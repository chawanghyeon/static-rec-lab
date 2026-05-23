"""DuckDB 기반 item KNN parquet 학습 backend."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import duckdb

from recsys.baseline.item_knn_model import CooccurrenceCandidate, ItemKNNModel
from recsys.baseline.popularity import PopularItem


def fit_item_knn_model_from_parquet(
    train_parquet: str | Path,
    *,
    max_candidates_per_item: int,
    max_history_items: int,
) -> ItemKNNModel:
    """train parquet 파일에서 전체 example 기반 item KNN model을 학습한다."""
    parquet_path = _sql_string(train_parquet)
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("SET preserve_insertion_order = false")
        popularity_items = _fit_popularity_items_with_duckdb(connection, parquet_path)
        neighbors = _fit_neighbors_with_duckdb(
            connection=connection,
            parquet_path=parquet_path,
            max_candidates_per_item=max_candidates_per_item,
            max_history_items=max_history_items,
        )
        count_row = connection.execute(
            f"SELECT count(*) FROM read_parquet('{parquet_path}')"
        ).fetchone()
        if count_row is None:
            msg = "train parquet row count를 계산할 수 없습니다."
            raise ValueError(msg)
        num_train_examples = int(count_row[0])
    finally:
        connection.close()

    return ItemKNNModel(
        neighbors=neighbors,
        popularity_items=popularity_items,
        num_train_examples=num_train_examples,
        max_candidates_per_item=max_candidates_per_item,
        max_history_items=max_history_items,
    )


def _fit_popularity_items_with_duckdb(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
) -> tuple[PopularItem, ...]:
    rows = connection.execute(
        f"""
        WITH counts AS (
            SELECT
                target_item_id AS item_id,
                count(*) AS item_count
            FROM read_parquet('{parquet_path}')
            GROUP BY target_item_id
        )
        SELECT
            item_id,
            item_count,
            row_number() OVER (ORDER BY item_count DESC, item_id ASC) AS item_rank
        FROM counts
        ORDER BY item_rank
        """
    ).fetchall()
    return tuple(
        PopularItem(item_id=int(item_id), count=int(count), rank=int(rank))
        for item_id, count, rank in rows
    )


def _fit_neighbors_with_duckdb(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
    max_candidates_per_item: int,
    max_history_items: int,
) -> Mapping[int, tuple[CooccurrenceCandidate, ...]]:
    rows = connection.execute(
        f"""
        WITH popularity AS (
            SELECT
                target_item_id AS item_id,
                count(*) AS item_count,
                row_number() OVER (
                    ORDER BY count(*) DESC, target_item_id ASC
                ) AS item_rank
            FROM read_parquet('{parquet_path}')
            GROUP BY target_item_id
        ),
        examples AS (
            SELECT
                row_number() OVER () AS example_id,
                target_item_id,
                list_slice(
                    history_item_ids,
                    greatest(len(history_item_ids) - ? + 1, 1),
                    len(history_item_ids)
                ) AS history_item_ids
            FROM read_parquet('{parquet_path}')
        ),
        history_pairs AS (
            SELECT DISTINCT
                example_id,
                unnest(history_item_ids) AS source_item_id,
                target_item_id
            FROM examples
        ),
        pair_counts AS (
            SELECT
                source_item_id,
                target_item_id,
                count(*) AS pair_count
            FROM history_pairs
            WHERE source_item_id <> target_item_id
            GROUP BY source_item_id, target_item_id
        ),
        ranked AS (
            SELECT
                pair_counts.source_item_id,
                pair_counts.target_item_id,
                pair_counts.pair_count,
                row_number() OVER (
                    PARTITION BY pair_counts.source_item_id
                    ORDER BY
                        pair_counts.pair_count DESC,
                        coalesce(popularity.item_rank, 9223372036854775807),
                        pair_counts.target_item_id ASC
                ) AS candidate_rank
            FROM pair_counts
            LEFT JOIN popularity
                ON pair_counts.target_item_id = popularity.item_id
        )
        SELECT source_item_id, target_item_id, pair_count
        FROM ranked
        WHERE candidate_rank <= ?
        ORDER BY source_item_id ASC, candidate_rank ASC
        """,
        [max_history_items, max_candidates_per_item],
    ).fetchall()

    neighbors: dict[int, list[CooccurrenceCandidate]] = defaultdict(list)
    for source_item_id, target_item_id, count in rows:
        neighbors[int(source_item_id)].append(
            CooccurrenceCandidate(item_id=int(target_item_id), count=int(count))
        )
    return {source_item_id: tuple(candidates) for source_item_id, candidates in neighbors.items()}


def _sql_string(path: str | Path) -> str:
    return str(Path(path)).replace("'", "''")

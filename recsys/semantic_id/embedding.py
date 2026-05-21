"""Item interaction embedding 생성."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import normalize  # type: ignore[import-untyped]

from recsys.data import coerce_item_ids


@dataclass(frozen=True)
class ItemEmbeddingConfig:
    """interaction embedding 생성 설정."""

    n_components: int = 32
    max_history_items: int = 50
    batch_size: int = 50_000
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_components < 1:
            msg = "n_components는 1 이상이어야 합니다."
            raise ValueError(msg)
        if self.max_history_items < 1:
            msg = "max_history_items는 1 이상이어야 합니다."
            raise ValueError(msg)
        if self.batch_size < 1:
            msg = "batch_size는 1 이상이어야 합니다."
            raise ValueError(msg)


@dataclass(frozen=True)
class ItemEmbeddings:
    """item_id와 dense embedding 행렬."""

    item_ids: tuple[int, ...]
    embeddings: np.ndarray
    num_examples: int
    num_embedding_examples: int
    num_context_edges: int

    @property
    def num_items(self) -> int:
        return len(self.item_ids)

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])


def build_item_interaction_embeddings(
    train_frame: pd.DataFrame,
    config: ItemEmbeddingConfig,
) -> ItemEmbeddings:
    """DataFrame의 전체 history-target co-occurrence로 item embedding을 만든다."""
    _validate_train_frame(train_frame)
    item_ids = _collect_item_ids_from_frame(train_frame)
    if not item_ids:
        msg = "embedding을 만들 item이 없습니다."
        raise ValueError(msg)

    state = _EmbeddingBuildState.create(item_ids, config)
    for start in range(0, len(train_frame), config.batch_size):
        chunk = train_frame.iloc[start : start + config.batch_size]
        target_item_ids = np.asarray(
            [int(item_id) for item_id in chunk["target_item_id"].tolist()],
            dtype=np.int64,
        )
        history_values = cast(Iterable[Any], chunk["history_item_ids"].tolist())
        sources, parent_indices = _flatten_history_lists(
            history_values,
            config.max_history_items,
        )
        state.update(target_item_ids, sources, parent_indices)

    return state.to_item_embeddings()


def build_item_interaction_embeddings_from_parquet(
    train_parquet: str | Path,
    config: ItemEmbeddingConfig,
) -> ItemEmbeddings:
    """Parquet 파일을 배치 단위로 읽어 전체 history-target co-occurrence embedding을 만든다."""
    parquet_file = pq.ParquetFile(train_parquet)  # type: ignore[no-untyped-call]
    _validate_parquet_schema(parquet_file)
    item_ids = _collect_item_ids_from_parquet(parquet_file, config.batch_size)
    if not item_ids:
        msg = "embedding을 만들 item이 없습니다."
        raise ValueError(msg)

    state = _EmbeddingBuildState.create(item_ids, config)
    for batch in parquet_file.iter_batches(  # type: ignore[no-untyped-call]
        batch_size=config.batch_size,
        columns=["history_item_ids", "target_item_id"],
    ):
        target_item_ids = _arrow_int_array_to_numpy(batch.column("target_item_id"))
        sources, parent_indices = _flatten_arrow_history_array(
            batch.column("history_item_ids"),
            config.max_history_items,
        )
        state.update(target_item_ids, sources, parent_indices)

    return state.to_item_embeddings()


@dataclass
class _EmbeddingBuildState:
    item_ids: tuple[int, ...]
    item_ids_array: np.ndarray
    dense_context: np.ndarray
    target_counts: np.ndarray
    source_counts: np.ndarray
    num_examples: int
    num_context_edges: int
    config: ItemEmbeddingConfig

    @classmethod
    def create(
        cls,
        item_ids: tuple[int, ...],
        config: ItemEmbeddingConfig,
    ) -> _EmbeddingBuildState:
        num_items = len(item_ids)
        return cls(
            item_ids=item_ids,
            item_ids_array=np.asarray(item_ids, dtype=np.int64),
            dense_context=np.zeros((num_items, config.n_components), dtype=np.float64),
            target_counts=np.zeros(num_items, dtype=np.float64),
            source_counts=np.zeros(num_items, dtype=np.float64),
            num_examples=0,
            num_context_edges=0,
            config=config,
        )

    def update(
        self,
        target_item_ids: np.ndarray,
        source_item_ids: np.ndarray,
        parent_indices: np.ndarray,
    ) -> None:
        target_indices = _lookup_item_indices(self.item_ids_array, target_item_ids)
        self.target_counts += np.bincount(
            target_indices,
            minlength=len(self.item_ids),
        ).astype(np.float64)
        self.num_examples += len(target_item_ids)

        if len(source_item_ids) == 0:
            return

        source_indices = _lookup_item_indices(self.item_ids_array, source_item_ids)
        self.source_counts += np.bincount(
            source_indices,
            minlength=len(self.item_ids),
        ).astype(np.float64)

        edge_target_indices = target_indices[parent_indices]
        edge_target_item_ids = target_item_ids[parent_indices]
        not_self = source_item_ids != edge_target_item_ids
        if not np.any(not_self):
            return

        context_targets = edge_target_indices[not_self]
        context_sources = source_item_ids[not_self]
        buckets = _hash_buckets(
            context_sources,
            self.config.n_components,
            self.config.random_state,
        )
        signs = _hash_signs(context_sources, self.config.random_state)
        linear_indices = context_targets * self.config.n_components + buckets
        projected_counts = np.bincount(
            linear_indices,
            weights=signs,
            minlength=len(self.item_ids) * self.config.n_components,
        )
        self.dense_context += projected_counts.reshape(
            len(self.item_ids),
            self.config.n_components,
        )
        self.num_context_edges += int(np.count_nonzero(not_self))

    def to_item_embeddings(self) -> ItemEmbeddings:
        dense_features = np.column_stack(
            [
                self.dense_context,
                _safe_log_scale(self.target_counts),
                _safe_log_scale(self.source_counts),
                _normalized_item_order(len(self.item_ids)),
            ]
        )
        embeddings = normalize(dense_features, norm="l2", axis=1)
        return ItemEmbeddings(
            item_ids=self.item_ids,
            embeddings=np.asarray(embeddings, dtype=np.float64),
            num_examples=self.num_examples,
            num_embedding_examples=self.num_examples,
            num_context_edges=self.num_context_edges,
        )


def _collect_item_ids_from_frame(train_frame: pd.DataFrame) -> tuple[int, ...]:
    item_ids = {
        int(item_id)
        for item_id in cast(Iterable[Any], train_frame["target_item_id"].unique().tolist())
    }
    for history_item_ids in cast(Iterable[Any], train_frame["history_item_ids"].tolist()):
        item_ids.update(coerce_item_ids(history_item_ids))
    return tuple(sorted(item_ids))


def _collect_item_ids_from_parquet(
    parquet_file: pq.ParquetFile,
    batch_size: int,
) -> tuple[int, ...]:
    item_ids: set[int] = set()
    for batch in parquet_file.iter_batches(  # type: ignore[no-untyped-call]
        batch_size=batch_size,
        columns=["history_item_ids", "target_item_id"],
    ):
        target_item_ids = _arrow_int_array_to_numpy(batch.column("target_item_id"))
        item_ids.update(int(item_id) for item_id in np.unique(target_item_ids).tolist())

        history_values = _arrow_list_values_to_numpy(batch.column("history_item_ids"))
        if len(history_values) > 0:
            item_ids.update(int(item_id) for item_id in np.unique(history_values).tolist())
    return tuple(sorted(item_ids))


def _flatten_history_lists(
    history_values: Iterable[Any],
    max_history_items: int,
) -> tuple[np.ndarray, np.ndarray]:
    sources: list[int] = []
    parent_indices: list[int] = []
    for parent_index, history_value in enumerate(history_values):
        history_item_ids = coerce_item_ids(history_value)[-max_history_items:]
        sources.extend(history_item_ids)
        parent_indices.extend([parent_index] * len(history_item_ids))

    return (
        np.asarray(sources, dtype=np.int64),
        np.asarray(parent_indices, dtype=np.int64),
    )


def _flatten_arrow_history_array(
    history_array: pa.Array,
    max_history_items: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = _arrow_offsets_to_numpy(history_array)
    values = _arrow_list_values_to_numpy(history_array)
    if len(values) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    lengths = offsets[1:] - offsets[:-1]
    parent_indices = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    local_offsets = np.repeat(offsets[:-1], lengths)
    positions = np.arange(len(values), dtype=np.int64) - local_offsets
    history_starts = np.repeat(np.maximum(lengths - max_history_items, 0), lengths)
    keep = positions >= history_starts
    return (
        np.asarray(values[keep], dtype=np.int64),
        np.asarray(parent_indices[keep], dtype=np.int64),
    )


def _arrow_int_array_to_numpy(array: pa.Array) -> np.ndarray:
    return np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.int64)


def _arrow_offsets_to_numpy(array: pa.Array) -> np.ndarray:
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if not isinstance(array, pa.ListArray | pa.LargeListArray):
        msg = "history_item_ids는 list array여야 합니다."
        raise ValueError(msg)
    return np.asarray(array.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)


def _arrow_list_values_to_numpy(array: pa.Array) -> np.ndarray:
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if not isinstance(array, pa.ListArray | pa.LargeListArray):
        msg = "history_item_ids는 list array여야 합니다."
        raise ValueError(msg)
    return np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.int64)


def _lookup_item_indices(item_ids_array: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(item_ids_array, item_ids)
    if np.any(indices >= len(item_ids_array)) or np.any(item_ids_array[indices] != item_ids):
        msg = "item index에서 찾을 수 없는 item_id가 있습니다."
        raise ValueError(msg)
    return np.asarray(indices, dtype=np.int64)


def _hash_buckets(
    item_ids: np.ndarray,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    values = item_ids.astype(np.uint64, copy=False)
    seed = np.uint64(random_state + 0x9E3779B1)
    hashed = values * np.uint64(11400714819323198485) + seed
    return np.asarray(hashed % np.uint64(n_components), dtype=np.int64)


def _hash_signs(item_ids: np.ndarray, random_state: int) -> np.ndarray:
    values = item_ids.astype(np.uint64, copy=False)
    seed = np.uint64(random_state + 0x85EBCA6B)
    hashed = values * np.uint64(14029467366897019727) + seed
    return np.where((hashed & np.uint64(1)) == 0, 1.0, -1.0).astype(np.float64)


def _safe_log_scale(values: np.ndarray) -> np.ndarray:
    scaled = np.asarray(np.log1p(values), dtype=np.float64)
    max_value = float(scaled.max())
    if max_value == 0.0:
        return scaled
    return np.asarray(scaled / max_value, dtype=np.float64)


def _normalized_item_order(num_items: int) -> np.ndarray:
    if num_items == 1:
        return np.zeros(1, dtype=np.float64)
    return np.linspace(0.0, 1.0, num_items, dtype=np.float64)


def _validate_train_frame(train_frame: pd.DataFrame) -> None:
    required_columns = {"history_item_ids", "target_item_id"}
    missing_columns = sorted(required_columns - set(train_frame.columns))
    if missing_columns:
        msg = f"train 데이터에 필요한 컬럼이 없습니다: {missing_columns}"
        raise ValueError(msg)


def _validate_parquet_schema(parquet_file: pq.ParquetFile) -> None:
    schema_names = set(parquet_file.schema_arrow.names)
    required_columns = {"history_item_ids", "target_item_id"}
    missing_columns = sorted(required_columns - schema_names)
    if missing_columns:
        msg = f"train parquet에 필요한 컬럼이 없습니다: {missing_columns}"
        raise ValueError(msg)

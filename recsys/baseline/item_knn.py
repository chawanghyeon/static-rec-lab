"""Item co-occurrence baseline recommender."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from recsys.baseline.popularity import PopularItem, fit_popularity_model
from recsys.evaluation import RankingMetrics, evaluate_ranking_at_k

DEFAULT_MAX_CANDIDATES_PER_ITEM: Final[int] = 200
DEFAULT_MAX_HISTORY_ITEMS: Final[int] = 50


@dataclass(frozen=True)
class CooccurrenceCandidate:
    """특정 source item과 함께 등장한 candidate item."""

    item_id: int
    count: int


@dataclass(frozen=True)
class ItemKNNModel:
    """Item co-occurrence 기반 baseline."""

    neighbors: Mapping[int, tuple[CooccurrenceCandidate, ...]]
    popularity_items: tuple[PopularItem, ...]
    num_train_examples: int
    max_candidates_per_item: int
    max_history_items: int

    def recommend(self, history_item_ids: Iterable[int], k: int) -> list[int]:
        """history item들의 co-occurrence score를 합산해 top-K item을 추천한다."""
        if k < 1:
            msg = "k는 1 이상이어야 합니다."
            raise ValueError(msg)

        history = [int(item_id) for item_id in history_item_ids]
        seen = set(history)
        query_items = history[-self.max_history_items :]
        popularity_rank = {item.item_id: item.rank for item in self.popularity_items}

        scores: dict[int, int] = defaultdict(int)
        for source_item_id in query_items:
            for candidate in self.neighbors.get(source_item_id, ()):
                if candidate.item_id in seen:
                    continue
                scores[candidate.item_id] += candidate.count

        ranked_candidates = sorted(
            scores,
            key=lambda item_id: (
                -scores[item_id],
                popularity_rank.get(item_id, len(popularity_rank) + 1),
                item_id,
            ),
        )
        recommendations = ranked_candidates[:k]
        recommended = set(recommendations)

        for item in self.popularity_items:
            if len(recommendations) == k:
                break
            if item.item_id in seen or item.item_id in recommended:
                continue
            recommendations.append(item.item_id)
            recommended.add(item.item_id)

        return recommendations

    @property
    def num_items(self) -> int:
        return len(self.neighbors)


def fit_item_knn_model(
    train_frame: pd.DataFrame,
    max_candidates_per_item: int = DEFAULT_MAX_CANDIDATES_PER_ITEM,
    max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS,
) -> ItemKNNModel:
    """train split의 history-target co-occurrence로 item KNN baseline을 학습한다."""
    _validate_train_frame(train_frame)
    if max_candidates_per_item < 1:
        msg = "max_candidates_per_item은 1 이상이어야 합니다."
        raise ValueError(msg)
    if max_history_items < 1:
        msg = "max_history_items는 1 이상이어야 합니다."
        raise ValueError(msg)

    popularity_model = fit_popularity_model(train_frame)
    popularity_rank = {item.item_id: item.rank for item in popularity_model.items}
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for record in train_frame[["history_item_ids", "target_item_id"]].to_dict("records"):
        target_item_id = int(record["target_item_id"])
        history_item_ids = _coerce_item_ids(record["history_item_ids"])[-max_history_items:]
        for source_item_id in set(history_item_ids):
            if source_item_id == target_item_id:
                continue
            counts[source_item_id][target_item_id] += 1

    neighbors = {
        source_item_id: tuple(
            CooccurrenceCandidate(item_id=candidate_item_id, count=count)
            for candidate_item_id, count in sorted(
                candidate_counts.items(),
                key=lambda item_count: (
                    -item_count[1],
                    popularity_rank.get(item_count[0], len(popularity_rank) + 1),
                    item_count[0],
                ),
            )[:max_candidates_per_item]
        )
        for source_item_id, candidate_counts in counts.items()
    }
    return ItemKNNModel(
        neighbors=neighbors,
        popularity_items=popularity_model.items,
        num_train_examples=len(train_frame),
        max_candidates_per_item=max_candidates_per_item,
        max_history_items=max_history_items,
    )


def fit_item_knn_model_from_parquet(
    train_parquet: str | Path,
    max_candidates_per_item: int = DEFAULT_MAX_CANDIDATES_PER_ITEM,
    max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS,
) -> ItemKNNModel:
    """train parquet 파일에서 item KNN model을 학습한다."""
    return fit_item_knn_model(
        pd.read_parquet(train_parquet),
        max_candidates_per_item=max_candidates_per_item,
        max_history_items=max_history_items,
    )


def save_item_knn_model(model: ItemKNNModel, output_path: str | Path) -> Path:
    """item KNN model을 JSON artifact로 저장한다."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "item_knn",
        "num_train_examples": model.num_train_examples,
        "max_candidates_per_item": model.max_candidates_per_item,
        "max_history_items": model.max_history_items,
        "popularity_items": [asdict(item) for item in model.popularity_items],
        "neighbors": {
            str(source_item_id): [asdict(candidate) for candidate in candidates]
            for source_item_id, candidates in model.neighbors.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_item_knn_model(model_path: str | Path) -> ItemKNNModel:
    """JSON artifact에서 item KNN model을 로드한다."""
    payload = json.loads(Path(model_path).read_text(encoding="utf-8"))
    _validate_model_payload(payload)
    popularity_items = tuple(
        PopularItem(
            item_id=int(item["item_id"]),
            count=int(item["count"]),
            rank=int(item["rank"]),
        )
        for item in payload["popularity_items"]
    )
    neighbors = {
        int(source_item_id): tuple(
            CooccurrenceCandidate(
                item_id=int(candidate["item_id"]),
                count=int(candidate["count"]),
            )
            for candidate in candidates
        )
        for source_item_id, candidates in payload["neighbors"].items()
    }
    return ItemKNNModel(
        neighbors=neighbors,
        popularity_items=popularity_items,
        num_train_examples=int(payload["num_train_examples"]),
        max_candidates_per_item=int(payload["max_candidates_per_item"]),
        max_history_items=int(payload["max_history_items"]),
    )


def evaluate_item_knn_model(
    model: ItemKNNModel,
    eval_frame: pd.DataFrame,
    cutoffs: Sequence[int],
) -> dict[int, RankingMetrics]:
    """평가 split에 대해 item KNN model의 ranking metric을 계산한다."""
    if "history_item_ids" not in eval_frame.columns or "target_item_id" not in eval_frame.columns:
        msg = "eval 데이터에 history_item_ids와 target_item_id 컬럼이 필요합니다."
        raise ValueError(msg)

    max_k = max(cutoffs)
    recommendations = [
        model.recommend(_coerce_item_ids(row.history_item_ids), max_k)
        for row in eval_frame.itertuples(index=False)
    ]
    relevant_items = [[int(target_item_id)] for target_item_id in eval_frame["target_item_id"]]
    return {
        cutoff: evaluate_ranking_at_k(recommendations, relevant_items, cutoff) for cutoff in cutoffs
    }


def _validate_train_frame(train_frame: pd.DataFrame) -> None:
    required_columns = {"history_item_ids", "target_item_id"}
    missing_columns = sorted(required_columns - set(train_frame.columns))
    if missing_columns:
        msg = f"train 데이터에 필요한 컬럼이 없습니다: {missing_columns}"
        raise ValueError(msg)


def _coerce_item_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(item_id) for item_id in value]


def _validate_model_payload(payload: Any) -> None:
    if not isinstance(payload, dict):
        msg = "item KNN model payload는 JSON object여야 합니다."
        raise ValueError(msg)
    if payload.get("model") != "item_knn":
        msg = "item KNN model artifact가 아닙니다."
        raise ValueError(msg)
    required_fields = {
        "num_train_examples",
        "max_candidates_per_item",
        "max_history_items",
        "popularity_items",
        "neighbors",
    }
    missing_fields = sorted(required_fields - set(payload))
    if missing_fields:
        msg = f"item KNN model artifact에 필수 필드가 없습니다: {missing_fields}"
        raise ValueError(msg)

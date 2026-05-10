"""Popularity baseline recommender."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from recsys.evaluation import RankingMetrics, evaluate_ranking_at_k

DEFAULT_CUTOFFS: Final[tuple[int, int]] = (10, 20)


@dataclass(frozen=True)
class PopularItem:
    """Popularity ranking에 포함된 item."""

    item_id: int
    count: int
    rank: int


@dataclass(frozen=True)
class PopularityModel:
    """전역 item popularity baseline."""

    items: tuple[PopularItem, ...]
    num_train_examples: int

    def recommend(self, history_item_ids: Iterable[int], k: int) -> list[int]:
        """history에 이미 등장한 item을 제외하고 top-K item을 추천한다."""
        if k < 1:
            msg = "k는 1 이상이어야 합니다."
            raise ValueError(msg)

        seen = {int(item_id) for item_id in history_item_ids}
        recommendations: list[int] = []
        for item in self.items:
            if item.item_id in seen:
                continue
            recommendations.append(item.item_id)
            if len(recommendations) == k:
                break
        return recommendations

    @property
    def num_items(self) -> int:
        return len(self.items)


def fit_popularity_model(train_frame: pd.DataFrame) -> PopularityModel:
    """train split의 target_item_id 빈도로 popularity model을 학습한다."""
    if "target_item_id" not in train_frame.columns:
        msg = "train 데이터에 target_item_id 컬럼이 필요합니다."
        raise ValueError(msg)

    counts = (
        train_frame["target_item_id"]
        .astype("int64")
        .value_counts(sort=False)
        .rename_axis("item_id")
        .reset_index(name="count")
        .sort_values(["count", "item_id"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    items = tuple(
        PopularItem(
            item_id=int(record["item_id"]),
            count=int(record["count"]),
            rank=rank,
        )
        for rank, record in enumerate(counts.to_dict("records"), start=1)
    )
    return PopularityModel(items=items, num_train_examples=len(train_frame))


def fit_popularity_model_from_parquet(train_parquet: str | Path) -> PopularityModel:
    """train parquet 파일에서 popularity model을 학습한다."""
    return fit_popularity_model(pd.read_parquet(train_parquet))


def save_popularity_model(model: PopularityModel, output_path: str | Path) -> Path:
    """popularity model을 JSON artifact로 저장한다."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "popularity",
        "num_train_examples": model.num_train_examples,
        "items": [asdict(item) for item in model.items],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_popularity_model(model_path: str | Path) -> PopularityModel:
    """JSON artifact에서 popularity model을 로드한다."""
    payload = json.loads(Path(model_path).read_text(encoding="utf-8"))
    _validate_model_payload(payload)
    items = tuple(
        PopularItem(
            item_id=int(item["item_id"]),
            count=int(item["count"]),
            rank=int(item["rank"]),
        )
        for item in payload["items"]
    )
    return PopularityModel(items=items, num_train_examples=int(payload["num_train_examples"]))


def evaluate_popularity_model(
    model: PopularityModel,
    eval_frame: pd.DataFrame,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> dict[int, RankingMetrics]:
    """평가 split에 대해 popularity model의 ranking metric을 계산한다."""
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


def write_baseline_report(
    report_path: str | Path,
    model: PopularityModel,
    valid_metrics: dict[int, RankingMetrics],
    test_metrics: dict[int, RankingMetrics],
) -> Path:
    """baseline 평가 결과를 한국어 markdown 리포트로 저장한다."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline 평가 리포트",
        "",
        "## 설정",
        "",
        "- 모델: Popularity baseline",
        f"- 학습 예제 수: {model.num_train_examples}",
        f"- popularity item 수: {model.num_items}",
        "- 추천 시 사용자 history에 이미 포함된 item은 제외합니다.",
        "",
        "## 평가 결과",
        "",
        "| split | k | Recall@K | NDCG@K | MRR@K |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(_metrics_table_rows("valid", valid_metrics))
    lines.extend(_metrics_table_rows("test", test_metrics))
    lines.extend(
        [
            "",
            "## 해석",
            "",
            "Popularity baseline은 개인화 없이 전체 train split에서 자주 등장한 item을 추천합니다.",
            "이 결과는 이후 item co-occurrence baseline과 Generative Retrieval 모델의 "
            "하한선으로 사용합니다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _metrics_table_rows(split_name: str, metrics_by_k: dict[int, RankingMetrics]) -> list[str]:
    return [
        (f"| {split_name} | {k} | {metrics.recall:.6f} | {metrics.ndcg:.6f} | {metrics.mrr:.6f} |")
        for k, metrics in sorted(metrics_by_k.items())
    ]


def _coerce_item_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(item_id) for item_id in value]


def _validate_model_payload(payload: Any) -> None:
    if not isinstance(payload, dict):
        msg = "popularity model payload는 JSON object여야 합니다."
        raise ValueError(msg)
    if payload.get("model") != "popularity":
        msg = "popularity model artifact가 아닙니다."
        raise ValueError(msg)
    if "num_train_examples" not in payload or "items" not in payload:
        msg = "popularity model artifact에 필수 필드가 없습니다."
        raise ValueError(msg)

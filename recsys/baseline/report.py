"""Baseline 평가 리포트 생성."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from recsys.evaluation import RankingMetrics


@dataclass(frozen=True)
class BaselineEvaluation:
    """한 baseline 모델의 valid/test 평가 결과."""

    model_name: str
    description: str
    num_train_examples: int
    num_items: int
    valid_metrics: dict[int, RankingMetrics]
    test_metrics: dict[int, RankingMetrics]


def write_baseline_comparison_report(
    report_path: str | Path,
    evaluations: Sequence[BaselineEvaluation],
) -> Path:
    """여러 baseline 결과를 한국어 markdown 리포트로 저장한다."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline 평가 리포트",
        "",
        "## 설정",
        "",
    ]

    for evaluation in evaluations:
        lines.extend(
            [
                f"### {evaluation.model_name}",
                "",
                f"- 설명: {evaluation.description}",
                f"- 학습 예제 수: {evaluation.num_train_examples}",
                f"- item 수: {evaluation.num_items}",
                "",
            ]
        )

    lines.extend(
        [
            "## 평가 결과",
            "",
            "| model | split | k | Recall@K | NDCG@K | MRR@K |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for evaluation in evaluations:
        lines.extend(_metrics_table_rows(evaluation.model_name, "valid", evaluation.valid_metrics))
        lines.extend(_metrics_table_rows(evaluation.model_name, "test", evaluation.test_metrics))

    lines.extend(
        [
            "",
            "## 해석",
            "",
            "Popularity baseline은 개인화 없이 전체 train split에서 자주 등장한 item을 추천합니다.",
            "Item co-occurrence baseline은 사용자 positive history item과 함께 등장한 "
            "target item을 집계해 개인화된 추천을 만듭니다.",
            "이 결과는 이후 Generative Retrieval 모델의 비교 기준으로 사용합니다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _metrics_table_rows(
    model_name: str,
    split_name: str,
    metrics_by_k: dict[int, RankingMetrics],
) -> list[str]:
    return [
        (
            f"| {model_name} | {split_name} | {k} | "
            f"{metrics.recall:.6f} | {metrics.ndcg:.6f} | {metrics.mrr:.6f} |"
        )
        for k, metrics in sorted(metrics_by_k.items())
    ]

"""Baseline 평가 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from recsys.baseline import (
    DEFAULT_CUTOFFS,
    BaselineEvaluation,
    evaluate_item_knn_model,
    evaluate_popularity_model,
    load_item_knn_model,
    load_popularity_model,
    write_baseline_comparison_report,
)

BASELINE_TYPES = ("popularity", "item_knn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline 평가")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/baseline"),
        help="baseline model JSON 디렉터리",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=BASELINE_TYPES,
        default=list(BASELINE_TYPES),
        help="평가할 baseline 종류",
    )
    parser.add_argument(
        "--valid-parquet",
        type=Path,
        default=Path("data/processed/valid.parquet"),
        help="전처리된 valid parquet 경로",
    )
    parser.add_argument(
        "--test-parquet",
        type=Path,
        default=Path("data/processed/test.parquet"),
        help="전처리된 test parquet 경로",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/baseline.md"),
        help="baseline 평가 리포트 저장 경로",
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=list(DEFAULT_CUTOFFS),
        help="평가 cutoff K 목록",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    valid_frame = pd.read_parquet(args.valid_parquet)
    test_frame = pd.read_parquet(args.test_parquet)
    evaluations: list[BaselineEvaluation] = []

    if "popularity" in args.model_types:
        popularity_model = load_popularity_model(args.model_dir / "popularity.json")
        valid_metrics = evaluate_popularity_model(
            popularity_model, valid_frame, cutoffs=args.cutoffs
        )
        test_metrics = evaluate_popularity_model(popularity_model, test_frame, cutoffs=args.cutoffs)
        evaluations.append(
            BaselineEvaluation(
                model_name="Popularity",
                description="전체 train split의 target item 빈도 기반 baseline",
                num_train_examples=popularity_model.num_train_examples,
                num_items=popularity_model.num_items,
                valid_metrics=valid_metrics,
                test_metrics=test_metrics,
            )
        )

    if "item_knn" in args.model_types:
        item_knn_model = load_item_knn_model(args.model_dir / "item_knn.json")
        valid_metrics = evaluate_item_knn_model(item_knn_model, valid_frame, cutoffs=args.cutoffs)
        test_metrics = evaluate_item_knn_model(item_knn_model, test_frame, cutoffs=args.cutoffs)
        evaluations.append(
            BaselineEvaluation(
                model_name="Item co-occurrence",
                description="사용자 history item과 target item의 co-occurrence count 기반 baseline",
                num_train_examples=item_knn_model.num_train_examples,
                num_items=item_knn_model.num_items,
                valid_metrics=valid_metrics,
                test_metrics=test_metrics,
            )
        )

    report_path = write_baseline_comparison_report(args.report_path, evaluations)

    print("Baseline 평가 완료")
    for evaluation in evaluations:
        for split_name, metrics_by_k in (
            ("valid", evaluation.valid_metrics),
            ("test", evaluation.test_metrics),
        ):
            for k, metrics in sorted(metrics_by_k.items()):
                print(
                    f"- {evaluation.model_name} {split_name}@{k}: "
                    f"Recall={metrics.recall:.6f}, "
                    f"NDCG={metrics.ndcg:.6f}, "
                    f"MRR={metrics.mrr:.6f}"
                )
    print(f"- report: {report_path}")


if __name__ == "__main__":
    main()

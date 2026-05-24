"""Baseline 학습/평가 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from recsys.baseline import (
    DEFAULT_CUTOFFS,
    DEFAULT_MAX_CANDIDATES_PER_ITEM,
    DEFAULT_MAX_HISTORY_ITEMS,
    BaselineEvaluation,
    evaluate_item_knn_model,
    evaluate_popularity_model,
    fit_item_knn_model_from_parquet,
    fit_popularity_model_from_parquet,
    load_item_knn_model,
    load_popularity_model,
    save_item_knn_model,
    save_popularity_model,
    write_baseline_comparison_report,
)

BASELINE_TYPES = ("popularity", "item_knn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline 학습/평가")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(subparsers)
    _add_eval_parser(subparsers)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_baseline(args)
    elif args.command == "eval":
        eval_baseline(args)
    else:  # pragma: no cover
        raise ValueError(f"알 수 없는 baseline command입니다: {args.command}")


def _add_train_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("train", help="baseline 모델 학습")
    parser.add_argument(
        "--train-parquet",
        type=Path,
        default=Path("data/processed/train.parquet"),
        help="전처리된 train parquet 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/baseline"),
        help="baseline model JSON 저장 디렉터리",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=BASELINE_TYPES,
        default=list(BASELINE_TYPES),
        help="학습할 baseline 종류",
    )
    parser.add_argument(
        "--max-candidates-per-item",
        type=int,
        default=DEFAULT_MAX_CANDIDATES_PER_ITEM,
        help="item KNN에서 source item별 저장할 최대 candidate 수",
    )
    parser.add_argument(
        "--max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
        help="item KNN 학습과 추천에서 사용할 최근 history item 수",
    )


def _add_eval_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("eval", help="baseline 모델 평가")
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
        default=Path("reports/local/baseline.md"),
        help="baseline 평가 리포트 저장 경로",
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=list(DEFAULT_CUTOFFS),
        help="평가 cutoff K 목록",
    )


def train_baseline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)

    if "popularity" in args.model_types:
        popularity_model = fit_popularity_model_from_parquet(args.train_parquet)
        output_path = save_popularity_model(popularity_model, output_dir / "popularity.json")
        print("Popularity baseline 학습 완료")
        print(f"- train examples: {popularity_model.num_train_examples}")
        print(f"- items: {popularity_model.num_items}")
        print(f"- artifact: {output_path}")

    if "item_knn" in args.model_types:
        item_knn_model = fit_item_knn_model_from_parquet(
            args.train_parquet,
            max_candidates_per_item=args.max_candidates_per_item,
            max_history_items=args.max_history_items,
        )
        output_path = save_item_knn_model(item_knn_model, output_dir / "item_knn.json")
        print("Item co-occurrence baseline 학습 완료")
        print(f"- train examples: {item_knn_model.num_train_examples}")
        print(f"- source items: {item_knn_model.num_items}")
        print(f"- artifact: {output_path}")


def eval_baseline(args: argparse.Namespace) -> None:
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
                description=(
                    "사용자 positive history item과 target item의 co-occurrence count 기반 baseline"
                ),
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

"""Baseline 학습 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from recsys.baseline import (
    DEFAULT_MAX_CANDIDATES_PER_ITEM,
    DEFAULT_MAX_HISTORY_ITEMS,
    fit_item_knn_model_from_parquet,
    fit_popularity_model_from_parquet,
    save_item_knn_model,
    save_popularity_model,
)

BASELINE_TYPES = ("popularity", "item_knn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline 학습")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()

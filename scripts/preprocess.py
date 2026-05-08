"""MovieLens 전처리 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from recsys.data import PreprocessConfig, preprocess_ratings_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MovieLens ratings.csv 전처리")
    parser.add_argument(
        "--ratings-csv",
        type=Path,
        default=Path("data/raw/ratings.csv"),
        help="MovieLens ratings.csv 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="전처리 parquet 출력 디렉터리",
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=5,
        help="사용자별 최소 interaction 수",
    )
    parser.add_argument(
        "--max-history-length",
        type=int,
        default=50,
        help="history 최대 길이. 0 이하이면 제한하지 않습니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_history_length = args.max_history_length if args.max_history_length > 0 else None
    result = preprocess_ratings_csv(
        ratings_csv=args.ratings_csv,
        output_dir=args.output_dir,
        config=PreprocessConfig(
            min_interactions=args.min_interactions,
            max_history_length=max_history_length,
        ),
    )

    print(f"전처리 완료: users={result.num_users}, interactions={result.num_interactions}")
    for split_name, output_path in result.paths.items():
        print(f"- {split_name}: {result.split_counts[split_name]} examples -> {output_path}")


if __name__ == "__main__":
    main()

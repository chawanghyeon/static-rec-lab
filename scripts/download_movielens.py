"""MovieLens 다운로드 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from recsys.data import (
    DEFAULT_MOVIELENS_DATASET,
    DEFAULT_MOVIELENS_URL,
    download_movielens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MovieLens 데이터 다운로드")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="원본 데이터 다운로드 디렉터리",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_MOVIELENS_DATASET,
        help="압축 파일 내부 데이터셋 디렉터리 이름",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_MOVIELENS_URL,
        help="MovieLens zip 다운로드 URL",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="이미 받은 파일도 다시 다운로드하고 압축 해제합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = download_movielens(
        output_dir=args.output_dir,
        url=args.url,
        dataset_name=args.dataset_name,
        force=args.force,
    )

    print("MovieLens 다운로드 완료")
    print(f"- dataset: {result.dataset_name}")
    print(f"- zip: {result.zip_path}")
    print(f"- ratings: {result.ratings_csv}")


if __name__ == "__main__":
    main()

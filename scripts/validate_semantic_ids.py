"""Semantic ID artifact 검증 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from recsys.semantic_id import validate_semantic_id_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic ID artifact 검증")
    parser.add_argument(
        "--semantic-id-path",
        type=Path,
        default=Path("artifacts/semantic_id/semantic_ids.json"),
        help="Semantic ID codec JSON 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_semantic_id_file(args.semantic_id_path)
    print("Semantic ID 검증 완료")
    print(f"- items: {result.num_items}")
    print(f"- semantic_id_length: {result.semantic_id_length}")


if __name__ == "__main__":
    main()

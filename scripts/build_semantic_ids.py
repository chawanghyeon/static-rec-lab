"""Semantic ID 생성 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from recsys.semantic_id import (
    HierarchicalKMeansConfig,
    ItemEmbeddingConfig,
    build_item_interaction_embeddings_from_parquet,
    build_semantic_id_codec,
    write_semantic_id_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic ID 생성")
    parser.add_argument(
        "--train-parquet",
        type=Path,
        default=Path("data/processed/train.parquet"),
        help="전처리된 train parquet 경로",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/semantic_id/semantic_ids.json"),
        help="Semantic ID codec JSON 저장 경로",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/semantic_id.md"),
        help="Semantic ID 리포트 저장 경로",
    )
    parser.add_argument("--depth", type=int, default=4, help="Semantic ID depth")
    parser.add_argument("--branching-factor", type=int, default=16, help="분기 수")
    parser.add_argument(
        "--n-components", type=int, default=32, help="co-occurrence projection 차원"
    )
    parser.add_argument("--max-history-items", type=int, default=50, help="최근 history item 수")
    parser.add_argument("--random-state", type=int, default=42, help="랜덤 시드")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = build_item_interaction_embeddings_from_parquet(
        args.train_parquet,
        ItemEmbeddingConfig(
            n_components=args.n_components,
            max_history_items=args.max_history_items,
            random_state=args.random_state,
        ),
    )
    result = build_semantic_id_codec(
        embeddings,
        HierarchicalKMeansConfig(
            depth=args.depth,
            branching_factor=args.branching_factor,
            random_state=args.random_state,
        ),
    )
    output_path = result.codec.save_json(args.output_path)
    report_path = write_semantic_id_report(args.report_path, result, output_path)

    print("Semantic ID 생성 완료")
    print(f"- items: {result.codec.num_items}")
    print(f"- semantic_id_length: {result.codec.semantic_id_length}")
    print(f"- train_examples: {result.num_examples}")
    print(f"- embedding_examples: {result.num_embedding_examples}")
    print(f"- embedding_dim: {result.embedding_dim}")
    print(f"- output: {output_path}")
    print(f"- report: {report_path}")


if __name__ == "__main__":
    main()

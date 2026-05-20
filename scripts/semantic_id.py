"""Semantic ID 생성/검증/static_decoding index CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from recsys.decoding import StaticDecodingIndex
from recsys.semantic_id import (
    HierarchicalKMeansConfig,
    ItemEmbeddingConfig,
    SemanticIdCodec,
    build_item_interaction_embeddings_from_parquet,
    build_semantic_id_codec,
    validate_semantic_id_file,
    write_semantic_id_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic ID artifact 관리")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_build_parser(subparsers)
    _add_validate_parser(subparsers)
    _add_static_index_parser(subparsers)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build_semantic_ids(args)
    elif args.command == "validate":
        validate_semantic_ids(args)
    elif args.command == "build-static-index":
        build_static_decoding_index(args)
    else:  # pragma: no cover
        raise ValueError(f"알 수 없는 Semantic ID command입니다: {args.command}")


def _add_build_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("build", help="Semantic ID 생성")
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
        default=Path("reports/local/semantic_id.md"),
        help="Semantic ID 리포트 저장 경로",
    )
    parser.add_argument("--depth", type=int, default=4, help="Semantic ID depth")
    parser.add_argument("--branching-factor", type=int, default=16, help="분기 수")
    parser.add_argument(
        "--n-components", type=int, default=32, help="co-occurrence projection 차원"
    )
    parser.add_argument("--max-history-items", type=int, default=50, help="최근 history item 수")
    parser.add_argument("--random-state", type=int, default=42, help="랜덤 시드")


def _add_validate_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("validate", help="Semantic ID artifact 검증")
    parser.add_argument(
        "--semantic-id-path",
        type=Path,
        default=Path("artifacts/semantic_id/semantic_ids.json"),
        help="Semantic ID codec JSON 경로",
    )


def _add_static_index_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("build-static-index", help="static_decoding index 생성")
    parser.add_argument(
        "--semantic-id-path",
        type=Path,
        default=Path("artifacts/semantic_id/semantic_ids.json"),
        help="Semantic ID codec JSON 경로",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/semantic_id/static_decoding_index.npz"),
        help="static_decoding index npz 저장 경로",
    )
    parser.add_argument(
        "--dense-lookup-layers",
        default="auto",
        help="static_decoding dense lookup layer 수. auto면 min(2, depth - 1)을 사용한다.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Semantic token vocab size. 생략하면 artifact에서 추론한다.",
    )


def build_semantic_ids(args: argparse.Namespace) -> None:
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


def validate_semantic_ids(args: argparse.Namespace) -> None:
    result = validate_semantic_id_file(args.semantic_id_path)
    print("Semantic ID 검증 완료")
    print(f"- items: {result.num_items}")
    print(f"- semantic_id_length: {result.semantic_id_length}")


def build_static_decoding_index(args: argparse.Namespace) -> None:
    codec = SemanticIdCodec.load_json(args.semantic_id_path)
    dense_lookup_layers = resolve_dense_lookup_layers(
        codec=codec,
        value=str(args.dense_lookup_layers),
    )
    index = StaticDecodingIndex.from_codec(
        codec,
        vocab_size=args.vocab_size,
        dense_lookup_layers=dense_lookup_layers,
    )
    output_path = index.save_npz(args.output_path)

    print("static_decoding index 생성 완료")
    print(f"- source_repository: {index.source_repository}")
    print(f"- source_commit: {index.source_commit}")
    print(f"- items: {codec.num_items}")
    print(f"- semantic_id_depth: {index.semantic_id_depth}")
    print(f"- vocab_size: {index.vocab_size}")
    print(f"- dense_lookup_layers: {index.dense_lookup_layers}")
    print(f"- packed_csr_shape: {tuple(index.packed_csr.shape)}")
    print(f"- csr_indptr_shape: {tuple(index.csr_indptr.shape)}")
    print(f"- dense_mask_shape: {tuple(index.dense_mask.shape)}")
    print(f"- output: {output_path}")


def resolve_dense_lookup_layers(*, codec: SemanticIdCodec, value: str) -> int:
    """CLI 입력에서 static_decoding dense lookup layer 수를 결정한다."""
    depths = {len(semantic_id) for semantic_id in codec.item_to_semantic_id.values()}
    if len(depths) != 1:
        msg = f"모든 Semantic ID 길이가 같아야 합니다: {sorted(depths)}"
        raise ValueError(msg)
    depth = next(iter(depths), 0)
    if depth < 2:
        msg = "static_decoding build_static_index는 길이 2 이상의 Semantic ID가 필요합니다."
        raise ValueError(msg)
    if value == "auto":
        return min(2, depth - 1)

    dense_lookup_layers = int(value)
    if dense_lookup_layers < 1 or dense_lookup_layers >= depth:
        msg = (
            "dense_lookup_layers는 1 이상이고 Semantic ID 길이보다 작아야 합니다: "
            f"dense_lookup_layers={dense_lookup_layers}, depth={depth}"
        )
        raise ValueError(msg)
    return dense_lookup_layers


if __name__ == "__main__":
    main()

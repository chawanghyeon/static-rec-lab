"""static_decoding index artifact 생성 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from recsys.decoding import StaticDecodingIndex
from recsys.semantic_id import SemanticIdCodec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="static_decoding index 생성")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

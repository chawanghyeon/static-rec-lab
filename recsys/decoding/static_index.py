"""STATIC decoding index construction helpers."""

from __future__ import annotations

from pathlib import Path

from recsys.decoding.static_artifact import (
    StaticDecodingIndex,
    validate_static_decoding_index_matches_codec,
)
from recsys.semantic_id import SemanticIdCodec

STATIC_DECODING_DECODER_NAME = "static_decoding_pt"


def resolve_dense_lookup_layers(*, codec: SemanticIdCodec, value: str = "auto") -> int:
    """Semantic ID depth와 CLI/API 입력에서 dense lookup layer 수를 결정한다."""
    depth = _semantic_id_depth(codec)
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


def load_or_build_static_decoding_index(
    *,
    codec: SemanticIdCodec,
    static_decoding_index_path: str | Path | None,
    dense_lookup_layers: str = "auto",
    vocab_size: int | None = None,
) -> StaticDecodingIndex:
    """저장된 index를 로드하거나 codec에서 새 STATIC decoding index를 만든다."""
    if static_decoding_index_path is None:
        return StaticDecodingIndex.from_codec(
            codec,
            vocab_size=vocab_size,
            dense_lookup_layers=resolve_dense_lookup_layers(
                codec=codec,
                value=dense_lookup_layers,
            ),
        )

    index = StaticDecodingIndex.load_npz(static_decoding_index_path)
    validate_static_decoding_index_matches_codec(index=index, codec=codec)
    return index


def _semantic_id_depth(codec: SemanticIdCodec) -> int:
    depths = {len(semantic_id) for semantic_id in codec.item_to_semantic_id.values()}
    if len(depths) != 1:
        msg = f"모든 Semantic ID 길이가 같아야 합니다: {sorted(depths)}"
        raise ValueError(msg)
    depth = next(iter(depths), 0)
    if depth < 2:
        msg = "static_decoding build_static_index는 길이 2 이상의 Semantic ID가 필요합니다."
        raise ValueError(msg)
    return depth

"""Decoder benchmark Semantic ID input helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from recsys.benchmark.decoder_schema import DecoderBenchmarkError
from recsys.semantic_id import SemanticId, SemanticIdCodec


def load_semantic_ids(path: str | Path) -> tuple[SemanticId, ...]:
    """Semantic ID codec JSON에서 Semantic ID 목록을 로드한다."""
    codec = SemanticIdCodec.load_json(path)
    return tuple(
        codec.item_to_semantic_id[item_id] for item_id in sorted(codec.item_to_semantic_id)
    )


def generate_synthetic_semantic_ids(
    *,
    num_semantic_ids: int,
    depth: int,
    vocab_size: int,
    random_seed: int,
) -> tuple[SemanticId, ...]:
    """재현 가능한 synthetic Semantic ID 목록을 생성한다."""
    if num_semantic_ids < 1:
        msg = "num_semantic_ids는 1 이상이어야 합니다."
        raise DecoderBenchmarkError(msg)
    if depth < 1:
        msg = "depth는 1 이상이어야 합니다."
        raise DecoderBenchmarkError(msg)
    if vocab_size < 1:
        msg = "vocab_size는 1 이상이어야 합니다."
        raise DecoderBenchmarkError(msg)
    if vocab_size**depth < num_semantic_ids:
        msg = (
            "vocab_size와 depth 조합으로 요청한 수만큼 고유 Semantic ID를 만들 수 없습니다: "
            f"vocab_size={vocab_size}, depth={depth}, num_semantic_ids={num_semantic_ids}"
        )
        raise DecoderBenchmarkError(msg)

    rng = np.random.default_rng(random_seed)
    semantic_ids: set[SemanticId] = set()
    while len(semantic_ids) < num_semantic_ids:
        remaining = num_semantic_ids - len(semantic_ids)
        candidate_count = max(remaining * 2, 1024)
        candidates = rng.integers(
            0,
            vocab_size,
            size=(candidate_count, depth),
            dtype=np.int64,
        )
        for row in candidates:
            semantic_ids.add(tuple(int(token) for token in row))
            if len(semantic_ids) >= num_semantic_ids:
                break

    return tuple(sorted(semantic_ids))


def normalize_semantic_ids(semantic_ids: Sequence[Sequence[int]]) -> tuple[SemanticId, ...]:
    """Benchmark 입력 Semantic ID를 tuple 기반 정수 sequence로 정규화한다."""
    normalized = tuple(tuple(int(token) for token in semantic_id) for semantic_id in semantic_ids)
    if not normalized:
        msg = "benchmark Semantic ID 목록은 비어 있을 수 없습니다."
        raise DecoderBenchmarkError(msg)
    if any(not semantic_id for semantic_id in normalized):
        msg = "benchmark Semantic ID sequence는 비어 있을 수 없습니다."
        raise DecoderBenchmarkError(msg)
    return normalized


def infer_vocab_size(semantic_ids: Iterable[Sequence[int]]) -> int:
    """Semantic ID token 목록에서 필요한 vocab size를 추론한다."""
    max_token = max((token for semantic_id in semantic_ids for token in semantic_id), default=-1)
    return max_token + 1 if max_token >= 0 else 1

"""Semantic ID artifact 검증."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from recsys.semantic_id.codec import SemanticIdCodec


@dataclass(frozen=True)
class SemanticIdValidationResult:
    """Semantic ID artifact 검증 결과."""

    num_items: int
    semantic_id_length: int


def validate_semantic_id_file(path: str | Path) -> SemanticIdValidationResult:
    """Semantic ID JSON artifact를 로드하고 codec 조건을 검증한다."""
    codec = SemanticIdCodec.load_json(path)
    return SemanticIdValidationResult(
        num_items=codec.num_items,
        semantic_id_length=codec.semantic_id_length,
    )

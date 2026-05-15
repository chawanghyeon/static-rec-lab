"""Recommendation service interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RecommendedItem:
    """추천 결과 item."""

    item_id: int
    title: str
    semantic_id: tuple[int, ...]
    score: float


class RecommendationService(Protocol):
    """Recommendation endpoint가 사용하는 service contract."""

    model_name: str
    decoder_name: str

    def recommend(self, *, user_id: int, k: int) -> list[RecommendedItem]:
        """사용자별 추천 item을 반환한다."""

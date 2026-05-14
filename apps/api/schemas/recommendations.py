"""Recommendation API 응답 schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    """추천 item 응답."""

    item_id: int
    title: str
    semantic_id: list[int]
    score: float = Field(ge=0.0, le=1.0)


class RecommendationResponse(BaseModel):
    """사용자별 추천 응답."""

    user_id: int
    model: str
    decoder: str
    items: list[RecommendationItem]
    latency_ms: float = Field(ge=0.0)

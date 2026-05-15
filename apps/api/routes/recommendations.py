"""Recommendation API routes."""

from __future__ import annotations

import time
from typing import Annotated, cast

from fastapi import APIRouter, Depends, Path, Query, Request

from apps.api.schemas import RecommendationItem, RecommendationResponse
from apps.api.services import RecommendationService

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


def get_recommendation_service(request: Request) -> RecommendationService:
    """FastAPI app state에서 recommendation service를 꺼낸다."""
    return cast(RecommendationService, request.app.state.recommendation_service)


@router.get(
    "/users/{user_id}",
    response_model=RecommendationResponse,
    summary="사용자별 추천 item 조회",
)
def get_user_recommendations(
    user_id: Annotated[int, Path(ge=1)],
    recommendation_service: Annotated[
        RecommendationService,
        Depends(get_recommendation_service),
    ],
    k: Annotated[int, Query(ge=1)] = 20,
) -> RecommendationResponse:
    """설정된 recommendation service를 사용해 추천 결과를 반환한다."""
    started_at = time.perf_counter()
    items = recommendation_service.recommend(user_id=user_id, k=k)
    latency_ms = (time.perf_counter() - started_at) * 1000

    return RecommendationResponse(
        user_id=user_id,
        model=recommendation_service.model_name,
        decoder=recommendation_service.decoder_name,
        items=[
            RecommendationItem(
                item_id=item.item_id,
                title=item.title,
                semantic_id=list(item.semantic_id),
                score=item.score,
            )
            for item in items
        ],
        latency_ms=latency_ms,
    )

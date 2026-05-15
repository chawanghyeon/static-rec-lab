"""API service 패키지."""

from __future__ import annotations

import os
from pathlib import Path

from apps.api.services.base import RecommendationService, RecommendedItem
from apps.api.services.mock_recommender import MockRecommendationService
from apps.api.services.model_recommender import (
    ModelRecommendationConfig,
    ModelRecommendationService,
    load_user_histories,
)

__all__ = [
    "MockRecommendationService",
    "ModelRecommendationConfig",
    "ModelRecommendationService",
    "RecommendationService",
    "RecommendedItem",
    "build_recommendation_service_from_environment",
    "load_user_histories",
]


def build_recommendation_service_from_environment() -> RecommendationService:
    """모델 환경변수가 모두 없을 때만 mock service를 사용한다."""
    checkpoint_path = os.getenv("STATIC_REC_GENERATIVE_CHECKPOINT")
    semantic_id_path = os.getenv("STATIC_REC_SEMANTIC_ID_PATH")
    user_history_path = os.getenv("STATIC_REC_USER_HISTORY_PARQUET")
    configured_values = {
        "STATIC_REC_GENERATIVE_CHECKPOINT": checkpoint_path,
        "STATIC_REC_SEMANTIC_ID_PATH": semantic_id_path,
        "STATIC_REC_USER_HISTORY_PARQUET": user_history_path,
    }
    provided_values = {name: value for name, value in configured_values.items() if value}
    if not provided_values:
        return MockRecommendationService()
    if len(provided_values) != len(configured_values):
        missing_names = sorted(set(configured_values) - set(provided_values))
        msg = f"model-backed API 설정에 필요한 환경변수가 빠졌습니다: {missing_names}"
        raise ValueError(msg)

    paths = {name: Path(value) for name, value in provided_values.items() if value is not None}
    missing_paths = {name: path for name, path in paths.items() if not path.exists()}
    if missing_paths:
        formatted = {name: str(path) for name, path in missing_paths.items()}
        msg = f"model-backed API 설정 파일을 찾을 수 없습니다: {formatted}"
        raise FileNotFoundError(msg)

    return ModelRecommendationService.from_config(
        ModelRecommendationConfig(
            checkpoint_path=paths["STATIC_REC_GENERATIVE_CHECKPOINT"],
            semantic_id_path=paths["STATIC_REC_SEMANTIC_ID_PATH"],
            user_history_path=paths["STATIC_REC_USER_HISTORY_PARQUET"],
            device=os.getenv("STATIC_REC_DEVICE", "cpu"),
            beam_size=int(os.getenv("STATIC_REC_BEAM_SIZE", "50")),
        )
    )

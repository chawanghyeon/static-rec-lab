"""API service 패키지."""

from __future__ import annotations

from pathlib import Path

from apps.api.services.base import RecommendationService, RecommendedItem
from apps.api.services.model_recommender import (
    ModelRecommendationConfig,
    ModelRecommendationService,
    UserHistory,
    load_user_histories,
)

__all__ = [
    "DEFAULT_MODEL_CONFIG",
    "ModelRecommendationConfig",
    "ModelRecommendationService",
    "RecommendationService",
    "RecommendedItem",
    "UserHistory",
    "build_default_recommendation_service",
    "load_user_histories",
]


DEFAULT_MODEL_CONFIG = ModelRecommendationConfig(
    checkpoint_path=Path("artifacts/generative/ml-32m/model.pt"),
    semantic_id_path=Path("artifacts/semantic_id/ml-32m/semantic_ids.json"),
    user_history_path=Path("data/processed/ml-32m/valid.parquet"),
    device="cpu",
    beam_size=20,
    static_decoding_index_path=Path("artifacts/semantic_id/ml-32m/static_decoding_index.npz"),
)


def build_default_recommendation_service() -> RecommendationService:
    """기본 artifact 기반 recommendation service를 생성한다."""
    _validate_model_config_paths(DEFAULT_MODEL_CONFIG)
    return ModelRecommendationService.from_config(DEFAULT_MODEL_CONFIG)


def _validate_model_config_paths(config: ModelRecommendationConfig) -> None:
    missing_paths = {
        "checkpoint_path": config.checkpoint_path,
        "semantic_id_path": config.semantic_id_path,
        "user_history_path": config.user_history_path,
    }
    if config.static_decoding_index_path is not None:
        missing_paths["static_decoding_index_path"] = config.static_decoding_index_path
    missing_paths = {name: path for name, path in missing_paths.items() if not path.exists()}
    if missing_paths:
        formatted = {name: str(path) for name, path in missing_paths.items()}
        msg = f"serving artifact를 찾을 수 없습니다: {formatted}"
        raise FileNotFoundError(msg)

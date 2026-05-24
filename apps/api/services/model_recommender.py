"""Checkpoint 기반 generative retrieval recommendation service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch

from apps.api.services.base import RecommendedItem
from recsys.data import coerce_feedback_ids, coerce_item_ids
from recsys.decoding import (
    STATIC_DECODING_DECODER_NAME,
    StaticDecodingIndex,
    StaticDecodingTorchIndex,
    load_or_build_static_decoding_index,
)
from recsys.models import (
    generate_semantic_ids_with_static_decoding,
    load_checkpoint,
    resolve_torch_device,
)
from recsys.semantic_id import SemanticIdCodec, UnknownSemanticIdError


@dataclass(frozen=True)
class ModelRecommendationConfig:
    """Model-backed recommendation service 설정."""

    checkpoint_path: Path
    semantic_id_path: Path
    user_history_path: Path
    movie_metadata_path: Path
    device: str = "cpu"
    beam_size: int = 50
    static_decoding_index_path: Path | None = None


@dataclass(frozen=True)
class UserHistory:
    """API serving에 사용할 사용자 history."""

    item_ids: tuple[int, ...]
    feedback_ids: tuple[int, ...]


class ModelRecommendationService:
    """학습된 checkpoint와 STATIC decoder를 사용하는 recommendation service."""

    model_name = "generative-retrieval-static"

    def __init__(
        self,
        *,
        model: Any,
        item_to_index: Mapping[int, int],
        codec: SemanticIdCodec,
        user_histories: Mapping[int, UserHistory],
        movie_titles: Mapping[int, str],
        device: torch.device,
        beam_size: int = 50,
        static_decoding_index_path: Path | None = None,
    ) -> None:
        self._model = model
        self._item_to_index = dict(item_to_index)
        self._codec = codec
        self._device = device
        self.decoder_name = STATIC_DECODING_DECODER_NAME
        self._static_decoding_index: StaticDecodingIndex = load_or_build_static_decoding_index(
            codec=codec,
            static_decoding_index_path=static_decoding_index_path,
        )
        self._static_decoding_torch_index: StaticDecodingTorchIndex = (
            self._static_decoding_index.to_torch(device)
        )
        self._user_histories = {
            int(user_id): history for user_id, history in user_histories.items()
        }
        self._movie_titles = dict(movie_titles)
        _validate_movie_titles(codec=codec, movie_titles=self._movie_titles)
        self._beam_size = beam_size

    @classmethod
    def from_config(cls, config: ModelRecommendationConfig) -> ModelRecommendationService:
        """파일 경로 기반 설정에서 service를 생성한다."""
        device = resolve_torch_device(config.device)
        model, item_to_index = load_checkpoint(config.checkpoint_path, device=device)
        codec = SemanticIdCodec.load_json(config.semantic_id_path)
        user_histories = load_user_histories(config.user_history_path)
        movie_titles = load_movie_titles(config.movie_metadata_path)
        return cls(
            model=model,
            item_to_index=item_to_index,
            codec=codec,
            user_histories=user_histories,
            movie_titles=movie_titles,
            device=device,
            beam_size=config.beam_size,
            static_decoding_index_path=config.static_decoding_index_path,
        )

    def recommend(self, *, user_id: int, k: int) -> list[RecommendedItem]:
        """학습된 모델로 user_id에 대한 추천을 생성한다."""
        if k < 1:
            msg = "k는 1 이상이어야 합니다."
            raise ValueError(msg)

        history = self._user_histories.get(user_id, UserHistory(item_ids=(), feedback_ids=()))
        history_item_ids = set(history.item_ids)
        result_limit = max(self._beam_size, k + len(history_item_ids))
        beam_results = generate_semantic_ids_with_static_decoding(
            model=self._model,
            index=self._static_decoding_index,
            torch_index=self._static_decoding_torch_index,
            history_item_ids=history.item_ids,
            history_feedback_ids=history.feedback_ids,
            item_to_index=self._item_to_index,
            beam_size=result_limit,
            max_results=result_limit,
            device=self._device,
        )
        recommendations: list[RecommendedItem] = []
        seen_items: set[int] = set()
        for result in beam_results:
            try:
                item_id = self._codec.decode_semantic_id(result.semantic_id)
            except UnknownSemanticIdError:
                continue
            if item_id in history_item_ids or item_id in seen_items:
                continue
            recommendations.append(
                RecommendedItem(
                    item_id=item_id,
                    title=self._movie_titles[item_id],
                    semantic_id=result.semantic_id,
                    score=round(result.score, 6),
                )
            )
            seen_items.add(item_id)
            if len(recommendations) >= k:
                break
        return recommendations


def load_user_histories(path: str | Path) -> dict[int, UserHistory]:
    """전처리 parquet에서 user_id별 최신 history를 로드한다."""
    frame = pd.read_parquet(path)
    required_columns = {"user_id", "history_item_ids", "history_feedback_ids"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        msg = f"user history parquet에 필요한 컬럼이 없습니다: {missing_columns}"
        raise ValueError(msg)

    if "target_timestamp" in frame.columns:
        frame = frame.sort_values(["user_id", "target_timestamp"], kind="mergesort")

    histories: dict[int, UserHistory] = {}
    for row in frame.itertuples(index=False):
        user_id = int(cast(Any, row.user_id))
        item_ids = coerce_item_ids(row.history_item_ids)
        feedback_ids = coerce_feedback_ids(row.history_feedback_ids)
        if len(item_ids) != len(feedback_ids):
            msg = "history_item_ids와 history_feedback_ids 길이가 같아야 합니다."
            raise ValueError(msg)
        histories[user_id] = UserHistory(item_ids=item_ids, feedback_ids=feedback_ids)
    return histories


def load_movie_titles(path: str | Path) -> dict[int, str]:
    """MovieLens movies.csv에서 movieId별 title을 로드한다."""
    frame = pd.read_csv(path)
    required_columns = {"movieId", "title"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        msg = f"MovieLens metadata에 필요한 컬럼이 없습니다: {missing_columns}"
        raise ValueError(msg)

    titles: dict[int, str] = {}
    for row in frame.itertuples(index=False):
        movie_id = int(cast(Any, row.movieId))
        title = str(cast(Any, row.title))
        titles[movie_id] = title
    return titles


def _validate_movie_titles(
    *,
    codec: SemanticIdCodec,
    movie_titles: Mapping[int, str],
) -> None:
    missing_item_ids = sorted(set(codec.item_to_semantic_id) - set(movie_titles))
    if missing_item_ids:
        sample = missing_item_ids[:10]
        msg = (
            f"semantic ID artifact에 있지만 MovieLens metadata에 없는 item_id가 있습니다: {sample}"
        )
        raise ValueError(msg)

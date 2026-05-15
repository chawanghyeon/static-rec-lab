"""Checkpoint 기반 generative retrieval recommendation service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch

from apps.api.services.base import RecommendedItem
from recsys.decoding import StaticTransitionMatrixDecoder
from recsys.models import generate_semantic_ids, load_checkpoint
from recsys.semantic_id import SemanticIdCodec, UnknownSemanticIdError


@dataclass(frozen=True)
class ModelRecommendationConfig:
    """Model-backed recommendation service 설정."""

    checkpoint_path: Path
    semantic_id_path: Path
    user_history_path: Path
    device: str = "cpu"
    beam_size: int = 50


class ModelRecommendationService:
    """학습된 checkpoint와 STATIC decoder를 사용하는 recommendation service."""

    model_name = "generative-retrieval-static"
    decoder_name = "static_sparse_matrix"

    def __init__(
        self,
        *,
        model: Any,
        item_to_index: Mapping[int, int],
        codec: SemanticIdCodec,
        user_histories: Mapping[int, Sequence[int]],
        device: torch.device,
        beam_size: int = 50,
    ) -> None:
        self._model = model
        self._item_to_index = dict(item_to_index)
        self._codec = codec
        self._decoder = StaticTransitionMatrixDecoder.from_codec(codec)
        self._user_histories = {
            int(user_id): tuple(int(item_id) for item_id in history)
            for user_id, history in user_histories.items()
        }
        self._device = device
        self._beam_size = beam_size

    @classmethod
    def from_config(cls, config: ModelRecommendationConfig) -> ModelRecommendationService:
        """파일 경로 기반 설정에서 service를 생성한다."""
        device = _resolve_device(config.device)
        model, item_to_index = load_checkpoint(config.checkpoint_path, device=device)
        codec = SemanticIdCodec.load_json(config.semantic_id_path)
        user_histories = load_user_histories(config.user_history_path)
        return cls(
            model=model,
            item_to_index=item_to_index,
            codec=codec,
            user_histories=user_histories,
            device=device,
            beam_size=config.beam_size,
        )

    def recommend(self, *, user_id: int, k: int) -> list[RecommendedItem]:
        """학습된 모델로 user_id에 대한 추천을 생성한다."""
        if k < 1:
            msg = "k는 1 이상이어야 합니다."
            raise ValueError(msg)

        history = self._user_histories.get(user_id, ())
        beam_results = generate_semantic_ids(
            model=self._model,
            decoder=self._decoder,
            history_item_ids=history,
            item_to_index=self._item_to_index,
            beam_size=max(self._beam_size, k),
            max_results=k,
            device=self._device,
        )
        recommendations: list[RecommendedItem] = []
        seen_items: set[int] = set()
        for result in beam_results:
            try:
                item_id = self._codec.decode_semantic_id(result.semantic_id)
            except UnknownSemanticIdError:
                continue
            if item_id in seen_items:
                continue
            recommendations.append(
                RecommendedItem(
                    item_id=item_id,
                    title=f"Item {item_id}",
                    semantic_id=result.semantic_id,
                    score=round(result.score, 6),
                )
            )
            seen_items.add(item_id)
            if len(recommendations) >= k:
                break
        return recommendations


def load_user_histories(path: str | Path) -> dict[int, tuple[int, ...]]:
    """전처리 parquet에서 user_id별 최신 history를 로드한다."""
    frame = pd.read_parquet(path)
    required_columns = {"user_id", "history_item_ids"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        msg = f"user history parquet에 필요한 컬럼이 없습니다: {missing_columns}"
        raise ValueError(msg)

    if "target_timestamp" in frame.columns:
        frame = frame.sort_values(["user_id", "target_timestamp"], kind="mergesort")

    histories: dict[int, tuple[int, ...]] = {}
    for row in frame.itertuples(index=False):
        user_id = int(cast(Any, row.user_id))
        histories[user_id] = _normalize_history(row.history_item_ids)
    return histories


def _normalize_history(history: object) -> tuple[int, ...]:
    if history is None:
        return ()
    if isinstance(history, float) and pd.isna(history):
        return ()
    if isinstance(history, str):
        msg = "history_item_ids는 문자열이 아니라 정수 sequence여야 합니다."
        raise ValueError(msg)
    return tuple(int(cast(Any, item_id)) for item_id in cast(Sequence[object], history))


def _resolve_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        return torch.device("cuda")
    if value == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

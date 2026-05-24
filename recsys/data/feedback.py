"""Explicit feedback helpers for MovieLens ratings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral
from typing import Any, Final, cast

PAD_FEEDBACK_ID: Final[int] = 0
NEGATIVE_FEEDBACK_ID: Final[int] = 1
NEUTRAL_FEEDBACK_ID: Final[int] = 2
POSITIVE_FEEDBACK_ID: Final[int] = 3
FEEDBACK_VOCAB_SIZE: Final[int] = 4


def feedback_id_for_rating(
    rating: float,
    *,
    positive_rating: float = 4.0,
    neutral_rating: float = 3.0,
) -> int:
    """MovieLens rating을 negative/neutral/positive feedback id로 변환한다."""
    if neutral_rating > positive_rating:
        msg = "neutral_rating은 positive_rating보다 클 수 없습니다."
        raise ValueError(msg)
    if rating >= positive_rating:
        return POSITIVE_FEEDBACK_ID
    if rating >= neutral_rating:
        return NEUTRAL_FEEDBACK_ID
    return NEGATIVE_FEEDBACK_ID


def coerce_feedback_ids(
    value: object,
    *,
    field_name: str = "history_feedback_ids",
) -> tuple[int, ...]:
    """정수 feedback id sequence를 tuple로 정규화한다."""
    if value is None:
        return ()
    if isinstance(value, float) and math.isnan(value):
        return ()
    if isinstance(value, str):
        msg = f"{field_name}는 문자열이 아니라 정수 sequence여야 합니다."
        raise ValueError(msg)
    if isinstance(value, Integral):
        return (_validate_feedback_id(int(value), field_name=field_name),)
    return tuple(
        _validate_feedback_id(int(cast(Any, feedback_id)), field_name=field_name)
        for feedback_id in cast(Sequence[object], value)
    )


def _validate_feedback_id(feedback_id: int, *, field_name: str) -> int:
    if feedback_id <= PAD_FEEDBACK_ID or feedback_id >= FEEDBACK_VOCAB_SIZE:
        msg = f"{field_name}에는 실제 interaction feedback id만 들어갈 수 있습니다: {feedback_id}"
        raise ValueError(msg)
    return feedback_id

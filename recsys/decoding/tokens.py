"""Decoder token 정규화 helper."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def normalize_tokens(tokens: Iterable[Any], *, allow_empty: bool = False) -> tuple[int, ...]:
    """정수 token sequence를 tuple로 정규화한다."""
    if isinstance(tokens, str):
        msg = "token sequence는 문자열이 아니라 정수 iterable이어야 합니다."
        raise ValueError(msg)

    normalized = tuple(normalize_token(token) for token in tokens)
    if not normalized and not allow_empty:
        msg = "Semantic ID sequence는 비어 있을 수 없습니다."
        raise ValueError(msg)
    return normalized


def try_normalize_tokens(tokens: Iterable[Any], *, allow_empty: bool) -> tuple[int, ...] | None:
    """정규화 실패 시 None을 반환한다."""
    try:
        return normalize_tokens(tokens, allow_empty=allow_empty)
    except ValueError:
        return None


def normalize_token(token: Any) -> int:
    """단일 decoder token을 음수가 아닌 정수로 정규화한다."""
    if isinstance(token, (bool, np.bool_)) or not isinstance(token, (int, np.integer)):
        msg = f"token은 bool이 아닌 정수여야 합니다: {token!r}"
        raise ValueError(msg)
    normalized = int(token)
    if normalized < 0:
        msg = f"token은 0 이상이어야 합니다: {token}"
        raise ValueError(msg)
    return normalized

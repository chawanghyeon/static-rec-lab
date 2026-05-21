"""Sequential recommendation item sequence helper."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral
from typing import Any, cast


def coerce_item_ids(value: object, *, field_name: str = "history_item_ids") -> tuple[int, ...]:
    """정수 item_id sequence를 tuple로 정규화한다."""
    if value is None:
        return ()
    if isinstance(value, float) and math.isnan(value):
        return ()
    if isinstance(value, str):
        msg = f"{field_name}는 문자열이 아니라 정수 sequence여야 합니다."
        raise ValueError(msg)
    if isinstance(value, Integral):
        return (int(value),)
    return tuple(int(cast(Any, item_id)) for item_id in cast(Sequence[object], value))

"""static_decoding benchmark 공통 helper."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def throughput_rows_per_second(
    *,
    batch_size: int,
    iterations: int,
    elapsed_ns: int,
) -> float:
    """측정 시간을 rows/s로 변환한다."""
    elapsed_seconds = elapsed_ns / 1_000_000_000
    return batch_size * iterations / elapsed_seconds


def jax_module() -> Any:
    """JAX module을 지연 import한다."""
    return import_module("jax")


def jax_numpy() -> Any:
    """jax.numpy module을 지연 import한다."""
    return import_module("jax.numpy")


def block_until_ready(value: Any) -> None:
    """JAX 비동기 실행 결과를 benchmark 측정 전에 동기화한다."""
    if isinstance(value, tuple):
        for item in value:
            block_until_ready(item)
        return
    block_until_ready_method = getattr(value, "block_until_ready", None)
    if callable(block_until_ready_method):
        block_until_ready_method()

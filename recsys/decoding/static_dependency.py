"""외부 `static_decoding` package dependency loader."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, cast

import numpy as np
import torch

STATIC_DECODING_REPOSITORY = "https://github.com/youtube/static-constraint-decoding"
STATIC_DECODING_COMMIT = "c24f9dc8b9b8045716fff7ef750a1f0cb31c6f57"

StaticDecodingBuildIndex = Callable[
    [np.ndarray, int, int],
    tuple[np.ndarray, np.ndarray, tuple[int, ...], np.ndarray, np.ndarray, np.ndarray],
]
StaticDecodingSparseMaskKernel = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.device],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]
StaticDecodingJaxSparseMaskKernel = Callable[
    [Any, Any, Any, Any, int, int],
    tuple[Any, Any, Any],
]
StaticDecodingJaxSparseTransitionHarness = Callable[
    [
        Any,
        Any,
        int,
        int,
        int,
        int,
        int,
        int,
        tuple[int, ...],
        Any,
        Any,
        Any,
        Any,
        Any,
        int,
    ],
    Any,
]
StaticDecodingSparseTransitionHarness = Callable[
    [
        torch.nn.Module,
        int,
        int,
        int,
        int,
        int,
        int,
        tuple[int, ...],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.device,
        int,
    ],
    torch.Tensor,
]
StaticDecodingRandomModelFactory = Callable[[int, torch.device], torch.nn.Module]
StaticDecodingJaxRandomModelFactory = Callable[[int], Any]


class StaticDecodingDependencyError(ImportError):
    """STATIC package를 사용할 수 없을 때 발생하는 오류."""


def load_static_decoding_build_static_index() -> StaticDecodingBuildIndex:
    try:
        module = import_module("static_decoding.csr_utils")
    except ImportError as exc:
        msg = (
            "static_decoding package `static_decoding`을 import할 수 없습니다. "
            "`uv lock` 및 `uv sync` 후 다시 실행하세요."
        )
        raise StaticDecodingDependencyError(msg) from exc
    return cast(StaticDecodingBuildIndex, module.build_static_index)


def load_static_decoding_sparse_mask_kernel() -> StaticDecodingSparseMaskKernel:
    try:
        module = import_module("static_decoding.decoding_pt")
    except ImportError as exc:
        msg = (
            "static_decoding PyTorch decoder를 import할 수 없습니다. "
            "`uv lock` 및 `uv sync` 후 다시 실행하세요."
        )
        raise StaticDecodingDependencyError(msg) from exc
    return cast(StaticDecodingSparseMaskKernel, module.generate_and_apply_logprobs_mask)


def load_static_decoding_jax_sparse_mask_kernel() -> StaticDecodingJaxSparseMaskKernel:
    try:
        module = import_module("static_decoding.decoding_jax")
    except ImportError as exc:
        msg = (
            "static_decoding JAX decoder를 import할 수 없습니다. "
            "`uv lock` 및 `uv sync` 후 다시 실행하세요."
        )
        raise StaticDecodingDependencyError(msg) from exc
    return cast(StaticDecodingJaxSparseMaskKernel, module.generate_and_apply_logprobs_mask)


def load_static_decoding_jax_sparse_transition_harness() -> (
    StaticDecodingJaxSparseTransitionHarness
):
    try:
        module = import_module("static_decoding.decoding_jax")
    except ImportError as exc:
        msg = (
            "static_decoding JAX decoder를 import할 수 없습니다. "
            "`uv lock` 및 `uv sync` 후 다시 실행하세요."
        )
        raise StaticDecodingDependencyError(msg) from exc
    return cast(StaticDecodingJaxSparseTransitionHarness, module.sparse_transition_jax)


def load_static_decoding_jax_random_model_factory() -> StaticDecodingJaxRandomModelFactory:
    try:
        module = import_module("static_decoding.decoding_jax")
    except ImportError as exc:
        msg = (
            "static_decoding JAX decoder를 import할 수 없습니다. "
            "`uv lock` 및 `uv sync` 후 다시 실행하세요."
        )
        raise StaticDecodingDependencyError(msg) from exc
    return cast(StaticDecodingJaxRandomModelFactory, module.RandomModel)


def load_static_decoding_jax() -> Any:
    try:
        return import_module("jax")
    except ImportError as exc:
        msg = (
            "JAX를 import할 수 없습니다. static_decoding JAX decoder를 사용하려면 "
            "`uv lock` 및 `uv sync` 후 다시 실행하세요."
        )
        raise StaticDecodingDependencyError(msg) from exc


def load_static_decoding_jax_numpy() -> Any:
    try:
        return import_module("jax.numpy")
    except ImportError as exc:
        msg = (
            "JAX numpy를 import할 수 없습니다. static_decoding JAX decoder를 사용하려면 "
            "`uv lock` 및 `uv sync` 후 다시 실행하세요."
        )
        raise StaticDecodingDependencyError(msg) from exc


def load_static_decoding_sparse_transition_harness() -> StaticDecodingSparseTransitionHarness:
    try:
        module = import_module("static_decoding.decoding_pt")
    except ImportError as exc:
        msg = (
            "static_decoding PyTorch decoder를 import할 수 없습니다. "
            "`uv lock` 및 `uv sync` 후 다시 실행하세요."
        )
        raise StaticDecodingDependencyError(msg) from exc
    return cast(StaticDecodingSparseTransitionHarness, module.sparse_transition_torch)


def load_static_decoding_random_model_factory() -> StaticDecodingRandomModelFactory:
    try:
        module = import_module("static_decoding.decoding_pt")
    except ImportError as exc:
        msg = (
            "static_decoding PyTorch decoder를 import할 수 없습니다. "
            "`uv lock` 및 `uv sync` 후 다시 실행하세요."
        )
        raise StaticDecodingDependencyError(msg) from exc
    return cast(StaticDecodingRandomModelFactory, module.RandomModel)

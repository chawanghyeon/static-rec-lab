"""STATIC decoding runtime kernel/harness wrapper."""

from __future__ import annotations

from typing import Any

import torch

from recsys.decoding.static_artifact import StaticDecodingIndex, StaticDecodingTorchIndex
from recsys.decoding.static_dependency import (
    load_static_decoding_jax,
    load_static_decoding_jax_numpy,
    load_static_decoding_jax_random_model_factory,
    load_static_decoding_jax_sparse_mask_kernel,
    load_static_decoding_jax_sparse_transition_harness,
    load_static_decoding_random_model_factory,
    load_static_decoding_sparse_mask_kernel,
    load_static_decoding_sparse_transition_harness,
)


def static_decoding_generate_and_apply_logprobs_mask(
    *,
    flat_logprobs: torch.Tensor,
    flat_states: torch.Tensor,
    index: StaticDecodingTorchIndex,
    prefix_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """static_decoding PyTorch sparse candidate extraction kernel을 호출한다."""
    if prefix_length < index.dense_lookup_layers:
        msg = (
            "static_decoding sparse kernel은 dense lookup 이후 prefix에서만 호출할 수 있습니다: "
            f"prefix_length={prefix_length}, dense_lookup_layers={index.dense_lookup_layers}"
        )
        raise ValueError(msg)
    if prefix_length < 0 or prefix_length >= index.semantic_id_depth:
        msg = f"prefix_length가 Semantic ID depth 범위를 벗어났습니다: {prefix_length}"
        raise ValueError(msg)

    limit = max(1, int(index.layer_max_branches[prefix_length]))
    kernel = load_static_decoding_sparse_mask_kernel()
    return kernel(
        flat_logprobs,
        flat_states,
        index.packed_csr,
        index.csr_indptr,
        limit,
        index.vocab_size,
        index.device,
    )


def static_decoding_generate_and_apply_logprobs_mask_jax(
    *,
    flat_logprobs: Any,
    flat_states: Any,
    index: StaticDecodingIndex,
    prefix_length: int,
) -> tuple[Any, Any, Any]:
    """static_decoding JAX sparse candidate extraction kernel을 호출한다."""
    if prefix_length < index.dense_lookup_layers:
        msg = (
            "static_decoding JAX sparse kernel은 dense lookup 이후 prefix에서만 "
            "호출할 수 있습니다: "
            f"prefix_length={prefix_length}, dense_lookup_layers={index.dense_lookup_layers}"
        )
        raise ValueError(msg)
    if prefix_length < 0 or prefix_length >= index.semantic_id_depth:
        msg = f"prefix_length가 Semantic ID depth 범위를 벗어났습니다: {prefix_length}"
        raise ValueError(msg)

    jnp = load_static_decoding_jax_numpy()
    limit = max(1, int(index.layer_max_branches[prefix_length]))
    kernel = load_static_decoding_jax_sparse_mask_kernel()
    return kernel(
        flat_logprobs,
        flat_states,
        jnp.asarray(index.packed_csr),
        jnp.asarray(index.csr_indptr),
        limit,
        index.vocab_size,
    )


@torch.inference_mode()
def static_decoding_sparse_transition_torch(
    *,
    model: torch.nn.Module,
    index: StaticDecodingIndex,
    batch_size: int,
    beam_size: int,
    tokens_per_beam: int,
    start_token: int,
    device: torch.device,
    torch_index: StaticDecodingTorchIndex | None = None,
) -> torch.Tensor:
    """static_decoding `sparse_transition_torch` decoding harness를 호출한다."""
    if batch_size < 1:
        msg = "batch_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    if beam_size < 1:
        msg = "beam_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    if tokens_per_beam < 1:
        msg = "tokens_per_beam은 1 이상이어야 합니다."
        raise ValueError(msg)
    if index.dense_lookup_layers not in (1, 2):
        msg = "static_decoding PyTorch harness는 dense_lookup_layers=1 또는 2만 지원합니다."
        raise ValueError(msg)

    resolved_torch_index = index.to_torch(device) if torch_index is None else torch_index
    if resolved_torch_index.device != device:
        msg = (
            "torch_index device가 요청한 device와 다릅니다: "
            f"torch_index={resolved_torch_index.device}, device={device}"
        )
        raise ValueError(msg)
    harness = load_static_decoding_sparse_transition_harness()
    return harness(
        model,
        batch_size,
        beam_size,
        tokens_per_beam,
        start_token,
        index.semantic_id_depth,
        index.vocab_size,
        index.layer_max_branches,
        resolved_torch_index.packed_csr,
        resolved_torch_index.csr_indptr,
        resolved_torch_index.start_mask,
        resolved_torch_index.dense_mask,
        resolved_torch_index.dense_states,
        device,
        index.dense_lookup_layers,
    )


def static_decoding_sparse_transition_jax(
    *,
    index: StaticDecodingIndex,
    batch_size: int,
    beam_size: int,
    tokens_per_beam: int,
    start_token: int,
    random_seed: int = 0,
    model: Any | None = None,
    key: Any | None = None,
) -> Any:
    """static_decoding `sparse_transition_jax` decoding harness를 호출한다."""
    if batch_size < 1:
        msg = "batch_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    if beam_size < 1:
        msg = "beam_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    if tokens_per_beam < 1:
        msg = "tokens_per_beam은 1 이상이어야 합니다."
        raise ValueError(msg)
    if index.dense_lookup_layers not in (1, 2):
        msg = "static_decoding JAX harness는 dense_lookup_layers=1 또는 2만 지원합니다."
        raise ValueError(msg)

    jax = load_static_decoding_jax()
    jnp = load_static_decoding_jax_numpy()
    resolved_model = (
        build_static_decoding_jax_random_model(vocab_size=index.vocab_size)
        if model is None
        else model
    )
    resolved_key = jax.random.PRNGKey(random_seed) if key is None else key
    harness = load_static_decoding_jax_sparse_transition_harness()
    return harness(
        resolved_model,
        resolved_key,
        batch_size,
        beam_size,
        tokens_per_beam,
        start_token,
        index.semantic_id_depth,
        index.vocab_size,
        index.layer_max_branches,
        jnp.asarray(index.packed_csr),
        jnp.asarray(index.csr_indptr),
        jnp.asarray(index.start_mask),
        jnp.asarray(index.dense_mask),
        jnp.asarray(index.dense_states),
        index.dense_lookup_layers,
    )


def build_static_decoding_random_model(*, vocab_size: int, device: torch.device) -> torch.nn.Module:
    """STATIC PyTorch `RandomModel` benchmark model을 만든다."""
    if vocab_size < 1:
        msg = "vocab_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    factory = load_static_decoding_random_model_factory()
    return factory(vocab_size, device)


def build_static_decoding_jax_random_model(*, vocab_size: int) -> Any:
    """STATIC JAX `RandomModel` benchmark model을 만든다."""
    if vocab_size < 1:
        msg = "vocab_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    factory = load_static_decoding_jax_random_model_factory()
    return factory(vocab_size)

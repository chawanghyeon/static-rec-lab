"""STATIC decoding compatibility exports."""

from __future__ import annotations

from recsys.decoding.static_artifact import StaticDecodingIndex, StaticDecodingTorchIndex
from recsys.decoding.static_beam_search import (
    StaticDecodingBeamLogitProvider,
    static_decoding_constrained_beam_search,
)
from recsys.decoding.static_runtime import (
    build_static_decoding_jax_random_model,
    build_static_decoding_random_model,
    static_decoding_generate_and_apply_logprobs_mask,
    static_decoding_generate_and_apply_logprobs_mask_jax,
    static_decoding_sparse_transition_jax,
    static_decoding_sparse_transition_torch,
)

__all__ = [
    "StaticDecodingBeamLogitProvider",
    "StaticDecodingIndex",
    "StaticDecodingTorchIndex",
    "build_static_decoding_jax_random_model",
    "build_static_decoding_random_model",
    "static_decoding_constrained_beam_search",
    "static_decoding_generate_and_apply_logprobs_mask",
    "static_decoding_generate_and_apply_logprobs_mask_jax",
    "static_decoding_sparse_transition_jax",
    "static_decoding_sparse_transition_torch",
]

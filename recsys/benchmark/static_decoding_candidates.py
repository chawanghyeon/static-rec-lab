"""static_decoding sparse candidate benchmark."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from recsys.benchmark.decoder_schema import (
    DecoderBenchmarkError,
    StaticDecodingJaxKernelBenchmarkResult,
    StaticDecodingKernelBenchmarkResult,
)
from recsys.benchmark.static_decoding_helpers import (
    block_until_ready,
    jax_numpy,
    throughput_rows_per_second,
)
from recsys.decoding import (
    StaticDecodingIndex,
    StaticDecodingTorchIndex,
    static_decoding_generate_and_apply_logprobs_mask,
    static_decoding_generate_and_apply_logprobs_mask_jax,
)


def benchmark_static_decoding_candidates(
    *,
    index: StaticDecodingIndex,
    torch_index: StaticDecodingTorchIndex,
    prefix_batches: np.ndarray,
    state_batches: np.ndarray,
    batch_size: int,
    warmup_iterations: int,
    iterations: int,
) -> tuple[StaticDecodingKernelBenchmarkResult, StaticDecodingJaxKernelBenchmarkResult]:
    """PyTorch/JAX sparse candidate extraction benchmark를 실행한다."""
    if not _verify_torch_candidate_equivalence(
        index=index,
        torch_index=torch_index,
        prefix_batches=prefix_batches,
        state_batches=state_batches,
    ):
        msg = f"batch_size={batch_size}에서 static_decoding 후보 token이 index와 다릅니다."
        raise DecoderBenchmarkError(msg)

    elapsed_ns, checksum = _measure_torch_candidate_generation(
        torch_index=torch_index,
        state_batches=state_batches,
        warmup_iterations=warmup_iterations,
    )
    if checksum < 1:
        msg = f"batch_size={batch_size}에서 static_decoding 후보 checksum이 비어 있습니다."
        raise DecoderBenchmarkError(msg)
    torch_result = StaticDecodingKernelBenchmarkResult(
        batch_size=batch_size,
        candidates_identical=True,
        static_decoding_latency_ms=elapsed_ns / iterations / 1_000_000,
        static_decoding_throughput_rows_per_s=throughput_rows_per_second(
            batch_size=batch_size,
            iterations=iterations,
            elapsed_ns=elapsed_ns,
        ),
    )

    if not _verify_jax_candidate_equivalence(
        index=index,
        prefix_batches=prefix_batches,
        state_batches=state_batches,
    ):
        msg = f"batch_size={batch_size}에서 static_decoding JAX 후보 token이 다릅니다."
        raise DecoderBenchmarkError(msg)

    jax_elapsed_ns, jax_checksum = _measure_jax_candidate_generation(
        index=index,
        state_batches=state_batches,
        warmup_iterations=warmup_iterations,
    )
    if jax_checksum < 1:
        msg = f"batch_size={batch_size}에서 static_decoding JAX 후보 checksum이 비어 있습니다."
        raise DecoderBenchmarkError(msg)
    jax_result = StaticDecodingJaxKernelBenchmarkResult(
        batch_size=batch_size,
        candidates_identical=True,
        static_decoding_jax_latency_ms=jax_elapsed_ns / iterations / 1_000_000,
        static_decoding_jax_throughput_rows_per_s=throughput_rows_per_second(
            batch_size=batch_size,
            iterations=iterations,
            elapsed_ns=jax_elapsed_ns,
        ),
    )
    return torch_result, jax_result


def _verify_torch_candidate_equivalence(
    *,
    index: StaticDecodingIndex,
    torch_index: StaticDecodingTorchIndex,
    prefix_batches: np.ndarray,
    state_batches: np.ndarray,
) -> bool:
    for prefix_batch, state_batch in zip(prefix_batches, state_batches, strict=True):
        candidate_logprobs, candidate_tokens, _ = _torch_candidates_for_states(
            torch_index=torch_index,
            state_batch=state_batch,
            vocab_size=index.vocab_size,
            prefix_length=index.dense_lookup_layers,
        )
        for row_index, prefix in enumerate(prefix_batch):
            finite_tokens = tuple(
                int(token)
                for token in candidate_tokens[row_index][
                    torch.isfinite(candidate_logprobs[row_index])
                ]
            )
            if finite_tokens != index.allowed_next_tokens(prefix.tolist()):
                return False
    return True


def _measure_torch_candidate_generation(
    *,
    torch_index: StaticDecodingTorchIndex,
    state_batches: np.ndarray,
    warmup_iterations: int,
) -> tuple[int, int]:
    checksum = 0
    for states in state_batches[:warmup_iterations]:
        candidate_logprobs, _, _ = _torch_candidates_for_states(
            torch_index=torch_index,
            state_batch=states,
            vocab_size=torch_index.vocab_size,
            prefix_length=torch_index.dense_lookup_layers,
        )
        checksum += int(torch.isfinite(candidate_logprobs).sum().item())

    start_ns = time.perf_counter_ns()
    for states in state_batches[warmup_iterations:]:
        candidate_logprobs, _, _ = _torch_candidates_for_states(
            torch_index=torch_index,
            state_batch=states,
            vocab_size=torch_index.vocab_size,
            prefix_length=torch_index.dense_lookup_layers,
        )
        checksum += int(torch.isfinite(candidate_logprobs).sum().item())
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns, checksum


def _verify_jax_candidate_equivalence(
    *,
    index: StaticDecodingIndex,
    prefix_batches: np.ndarray,
    state_batches: np.ndarray,
) -> bool:
    for prefix_batch, state_batch in zip(prefix_batches, state_batches, strict=True):
        candidate_logprobs, candidate_tokens, _ = _jax_candidates_for_states(
            index=index,
            state_batch=state_batch,
            vocab_size=index.vocab_size,
            prefix_length=index.dense_lookup_layers,
        )
        candidate_logprobs_array = np.asarray(candidate_logprobs)
        candidate_tokens_array = np.asarray(candidate_tokens)
        for row_index, prefix in enumerate(prefix_batch):
            finite_tokens = tuple(
                int(token)
                for token in candidate_tokens_array[row_index][
                    np.isfinite(candidate_logprobs_array[row_index])
                ]
            )
            if finite_tokens != index.allowed_next_tokens(prefix.tolist()):
                return False
    return True


def _measure_jax_candidate_generation(
    *,
    index: StaticDecodingIndex,
    state_batches: np.ndarray,
    warmup_iterations: int,
) -> tuple[int, int]:
    checksum = 0
    for states in state_batches[:warmup_iterations]:
        candidate_logprobs, _, _ = _jax_candidates_for_states(
            index=index,
            state_batch=states,
            vocab_size=index.vocab_size,
            prefix_length=index.dense_lookup_layers,
        )
        checksum += int(np.isfinite(np.asarray(candidate_logprobs)).sum())

    start_ns = time.perf_counter_ns()
    for states in state_batches[warmup_iterations:]:
        candidate_logprobs, _, _ = _jax_candidates_for_states(
            index=index,
            state_batch=states,
            vocab_size=index.vocab_size,
            prefix_length=index.dense_lookup_layers,
        )
        checksum += int(np.isfinite(np.asarray(candidate_logprobs)).sum())
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns, checksum


def _torch_candidates_for_states(
    *,
    torch_index: StaticDecodingTorchIndex,
    state_batch: np.ndarray,
    vocab_size: int,
    prefix_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = torch.as_tensor(state_batch, dtype=torch.long)
    logprobs = torch.zeros((len(state_batch), vocab_size), dtype=torch.float32)
    return static_decoding_generate_and_apply_logprobs_mask(
        flat_logprobs=logprobs,
        flat_states=states,
        index=torch_index,
        prefix_length=prefix_length,
    )


def _jax_candidates_for_states(
    *,
    index: StaticDecodingIndex,
    state_batch: np.ndarray,
    vocab_size: int,
    prefix_length: int,
) -> tuple[Any, Any, Any]:
    jnp = jax_numpy()
    states = jnp.asarray(state_batch, dtype=jnp.int32)
    logprobs = jnp.zeros((len(state_batch), vocab_size), dtype=jnp.float32)
    result = static_decoding_generate_and_apply_logprobs_mask_jax(
        flat_logprobs=logprobs,
        flat_states=states,
        index=index,
        prefix_length=prefix_length,
    )
    block_until_ready(result)
    return result

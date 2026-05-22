"""static_decoding PyTorch/JAX benchmark 실행 로직."""

from __future__ import annotations

import time
from collections.abc import Sequence
from importlib import import_module
from typing import Any

import numpy as np
import torch

from recsys.benchmark.decoder_schema import (
    DecoderBenchmarkError,
    StaticDecodingHarnessBenchmarkResult,
    StaticDecodingJaxHarnessBenchmarkResult,
    StaticDecodingJaxKernelBenchmarkResult,
    StaticDecodingKernelBenchmarkResult,
)
from recsys.decoding import (
    StaticDecodingIndex,
    StaticDecodingTorchIndex,
    build_static_decoding_jax_random_model,
    build_static_decoding_random_model,
    static_decoding_generate_and_apply_logprobs_mask,
    static_decoding_generate_and_apply_logprobs_mask_jax,
    static_decoding_sparse_transition_jax,
    static_decoding_sparse_transition_torch,
)

StaticDecodingBenchmarkResults = tuple[
    tuple[StaticDecodingKernelBenchmarkResult, ...],
    tuple[StaticDecodingHarnessBenchmarkResult, ...],
    tuple[StaticDecodingJaxKernelBenchmarkResult, ...],
    tuple[StaticDecodingJaxHarnessBenchmarkResult, ...],
]


def run_static_decoding_benchmarks(
    semantic_ids: Sequence[Sequence[int]],
    *,
    vocab_size: int,
    batch_sizes: Sequence[int],
    warmup_iterations: int,
    iterations: int,
    random_seed: int,
) -> StaticDecodingBenchmarkResults:
    """static_decoding kernel과 harness benchmark를 실행한다."""
    semantic_id_array = np.asarray(semantic_ids, dtype=np.int64)
    if semantic_id_array.ndim != 2:
        msg = "Semantic ID는 2차원 배열로 변환되어야 합니다."
        raise DecoderBenchmarkError(msg)
    if semantic_id_array.shape[1] <= 1:
        return (), (), (), ()

    rng = np.random.default_rng(random_seed)
    dense_lookup_layers = min(2, semantic_id_array.shape[1] - 1)
    index = StaticDecodingIndex.from_semantic_ids(
        semantic_ids,
        vocab_size=vocab_size,
        dense_lookup_layers=dense_lookup_layers,
    )
    torch_index = index.to_torch(torch.device("cpu"))

    kernel_results: list[StaticDecodingKernelBenchmarkResult] = []
    harness_results: list[StaticDecodingHarnessBenchmarkResult] = []
    jax_kernel_results: list[StaticDecodingJaxKernelBenchmarkResult] = []
    jax_harness_results: list[StaticDecodingJaxHarnessBenchmarkResult] = []

    for batch_size in batch_sizes:
        prefix_batches, state_batches = _sample_prefix_state_batches(
            rng=rng,
            semantic_id_array=semantic_id_array,
            index=index,
            batch_size=batch_size,
            total_iterations=warmup_iterations + iterations,
        )
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
        kernel_results.append(
            StaticDecodingKernelBenchmarkResult(
                batch_size=batch_size,
                candidates_identical=True,
                static_decoding_latency_ms=elapsed_ns / iterations / 1_000_000,
                static_decoding_throughput_rows_per_s=_throughput_rows_per_second(
                    batch_size=batch_size,
                    iterations=iterations,
                    elapsed_ns=elapsed_ns,
                ),
            )
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
        jax_kernel_results.append(
            StaticDecodingJaxKernelBenchmarkResult(
                batch_size=batch_size,
                candidates_identical=True,
                static_decoding_jax_latency_ms=jax_elapsed_ns / iterations / 1_000_000,
                static_decoding_jax_throughput_rows_per_s=_throughput_rows_per_second(
                    batch_size=batch_size,
                    iterations=iterations,
                    elapsed_ns=jax_elapsed_ns,
                ),
            )
        )

    torch.manual_seed(random_seed)
    torch_model = build_static_decoding_random_model(
        vocab_size=index.vocab_size,
        device=torch.device("cpu"),
    )
    jax = _jax_module()
    jax_model = build_static_decoding_jax_random_model(vocab_size=index.vocab_size)
    jax_key = jax.random.PRNGKey(random_seed)
    harness_torch_index = index.to_torch(torch.device("cpu"))
    for batch_size in batch_sizes:
        generated_semantic_ids = static_decoding_sparse_transition_torch(
            model=torch_model,
            index=index,
            batch_size=batch_size,
            beam_size=1,
            tokens_per_beam=1,
            start_token=0,
            device=torch.device("cpu"),
            torch_index=harness_torch_index,
        )
        if not _verify_torch_harness_sequences(
            index=index,
            generated_semantic_ids=generated_semantic_ids,
        ):
            msg = (
                "static_decoding sparse_transition_torch harness가 유효하지 않은 "
                f"Semantic ID를 생성했습니다: batch_size={batch_size}"
            )
            raise DecoderBenchmarkError(msg)

        harness_elapsed_ns, harness_checksum = _measure_torch_harness_generation(
            model=torch_model,
            index=index,
            torch_index=harness_torch_index,
            batch_size=batch_size,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
        )
        if harness_checksum < 0:
            msg = f"batch_size={batch_size}에서 static_decoding harness checksum이 잘못되었습니다."
            raise DecoderBenchmarkError(msg)
        harness_results.append(
            StaticDecodingHarnessBenchmarkResult(
                batch_size=batch_size,
                sequences_valid=True,
                static_decoding_harness_latency_ms=harness_elapsed_ns / iterations / 1_000_000,
                static_decoding_harness_throughput_rows_per_s=_throughput_rows_per_second(
                    batch_size=batch_size,
                    iterations=iterations,
                    elapsed_ns=harness_elapsed_ns,
                ),
            )
        )

        generated_jax_semantic_ids = static_decoding_sparse_transition_jax(
            index=index,
            batch_size=batch_size,
            beam_size=1,
            tokens_per_beam=1,
            start_token=0,
            model=jax_model,
            key=jax_key,
        )
        if not _verify_jax_harness_sequences(
            index=index,
            generated_semantic_ids=generated_jax_semantic_ids,
        ):
            msg = (
                "static_decoding sparse_transition_jax harness가 유효하지 않은 "
                f"Semantic ID를 생성했습니다: batch_size={batch_size}"
            )
            raise DecoderBenchmarkError(msg)

        jax_harness_elapsed_ns, jax_harness_checksum = _measure_jax_harness_generation(
            model=jax_model,
            key=jax_key,
            index=index,
            batch_size=batch_size,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
        )
        if jax_harness_checksum < 0:
            msg = (
                f"batch_size={batch_size}에서 static_decoding JAX harness checksum이 "
                "잘못되었습니다."
            )
            raise DecoderBenchmarkError(msg)
        jax_harness_results.append(
            StaticDecodingJaxHarnessBenchmarkResult(
                batch_size=batch_size,
                sequences_valid=True,
                static_decoding_jax_harness_latency_ms=jax_harness_elapsed_ns
                / iterations
                / 1_000_000,
                static_decoding_jax_harness_throughput_rows_per_s=_throughput_rows_per_second(
                    batch_size=batch_size,
                    iterations=iterations,
                    elapsed_ns=jax_harness_elapsed_ns,
                ),
            )
        )

    return (
        tuple(kernel_results),
        tuple(harness_results),
        tuple(jax_kernel_results),
        tuple(jax_harness_results),
    )


def _sample_prefix_state_batches(
    *,
    rng: np.random.Generator,
    semantic_id_array: np.ndarray,
    index: StaticDecodingIndex,
    batch_size: int,
    total_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    prefix_length = index.dense_lookup_layers
    row_indices = rng.integers(
        0,
        semantic_id_array.shape[0],
        size=(total_iterations, batch_size),
        dtype=np.int64,
    )
    prefix_batches = semantic_id_array[row_indices, :prefix_length]
    state_batches = np.zeros((total_iterations, batch_size), dtype=np.int64)
    for batch_index, prefixes in enumerate(prefix_batches):
        for row_index, prefix in enumerate(prefixes):
            state = index.state_for_prefix(prefix.tolist())
            if state is None:
                msg = f"static_decoding prefix state를 찾을 수 없습니다: {prefix.tolist()}"
                raise DecoderBenchmarkError(msg)
            state_batches[batch_index, row_index] = state
    return prefix_batches, state_batches


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


def _measure_torch_harness_generation(
    *,
    model: torch.nn.Module,
    index: StaticDecodingIndex,
    torch_index: StaticDecodingTorchIndex,
    batch_size: int,
    warmup_iterations: int,
    iterations: int,
) -> tuple[int, int]:
    checksum = 0
    device = torch.device("cpu")
    for _ in range(warmup_iterations):
        generated_semantic_ids = static_decoding_sparse_transition_torch(
            model=model,
            index=index,
            batch_size=batch_size,
            beam_size=1,
            tokens_per_beam=1,
            start_token=0,
            device=device,
            torch_index=torch_index,
        )
        checksum += int(generated_semantic_ids.sum().item())

    start_ns = time.perf_counter_ns()
    for _ in range(iterations):
        generated_semantic_ids = static_decoding_sparse_transition_torch(
            model=model,
            index=index,
            batch_size=batch_size,
            beam_size=1,
            tokens_per_beam=1,
            start_token=0,
            device=device,
            torch_index=torch_index,
        )
        checksum += int(generated_semantic_ids.sum().item())
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns, checksum


def _measure_jax_harness_generation(
    *,
    model: Any,
    key: Any,
    index: StaticDecodingIndex,
    batch_size: int,
    warmup_iterations: int,
    iterations: int,
) -> tuple[int, int]:
    checksum = 0
    for _ in range(warmup_iterations):
        generated_semantic_ids = static_decoding_sparse_transition_jax(
            index=index,
            batch_size=batch_size,
            beam_size=1,
            tokens_per_beam=1,
            start_token=0,
            model=model,
            key=key,
        )
        _block_until_ready(generated_semantic_ids)
        checksum += int(np.asarray(generated_semantic_ids).sum())

    start_ns = time.perf_counter_ns()
    for _ in range(iterations):
        generated_semantic_ids = static_decoding_sparse_transition_jax(
            index=index,
            batch_size=batch_size,
            beam_size=1,
            tokens_per_beam=1,
            start_token=0,
            model=model,
            key=key,
        )
        _block_until_ready(generated_semantic_ids)
        checksum += int(np.asarray(generated_semantic_ids).sum())
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns, checksum


def _verify_torch_harness_sequences(
    *,
    index: StaticDecodingIndex,
    generated_semantic_ids: torch.Tensor,
) -> bool:
    if generated_semantic_ids.ndim not in (2, 3):
        return False
    if generated_semantic_ids.shape[-1] != index.semantic_id_depth:
        return False
    flat_semantic_ids = generated_semantic_ids.reshape(-1, index.semantic_id_depth)
    return all(
        index.contains(tuple(int(token) for token in row))
        for row in flat_semantic_ids.detach().cpu().tolist()
    )


def _verify_jax_harness_sequences(
    *,
    index: StaticDecodingIndex,
    generated_semantic_ids: Any,
) -> bool:
    semantic_id_array = np.asarray(generated_semantic_ids)
    if semantic_id_array.ndim not in (2, 3):
        return False
    if semantic_id_array.shape[-1] != index.semantic_id_depth:
        return False
    flat_semantic_ids = semantic_id_array.reshape(-1, index.semantic_id_depth)
    return all(index.contains(tuple(int(token) for token in row)) for row in flat_semantic_ids)


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
    jnp = _jax_numpy()
    states = jnp.asarray(state_batch, dtype=jnp.int32)
    logprobs = jnp.zeros((len(state_batch), vocab_size), dtype=jnp.float32)
    result = static_decoding_generate_and_apply_logprobs_mask_jax(
        flat_logprobs=logprobs,
        flat_states=states,
        index=index,
        prefix_length=prefix_length,
    )
    _block_until_ready(result)
    return result


def _jax_module() -> Any:
    return import_module("jax")


def _jax_numpy() -> Any:
    return import_module("jax.numpy")


def _block_until_ready(value: Any) -> None:
    if isinstance(value, tuple):
        for item in value:
            _block_until_ready(item)
        return
    block_until_ready = getattr(value, "block_until_ready", None)
    if callable(block_until_ready):
        block_until_ready()


def _throughput_rows_per_second(
    *,
    batch_size: int,
    iterations: int,
    elapsed_ns: int,
) -> float:
    elapsed_seconds = elapsed_ns / 1_000_000_000
    return batch_size * iterations / elapsed_seconds

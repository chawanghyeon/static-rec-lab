"""static_decoding sparse transition harness benchmark."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from recsys.benchmark.decoder_schema import (
    DecoderBenchmarkError,
    StaticDecodingHarnessBenchmarkResult,
    StaticDecodingJaxHarnessBenchmarkResult,
)
from recsys.benchmark.static_decoding_helpers import (
    block_until_ready,
    jax_module,
    throughput_rows_per_second,
)
from recsys.decoding import (
    StaticDecodingIndex,
    StaticDecodingTorchIndex,
    build_static_decoding_jax_random_model,
    build_static_decoding_random_model,
    static_decoding_sparse_transition_jax,
    static_decoding_sparse_transition_torch,
)


def run_static_decoding_harness_benchmarks(
    *,
    index: StaticDecodingIndex,
    torch_index: StaticDecodingTorchIndex,
    batch_sizes: Sequence[int],
    warmup_iterations: int,
    iterations: int,
    random_seed: int,
) -> tuple[
    tuple[StaticDecodingHarnessBenchmarkResult, ...],
    tuple[StaticDecodingJaxHarnessBenchmarkResult, ...],
]:
    """PyTorch/JAX sparse transition harness benchmark를 실행한다."""
    torch.manual_seed(random_seed)
    torch_model = build_static_decoding_random_model(
        vocab_size=index.vocab_size,
        device=torch.device("cpu"),
    )
    jax = jax_module()
    jax_model = build_static_decoding_jax_random_model(vocab_size=index.vocab_size)
    jax_key = jax.random.PRNGKey(random_seed)

    torch_results: list[StaticDecodingHarnessBenchmarkResult] = []
    jax_results: list[StaticDecodingJaxHarnessBenchmarkResult] = []
    for batch_size in batch_sizes:
        torch_results.append(
            _benchmark_torch_harness(
                model=torch_model,
                index=index,
                torch_index=torch_index,
                batch_size=batch_size,
                warmup_iterations=warmup_iterations,
                iterations=iterations,
            )
        )
        jax_results.append(
            _benchmark_jax_harness(
                model=jax_model,
                key=jax_key,
                index=index,
                batch_size=batch_size,
                warmup_iterations=warmup_iterations,
                iterations=iterations,
            )
        )
    return tuple(torch_results), tuple(jax_results)


def _benchmark_torch_harness(
    *,
    model: torch.nn.Module,
    index: StaticDecodingIndex,
    torch_index: StaticDecodingTorchIndex,
    batch_size: int,
    warmup_iterations: int,
    iterations: int,
) -> StaticDecodingHarnessBenchmarkResult:
    generated_semantic_ids = static_decoding_sparse_transition_torch(
        model=model,
        index=index,
        batch_size=batch_size,
        beam_size=1,
        tokens_per_beam=1,
        start_token=0,
        device=torch.device("cpu"),
        torch_index=torch_index,
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

    elapsed_ns, checksum = _measure_torch_harness_generation(
        model=model,
        index=index,
        torch_index=torch_index,
        batch_size=batch_size,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )
    if checksum < 0:
        msg = f"batch_size={batch_size}에서 static_decoding harness checksum이 잘못되었습니다."
        raise DecoderBenchmarkError(msg)
    return StaticDecodingHarnessBenchmarkResult(
        batch_size=batch_size,
        sequences_valid=True,
        static_decoding_harness_latency_ms=elapsed_ns / iterations / 1_000_000,
        static_decoding_harness_throughput_rows_per_s=throughput_rows_per_second(
            batch_size=batch_size,
            iterations=iterations,
            elapsed_ns=elapsed_ns,
        ),
    )


def _benchmark_jax_harness(
    *,
    model: Any,
    key: Any,
    index: StaticDecodingIndex,
    batch_size: int,
    warmup_iterations: int,
    iterations: int,
) -> StaticDecodingJaxHarnessBenchmarkResult:
    generated_semantic_ids = static_decoding_sparse_transition_jax(
        index=index,
        batch_size=batch_size,
        beam_size=1,
        tokens_per_beam=1,
        start_token=0,
        model=model,
        key=key,
    )
    if not _verify_jax_harness_sequences(
        index=index,
        generated_semantic_ids=generated_semantic_ids,
    ):
        msg = (
            "static_decoding sparse_transition_jax harness가 유효하지 않은 "
            f"Semantic ID를 생성했습니다: batch_size={batch_size}"
        )
        raise DecoderBenchmarkError(msg)

    elapsed_ns, checksum = _measure_jax_harness_generation(
        model=model,
        key=key,
        index=index,
        batch_size=batch_size,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )
    if checksum < 0:
        msg = f"batch_size={batch_size}에서 static_decoding JAX harness checksum이 잘못되었습니다."
        raise DecoderBenchmarkError(msg)
    return StaticDecodingJaxHarnessBenchmarkResult(
        batch_size=batch_size,
        sequences_valid=True,
        static_decoding_jax_harness_latency_ms=elapsed_ns / iterations / 1_000_000,
        static_decoding_jax_harness_throughput_rows_per_s=throughput_rows_per_second(
            batch_size=batch_size,
            iterations=iterations,
            elapsed_ns=elapsed_ns,
        ),
    )


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
        block_until_ready(generated_semantic_ids)
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
        block_until_ready(generated_semantic_ids)
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

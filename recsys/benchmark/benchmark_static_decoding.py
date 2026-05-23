"""static_decoding benchmark 실행 조립부."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from recsys.benchmark.decoder_schema import (
    DecoderBenchmarkError,
    StaticDecodingHarnessBenchmarkResult,
    StaticDecodingJaxHarnessBenchmarkResult,
    StaticDecodingJaxKernelBenchmarkResult,
    StaticDecodingKernelBenchmarkResult,
)
from recsys.benchmark.static_decoding_candidates import benchmark_static_decoding_candidates
from recsys.benchmark.static_decoding_harness import run_static_decoding_harness_benchmarks
from recsys.decoding import StaticDecodingIndex

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
    """static_decoding candidate kernel과 sparse transition harness를 실행한다."""
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
    jax_kernel_results: list[StaticDecodingJaxKernelBenchmarkResult] = []
    for batch_size in batch_sizes:
        prefix_batches, state_batches = _sample_prefix_state_batches(
            rng=rng,
            semantic_id_array=semantic_id_array,
            index=index,
            batch_size=batch_size,
            total_iterations=warmup_iterations + iterations,
        )
        torch_result, jax_result = benchmark_static_decoding_candidates(
            index=index,
            torch_index=torch_index,
            prefix_batches=prefix_batches,
            state_batches=state_batches,
            batch_size=batch_size,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
        )
        kernel_results.append(torch_result)
        jax_kernel_results.append(jax_result)

    harness_results, jax_harness_results = run_static_decoding_harness_benchmarks(
        index=index,
        torch_index=torch_index,
        batch_sizes=batch_sizes,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        random_seed=random_seed,
    )

    return (
        tuple(kernel_results),
        harness_results,
        tuple(jax_kernel_results),
        jax_harness_results,
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

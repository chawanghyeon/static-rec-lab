"""Naive trie와 검증용 matrix decoder benchmark."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np

from recsys.benchmark.benchmark_static_decoding import run_static_decoding_benchmarks
from recsys.benchmark.decoder_data import (
    generate_synthetic_semantic_ids,
    infer_vocab_size,
    load_semantic_ids,
    normalize_semantic_ids,
)
from recsys.benchmark.decoder_report import write_decoder_benchmark_report
from recsys.benchmark.decoder_schema import (
    DecoderBenchmarkConfig,
    DecoderBenchmarkError,
    DecoderBenchmarkResult,
    DecoderBenchmarkSummary,
    StaticDecodingHarnessBenchmarkResult,
    StaticDecodingJaxHarnessBenchmarkResult,
    StaticDecodingJaxKernelBenchmarkResult,
    StaticDecodingKernelBenchmarkResult,
)
from recsys.decoding.naive_trie import SemanticIdTrie
from recsys.decoding.static_matrix import INVALID_STATE, StaticTransitionMatrixDecoder

__all__ = [
    "DecoderBenchmarkConfig",
    "DecoderBenchmarkError",
    "DecoderBenchmarkResult",
    "DecoderBenchmarkSummary",
    "StaticDecodingHarnessBenchmarkResult",
    "StaticDecodingJaxHarnessBenchmarkResult",
    "StaticDecodingJaxKernelBenchmarkResult",
    "StaticDecodingKernelBenchmarkResult",
    "generate_synthetic_semantic_ids",
    "load_semantic_ids",
    "run_decoder_benchmark",
    "write_decoder_benchmark_report",
]


def run_decoder_benchmark(
    semantic_ids: Sequence[Sequence[int]] | None = None,
    *,
    config: DecoderBenchmarkConfig | None = None,
    source: str = "synthetic",
) -> DecoderBenchmarkSummary:
    """Naive trie와 검증용 matrix decoder의 mask 생성 성능을 비교한다."""
    config = DecoderBenchmarkConfig() if config is None else config
    _validate_config(config)
    resolved_semantic_ids = (
        generate_synthetic_semantic_ids(
            num_semantic_ids=config.num_semantic_ids,
            depth=config.semantic_id_depth,
            vocab_size=config.vocab_size,
            random_seed=config.random_seed,
        )
        if semantic_ids is None
        else normalize_semantic_ids(semantic_ids)
    )
    vocab_size = max(config.vocab_size, infer_vocab_size(resolved_semantic_ids))
    trie = SemanticIdTrie(resolved_semantic_ids)
    decoder = StaticTransitionMatrixDecoder.from_trie(trie, vocab_size=vocab_size)
    rng = np.random.default_rng(config.random_seed)

    results: list[DecoderBenchmarkResult] = []
    for batch_size in config.batch_sizes:
        state_batches = _sample_state_batches(
            rng=rng,
            num_states=decoder.num_states,
            batch_size=batch_size,
            total_iterations=config.warmup_iterations + config.iterations,
        )
        masks_identical = _verify_mask_equivalence(
            trie=trie,
            decoder=decoder,
            state_batches=state_batches,
            vocab_size=vocab_size,
        )
        if not masks_identical:
            msg = f"batch_size={batch_size}에서 naive trie와 검증용 matrix mask가 다릅니다."
            raise DecoderBenchmarkError(msg)

        naive_elapsed_ns, naive_checksum = _measure_mask_generation(
            lambda states: _naive_allowed_token_mask(trie, states, vocab_size),
            state_batches,
            warmup_iterations=config.warmup_iterations,
        )
        static_elapsed_ns, static_checksum = _measure_mask_generation(
            decoder.allowed_token_mask,
            state_batches,
            warmup_iterations=config.warmup_iterations,
        )
        if naive_checksum != static_checksum:
            msg = f"batch_size={batch_size}에서 benchmark 중 mask checksum이 다릅니다."
            raise DecoderBenchmarkError(msg)

        naive_latency_ms = naive_elapsed_ns / config.iterations / 1_000_000
        static_latency_ms = static_elapsed_ns / config.iterations / 1_000_000
        naive_throughput = _throughput_rows_per_second(
            batch_size=batch_size,
            iterations=config.iterations,
            elapsed_ns=naive_elapsed_ns,
        )
        static_throughput = _throughput_rows_per_second(
            batch_size=batch_size,
            iterations=config.iterations,
            elapsed_ns=static_elapsed_ns,
        )
        results.append(
            DecoderBenchmarkResult(
                batch_size=batch_size,
                masks_identical=True,
                naive_latency_ms=naive_latency_ms,
                static_latency_ms=static_latency_ms,
                naive_throughput_rows_per_s=naive_throughput,
                static_throughput_rows_per_s=static_throughput,
                speedup=naive_latency_ms / static_latency_ms,
            )
        )

    (
        static_decoding_kernel_results,
        static_decoding_harness_results,
        static_decoding_jax_kernel_results,
        static_decoding_jax_harness_results,
    ) = run_static_decoding_benchmarks(
        resolved_semantic_ids,
        vocab_size=vocab_size,
        batch_sizes=config.batch_sizes,
        warmup_iterations=config.warmup_iterations,
        iterations=config.iterations,
        random_seed=config.random_seed,
    )

    return DecoderBenchmarkSummary(
        config=config,
        source=source,
        num_semantic_ids=len(resolved_semantic_ids),
        num_states=decoder.num_states,
        vocab_size=vocab_size,
        results=tuple(results),
        static_decoding_kernel_results=tuple(static_decoding_kernel_results),
        static_decoding_harness_results=tuple(static_decoding_harness_results),
        static_decoding_jax_kernel_results=tuple(static_decoding_jax_kernel_results),
        static_decoding_jax_harness_results=tuple(static_decoding_jax_harness_results),
    )


def _validate_config(config: DecoderBenchmarkConfig) -> None:
    if not config.batch_sizes:
        msg = "batch_sizes는 비어 있을 수 없습니다."
        raise DecoderBenchmarkError(msg)
    if any(batch_size < 1 for batch_size in config.batch_sizes):
        msg = f"모든 batch size는 1 이상이어야 합니다: {config.batch_sizes}"
        raise DecoderBenchmarkError(msg)
    if config.warmup_iterations < 0:
        msg = "warmup_iterations는 0 이상이어야 합니다."
        raise DecoderBenchmarkError(msg)
    if config.iterations < 1:
        msg = "iterations는 1 이상이어야 합니다."
        raise DecoderBenchmarkError(msg)
    if config.num_semantic_ids < 1:
        msg = "num_semantic_ids는 1 이상이어야 합니다."
        raise DecoderBenchmarkError(msg)
    if config.semantic_id_depth < 1:
        msg = "semantic_id_depth는 1 이상이어야 합니다."
        raise DecoderBenchmarkError(msg)
    if config.vocab_size < 1:
        msg = "vocab_size는 1 이상이어야 합니다."
        raise DecoderBenchmarkError(msg)


def _sample_state_batches(
    *,
    rng: np.random.Generator,
    num_states: int,
    batch_size: int,
    total_iterations: int,
) -> np.ndarray:
    return rng.integers(
        0,
        num_states,
        size=(total_iterations, batch_size),
        dtype=np.int64,
    )


def _verify_mask_equivalence(
    *,
    trie: SemanticIdTrie,
    decoder: StaticTransitionMatrixDecoder,
    state_batches: np.ndarray,
    vocab_size: int,
) -> bool:
    invalid_state = np.array([INVALID_STATE], dtype=np.int64)
    for batch_index, states in enumerate(state_batches):
        states_to_check = np.concatenate([states, invalid_state]) if batch_index == 0 else states
        naive_mask = _naive_allowed_token_mask(trie, states_to_check, vocab_size)
        static_mask = decoder.allowed_token_mask(states_to_check)
        if not np.array_equal(naive_mask, static_mask):
            return False
    return True


def _measure_mask_generation(
    build_mask: Callable[[np.ndarray], np.ndarray],
    state_batches: np.ndarray,
    *,
    warmup_iterations: int,
) -> tuple[int, int]:
    checksum = 0
    for states in state_batches[:warmup_iterations]:
        checksum += int(build_mask(states).sum())

    start_ns = time.perf_counter_ns()
    for states in state_batches[warmup_iterations:]:
        checksum += int(build_mask(states).sum())
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns, checksum


def _naive_allowed_token_mask(
    trie: SemanticIdTrie,
    states: np.ndarray,
    vocab_size: int,
) -> np.ndarray:
    flat_states = states.reshape(-1)
    mask = np.zeros((len(flat_states), vocab_size), dtype=bool)
    for row_index, state in enumerate(flat_states):
        normalized_state = int(state)
        if normalized_state < 0 or normalized_state >= trie.num_nodes:
            continue
        for token in trie.allowed_next_tokens_for_state(normalized_state):
            mask[row_index, token] = True
    return mask.reshape((*states.shape, vocab_size))


def _throughput_rows_per_second(
    *,
    batch_size: int,
    iterations: int,
    elapsed_ns: int,
) -> float:
    elapsed_seconds = elapsed_ns / 1_000_000_000
    return batch_size * iterations / elapsed_seconds

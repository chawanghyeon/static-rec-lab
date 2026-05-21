"""Decoder benchmark 결과 schema."""

from __future__ import annotations

from dataclasses import dataclass


class DecoderBenchmarkError(ValueError):
    """Decoder benchmark 설정 또는 검증 오류."""


@dataclass(frozen=True)
class DecoderBenchmarkConfig:
    """Decoder benchmark 실행 설정."""

    batch_sizes: tuple[int, ...] = (1, 32, 128, 512)
    warmup_iterations: int = 10
    iterations: int = 100
    num_semantic_ids: int = 4096
    semantic_id_depth: int = 4
    vocab_size: int = 128
    random_seed: int = 42


@dataclass(frozen=True)
class DecoderBenchmarkResult:
    """Batch size별 decoder benchmark 결과."""

    batch_size: int
    masks_identical: bool
    naive_latency_ms: float
    static_latency_ms: float
    naive_throughput_rows_per_s: float
    static_throughput_rows_per_s: float
    speedup: float


@dataclass(frozen=True)
class StaticDecodingKernelBenchmarkResult:
    """static_decoding PyTorch sparse candidate extraction 결과."""

    batch_size: int
    candidates_identical: bool
    static_decoding_latency_ms: float
    static_decoding_throughput_rows_per_s: float


@dataclass(frozen=True)
class StaticDecodingHarnessBenchmarkResult:
    """static_decoding PyTorch sparse_transition_torch harness 결과."""

    batch_size: int
    sequences_valid: bool
    static_decoding_harness_latency_ms: float
    static_decoding_harness_throughput_rows_per_s: float


@dataclass(frozen=True)
class StaticDecodingJaxKernelBenchmarkResult:
    """static_decoding JAX sparse candidate extraction 결과."""

    batch_size: int
    candidates_identical: bool
    static_decoding_jax_latency_ms: float
    static_decoding_jax_throughput_rows_per_s: float


@dataclass(frozen=True)
class StaticDecodingJaxHarnessBenchmarkResult:
    """static_decoding JAX sparse_transition_jax harness 결과."""

    batch_size: int
    sequences_valid: bool
    static_decoding_jax_harness_latency_ms: float
    static_decoding_jax_harness_throughput_rows_per_s: float


@dataclass(frozen=True)
class DecoderBenchmarkSummary:
    """Decoder benchmark 전체 결과."""

    config: DecoderBenchmarkConfig
    source: str
    num_semantic_ids: int
    num_states: int
    vocab_size: int
    results: tuple[DecoderBenchmarkResult, ...]
    static_decoding_kernel_results: tuple[StaticDecodingKernelBenchmarkResult, ...] = ()
    static_decoding_harness_results: tuple[StaticDecodingHarnessBenchmarkResult, ...] = ()
    static_decoding_jax_kernel_results: tuple[StaticDecodingJaxKernelBenchmarkResult, ...] = ()
    static_decoding_jax_harness_results: tuple[StaticDecodingJaxHarnessBenchmarkResult, ...] = ()

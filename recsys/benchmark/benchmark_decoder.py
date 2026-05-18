"""Naive trie와 검증용 matrix decoder benchmark."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
from recsys.decoding.naive_trie import SemanticIdTrie
from recsys.decoding.static_matrix import INVALID_STATE, StaticTransitionMatrixDecoder
from recsys.semantic_id import SemanticId, SemanticIdCodec


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


def load_semantic_ids(path: str | Path) -> tuple[SemanticId, ...]:
    """Semantic ID codec JSON에서 Semantic ID 목록을 로드한다."""
    codec = SemanticIdCodec.load_json(path)
    return tuple(
        codec.item_to_semantic_id[item_id] for item_id in sorted(codec.item_to_semantic_id)
    )


def generate_synthetic_semantic_ids(
    *,
    num_semantic_ids: int,
    depth: int,
    vocab_size: int,
    random_seed: int,
) -> tuple[SemanticId, ...]:
    """재현 가능한 synthetic Semantic ID 목록을 생성한다."""
    if num_semantic_ids < 1:
        msg = "num_semantic_ids는 1 이상이어야 합니다."
        raise DecoderBenchmarkError(msg)
    if depth < 1:
        msg = "depth는 1 이상이어야 합니다."
        raise DecoderBenchmarkError(msg)
    if vocab_size < 1:
        msg = "vocab_size는 1 이상이어야 합니다."
        raise DecoderBenchmarkError(msg)
    if vocab_size**depth < num_semantic_ids:
        msg = (
            "vocab_size와 depth 조합으로 요청한 수만큼 고유 Semantic ID를 만들 수 없습니다: "
            f"vocab_size={vocab_size}, depth={depth}, num_semantic_ids={num_semantic_ids}"
        )
        raise DecoderBenchmarkError(msg)

    rng = np.random.default_rng(random_seed)
    semantic_ids: set[SemanticId] = set()
    while len(semantic_ids) < num_semantic_ids:
        remaining = num_semantic_ids - len(semantic_ids)
        candidate_count = max(remaining * 2, 1024)
        candidates = rng.integers(
            0,
            vocab_size,
            size=(candidate_count, depth),
            dtype=np.int64,
        )
        for row in candidates:
            semantic_ids.add(tuple(int(token) for token in row))
            if len(semantic_ids) >= num_semantic_ids:
                break

    return tuple(sorted(semantic_ids))


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
        else _normalize_semantic_ids(semantic_ids)
    )
    vocab_size = max(config.vocab_size, _infer_vocab_size(resolved_semantic_ids))
    trie = SemanticIdTrie(resolved_semantic_ids)
    decoder = StaticTransitionMatrixDecoder.from_trie(trie, vocab_size=vocab_size)
    rng = np.random.default_rng(config.random_seed)

    results: list[DecoderBenchmarkResult] = []
    static_decoding_kernel_results: list[StaticDecodingKernelBenchmarkResult] = []
    static_decoding_harness_results: list[StaticDecodingHarnessBenchmarkResult] = []
    static_decoding_jax_kernel_results: list[StaticDecodingJaxKernelBenchmarkResult] = []
    static_decoding_jax_harness_results: list[StaticDecodingJaxHarnessBenchmarkResult] = []
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

    if config.semantic_id_depth > 1:
        dense_lookup_layers = min(2, config.semantic_id_depth - 1)
        static_decoding_index = StaticDecodingIndex.from_semantic_ids(
            resolved_semantic_ids,
            vocab_size=vocab_size,
            dense_lookup_layers=dense_lookup_layers,
        )
        torch_index = static_decoding_index.to_torch(torch.device("cpu"))
        semantic_id_array = np.asarray(resolved_semantic_ids, dtype=np.int64)
        for batch_size in config.batch_sizes:
            prefix_batches, state_batches = _sample_static_decoding_prefix_state_batches(
                rng=rng,
                semantic_id_array=semantic_id_array,
                index=static_decoding_index,
                batch_size=batch_size,
                total_iterations=config.warmup_iterations + config.iterations,
            )
            candidates_identical = _verify_static_decoding_candidate_equivalence(
                index=static_decoding_index,
                torch_index=torch_index,
                prefix_batches=prefix_batches,
                state_batches=state_batches,
            )
            if not candidates_identical:
                msg = f"batch_size={batch_size}에서 STATIC 후보 token이 index와 다릅니다."
                raise DecoderBenchmarkError(msg)

            static_decoding_elapsed_ns, static_decoding_checksum = (
                _measure_static_decoding_candidate_generation(
                    torch_index=torch_index,
                    state_batches=state_batches,
                    warmup_iterations=config.warmup_iterations,
                )
            )
            if static_decoding_checksum < 1:
                msg = f"batch_size={batch_size}에서 STATIC 후보 checksum이 비어 있습니다."
                raise DecoderBenchmarkError(msg)
            static_decoding_kernel_results.append(
                StaticDecodingKernelBenchmarkResult(
                    batch_size=batch_size,
                    candidates_identical=True,
                    static_decoding_latency_ms=static_decoding_elapsed_ns
                    / config.iterations
                    / 1_000_000,
                    static_decoding_throughput_rows_per_s=_throughput_rows_per_second(
                        batch_size=batch_size,
                        iterations=config.iterations,
                        elapsed_ns=static_decoding_elapsed_ns,
                    ),
                )
            )
            jax_candidates_identical = _verify_static_decoding_jax_candidate_equivalence(
                index=static_decoding_index,
                prefix_batches=prefix_batches,
                state_batches=state_batches,
            )
            if not jax_candidates_identical:
                msg = f"batch_size={batch_size}에서 STATIC JAX 후보 token이 다릅니다."
                raise DecoderBenchmarkError(msg)

            static_decoding_jax_elapsed_ns, static_decoding_jax_checksum = (
                _measure_static_decoding_jax_candidate_generation(
                    index=static_decoding_index,
                    state_batches=state_batches,
                    warmup_iterations=config.warmup_iterations,
                )
            )
            if static_decoding_jax_checksum < 1:
                msg = f"batch_size={batch_size}에서 STATIC JAX 후보 checksum이 비어 있습니다."
                raise DecoderBenchmarkError(msg)
            static_decoding_jax_kernel_results.append(
                StaticDecodingJaxKernelBenchmarkResult(
                    batch_size=batch_size,
                    candidates_identical=True,
                    static_decoding_jax_latency_ms=static_decoding_jax_elapsed_ns
                    / config.iterations
                    / 1_000_000,
                    static_decoding_jax_throughput_rows_per_s=_throughput_rows_per_second(
                        batch_size=batch_size,
                        iterations=config.iterations,
                        elapsed_ns=static_decoding_jax_elapsed_ns,
                    ),
                )
            )
        torch.manual_seed(config.random_seed)
        harness_model = build_static_decoding_random_model(
            vocab_size=static_decoding_index.vocab_size,
            device=torch.device("cpu"),
        )
        jax = _jax_module()
        jax_harness_model = build_static_decoding_jax_random_model(
            vocab_size=static_decoding_index.vocab_size
        )
        jax_key = jax.random.PRNGKey(config.random_seed)
        harness_torch_index = static_decoding_index.to_torch(torch.device("cpu"))
        for batch_size in config.batch_sizes:
            generated_semantic_ids = static_decoding_sparse_transition_torch(
                model=harness_model,
                index=static_decoding_index,
                batch_size=batch_size,
                beam_size=1,
                tokens_per_beam=1,
                start_token=0,
                device=torch.device("cpu"),
                torch_index=harness_torch_index,
            )
            sequences_valid = _verify_static_decoding_harness_sequences(
                index=static_decoding_index,
                generated_semantic_ids=generated_semantic_ids,
            )
            if not sequences_valid:
                msg = (
                    "static_decoding sparse_transition_torch harness가 유효하지 않은 Semantic ID를 "
                    f"생성했습니다: batch_size={batch_size}"
                )
                raise DecoderBenchmarkError(msg)
            static_decoding_harness_elapsed_ns, static_decoding_harness_checksum = (
                _measure_static_decoding_harness_generation(
                    model=harness_model,
                    index=static_decoding_index,
                    torch_index=harness_torch_index,
                    batch_size=batch_size,
                    warmup_iterations=config.warmup_iterations,
                    iterations=config.iterations,
                )
            )
            if static_decoding_harness_checksum < 0:
                msg = (
                    f"batch_size={batch_size}에서 static_decoding harness checksum이 "
                    "잘못되었습니다."
                )
                raise DecoderBenchmarkError(msg)
            static_decoding_harness_results.append(
                StaticDecodingHarnessBenchmarkResult(
                    batch_size=batch_size,
                    sequences_valid=True,
                    static_decoding_harness_latency_ms=static_decoding_harness_elapsed_ns
                    / config.iterations
                    / 1_000_000,
                    static_decoding_harness_throughput_rows_per_s=_throughput_rows_per_second(
                        batch_size=batch_size,
                        iterations=config.iterations,
                        elapsed_ns=static_decoding_harness_elapsed_ns,
                    ),
                )
            )
            generated_jax_semantic_ids = static_decoding_sparse_transition_jax(
                index=static_decoding_index,
                batch_size=batch_size,
                beam_size=1,
                tokens_per_beam=1,
                start_token=0,
                model=jax_harness_model,
                key=jax_key,
            )
            jax_sequences_valid = _verify_static_decoding_jax_harness_sequences(
                index=static_decoding_index,
                generated_semantic_ids=generated_jax_semantic_ids,
            )
            if not jax_sequences_valid:
                msg = (
                    "static_decoding sparse_transition_jax harness가 유효하지 않은 Semantic ID를 "
                    f"생성했습니다: batch_size={batch_size}"
                )
                raise DecoderBenchmarkError(msg)
            static_decoding_jax_harness_elapsed_ns, static_decoding_jax_harness_checksum = (
                _measure_static_decoding_jax_harness_generation(
                    model=jax_harness_model,
                    key=jax_key,
                    index=static_decoding_index,
                    batch_size=batch_size,
                    warmup_iterations=config.warmup_iterations,
                    iterations=config.iterations,
                )
            )
            if static_decoding_jax_harness_checksum < 0:
                msg = (
                    f"batch_size={batch_size}에서 static_decoding JAX harness checksum이 "
                    "잘못되었습니다."
                )
                raise DecoderBenchmarkError(msg)
            static_decoding_jax_harness_results.append(
                StaticDecodingJaxHarnessBenchmarkResult(
                    batch_size=batch_size,
                    sequences_valid=True,
                    static_decoding_jax_harness_latency_ms=static_decoding_jax_harness_elapsed_ns
                    / config.iterations
                    / 1_000_000,
                    static_decoding_jax_harness_throughput_rows_per_s=_throughput_rows_per_second(
                        batch_size=batch_size,
                        iterations=config.iterations,
                        elapsed_ns=static_decoding_jax_harness_elapsed_ns,
                    ),
                )
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


def write_decoder_benchmark_report(
    path: str | Path,
    summary: DecoderBenchmarkSummary,
) -> Path:
    """Decoder benchmark 결과를 한국어 markdown report로 저장한다."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Decoder 벤치마크 리포트",
        "",
        "naive trie decoder와 검증용 sparse transition matrix decoder의 "
        "allowed-token mask 생성 성능을 비교합니다.",
        "",
        "## 설정",
        "",
        f"- 데이터 소스: `{summary.source}`",
        f"- Semantic ID 수: {summary.num_semantic_ids:,}",
        f"- trie state 수: {summary.num_states:,}",
        f"- vocab size: {summary.vocab_size:,}",
        f"- batch sizes: {', '.join(str(size) for size in summary.config.batch_sizes)}",
        f"- warmup iterations: {summary.config.warmup_iterations:,}",
        f"- measured iterations: {summary.config.iterations:,}",
        f"- random seed: {summary.config.random_seed}",
        "",
        "## 결과",
        "",
        "| batch_size | mask 일치 | naive latency ms | matrix latency ms | "
        "naive rows/s | matrix rows/s | speedup |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for result in summary.results:
        lines.append(
            f"| {result.batch_size:,} | {'예' if result.masks_identical else '아니오'} | "
            f"{result.naive_latency_ms:.4f} | {result.static_latency_ms:.4f} | "
            f"{result.naive_throughput_rows_per_s:,.2f} | "
            f"{result.static_throughput_rows_per_s:,.2f} | {result.speedup:.2f}x |"
        )
    if summary.static_decoding_kernel_results:
        lines.extend(
            [
                "",
                "## static_decoding PyTorch kernel",
                "",
                "`static_decoding.decoding_pt.generate_and_apply_logprobs_mask`를 "
                "사용해 CSR sparse tail에서 유효 child token 후보를 추출하는 성능을 "
                "측정합니다. 이 값은 전체 vocab boolean mask 생성이 아니라 candidate "
                "gather latency입니다.",
                "",
                "| batch_size | 후보 일치 | static_decoding latency ms | static_decoding rows/s |",
                "|---:|:---:|---:|---:|",
            ]
        )
        for static_decoding_result in summary.static_decoding_kernel_results:
            lines.append(
                f"| {static_decoding_result.batch_size:,} | "
                f"{'예' if static_decoding_result.candidates_identical else '아니오'} | "
                f"{static_decoding_result.static_decoding_latency_ms:.4f} | "
                f"{static_decoding_result.static_decoding_throughput_rows_per_s:,.2f} |"
            )
    if summary.static_decoding_harness_results:
        lines.extend(
            [
                "",
                "## static_decoding sparse_transition_torch harness",
                "",
                "`static_decoding.decoding_pt.sparse_transition_torch`를 그대로 호출해 "
                "`static_decoding` PyTorch decoding loop가 생성한 Semantic ID가 모두 유효한지 "
                "검증하고 end-to-end harness latency를 측정합니다. 이 harness는 "
                "`static_decoding.decoding_pt.RandomModel`을 사용하므로 추천 모델 품질 "
                "평가는 아니며, static_decoding 호출 경로와 constrained generation 동작 검증에 "
                "초점을 둡니다.",
                "",
                "| batch_size | 생성 ID 유효 | static_decoding harness latency ms | "
                "static_decoding harness rows/s |",
                "|---:|:---:|---:|---:|",
            ]
        )
        for harness_result in summary.static_decoding_harness_results:
            lines.append(
                f"| {harness_result.batch_size:,} | "
                f"{'예' if harness_result.sequences_valid else '아니오'} | "
                f"{harness_result.static_decoding_harness_latency_ms:.4f} | "
                f"{harness_result.static_decoding_harness_throughput_rows_per_s:,.2f} |"
            )
    if summary.static_decoding_jax_kernel_results:
        lines.extend(
            [
                "",
                "## static_decoding JAX kernel",
                "",
                "`static_decoding.decoding_jax.generate_and_apply_logprobs_mask`를 "
                "사용해 CSR sparse tail에서 유효 child token 후보를 추출하는 성능을 "
                "측정합니다. 이 값은 CPU 환경의 JAX candidate gather latency입니다.",
                "",
                "| batch_size | 후보 일치 | static_decoding JAX latency ms | "
                "static_decoding JAX rows/s |",
                "|---:|:---:|---:|---:|",
            ]
        )
        for jax_result in summary.static_decoding_jax_kernel_results:
            lines.append(
                f"| {jax_result.batch_size:,} | "
                f"{'예' if jax_result.candidates_identical else '아니오'} | "
                f"{jax_result.static_decoding_jax_latency_ms:.4f} | "
                f"{jax_result.static_decoding_jax_throughput_rows_per_s:,.2f} |"
            )
    if summary.static_decoding_jax_harness_results:
        lines.extend(
            [
                "",
                "## static_decoding sparse_transition_jax harness",
                "",
                "`static_decoding.decoding_jax.sparse_transition_jax`를 그대로 호출해 "
                "static_decoding JAX decoding loop가 생성한 Semantic ID가 모두 유효한지 검증하고 "
                "end-to-end harness latency를 측정합니다. 이 harness는 "
                "`static_decoding.decoding_jax.RandomModel`을 사용합니다.",
                "",
                "| batch_size | 생성 ID 유효 | static_decoding JAX harness latency ms | "
                "static_decoding JAX harness rows/s |",
                "|---:|:---:|---:|---:|",
            ]
        )
        for jax_harness_result in summary.static_decoding_jax_harness_results:
            lines.append(
                f"| {jax_harness_result.batch_size:,} | "
                f"{'예' if jax_harness_result.sequences_valid else '아니오'} | "
                f"{jax_harness_result.static_decoding_jax_harness_latency_ms:.4f} | "
                f"{jax_harness_result.static_decoding_jax_harness_throughput_rows_per_s:,.2f} |"
            )
    lines.extend(
        [
            "",
            "## 검증",
            "",
            "- 각 batch size의 모든 sampled state batch에서 naive trie와 검증용 matrix decoder의 "
            "allowed-token mask가 동일한지 먼저 확인합니다.",
            "- benchmark 실행 중 생성된 mask checksum도 비교해서 측정 루프 안의 결과 차이를 "
            "감지합니다.",
            "- throughput은 measured iterations 동안 생성한 mask row 수를 총 소요 시간으로 "
            "나눈 값입니다.",
            "- static_decoding PyTorch kernel은 동일 prefix batch에서 static_decoding index의 "
            "allowed-token 후보와 일치하는지 검증합니다.",
            "- static_decoding sparse_transition_torch harness는 생성된 모든 Semantic ID가 "
            "static_decoding index에 존재하는지 검증합니다.",
            "- static_decoding JAX kernel과 sparse_transition_jax harness도 동일한 "
            "static_decoding index에서 후보 token 및 생성 Semantic ID validity를 검증합니다.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


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


def _normalize_semantic_ids(semantic_ids: Sequence[Sequence[int]]) -> tuple[SemanticId, ...]:
    normalized = tuple(tuple(int(token) for token in semantic_id) for semantic_id in semantic_ids)
    if not normalized:
        msg = "benchmark Semantic ID 목록은 비어 있을 수 없습니다."
        raise DecoderBenchmarkError(msg)
    if any(not semantic_id for semantic_id in normalized):
        msg = "benchmark Semantic ID sequence는 비어 있을 수 없습니다."
        raise DecoderBenchmarkError(msg)
    return normalized


def _infer_vocab_size(semantic_ids: Iterable[Sequence[int]]) -> int:
    max_token = max((token for semantic_id in semantic_ids for token in semantic_id), default=-1)
    return max_token + 1 if max_token >= 0 else 1


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


def _sample_static_decoding_prefix_state_batches(
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
                msg = f"STATIC prefix state를 찾을 수 없습니다: {prefix.tolist()}"
                raise DecoderBenchmarkError(msg)
            state_batches[batch_index, row_index] = state
    return prefix_batches, state_batches


def _verify_static_decoding_candidate_equivalence(
    *,
    index: StaticDecodingIndex,
    torch_index: StaticDecodingTorchIndex,
    prefix_batches: np.ndarray,
    state_batches: np.ndarray,
) -> bool:
    for prefix_batch, state_batch in zip(prefix_batches, state_batches, strict=True):
        candidate_logprobs, candidate_tokens, _ = _static_decoding_candidates_for_states(
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


def _measure_static_decoding_candidate_generation(
    *,
    torch_index: StaticDecodingTorchIndex,
    state_batches: np.ndarray,
    warmup_iterations: int,
) -> tuple[int, int]:
    checksum = 0
    for states in state_batches[:warmup_iterations]:
        candidate_logprobs, _, _ = _static_decoding_candidates_for_states(
            torch_index=torch_index,
            state_batch=states,
            vocab_size=torch_index.vocab_size,
            prefix_length=torch_index.dense_lookup_layers,
        )
        checksum += int(torch.isfinite(candidate_logprobs).sum().item())

    start_ns = time.perf_counter_ns()
    for states in state_batches[warmup_iterations:]:
        candidate_logprobs, _, _ = _static_decoding_candidates_for_states(
            torch_index=torch_index,
            state_batch=states,
            vocab_size=torch_index.vocab_size,
            prefix_length=torch_index.dense_lookup_layers,
        )
        checksum += int(torch.isfinite(candidate_logprobs).sum().item())
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns, checksum


def _verify_static_decoding_jax_candidate_equivalence(
    *,
    index: StaticDecodingIndex,
    prefix_batches: np.ndarray,
    state_batches: np.ndarray,
) -> bool:
    for prefix_batch, state_batch in zip(prefix_batches, state_batches, strict=True):
        candidate_logprobs, candidate_tokens, _ = _static_decoding_jax_candidates_for_states(
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


def _measure_static_decoding_jax_candidate_generation(
    *,
    index: StaticDecodingIndex,
    state_batches: np.ndarray,
    warmup_iterations: int,
) -> tuple[int, int]:
    checksum = 0
    for states in state_batches[:warmup_iterations]:
        candidate_logprobs, _, _ = _static_decoding_jax_candidates_for_states(
            index=index,
            state_batch=states,
            vocab_size=index.vocab_size,
            prefix_length=index.dense_lookup_layers,
        )
        checksum += int(np.isfinite(np.asarray(candidate_logprobs)).sum())

    start_ns = time.perf_counter_ns()
    for states in state_batches[warmup_iterations:]:
        candidate_logprobs, _, _ = _static_decoding_jax_candidates_for_states(
            index=index,
            state_batch=states,
            vocab_size=index.vocab_size,
            prefix_length=index.dense_lookup_layers,
        )
        checksum += int(np.isfinite(np.asarray(candidate_logprobs)).sum())
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns, checksum


def _measure_static_decoding_harness_generation(
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


def _measure_static_decoding_jax_harness_generation(
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


def _verify_static_decoding_harness_sequences(
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


def _verify_static_decoding_jax_harness_sequences(
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


def _static_decoding_candidates_for_states(
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


def _static_decoding_jax_candidates_for_states(
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

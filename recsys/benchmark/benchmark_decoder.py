"""Naive trie와 STATIC-style decoder benchmark."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recsys.decoding import INVALID_STATE, SemanticIdTrie, StaticTransitionMatrixDecoder
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
class DecoderBenchmarkSummary:
    """Decoder benchmark 전체 결과."""

    config: DecoderBenchmarkConfig
    source: str
    num_semantic_ids: int
    num_states: int
    vocab_size: int
    results: tuple[DecoderBenchmarkResult, ...]


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
    """Naive trie와 STATIC-style decoder의 mask 생성 성능을 비교한다."""
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
            msg = f"batch_size={batch_size}에서 naive trie와 STATIC-style mask가 다릅니다."
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

    return DecoderBenchmarkSummary(
        config=config,
        source=source,
        num_semantic_ids=len(resolved_semantic_ids),
        num_states=decoder.num_states,
        vocab_size=vocab_size,
        results=tuple(results),
    )


def write_decoder_benchmark_report(
    path: str | Path,
    summary: DecoderBenchmarkSummary,
) -> Path:
    """Decoder benchmark 결과를 markdown report로 저장한다."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Decoder Benchmark",
        "",
        "naive trie decoder와 STATIC-style sparse transition matrix decoder의 "
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
        "| batch_size | mask 일치 | naive latency ms | STATIC latency ms | "
        "naive rows/s | STATIC rows/s | speedup |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for result in summary.results:
        lines.append(
            f"| {result.batch_size:,} | {'yes' if result.masks_identical else 'no'} | "
            f"{result.naive_latency_ms:.4f} | {result.static_latency_ms:.4f} | "
            f"{result.naive_throughput_rows_per_s:,.2f} | "
            f"{result.static_throughput_rows_per_s:,.2f} | {result.speedup:.2f}x |"
        )
    lines.extend(
        [
            "",
            "## 검증",
            "",
            "- 각 batch size의 모든 sampled state batch에서 naive trie와 STATIC-style decoder의 "
            "allowed-token mask가 동일한지 먼저 확인합니다.",
            "- benchmark 실행 중 생성된 mask checksum도 비교해서 측정 루프 안의 결과 차이를 "
            "감지합니다.",
            "- throughput은 measured iterations 동안 생성한 mask row 수를 총 소요 시간으로 "
            "나눈 값입니다.",
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

"""Decoder benchmark markdown 리포트 작성."""

from __future__ import annotations

from pathlib import Path

from recsys.benchmark.decoder_schema import DecoderBenchmarkSummary


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

    _append_static_decoding_kernel_section(lines, summary)
    _append_static_decoding_harness_section(lines, summary)
    _append_static_decoding_jax_kernel_section(lines, summary)
    _append_static_decoding_jax_harness_section(lines, summary)
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


def _append_static_decoding_kernel_section(
    lines: list[str],
    summary: DecoderBenchmarkSummary,
) -> None:
    if not summary.static_decoding_kernel_results:
        return
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
    for result in summary.static_decoding_kernel_results:
        lines.append(
            f"| {result.batch_size:,} | {'예' if result.candidates_identical else '아니오'} | "
            f"{result.static_decoding_latency_ms:.4f} | "
            f"{result.static_decoding_throughput_rows_per_s:,.2f} |"
        )


def _append_static_decoding_harness_section(
    lines: list[str],
    summary: DecoderBenchmarkSummary,
) -> None:
    if not summary.static_decoding_harness_results:
        return
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
    for result in summary.static_decoding_harness_results:
        lines.append(
            f"| {result.batch_size:,} | {'예' if result.sequences_valid else '아니오'} | "
            f"{result.static_decoding_harness_latency_ms:.4f} | "
            f"{result.static_decoding_harness_throughput_rows_per_s:,.2f} |"
        )


def _append_static_decoding_jax_kernel_section(
    lines: list[str],
    summary: DecoderBenchmarkSummary,
) -> None:
    if not summary.static_decoding_jax_kernel_results:
        return
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
    for result in summary.static_decoding_jax_kernel_results:
        lines.append(
            f"| {result.batch_size:,} | {'예' if result.candidates_identical else '아니오'} | "
            f"{result.static_decoding_jax_latency_ms:.4f} | "
            f"{result.static_decoding_jax_throughput_rows_per_s:,.2f} |"
        )


def _append_static_decoding_jax_harness_section(
    lines: list[str],
    summary: DecoderBenchmarkSummary,
) -> None:
    if not summary.static_decoding_jax_harness_results:
        return
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
    for result in summary.static_decoding_jax_harness_results:
        lines.append(
            f"| {result.batch_size:,} | {'예' if result.sequences_valid else '아니오'} | "
            f"{result.static_decoding_jax_harness_latency_ms:.4f} | "
            f"{result.static_decoding_jax_harness_throughput_rows_per_s:,.2f} |"
        )

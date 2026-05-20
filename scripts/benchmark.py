"""Decoder/serving benchmark CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from apps.api.services import (
    MockRecommendationService,
    build_recommendation_service_from_environment,
)
from recsys.benchmark import (
    DecoderBenchmarkConfig,
    ServingBenchmarkConfig,
    load_semantic_ids,
    run_decoder_benchmark,
    run_serving_benchmark,
    write_decoder_benchmark_report,
    write_serving_benchmark_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark 실행")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_decoder_parser(subparsers)
    _add_serving_parser(subparsers)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "decoder":
        benchmark_decoder(args)
    elif args.command == "serving":
        benchmark_serving(args)
    else:  # pragma: no cover
        raise ValueError(f"알 수 없는 benchmark command입니다: {args.command}")


def _add_decoder_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("decoder", help="decoder benchmark")
    parser.add_argument(
        "--semantic-id-path",
        type=Path,
        default=None,
        help="Semantic ID codec JSON 경로. 생략하면 synthetic Semantic ID를 사용한다.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/local/decoder_benchmark.md"),
        help="decoder benchmark markdown report 저장 경로",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 32, 128, 512],
        help="benchmark batch size 목록",
    )
    parser.add_argument("--warmup-iterations", type=int, default=10, help="warmup 반복 수")
    parser.add_argument("--iterations", type=int, default=100, help="측정 반복 수")
    parser.add_argument(
        "--num-semantic-ids",
        type=int,
        default=4096,
        help="synthetic Semantic ID 수",
    )
    parser.add_argument("--semantic-id-depth", type=int, default=4, help="synthetic depth")
    parser.add_argument("--vocab-size", type=int, default=128, help="decoder vocab size")
    parser.add_argument("--random-seed", type=int, default=42, help="랜덤 시드")


def _add_serving_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("serving", help="serving benchmark")
    parser.add_argument(
        "--service",
        choices=["mock", "environment"],
        default="mock",
        help="benchmark 대상 service. environment는 STATIC_REC_* 환경변수 설정을 사용한다.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/local/serving_benchmark.md"),
        help="serving benchmark 리포트 저장 경로",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-user-id", type=int, default=1)
    parser.add_argument("--max-user-id", type=int, default=10_000)


def benchmark_decoder(args: argparse.Namespace) -> None:
    config = DecoderBenchmarkConfig(
        batch_sizes=tuple(args.batch_sizes),
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        num_semantic_ids=args.num_semantic_ids,
        semantic_id_depth=args.semantic_id_depth,
        vocab_size=args.vocab_size,
        random_seed=args.random_seed,
    )
    semantic_ids = None
    source = "synthetic"
    if args.semantic_id_path is not None:
        semantic_ids = load_semantic_ids(args.semantic_id_path)
        source = str(args.semantic_id_path)

    summary = run_decoder_benchmark(semantic_ids, config=config, source=source)
    report_path = write_decoder_benchmark_report(args.report_path, summary)

    print("Decoder benchmark 완료")
    print(f"- source: {summary.source}")
    print(f"- semantic_ids: {summary.num_semantic_ids}")
    print(f"- states: {summary.num_states}")
    print(f"- vocab_size: {summary.vocab_size}")
    for result in summary.results:
        print(
            f"- batch={result.batch_size}: "
            f"naive={result.naive_latency_ms:.4f}ms, "
            f"static={result.static_latency_ms:.4f}ms, "
            f"speedup={result.speedup:.2f}x, "
            f"mask_equal={result.masks_identical}"
        )
    for static_decoding_result in summary.static_decoding_kernel_results:
        print(
            f"- static_decoding batch={static_decoding_result.batch_size}: "
            f"candidate_gather={static_decoding_result.static_decoding_latency_ms:.4f}ms, "
            f"rows/s={static_decoding_result.static_decoding_throughput_rows_per_s:.2f}, "
            f"candidate_equal={static_decoding_result.candidates_identical}"
        )
    for harness_result in summary.static_decoding_harness_results:
        print(
            f"- static_decoding harness batch={harness_result.batch_size}: "
            f"latency={harness_result.static_decoding_harness_latency_ms:.4f}ms, "
            f"rows/s={harness_result.static_decoding_harness_throughput_rows_per_s:.2f}, "
            f"valid={harness_result.sequences_valid}"
        )
    for jax_result in summary.static_decoding_jax_kernel_results:
        print(
            f"- static_decoding jax batch={jax_result.batch_size}: "
            f"candidate_gather={jax_result.static_decoding_jax_latency_ms:.4f}ms, "
            f"rows/s={jax_result.static_decoding_jax_throughput_rows_per_s:.2f}, "
            f"candidate_equal={jax_result.candidates_identical}"
        )
    for jax_harness_result in summary.static_decoding_jax_harness_results:
        print(
            f"- static_decoding jax harness batch={jax_harness_result.batch_size}: "
            f"latency={jax_harness_result.static_decoding_jax_harness_latency_ms:.4f}ms, "
            f"rows/s={jax_harness_result.static_decoding_jax_harness_throughput_rows_per_s:.2f}, "
            f"valid={jax_harness_result.sequences_valid}"
        )
    print(f"- report: {report_path}")


def benchmark_serving(args: argparse.Namespace) -> None:
    service = (
        MockRecommendationService()
        if args.service == "mock"
        else build_recommendation_service_from_environment()
    )
    config = ServingBenchmarkConfig(
        batch_sizes=tuple(args.batch_sizes),
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        k=args.k,
        random_seed=args.random_seed,
        min_user_id=args.min_user_id,
        max_user_id=args.max_user_id,
    )
    summary = run_serving_benchmark(service, config=config, source=args.service)
    report_path = write_serving_benchmark_report(args.report_path, summary)

    print("Serving benchmark 완료")
    print(f"- 서비스: {summary.source}")
    print(f"- model: {summary.model_name}")
    print(f"- decoder: {summary.decoder_name}")
    for result in summary.results:
        print(
            f"- batch_size={result.batch_size}: "
            f"mean_latency_ms={result.mean_latency_ms:.4f}, "
            f"p95_latency_ms={result.p95_latency_ms:.4f}, "
            f"throughput_req_s={result.throughput_requests_per_s:.2f}"
        )
    print(f"- report: {report_path}")


if __name__ == "__main__":
    main()

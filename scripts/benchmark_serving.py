"""추천 서비스 serving benchmark CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from apps.api.services import (
    MockRecommendationService,
    build_recommendation_service_from_environment,
)
from recsys.benchmark import (
    ServingBenchmarkConfig,
    run_serving_benchmark,
    write_serving_benchmark_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="추천 서비스 serving benchmark")
    parser.add_argument(
        "--service",
        choices=["mock", "environment"],
        default="mock",
        help="benchmark 대상 service. environment는 STATIC_REC_* 환경변수 설정을 사용한다.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/serving_benchmark.md"),
        help="serving benchmark 리포트 저장 경로",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-user-id", type=int, default=1)
    parser.add_argument("--max-user-id", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

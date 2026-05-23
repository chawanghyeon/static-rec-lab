"""추천 서비스 serving latency benchmark."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from string import Formatter
from typing import Any, Protocol, cast

import numpy as np


class ServingBenchmarkError(ValueError):
    """Serving benchmark 설정 또는 실행 오류."""


class ServingBenchmarkService(Protocol):
    """Serving benchmark가 요구하는 추천 서비스 contract."""

    model_name: str
    decoder_name: str

    def recommend(self, *, user_id: int, k: int) -> Sequence[object]:
        """사용자별 추천 결과를 반환한다."""


class HttpBenchmarkResponse(Protocol):
    """HTTP benchmark가 요구하는 response contract."""

    status_code: int

    def json(self) -> Any:
        """JSON response body를 반환한다."""


class HttpBenchmarkClient(Protocol):
    """HTTP benchmark가 요구하는 client contract."""

    def get(self, url: str) -> HttpBenchmarkResponse:
        """GET request를 실행한다."""


@dataclass(frozen=True)
class ServingBenchmarkConfig:
    """Serving benchmark 실행 설정."""

    batch_sizes: tuple[int, ...] = (1, 32, 128)
    warmup_iterations: int = 3
    iterations: int = 10
    k: int = 20
    random_seed: int = 42
    min_user_id: int = 1
    max_user_id: int = 10_000
    endpoint_template: str = "/recommendations/users/{user_id}?k={k}"


@dataclass(frozen=True)
class ServingBenchmarkResult:
    """Batch size별 serving benchmark 결과."""

    batch_size: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    throughput_requests_per_s: float
    total_requests: int


@dataclass(frozen=True)
class ServingBenchmarkSummary:
    """Serving benchmark 전체 결과."""

    config: ServingBenchmarkConfig
    source: str
    model_name: str
    decoder_name: str
    results: tuple[ServingBenchmarkResult, ...]


@dataclass(frozen=True)
class HttpEndpointBenchmarkSummary:
    """HTTP endpoint benchmark 전체 결과."""

    config: ServingBenchmarkConfig
    source: str
    model_name: str
    decoder_name: str
    endpoint_template: str
    results: tuple[ServingBenchmarkResult, ...]


def run_serving_benchmark(
    service: ServingBenchmarkService,
    *,
    config: ServingBenchmarkConfig | None = None,
    source: str = "service",
) -> ServingBenchmarkSummary:
    """추천 서비스의 사용자별 serving latency를 측정한다."""
    config = ServingBenchmarkConfig() if config is None else config
    _validate_config(config)
    rng = np.random.default_rng(config.random_seed)
    results: list[ServingBenchmarkResult] = []

    for batch_size in config.batch_sizes:
        user_batches = _sample_user_batches(
            rng=rng,
            batch_size=batch_size,
            total_iterations=config.warmup_iterations + config.iterations,
            min_user_id=config.min_user_id,
            max_user_id=config.max_user_id,
        )
        _warm_up_service(
            service=service,
            user_batches=user_batches[: config.warmup_iterations],
            k=config.k,
        )
        request_latencies_ns, measured_elapsed_ns = _measure_service(
            service=service,
            user_batches=user_batches[config.warmup_iterations :],
            k=config.k,
        )
        results.append(
            _summarize_result(
                batch_size=batch_size,
                request_latencies_ns=request_latencies_ns,
                measured_elapsed_ns=measured_elapsed_ns,
            )
        )

    return ServingBenchmarkSummary(
        config=config,
        source=source,
        model_name=service.model_name,
        decoder_name=service.decoder_name,
        results=tuple(results),
    )


def write_serving_benchmark_report(
    path: str | Path,
    summary: ServingBenchmarkSummary,
) -> Path:
    """Serving benchmark 결과를 한국어 markdown report로 저장한다."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Serving 벤치마크 리포트",
        "",
        "추천 서비스의 사용자별 추천 생성 latency를 측정합니다.",
        "이 벤치마크는 HTTP 서버를 띄우지 않고 서비스 contract를 직접 호출하므로,",
        "FastAPI 직렬화, validation, network overhead를 제외한 모델 및 decoder 비용을 봅니다.",
        "",
        "## 설정",
        "",
        f"- 서비스 소스: `{summary.source}`",
        f"- model: `{summary.model_name}`",
        f"- decoder: `{summary.decoder_name}`",
        f"- k: {summary.config.k}",
        f"- batch sizes: {', '.join(str(size) for size in summary.config.batch_sizes)}",
        f"- warmup iterations: {summary.config.warmup_iterations:,}",
        f"- measured iterations: {summary.config.iterations:,}",
        f"- random seed: {summary.config.random_seed}",
        f"- user_id range: {summary.config.min_user_id}..{summary.config.max_user_id}",
        "",
        "## 결과",
        "",
        "| batch_size | 요청 수 | 평균 latency ms | p50 latency ms | p95 latency ms | "
        "최대 latency ms | throughput req/s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in summary.results:
        lines.append(
            f"| {result.batch_size:,} | {result.total_requests:,} | "
            f"{result.mean_latency_ms:.4f} | {result.p50_latency_ms:.4f} | "
            f"{result.p95_latency_ms:.4f} | {result.max_latency_ms:.4f} | "
            f"{result.throughput_requests_per_s:,.2f} |"
        )

    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- latency는 `RecommendationService.recommend(user_id, k)` 단일 호출 기준입니다.",
            "- batch_size는 한 측정 iteration에서 순차 호출한 user request 수입니다.",
            "- 실제 HTTP endpoint latency는 FastAPI 직렬화, validation, network overhead가 "
            "추가될 수 있습니다.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_http_endpoint_benchmark(
    client: HttpBenchmarkClient,
    *,
    config: ServingBenchmarkConfig | None = None,
    source: str = "http-endpoint",
) -> HttpEndpointBenchmarkSummary:
    """FastAPI/HTTP endpoint latency를 측정한다."""
    config = ServingBenchmarkConfig() if config is None else config
    _validate_config(config)
    _validate_endpoint_template(config.endpoint_template)
    rng = np.random.default_rng(config.random_seed)
    results: list[ServingBenchmarkResult] = []
    model_name = ""
    decoder_name = ""

    for batch_size in config.batch_sizes:
        user_batches = _sample_user_batches(
            rng=rng,
            batch_size=batch_size,
            total_iterations=config.warmup_iterations + config.iterations,
            min_user_id=config.min_user_id,
            max_user_id=config.max_user_id,
        )
        _warm_up_http_endpoint(
            client=client,
            user_batches=user_batches[: config.warmup_iterations],
            k=config.k,
            endpoint_template=config.endpoint_template,
        )
        request_latencies_ns, measured_elapsed_ns, batch_model_name, batch_decoder_name = (
            _measure_http_endpoint(
                client=client,
                user_batches=user_batches[config.warmup_iterations :],
                k=config.k,
                endpoint_template=config.endpoint_template,
            )
        )
        if not model_name:
            model_name = batch_model_name
            decoder_name = batch_decoder_name
        results.append(
            _summarize_result(
                batch_size=batch_size,
                request_latencies_ns=request_latencies_ns,
                measured_elapsed_ns=measured_elapsed_ns,
            )
        )

    return HttpEndpointBenchmarkSummary(
        config=config,
        source=source,
        model_name=model_name,
        decoder_name=decoder_name,
        endpoint_template=config.endpoint_template,
        results=tuple(results),
    )


def write_http_endpoint_benchmark_report(
    path: str | Path,
    summary: HttpEndpointBenchmarkSummary,
) -> Path:
    """HTTP endpoint benchmark 결과를 한국어 markdown report로 저장한다."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HTTP Endpoint 벤치마크 리포트",
        "",
        "FastAPI recommendation endpoint를 HTTP client contract로 호출해 latency를 측정합니다.",
        "이 벤치마크는 endpoint routing, request validation, response model serialization,",
        "client-side JSON decode 비용을 포함합니다. 별도 network hop은 포함하지 않습니다.",
        "",
        "## 설정",
        "",
        f"- 서비스 소스: `{summary.source}`",
        f"- endpoint: `{summary.endpoint_template}`",
        f"- model: `{summary.model_name}`",
        f"- decoder: `{summary.decoder_name}`",
        f"- k: {summary.config.k}",
        f"- batch sizes: {', '.join(str(size) for size in summary.config.batch_sizes)}",
        f"- warmup iterations: {summary.config.warmup_iterations:,}",
        f"- measured iterations: {summary.config.iterations:,}",
        f"- random seed: {summary.config.random_seed}",
        f"- user_id range: {summary.config.min_user_id}..{summary.config.max_user_id}",
        "",
        "## 결과",
        "",
        "| batch_size | 요청 수 | 평균 latency ms | p50 latency ms | p95 latency ms | "
        "최대 latency ms | throughput req/s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in summary.results:
        lines.append(
            f"| {result.batch_size:,} | {result.total_requests:,} | "
            f"{result.mean_latency_ms:.4f} | {result.p50_latency_ms:.4f} | "
            f"{result.p95_latency_ms:.4f} | {result.max_latency_ms:.4f} | "
            f"{result.throughput_requests_per_s:,.2f} |"
        )

    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- latency는 `GET /recommendations/users/{user_id}?k=...` 호출 기준입니다.",
            "- ASGI test client를 사용하므로 FastAPI endpoint 비용은 포함하지만 network hop은 "
            "포함하지 않습니다.",
            "- service 직접 호출 benchmark와 비교하면 API routing, validation, serialization, "
            "JSON decode overhead를 볼 수 있습니다.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _validate_config(config: ServingBenchmarkConfig) -> None:
    if not config.batch_sizes:
        msg = "batch_sizes는 비어 있을 수 없습니다."
        raise ServingBenchmarkError(msg)
    if any(batch_size < 1 for batch_size in config.batch_sizes):
        msg = f"모든 batch size는 1 이상이어야 합니다: {config.batch_sizes}"
        raise ServingBenchmarkError(msg)
    if config.warmup_iterations < 0:
        msg = "warmup_iterations는 0 이상이어야 합니다."
        raise ServingBenchmarkError(msg)
    if config.iterations < 1:
        msg = "iterations는 1 이상이어야 합니다."
        raise ServingBenchmarkError(msg)
    if config.k < 1:
        msg = "k는 1 이상이어야 합니다."
        raise ServingBenchmarkError(msg)
    if config.min_user_id < 1:
        msg = "min_user_id는 1 이상이어야 합니다."
        raise ServingBenchmarkError(msg)
    if config.max_user_id < config.min_user_id:
        msg = "max_user_id는 min_user_id 이상이어야 합니다."
        raise ServingBenchmarkError(msg)


def _validate_endpoint_template(endpoint_template: str) -> None:
    field_names = {
        field_name
        for _, field_name, _, _ in Formatter().parse(endpoint_template)
        if field_name is not None
    }
    required_fields = {"user_id", "k"}
    if not required_fields.issubset(field_names):
        msg = (
            "endpoint_template에는 {user_id}와 {k} placeholder가 필요합니다: "
            f"{endpoint_template}"
        )
        raise ServingBenchmarkError(msg)


def _sample_user_batches(
    *,
    rng: np.random.Generator,
    batch_size: int,
    total_iterations: int,
    min_user_id: int,
    max_user_id: int,
) -> np.ndarray:
    return rng.integers(
        min_user_id,
        max_user_id + 1,
        size=(total_iterations, batch_size),
        dtype=np.int64,
    )


def _warm_up_service(
    *,
    service: ServingBenchmarkService,
    user_batches: np.ndarray,
    k: int,
) -> None:
    for user_ids in user_batches:
        for user_id in user_ids:
            service.recommend(user_id=int(user_id), k=k)


def _measure_service(
    *,
    service: ServingBenchmarkService,
    user_batches: np.ndarray,
    k: int,
) -> tuple[tuple[int, ...], int]:
    request_latencies_ns: list[int] = []
    started_ns = time.perf_counter_ns()
    for user_ids in user_batches:
        for user_id in user_ids:
            request_started_ns = time.perf_counter_ns()
            recommendations = service.recommend(user_id=int(user_id), k=k)
            request_latencies_ns.append(time.perf_counter_ns() - request_started_ns)
            if len(recommendations) > k:
                msg = f"service가 k보다 많은 추천을 반환했습니다: {len(recommendations)} > {k}"
                raise ServingBenchmarkError(msg)
    elapsed_ns = time.perf_counter_ns() - started_ns
    return tuple(request_latencies_ns), elapsed_ns


def _warm_up_http_endpoint(
    *,
    client: HttpBenchmarkClient,
    user_batches: np.ndarray,
    k: int,
    endpoint_template: str,
) -> None:
    for user_ids in user_batches:
        for user_id in user_ids:
            _request_http_endpoint(
                client=client,
                user_id=int(user_id),
                k=k,
                endpoint_template=endpoint_template,
            )


def _measure_http_endpoint(
    *,
    client: HttpBenchmarkClient,
    user_batches: np.ndarray,
    k: int,
    endpoint_template: str,
) -> tuple[tuple[int, ...], int, str, str]:
    request_latencies_ns: list[int] = []
    model_name = ""
    decoder_name = ""
    started_ns = time.perf_counter_ns()
    for user_ids in user_batches:
        for user_id in user_ids:
            request_started_ns = time.perf_counter_ns()
            payload = _request_http_endpoint(
                client=client,
                user_id=int(user_id),
                k=k,
                endpoint_template=endpoint_template,
            )
            request_latencies_ns.append(time.perf_counter_ns() - request_started_ns)
            if not model_name:
                model_name = str(payload["model"])
                decoder_name = str(payload["decoder"])
    elapsed_ns = time.perf_counter_ns() - started_ns
    return tuple(request_latencies_ns), elapsed_ns, model_name, decoder_name


def _request_http_endpoint(
    *,
    client: HttpBenchmarkClient,
    user_id: int,
    k: int,
    endpoint_template: str,
) -> dict[str, Any]:
    url = endpoint_template.format(user_id=user_id, k=k)
    response = client.get(url)
    if response.status_code != 200:
        msg = f"HTTP endpoint가 200이 아닌 상태를 반환했습니다: {response.status_code}"
        raise ServingBenchmarkError(msg)
    payload = cast(dict[str, Any], response.json())
    _validate_http_payload(payload=payload, expected_user_id=user_id, k=k)
    return payload


def _validate_http_payload(*, payload: dict[str, Any], expected_user_id: int, k: int) -> None:
    required_fields = {"user_id", "model", "decoder", "items", "latency_ms"}
    missing_fields = sorted(required_fields - set(payload))
    if missing_fields:
        msg = f"HTTP response에 필요한 필드가 없습니다: {missing_fields}"
        raise ServingBenchmarkError(msg)
    if int(payload["user_id"]) != expected_user_id:
        msg = f"HTTP response user_id가 요청과 다릅니다: {payload['user_id']} != {expected_user_id}"
        raise ServingBenchmarkError(msg)
    items = payload["items"]
    if not isinstance(items, list):
        msg = "HTTP response items는 list여야 합니다."
        raise ServingBenchmarkError(msg)
    if len(items) > k:
        msg = f"HTTP endpoint가 k보다 많은 추천을 반환했습니다: {len(items)} > {k}"
        raise ServingBenchmarkError(msg)
    if float(payload["latency_ms"]) < 0:
        msg = "HTTP response latency_ms는 0 이상이어야 합니다."
        raise ServingBenchmarkError(msg)


def _summarize_result(
    *,
    batch_size: int,
    request_latencies_ns: Sequence[int],
    measured_elapsed_ns: int,
) -> ServingBenchmarkResult:
    if not request_latencies_ns:
        msg = "측정된 request latency가 없습니다."
        raise ServingBenchmarkError(msg)
    latencies_ms = np.asarray(request_latencies_ns, dtype=np.float64) / 1_000_000
    total_requests = len(request_latencies_ns)
    return ServingBenchmarkResult(
        batch_size=batch_size,
        mean_latency_ms=fmean(float(latency) for latency in latencies_ms),
        p50_latency_ms=float(np.percentile(latencies_ms, 50)),
        p95_latency_ms=float(np.percentile(latencies_ms, 95)),
        max_latency_ms=float(latencies_ms.max()),
        throughput_requests_per_s=total_requests / max(measured_elapsed_ns / 1_000_000_000, 1e-12),
        total_requests=total_requests,
    )

from pathlib import Path

import pytest

from apps.api.services.base import RecommendedItem
from recsys.benchmark import (
    ServingBenchmarkConfig,
    ServingBenchmarkError,
    run_http_endpoint_benchmark,
    run_serving_benchmark,
    write_http_endpoint_benchmark_report,
    write_serving_benchmark_report,
)


def test_run_serving_benchmark_supports_configured_batch_sizes() -> None:
    summary = run_serving_benchmark(
        _StaticRecommendationService(),
        config=ServingBenchmarkConfig(
            batch_sizes=(1, 4),
            warmup_iterations=1,
            iterations=2,
            k=3,
            random_seed=7,
            min_user_id=1,
            max_user_id=5,
        ),
        source="test",
    )

    assert summary.source == "test"
    assert summary.model_name == "test-model"
    assert summary.decoder_name == "test-decoder"
    assert [result.batch_size for result in summary.results] == [1, 4]
    assert [result.total_requests for result in summary.results] == [2, 8]
    for result in summary.results:
        assert result.mean_latency_ms >= 0
        assert result.p50_latency_ms >= 0
        assert result.p95_latency_ms >= 0
        assert result.max_latency_ms >= 0
        assert result.throughput_requests_per_s > 0


def test_run_serving_benchmark_rejects_invalid_config() -> None:
    with pytest.raises(ServingBenchmarkError, match="batch size"):
        run_serving_benchmark(
            _StaticRecommendationService(),
            config=ServingBenchmarkConfig(batch_sizes=(0,)),
        )


def test_run_serving_benchmark_rejects_too_many_recommendations() -> None:
    with pytest.raises(ServingBenchmarkError, match="k보다 많은 추천"):
        run_serving_benchmark(
            _TooManyRecommendationService(),
            config=ServingBenchmarkConfig(
                batch_sizes=(1,),
                warmup_iterations=0,
                iterations=1,
                k=1,
            ),
        )


def test_write_serving_benchmark_report(tmp_path: Path) -> None:
    summary = run_serving_benchmark(
        _StaticRecommendationService(),
        config=ServingBenchmarkConfig(
            batch_sizes=(1,),
            warmup_iterations=0,
            iterations=1,
            k=2,
        ),
        source="test",
    )
    report_path = write_serving_benchmark_report(tmp_path / "serving_benchmark.md", summary)

    report = report_path.read_text(encoding="utf-8")
    assert "# Serving 벤치마크 리포트" in report
    assert "평균 latency ms" in report
    assert "throughput req/s" in report
    assert "test-model" in report


def test_run_http_endpoint_benchmark_supports_configured_batch_sizes() -> None:
    summary = run_http_endpoint_benchmark(
        _StaticHttpClient(),
        config=ServingBenchmarkConfig(
            batch_sizes=(1, 4),
            warmup_iterations=1,
            iterations=2,
            k=3,
            random_seed=7,
            min_user_id=1,
            max_user_id=5,
        ),
        source="test-http",
    )

    assert summary.source == "test-http"
    assert summary.model_name == "http-test-model"
    assert summary.decoder_name == "http-test-decoder"
    assert summary.endpoint_template == "/recommendations/users/{user_id}?k={k}"
    assert [result.batch_size for result in summary.results] == [1, 4]
    assert [result.total_requests for result in summary.results] == [2, 8]
    for result in summary.results:
        assert result.mean_latency_ms >= 0
        assert result.p50_latency_ms >= 0
        assert result.p95_latency_ms >= 0
        assert result.max_latency_ms >= 0
        assert result.throughput_requests_per_s > 0


def test_run_http_endpoint_benchmark_rejects_bad_response() -> None:
    with pytest.raises(ServingBenchmarkError, match="200이 아닌"):
        run_http_endpoint_benchmark(
            _FailingHttpClient(),
            config=ServingBenchmarkConfig(
                batch_sizes=(1,),
                warmup_iterations=0,
                iterations=1,
                k=1,
            ),
        )


def test_write_http_endpoint_benchmark_report(tmp_path: Path) -> None:
    summary = run_http_endpoint_benchmark(
        _StaticHttpClient(),
        config=ServingBenchmarkConfig(
            batch_sizes=(1,),
            warmup_iterations=0,
            iterations=1,
            k=2,
        ),
        source="test-http",
    )
    report_path = write_http_endpoint_benchmark_report(
        tmp_path / "http_serving_benchmark.md",
        summary,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "# HTTP Endpoint 벤치마크 리포트" in report
    assert "response model serialization" in report
    assert "throughput req/s" in report
    assert "http-test-model" in report


class _StaticRecommendationService:
    model_name = "test-model"
    decoder_name = "test-decoder"

    def recommend(self, *, user_id: int, k: int) -> list[RecommendedItem]:
        return [
            RecommendedItem(
                item_id=user_id * 100 + index,
                title=f"Item {index}",
                semantic_id=(index,),
                score=1.0 / (index + 1),
            )
            for index in range(k)
        ]


class _TooManyRecommendationService:
    model_name = "bad-model"
    decoder_name = "bad-decoder"

    def recommend(self, *, user_id: int, k: int) -> list[RecommendedItem]:
        return [
            RecommendedItem(
                item_id=user_id * 100 + index,
                title=f"Item {index}",
                semantic_id=(index,),
                score=1.0,
            )
            for index in range(k + 1)
        ]


class _StaticHttpResponse:
    status_code = 200

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


class _FailingHttpResponse:
    status_code = 500

    def json(self) -> dict[str, object]:
        return {}


class _StaticHttpClient:
    def get(self, url: str) -> _StaticHttpResponse:
        user_id_text = url.removeprefix("/recommendations/users/").split("?", maxsplit=1)[0]
        k_text = url.rsplit("k=", maxsplit=1)[-1]
        user_id = int(user_id_text)
        k = int(k_text)
        return _StaticHttpResponse(
            {
                "user_id": user_id,
                "model": "http-test-model",
                "decoder": "http-test-decoder",
                "latency_ms": 0.1,
                "items": [
                    {
                        "item_id": user_id * 100 + index,
                        "title": f"Item {index}",
                        "semantic_id": [index],
                        "score": 1.0 / (index + 1),
                    }
                    for index in range(k)
                ],
            }
        )


class _FailingHttpClient:
    def get(self, url: str) -> _FailingHttpResponse:
        return _FailingHttpResponse()

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from apps.api.services import (
    MockRecommendationService,
    RecommendedItem,
    build_recommendation_service_from_environment,
)


class _StaticRecommendationService:
    model_name = "test-model"
    decoder_name = "test-decoder"

    def recommend(self, *, user_id: int, k: int) -> list[RecommendedItem]:
        return [
            RecommendedItem(
                item_id=user_id + index,
                title=f"Test Item {index}",
                semantic_id=(index, index + 1),
                score=1.0 - index * 0.01,
            )
            for index in range(k)
        ]


def test_get_user_recommendations_returns_mock_items() -> None:
    client = TestClient(create_app())

    response = client.get("/recommendations/users/123?k=3")

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == 123
    assert payload["model"] == "mock-generative-retrieval-static"
    assert payload["decoder"] == "static_sparse_matrix"
    assert payload["latency_ms"] >= 0
    assert len(payload["items"]) == 3
    scores = [item["score"] for item in payload["items"]]
    assert scores == sorted(scores, reverse=True)
    for item in payload["items"]:
        assert set(item) == {"item_id", "title", "semantic_id", "score"}
        assert isinstance(item["item_id"], int)
        assert isinstance(item["title"], str)
        assert len(item["semantic_id"]) == 4
        assert 0 <= item["score"] <= 1


def test_get_user_recommendations_uses_default_k() -> None:
    client = TestClient(create_app())

    response = client.get("/recommendations/users/123")

    assert response.status_code == 200
    assert len(response.json()["items"]) == 20


def test_get_user_recommendations_is_deterministic_for_same_user() -> None:
    client = TestClient(create_app())

    first = client.get("/recommendations/users/123?k=5").json()
    second = client.get("/recommendations/users/123?k=5").json()

    assert first["items"] == second["items"]


def test_get_user_recommendations_validates_user_id_and_k() -> None:
    client = TestClient(create_app())

    assert client.get("/recommendations/users/0?k=3").status_code == 422
    assert client.get("/recommendations/users/123?k=0").status_code == 422


def test_get_user_recommendations_does_not_apply_arbitrary_k_cap() -> None:
    client = TestClient(create_app())

    response = client.get("/recommendations/users/123?k=101")

    assert response.status_code == 200
    assert len(response.json()["items"]) == 101


def test_get_user_recommendations_uses_injected_service() -> None:
    client = TestClient(create_app(_StaticRecommendationService()))

    response = client.get("/recommendations/users/10?k=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "test-model"
    assert payload["decoder"] == "test-decoder"
    assert [item["item_id"] for item in payload["items"]] == [10, 11]


def test_build_recommendation_service_uses_mock_when_model_env_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STATIC_REC_GENERATIVE_CHECKPOINT", raising=False)
    monkeypatch.delenv("STATIC_REC_SEMANTIC_ID_PATH", raising=False)
    monkeypatch.delenv("STATIC_REC_USER_HISTORY_PARQUET", raising=False)

    service = build_recommendation_service_from_environment()

    assert isinstance(service, MockRecommendationService)


def test_build_recommendation_service_rejects_partial_model_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_REC_GENERATIVE_CHECKPOINT", "/tmp/model.pt")
    monkeypatch.delenv("STATIC_REC_SEMANTIC_ID_PATH", raising=False)
    monkeypatch.delenv("STATIC_REC_USER_HISTORY_PARQUET", raising=False)

    with pytest.raises(ValueError, match="환경변수"):
        build_recommendation_service_from_environment()


def test_build_recommendation_service_rejects_missing_model_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_REC_GENERATIVE_CHECKPOINT", "/tmp/missing-model.pt")
    monkeypatch.setenv("STATIC_REC_SEMANTIC_ID_PATH", "/tmp/missing-semantic.json")
    monkeypatch.setenv("STATIC_REC_USER_HISTORY_PARQUET", "/tmp/missing-history.parquet")

    with pytest.raises(FileNotFoundError, match="찾을 수 없습니다"):
        build_recommendation_service_from_environment()

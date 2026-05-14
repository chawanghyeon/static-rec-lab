from fastapi.testclient import TestClient

from apps.api.main import create_app


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

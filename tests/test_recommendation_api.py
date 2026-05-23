from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import apps.api.services as api_services
from apps.api.main import create_app
from apps.api.services import ModelRecommendationConfig, build_default_recommendation_service
from recsys.decoding import StaticDecodingIndex
from recsys.models import (
    GenerativeRetriever,
    GenerativeRetrieverConfig,
    GenerativeTrainingMetrics,
    build_item_index_from_codec,
    infer_semantic_vocab_size,
    save_checkpoint,
)
from recsys.semantic_id import SemanticIdCodec


@pytest.fixture(scope="module")
def model_config(tmp_path_factory: pytest.TempPathFactory) -> ModelRecommendationConfig:
    tmp_path = tmp_path_factory.mktemp("api_model")
    codec = SemanticIdCodec(
        {item_id: [index // 16, index % 16] for index, item_id in enumerate(range(1000, 1128))}
    )
    semantic_id_path = codec.save_json(tmp_path / "semantic_ids.json")
    static_decoding_index_path = StaticDecodingIndex.from_codec(
        codec,
        dense_lookup_layers=1,
    ).save_npz(tmp_path / "static_decoding_index.npz")
    history_path = tmp_path / "histories.parquet"
    pd.DataFrame(
        {
            "user_id": [10, 123, 123],
            "history_item_ids": [[1002], [1000], [1000, 1001]],
            "target_timestamp": [1, 1, 2],
        }
    ).to_parquet(history_path, index=False)

    item_to_index = build_item_index_from_codec(codec)
    model = GenerativeRetriever(
        GenerativeRetrieverConfig(
            item_vocab_size=len(item_to_index) + 2,
            semantic_vocab_size=infer_semantic_vocab_size(codec),
            semantic_id_length=codec.semantic_id_length,
            max_history_length=4,
            d_model=16,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
    )
    checkpoint_path = save_checkpoint(
        tmp_path / "model.pt",
        model=model,
        item_to_index=item_to_index,
        metrics=GenerativeTrainingMetrics(
            loss=0.0,
            token_accuracy=0.0,
            sequence_accuracy=0.0,
            num_examples=0,
        ),
    )
    return ModelRecommendationConfig(
        checkpoint_path=checkpoint_path,
        semantic_id_path=semantic_id_path,
        user_history_path=history_path,
        device="cpu",
        beam_size=128,
        static_decoding_index_path=static_decoding_index_path,
    )


@pytest.fixture
def client(
    model_config: ModelRecommendationConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    monkeypatch.setattr(api_services, "DEFAULT_MODEL_CONFIG", model_config)

    with TestClient(create_app()) as test_client:
        yield test_client


def test_get_user_recommendations_returns_model_items(client: TestClient) -> None:

    response = client.get("/recommendations/users/123?k=3")

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == 123
    assert payload["model"] == "generative-retrieval-static"
    assert payload["decoder"] == "static_decoding_pt"
    assert payload["latency_ms"] >= 0
    assert len(payload["items"]) == 3
    scores = [item["score"] for item in payload["items"]]
    assert scores == sorted(scores, reverse=True)
    for item in payload["items"]:
        assert set(item) == {"item_id", "title", "semantic_id", "score"}
        assert isinstance(item["item_id"], int)
        assert item["title"] == f"Item {item['item_id']}"
        assert len(item["semantic_id"]) == 2
        assert 0 <= item["score"] <= 1


def test_get_user_recommendations_uses_default_k(client: TestClient) -> None:
    response = client.get("/recommendations/users/123")

    assert response.status_code == 200
    assert len(response.json()["items"]) == 20


def test_get_user_recommendations_is_deterministic_for_same_user(client: TestClient) -> None:
    first = client.get("/recommendations/users/123?k=5").json()
    second = client.get("/recommendations/users/123?k=5").json()

    assert first["items"] == second["items"]


def test_get_user_recommendations_validates_user_id_and_k(client: TestClient) -> None:
    assert client.get("/recommendations/users/0?k=3").status_code == 422
    assert client.get("/recommendations/users/123?k=0").status_code == 422


def test_get_user_recommendations_does_not_apply_arbitrary_k_cap(client: TestClient) -> None:
    response = client.get("/recommendations/users/123?k=101")

    assert response.status_code == 200
    assert len(response.json()["items"]) == 101


def test_build_default_service_rejects_missing_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_services,
        "DEFAULT_MODEL_CONFIG",
        ModelRecommendationConfig(
            checkpoint_path=tmp_path / "model.pt",
            semantic_id_path=tmp_path / "semantic_ids.json",
            user_history_path=tmp_path / "valid.parquet",
            device="cpu",
            beam_size=20,
            static_decoding_index_path=tmp_path / "static_decoding_index.npz",
        ),
    )

    with pytest.raises(FileNotFoundError, match="serving artifact"):
        build_default_recommendation_service()


def test_app_startup_rejects_missing_default_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_services,
        "DEFAULT_MODEL_CONFIG",
        ModelRecommendationConfig(
            checkpoint_path=tmp_path / "model.pt",
            semantic_id_path=tmp_path / "semantic_ids.json",
            user_history_path=tmp_path / "valid.parquet",
            device="cpu",
            beam_size=20,
            static_decoding_index_path=tmp_path / "static_decoding_index.npz",
        ),
    )

    with (
        pytest.raises(FileNotFoundError, match="serving artifact"),
        TestClient(create_app()),
    ):
        pass


def test_build_default_service_rejects_missing_static_decoding_index(
    model_config: ModelRecommendationConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_services,
        "DEFAULT_MODEL_CONFIG",
        replace(model_config, static_decoding_index_path=tmp_path / "static_decoding_index.npz"),
    )

    with pytest.raises(FileNotFoundError, match="static_decoding_index_path"):
        build_default_recommendation_service()

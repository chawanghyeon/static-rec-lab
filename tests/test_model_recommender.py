from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from apps.api.services import ModelRecommendationConfig, ModelRecommendationService, UserHistory
from recsys.decoding import BeamSearchResult, StaticDecodingIndex
from recsys.models import (
    GenerativeRetriever,
    GenerativeRetrieverConfig,
    GenerativeTrainingMetrics,
    save_checkpoint,
)
from recsys.semantic_id import SemanticIdCodec


def test_model_recommendation_service_returns_constrained_items(tmp_path: Path) -> None:
    codec = SemanticIdCodec({20: [0, 1], 30: [1, 2], 40: [2, 3]})
    semantic_id_path = codec.save_json(tmp_path / "semantic_ids.json")
    static_decoding_index_path = StaticDecodingIndex.from_codec(
        codec,
        dense_lookup_layers=1,
    ).save_npz(tmp_path / "static_decoding_index.npz")
    history_path = tmp_path / "histories.parquet"
    pd.DataFrame(
        {
            "user_id": [1, 1],
            "history_item_ids": [[10], [10, 20]],
            "history_feedback_ids": [[3], [3, 3]],
            "target_timestamp": [1, 2],
        }
    ).to_parquet(history_path, index=False)
    movie_path = tmp_path / "movies.csv"
    movie_titles = {
        20: "Movie Twenty",
        30: "Movie Thirty",
        40: "Movie Forty",
    }
    pd.DataFrame(
        {
            "movieId": list(movie_titles),
            "title": list(movie_titles.values()),
            "genres": ["Action", "Drama", "Comedy"],
        }
    ).to_csv(movie_path, index=False)
    model = GenerativeRetriever(
        GenerativeRetrieverConfig(
            item_vocab_size=6,
            semantic_vocab_size=7,
            semantic_id_length=2,
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
        item_to_index={10: 2, 20: 3, 30: 4, 40: 5},
        metrics=GenerativeTrainingMetrics(
            loss=0.0,
            token_accuracy=0.0,
            sequence_accuracy=0.0,
            num_examples=0,
        ),
    )
    service = ModelRecommendationService.from_config(
        ModelRecommendationConfig(
            checkpoint_path=checkpoint_path,
            semantic_id_path=semantic_id_path,
            user_history_path=history_path,
            movie_metadata_path=movie_path,
            device="cpu",
            beam_size=3,
            static_decoding_index_path=static_decoding_index_path,
        )
    )

    recommendations = service.recommend(user_id=1, k=2)

    assert len(recommendations) == 2
    assert service.model_name == "generative-retrieval-static"
    assert service.decoder_name == "static_decoding_pt"
    for item in recommendations:
        assert codec.has_semantic_id(item.semantic_id)
        assert item.title == movie_titles[item.item_id]
        assert 0 <= item.score <= 1


def test_model_recommendation_service_filters_history_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = SemanticIdCodec({20: [0, 0], 30: [0, 1], 40: [0, 2]})
    captured_kwargs: dict[str, Any] = {}

    def fake_generate(**kwargs: Any) -> tuple[BeamSearchResult, ...]:
        captured_kwargs.update(kwargs)
        return (
            BeamSearchResult(semantic_id=(0, 0), score=0.9, log_score=-0.1),
            BeamSearchResult(semantic_id=(0, 1), score=0.8, log_score=-0.2),
            BeamSearchResult(semantic_id=(0, 2), score=0.7, log_score=-0.3),
        )

    monkeypatch.setattr(
        "apps.api.services.model_recommender.generate_semantic_ids_with_static_decoding",
        fake_generate,
    )
    service = ModelRecommendationService(
        model=object(),
        item_to_index={20: 2, 30: 3, 40: 4},
        codec=codec,
        user_histories={1: UserHistory(item_ids=(20,), feedback_ids=(3,))},
        movie_titles={
            20: "Seen Movie",
            30: "First New Movie",
            40: "Second New Movie",
        },
        device=torch.device("cpu"),
        beam_size=1,
        static_decoding_index_path=None,
    )

    recommendations = service.recommend(user_id=1, k=2)

    assert [item.item_id for item in recommendations] == [30, 40]
    assert captured_kwargs["beam_size"] == 3
    assert captured_kwargs["max_results"] == 3

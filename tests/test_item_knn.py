from pathlib import Path

import pandas as pd

from recsys.baseline import (
    CooccurrenceCandidate,
    ItemKNNModel,
    PopularItem,
    evaluate_item_knn_model,
    fit_item_knn_model,
    load_item_knn_model,
    save_item_knn_model,
)


def test_fit_item_knn_model_builds_neighbors_from_history_targets() -> None:
    train_frame = pd.DataFrame(
        {
            "history_item_ids": [[1, 2], [1], [2], [1, 3]],
            "target_item_id": [10, 10, 20, 20],
        }
    )

    model = fit_item_knn_model(train_frame, max_candidates_per_item=10, max_history_items=50)

    assert model.neighbors[1] == (
        CooccurrenceCandidate(item_id=10, count=2),
        CooccurrenceCandidate(item_id=20, count=1),
    )
    assert model.neighbors[2] == (
        CooccurrenceCandidate(item_id=10, count=1),
        CooccurrenceCandidate(item_id=20, count=1),
    )


def test_item_knn_recommend_aggregates_history_scores_and_excludes_seen() -> None:
    model = ItemKNNModel(
        neighbors={
            1: (
                CooccurrenceCandidate(item_id=10, count=3),
                CooccurrenceCandidate(item_id=20, count=1),
            ),
            2: (
                CooccurrenceCandidate(item_id=20, count=5),
                CooccurrenceCandidate(item_id=30, count=1),
            ),
        },
        popularity_items=(
            PopularItem(item_id=10, count=3, rank=1),
            PopularItem(item_id=20, count=2, rank=2),
            PopularItem(item_id=30, count=1, rank=3),
            PopularItem(item_id=40, count=1, rank=4),
        ),
        num_train_examples=4,
        max_candidates_per_item=10,
        max_history_items=50,
    )

    assert model.recommend(history_item_ids=[1, 2], k=3) == [20, 10, 30]


def test_item_knn_recommend_falls_back_to_popularity() -> None:
    model = ItemKNNModel(
        neighbors={},
        popularity_items=(
            PopularItem(item_id=10, count=3, rank=1),
            PopularItem(item_id=20, count=2, rank=2),
        ),
        num_train_examples=5,
        max_candidates_per_item=10,
        max_history_items=50,
    )

    assert model.recommend(history_item_ids=[10], k=2) == [20]


def test_item_knn_model_round_trip_json(tmp_path: Path) -> None:
    model = ItemKNNModel(
        neighbors={1: (CooccurrenceCandidate(item_id=10, count=3),)},
        popularity_items=(PopularItem(item_id=10, count=3, rank=1),),
        num_train_examples=3,
        max_candidates_per_item=10,
        max_history_items=50,
    )
    model_path = tmp_path / "item_knn.json"

    save_item_knn_model(model, model_path)

    assert load_item_knn_model(model_path) == model


def test_evaluate_item_knn_model_reports_metrics() -> None:
    model = ItemKNNModel(
        neighbors={1: (CooccurrenceCandidate(item_id=10, count=3),)},
        popularity_items=(
            PopularItem(item_id=10, count=3, rank=1),
            PopularItem(item_id=20, count=2, rank=2),
        ),
        num_train_examples=3,
        max_candidates_per_item=10,
        max_history_items=50,
    )
    eval_frame = pd.DataFrame({"history_item_ids": [[1], [20]], "target_item_id": [10, 10]})

    metrics = evaluate_item_knn_model(model, eval_frame, cutoffs=[1])

    assert metrics[1].recall == 1.0

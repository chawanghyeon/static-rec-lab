from pathlib import Path

import pandas as pd

from recsys.data import (
    PreprocessConfig,
    filter_users_by_min_interactions,
    load_ratings_csv,
    make_sequential_splits,
    sort_interactions,
    write_split_parquets,
)


def test_load_ratings_csv_normalizes_movielens_columns(tmp_path: Path) -> None:
    ratings_csv = tmp_path / "ratings.csv"
    pd.DataFrame(
        {
            "userId": [1],
            "movieId": [10],
            "rating": [4.5],
            "timestamp": [100],
        }
    ).to_csv(ratings_csv, index=False)

    ratings = load_ratings_csv(ratings_csv)

    assert ratings.to_dict("records") == [
        {
            "user_id": 1,
            "item_id": 10,
            "rating": 4.5,
            "timestamp": 100,
        }
    ]


def test_sort_interactions_orders_by_user_and_timestamp() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [2, 1, 2, 1],
            "item_id": [20, 11, 21, 10],
            "rating": [4.0, 3.0, 5.0, 4.0],
            "timestamp": [30, 20, 10, 10],
        }
    )

    sorted_interactions = sort_interactions(interactions)

    assert sorted_interactions[["user_id", "item_id", "timestamp"]].to_dict("records") == [
        {"user_id": 1, "item_id": 10, "timestamp": 10},
        {"user_id": 1, "item_id": 11, "timestamp": 20},
        {"user_id": 2, "item_id": 21, "timestamp": 10},
        {"user_id": 2, "item_id": 20, "timestamp": 30},
    ]


def test_filter_users_by_min_interactions_removes_short_histories() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2],
            "item_id": [10, 11, 20, 21, 22],
            "rating": [4.0, 4.0, 5.0, 5.0, 5.0],
            "timestamp": [1, 2, 1, 2, 3],
        }
    )

    filtered = filter_users_by_min_interactions(interactions, min_interactions=3)

    assert filtered["user_id"].tolist() == [2, 2, 2]


def test_make_sequential_splits_uses_last_items_for_valid_and_test() -> None:
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 1, 2, 2],
            "movieId": [10, 11, 12, 13, 14, 20, 21],
            "rating": [4.0, 4.0, 5.0, 5.0, 3.0, 5.0, 5.0],
            "timestamp": [1, 2, 3, 4, 5, 1, 2],
        }
    )

    splits = make_sequential_splits(
        ratings,
        PreprocessConfig(min_interactions=3, max_history_length=None),
    )

    assert splits["train"].to_dict("records") == [
        {
            "user_id": 1,
            "history_item_ids": [10],
            "target_item_id": 11,
            "target_timestamp": 2,
            "history_length": 1,
        },
        {
            "user_id": 1,
            "history_item_ids": [10, 11],
            "target_item_id": 12,
            "target_timestamp": 3,
            "history_length": 2,
        },
    ]
    assert splits["valid"].to_dict("records") == [
        {
            "user_id": 1,
            "history_item_ids": [10, 11, 12],
            "target_item_id": 13,
            "target_timestamp": 4,
            "history_length": 3,
        }
    ]
    assert splits["test"].to_dict("records") == [
        {
            "user_id": 1,
            "history_item_ids": [10, 11, 12, 13],
            "target_item_id": 14,
            "target_timestamp": 5,
            "history_length": 4,
        }
    ]


def test_make_sequential_splits_truncates_history() -> None:
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 1],
            "movieId": [10, 11, 12, 13, 14],
            "rating": [4.0, 4.0, 5.0, 5.0, 3.0],
            "timestamp": [1, 2, 3, 4, 5],
        }
    )

    splits = make_sequential_splits(
        ratings,
        PreprocessConfig(min_interactions=3, max_history_length=2),
    )

    assert splits["test"].iloc[0]["history_item_ids"] == [12, 13]
    assert splits["test"].iloc[0]["history_length"] == 2


def test_write_split_parquets_creates_expected_files(tmp_path: Path) -> None:
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1],
            "movieId": [10, 11, 12],
            "rating": [4.0, 4.0, 5.0],
            "timestamp": [1, 2, 3],
        }
    )
    splits = make_sequential_splits(
        ratings,
        PreprocessConfig(min_interactions=3, max_history_length=None),
    )

    paths = write_split_parquets(splits, tmp_path)

    assert sorted(paths) == ["test", "train", "valid"]
    assert (tmp_path / "train.parquet").is_file()
    assert (tmp_path / "valid.parquet").is_file()
    assert (tmp_path / "test.parquet").is_file()
    assert pd.read_parquet(paths["valid"]).to_dict("records") == [
        {
            "user_id": 1,
            "history_item_ids": [10],
            "target_item_id": 11,
            "target_timestamp": 2,
            "history_length": 1,
        }
    ]

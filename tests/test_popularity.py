from pathlib import Path

import pandas as pd

from recsys.baseline import (
    PopularItem,
    PopularityModel,
    evaluate_popularity_model,
    fit_popularity_model,
    load_popularity_model,
    save_popularity_model,
    write_baseline_report,
)


def test_fit_popularity_model_ranks_by_count_then_item_id() -> None:
    train_frame = pd.DataFrame({"target_item_id": [20, 10, 20, 30, 10, 20, 10]})

    model = fit_popularity_model(train_frame)

    assert model.items == (
        PopularItem(item_id=10, count=3, rank=1),
        PopularItem(item_id=20, count=3, rank=2),
        PopularItem(item_id=30, count=1, rank=3),
    )
    assert model.num_train_examples == 7


def test_popularity_model_recommend_excludes_history_items() -> None:
    model = PopularityModel(
        items=(
            PopularItem(item_id=10, count=3, rank=1),
            PopularItem(item_id=20, count=2, rank=2),
            PopularItem(item_id=30, count=1, rank=3),
        ),
        num_train_examples=6,
    )

    assert model.recommend(history_item_ids=[10], k=2) == [20, 30]


def test_popularity_model_round_trip_json(tmp_path: Path) -> None:
    model = PopularityModel(
        items=(PopularItem(item_id=10, count=3, rank=1),),
        num_train_examples=3,
    )
    model_path = tmp_path / "popularity.json"

    save_popularity_model(model, model_path)

    assert load_popularity_model(model_path) == model


def test_evaluate_popularity_model_reports_metrics() -> None:
    model = PopularityModel(
        items=(
            PopularItem(item_id=10, count=3, rank=1),
            PopularItem(item_id=20, count=2, rank=2),
            PopularItem(item_id=30, count=1, rank=3),
        ),
        num_train_examples=6,
    )
    eval_frame = pd.DataFrame(
        {
            "history_item_ids": [[10], [30]],
            "target_item_id": [20, 20],
        }
    )

    metrics = evaluate_popularity_model(model, eval_frame, cutoffs=[1, 2])

    assert metrics[1].recall == 0.5
    assert metrics[1].ndcg == 0.5
    assert metrics[2].recall == 1.0
    assert metrics[2].mrr == 0.75


def test_write_baseline_report_creates_markdown(tmp_path: Path) -> None:
    model = PopularityModel(
        items=(PopularItem(item_id=10, count=3, rank=1),),
        num_train_examples=3,
    )
    eval_frame = pd.DataFrame({"history_item_ids": [[]], "target_item_id": [10]})
    metrics = evaluate_popularity_model(model, eval_frame, cutoffs=[1])

    report_path = write_baseline_report(
        report_path=tmp_path / "baseline.md",
        model=model,
        valid_metrics=metrics,
        test_metrics=metrics,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "# Baseline 평가 리포트" in report
    assert "| valid | 1 | 1.000000 | 1.000000 | 1.000000 |" in report

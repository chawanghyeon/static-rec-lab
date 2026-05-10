from pathlib import Path

from recsys.baseline import BaselineEvaluation, write_baseline_comparison_report
from recsys.evaluation import RankingMetrics


def test_write_baseline_comparison_report_contains_multiple_models(tmp_path: Path) -> None:
    metrics = {10: RankingMetrics(k=10, recall=0.1, ndcg=0.2, mrr=0.3)}

    report_path = write_baseline_comparison_report(
        report_path=tmp_path / "baseline.md",
        evaluations=[
            BaselineEvaluation(
                model_name="Popularity",
                description="popularity",
                num_train_examples=100,
                num_items=10,
                valid_metrics=metrics,
                test_metrics=metrics,
            ),
            BaselineEvaluation(
                model_name="Item co-occurrence",
                description="co-occurrence",
                num_train_examples=100,
                num_items=9,
                valid_metrics=metrics,
                test_metrics=metrics,
            ),
        ],
    )

    report = report_path.read_text(encoding="utf-8")
    assert "| Popularity | valid | 10 | 0.100000 | 0.200000 | 0.300000 |" in report
    assert "| Item co-occurrence | test | 10 | 0.100000 | 0.200000 | 0.300000 |" in report

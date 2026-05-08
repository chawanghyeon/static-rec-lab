import math

import pytest

from recsys.evaluation import (
    RankingMetrics,
    evaluate_ranking_at_k,
    mean_ndcg_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_counts_unique_hits() -> None:
    assert recall_at_k([1, 1, 3], [1, 2], k=3) == 0.5


def test_recall_at_k_respects_cutoff() -> None:
    assert recall_at_k([1, 2, 3], [1, 2, 3], k=2) == 2 / 3


def test_recall_at_k_returns_zero_for_empty_relevant_items() -> None:
    assert recall_at_k([1, 2, 3], [], k=3) == 0.0


def test_ndcg_at_k_is_one_for_perfect_ranking() -> None:
    assert ndcg_at_k([1, 2, 3], [1, 2, 3], k=3) == 1.0


def test_ndcg_at_k_penalizes_late_hits() -> None:
    actual = ndcg_at_k([3, 2, 1], [1, 2], k=3)
    expected_dcg = (1.0 / math.log2(3)) + (1.0 / math.log2(4))
    expected_idcg = 1.0 + (1.0 / math.log2(3))

    assert actual == pytest.approx(expected_dcg / expected_idcg)


def test_ndcg_at_k_penalizes_duplicate_recommendations() -> None:
    actual = ndcg_at_k([1, 1, 2], [1, 2], k=3)
    expected_dcg = 1.0 + (1.0 / math.log2(4))
    expected_idcg = 1.0 + (1.0 / math.log2(3))

    assert actual == pytest.approx(expected_dcg / expected_idcg)


def test_ndcg_at_k_returns_zero_for_empty_relevant_items() -> None:
    assert ndcg_at_k([1, 2, 3], [], k=3) == 0.0


def test_reciprocal_rank_returns_first_hit_rank() -> None:
    assert reciprocal_rank([4, 5, 2], [2], k=None) == pytest.approx(1 / 3)


def test_reciprocal_rank_respects_cutoff() -> None:
    assert reciprocal_rank([4, 5, 2], [2], k=2) == 0.0


def test_mean_metrics_average_over_queries() -> None:
    recommendations = [[1, 2, 3], [3, 4, 5]]
    relevant_items = [[2], [5]]

    assert mean_recall_at_k(recommendations, relevant_items, k=2) == 0.5
    assert mean_ndcg_at_k(recommendations, relevant_items, k=3) == pytest.approx(
        ((1.0 / math.log2(3)) + (1.0 / math.log2(4))) / 2
    )
    assert mean_reciprocal_rank(recommendations, relevant_items) == pytest.approx(
        ((1 / 2) + (1 / 3)) / 2
    )


def test_evaluate_ranking_at_k_returns_metric_bundle() -> None:
    metrics = evaluate_ranking_at_k(
        recommendations=[[1, 2, 3], [3, 4, 5]],
        relevant_items=[[2], [6]],
        k=2,
    )

    assert isinstance(metrics, RankingMetrics)
    assert metrics.k == 2
    assert metrics.recall == 0.5
    assert metrics.ndcg == pytest.approx((1 / math.log2(3)) / 2)
    assert metrics.mrr == 0.25


def test_evaluate_ranking_at_k_handles_single_pass_iterables() -> None:
    metrics = evaluate_ranking_at_k(
        recommendations=[iter([1, 2, 3])],
        relevant_items=[iter([2])],
        k=3,
    )

    assert metrics.recall == 1.0
    assert metrics.ndcg == pytest.approx(1 / math.log2(3))
    assert metrics.mrr == 0.5


def test_mean_metrics_handle_single_pass_iterables() -> None:
    assert mean_recall_at_k([iter([1, 2, 3])], [iter([2])], k=3) == 1.0
    assert mean_ndcg_at_k([iter([1, 2, 3])], [iter([2])], k=3) == pytest.approx(1 / math.log2(3))
    assert mean_reciprocal_rank([iter([1, 2, 3])], [iter([2])], k=3) == 0.5


def test_metrics_reject_invalid_k() -> None:
    with pytest.raises(ValueError, match="k"):
        recall_at_k([1], [1], k=0)

    with pytest.raises(ValueError, match="k"):
        ndcg_at_k([1], [1], k=0)

    with pytest.raises(ValueError, match="k"):
        reciprocal_rank([1], [1], k=0)


def test_mean_metrics_require_matching_query_counts() -> None:
    with pytest.raises(ValueError, match="길이"):
        mean_recall_at_k([[1]], [[1], [2]], k=1)


def test_mean_metrics_return_zero_for_empty_inputs() -> None:
    assert mean_recall_at_k([], [], k=1) == 0.0
    assert mean_ndcg_at_k([], [], k=1) == 0.0
    assert mean_reciprocal_rank([], []) == 0.0

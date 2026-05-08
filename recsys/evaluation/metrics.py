"""추천 ranking 평가 지표."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import islice
from statistics import fmean

ItemId = int


@dataclass(frozen=True)
class RankingMetrics:
    """한 cutoff K에서 계산한 평균 ranking 지표."""

    k: int
    recall: float
    ndcg: float
    mrr: float


def recall_at_k(
    recommended_ids: Iterable[ItemId],
    relevant_ids: Iterable[ItemId],
    k: int,
) -> float:
    """Recall@K를 계산한다.

    중복 추천은 여러 번 맞힌 것으로 세지 않는다. 다만 중복 item도 ranking 위치 하나를
    차지하므로, NDCG에서는 자연스럽게 penalty가 된다.
    """
    _validate_k(k)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    hits = _unique_hits_at_k(recommended_ids, relevant, k)
    return hits / len(relevant)


def ndcg_at_k(
    recommended_ids: Iterable[ItemId],
    relevant_ids: Iterable[ItemId],
    k: int,
) -> float:
    """Binary relevance 기준 NDCG@K를 계산한다."""
    _validate_k(k)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    dcg = 0.0
    seen_hits: set[ItemId] = set()
    for rank, item_id in enumerate(recommended_ids, start=1):
        if rank > k:
            break
        if item_id in relevant and item_id not in seen_hits:
            dcg += 1.0 / math.log2(rank + 1)
            seen_hits.add(item_id)

    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def reciprocal_rank(
    recommended_ids: Iterable[ItemId],
    relevant_ids: Iterable[ItemId],
    k: int | None = None,
) -> float:
    """첫 번째 relevant item의 reciprocal rank를 계산한다."""
    if k is not None:
        _validate_k(k)

    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    for rank, item_id in enumerate(recommended_ids, start=1):
        if k is not None and rank > k:
            break
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_recall_at_k(
    recommendations: Sequence[Iterable[ItemId]],
    relevant_items: Sequence[Iterable[ItemId]],
    k: int,
) -> float:
    """여러 query의 평균 Recall@K를 계산한다."""
    pairs = _materialize_query_pairs(recommendations, relevant_items, k)
    if not pairs:
        return 0.0

    return fmean(
        recall_at_k(recommended_ids, relevant_ids, k) for recommended_ids, relevant_ids in pairs
    )


def mean_ndcg_at_k(
    recommendations: Sequence[Iterable[ItemId]],
    relevant_items: Sequence[Iterable[ItemId]],
    k: int,
) -> float:
    """여러 query의 평균 NDCG@K를 계산한다."""
    pairs = _materialize_query_pairs(recommendations, relevant_items, k)
    if not pairs:
        return 0.0

    return fmean(
        ndcg_at_k(recommended_ids, relevant_ids, k) for recommended_ids, relevant_ids in pairs
    )


def mean_reciprocal_rank(
    recommendations: Sequence[Iterable[ItemId]],
    relevant_items: Sequence[Iterable[ItemId]],
    k: int | None = None,
) -> float:
    """여러 query의 MRR을 계산한다."""
    pairs = _materialize_query_pairs(recommendations, relevant_items, k)
    if not pairs:
        return 0.0

    return fmean(
        reciprocal_rank(recommended_ids, relevant_ids, k) for recommended_ids, relevant_ids in pairs
    )


def evaluate_ranking_at_k(
    recommendations: Sequence[Iterable[ItemId]],
    relevant_items: Sequence[Iterable[ItemId]],
    k: int,
) -> RankingMetrics:
    """여러 query의 Recall@K, NDCG@K, MRR@K를 함께 계산한다."""
    pairs = _materialize_query_pairs(recommendations, relevant_items, k)
    if not pairs:
        return RankingMetrics(k=k, recall=0.0, ndcg=0.0, mrr=0.0)

    return RankingMetrics(
        k=k,
        recall=fmean(
            recall_at_k(recommended_ids, relevant_ids, k) for recommended_ids, relevant_ids in pairs
        ),
        ndcg=fmean(
            ndcg_at_k(recommended_ids, relevant_ids, k) for recommended_ids, relevant_ids in pairs
        ),
        mrr=fmean(
            reciprocal_rank(recommended_ids, relevant_ids, k)
            for recommended_ids, relevant_ids in pairs
        ),
    )


def _unique_hits_at_k(
    recommended_ids: Iterable[ItemId],
    relevant: set[ItemId],
    k: int,
) -> int:
    seen_hits: set[ItemId] = set()
    for rank, item_id in enumerate(recommended_ids, start=1):
        if rank > k:
            break
        if item_id in relevant:
            seen_hits.add(item_id)
    return len(seen_hits)


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k는 1 이상이어야 합니다.")


def _validate_same_length(
    recommendations: Sequence[Iterable[ItemId]],
    relevant_items: Sequence[Iterable[ItemId]],
) -> None:
    if len(recommendations) != len(relevant_items):
        raise ValueError("recommendations와 relevant_items의 길이가 같아야 합니다.")


def _materialize_query_pairs(
    recommendations: Sequence[Iterable[ItemId]],
    relevant_items: Sequence[Iterable[ItemId]],
    k: int | None,
) -> list[tuple[tuple[ItemId, ...], tuple[ItemId, ...]]]:
    if k is not None:
        _validate_k(k)
    _validate_same_length(recommendations, relevant_items)

    return [
        (
            tuple(islice(recommended_ids, k)) if k is not None else tuple(recommended_ids),
            tuple(relevant_ids),
        )
        for recommended_ids, relevant_ids in zip(recommendations, relevant_items, strict=True)
    ]

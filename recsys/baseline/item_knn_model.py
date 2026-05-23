"""Item KNN baseline model 타입과 추천 로직."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property

from recsys.baseline.popularity import PopularItem


@dataclass(frozen=True)
class CooccurrenceCandidate:
    """특정 source item과 함께 등장한 candidate item."""

    item_id: int
    count: int


@dataclass(frozen=True)
class ItemKNNModel:
    """Item co-occurrence 기반 baseline."""

    neighbors: Mapping[int, tuple[CooccurrenceCandidate, ...]]
    popularity_items: tuple[PopularItem, ...]
    num_train_examples: int
    max_candidates_per_item: int
    max_history_items: int

    @cached_property
    def popularity_rank(self) -> dict[int, int]:
        """item_id별 popularity rank cache."""
        return {item.item_id: item.rank for item in self.popularity_items}

    def recommend(self, history_item_ids: Iterable[int], k: int) -> list[int]:
        """history item들의 co-occurrence score를 합산해 top-K item을 추천한다."""
        if k < 1:
            msg = "k는 1 이상이어야 합니다."
            raise ValueError(msg)

        history = [int(item_id) for item_id in history_item_ids]
        seen = set(history)
        query_items = history[-self.max_history_items :]

        scores: dict[int, int] = defaultdict(int)
        for source_item_id in query_items:
            for candidate in self.neighbors.get(source_item_id, ()):
                if candidate.item_id in seen:
                    continue
                scores[candidate.item_id] += candidate.count

        ranked_candidates = sorted(
            scores,
            key=lambda item_id: (
                -scores[item_id],
                self.popularity_rank.get(item_id, len(self.popularity_rank) + 1),
                item_id,
            ),
        )
        recommendations = ranked_candidates[:k]
        recommended = set(recommendations)

        for item in self.popularity_items:
            if len(recommendations) == k:
                break
            if item.item_id in seen or item.item_id in recommended:
                continue
            recommendations.append(item.item_id)
            recommended.add(item.item_id)

        return recommendations

    @property
    def num_items(self) -> int:
        """neighbor를 가진 source item 수."""
        return len(self.neighbors)

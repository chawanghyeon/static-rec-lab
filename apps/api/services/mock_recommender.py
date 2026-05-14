"""학습된 모델이 없을 때 사용하는 deterministic mock recommender."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class RecommendedItem:
    """추천 결과 item."""

    item_id: int
    title: str
    semantic_id: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class _CatalogItem:
    item_id: int
    title: str
    semantic_id: tuple[int, ...]


class MockRecommendationService:
    """실제 generative retrieval model 전까지 API contract를 검증하는 mock service."""

    model_name = "mock-generative-retrieval-static"
    decoder_name = "static_sparse_matrix"

    def __init__(self, catalog: tuple[_CatalogItem, ...] | None = None) -> None:
        self._catalog = _build_default_catalog() if catalog is None else catalog

    def recommend(self, *, user_id: int, k: int) -> list[RecommendedItem]:
        """user_id 기준 deterministic ranking을 반환한다."""
        if k < 1:
            msg = "k는 1 이상이어야 합니다."
            raise ValueError(msg)

        scored_items = [
            RecommendedItem(
                item_id=item.item_id,
                title=item.title,
                semantic_id=item.semantic_id,
                score=_score_item(user_id=user_id, item_id=item.item_id),
            )
            for item in _catalog_with_min_size(self._catalog, k)
        ]
        scored_items.sort(key=lambda item: (-item.score, item.item_id))
        return scored_items[:k]


def _score_item(*, user_id: int, item_id: int) -> float:
    payload = f"{user_id}:{item_id}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    raw_score = int.from_bytes(digest, byteorder="big") / ((1 << 64) - 1)
    return round(raw_score, 6)


def _build_default_catalog() -> tuple[_CatalogItem, ...]:
    seed_items = [
        (2959, "Fight Club"),
        (2571, "The Matrix"),
        (356, "Forrest Gump"),
        (318, "The Shawshank Redemption"),
        (296, "Pulp Fiction"),
        (593, "The Silence of the Lambs"),
        (260, "Star Wars: Episode IV - A New Hope"),
        (1196, "Star Wars: Episode V - The Empire Strikes Back"),
        (50, "The Usual Suspects"),
        (527, "Schindler's List"),
    ]
    generated_items = [(100_000 + index, f"Mock Movie {index:03d}") for index in range(118)]
    return tuple(
        _CatalogItem(
            item_id=item_id,
            title=title,
            semantic_id=_semantic_id_for_index(index),
        )
        for index, (item_id, title) in enumerate([*seed_items, *generated_items])
    )


def _catalog_with_min_size(
    catalog: tuple[_CatalogItem, ...],
    min_size: int,
) -> tuple[_CatalogItem, ...]:
    if len(catalog) >= min_size:
        return catalog
    generated_items = tuple(
        _generated_catalog_item(index) for index in range(len(catalog), min_size)
    )
    return (*catalog, *generated_items)


def _generated_catalog_item(index: int) -> _CatalogItem:
    return _CatalogItem(
        item_id=100_000 + index,
        title=f"Mock Movie {index:03d}",
        semantic_id=_semantic_id_for_index(index),
    )


def _semantic_id_for_index(index: int) -> tuple[int, ...]:
    return (
        index // 4096,
        (index // 256) % 16,
        (index // 16) % 16,
        index % 16,
    )

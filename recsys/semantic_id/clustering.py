"""Hierarchical balanced k-means 기반 Semantic ID 생성."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans  # type: ignore[import-untyped]

from recsys.semantic_id.codec import SemanticId, SemanticIdCodec
from recsys.semantic_id.embedding import ItemEmbeddings


@dataclass(frozen=True)
class HierarchicalKMeansConfig:
    """Semantic ID 생성을 위한 hierarchical balanced k-means 설정."""

    depth: int = 4
    branching_factor: int = 16
    random_state: int = 42
    n_init: int = 10
    max_iter: int = 300

    def __post_init__(self) -> None:
        if self.depth < 1:
            msg = "depth는 1 이상이어야 합니다."
            raise ValueError(msg)
        if self.branching_factor < 2:
            msg = "branching_factor는 2 이상이어야 합니다."
            raise ValueError(msg)
        if self.n_init < 1:
            msg = "n_init은 1 이상이어야 합니다."
            raise ValueError(msg)
        if self.max_iter < 1:
            msg = "max_iter는 1 이상이어야 합니다."
            raise ValueError(msg)

    @property
    def capacity(self) -> int:
        return int(self.branching_factor**self.depth)


@dataclass(frozen=True)
class SemanticIdBuildResult:
    """Semantic ID 생성 결과."""

    codec: SemanticIdCodec
    config: HierarchicalKMeansConfig
    embedding_dim: int
    num_context_edges: int


def build_semantic_id_codec(
    item_embeddings: ItemEmbeddings,
    config: HierarchicalKMeansConfig,
) -> SemanticIdBuildResult:
    """item embedding을 hierarchical balanced k-means로 clustering해 Semantic ID codec을 만든다."""
    mapping = build_semantic_id_mapping(
        item_ids=item_embeddings.item_ids,
        embeddings=item_embeddings.embeddings,
        config=config,
    )
    codec = SemanticIdCodec(mapping)
    return SemanticIdBuildResult(
        codec=codec,
        config=config,
        embedding_dim=item_embeddings.embedding_dim,
        num_context_edges=item_embeddings.num_context_edges,
    )


def build_semantic_id_mapping(
    item_ids: tuple[int, ...],
    embeddings: np.ndarray,
    config: HierarchicalKMeansConfig,
) -> dict[int, SemanticId]:
    """item_id별 고정 길이 Semantic ID를 생성한다."""
    _validate_inputs(item_ids, embeddings, config)
    token_paths: dict[int, list[int]] = {item_id: [] for item_id in item_ids}
    indices = np.arange(len(item_ids), dtype=np.int64)
    _assign_tokens(
        item_ids=item_ids,
        embeddings=np.asarray(embeddings, dtype=np.float64),
        indices=indices,
        level=0,
        config=config,
        token_paths=token_paths,
    )
    return {item_id: tuple(tokens) for item_id, tokens in token_paths.items()}


def write_semantic_id_report(
    report_path: str | Path,
    result: SemanticIdBuildResult,
    output_path: str | Path,
) -> Path:
    """Semantic ID 생성 결과를 한국어 markdown 리포트로 저장한다."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Semantic ID 리포트",
        "",
        "## 생성 방식",
        "",
        "- train split의 history-target co-occurrence로 item interaction matrix를 만듭니다.",
        "- sparse matrix를 Truncated SVD로 축소한 뒤 item 빈도와 정규화된 item 순서 "
        "feature를 더합니다.",
        "- hierarchical balanced k-means로 token path를 만들고, 각 subtree capacity를 "
        "넘지 않게 balanced chunk로 나눕니다.",
        "- 이 방식은 clustering 구조를 사용하면서도 모든 item에 중복 없는 고정 길이 "
        "Semantic ID를 부여하기 위한 구현입니다.",
        "",
        "## 설정",
        "",
        f"- depth: {result.config.depth}",
        f"- branching factor: {result.config.branching_factor}",
        f"- capacity: {result.config.capacity}",
        f"- item 수: {result.codec.num_items}",
        f"- Semantic ID 길이: {result.codec.semantic_id_length}",
        f"- embedding dimension: {result.embedding_dim}",
        f"- context edge 수: {result.num_context_edges}",
        f"- output: `{output_path}`",
        "",
        "## 검증",
        "",
        "- 모든 item에 Semantic ID가 부여되었습니다.",
        "- 모든 Semantic ID는 고정 길이입니다.",
        "- 중복 Semantic ID는 없습니다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _assign_tokens(
    item_ids: tuple[int, ...],
    embeddings: np.ndarray,
    indices: np.ndarray,
    level: int,
    config: HierarchicalKMeansConfig,
    token_paths: dict[int, list[int]],
) -> None:
    remaining_depth = config.depth - level
    if remaining_depth == 0:
        return

    _validate_node_capacity(len(indices), config.branching_factor, remaining_depth)
    child_capacity = config.branching_factor ** (remaining_depth - 1)
    num_children = max(1, math.ceil(len(indices) / child_capacity))

    # 각 subtree capacity를 넘지 않도록 k-means cluster order를 balanced chunk로 나눈다.
    if num_children == 1:
        chunks = [indices]
    else:
        ordered_indices = _order_indices_by_kmeans(
            embeddings=embeddings,
            indices=indices,
            num_children=num_children,
            config=config,
        )
        chunks = [
            ordered_indices[start : start + child_capacity]
            for start in range(0, len(ordered_indices), child_capacity)
        ]

    for token, child_indices in enumerate(chunks):
        for index_value in child_indices:
            token_paths[item_ids[int(index_value)]].append(token)
        _assign_tokens(
            item_ids=item_ids,
            embeddings=embeddings,
            indices=child_indices,
            level=level + 1,
            config=config,
            token_paths=token_paths,
        )


def _order_indices_by_kmeans(
    embeddings: np.ndarray,
    indices: np.ndarray,
    num_children: int,
    config: HierarchicalKMeansConfig,
) -> np.ndarray:
    local_embeddings = embeddings[indices]
    model = KMeans(
        n_clusters=num_children,
        random_state=config.random_state,
        n_init=config.n_init,
        max_iter=config.max_iter,
    )
    labels = model.fit_predict(local_embeddings)
    centers = model.cluster_centers_
    label_order = {
        label: order
        for order, label in enumerate(
            sorted(range(num_children), key=lambda label: tuple(centers[label].tolist()))
        )
    }
    distances = np.linalg.norm(local_embeddings - centers[labels], axis=1)
    ordered_positions = sorted(
        range(len(indices)),
        key=lambda position: (
            label_order[int(labels[position])],
            float(distances[position]),
            int(indices[position]),
        ),
    )
    return indices[ordered_positions]


def _validate_inputs(
    item_ids: tuple[int, ...],
    embeddings: np.ndarray,
    config: HierarchicalKMeansConfig,
) -> None:
    if not item_ids:
        msg = "item_ids는 비어 있을 수 없습니다."
        raise ValueError(msg)
    if len(set(item_ids)) != len(item_ids):
        msg = "item_ids에 중복이 있습니다."
        raise ValueError(msg)
    if embeddings.ndim != 2:
        msg = "embeddings는 2차원 행렬이어야 합니다."
        raise ValueError(msg)
    if embeddings.shape[0] != len(item_ids):
        msg = "embeddings 행 수와 item_ids 길이가 다릅니다."
        raise ValueError(msg)
    if config.capacity < len(item_ids):
        msg = (
            "Semantic ID capacity가 item 수보다 작습니다: "
            f"capacity={config.capacity}, items={len(item_ids)}"
        )
        raise ValueError(msg)


def _validate_node_capacity(
    num_items: int,
    branching_factor: int,
    remaining_depth: int,
) -> None:
    capacity = branching_factor**remaining_depth
    if num_items > capacity:
        msg = f"node capacity를 초과했습니다: capacity={capacity}, items={num_items}"
        raise ValueError(msg)

"""Item interaction embedding 생성."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse  # type: ignore[import-untyped]
from sklearn.decomposition import TruncatedSVD  # type: ignore[import-untyped]
from sklearn.preprocessing import normalize  # type: ignore[import-untyped]


@dataclass(frozen=True)
class ItemEmbeddingConfig:
    """interaction embedding 생성 설정."""

    n_components: int = 32
    max_history_items: int = 50
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_components < 1:
            msg = "n_components는 1 이상이어야 합니다."
            raise ValueError(msg)
        if self.max_history_items < 1:
            msg = "max_history_items는 1 이상이어야 합니다."
            raise ValueError(msg)


@dataclass(frozen=True)
class ItemEmbeddings:
    """item_id와 dense embedding 행렬."""

    item_ids: tuple[int, ...]
    embeddings: np.ndarray
    num_examples: int
    num_context_edges: int

    @property
    def num_items(self) -> int:
        return len(self.item_ids)

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])


def build_item_interaction_embeddings(
    train_frame: pd.DataFrame,
    config: ItemEmbeddingConfig,
) -> ItemEmbeddings:
    """train split의 history-target co-occurrence로 item embedding을 만든다."""
    _validate_train_frame(train_frame)
    records = cast(
        list[dict[str, Any]],
        train_frame[["history_item_ids", "target_item_id"]].to_dict("records"),
    )
    item_ids = _collect_item_ids(records)
    if not item_ids:
        msg = "embedding을 만들 item이 없습니다."
        raise ValueError(msg)

    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    target_counts = np.zeros(len(item_ids), dtype=np.float64)
    source_counts = np.zeros(len(item_ids), dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for record in records:
        target_item_id = int(record["target_item_id"])
        target_index = item_index[target_item_id]
        target_counts[target_index] += 1.0

        history_item_ids = _coerce_item_ids(record["history_item_ids"])[-config.max_history_items :]
        for source_item_id in history_item_ids:
            source_index = item_index[source_item_id]
            source_counts[source_index] += 1.0
            if source_item_id == target_item_id:
                continue
            rows.append(target_index)
            cols.append(source_index)
            data.append(1.0)

    context_matrix = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(item_ids), len(item_ids)),
        dtype=np.float64,
    ).tocsr()
    dense_context = _reduce_sparse_context_matrix(context_matrix, config)
    dense_features = np.column_stack(
        [
            dense_context,
            _safe_log_scale(target_counts),
            _safe_log_scale(source_counts),
            _normalized_item_order(len(item_ids)),
        ]
    )
    embeddings = normalize(dense_features, norm="l2", axis=1)
    return ItemEmbeddings(
        item_ids=item_ids,
        embeddings=np.asarray(embeddings, dtype=np.float64),
        num_examples=len(train_frame),
        num_context_edges=len(data),
    )


def _reduce_sparse_context_matrix(
    context_matrix: sparse.csr_matrix,
    config: ItemEmbeddingConfig,
) -> np.ndarray:
    min_shape = min(context_matrix.shape)
    if context_matrix.nnz == 0 or min_shape <= 1:
        return np.zeros((context_matrix.shape[0], 1), dtype=np.float64)

    n_components = min(config.n_components, min_shape - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=config.random_state)
    return np.asarray(svd.fit_transform(context_matrix), dtype=np.float64)


def _collect_item_ids(records: Iterable[dict[str, Any]]) -> tuple[int, ...]:
    item_ids: set[int] = set()
    for record in records:
        item_ids.add(int(record["target_item_id"]))
        item_ids.update(_coerce_item_ids(record["history_item_ids"]))
    return tuple(sorted(item_ids))


def _coerce_item_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(item_id) for item_id in value]


def _safe_log_scale(values: np.ndarray) -> np.ndarray:
    scaled = np.asarray(np.log1p(values), dtype=np.float64)
    max_value = float(scaled.max())
    if max_value == 0.0:
        return scaled
    return np.asarray(scaled / max_value, dtype=np.float64)


def _normalized_item_order(num_items: int) -> np.ndarray:
    if num_items == 1:
        return np.zeros(1, dtype=np.float64)
    return np.linspace(0.0, 1.0, num_items, dtype=np.float64)


def _validate_train_frame(train_frame: pd.DataFrame) -> None:
    required_columns = {"history_item_ids", "target_item_id"}
    missing_columns = sorted(required_columns - set(train_frame.columns))
    if missing_columns:
        msg = f"train 데이터에 필요한 컬럼이 없습니다: {missing_columns}"
        raise ValueError(msg)

import numpy as np
import pandas as pd
import pytest

from recsys.semantic_id import ItemEmbeddingConfig, build_item_interaction_embeddings


def test_build_item_interaction_embeddings_includes_history_and_target_items() -> None:
    train_frame = pd.DataFrame(
        {
            "history_item_ids": [[1, 2], [2, 3], [4]],
            "target_item_id": [3, 4, 5],
        }
    )

    result = build_item_interaction_embeddings(
        train_frame,
        ItemEmbeddingConfig(n_components=2, max_history_items=10, random_state=42),
    )

    assert result.item_ids == (1, 2, 3, 4, 5)
    assert result.num_items == 5
    assert result.num_examples == 3
    assert result.num_context_edges == 5
    assert result.embeddings.shape[0] == 5
    assert np.isfinite(result.embeddings).all()


def test_build_item_interaction_embeddings_respects_max_history_items() -> None:
    train_frame = pd.DataFrame({"history_item_ids": [[1, 2, 3]], "target_item_id": [4]})

    result = build_item_interaction_embeddings(
        train_frame,
        ItemEmbeddingConfig(n_components=2, max_history_items=1, random_state=42),
    )

    assert result.num_context_edges == 1


def test_build_item_interaction_embeddings_requires_columns() -> None:
    with pytest.raises(ValueError, match="필요한 컬럼"):
        build_item_interaction_embeddings(
            pd.DataFrame({"target_item_id": [1]}),
            ItemEmbeddingConfig(),
        )

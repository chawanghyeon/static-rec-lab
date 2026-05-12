import numpy as np
import pytest

from recsys.semantic_id import (
    HierarchicalKMeansConfig,
    ItemEmbeddings,
    build_semantic_id_codec,
    build_semantic_id_mapping,
)


def test_build_semantic_id_mapping_assigns_fixed_length_unique_ids() -> None:
    item_ids = (10, 20, 30, 40)
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [10.0, 11.0],
        ],
        dtype=np.float64,
    )

    mapping = build_semantic_id_mapping(
        item_ids,
        embeddings,
        HierarchicalKMeansConfig(depth=2, branching_factor=2, random_state=42, n_init=1),
    )

    assert set(mapping) == set(item_ids)
    assert {len(semantic_id) for semantic_id in mapping.values()} == {2}
    assert len(set(mapping.values())) == len(item_ids)


def test_build_semantic_id_mapping_rejects_insufficient_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        build_semantic_id_mapping(
            item_ids=(1, 2, 3),
            embeddings=np.ones((3, 2), dtype=np.float64),
            config=HierarchicalKMeansConfig(depth=1, branching_factor=2),
        )


def test_build_semantic_id_codec_returns_valid_codec() -> None:
    item_embeddings = ItemEmbeddings(
        item_ids=(1, 2),
        embeddings=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        num_examples=2,
        num_context_edges=1,
    )

    result = build_semantic_id_codec(
        item_embeddings,
        HierarchicalKMeansConfig(depth=1, branching_factor=2, random_state=42, n_init=1),
    )

    assert result.codec.num_items == 2
    assert result.codec.semantic_id_length == 1
    assert result.embedding_dim == 2
    assert result.num_context_edges == 1

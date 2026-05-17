from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from recsys.decoding import (
    StaticDecodingIndex,
    build_static_decoding_random_model,
    static_decoding_constrained_beam_search,
    static_decoding_generate_and_apply_logprobs_mask,
    static_decoding_generate_and_apply_logprobs_mask_jax,
    static_decoding_sparse_transition_jax,
    static_decoding_sparse_transition_torch,
    validate_static_decoding_index_matches_codec,
)
from recsys.decoding.naive_trie import SemanticIdTrie
from recsys.semantic_id import SemanticIdCodec
from scripts.build_static_decoding_index import resolve_dense_lookup_layers


def test_static_decoding_index_matches_naive_trie_prefixes() -> None:
    semantic_ids = [(1, 2, 3), (1, 2, 4), (1, 5, 6), (7, 8, 9)]
    trie = SemanticIdTrie(semantic_ids)
    index = StaticDecodingIndex.from_semantic_ids(
        semantic_ids,
        vocab_size=10,
        dense_lookup_layers=2,
    )

    assert index.start_mask.shape == (10,)
    assert index.dense_mask.shape == (10, 10)
    assert index.dense_states.shape == (10, 10)
    assert index.allowed_next_tokens([]) == trie.allowed_next_tokens([])

    prefixes = [[], [1], [1, 2], [1, 5], [7], [7, 8], [9], [1, 9], [1, 2, 3]]
    for prefix in prefixes:
        assert index.allowed_next_tokens(prefix) == trie.allowed_next_tokens(prefix)

    assert index.contains([1, 2, 3])
    assert index.contains([7, 8, 9])
    assert not index.contains([1, 2, 9])


def test_static_decoding_sparse_kernel_matches_index_allowed_tokens() -> None:
    index = StaticDecodingIndex.from_semantic_ids(
        [(1, 2, 3), (1, 2, 4), (7, 8, 9)],
        vocab_size=10,
        dense_lookup_layers=2,
    )
    device = torch.device("cpu")
    torch_index = index.to_torch(device)
    states = torch.tensor(
        [
            index.state_for_prefix([1, 2]),
            index.state_for_prefix([7, 8]),
        ],
        dtype=torch.long,
        device=device,
    )
    logprobs = torch.zeros((2, index.vocab_size), dtype=torch.float32, device=device)

    candidate_logprobs, candidate_tokens, candidate_states = (
        static_decoding_generate_and_apply_logprobs_mask(
            flat_logprobs=logprobs,
            flat_states=states,
            index=torch_index,
            prefix_length=2,
        )
    )

    assert candidate_logprobs.shape == candidate_tokens.shape == candidate_states.shape
    finite_tokens = [
        tuple(
            int(token) for token in candidate_tokens[row][torch.isfinite(candidate_logprobs[row])]
        )
        for row in range(candidate_tokens.shape[0])
    ]
    assert finite_tokens == [
        index.allowed_next_tokens([1, 2]),
        index.allowed_next_tokens([7, 8]),
    ]


def test_static_decoding_jax_sparse_kernel_matches_index_allowed_tokens() -> None:
    index = StaticDecodingIndex.from_semantic_ids(
        [(1, 2, 3), (1, 2, 4), (7, 8, 9)],
        vocab_size=10,
        dense_lookup_layers=2,
    )
    states = jnp.asarray(
        [
            index.state_for_prefix([1, 2]),
            index.state_for_prefix([7, 8]),
        ],
        dtype=jnp.int32,
    )
    logprobs = jnp.zeros((2, index.vocab_size), dtype=jnp.float32)

    candidate_logprobs, candidate_tokens, candidate_states = (
        static_decoding_generate_and_apply_logprobs_mask_jax(
            flat_logprobs=logprobs,
            flat_states=states,
            index=index,
            prefix_length=2,
        )
    )

    candidate_logprobs_array = np.asarray(candidate_logprobs)
    candidate_tokens_array = np.asarray(candidate_tokens)
    candidate_states_array = np.asarray(candidate_states)
    assert (
        candidate_logprobs_array.shape
        == candidate_tokens_array.shape
        == candidate_states_array.shape
    )
    finite_tokens = [
        tuple(
            int(token)
            for token in candidate_tokens_array[row][np.isfinite(candidate_logprobs_array[row])]
        )
        for row in range(candidate_tokens_array.shape[0])
    ]
    assert finite_tokens == [
        index.allowed_next_tokens([1, 2]),
        index.allowed_next_tokens([7, 8]),
    ]


def test_static_decoding_index_supports_dense_lookup_one() -> None:
    semantic_ids = [(3, 4), (3, 5), (6, 7)]
    trie = SemanticIdTrie(semantic_ids)
    index = StaticDecodingIndex.from_semantic_ids(
        semantic_ids,
        vocab_size=8,
        dense_lookup_layers=1,
    )

    for prefix in ([], [3], [6], [3, 4], [1]):
        assert index.allowed_next_tokens(prefix) == trie.allowed_next_tokens(prefix)

    assert index.contains([3, 4])
    assert index.contains([6, 7])
    assert not index.contains([3, 7])


def test_static_decoding_constrained_beam_search_only_returns_valid_semantic_ids() -> None:
    index = StaticDecodingIndex.from_semantic_ids(
        [(1, 2, 3), (1, 2, 4), (5, 6, 7)],
        vocab_size=10,
        dense_lookup_layers=2,
    )

    def logits_provider(
        prefixes: tuple[tuple[int, ...], ...],
        _states: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        logits = torch.full((len(prefixes), index.vocab_size), -10.0)
        logits[:, 9] = 100.0
        if step == 0:
            logits[:, 1] = 2.0
            logits[:, 5] = 1.0
            return logits
        for row_index, prefix in enumerate(prefixes):
            if prefix == (1,):
                logits[row_index, 2] = 3.0
            elif prefix == (5,):
                logits[row_index, 6] = 1.0
            elif prefix == (1, 2):
                logits[row_index, 4] = 4.0
                logits[row_index, 3] = 2.0
            elif prefix == (5, 6):
                logits[row_index, 7] = 1.0
        return logits

    results = static_decoding_constrained_beam_search(
        index=index,
        beam_size=3,
        logits_provider=logits_provider,
        device=torch.device("cpu"),
    )

    assert results[0].semantic_id == (1, 2, 4)
    assert {result.semantic_id for result in results} <= {(1, 2, 3), (1, 2, 4), (5, 6, 7)}
    assert all(index.contains(result.semantic_id) for result in results)


def test_static_decoding_constrained_beam_search_supports_beam_larger_than_vocab() -> None:
    index = StaticDecodingIndex.from_semantic_ids(
        [(0, 1, 2), (0, 2, 3), (3, 0, 1)],
        vocab_size=4,
        dense_lookup_layers=2,
    )

    def logits_provider(
        prefixes: tuple[tuple[int, ...], ...],
        _states: torch.Tensor,
        _step: int,
    ) -> torch.Tensor:
        logits = torch.zeros((len(prefixes), index.vocab_size))
        logits[:, 3] = 5.0
        return logits

    results = static_decoding_constrained_beam_search(
        index=index,
        beam_size=8,
        logits_provider=logits_provider,
        device=torch.device("cpu"),
    )

    assert results
    assert all(index.contains(result.semantic_id) for result in results)


def test_static_decoding_sparse_transition_torch_harness_returns_valid_semantic_ids() -> None:
    index = StaticDecodingIndex.from_semantic_ids(
        [(1, 2, 3), (1, 2, 4), (5, 6, 7), (6, 0, 1)],
        vocab_size=8,
        dense_lookup_layers=2,
    )
    model = _FixedLogitModel(vocab_size=index.vocab_size)

    token_buffer = static_decoding_sparse_transition_torch(
        model=model,
        index=index,
        batch_size=2,
        beam_size=2,
        tokens_per_beam=2,
        start_token=0,
        device=torch.device("cpu"),
    )

    assert token_buffer.shape == (2, 2, 3)
    for semantic_id in token_buffer.reshape(-1, token_buffer.shape[-1]).tolist():
        assert index.contains(semantic_id)


def test_static_decoding_sparse_transition_jax_harness_returns_valid_semantic_ids() -> None:
    index = StaticDecodingIndex.from_semantic_ids(
        [(1, 2, 3), (1, 2, 4), (5, 6, 7), (6, 0, 1)],
        vocab_size=8,
        dense_lookup_layers=2,
    )

    token_buffer = static_decoding_sparse_transition_jax(
        index=index,
        batch_size=1,
        beam_size=1,
        tokens_per_beam=1,
        start_token=0,
        random_seed=0,
    )

    token_buffer_array = np.asarray(token_buffer)
    assert token_buffer_array.shape == (1, 1, 3)
    for semantic_id in token_buffer_array.reshape(-1, token_buffer_array.shape[-1]).tolist():
        assert index.contains(semantic_id)


def test_build_static_decoding_random_model_uses_static_decoding_shape_contract() -> None:
    model = build_static_decoding_random_model(vocab_size=8, device=torch.device("cpu"))

    logits = model(torch.zeros((3, 1), dtype=torch.long))

    assert logits.shape == (3, 1, 8)


def test_resolve_dense_lookup_layers_uses_static_decoding_auto_rule() -> None:
    codec = SemanticIdCodec({10: [1, 2, 3], 20: [1, 4, 5]})

    assert resolve_dense_lookup_layers(codec=codec, value="auto") == 2
    assert resolve_dense_lookup_layers(codec=codec, value="1") == 1


def test_validate_static_decoding_index_matches_codec_detects_missing_ids() -> None:
    codec = SemanticIdCodec({10: [1, 2], 20: [3, 4]})
    index = StaticDecodingIndex.from_semantic_ids(
        [(1, 2)],
        vocab_size=5,
        dense_lookup_layers=1,
    )

    with pytest.raises(ValueError, match="누락"):
        validate_static_decoding_index_matches_codec(index=index, codec=codec)


def test_static_decoding_index_round_trips_npz(tmp_path: Path) -> None:
    index = StaticDecodingIndex.from_semantic_ids(
        [(1, 2, 3), (4, 5, 6)],
        vocab_size=8,
        dense_lookup_layers=2,
    )

    path = index.save_npz(tmp_path / "static_decoding.npz")
    restored = StaticDecodingIndex.load_npz(path)

    assert restored.layer_max_branches == index.layer_max_branches
    assert np.array_equal(restored.packed_csr, index.packed_csr)
    assert np.array_equal(restored.csr_indptr, index.csr_indptr)
    assert restored.allowed_next_tokens([1, 2]) == (3,)


def test_static_decoding_index_rejects_dense_lookup_equal_to_depth() -> None:
    with pytest.raises(ValueError, match="dense_lookup_layers"):
        StaticDecodingIndex.from_semantic_ids(
            [(1, 2)],
            vocab_size=4,
            dense_lookup_layers=2,
        )


class _FixedLogitModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        logits = torch.full((batch_size, 1, self.vocab_size), -10.0)
        logits[:, :, 7] = 100.0
        for token in range(self.vocab_size):
            logits[:, :, token] = float(token)
        return logits

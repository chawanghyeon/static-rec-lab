from typing import Any

import numpy as np
import pytest

from recsys.decoding import INVALID_STATE, SemanticIdTrie, StaticTransitionMatrixDecoder
from recsys.semantic_id import SemanticIdCodec


def test_static_transition_matrix_flattens_trie_states() -> None:
    trie = SemanticIdTrie([(1, 2, 3), (1, 2, 4), (5, 0)])

    decoder = StaticTransitionMatrixDecoder.from_trie(trie)

    assert decoder.num_states == trie.num_nodes
    assert decoder.root_state == trie.root_state
    assert decoder.vocab_size == 6
    assert decoder.transition_matrix.shape == (trie.num_nodes, 6)
    assert decoder.transition_matrix.nnz == sum(1 for _ in trie.iter_transitions())


def test_static_transition_matrix_allowed_tokens_match_naive_trie() -> None:
    semantic_ids = [(1, 2, 3), (1, 2, 4), (1, 5, 6), (7, 8)]
    trie = SemanticIdTrie(semantic_ids)
    decoder = StaticTransitionMatrixDecoder.from_trie(trie)

    prefixes: list[list[Any]] = [[], [1], [1, 2], [1, 5], [7], [9], [1, -1], ["1"]]
    for prefix in prefixes:
        assert decoder.allowed_next_tokens(prefix) == trie.allowed_next_tokens(prefix)


def test_static_transition_matrix_batch_next_state_updates_states() -> None:
    decoder = StaticTransitionMatrixDecoder.from_semantic_ids([(1, 2, 3), (1, 4, 5), (6, 7, 8)])
    state_after_1 = decoder.state_for_prefix([1])

    assert state_after_1 is not None
    states = [decoder.root_state, decoder.root_state, state_after_1, state_after_1, INVALID_STATE]
    tokens = [1, 6, 2, 9, 1]

    next_states = decoder.batch_next_state(states, tokens)

    assert next_states[0] == decoder.state_for_prefix([1])
    assert next_states[1] == decoder.state_for_prefix([6])
    assert next_states[2] == decoder.state_for_prefix([1, 2])
    assert next_states[3] == INVALID_STATE
    assert next_states[4] == INVALID_STATE


def test_static_transition_matrix_batch_next_state_accepts_numpy_arrays() -> None:
    decoder = StaticTransitionMatrixDecoder.from_semantic_ids([(1, 2), (3, 4)])

    states = np.array([[decoder.root_state, decoder.root_state]])
    tokens = np.array([[1, 9]])

    next_states = decoder.batch_next_state(states, tokens)

    assert next_states.shape == (1, 2)
    assert next_states[0, 0] == decoder.state_for_prefix([1])
    assert next_states[0, 1] == INVALID_STATE


def test_static_transition_matrix_next_state_accepts_numpy_integer_state() -> None:
    decoder = StaticTransitionMatrixDecoder.from_semantic_ids([(1, 2)])

    next_state = decoder.next_state(np.int64(decoder.root_state), np.int64(1))

    assert next_state == decoder.state_for_prefix([1])


def test_static_transition_matrix_allowed_mask_matches_naive_trie() -> None:
    semantic_ids = [(1, 2, 3), (1, 2, 4), (5, 6, 7)]
    trie = SemanticIdTrie(semantic_ids)
    decoder = StaticTransitionMatrixDecoder.from_trie(trie, vocab_size=8)
    prefixes = [[], [1], [1, 2], [5], [9]]
    states = [
        decoder.state_for_prefix(prefix)
        if decoder.state_for_prefix(prefix) is not None
        else INVALID_STATE
        for prefix in prefixes
    ]

    mask = decoder.allowed_token_mask(states)

    assert mask.shape == (len(prefixes), 8)
    for row_index, prefix in enumerate(prefixes):
        allowed_from_mask = tuple(np.flatnonzero(mask[row_index]).tolist())
        assert allowed_from_mask == trie.allowed_next_tokens(prefix)


def test_static_transition_matrix_terminal_states_match_trie() -> None:
    semantic_ids = [(1, 2), (1, 2, 3), (4,)]
    trie = SemanticIdTrie(semantic_ids)
    decoder = StaticTransitionMatrixDecoder.from_trie(trie)

    for prefix in ([1, 2], [1, 2, 3], [4]):
        state = decoder.state_for_prefix(prefix)
        assert state is not None
        assert decoder.is_terminal_state(state) == trie.is_terminal_prefix(prefix)


def test_static_transition_matrix_can_be_built_from_codec() -> None:
    codec = SemanticIdCodec({10: [1, 2, 3], 20: [1, 2, 4]})

    decoder = StaticTransitionMatrixDecoder.from_codec(codec)

    assert decoder.allowed_next_tokens([1, 2]) == (3, 4)


def test_static_transition_matrix_rejects_small_vocab_size() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        StaticTransitionMatrixDecoder.from_semantic_ids([(3, 4)], vocab_size=4)


def test_static_transition_matrix_rejects_bad_batch_shapes() -> None:
    decoder = StaticTransitionMatrixDecoder.from_semantic_ids([(1, 2)])

    with pytest.raises(ValueError, match="shape"):
        decoder.batch_next_state([decoder.root_state], [1, 2])

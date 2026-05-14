import pytest

from recsys.decoding import SemanticIdTrie
from recsys.semantic_id import SemanticIdCodec


def test_semantic_id_trie_inserts_sequences_and_returns_allowed_next_tokens() -> None:
    trie = SemanticIdTrie([(1, 2, 3), (1, 2, 4), (1, 5, 6), (7, 8)])

    assert trie.num_sequences == 4
    assert trie.max_depth == 3
    assert trie.allowed_next_tokens([]) == (1, 7)
    assert trie.allowed_next_tokens([1]) == (2, 5)
    assert trie.allowed_next_tokens([1, 2]) == (3, 4)
    assert trie.allowed_next_tokens([7]) == (8,)


def test_semantic_id_trie_returns_empty_candidates_for_invalid_prefix() -> None:
    trie = SemanticIdTrie([(1, 2, 3), (1, 5, 6)])

    assert trie.allowed_next_tokens([9]) == ()
    assert trie.allowed_next_tokens([1, 9]) == ()
    assert trie.allowed_next_tokens([1, -1]) == ()
    assert trie.allowed_next_tokens(["1"]) == ()
    assert not trie.is_valid_prefix([1, 9])


def test_semantic_id_trie_handles_multiple_depths_and_terminal_prefixes() -> None:
    trie = SemanticIdTrie([(1, 2), (1, 2, 3), (4,)])

    assert trie.contains([1, 2])
    assert trie.contains([1, 2, 3])
    assert trie.contains([4])
    assert trie.is_terminal_prefix([1, 2])
    assert trie.allowed_next_tokens([1, 2]) == (3,)
    assert trie.allowed_next_tokens([4]) == ()


def test_semantic_id_trie_deduplicates_repeated_sequences() -> None:
    trie = SemanticIdTrie([(1, 2), (1, 2)])

    assert trie.num_sequences == 1
    assert trie.contains([1, 2])


def test_semantic_id_trie_supports_state_transitions() -> None:
    trie = SemanticIdTrie([(1, 2, 3), (1, 4, 5)])

    state = trie.next_state(trie.root_state, 1)
    assert state is not None
    resolved_state = state
    assert trie.allowed_next_tokens_for_state(resolved_state) == (2, 4)

    next_state = trie.next_state(resolved_state, 9)
    assert next_state is None


def test_semantic_id_trie_can_be_built_from_codec() -> None:
    codec = SemanticIdCodec({10: [1, 2, 3], 20: [1, 2, 4]})

    trie = SemanticIdTrie.from_codec(codec)

    assert trie.allowed_next_tokens([1, 2]) == (3, 4)
    assert trie.contains(codec.encode_item(10))


@pytest.mark.parametrize("bad_sequence", [[], ["1"], [1, -1], [True]])
def test_semantic_id_trie_rejects_invalid_insert_sequences(
    bad_sequence: list[object],
) -> None:
    trie = SemanticIdTrie()

    with pytest.raises(ValueError):
        trie.insert(bad_sequence)


@pytest.mark.parametrize("bad_state", [-1, 999, True, "0"])
def test_semantic_id_trie_rejects_invalid_states(bad_state: object) -> None:
    trie = SemanticIdTrie([(1, 2)])

    with pytest.raises(ValueError):
        trie.allowed_next_tokens_for_state(bad_state)  # type: ignore[arg-type]

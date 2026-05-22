"""Naive trie 기반 constrained decoder."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from recsys.decoding.tokens import (
    normalize_token as _normalize_token,
)
from recsys.decoding.tokens import (
    normalize_tokens as _normalize_tokens,
)
from recsys.decoding.tokens import (
    try_normalize_tokens as _try_normalize_tokens,
)
from recsys.semantic_id import SemanticIdCodec


@dataclass
class _TrieNode:
    children: dict[int, int] = field(default_factory=dict)
    is_terminal: bool = False


class SemanticIdTrie:
    """유효한 Semantic ID token sequence만 따라갈 수 있는 prefix trie."""

    def __init__(self, semantic_ids: Iterable[Iterable[Any]] = ()) -> None:
        self._nodes: list[_TrieNode] = [_TrieNode()]
        self._num_sequences = 0
        self._max_depth = 0

        for semantic_id in semantic_ids:
            self.insert(semantic_id)

    @classmethod
    def from_codec(cls, codec: SemanticIdCodec) -> SemanticIdTrie:
        """SemanticIdCodec의 mapping으로 trie를 만든다."""
        return cls(codec.item_to_semantic_id.values())

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_sequences(self) -> int:
        return self._num_sequences

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @property
    def root_state(self) -> int:
        return 0

    def insert(self, semantic_id: Iterable[Any]) -> None:
        """유효한 Semantic ID sequence를 trie에 삽입한다."""
        tokens = _normalize_tokens(semantic_id, allow_empty=False)
        node_index = self.root_state

        for token in tokens:
            node = self._nodes[node_index]
            if token not in node.children:
                node.children[token] = len(self._nodes)
                self._nodes.append(_TrieNode())
            node_index = node.children[token]

        terminal_node = self._nodes[node_index]
        if not terminal_node.is_terminal:
            terminal_node.is_terminal = True
            self._num_sequences += 1
            self._max_depth = max(self._max_depth, len(tokens))

    def allowed_next_tokens(self, prefix: Iterable[Any]) -> tuple[int, ...]:
        """prefix 뒤에 올 수 있는 token 목록을 정렬된 tuple로 반환한다."""
        node_index = self.state_for_prefix(prefix)
        if node_index is None:
            return ()
        return tuple(sorted(self._nodes[node_index].children))

    def state_for_prefix(self, prefix: Iterable[Any]) -> int | None:
        """prefix가 가리키는 trie state를 반환한다. 유효하지 않으면 None이다."""
        tokens = _try_normalize_tokens(prefix, allow_empty=True)
        if tokens is None:
            return None

        node_index = self.root_state

        for token in tokens:
            next_node_index = self._nodes[node_index].children.get(token)
            if next_node_index is None:
                return None
            node_index = next_node_index
        return node_index

    def next_state(self, state: int, token: Any) -> int | None:
        """현재 state에서 token을 소비했을 때의 다음 state를 반환한다."""
        self._validate_state(state)
        normalized_token = _normalize_token(token)
        return self._nodes[state].children.get(normalized_token)

    def allowed_next_tokens_for_state(self, state: int) -> tuple[int, ...]:
        """state 기준으로 다음에 허용되는 token 목록을 반환한다."""
        self._validate_state(state)
        return tuple(sorted(self._nodes[state].children))

    def iter_transitions(self) -> Iterator[tuple[int, int, int]]:
        """(source_state, token, target_state) transition을 state 순서로 순회한다."""
        for source_state, node in enumerate(self._nodes):
            for token, target_state in sorted(node.children.items()):
                yield source_state, token, target_state

    def is_terminal_state(self, state: int) -> bool:
        """state가 완성된 Semantic ID를 가리키는지 확인한다."""
        self._validate_state(state)
        return self._nodes[state].is_terminal

    def contains(self, semantic_id: Iterable[Any]) -> bool:
        """sequence가 완전한 Semantic ID로 삽입되어 있는지 확인한다."""
        node_index = self.state_for_prefix(semantic_id)
        return node_index is not None and self._nodes[node_index].is_terminal

    def is_valid_prefix(self, prefix: Iterable[Any]) -> bool:
        """prefix가 trie에 존재하는 경로인지 확인한다."""
        return self.state_for_prefix(prefix) is not None

    def is_terminal_prefix(self, prefix: Iterable[Any]) -> bool:
        """prefix 자체가 완성된 Semantic ID인지 확인한다."""
        node_index = self.state_for_prefix(prefix)
        return node_index is not None and self._nodes[node_index].is_terminal

    def _validate_state(self, state: int) -> None:
        if isinstance(state, bool) or not isinstance(state, int):
            msg = "state는 정수여야 합니다."
            raise ValueError(msg)
        if state < 0 or state >= len(self._nodes):
            msg = f"알 수 없는 trie state입니다: {state}"
            raise ValueError(msg)

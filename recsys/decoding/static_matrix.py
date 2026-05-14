"""STATIC-style sparse transition matrix decoder."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse  # type: ignore[import-untyped]

from recsys.decoding.naive_trie import SemanticIdTrie
from recsys.semantic_id import SemanticIdCodec

INVALID_STATE = -1


@dataclass(frozen=True)
class StaticTransitionMatrixDecoder:
    """Trie transition을 CSR sparse matrix로 표현한 constrained decoder."""

    transition_matrix: sparse.csr_matrix
    terminal_state_mask: np.ndarray
    root_state: int = 0

    @classmethod
    def from_semantic_ids(
        cls,
        semantic_ids: Iterable[Iterable[Any]],
        *,
        vocab_size: int | None = None,
    ) -> StaticTransitionMatrixDecoder:
        """Semantic ID sequence 목록으로 sparse transition decoder를 만든다."""
        return cls.from_trie(SemanticIdTrie(semantic_ids), vocab_size=vocab_size)

    @classmethod
    def from_codec(
        cls,
        codec: SemanticIdCodec,
        *,
        vocab_size: int | None = None,
    ) -> StaticTransitionMatrixDecoder:
        """SemanticIdCodec의 mapping으로 sparse transition decoder를 만든다."""
        return cls.from_trie(SemanticIdTrie.from_codec(codec), vocab_size=vocab_size)

    @classmethod
    def from_trie(
        cls,
        trie: SemanticIdTrie,
        *,
        vocab_size: int | None = None,
    ) -> StaticTransitionMatrixDecoder:
        """Naive trie의 integer state transition을 CSR sparse matrix로 flatten한다."""
        transitions = tuple(trie.iter_transitions())
        max_token = max((token for _, token, _ in transitions), default=-1)
        resolved_vocab_size = max_token + 1 if vocab_size is None else vocab_size
        if resolved_vocab_size < 1:
            resolved_vocab_size = 1
        if max_token >= resolved_vocab_size:
            msg = (
                "vocab_size가 transition token을 담기에 작습니다: "
                f"vocab_size={resolved_vocab_size}, max_token={max_token}"
            )
            raise ValueError(msg)

        source_states = [source_state for source_state, _, _ in transitions]
        token_ids = [token for _, token, _ in transitions]
        # CSR zero는 missing transition을 의미하므로 next_state는 1을 더해서 저장한다.
        encoded_next_states = [target_state + 1 for _, _, target_state in transitions]
        transition_matrix = sparse.csr_matrix(
            (encoded_next_states, (source_states, token_ids)),
            shape=(trie.num_nodes, resolved_vocab_size),
            dtype=np.int64,
        )
        terminal_state_mask = np.array(
            [trie.is_terminal_state(state) for state in range(trie.num_nodes)],
            dtype=bool,
        )
        return cls(
            transition_matrix=transition_matrix,
            terminal_state_mask=terminal_state_mask,
            root_state=trie.root_state,
        )

    @property
    def num_states(self) -> int:
        return int(self.transition_matrix.shape[0])

    @property
    def vocab_size(self) -> int:
        return int(self.transition_matrix.shape[1])

    def allowed_next_tokens(self, prefix: Iterable[Any]) -> tuple[int, ...]:
        """prefix 뒤에 올 수 있는 token 목록을 반환한다."""
        state = self.state_for_prefix(prefix)
        if state is None:
            return ()
        return self.allowed_next_tokens_for_state(state)

    def state_for_prefix(self, prefix: Iterable[Any]) -> int | None:
        """prefix를 소비한 뒤의 state를 반환한다. 유효하지 않으면 None이다."""
        tokens = _try_normalize_tokens(prefix, allow_empty=True)
        if tokens is None:
            return None

        state = self.root_state
        for token in tokens:
            next_state = self.next_state(state, token)
            if next_state is None:
                return None
            state = next_state
        return state

    def next_state(self, state: Any, token: Any) -> int | None:
        """단일 state에서 token을 소비한 다음 state를 반환한다."""
        state = self._validate_state(state)
        normalized_token = _normalize_token(token)
        if normalized_token >= self.vocab_size:
            return None

        encoded_next_state = int(self.transition_matrix[state, normalized_token])
        if encoded_next_state == 0:
            return None
        return encoded_next_state - 1

    def batch_next_state(
        self,
        states: Iterable[Any],
        tokens: Iterable[Any],
    ) -> np.ndarray:
        """batch state와 token을 한 번에 업데이트한다. invalid transition은 -1이다."""
        state_array = _coerce_int_array(states)
        token_array = _coerce_int_array(tokens)
        if state_array.shape != token_array.shape:
            msg = (
                "states와 tokens shape가 같아야 합니다: "
                f"states={state_array.shape}, tokens={token_array.shape}"
            )
            raise ValueError(msg)

        result = np.full(state_array.shape, INVALID_STATE, dtype=np.int64)
        valid_mask = (
            (state_array >= 0)
            & (state_array < self.num_states)
            & (token_array >= 0)
            & (token_array < self.vocab_size)
        )
        if not np.any(valid_mask):
            return result

        encoded_next_states = np.asarray(
            self.transition_matrix[state_array[valid_mask], token_array[valid_mask]]
        ).ravel()
        valid_transitions = encoded_next_states > 0
        updated_states = np.full(encoded_next_states.shape, INVALID_STATE, dtype=np.int64)
        updated_states[valid_transitions] = encoded_next_states[valid_transitions] - 1
        result[valid_mask] = updated_states
        return result

    def allowed_next_tokens_for_state(self, state: Any) -> tuple[int, ...]:
        """state 기준으로 다음에 허용되는 token 목록을 반환한다."""
        state = self._validate_state(state)
        row = self.transition_matrix.getrow(state)
        return tuple(int(token) for token in row.indices)

    def allowed_token_mask(self, states: Iterable[Any]) -> np.ndarray:
        """batch state별 allowed token boolean mask를 반환한다."""
        state_array = _coerce_int_array(states)
        flat_states = state_array.reshape(-1)
        mask = np.zeros((len(flat_states), self.vocab_size), dtype=bool)
        valid_mask = (flat_states >= 0) & (flat_states < self.num_states)
        if np.any(valid_mask):
            mask[valid_mask] = self.transition_matrix[flat_states[valid_mask]].toarray() > 0
        return mask.reshape((*state_array.shape, self.vocab_size))

    def is_terminal_state(self, state: Any) -> bool:
        """state가 완성된 Semantic ID를 가리키는지 확인한다."""
        state = self._validate_state(state)
        return bool(self.terminal_state_mask[state])

    def _validate_state(self, state: Any) -> int:
        if isinstance(state, (bool, np.bool_)) or not isinstance(state, (int, np.integer)):
            msg = "state는 정수여야 합니다."
            raise ValueError(msg)
        normalized_state = int(state)
        if normalized_state < 0 or normalized_state >= self.num_states:
            msg = f"알 수 없는 transition state입니다: {normalized_state}"
            raise ValueError(msg)
        return normalized_state


def _coerce_int_array(values: Iterable[Any]) -> np.ndarray:
    if isinstance(values, str):
        msg = "batch 값은 문자열이 아니라 정수 iterable이어야 합니다."
        raise ValueError(msg)

    array = values if isinstance(values, np.ndarray) else np.asarray(list(values))

    if array.size == 0:
        return array.astype(np.int64)
    if np.issubdtype(array.dtype, np.bool_):
        msg = "state/token batch 값은 bool이 아닌 정수여야 합니다."
        raise ValueError(msg)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int64, copy=False)

    if array.dtype == np.dtype("O"):
        normalized = [_normalize_token_like_value(value) for value in array.reshape(-1)]
        return np.asarray(normalized, dtype=np.int64).reshape(array.shape)

    msg = "state/token batch 값은 bool이 아닌 정수여야 합니다."
    raise ValueError(msg)


def _normalize_tokens(
    tokens: Iterable[Any],
    *,
    allow_empty: bool,
) -> tuple[int, ...]:
    if isinstance(tokens, str):
        msg = "token sequence는 문자열이 아니라 정수 iterable이어야 합니다."
        raise ValueError(msg)

    normalized = tuple(_normalize_token(token) for token in tokens)
    if not normalized and not allow_empty:
        msg = "Semantic ID sequence는 비어 있을 수 없습니다."
        raise ValueError(msg)
    return normalized


def _try_normalize_tokens(
    tokens: Iterable[Any],
    *,
    allow_empty: bool,
) -> tuple[int, ...] | None:
    try:
        return _normalize_tokens(tokens, allow_empty=allow_empty)
    except ValueError:
        return None


def _normalize_token(token: Any) -> int:
    if isinstance(token, (bool, np.bool_)) or not isinstance(token, (int, np.integer)):
        msg = f"token은 bool이 아닌 정수여야 합니다: {token!r}"
        raise ValueError(msg)
    if token < 0:
        msg = f"token은 0 이상이어야 합니다: {token}"
        raise ValueError(msg)
    return int(token)


def _normalize_token_like_value(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        msg = f"state/token batch 값은 bool이 아닌 정수여야 합니다: {value!r}"
        raise ValueError(msg)
    return int(value)

"""STATIC decoding index artifact 관리."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from recsys.decoding.static_dependency import (
    STATIC_DECODING_COMMIT,
    STATIC_DECODING_REPOSITORY,
    load_static_decoding_build_static_index,
)
from recsys.decoding.tokens import (
    normalize_tokens as _normalize_tokens,
)
from recsys.decoding.tokens import (
    try_normalize_tokens as _try_normalize_tokens,
)
from recsys.semantic_id import SemanticIdCodec


@dataclass(frozen=True)
class StaticDecodingTorchIndex:
    """STATIC sparse transition kernel에 전달할 Torch tensor 묶음."""

    packed_csr: torch.Tensor
    csr_indptr: torch.Tensor
    layer_max_branches: tuple[int, ...]
    start_mask: torch.Tensor
    dense_mask: torch.Tensor
    dense_states: torch.Tensor
    vocab_size: int
    semantic_id_depth: int
    dense_lookup_layers: int
    device: torch.device


@dataclass(frozen=True)
class StaticDecodingIndex:
    """`static_decoding.csr_utils.build_static_index`로 만든 STATIC index."""

    packed_csr: np.ndarray
    csr_indptr: np.ndarray
    layer_max_branches: tuple[int, ...]
    start_mask: np.ndarray
    dense_mask: np.ndarray
    dense_states: np.ndarray
    vocab_size: int
    semantic_id_depth: int
    dense_lookup_layers: int
    source_repository: str = STATIC_DECODING_REPOSITORY
    source_commit: str = STATIC_DECODING_COMMIT

    @classmethod
    def from_semantic_ids(
        cls,
        semantic_ids: Iterable[Iterable[Any]],
        *,
        vocab_size: int | None = None,
        dense_lookup_layers: int = 2,
    ) -> StaticDecodingIndex:
        """Semantic ID sequence 목록에서 STATIC index를 만든다."""
        semantic_id_array = _normalize_semantic_ids(semantic_ids)
        resolved_vocab_size = _resolve_vocab_size(semantic_id_array, vocab_size)
        _validate_dense_lookup_layers(
            dense_lookup_layers=dense_lookup_layers,
            semantic_id_depth=int(semantic_id_array.shape[1]),
        )
        if int(semantic_id_array.max()) >= resolved_vocab_size:
            msg = (
                "vocab_size가 Semantic ID token을 담기에 작습니다: "
                f"vocab_size={resolved_vocab_size}, max_token={int(semantic_id_array.max())}"
            )
            raise ValueError(msg)

        build_static_index = load_static_decoding_build_static_index()
        packed_csr, csr_indptr, layer_max_branches, start_mask, dense_mask, dense_states = (
            build_static_index(semantic_id_array, resolved_vocab_size, dense_lookup_layers)
        )
        return cls(
            packed_csr=np.asarray(packed_csr, dtype=np.int64),
            csr_indptr=np.asarray(csr_indptr, dtype=np.int64),
            layer_max_branches=tuple(int(value) for value in layer_max_branches),
            start_mask=np.asarray(start_mask, dtype=bool),
            dense_mask=np.asarray(dense_mask, dtype=bool),
            dense_states=np.asarray(dense_states, dtype=np.int64),
            vocab_size=resolved_vocab_size,
            semantic_id_depth=int(semantic_id_array.shape[1]),
            dense_lookup_layers=dense_lookup_layers,
        )

    @classmethod
    def from_codec(
        cls,
        codec: SemanticIdCodec,
        *,
        vocab_size: int | None = None,
        dense_lookup_layers: int = 2,
    ) -> StaticDecodingIndex:
        """Semantic ID codec에서 STATIC index를 만든다."""
        return cls.from_semantic_ids(
            codec.item_to_semantic_id.values(),
            vocab_size=vocab_size,
            dense_lookup_layers=dense_lookup_layers,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> StaticDecodingIndex:
        """`save_npz`로 저장한 STATIC index artifact를 로드한다."""
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                packed_csr=np.asarray(data["packed_csr"], dtype=np.int64),
                csr_indptr=np.asarray(data["csr_indptr"], dtype=np.int64),
                layer_max_branches=tuple(int(value) for value in data["layer_max_branches"]),
                start_mask=np.asarray(data["start_mask"], dtype=bool),
                dense_mask=np.asarray(data["dense_mask"], dtype=bool),
                dense_states=np.asarray(data["dense_states"], dtype=np.int64),
                vocab_size=int(data["vocab_size"]),
                semantic_id_depth=int(data["semantic_id_depth"]),
                dense_lookup_layers=int(data["dense_lookup_layers"]),
            )

    def save_npz(self, path: str | Path) -> Path:
        """STATIC index artifact를 압축 npz 파일로 저장한다."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            packed_csr=self.packed_csr,
            csr_indptr=self.csr_indptr,
            layer_max_branches=np.asarray(self.layer_max_branches, dtype=np.int64),
            start_mask=self.start_mask,
            dense_mask=self.dense_mask,
            dense_states=self.dense_states,
            vocab_size=np.asarray(self.vocab_size, dtype=np.int64),
            semantic_id_depth=np.asarray(self.semantic_id_depth, dtype=np.int64),
            dense_lookup_layers=np.asarray(self.dense_lookup_layers, dtype=np.int64),
        )
        return output_path

    def to_torch(self, device: torch.device) -> StaticDecodingTorchIndex:
        """STATIC index array를 Torch tensor로 옮긴다."""
        return StaticDecodingTorchIndex(
            packed_csr=torch.as_tensor(self.packed_csr, dtype=torch.long, device=device),
            csr_indptr=torch.as_tensor(self.csr_indptr, dtype=torch.long, device=device),
            layer_max_branches=self.layer_max_branches,
            start_mask=torch.as_tensor(self.start_mask, dtype=torch.bool, device=device),
            dense_mask=torch.as_tensor(self.dense_mask, dtype=torch.bool, device=device),
            dense_states=torch.as_tensor(self.dense_states, dtype=torch.long, device=device),
            vocab_size=self.vocab_size,
            semantic_id_depth=self.semantic_id_depth,
            dense_lookup_layers=self.dense_lookup_layers,
            device=device,
        )

    def allowed_next_tokens(self, prefix: Iterable[Any]) -> tuple[int, ...]:
        """STATIC index 기준으로 prefix 뒤에 올 수 있는 token을 반환한다."""
        tokens = _try_normalize_tokens(prefix, allow_empty=True)
        if tokens is None or len(tokens) >= self.semantic_id_depth:
            return ()
        if any(token >= self.vocab_size for token in tokens):
            return ()
        if not tokens:
            return _nonzero_tokens(self.start_mask)
        if len(tokens) < self.dense_lookup_layers:
            dense_slice = self.dense_mask[tokens]
            if dense_slice.ndim > 1:
                dense_slice = dense_slice.reshape(dense_slice.shape[0], -1).any(axis=1)
            return _nonzero_tokens(dense_slice)

        state = self.state_for_prefix(tokens)
        if state is None or state == 0:
            return ()
        return self._csr_allowed_tokens(state)

    def state_for_prefix(self, prefix: Iterable[Any]) -> int | None:
        """prefix가 도달하는 STATIC state를 반환한다."""
        tokens = _try_normalize_tokens(prefix, allow_empty=True)
        if tokens is None or not tokens:
            return None
        if any(token >= self.vocab_size for token in tokens):
            return None
        if len(tokens) < self.dense_lookup_layers:
            return None

        dense_prefix = tokens[: self.dense_lookup_layers]
        if not bool(self.dense_mask[dense_prefix]):
            return None
        state = int(self.dense_states[dense_prefix])

        for token in tokens[self.dense_lookup_layers :]:
            if state == 0:
                return None
            next_state = self._csr_next_state(state, token)
            if next_state is None:
                return None
            state = next_state
        return state

    def contains(self, semantic_id: Iterable[Any]) -> bool:
        """완성된 Semantic ID sequence가 STATIC index에 존재하는지 확인한다."""
        tokens = _try_normalize_tokens(semantic_id, allow_empty=False)
        if tokens is None or len(tokens) != self.semantic_id_depth:
            return False
        return self.state_for_prefix(tokens) == 0

    def _csr_allowed_tokens(self, state: int) -> tuple[int, ...]:
        start, end = self._csr_bounds(state)
        if end <= start:
            return ()
        row = self.packed_csr[start:end]
        tokens = row[:, 0]
        return tuple(sorted({int(token) for token in tokens if 0 <= int(token) < self.vocab_size}))

    def _csr_next_state(self, state: int, token: int) -> int | None:
        start, end = self._csr_bounds(state)
        if end <= start:
            return None
        row = self.packed_csr[start:end]
        matches = row[row[:, 0] == token]
        if len(matches) == 0:
            return None
        return int(matches[0, 1])

    def _csr_bounds(self, state: int) -> tuple[int, int]:
        if state < 0 or state + 1 >= len(self.csr_indptr):
            msg = f"알 수 없는 STATIC state입니다: {state}"
            raise ValueError(msg)
        return int(self.csr_indptr[state]), int(self.csr_indptr[state + 1])


def validate_static_decoding_index_matches_codec(
    *,
    index: StaticDecodingIndex,
    codec: SemanticIdCodec,
) -> None:
    """STATIC index가 codec의 모든 Semantic ID를 표현하는지 검증한다."""
    if index.semantic_id_depth != codec.semantic_id_length:
        msg = (
            "STATIC index depth와 Semantic ID codec 길이가 같아야 합니다: "
            f"index={index.semantic_id_depth}, codec={codec.semantic_id_length}"
        )
        raise ValueError(msg)

    max_token = max(
        (token for semantic_id in codec.item_to_semantic_id.values() for token in semantic_id),
        default=-1,
    )
    if max_token >= index.vocab_size:
        msg = (
            "STATIC index vocab_size가 Semantic ID token을 담기에 작습니다: "
            f"vocab_size={index.vocab_size}, max_token={max_token}"
        )
        raise ValueError(msg)

    missing_semantic_ids = [
        semantic_id
        for semantic_id in codec.item_to_semantic_id.values()
        if not index.contains(semantic_id)
    ]
    if missing_semantic_ids:
        msg = (
            "STATIC index에 codec Semantic ID가 누락되어 있습니다: "
            f"missing_count={len(missing_semantic_ids)}"
        )
        raise ValueError(msg)


def _normalize_semantic_ids(semantic_ids: Iterable[Iterable[Any]]) -> np.ndarray:
    normalized = tuple(sorted({_normalize_tokens(semantic_id) for semantic_id in semantic_ids}))
    if not normalized:
        msg = "Semantic ID 목록은 비어 있을 수 없습니다."
        raise ValueError(msg)
    depths = {len(semantic_id) for semantic_id in normalized}
    if len(depths) != 1:
        msg = f"모든 Semantic ID 길이가 같아야 합니다: {sorted(depths)}"
        raise ValueError(msg)
    return np.asarray(normalized, dtype=np.int64)


def _resolve_vocab_size(semantic_id_array: np.ndarray, vocab_size: int | None) -> int:
    inferred_vocab_size = int(semantic_id_array.max()) + 1
    if vocab_size is None:
        return max(1, inferred_vocab_size)
    if vocab_size < 1:
        msg = "vocab_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    return vocab_size


def _validate_dense_lookup_layers(*, dense_lookup_layers: int, semantic_id_depth: int) -> None:
    if dense_lookup_layers < 1:
        msg = "dense_lookup_layers는 1 이상이어야 합니다."
        raise ValueError(msg)
    if dense_lookup_layers >= semantic_id_depth:
        msg = (
            "static_decoding build_static_index는 dense_lookup_layers가 Semantic ID "
            "길이보다 작아야 합니다: "
            f"dense_lookup_layers={dense_lookup_layers}, depth={semantic_id_depth}"
        )
        raise ValueError(msg)


def _nonzero_tokens(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(token) for token in np.flatnonzero(mask))

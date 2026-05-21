"""Adapter around YouTube's STATIC implementation."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from recsys.decoding.beam_search import BeamSearchResult
from recsys.decoding.static_dependency import (
    STATIC_DECODING_COMMIT,
    STATIC_DECODING_REPOSITORY,
    load_static_decoding_build_static_index,
    load_static_decoding_jax,
    load_static_decoding_jax_numpy,
    load_static_decoding_jax_random_model_factory,
    load_static_decoding_jax_sparse_mask_kernel,
    load_static_decoding_jax_sparse_transition_harness,
    load_static_decoding_random_model_factory,
    load_static_decoding_sparse_mask_kernel,
    load_static_decoding_sparse_transition_harness,
)
from recsys.semantic_id import SemanticIdCodec

StaticDecodingBeamLogitProvider = Callable[
    [tuple[tuple[int, ...], ...], torch.Tensor, int],
    torch.Tensor,
]


@dataclass(frozen=True)
class StaticDecodingTorchIndex:
    """Torch tensors for the STATIC sparse transition kernel."""

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
    """STATIC index built by `static_decoding.csr_utils.build_static_index`."""

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
        """Build an STATIC index from Semantic ID sequences."""
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
        """Build an STATIC index from a Semantic ID codec."""
        return cls.from_semantic_ids(
            codec.item_to_semantic_id.values(),
            vocab_size=vocab_size,
            dense_lookup_layers=dense_lookup_layers,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> StaticDecodingIndex:
        """Load an STATIC index artifact saved by `save_npz`."""
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
        """Save the STATIC index artifact as a compressed npz file."""
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
        """Move STATIC index arrays to torch tensors."""
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
        """Return valid next tokens for a prefix according to the static_decoding index."""
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
        """Return the STATIC state reached by a prefix, if it is materialized."""
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
        """Return whether a full Semantic ID sequence is valid in the static_decoding index."""
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


def static_decoding_generate_and_apply_logprobs_mask(
    *,
    flat_logprobs: torch.Tensor,
    flat_states: torch.Tensor,
    index: StaticDecodingTorchIndex,
    prefix_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the static_decoding PyTorch sparse candidate extraction kernel."""
    if prefix_length < index.dense_lookup_layers:
        msg = (
            "static_decoding sparse kernel은 dense lookup 이후 prefix에서만 호출할 수 있습니다: "
            f"prefix_length={prefix_length}, dense_lookup_layers={index.dense_lookup_layers}"
        )
        raise ValueError(msg)
    if prefix_length < 0 or prefix_length >= index.semantic_id_depth:
        msg = f"prefix_length가 Semantic ID depth 범위를 벗어났습니다: {prefix_length}"
        raise ValueError(msg)

    limit = max(1, int(index.layer_max_branches[prefix_length]))
    kernel = load_static_decoding_sparse_mask_kernel()
    return kernel(
        flat_logprobs,
        flat_states,
        index.packed_csr,
        index.csr_indptr,
        limit,
        index.vocab_size,
        index.device,
    )


def static_decoding_generate_and_apply_logprobs_mask_jax(
    *,
    flat_logprobs: Any,
    flat_states: Any,
    index: StaticDecodingIndex,
    prefix_length: int,
) -> tuple[Any, Any, Any]:
    """Call the static_decoding JAX sparse candidate extraction kernel."""
    if prefix_length < index.dense_lookup_layers:
        msg = (
            "static_decoding JAX sparse kernel은 dense lookup 이후 prefix에서만 "
            "호출할 수 있습니다: "
            f"prefix_length={prefix_length}, dense_lookup_layers={index.dense_lookup_layers}"
        )
        raise ValueError(msg)
    if prefix_length < 0 or prefix_length >= index.semantic_id_depth:
        msg = f"prefix_length가 Semantic ID depth 범위를 벗어났습니다: {prefix_length}"
        raise ValueError(msg)

    jnp = load_static_decoding_jax_numpy()
    limit = max(1, int(index.layer_max_branches[prefix_length]))
    kernel = load_static_decoding_jax_sparse_mask_kernel()
    return kernel(
        flat_logprobs,
        flat_states,
        jnp.asarray(index.packed_csr),
        jnp.asarray(index.csr_indptr),
        limit,
        index.vocab_size,
    )


@torch.inference_mode()
def static_decoding_sparse_transition_torch(
    *,
    model: torch.nn.Module,
    index: StaticDecodingIndex,
    batch_size: int,
    beam_size: int,
    tokens_per_beam: int,
    start_token: int,
    device: torch.device,
    torch_index: StaticDecodingTorchIndex | None = None,
) -> torch.Tensor:
    """Call the static_decoding `sparse_transition_torch` decoding harness."""
    if batch_size < 1:
        msg = "batch_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    if beam_size < 1:
        msg = "beam_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    if tokens_per_beam < 1:
        msg = "tokens_per_beam은 1 이상이어야 합니다."
        raise ValueError(msg)
    if index.dense_lookup_layers not in (1, 2):
        msg = "static_decoding PyTorch harness는 dense_lookup_layers=1 또는 2만 지원합니다."
        raise ValueError(msg)

    resolved_torch_index = index.to_torch(device) if torch_index is None else torch_index
    if resolved_torch_index.device != device:
        msg = (
            "torch_index device가 요청한 device와 다릅니다: "
            f"torch_index={resolved_torch_index.device}, device={device}"
        )
        raise ValueError(msg)
    harness = load_static_decoding_sparse_transition_harness()
    return harness(
        model,
        batch_size,
        beam_size,
        tokens_per_beam,
        start_token,
        index.semantic_id_depth,
        index.vocab_size,
        index.layer_max_branches,
        resolved_torch_index.packed_csr,
        resolved_torch_index.csr_indptr,
        resolved_torch_index.start_mask,
        resolved_torch_index.dense_mask,
        resolved_torch_index.dense_states,
        device,
        index.dense_lookup_layers,
    )


def static_decoding_sparse_transition_jax(
    *,
    index: StaticDecodingIndex,
    batch_size: int,
    beam_size: int,
    tokens_per_beam: int,
    start_token: int,
    random_seed: int = 0,
    model: Any | None = None,
    key: Any | None = None,
) -> Any:
    """Call the static_decoding `sparse_transition_jax` decoding harness."""
    if batch_size < 1:
        msg = "batch_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    if beam_size < 1:
        msg = "beam_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    if tokens_per_beam < 1:
        msg = "tokens_per_beam은 1 이상이어야 합니다."
        raise ValueError(msg)
    if index.dense_lookup_layers not in (1, 2):
        msg = "static_decoding JAX harness는 dense_lookup_layers=1 또는 2만 지원합니다."
        raise ValueError(msg)

    jax = load_static_decoding_jax()
    jnp = load_static_decoding_jax_numpy()
    resolved_model = (
        build_static_decoding_jax_random_model(vocab_size=index.vocab_size)
        if model is None
        else model
    )
    resolved_key = jax.random.PRNGKey(random_seed) if key is None else key
    harness = load_static_decoding_jax_sparse_transition_harness()
    return harness(
        resolved_model,
        resolved_key,
        batch_size,
        beam_size,
        tokens_per_beam,
        start_token,
        index.semantic_id_depth,
        index.vocab_size,
        index.layer_max_branches,
        jnp.asarray(index.packed_csr),
        jnp.asarray(index.csr_indptr),
        jnp.asarray(index.start_mask),
        jnp.asarray(index.dense_mask),
        jnp.asarray(index.dense_states),
        index.dense_lookup_layers,
    )


def build_static_decoding_random_model(*, vocab_size: int, device: torch.device) -> torch.nn.Module:
    """Build the STATIC PyTorch `RandomModel` benchmark model."""
    if vocab_size < 1:
        msg = "vocab_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    factory = load_static_decoding_random_model_factory()
    return factory(vocab_size, device)


def build_static_decoding_jax_random_model(*, vocab_size: int) -> Any:
    """Build the STATIC JAX `RandomModel` benchmark model."""
    if vocab_size < 1:
        msg = "vocab_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    factory = load_static_decoding_jax_random_model_factory()
    return factory(vocab_size)


def validate_static_decoding_index_matches_codec(
    *,
    index: StaticDecodingIndex,
    codec: SemanticIdCodec,
) -> None:
    """Validate that an STATIC index can represent every codec Semantic ID."""
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


@torch.inference_mode()
def static_decoding_constrained_beam_search(
    *,
    index: StaticDecodingIndex,
    beam_size: int,
    logits_provider: StaticDecodingBeamLogitProvider,
    device: torch.device,
    max_results: int | None = None,
    tokens_per_beam: int | None = None,
    torch_index: StaticDecodingTorchIndex | None = None,
) -> tuple[BeamSearchResult, ...]:
    """Beam search that uses the STATIC PyTorch masking kernel."""
    if beam_size < 1:
        msg = "beam_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    if max_results is not None and max_results < 1:
        msg = "max_results는 None이거나 1 이상이어야 합니다."
        raise ValueError(msg)
    if index.dense_lookup_layers not in (1, 2):
        msg = (
            "static_decoding PyTorch harness와 동일하게 dense_lookup_layers는 "
            "1 또는 2만 지원합니다."
        )
        raise ValueError(msg)

    result_limit = beam_size if max_results is None else max_results
    requested_candidates = beam_size if tokens_per_beam is None else tokens_per_beam
    if requested_candidates < 1:
        msg = "tokens_per_beam은 1 이상이어야 합니다."
        raise ValueError(msg)
    per_beam_candidates = min(requested_candidates, index.vocab_size)

    resolved_torch_index = index.to_torch(device) if torch_index is None else torch_index
    if resolved_torch_index.device != device:
        msg = (
            "torch_index device가 요청한 device와 다릅니다: "
            f"torch_index={resolved_torch_index.device}, device={device}"
        )
        raise ValueError(msg)
    prefixes: tuple[tuple[int, ...], ...] = ((),)
    empty_states = torch.empty((1,), dtype=torch.long, device=device)
    initial_logits = _validate_torch_logits(
        logits_provider(prefixes, empty_states, 0),
        num_rows=1,
        vocab_size=index.vocab_size,
        device=device,
    )
    initial_logprobs = F.log_softmax(initial_logits, dim=-1)
    initial_logprobs = torch.where(
        resolved_torch_index.start_mask.unsqueeze(0),
        initial_logprobs,
        _negative_inf_like(initial_logprobs),
    )

    start_count = int(resolved_torch_index.start_mask.sum().item())
    if start_count == 0:
        return ()
    initial_k = min(beam_size, start_count)
    current_scores, top_tokens = torch.topk(initial_logprobs, initial_k, dim=-1)
    current_scores = current_scores.squeeze(0)
    top_tokens = top_tokens.squeeze(0)
    token_buffer = torch.full(
        (initial_k, index.semantic_id_depth),
        -1,
        dtype=torch.long,
        device=device,
    )
    token_buffer[:, 0] = top_tokens
    current_states = top_tokens + 1

    for prefix_length in range(1, index.semantic_id_depth):
        prefixes = _prefixes_from_buffer(token_buffer, prefix_length)
        logits = _validate_torch_logits(
            logits_provider(prefixes, current_states, prefix_length),
            num_rows=len(prefixes),
            vocab_size=index.vocab_size,
            device=device,
        )
        logprobs = F.log_softmax(logits, dim=-1)

        if prefix_length < index.dense_lookup_layers:
            candidate_logprobs, candidate_tokens, candidate_states = _dense_candidates(
                logprobs=logprobs,
                states=current_states,
                torch_index=resolved_torch_index,
                tokens_per_beam=per_beam_candidates,
            )
        else:
            candidate_logprobs, candidate_tokens, candidate_states = (
                static_decoding_generate_and_apply_logprobs_mask(
                    flat_logprobs=logprobs,
                    flat_states=current_states,
                    index=resolved_torch_index,
                    prefix_length=prefix_length,
                )
            )

        limit = candidate_logprobs.shape[1]
        flat_scores = (current_scores.unsqueeze(1) + candidate_logprobs).reshape(-1)
        finite_count = int(torch.isfinite(flat_scores).sum().item())
        if finite_count == 0:
            return ()
        selected_count = min(beam_size, finite_count)
        next_scores, flat_indices = torch.topk(flat_scores, selected_count, dim=-1)
        parent_indices = torch.div(flat_indices, limit, rounding_mode="floor")

        token_buffer = token_buffer[parent_indices].clone()
        token_buffer[:, prefix_length] = candidate_tokens.reshape(-1)[flat_indices]
        current_states = candidate_states.reshape(-1)[flat_indices]
        current_scores = next_scores

    completed: list[BeamSearchResult] = []
    for row_index, state in enumerate(current_states.tolist()):
        if int(state) != 0:
            continue
        semantic_id = tuple(int(token) for token in token_buffer[row_index].tolist())
        log_score = float(current_scores[row_index].item())
        completed.append(
            BeamSearchResult(
                semantic_id=semantic_id,
                score=math.exp(log_score / max(len(semantic_id), 1)),
                log_score=log_score,
            )
        )
    completed.sort(key=lambda result: result.log_score, reverse=True)
    return tuple(completed[:result_limit])


def _dense_candidates(
    *,
    logprobs: torch.Tensor,
    states: torch.Tensor,
    torch_index: StaticDecodingTorchIndex,
    tokens_per_beam: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch_index.dense_lookup_layers != 2:
        msg = "dense_lookup_layers=1에서는 dense candidate 단계가 호출되지 않아야 합니다."
        raise ValueError(msg)
    parent_tokens = (states - 1).long()
    dense_masks = torch_index.dense_mask[parent_tokens]
    masked_logprobs = torch.where(dense_masks, logprobs, _negative_inf_like(logprobs))
    limit = min(tokens_per_beam, logprobs.shape[1])
    top_logprobs, top_tokens = torch.topk(masked_logprobs, limit, dim=-1)
    next_states = torch_index.dense_states[parent_tokens.unsqueeze(1), top_tokens.long()]
    return top_logprobs, top_tokens, next_states


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


def _normalize_tokens(tokens: Iterable[Any], *, allow_empty: bool = False) -> tuple[int, ...]:
    if isinstance(tokens, str):
        msg = "Semantic ID sequence는 문자열이 아니라 정수 iterable이어야 합니다."
        raise ValueError(msg)
    normalized = tuple(_normalize_token(token) for token in tokens)
    if not normalized and not allow_empty:
        msg = "Semantic ID sequence는 비어 있을 수 없습니다."
        raise ValueError(msg)
    return normalized


def _try_normalize_tokens(tokens: Iterable[Any], *, allow_empty: bool) -> tuple[int, ...] | None:
    try:
        return _normalize_tokens(tokens, allow_empty=allow_empty)
    except ValueError:
        return None


def _normalize_token(token: Any) -> int:
    if isinstance(token, (bool, np.bool_)) or not isinstance(token, (int, np.integer)):
        msg = f"token은 bool이 아닌 정수여야 합니다: {token!r}"
        raise ValueError(msg)
    if int(token) < 0:
        msg = f"token은 0 이상이어야 합니다: {token}"
        raise ValueError(msg)
    return int(token)


def _nonzero_tokens(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(token) for token in np.flatnonzero(mask))


def _validate_torch_logits(
    logits: torch.Tensor,
    *,
    num_rows: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    if logits.shape != (num_rows, vocab_size):
        msg = f"logits shape가 {(num_rows, vocab_size)}이어야 합니다: {tuple(logits.shape)}"
        raise ValueError(msg)
    return logits.to(device=device, dtype=torch.float32)


def _negative_inf_like(tensor: torch.Tensor) -> torch.Tensor:
    return torch.full_like(tensor, -float("inf"))


def _prefixes_from_buffer(
    token_buffer: torch.Tensor,
    prefix_length: int,
) -> tuple[tuple[int, ...], ...]:
    prefix_rows = token_buffer[:, :prefix_length].detach().cpu().tolist()
    return tuple(tuple(int(token) for token in row) for row in prefix_rows)

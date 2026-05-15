"""Constrained beam search for Semantic ID generation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from recsys.decoding.static_matrix import StaticTransitionMatrixDecoder

BeamLogitProvider = Callable[[tuple[tuple[int, ...], ...], np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class BeamSearchResult:
    """완성된 Semantic ID beam."""

    semantic_id: tuple[int, ...]
    score: float
    log_score: float


@dataclass(frozen=True)
class _Beam:
    semantic_id: tuple[int, ...]
    state: int
    log_score: float


def constrained_beam_search(
    *,
    decoder: StaticTransitionMatrixDecoder,
    depth: int,
    beam_size: int,
    logits_provider: BeamLogitProvider,
    max_results: int | None = None,
) -> tuple[BeamSearchResult, ...]:
    """STATIC mask를 적용해 유효한 Semantic ID만 생성한다."""
    if depth < 1:
        msg = "depth는 1 이상이어야 합니다."
        raise ValueError(msg)
    if beam_size < 1:
        msg = "beam_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    if max_results is not None and max_results < 1:
        msg = "max_results는 None이거나 1 이상이어야 합니다."
        raise ValueError(msg)

    result_limit = beam_size if max_results is None else max_results
    beams: tuple[_Beam, ...] = (_Beam(semantic_id=(), state=decoder.root_state, log_score=0.0),)

    for step in range(depth):
        prefixes = tuple(beam.semantic_id for beam in beams)
        states = np.asarray([beam.state for beam in beams], dtype=np.int64)
        logits = _validate_logits(
            logits_provider(prefixes, states, step), len(beams), decoder.vocab_size
        )
        allowed_mask = decoder.allowed_token_mask(states)
        candidates: list[_Beam] = []

        for row_index, beam in enumerate(beams):
            masked_logits = np.where(allowed_mask[row_index], logits[row_index], -np.inf)
            log_probs = _log_softmax(masked_logits)
            for token in np.flatnonzero(np.isfinite(log_probs)):
                next_state = decoder.next_state(beam.state, int(token))
                if next_state is None:
                    continue
                candidates.append(
                    _Beam(
                        semantic_id=(*beam.semantic_id, int(token)),
                        state=next_state,
                        log_score=beam.log_score + float(log_probs[token]),
                    )
                )

        if not candidates:
            return ()
        beams = tuple(
            sorted(candidates, key=lambda candidate: candidate.log_score, reverse=True)[:beam_size]
        )

    completed = [
        BeamSearchResult(
            semantic_id=beam.semantic_id,
            score=math.exp(beam.log_score / max(len(beam.semantic_id), 1)),
            log_score=beam.log_score,
        )
        for beam in beams
        if decoder.is_terminal_state(beam.state)
    ]
    completed.sort(key=lambda result: result.log_score, reverse=True)
    return tuple(completed[:result_limit])


def _validate_logits(logits: np.ndarray, num_beams: int, vocab_size: int) -> np.ndarray:
    array = np.asarray(logits, dtype=np.float64)
    expected_shape = (num_beams, vocab_size)
    if array.shape != expected_shape:
        msg = f"logits shape가 {expected_shape}이어야 합니다: {array.shape}"
        raise ValueError(msg)
    return array


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(logits)
    if not np.any(finite_mask):
        return np.full(logits.shape, -np.inf, dtype=np.float64)

    result = np.full(logits.shape, -np.inf, dtype=np.float64)
    finite_logits = logits[finite_mask]
    max_logit = np.max(finite_logits)
    log_denominator = max_logit + np.log(np.exp(finite_logits - max_logit).sum())
    result[finite_mask] = finite_logits - log_denominator
    return result

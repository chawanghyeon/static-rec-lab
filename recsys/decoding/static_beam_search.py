"""STATIC constrained beam search."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

from recsys.decoding.beam_search import BeamSearchResult
from recsys.decoding.static_artifact import StaticDecodingIndex, StaticDecodingTorchIndex
from recsys.decoding.static_runtime import static_decoding_generate_and_apply_logprobs_mask

StaticDecodingBeamLogitProvider = Callable[
    [tuple[tuple[int, ...], ...], torch.Tensor, int],
    torch.Tensor,
]


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
    """STATIC PyTorch masking kernel을 사용하는 beam search."""
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

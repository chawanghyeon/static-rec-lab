"""Generative retriever constrained inference utilities."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from recsys.decoding import (
    BeamSearchResult,
    StaticDecodingIndex,
    StaticDecodingTorchIndex,
    static_decoding_constrained_beam_search,
    static_decoding_generate_and_apply_logprobs_mask,
)
from recsys.models.dataset import (
    BOS_TOKEN_ID,
    PAD_TOKEN_ID,
    SEMANTIC_TOKEN_OFFSET,
    UNK_ITEM_INDEX,
)
from recsys.models.generative_retriever import GenerativeRetriever


@torch.no_grad()
def generate_semantic_ids_with_static_decoding(
    *,
    model: GenerativeRetriever,
    index: StaticDecodingIndex,
    torch_index: StaticDecodingTorchIndex | None = None,
    history_item_ids: Sequence[int],
    item_to_index: Mapping[int, int],
    beam_size: int,
    max_results: int,
    device: torch.device,
) -> tuple[BeamSearchResult, ...]:
    """static_decoding PyTorch kernel을 사용해 Semantic ID를 생성한다."""
    if beam_size < 1:
        msg = "beam_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    if max_results < 1:
        msg = "max_results는 1 이상이어야 합니다."
        raise ValueError(msg)
    if index.semantic_id_depth != model.config.semantic_id_length:
        msg = (
            "static_decoding STATIC index depth와 model Semantic ID length가 같아야 합니다: "
            f"index={index.semantic_id_depth}, model={model.config.semantic_id_length}"
        )
        raise ValueError(msg)

    model.eval()
    history_item_tensor, history_padding_mask = _build_history_tensors(
        history_item_ids=history_item_ids,
        item_to_index=item_to_index,
        max_history_length=model.config.max_history_length,
        device=device,
    )

    def logits_provider(
        prefixes: tuple[tuple[int, ...], ...],
        _states: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        batch_size = len(prefixes)
        decoder_input_ids = torch.full(
            (batch_size, model.config.semantic_id_length),
            PAD_TOKEN_ID,
            dtype=torch.long,
            device=device,
        )
        decoder_input_ids[:, 0] = BOS_TOKEN_ID
        for row_index, prefix in enumerate(prefixes):
            shifted_tokens = [
                token + SEMANTIC_TOKEN_OFFSET
                for token in prefix[: model.config.semantic_id_length - 1]
            ]
            if shifted_tokens:
                decoder_input_ids[
                    row_index,
                    1 : 1 + len(shifted_tokens),
                ] = torch.tensor(shifted_tokens, dtype=torch.long, device=device)

        logits = model(
            history_item_tensor.expand(batch_size, -1),
            history_padding_mask.expand(batch_size, -1),
            decoder_input_ids,
        )
        return _extract_raw_semantic_logits_tensor(logits[:, step, :], index.vocab_size)

    return static_decoding_constrained_beam_search(
        index=index,
        beam_size=beam_size,
        logits_provider=logits_provider,
        device=device,
        max_results=max_results,
        torch_index=torch_index,
    )


@torch.no_grad()
def generate_semantic_ids_batch_with_static_decoding(
    *,
    model: GenerativeRetriever,
    index: StaticDecodingIndex,
    torch_index: StaticDecodingTorchIndex | None = None,
    history_item_ids_batch: Sequence[Sequence[int]],
    item_to_index: Mapping[int, int],
    beam_size: int,
    max_results: int,
    device: torch.device,
) -> tuple[tuple[BeamSearchResult, ...], ...]:
    """여러 사용자 history를 묶어 static_decoding constrained beam search를 수행한다."""
    if beam_size < 1:
        msg = "beam_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    if max_results < 1:
        msg = "max_results는 1 이상이어야 합니다."
        raise ValueError(msg)
    if index.semantic_id_depth != model.config.semantic_id_length:
        msg = (
            "static_decoding STATIC index depth와 model Semantic ID length가 같아야 합니다: "
            f"index={index.semantic_id_depth}, model={model.config.semantic_id_length}"
        )
        raise ValueError(msg)
    if not history_item_ids_batch:
        return ()

    model.eval()
    resolved_torch_index = index.to_torch(device) if torch_index is None else torch_index
    if resolved_torch_index.device != device:
        msg = (
            "torch_index device가 요청한 device와 다릅니다: "
            f"torch_index={resolved_torch_index.device}, device={device}"
        )
        raise ValueError(msg)

    history_item_tensor, history_padding_mask = _build_history_tensors_batch(
        history_item_ids_batch=history_item_ids_batch,
        item_to_index=item_to_index,
        max_history_length=model.config.max_history_length,
        device=device,
    )
    batch_size = int(history_item_tensor.shape[0])
    depth = index.semantic_id_depth
    requested_candidates = min(beam_size, index.vocab_size)

    decoder_input_ids = torch.full(
        (batch_size, depth),
        PAD_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    decoder_input_ids[:, 0] = BOS_TOKEN_ID
    logits = model(history_item_tensor, history_padding_mask, decoder_input_ids)
    initial_logprobs = torch.log_softmax(
        _extract_raw_semantic_logits_tensor(logits[:, 0, :], index.vocab_size),
        dim=-1,
    )
    initial_logprobs = torch.where(
        resolved_torch_index.start_mask.unsqueeze(0),
        initial_logprobs,
        torch.full_like(initial_logprobs, -torch.inf),
    )

    start_count = int(resolved_torch_index.start_mask.sum().item())
    if start_count == 0:
        return tuple(() for _ in range(batch_size))
    beam_count = min(beam_size, start_count)
    current_scores, top_tokens = torch.topk(initial_logprobs, beam_count, dim=-1)
    token_buffer = torch.full(
        (batch_size, beam_count, depth),
        -1,
        dtype=torch.long,
        device=device,
    )
    token_buffer[:, :, 0] = top_tokens
    current_states = top_tokens + 1

    for prefix_length in range(1, depth):
        flat_beams = batch_size * beam_count
        decoder_input_ids = torch.full(
            (flat_beams, depth),
            PAD_TOKEN_ID,
            dtype=torch.long,
            device=device,
        )
        decoder_input_ids[:, 0] = BOS_TOKEN_ID
        prefix_tokens = token_buffer[:, :, :prefix_length].reshape(flat_beams, prefix_length)
        decoder_input_ids[:, 1 : 1 + prefix_length] = prefix_tokens + SEMANTIC_TOKEN_OFFSET

        expanded_history = history_item_tensor.repeat_interleave(beam_count, dim=0)
        expanded_padding_mask = history_padding_mask.repeat_interleave(beam_count, dim=0)
        logits = model(expanded_history, expanded_padding_mask, decoder_input_ids)
        logprobs = torch.log_softmax(
            _extract_raw_semantic_logits_tensor(logits[:, prefix_length, :], index.vocab_size),
            dim=-1,
        )
        flat_states = current_states.reshape(-1)
        if prefix_length < index.dense_lookup_layers:
            candidate_logprobs, candidate_tokens, candidate_states = _dense_candidates(
                logprobs=logprobs,
                states=flat_states,
                torch_index=resolved_torch_index,
                tokens_per_beam=requested_candidates,
            )
        else:
            candidate_logprobs, candidate_tokens, candidate_states = (
                static_decoding_generate_and_apply_logprobs_mask(
                    flat_logprobs=logprobs,
                    flat_states=flat_states,
                    index=resolved_torch_index,
                    prefix_length=prefix_length,
                )
            )

        candidate_count = int(candidate_logprobs.shape[1])
        total_scores = current_scores.reshape(
            batch_size, beam_count, 1
        ) + candidate_logprobs.reshape(batch_size, beam_count, candidate_count)
        flat_scores = total_scores.reshape(batch_size, beam_count * candidate_count)
        selected_count = min(beam_size, beam_count * candidate_count)
        next_scores, flat_indices = torch.topk(flat_scores, selected_count, dim=-1)
        parent_indices = torch.div(flat_indices, candidate_count, rounding_mode="floor")
        child_indices = flat_indices.remainder(candidate_count)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)

        candidate_tokens_view = candidate_tokens.reshape(batch_size, beam_count, candidate_count)
        candidate_states_view = candidate_states.reshape(batch_size, beam_count, candidate_count)
        token_buffer = token_buffer[batch_indices, parent_indices].clone()
        token_buffer[:, :, prefix_length] = candidate_tokens_view[
            batch_indices,
            parent_indices,
            child_indices,
        ]
        current_states = candidate_states_view[batch_indices, parent_indices, child_indices]
        current_scores = next_scores
        beam_count = selected_count

    results: list[tuple[BeamSearchResult, ...]] = []
    token_buffer_cpu = token_buffer.detach().cpu()
    current_states_cpu = current_states.detach().cpu()
    current_scores_cpu = current_scores.detach().cpu()
    for row_index in range(batch_size):
        row_results: list[BeamSearchResult] = []
        for beam_index in range(beam_count):
            if int(current_states_cpu[row_index, beam_index].item()) != 0:
                continue
            log_score = float(current_scores_cpu[row_index, beam_index].item())
            if not torch.isfinite(current_scores_cpu[row_index, beam_index]):
                continue
            semantic_id = tuple(int(token) for token in token_buffer_cpu[row_index, beam_index])
            row_results.append(
                BeamSearchResult(
                    semantic_id=semantic_id,
                    score=math.exp(log_score / max(len(semantic_id), 1)),
                    log_score=log_score,
                )
            )
        row_results.sort(key=lambda result: result.log_score, reverse=True)
        results.append(tuple(row_results[:max_results]))
    return tuple(results)


def _build_history_tensors(
    *,
    history_item_ids: Sequence[int],
    item_to_index: Mapping[int, int],
    max_history_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    history = tuple(history_item_ids)[-max_history_length:]
    if not history:
        history_indices: tuple[int, ...] = (UNK_ITEM_INDEX,)
    else:
        history_indices = tuple(
            item_to_index.get(int(item_id), UNK_ITEM_INDEX) for item_id in history
        )

    history_tensor = torch.tensor([history_indices], dtype=torch.long, device=device)
    padding_mask = torch.zeros(history_tensor.shape, dtype=torch.bool, device=device)
    return history_tensor, padding_mask


def _build_history_tensors_batch(
    *,
    history_item_ids_batch: Sequence[Sequence[int]],
    item_to_index: Mapping[int, int],
    max_history_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    histories: list[tuple[int, ...]] = []
    for history_item_ids in history_item_ids_batch:
        history = tuple(history_item_ids)[-max_history_length:]
        if not history:
            histories.append((UNK_ITEM_INDEX,))
        else:
            histories.append(
                tuple(item_to_index.get(int(item_id), UNK_ITEM_INDEX) for item_id in history)
            )
    padded_length = max(len(history) for history in histories)
    history_tensor = torch.full(
        (len(histories), padded_length),
        0,
        dtype=torch.long,
        device=device,
    )
    padding_mask = torch.ones(history_tensor.shape, dtype=torch.bool, device=device)
    for row_index, history in enumerate(histories):
        history_tensor[row_index, : len(history)] = torch.tensor(
            history,
            dtype=torch.long,
            device=device,
        )
        padding_mask[row_index, : len(history)] = False
    return history_tensor, padding_mask


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
    masked_logprobs = torch.where(dense_masks, logprobs, torch.full_like(logprobs, -torch.inf))
    limit = min(tokens_per_beam, logprobs.shape[1])
    top_logprobs, top_tokens = torch.topk(masked_logprobs, limit, dim=-1)
    next_states = torch_index.dense_states[parent_tokens.unsqueeze(1), top_tokens.long()]
    return top_logprobs, top_tokens, next_states


def _extract_raw_semantic_logits_tensor(
    shifted_logits: torch.Tensor,
    raw_vocab_size: int,
) -> torch.Tensor:
    raw_logits = torch.full(
        (shifted_logits.shape[0], raw_vocab_size),
        -torch.inf,
        dtype=shifted_logits.dtype,
        device=shifted_logits.device,
    )
    start = SEMANTIC_TOKEN_OFFSET
    end = min(shifted_logits.shape[1], start + raw_vocab_size)
    if end > start:
        raw_logits[:, : end - start] = shifted_logits[:, start:end]
    return raw_logits

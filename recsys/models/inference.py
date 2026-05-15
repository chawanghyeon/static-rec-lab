"""Generative retriever constrained inference utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from recsys.decoding import BeamSearchResult, StaticTransitionMatrixDecoder, constrained_beam_search
from recsys.models.dataset import (
    BOS_TOKEN_ID,
    PAD_TOKEN_ID,
    SEMANTIC_TOKEN_OFFSET,
    UNK_ITEM_INDEX,
)
from recsys.models.generative_retriever import GenerativeRetriever


@torch.no_grad()
def generate_semantic_ids(
    *,
    model: GenerativeRetriever,
    decoder: StaticTransitionMatrixDecoder,
    history_item_ids: Sequence[int],
    item_to_index: Mapping[int, int],
    beam_size: int,
    max_results: int,
    device: torch.device,
) -> tuple[BeamSearchResult, ...]:
    """모델 logits에 STATIC-style decoder mask를 적용해 Semantic ID를 생성한다."""
    if beam_size < 1:
        msg = "beam_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    if max_results < 1:
        msg = "max_results는 1 이상이어야 합니다."
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
        _states: np.ndarray,
        step: int,
    ) -> np.ndarray:
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
        return _extract_raw_semantic_logits(logits[:, step, :], decoder.vocab_size)

    return constrained_beam_search(
        decoder=decoder,
        depth=model.config.semantic_id_length,
        beam_size=beam_size,
        logits_provider=logits_provider,
        max_results=max_results,
    )


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


def _extract_raw_semantic_logits(shifted_logits: torch.Tensor, raw_vocab_size: int) -> np.ndarray:
    logits = shifted_logits.detach().cpu().numpy()
    raw_logits = np.full((logits.shape[0], raw_vocab_size), -np.inf, dtype=np.float64)
    start = SEMANTIC_TOKEN_OFFSET
    end = min(logits.shape[1], start + raw_vocab_size)
    if end > start:
        raw_logits[:, : end - start] = logits[:, start:end]
    return raw_logits

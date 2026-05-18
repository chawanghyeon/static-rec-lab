"""Generative retrieval 모델 학습/평가 유틸리티."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from recsys.models.dataset import PAD_TOKEN_ID, GenerativeBatch
from recsys.models.generative_retriever import GenerativeRetriever, GenerativeRetrieverConfig


@dataclass(frozen=True)
class GenerativeTrainingMetrics:
    """학습 또는 평가 metric."""

    loss: float
    token_accuracy: float
    sequence_accuracy: float
    num_examples: int


def train_one_epoch(
    model: GenerativeRetriever,
    dataloader: Iterable[GenerativeBatch],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    log_every_batches: int | None = None,
) -> GenerativeTrainingMetrics:
    """Teacher forcing으로 한 epoch 학습한다."""
    if log_every_batches is not None and log_every_batches < 1:
        msg = "log_every_batches는 None이거나 1 이상이어야 합니다."
        raise ValueError(msg)
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_tokens = 0
    correct_tokens = 0
    correct_sequences = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction="sum")

    for batch_index, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        logits = model(
            batch.history_item_ids,
            batch.history_padding_mask,
            batch.decoder_input_ids,
        )
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            batch.target_token_ids.reshape(-1),
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_examples += int(batch.target_token_ids.shape[0])
        total_tokens += _num_target_tokens(batch.target_token_ids)
        correct_tokens += _num_correct_tokens(logits.detach(), batch.target_token_ids)
        correct_sequences += _num_correct_sequences(logits.detach(), batch.target_token_ids)
        if log_every_batches is not None and batch_index % log_every_batches == 0:
            print(
                "train "
                f"batches={batch_index:,} "
                f"examples={total_examples:,} "
                f"loss={total_loss / max(total_tokens, 1):.6f}"
            )

    return _aggregate_metrics(
        total_loss=total_loss,
        total_examples=total_examples,
        total_tokens=total_tokens,
        correct_tokens=correct_tokens,
        correct_sequences=correct_sequences,
    )


@torch.no_grad()
def evaluate_model(
    model: GenerativeRetriever,
    dataloader: Iterable[GenerativeBatch],
    *,
    device: torch.device,
    log_every_batches: int | None = None,
) -> GenerativeTrainingMetrics:
    """Validation loss와 token/sequence accuracy를 계산한다."""
    if log_every_batches is not None and log_every_batches < 1:
        msg = "log_every_batches는 None이거나 1 이상이어야 합니다."
        raise ValueError(msg)
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_tokens = 0
    correct_tokens = 0
    correct_sequences = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction="sum")

    for batch_index, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        logits = model(
            batch.history_item_ids,
            batch.history_padding_mask,
            batch.decoder_input_ids,
        )
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            batch.target_token_ids.reshape(-1),
        )
        total_loss += float(loss.detach().cpu())
        total_examples += int(batch.target_token_ids.shape[0])
        total_tokens += _num_target_tokens(batch.target_token_ids)
        correct_tokens += _num_correct_tokens(logits, batch.target_token_ids)
        correct_sequences += _num_correct_sequences(logits, batch.target_token_ids)
        if log_every_batches is not None and batch_index % log_every_batches == 0:
            print(
                "eval "
                f"batches={batch_index:,} "
                f"examples={total_examples:,} "
                f"loss={total_loss / max(total_tokens, 1):.6f}"
            )

    return _aggregate_metrics(
        total_loss=total_loss,
        total_examples=total_examples,
        total_tokens=total_tokens,
        correct_tokens=correct_tokens,
        correct_sequences=correct_sequences,
    )


def move_batch_to_device(batch: GenerativeBatch, device: torch.device) -> GenerativeBatch:
    """Batch tensor를 device로 이동한다."""
    return GenerativeBatch(
        history_item_ids=batch.history_item_ids.to(device),
        history_padding_mask=batch.history_padding_mask.to(device),
        decoder_input_ids=batch.decoder_input_ids.to(device),
        target_token_ids=batch.target_token_ids.to(device),
    )


def save_checkpoint(
    path: str | Path,
    *,
    model: GenerativeRetriever,
    item_to_index: dict[int, int],
    metrics: GenerativeTrainingMetrics,
) -> Path:
    """모델 checkpoint를 저장한다."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config.to_dict(),
            "item_to_index": {str(item_id): index for item_id, index in item_to_index.items()},
            "metrics": {
                "loss": metrics.loss,
                "token_accuracy": metrics.token_accuracy,
                "sequence_accuracy": metrics.sequence_accuracy,
                "num_examples": metrics.num_examples,
            },
        },
        output_path,
    )
    return output_path


def load_checkpoint(
    path: str | Path, *, device: torch.device
) -> tuple[GenerativeRetriever, dict[int, int]]:
    """checkpoint에서 모델과 item index mapping을 로드한다."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        msg = "checkpoint payload는 dict여야 합니다."
        raise ValueError(msg)
    model_config = GenerativeRetrieverConfig(**payload["model_config"])
    model = GenerativeRetriever(model_config).to(device)
    state_dict = cast(dict[str, Any], payload["model_state_dict"])
    model.load_state_dict(state_dict)
    item_to_index = {
        int(item_id): int(index) for item_id, index in payload["item_to_index"].items()
    }
    return model, item_to_index


def _aggregate_metrics(
    *,
    total_loss: float,
    total_examples: int,
    total_tokens: int,
    correct_tokens: int,
    correct_sequences: int,
) -> GenerativeTrainingMetrics:
    if total_examples < 1:
        msg = "metric을 계산할 example이 없습니다."
        raise ValueError(msg)
    return GenerativeTrainingMetrics(
        loss=total_loss / max(total_tokens, 1),
        token_accuracy=correct_tokens / max(total_tokens, 1),
        sequence_accuracy=correct_sequences / total_examples,
        num_examples=total_examples,
    )


def _num_target_tokens(targets: torch.Tensor) -> int:
    return int((targets != PAD_TOKEN_ID).sum().item())


def _num_correct_tokens(logits: torch.Tensor, targets: torch.Tensor) -> int:
    predictions = logits.argmax(dim=-1)
    mask = targets != PAD_TOKEN_ID
    return int(((predictions == targets) & mask).sum().item())


def _num_correct_sequences(logits: torch.Tensor, targets: torch.Tensor) -> int:
    predictions = logits.argmax(dim=-1)
    mask = targets != PAD_TOKEN_ID
    per_token_match = (predictions == targets) | ~mask
    return int(per_token_match.all(dim=1).sum().item())

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
    model: nn.Module,
    dataloader: Iterable[GenerativeBatch],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    log_every_batches: int | None = None,
    compute_accuracy: bool = False,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> GenerativeTrainingMetrics:
    """Teacher forcing으로 한 epoch 학습한다."""
    if log_every_batches is not None and log_every_batches < 1:
        msg = "log_every_batches는 None이거나 1 이상이어야 합니다."
        raise ValueError(msg)
    model.train()
    total_loss: torch.Tensor | None = None
    total_examples = 0
    total_tokens = 0
    correct_tokens: torch.Tensor | int = 0
    correct_sequences: torch.Tensor | int = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction="sum")
    amp_module = cast(Any, torch.amp)
    scaler = amp_module.GradScaler(
        "cuda",
        enabled=use_amp and device.type == "cuda" and amp_dtype == torch.float16,
    )

    for batch_index, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=_use_autocast(device=device, use_amp=use_amp),
        ):
            logits = model(
                batch.history_item_ids,
                batch.history_padding_mask,
                batch.decoder_input_ids,
            )
            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]),
                batch.target_token_ids.reshape(-1),
            )
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        detached_loss = loss.detach()
        total_loss = detached_loss if total_loss is None else total_loss + detached_loss
        total_examples += int(batch.target_token_ids.shape[0])
        total_tokens += int(batch.target_token_ids.numel())
        if compute_accuracy:
            detached_logits = logits.detach()
            correct_tokens = _add_count(
                correct_tokens,
                _num_correct_tokens(detached_logits, batch.target_token_ids),
            )
            correct_sequences = _add_count(
                correct_sequences,
                _num_correct_sequences(detached_logits, batch.target_token_ids),
            )
        if log_every_batches is not None and batch_index % log_every_batches == 0:
            synced_total_loss = _loss_value(total_loss)
            print(
                "train "
                f"batches={batch_index:,} "
                f"examples={total_examples:,} "
                f"loss={synced_total_loss / max(total_tokens, 1):.6f}",
                flush=True,
            )

    return _aggregate_metrics(
        total_loss=_loss_value(total_loss),
        total_examples=total_examples,
        total_tokens=total_tokens,
        correct_tokens=correct_tokens,
        correct_sequences=correct_sequences,
    )


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    dataloader: Iterable[GenerativeBatch],
    *,
    device: torch.device,
    log_every_batches: int | None = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    compute_accuracy: bool = True,
) -> GenerativeTrainingMetrics:
    """Validation loss와 token/sequence accuracy를 계산한다."""
    if log_every_batches is not None and log_every_batches < 1:
        msg = "log_every_batches는 None이거나 1 이상이어야 합니다."
        raise ValueError(msg)
    model.eval()
    total_loss: torch.Tensor | None = None
    total_examples = 0
    total_tokens = 0
    correct_tokens: torch.Tensor | int = 0
    correct_sequences: torch.Tensor | int = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction="sum")

    for batch_index, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=_use_autocast(device=device, use_amp=use_amp),
        ):
            logits = model(
                batch.history_item_ids,
                batch.history_padding_mask,
                batch.decoder_input_ids,
            )
            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]),
                batch.target_token_ids.reshape(-1),
            )
        detached_loss = loss.detach()
        total_loss = detached_loss if total_loss is None else total_loss + detached_loss
        total_examples += int(batch.target_token_ids.shape[0])
        total_tokens += int(batch.target_token_ids.numel())
        if compute_accuracy:
            correct_tokens = _add_count(
                correct_tokens,
                _num_correct_tokens(logits, batch.target_token_ids),
            )
            correct_sequences = _add_count(
                correct_sequences,
                _num_correct_sequences(logits, batch.target_token_ids),
            )
        if log_every_batches is not None and batch_index % log_every_batches == 0:
            synced_total_loss = _loss_value(total_loss)
            print(
                "eval "
                f"batches={batch_index:,} "
                f"examples={total_examples:,} "
                f"loss={synced_total_loss / max(total_tokens, 1):.6f}",
                flush=True,
            )

    return _aggregate_metrics(
        total_loss=_loss_value(total_loss),
        total_examples=total_examples,
        total_tokens=total_tokens,
        correct_tokens=correct_tokens,
        correct_sequences=correct_sequences,
    )


def move_batch_to_device(batch: GenerativeBatch, device: torch.device) -> GenerativeBatch:
    """Batch tensor를 device로 이동한다."""
    return GenerativeBatch(
        history_item_ids=batch.history_item_ids.to(device, non_blocking=True),
        history_padding_mask=batch.history_padding_mask.to(device, non_blocking=True),
        decoder_input_ids=batch.decoder_input_ids.to(device, non_blocking=True),
        target_token_ids=batch.target_token_ids.to(device, non_blocking=True),
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
    correct_tokens: torch.Tensor | int,
    correct_sequences: torch.Tensor | int,
) -> GenerativeTrainingMetrics:
    if total_examples < 1:
        msg = "metric을 계산할 example이 없습니다."
        raise ValueError(msg)
    correct_token_count = _count_value(correct_tokens)
    correct_sequence_count = _count_value(correct_sequences)
    return GenerativeTrainingMetrics(
        loss=total_loss / max(total_tokens, 1),
        token_accuracy=correct_token_count / max(total_tokens, 1),
        sequence_accuracy=correct_sequence_count / total_examples,
        num_examples=total_examples,
    )


def _loss_value(loss: torch.Tensor | None) -> float:
    if loss is None:
        return 0.0
    return float(loss.detach().cpu())


def _count_value(count: torch.Tensor | int) -> int:
    if isinstance(count, int):
        return count
    return int(count.detach().cpu())


def _add_count(total: torch.Tensor | int, count: torch.Tensor) -> torch.Tensor:
    if isinstance(total, int):
        return count
    return total + count


def _use_autocast(*, device: torch.device, use_amp: bool) -> bool:
    return use_amp and device.type in {"cpu", "cuda", "mps"}


def _num_correct_tokens(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = logits.argmax(dim=-1)
    mask = targets != PAD_TOKEN_ID
    return ((predictions == targets) & mask).sum()


def _num_correct_sequences(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = logits.argmax(dim=-1)
    mask = targets != PAD_TOKEN_ID
    per_token_match = (predictions == targets) | ~mask
    return per_token_match.all(dim=1).sum()

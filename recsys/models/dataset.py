"""Generative retrieval 학습용 streaming Dataset."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from recsys.semantic_id import SemanticIdCodec

PAD_ITEM_INDEX = 0
UNK_ITEM_INDEX = 1
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
SEMANTIC_TOKEN_OFFSET = 2


@dataclass(frozen=True)
class GenerativeBatch:
    """Transformer 학습 batch."""

    history_item_ids: torch.Tensor
    history_padding_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    target_token_ids: torch.Tensor


class GenerativeParquetBatchIterableDataset(IterableDataset[GenerativeBatch]):
    """대용량 parquet에서 학습 batch를 직접 생성하는 Dataset."""

    def __init__(
        self,
        path: str | Path,
        codec: SemanticIdCodec,
        *,
        item_to_index: Mapping[int, int],
        batch_size: int,
        max_examples: int | None = None,
        parquet_batch_size: int = 65_536,
        fixed_history_length: int | None = None,
    ) -> None:
        if batch_size < 1:
            msg = "batch_size는 1 이상이어야 합니다."
            raise ValueError(msg)
        if max_examples is not None and max_examples < 1:
            msg = "max_examples는 None이거나 1 이상이어야 합니다."
            raise ValueError(msg)
        if parquet_batch_size < 1:
            msg = "parquet_batch_size는 1 이상이어야 합니다."
            raise ValueError(msg)
        if fixed_history_length is not None and fixed_history_length < 1:
            msg = "fixed_history_length는 None이거나 1 이상이어야 합니다."
            raise ValueError(msg)

        self._path = Path(path)
        self._item_index_lookup = _build_item_index_lookup(item_to_index)
        self._target_token_lookup = _build_target_token_lookup(codec)
        self._batch_size = batch_size
        self._max_examples = max_examples
        self._parquet_batch_size = parquet_batch_size
        self._fixed_history_length = fixed_history_length

    def __iter__(self) -> Iterator[GenerativeBatch]:
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        pq_module = cast(Any, pq)
        parquet_file = pq_module.ParquetFile(self._path)
        row_groups = _row_groups_for_worker(
            parquet_file=parquet_file,
            worker_id=worker_id,
            num_workers=num_workers,
        )
        if not row_groups:
            return

        active_workers = min(num_workers, int(parquet_file.num_row_groups))
        worker_max_examples = _max_examples_for_worker(
            max_examples=self._max_examples,
            worker_id=worker_id,
            num_workers=active_workers,
        )
        if worker_max_examples == 0:
            return

        yielded = 0
        for record_batch in parquet_file.iter_batches(
            batch_size=self._parquet_batch_size,
            row_groups=row_groups,
            columns=["history_item_ids", "target_item_id"],
        ):
            if worker_max_examples is not None and yielded >= worker_max_examples:
                return

            history_item_ids, history_padding_mask, target_token_ids = (
                _record_batch_to_generative_arrays(
                    record_batch,
                    item_index_lookup=self._item_index_lookup,
                    target_token_lookup=self._target_token_lookup,
                    fixed_history_length=self._fixed_history_length,
                )
            )
            if target_token_ids.shape[0] == 0:
                continue

            if worker_max_examples is not None:
                remaining = worker_max_examples - yielded
                if remaining <= 0:
                    return
                history_item_ids = history_item_ids[:remaining]
                history_padding_mask = history_padding_mask[:remaining]
                target_token_ids = target_token_ids[:remaining]

            for start in range(0, target_token_ids.shape[0], self._batch_size):
                end = start + self._batch_size
                yield _arrays_to_generative_batch(
                    history_item_ids[start:end],
                    history_padding_mask[start:end],
                    target_token_ids[start:end],
                )
                yielded += int(target_token_ids[start:end].shape[0])
                if worker_max_examples is not None and yielded >= worker_max_examples:
                    return


def _row_groups_for_worker(
    *,
    parquet_file: Any,
    worker_id: int,
    num_workers: int,
) -> list[int]:
    return [
        row_group_index
        for row_group_index in range(int(parquet_file.num_row_groups))
        if row_group_index % num_workers == worker_id
    ]


def _max_examples_for_worker(
    *,
    max_examples: int | None,
    worker_id: int,
    num_workers: int,
) -> int | None:
    if max_examples is None:
        return None
    examples_per_worker = max_examples // num_workers
    if worker_id < max_examples % num_workers:
        examples_per_worker += 1
    return examples_per_worker


def build_item_index_from_codec(codec: SemanticIdCodec) -> dict[int, int]:
    """Semantic ID codec item universe만 사용해 item vocabulary를 만든다."""
    return {
        item_id: index for index, item_id in enumerate(sorted(codec.item_to_semantic_id), start=2)
    }


def _build_item_index_lookup(item_to_index: Mapping[int, int]) -> np.ndarray:
    if not item_to_index:
        return np.asarray([UNK_ITEM_INDEX], dtype=np.int64)
    max_item_id = max(int(item_id) for item_id in item_to_index)
    lookup = np.full(max_item_id + 1, UNK_ITEM_INDEX, dtype=np.int64)
    for item_id, item_index in item_to_index.items():
        normalized_item_id = int(item_id)
        if normalized_item_id >= 0:
            lookup[normalized_item_id] = int(item_index)
    return lookup


def _build_target_token_lookup(codec: SemanticIdCodec) -> np.ndarray:
    max_item_id = max(int(item_id) for item_id in codec.item_to_semantic_id)
    lookup = np.full(
        (max_item_id + 1, codec.semantic_id_length),
        -1,
        dtype=np.int64,
    )
    for item_id, semantic_id in codec.item_to_semantic_id.items():
        normalized_item_id = int(item_id)
        if normalized_item_id >= 0:
            lookup[normalized_item_id] = np.asarray(
                [token + SEMANTIC_TOKEN_OFFSET for token in semantic_id],
                dtype=np.int64,
            )
    return lookup


def infer_semantic_vocab_size(codec: SemanticIdCodec) -> int:
    """Special token을 포함한 semantic token vocabulary 크기를 계산한다."""
    max_token = max(
        token for semantic_id in codec.item_to_semantic_id.values() for token in semantic_id
    )
    return max_token + SEMANTIC_TOKEN_OFFSET + 1


def _record_batch_to_generative_arrays(
    record_batch: Any,
    *,
    item_index_lookup: np.ndarray,
    target_token_lookup: np.ndarray,
    fixed_history_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_column = record_batch.column("history_item_ids")
    target_item_ids = np.asarray(
        record_batch.column("target_item_id").to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    num_rows = len(target_item_ids)
    if num_rows == 0:
        empty_history = np.empty((0, 1), dtype=np.int64)
        empty_padding = np.empty((0, 1), dtype=bool)
        empty_targets = np.empty((0, target_token_lookup.shape[1]), dtype=np.int64)
        return empty_history, empty_padding, empty_targets

    offsets = np.asarray(history_column.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
    values = np.asarray(history_column.values.to_numpy(zero_copy_only=False), dtype=np.int64)
    lengths = offsets[1:] - offsets[:-1]
    max_observed_length = int(lengths.max(initial=0))
    max_history_length = (
        max(max_observed_length, 1) if fixed_history_length is None else fixed_history_length
    )
    if max_observed_length > max_history_length:
        msg = (
            "record batch history length가 fixed_history_length를 초과했습니다: "
            f"{max_observed_length} > {max_history_length}"
        )
        raise ValueError(msg)
    history_item_ids = np.full((num_rows, max_history_length), PAD_ITEM_INDEX, dtype=np.int64)
    history_padding_mask = np.ones((num_rows, max_history_length), dtype=bool)

    if values.size:
        mapped_values = np.full(values.shape, UNK_ITEM_INDEX, dtype=np.int64)
        in_lookup = (values >= 0) & (values < item_index_lookup.shape[0])
        mapped_values[in_lookup] = item_index_lookup[values[in_lookup]]
        row_indices = np.repeat(np.arange(num_rows, dtype=np.int64), lengths)
        col_indices = np.arange(values.size, dtype=np.int64) - np.repeat(offsets[:-1], lengths)
        history_item_ids[row_indices, col_indices] = mapped_values
        history_padding_mask[row_indices, col_indices] = False

    in_target_lookup = (target_item_ids >= 0) & (target_item_ids < target_token_lookup.shape[0])
    safe_target_item_ids = np.where(in_target_lookup, target_item_ids, 0)
    target_token_ids = target_token_lookup[safe_target_item_ids]
    valid_targets = in_target_lookup & (target_token_ids[:, 0] >= 0)
    if not bool(valid_targets.all()):
        history_item_ids = history_item_ids[valid_targets]
        history_padding_mask = history_padding_mask[valid_targets]
        target_token_ids = target_token_ids[valid_targets]

    return history_item_ids, history_padding_mask, target_token_ids


def _arrays_to_generative_batch(
    history_item_ids: np.ndarray,
    history_padding_mask: np.ndarray,
    target_token_ids: np.ndarray,
) -> GenerativeBatch:
    target_tensor = torch.as_tensor(target_token_ids, dtype=torch.long)
    decoder_input_ids = torch.empty_like(target_tensor)
    decoder_input_ids[:, 0] = BOS_TOKEN_ID
    decoder_input_ids[:, 1:] = target_tensor[:, :-1]
    return GenerativeBatch(
        history_item_ids=torch.as_tensor(history_item_ids, dtype=torch.long),
        history_padding_mask=torch.as_tensor(history_padding_mask, dtype=torch.bool),
        decoder_input_ids=decoder_input_ids,
        target_token_ids=target_tensor,
    )

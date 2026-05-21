"""Generative retrieval 학습용 Dataset과 collate 함수."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from recsys.data import coerce_item_ids
from recsys.semantic_id import SemanticIdCodec, UnknownItemIdError

PAD_ITEM_INDEX = 0
UNK_ITEM_INDEX = 1
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
SEMANTIC_TOKEN_OFFSET = 2


@dataclass(frozen=True)
class GenerativeExample:
    """한 개의 history -> semantic_id 학습 예제."""

    history_item_indices: tuple[int, ...]
    target_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class GenerativeBatch:
    """Transformer 학습 batch."""

    history_item_ids: torch.Tensor
    history_padding_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    target_token_ids: torch.Tensor


@dataclass(frozen=True)
class GenerativeDatasetBundle:
    """Dataset과 vocabulary metadata."""

    dataset: GenerativeRetrievalDataset
    item_to_index: dict[int, int]
    item_vocab_size: int
    semantic_vocab_size: int
    semantic_id_length: int


class GenerativeRetrievalDataset(Dataset[GenerativeExample]):
    """전처리된 sequence example을 PyTorch Dataset으로 감싼다."""

    def __init__(self, examples: Sequence[GenerativeExample]) -> None:
        if not examples:
            msg = "GenerativeRetrievalDataset은 비어 있을 수 없습니다."
            raise ValueError(msg)
        self._examples = tuple(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> GenerativeExample:
        return self._examples[index]


class GenerativeParquetIterableDataset(IterableDataset[GenerativeExample]):
    """대용량 parquet를 batch 단위로 읽는 Generative Retrieval Dataset."""

    def __init__(
        self,
        path: str | Path,
        codec: SemanticIdCodec,
        *,
        item_to_index: Mapping[int, int],
        max_examples: int | None = None,
        parquet_batch_size: int = 65_536,
    ) -> None:
        if max_examples is not None and max_examples < 1:
            msg = "max_examples는 None이거나 1 이상이어야 합니다."
            raise ValueError(msg)
        if parquet_batch_size < 1:
            msg = "parquet_batch_size는 1 이상이어야 합니다."
            raise ValueError(msg)

        self._path = Path(path)
        self._codec = codec
        self._item_to_index = dict(item_to_index)
        self._max_examples = max_examples
        self._parquet_batch_size = parquet_batch_size

    def __iter__(self) -> Iterator[GenerativeExample]:
        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            msg = "GenerativeParquetIterableDataset은 num_workers=0 또는 1에서 사용하세요."
            raise RuntimeError(msg)

        yielded = 0
        pq_module = cast(Any, pq)
        parquet_file = pq_module.ParquetFile(self._path)
        for record_batch in parquet_file.iter_batches(
            batch_size=self._parquet_batch_size,
            columns=["history_item_ids", "target_item_id"],
        ):
            frame = record_batch.to_pandas()
            for row in frame.itertuples(index=False):
                target_item_id = int(cast(Any, row.target_item_id))
                try:
                    semantic_id = self._codec.encode_item(target_item_id)
                except UnknownItemIdError:
                    continue

                history = coerce_item_ids(row.history_item_ids)
                yield GenerativeExample(
                    history_item_indices=tuple(
                        self._item_to_index.get(item_id, UNK_ITEM_INDEX) for item_id in history
                    ),
                    target_token_ids=tuple(token + SEMANTIC_TOKEN_OFFSET for token in semantic_id),
                )
                yielded += 1
                if self._max_examples is not None and yielded >= self._max_examples:
                    return


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

        self._path = Path(path)
        self._item_index_lookup = _build_item_index_lookup(item_to_index)
        self._target_token_lookup = _build_target_token_lookup(codec)
        self._batch_size = batch_size
        self._max_examples = max_examples
        self._parquet_batch_size = parquet_batch_size

    def __iter__(self) -> Iterator[GenerativeBatch]:
        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            msg = "GenerativeParquetBatchIterableDataset은 num_workers=0 또는 1에서 사용하세요."
            raise RuntimeError(msg)

        yielded = 0
        pq_module = cast(Any, pq)
        parquet_file = pq_module.ParquetFile(self._path)

        for record_batch in parquet_file.iter_batches(
            batch_size=self._parquet_batch_size,
            columns=["history_item_ids", "target_item_id"],
        ):
            history_item_ids, history_padding_mask, target_token_ids = (
                _record_batch_to_generative_arrays(
                    record_batch,
                    item_index_lookup=self._item_index_lookup,
                    target_token_lookup=self._target_token_lookup,
                )
            )
            if target_token_ids.shape[0] == 0:
                continue

            if self._max_examples is not None:
                remaining = self._max_examples - yielded
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
                if self._max_examples is not None and yielded >= self._max_examples:
                    return


def build_generative_dataset(
    frame: pd.DataFrame,
    codec: SemanticIdCodec,
    *,
    item_to_index: Mapping[int, int] | None = None,
    max_examples: int | None = None,
) -> GenerativeDatasetBundle:
    """전처리 parquet DataFrame과 codec으로 학습 Dataset을 만든다."""
    if max_examples is not None and max_examples < 1:
        msg = "max_examples는 None이거나 1 이상이어야 합니다."
        raise ValueError(msg)

    resolved_item_to_index = (
        build_item_index(frame, codec) if item_to_index is None else dict(item_to_index)
    )
    semantic_vocab_size = infer_semantic_vocab_size(codec)
    semantic_id_length = codec.semantic_id_length
    examples: list[GenerativeExample] = []

    for row in frame.itertuples(index=False):
        target_item_id = int(cast(Any, row.target_item_id))
        try:
            semantic_id = codec.encode_item(target_item_id)
        except UnknownItemIdError:
            continue

        history = coerce_item_ids(row.history_item_ids)
        examples.append(
            GenerativeExample(
                history_item_indices=tuple(
                    resolved_item_to_index.get(item_id, UNK_ITEM_INDEX) for item_id in history
                ),
                target_token_ids=tuple(token + SEMANTIC_TOKEN_OFFSET for token in semantic_id),
            )
        )
        if max_examples is not None and len(examples) >= max_examples:
            break

    return GenerativeDatasetBundle(
        dataset=GenerativeRetrievalDataset(examples),
        item_to_index=resolved_item_to_index,
        item_vocab_size=max(resolved_item_to_index.values(), default=UNK_ITEM_INDEX) + 1,
        semantic_vocab_size=semantic_vocab_size,
        semantic_id_length=semantic_id_length,
    )


def build_item_index(frame: pd.DataFrame, codec: SemanticIdCodec) -> dict[int, int]:
    """history/target item과 codec item을 contiguous index로 변환한다."""
    item_ids: set[int] = set(codec.item_to_semantic_id)
    for row in frame.itertuples(index=False):
        item_ids.add(int(cast(Any, row.target_item_id)))
        item_ids.update(coerce_item_ids(row.history_item_ids))
    return {item_id: index for index, item_id in enumerate(sorted(item_ids), start=2)}


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


def collate_generative_examples(examples: Sequence[GenerativeExample]) -> GenerativeBatch:
    """가변 길이 history를 padding하고 decoder input/target을 만든다."""
    if not examples:
        msg = "batch examples는 비어 있을 수 없습니다."
        raise ValueError(msg)
    return _collate_history_and_targets(
        [example.history_item_indices for example in examples],
        [example.target_token_ids for example in examples],
    )


def _collate_history_and_targets(
    history_indices: Sequence[Sequence[int]],
    target_token_ids: Sequence[Sequence[int]],
) -> GenerativeBatch:
    if not history_indices or not target_token_ids:
        msg = "batch는 비어 있을 수 없습니다."
        raise ValueError(msg)
    if len(history_indices) != len(target_token_ids):
        msg = "history와 target batch 크기가 같아야 합니다."
        raise ValueError(msg)

    batch_size = len(history_indices)
    max_history_length = max(max(len(history), 1) for history in history_indices)
    semantic_id_length = len(target_token_ids[0])
    if any(len(target) != semantic_id_length for target in target_token_ids):
        msg = "batch 안의 target semantic_id 길이가 모두 같아야 합니다."
        raise ValueError(msg)

    history_item_ids = torch.full(
        (batch_size, max_history_length),
        PAD_ITEM_INDEX,
        dtype=torch.long,
    )
    history_padding_mask = torch.ones((batch_size, max_history_length), dtype=torch.bool)
    target_token_tensor = torch.empty((batch_size, semantic_id_length), dtype=torch.long)
    decoder_input_ids = torch.empty((batch_size, semantic_id_length), dtype=torch.long)

    for row_index, (history, target_tokens) in enumerate(
        zip(history_indices, target_token_ids, strict=True)
    ):
        history_length = len(history)
        if history_length > 0:
            history_item_ids[row_index, :history_length] = torch.tensor(
                history,
                dtype=torch.long,
            )
            history_padding_mask[row_index, :history_length] = False

        target = torch.tensor(target_tokens, dtype=torch.long)
        target_token_tensor[row_index] = target
        decoder_input_ids[row_index, 0] = BOS_TOKEN_ID
        decoder_input_ids[row_index, 1:] = target[:-1]

    return GenerativeBatch(
        history_item_ids=history_item_ids,
        history_padding_mask=history_padding_mask,
        decoder_input_ids=decoder_input_ids,
        target_token_ids=target_token_tensor,
    )


def _record_batch_to_generative_arrays(
    record_batch: Any,
    *,
    item_index_lookup: np.ndarray,
    target_token_lookup: np.ndarray,
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
    max_history_length = max(int(lengths.max(initial=0)), 1)
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


def _map_history_to_indices(
    history: object,
    item_to_index: Mapping[int, int],
) -> tuple[int, ...]:
    return tuple(item_to_index.get(item_id, UNK_ITEM_INDEX) for item_id in coerce_item_ids(history))

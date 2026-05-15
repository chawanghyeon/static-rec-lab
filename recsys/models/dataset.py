"""Generative retrieval 학습용 Dataset과 collate 함수."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
import torch
from torch.utils.data import Dataset

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

        history = _normalize_history(row.history_item_ids)
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
        item_ids.update(_normalize_history(row.history_item_ids))
    return {item_id: index for index, item_id in enumerate(sorted(item_ids), start=2)}


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

    batch_size = len(examples)
    max_history_length = max(max(len(example.history_item_indices), 1) for example in examples)
    semantic_id_length = len(examples[0].target_token_ids)
    if any(len(example.target_token_ids) != semantic_id_length for example in examples):
        msg = "batch 안의 target semantic_id 길이가 모두 같아야 합니다."
        raise ValueError(msg)

    history_item_ids = torch.full(
        (batch_size, max_history_length),
        PAD_ITEM_INDEX,
        dtype=torch.long,
    )
    history_padding_mask = torch.ones((batch_size, max_history_length), dtype=torch.bool)
    target_token_ids = torch.empty((batch_size, semantic_id_length), dtype=torch.long)
    decoder_input_ids = torch.empty((batch_size, semantic_id_length), dtype=torch.long)

    for row_index, example in enumerate(examples):
        history_length = len(example.history_item_indices)
        if history_length > 0:
            history_item_ids[row_index, :history_length] = torch.tensor(
                example.history_item_indices,
                dtype=torch.long,
            )
            history_padding_mask[row_index, :history_length] = False

        target = torch.tensor(example.target_token_ids, dtype=torch.long)
        target_token_ids[row_index] = target
        decoder_input_ids[row_index, 0] = BOS_TOKEN_ID
        decoder_input_ids[row_index, 1:] = target[:-1]

    return GenerativeBatch(
        history_item_ids=history_item_ids,
        history_padding_mask=history_padding_mask,
        decoder_input_ids=decoder_input_ids,
        target_token_ids=target_token_ids,
    )


def _normalize_history(history: object) -> tuple[int, ...]:
    if history is None:
        return ()
    if isinstance(history, float) and pd.isna(history):
        return ()
    if isinstance(history, str):
        msg = "history_item_ids는 문자열이 아니라 정수 sequence여야 합니다."
        raise ValueError(msg)
    return tuple(int(cast(Any, item_id)) for item_id in cast(Iterable[object], history))

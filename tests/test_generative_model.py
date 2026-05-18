from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from torch.utils.data import DataLoader

from recsys.decoding import StaticDecodingIndex
from recsys.models import (
    BOS_TOKEN_ID,
    GenerativeBatch,
    GenerativeRetriever,
    GenerativeRetrieverConfig,
    build_generative_dataset,
    collate_generative_examples,
    evaluate_model,
    generate_semantic_ids_with_static_decoding,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)
from recsys.semantic_id import SemanticIdCodec


def test_build_generative_dataset_maps_history_and_semantic_tokens() -> None:
    frame = _sample_frame()
    codec = _sample_codec()

    bundle = build_generative_dataset(frame, codec)

    assert len(bundle.dataset) == 3
    assert bundle.semantic_id_length == 3
    assert bundle.semantic_vocab_size == 7
    assert bundle.item_vocab_size >= 6
    example = bundle.dataset[0]
    assert example.history_item_indices
    assert example.target_token_ids == (2, 3, 4)


def test_collate_generative_examples_pads_history_and_shifts_decoder_input() -> None:
    bundle = build_generative_dataset(_sample_frame(), _sample_codec())

    batch = collate_generative_examples([bundle.dataset[0], bundle.dataset[1]])

    assert batch.history_item_ids.shape == (2, 2)
    assert batch.history_padding_mask.shape == (2, 2)
    assert batch.decoder_input_ids[:, 0].tolist() == [BOS_TOKEN_ID, BOS_TOKEN_ID]
    assert torch.equal(batch.decoder_input_ids[:, 1:], batch.target_token_ids[:, :-1])


def test_generative_retriever_forward_shape() -> None:
    bundle = build_generative_dataset(_sample_frame(), _sample_codec())
    batch = collate_generative_examples([bundle.dataset[0], bundle.dataset[1]])
    model = _tiny_model(bundle.item_vocab_size, bundle.semantic_vocab_size)

    logits = model(
        batch.history_item_ids,
        batch.history_padding_mask,
        batch.decoder_input_ids,
    )

    assert logits.shape == (2, bundle.semantic_id_length, bundle.semantic_vocab_size)


def test_train_evaluate_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    bundle = build_generative_dataset(_sample_frame(), _sample_codec())
    model = _tiny_model(bundle.item_vocab_size, bundle.semantic_vocab_size)
    dataloader = cast(
        Iterable[GenerativeBatch],
        DataLoader(
            bundle.dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_generative_examples,
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    train_metrics = train_one_epoch(model, dataloader, optimizer, device=device)
    eval_metrics = evaluate_model(model, dataloader, device=device)
    checkpoint_path = save_checkpoint(
        tmp_path / "model.pt",
        model=model,
        item_to_index=bundle.item_to_index,
        metrics=eval_metrics,
    )
    loaded_model, item_to_index = load_checkpoint(checkpoint_path, device=device)

    assert train_metrics.num_examples == len(bundle.dataset)
    assert eval_metrics.loss > 0
    assert item_to_index == bundle.item_to_index
    assert isinstance(loaded_model, GenerativeRetriever)


def test_generate_semantic_ids_with_static_decoding_returns_valid_results() -> None:
    torch.manual_seed(7)
    codec = _sample_codec()
    bundle = build_generative_dataset(_sample_frame(), codec)
    model = _tiny_model(bundle.item_vocab_size, bundle.semantic_vocab_size)
    index = StaticDecodingIndex.from_codec(
        codec,
        vocab_size=bundle.semantic_vocab_size,
        dense_lookup_layers=2,
    )

    results = generate_semantic_ids_with_static_decoding(
        model=model,
        index=index,
        history_item_ids=[10, 20],
        item_to_index=bundle.item_to_index,
        beam_size=3,
        max_results=2,
        device=torch.device("cpu"),
    )

    assert 1 <= len(results) <= 2
    for result in results:
        assert codec.has_semantic_id(result.semantic_id)
        assert index.contains(result.semantic_id)
        assert 0 <= result.score <= 1


def _tiny_model(item_vocab_size: int, semantic_vocab_size: int) -> GenerativeRetriever:
    return GenerativeRetriever(
        GenerativeRetrieverConfig(
            item_vocab_size=item_vocab_size,
            semantic_vocab_size=semantic_vocab_size,
            semantic_id_length=3,
            max_history_length=4,
            d_model=16,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
    )


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "history_item_ids": [[10], [10, 20], [30], [30, 40]],
            "target_item_id": [20, 30, 40, 999],
            "target_timestamp": [1, 2, 3, 4],
            "history_length": [1, 2, 1, 2],
        }
    )


def _sample_codec() -> SemanticIdCodec:
    return SemanticIdCodec(
        {
            20: [0, 1, 2],
            30: [1, 2, 3],
            40: [2, 3, 4],
        }
    )

from pathlib import Path

import pandas as pd
import torch

from recsys.decoding import StaticDecodingIndex
from recsys.models import (
    BOS_TOKEN_ID,
    GenerativeParquetBatchIterableDataset,
    GenerativeRetriever,
    GenerativeRetrieverConfig,
    build_item_index_from_codec,
    build_item_index_from_parquet,
    compute_target_coverage_from_parquet,
    evaluate_model,
    generate_semantic_ids_batch_with_static_decoding,
    generate_semantic_ids_with_static_decoding,
    infer_semantic_vocab_size,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)
from recsys.semantic_id import SemanticIdCodec


def test_generative_parquet_batch_iterable_dataset_streams_batches(tmp_path: Path) -> None:
    codec = _sample_codec()
    dataset = _sample_dataset(tmp_path, codec, batch_size=2, parquet_batch_size=2)

    batches = list(dataset)

    assert len(batches) == 2
    assert batches[0].history_item_ids.shape[0] == 2
    assert batches[0].history_feedback_ids.shape[0] == 2
    assert batches[0].target_token_ids.shape == (2, 3)
    assert batches[0].decoder_input_ids[:, 0].tolist() == [BOS_TOKEN_ID, BOS_TOKEN_ID]
    assert torch.equal(batches[0].decoder_input_ids[:, 1:], batches[0].target_token_ids[:, :-1])
    assert batches[0].history_item_ids[0, 0].item() == 1
    assert batches[0].history_feedback_ids[0, 0].item() == 3
    assert batches[0].target_token_ids[0].tolist() == [2, 3, 4]
    assert batches[1].history_item_ids.shape[0] == 1


def test_generative_parquet_batch_iterable_dataset_filters_unknown_targets(
    tmp_path: Path,
) -> None:
    codec = _sample_codec()
    dataset = _sample_dataset(tmp_path, codec, batch_size=8, parquet_batch_size=4)

    batch = next(iter(dataset))

    assert batch.target_token_ids.shape[0] == 3


def test_compute_target_coverage_from_parquet_counts_unknown_targets(
    tmp_path: Path,
) -> None:
    codec = _sample_codec()

    coverage = compute_target_coverage_from_parquet(
        _write_sample_parquet(tmp_path),
        codec,
        parquet_batch_size=2,
    )

    assert coverage.total_examples == 4
    assert coverage.known_target_examples == 3
    assert coverage.unknown_target_examples == 1
    assert coverage.unknown_target_rate == 0.25


def test_compute_target_coverage_from_parquet_respects_known_target_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "coverage.parquet"
    pd.DataFrame({"target_item_id": [999, 20, 998, 30, 40]}).to_parquet(
        path,
        index=False,
    )
    codec = _sample_codec()

    coverage = compute_target_coverage_from_parquet(
        path,
        codec,
        max_known_examples=2,
        parquet_batch_size=5,
    )

    assert coverage.total_examples == 4
    assert coverage.known_target_examples == 2
    assert coverage.unknown_target_examples == 2


def test_build_item_index_from_parquet_includes_full_history_items(tmp_path: Path) -> None:
    codec = _sample_codec()

    item_to_index = build_item_index_from_parquet(_write_sample_parquet(tmp_path), codec=codec)

    assert 10 in item_to_index
    assert 999 in item_to_index
    assert set(codec.item_to_semantic_id).issubset(item_to_index)


def test_generative_parquet_batch_iterable_dataset_respects_max_examples(
    tmp_path: Path,
) -> None:
    codec = _sample_codec()
    dataset = _sample_dataset(tmp_path, codec, batch_size=8, max_examples=2)

    batch = next(iter(dataset))

    assert batch.target_token_ids.shape[0] == 2


def test_generative_parquet_batch_iterable_dataset_supports_fixed_history_length(
    tmp_path: Path,
) -> None:
    codec = _sample_codec()
    dataset = _sample_dataset(tmp_path, codec, batch_size=2, fixed_history_length=4)

    batch = next(iter(dataset))

    assert batch.history_item_ids.shape == (2, 4)
    assert batch.history_feedback_ids.shape == (2, 4)
    assert batch.history_padding_mask.shape == (2, 4)
    assert batch.history_padding_mask[0].tolist() == [False, True, True, True]
    assert batch.history_padding_mask[1].tolist() == [False, False, True, True]
    assert batch.history_feedback_ids[0].tolist() == [3, 0, 0, 0]
    assert batch.history_feedback_ids[1].tolist() == [3, 3, 0, 0]


def test_generative_retriever_forward_shape(tmp_path: Path) -> None:
    codec = _sample_codec()
    batch = next(iter(_sample_dataset(tmp_path, codec, batch_size=2)))
    model = _tiny_model(_item_vocab_size(codec), infer_semantic_vocab_size(codec))

    logits = model(
        batch.history_item_ids,
        batch.history_feedback_ids,
        batch.history_padding_mask,
        batch.decoder_input_ids,
    )

    assert logits.shape == (2, codec.semantic_id_length, infer_semantic_vocab_size(codec))


def test_train_evaluate_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    codec = _sample_codec()
    item_to_index = build_item_index_from_codec(codec)
    model = _tiny_model(_item_vocab_size(codec), infer_semantic_vocab_size(codec))
    train_dataset = _sample_dataset(tmp_path, codec, batch_size=2)
    eval_dataset = _sample_dataset(tmp_path, codec, batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    train_metrics = train_one_epoch(model, train_dataset, optimizer, device=device)
    eval_metrics = evaluate_model(model, eval_dataset, device=device)
    checkpoint_path = save_checkpoint(
        tmp_path / "model.pt",
        model=model,
        item_to_index=item_to_index,
        metrics=eval_metrics,
    )
    loaded_model, loaded_item_to_index = load_checkpoint(checkpoint_path, device=device)

    assert train_metrics.num_examples == 3
    assert eval_metrics.loss > 0
    assert loaded_item_to_index == item_to_index
    assert isinstance(loaded_model, GenerativeRetriever)


def test_generate_semantic_ids_with_static_decoding_returns_valid_results() -> None:
    torch.manual_seed(7)
    codec = _sample_codec()
    item_to_index = build_item_index_from_codec(codec)
    semantic_vocab_size = infer_semantic_vocab_size(codec)
    model = _tiny_model(_item_vocab_size(codec), semantic_vocab_size)
    index = StaticDecodingIndex.from_codec(
        codec,
        vocab_size=semantic_vocab_size,
        dense_lookup_layers=2,
    )

    results = generate_semantic_ids_with_static_decoding(
        model=model,
        index=index,
        history_item_ids=[10, 20],
        item_to_index=item_to_index,
        beam_size=3,
        max_results=2,
        device=torch.device("cpu"),
    )

    assert 1 <= len(results) <= 2
    for result in results:
        assert codec.has_semantic_id(result.semantic_id)
        assert index.contains(result.semantic_id)
        assert 0 <= result.score <= 1


def test_generate_semantic_ids_batch_with_static_decoding_matches_single_results() -> None:
    torch.manual_seed(7)
    codec = _sample_codec()
    item_to_index = build_item_index_from_codec(codec)
    semantic_vocab_size = infer_semantic_vocab_size(codec)
    model = _tiny_model(_item_vocab_size(codec), semantic_vocab_size)
    index = StaticDecodingIndex.from_codec(
        codec,
        vocab_size=semantic_vocab_size,
        dense_lookup_layers=2,
    )
    histories = ([10, 20], [30, 40])

    single_results = [
        generate_semantic_ids_with_static_decoding(
            model=model,
            index=index,
            history_item_ids=history,
            item_to_index=item_to_index,
            beam_size=3,
            max_results=2,
            device=torch.device("cpu"),
        )
        for history in histories
    ]
    batch_results = generate_semantic_ids_batch_with_static_decoding(
        model=model,
        index=index,
        history_item_ids_batch=histories,
        item_to_index=item_to_index,
        beam_size=3,
        max_results=2,
        device=torch.device("cpu"),
    )

    assert [
        [beam_result.semantic_id for beam_result in row_results] for row_results in batch_results
    ] == [
        [beam_result.semantic_id for beam_result in row_results] for row_results in single_results
    ]


def _sample_dataset(
    tmp_path: Path,
    codec: SemanticIdCodec,
    *,
    batch_size: int,
    max_examples: int | None = None,
    parquet_batch_size: int = 2,
    fixed_history_length: int | None = None,
) -> GenerativeParquetBatchIterableDataset:
    return GenerativeParquetBatchIterableDataset(
        _write_sample_parquet(tmp_path),
        codec,
        item_to_index=build_item_index_from_codec(codec),
        batch_size=batch_size,
        max_examples=max_examples,
        parquet_batch_size=parquet_batch_size,
        fixed_history_length=fixed_history_length,
    )


def _write_sample_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "train.parquet"
    _sample_frame().to_parquet(path, index=False)
    return path


def _item_vocab_size(codec: SemanticIdCodec) -> int:
    item_to_index = build_item_index_from_codec(codec)
    return max(item_to_index.values(), default=1) + 1


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
            "history_feedback_ids": [[3], [3, 3], [1], [1, 3]],
            "positive_history_item_ids": [[10], [10, 20], [], [40]],
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

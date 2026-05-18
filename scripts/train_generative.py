"""Generative retrieval model 학습 CLI."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from recsys.models import (
    GenerativeBatch,
    GenerativeExample,
    GenerativeParquetBatchIterableDataset,
    GenerativeRetriever,
    GenerativeRetrieverConfig,
    build_generative_dataset,
    build_item_index_from_codec,
    collate_generative_examples,
    evaluate_model,
    infer_semantic_vocab_size,
    save_checkpoint,
    train_one_epoch,
)
from recsys.semantic_id import SemanticIdCodec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generative retrieval model 학습")
    parser.add_argument("--train-parquet", type=Path, default=Path("data/processed/train.parquet"))
    parser.add_argument("--valid-parquet", type=Path, default=Path("data/processed/valid.parquet"))
    parser.add_argument(
        "--semantic-id-path",
        type=Path,
        default=Path("artifacts/semantic_id/semantic_ids.json"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/generative/model.pt"),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-valid-examples", type=int, default=None)
    parser.add_argument("--max-history-length", type=int, default=50)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="대용량 parquet를 메모리에 모두 올리지 않고 batch 단위로 읽습니다.",
    )
    parser.add_argument("--parquet-batch-size", type=int, default=65_536)
    parser.add_argument("--log-every-batches", type=int, default=250)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-decoder-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs는 1 이상이어야 합니다.")
    if args.batch_size < 1:
        raise ValueError("batch-size는 1 이상이어야 합니다.")

    torch.manual_seed(args.random_seed)
    device = _resolve_device(args.device)
    codec = SemanticIdCodec.load_json(args.semantic_id_path)
    train_loader: Iterable[GenerativeBatch]
    valid_loader: Iterable[GenerativeBatch]
    if args.streaming:
        item_to_index = build_item_index_from_codec(codec)
        train_loader = GenerativeParquetBatchIterableDataset(
            args.train_parquet,
            codec,
            item_to_index=item_to_index,
            batch_size=args.batch_size,
            max_examples=args.max_train_examples,
            parquet_batch_size=args.parquet_batch_size,
        )
        valid_loader = GenerativeParquetBatchIterableDataset(
            args.valid_parquet,
            codec,
            item_to_index=item_to_index,
            batch_size=args.batch_size,
            max_examples=args.max_valid_examples,
            parquet_batch_size=args.parquet_batch_size,
        )
        item_vocab_size = max(item_to_index.values(), default=1) + 1
        semantic_vocab_size = infer_semantic_vocab_size(codec)
        semantic_id_length = codec.semantic_id_length
    else:
        train_frame = pd.read_parquet(args.train_parquet)
        valid_frame = pd.read_parquet(args.valid_parquet)
        train_bundle = build_generative_dataset(
            train_frame,
            codec,
            max_examples=args.max_train_examples,
        )
        valid_bundle = build_generative_dataset(
            valid_frame,
            codec,
            item_to_index=train_bundle.item_to_index,
            max_examples=args.max_valid_examples,
        )
        train_dataset: Dataset[GenerativeExample] | IterableDataset[GenerativeExample]
        valid_dataset: Dataset[GenerativeExample] | IterableDataset[GenerativeExample]
        train_dataset = train_bundle.dataset
        valid_dataset = valid_bundle.dataset
        item_to_index = train_bundle.item_to_index
        item_vocab_size = train_bundle.item_vocab_size
        semantic_vocab_size = train_bundle.semantic_vocab_size
        semantic_id_length = train_bundle.semantic_id_length
        train_loader = cast(
            Iterable[GenerativeBatch],
            DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_generative_examples,
            ),
        )
        valid_loader = cast(
            Iterable[GenerativeBatch],
            DataLoader(
                valid_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_generative_examples,
            ),
        )

    model = GenerativeRetriever(
        GenerativeRetrieverConfig(
            item_vocab_size=item_vocab_size,
            semantic_vocab_size=semantic_vocab_size,
            semantic_id_length=semantic_id_length,
            max_history_length=args.max_history_length,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_metrics = None
    valid_metrics = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            log_every_batches=args.log_every_batches,
        )
        valid_metrics = evaluate_model(
            model,
            valid_loader,
            device=device,
            log_every_batches=args.log_every_batches,
        )
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics.loss:.6f} "
            f"valid_loss={valid_metrics.loss:.6f} "
            f"valid_token_acc={valid_metrics.token_accuracy:.6f} "
            f"valid_sequence_acc={valid_metrics.sequence_accuracy:.6f}"
        )

    if valid_metrics is None:
        raise RuntimeError("학습 metric이 생성되지 않았습니다.")
    checkpoint_path = save_checkpoint(
        args.output_path,
        model=model,
        item_to_index=item_to_index,
        metrics=valid_metrics,
    )

    print("Generative retrieval 학습 완료")
    print(f"- train_examples: {train_metrics.num_examples if train_metrics is not None else 0}")
    print(f"- valid_examples: {valid_metrics.num_examples}")
    print(f"- item_vocab_size: {item_vocab_size}")
    print(f"- semantic_vocab_size: {semantic_vocab_size}")
    print(f"- checkpoint: {checkpoint_path}")


def _resolve_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        return torch.device("cuda")
    if value == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()

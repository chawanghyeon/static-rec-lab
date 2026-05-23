"""Generative Retrieval 학습/평가 CLI."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from recsys.evaluation import (
    STATIC_DECODING_DECODER_NAME,
    GenerativeRankingEvaluation,
    evaluate_generative_ranking_from_parquet,
    write_generative_ranking_report,
)
from recsys.models import (
    GenerativeBatch,
    GenerativeParquetBatchIterableDataset,
    GenerativeRetriever,
    GenerativeRetrieverConfig,
    GenerativeTrainingMetrics,
    build_item_index_from_codec,
    evaluate_model,
    infer_semantic_vocab_size,
    load_checkpoint,
    resolve_torch_device,
    save_checkpoint,
    train_one_epoch,
)
from recsys.semantic_id import SemanticIdCodec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generative Retrieval 학습/평가")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(subparsers)
    _add_eval_parser(subparsers)
    _add_eval_ranking_parser(subparsers)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_generative(args)
    elif args.command == "eval":
        eval_generative(args)
    elif args.command == "eval-ranking":
        eval_generative_ranking(args)
    else:  # pragma: no cover
        raise ValueError(f"알 수 없는 generative command입니다: {args.command}")


def _add_train_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("train", help="Generative Retrieval 모델 학습")
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
    parser.add_argument(
        "--eval-every-epochs",
        type=int,
        default=1,
        help="N epoch마다 validation을 실행합니다. 0이면 마지막 epoch 뒤에 한 번만 평가합니다.",
    )
    parser.add_argument(
        "--skip-valid-accuracy",
        action="store_true",
        help=(
            "학습 중 validation에서는 loss만 계산합니다. 최종 accuracy는 eval command로 계산합니다."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-valid-examples", type=int, default=None)
    parser.add_argument("--max-history-length", type=int, default=50)
    parser.add_argument("--parquet-batch-size", type=int, default=65_536)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-every-batches", type=int, default=250)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-decoder-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="torch.compile로 학습 모델을 컴파일합니다.",
    )
    parser.add_argument(
        "--full-train-metrics",
        action="store_true",
        help="학습 batch에서도 token/sequence accuracy를 계산합니다. 속도는 느려질 수 있습니다.",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default="high",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="가능한 device에서 mixed precision autocast를 사용합니다.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--random-seed", type=int, default=42)


def _add_eval_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("eval", help="Teacher-forcing 평가")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("artifacts/generative/model.pt"),
    )
    parser.add_argument("--eval-parquet", type=Path, default=Path("data/processed/valid.parquet"))
    parser.add_argument(
        "--semantic-id-path",
        type=Path,
        default=Path("artifacts/semantic_id/semantic_ids.json"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/local/generative.md"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--parquet-batch-size", type=int, default=65_536)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="가능한 device에서 mixed precision autocast를 사용합니다.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="float16",
    )


def _add_eval_ranking_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("eval-ranking", help="추천 ranking 평가")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("artifacts/generative/model.pt"),
        help="학습된 generative model checkpoint 경로",
    )
    parser.add_argument(
        "--semantic-id-path",
        type=Path,
        default=Path("artifacts/semantic_id/semantic_ids.json"),
        help="Semantic ID codec JSON 경로",
    )
    parser.add_argument("--valid-parquet", type=Path, default=Path("data/processed/valid.parquet"))
    parser.add_argument("--test-parquet", type=Path, default=Path("data/processed/test.parquet"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/local/generative_eval.md"),
        help="추천 ranking 평가 리포트 저장 경로",
    )
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--beam-size", type=int, default=50)
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=128,
        help="추천 ranking 생성 시 한 번에 처리할 query 수",
    )
    parser.add_argument(
        "--static-decoding-index-path",
        type=Path,
        default=None,
        help="static_decoding index npz artifact 경로",
    )
    parser.add_argument("--max-valid-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--parquet-batch-size", type=int, default=65_536)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")


def train_generative(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("epochs는 1 이상이어야 합니다.")
    if args.eval_every_epochs < 0:
        raise ValueError("eval-every-epochs는 0 이상이어야 합니다.")
    if args.batch_size < 1:
        raise ValueError("batch-size는 1 이상이어야 합니다.")
    _validate_loader_options(args.num_workers, args.prefetch_factor)
    torch.set_float32_matmul_precision(args.matmul_precision)
    amp_dtype = _resolve_amp_dtype(args.amp_dtype)

    torch.manual_seed(args.random_seed)
    device = resolve_torch_device(args.device)
    codec = SemanticIdCodec.load_json(args.semantic_id_path)
    item_to_index = build_item_index_from_codec(codec)
    train_dataset = GenerativeParquetBatchIterableDataset(
        args.train_parquet,
        codec,
        item_to_index=item_to_index,
        batch_size=args.batch_size,
        max_examples=args.max_train_examples,
        parquet_batch_size=args.parquet_batch_size,
        fixed_history_length=args.max_history_length,
    )
    valid_dataset = GenerativeParquetBatchIterableDataset(
        args.valid_parquet,
        codec,
        item_to_index=item_to_index,
        batch_size=args.batch_size,
        max_examples=args.max_valid_examples,
        parquet_batch_size=args.parquet_batch_size,
        fixed_history_length=args.max_history_length,
    )
    train_loader = _batch_stream(
        train_dataset,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        device=device,
    )
    valid_loader = _batch_stream(
        valid_dataset,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        device=device,
    )
    item_vocab_size = max(item_to_index.values(), default=1) + 1
    semantic_vocab_size = infer_semantic_vocab_size(codec)
    semantic_id_length = codec.semantic_id_length

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
    train_model = (
        cast(torch.nn.Module, cast(Any, torch.compile)(model)) if args.compile_model else model
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_metrics = None
    valid_metrics = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            train_model,
            train_loader,
            optimizer,
            device=device,
            log_every_batches=args.log_every_batches,
            compute_accuracy=args.full_train_metrics,
            use_amp=args.amp,
            amp_dtype=amp_dtype,
        )
        should_eval = _should_evaluate_epoch(
            epoch=epoch,
            epochs=args.epochs,
            eval_every_epochs=args.eval_every_epochs,
        )
        if should_eval:
            valid_metrics = evaluate_model(
                train_model,
                valid_loader,
                device=device,
                log_every_batches=args.log_every_batches,
                use_amp=args.amp,
                amp_dtype=amp_dtype,
                compute_accuracy=not args.skip_valid_accuracy,
            )
            valid_summary = _format_validation_summary(
                valid_metrics,
                include_accuracy=not args.skip_valid_accuracy,
            )
        else:
            valid_summary = "valid=skipped"
        print(f"epoch={epoch} train_loss={train_metrics.loss:.6f} {valid_summary}")

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


def eval_generative(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch-size는 1 이상이어야 합니다.")
    _validate_loader_options(args.num_workers, args.prefetch_factor)

    device = resolve_torch_device(args.device)
    amp_dtype = _resolve_amp_dtype(args.amp_dtype)
    model, item_to_index = load_checkpoint(args.checkpoint_path, device=device)
    codec = SemanticIdCodec.load_json(args.semantic_id_path)
    dataset = GenerativeParquetBatchIterableDataset(
        args.eval_parquet,
        codec,
        item_to_index=item_to_index,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        parquet_batch_size=args.parquet_batch_size,
        fixed_history_length=model.config.max_history_length,
    )
    dataloader = _batch_stream(
        dataset,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        device=device,
    )
    metrics = evaluate_model(
        model,
        dataloader,
        device=device,
        use_amp=args.amp,
        amp_dtype=amp_dtype,
    )
    report_path = _write_report(args.report_path, metrics, args.eval_parquet)

    print("Generative retrieval 평가 완료")
    print(f"- examples: {metrics.num_examples}")
    print(f"- loss: {metrics.loss:.6f}")
    print(f"- token_accuracy: {metrics.token_accuracy:.6f}")
    print(f"- sequence_accuracy: {metrics.sequence_accuracy:.6f}")
    print(f"- report: {report_path}")


def eval_generative_ranking(args: argparse.Namespace) -> None:
    if args.beam_size < 1:
        raise ValueError("beam-size는 1 이상이어야 합니다.")
    if args.inference_batch_size < 1:
        raise ValueError("inference-batch-size는 1 이상이어야 합니다.")

    device = resolve_torch_device(args.device)
    model, item_to_index = load_checkpoint(args.checkpoint_path, device=device)
    codec = SemanticIdCodec.load_json(args.semantic_id_path)

    evaluations = [
        evaluate_generative_ranking_from_parquet(
            model=model,
            item_to_index=item_to_index,
            codec=codec,
            eval_parquet=args.valid_parquet,
            split_name="valid",
            cutoffs=args.cutoffs,
            beam_size=args.beam_size,
            device=device,
            static_decoding_index_path=args.static_decoding_index_path,
            inference_batch_size=args.inference_batch_size,
            max_examples=args.max_valid_examples,
            parquet_batch_size=args.parquet_batch_size,
        ),
        evaluate_generative_ranking_from_parquet(
            model=model,
            item_to_index=item_to_index,
            codec=codec,
            eval_parquet=args.test_parquet,
            split_name="test",
            cutoffs=args.cutoffs,
            beam_size=args.beam_size,
            device=device,
            static_decoding_index_path=args.static_decoding_index_path,
            inference_batch_size=args.inference_batch_size,
            max_examples=args.max_test_examples,
            parquet_batch_size=args.parquet_batch_size,
        ),
    ]
    report_path = write_generative_ranking_report(
        args.report_path,
        evaluations,
        checkpoint_path=args.checkpoint_path,
        semantic_id_path=args.semantic_id_path,
        beam_size=args.beam_size,
        cutoffs=args.cutoffs,
        inference_batch_size=args.inference_batch_size,
        decoder_name=STATIC_DECODING_DECODER_NAME,
        static_decoding_index_path=args.static_decoding_index_path,
    )

    print("Generative Retrieval 추천 ranking 평가 완료")
    for evaluation in evaluations:
        _print_evaluation(evaluation)
    print(f"- report: {report_path}")


def _write_report(
    path: Path,
    metrics: GenerativeTrainingMetrics,
    eval_parquet: Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "# Generative Retrieval 평가 리포트",
                "",
                f"- 평가 데이터: `{eval_parquet}`",
                f"- examples: {metrics.num_examples:,}",
                f"- validation loss: {metrics.loss:.6f}",
                f"- token accuracy: {metrics.token_accuracy:.6f}",
                f"- sequence accuracy: {metrics.sequence_accuracy:.6f}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_path


def _print_evaluation(evaluation: GenerativeRankingEvaluation) -> None:
    for k, metrics in sorted(evaluation.metrics_by_k.items()):
        print(
            f"- {evaluation.split_name}@{k}: "
            f"Recall={metrics.recall:.6f}, "
            f"NDCG={metrics.ndcg:.6f}, "
            f"MRR={metrics.mrr:.6f}"
        )
    print(
        f"- {evaluation.split_name} invalid_generation_rate="
        f"{evaluation.invalid_generation_rate:.6f}, "
        f"unknown_targets={evaluation.unknown_target_examples}, "
        f"elapsed_ms={evaluation.elapsed_ms:.2f}"
    )


def _format_validation_summary(
    metrics: GenerativeTrainingMetrics,
    *,
    include_accuracy: bool,
) -> str:
    if not include_accuracy:
        return f"valid_loss={metrics.loss:.6f} valid_accuracy=skipped"
    return (
        f"valid_loss={metrics.loss:.6f} "
        f"valid_token_acc={metrics.token_accuracy:.6f} "
        f"valid_sequence_acc={metrics.sequence_accuracy:.6f}"
    )


def _batch_stream(
    dataset: GenerativeParquetBatchIterableDataset,
    *,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
) -> Iterable[GenerativeBatch]:
    if num_workers == 0:
        return dataset
    return cast(
        Iterable[GenerativeBatch],
        DataLoader(
            dataset,
            batch_size=None,
            collate_fn=cast(Any, _identity_batch),
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
            pin_memory=device.type == "cuda",
        ),
    )


def _identity_batch(batch: GenerativeBatch) -> GenerativeBatch:
    return batch


def _validate_loader_options(num_workers: int, prefetch_factor: int) -> None:
    if num_workers < 0:
        msg = "num-workers는 0 이상이어야 합니다."
        raise ValueError(msg)
    if prefetch_factor < 1:
        msg = "prefetch-factor는 1 이상이어야 합니다."
        raise ValueError(msg)


def _resolve_amp_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"지원하지 않는 amp dtype입니다: {name}")


def _should_evaluate_epoch(*, epoch: int, epochs: int, eval_every_epochs: int) -> bool:
    if epoch == epochs:
        return True
    if eval_every_epochs == 0:
        return False
    return epoch % eval_every_epochs == 0


if __name__ == "__main__":
    main()

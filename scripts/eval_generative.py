"""Generative retrieval model 평가 CLI."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from torch.utils.data import DataLoader

from recsys.models import (
    GenerativeBatch,
    GenerativeTrainingMetrics,
    build_generative_dataset,
    collate_generative_examples,
    evaluate_model,
    load_checkpoint,
)
from recsys.semantic_id import SemanticIdCodec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generative retrieval model 평가")
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
        default=Path("reports/generative.md"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size는 1 이상이어야 합니다.")

    device = _resolve_device(args.device)
    model, item_to_index = load_checkpoint(args.checkpoint_path, device=device)
    codec = SemanticIdCodec.load_json(args.semantic_id_path)
    eval_frame = pd.read_parquet(args.eval_parquet)
    bundle = build_generative_dataset(
        eval_frame,
        codec,
        item_to_index=item_to_index,
        max_examples=args.max_examples,
    )
    dataloader = cast(
        Iterable[GenerativeBatch],
        DataLoader(
            bundle.dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_generative_examples,
        ),
    )
    metrics = evaluate_model(model, dataloader, device=device)
    report_path = _write_report(args.report_path, metrics, args.eval_parquet)

    print("Generative retrieval 평가 완료")
    print(f"- examples: {metrics.num_examples}")
    print(f"- loss: {metrics.loss:.6f}")
    print(f"- token_accuracy: {metrics.token_accuracy:.6f}")
    print(f"- sequence_accuracy: {metrics.sequence_accuracy:.6f}")
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

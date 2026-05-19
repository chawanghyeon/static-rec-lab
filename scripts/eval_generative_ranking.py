"""Generative Retrieval 추천 ranking 평가 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from recsys.evaluation import (
    STATIC_DECODING_DECODER_NAME,
    GenerativeRankingEvaluation,
    evaluate_generative_ranking,
    write_generative_ranking_report,
)
from recsys.models import load_checkpoint
from recsys.semantic_id import SemanticIdCodec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generative Retrieval 추천 ranking 평가")
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
        default=Path("reports/generative_eval.md"),
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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.beam_size < 1:
        raise ValueError("beam-size는 1 이상이어야 합니다.")
    if args.inference_batch_size < 1:
        raise ValueError("inference-batch-size는 1 이상이어야 합니다.")

    device = _resolve_device(args.device)
    model, item_to_index = load_checkpoint(args.checkpoint_path, device=device)
    codec = SemanticIdCodec.load_json(args.semantic_id_path)
    valid_frame = _read_eval_frame(args.valid_parquet, args.max_valid_examples)
    test_frame = _read_eval_frame(args.test_parquet, args.max_test_examples)

    evaluations = [
        evaluate_generative_ranking(
            model=model,
            item_to_index=item_to_index,
            codec=codec,
            eval_frame=valid_frame,
            split_name="valid",
            cutoffs=args.cutoffs,
            beam_size=args.beam_size,
            device=device,
            static_decoding_index_path=args.static_decoding_index_path,
            inference_batch_size=args.inference_batch_size,
        ),
        evaluate_generative_ranking(
            model=model,
            item_to_index=item_to_index,
            codec=codec,
            eval_frame=test_frame,
            split_name="test",
            cutoffs=args.cutoffs,
            beam_size=args.beam_size,
            device=device,
            static_decoding_index_path=args.static_decoding_index_path,
            inference_batch_size=args.inference_batch_size,
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


def _read_eval_frame(path: Path, max_examples: int | None) -> pd.DataFrame:
    if max_examples is not None and max_examples < 1:
        raise ValueError("max examples는 None이거나 1 이상이어야 합니다.")
    frame = pd.read_parquet(path)
    if max_examples is None:
        return frame
    return frame.head(max_examples)


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

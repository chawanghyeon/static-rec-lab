"""Generative Retrieval 추천 ranking 평가."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import pandas as pd
import torch

from recsys.data import coerce_item_ids
from recsys.decoding import (
    STATIC_DECODING_DECODER_NAME,
    StaticDecodingIndex,
    StaticDecodingTorchIndex,
    load_or_build_static_decoding_index,
)
from recsys.evaluation.metrics import RankingMetrics, evaluate_ranking_at_k
from recsys.models import (
    GenerativeRetriever,
    generate_semantic_ids_batch_with_static_decoding,
    generate_semantic_ids_with_static_decoding,
)
from recsys.semantic_id import SemanticIdCodec, UnknownSemanticIdError


@dataclass(frozen=True)
class GenerativeRecommendationResult:
    """한 query에 대한 constrained generation 기반 추천 결과."""

    item_ids: tuple[int, ...]
    generated_sequences: int
    invalid_sequences: int
    history_filtered_items: int
    duplicate_items: int


@dataclass(frozen=True)
class GenerativeRankingEvaluation:
    """한 split의 Generative Retrieval 추천 ranking 평가 결과."""

    split_name: str
    metrics_by_k: dict[int, RankingMetrics]
    num_examples: int
    unknown_target_examples: int
    generated_sequences: int
    invalid_sequences: int
    history_filtered_items: int
    duplicate_items: int
    avg_recommendations: float
    elapsed_ms: float

    @property
    def invalid_generation_rate(self) -> float:
        """생성된 sequence 중 codec으로 decode할 수 없는 sequence 비율."""
        if self.generated_sequences == 0:
            return 0.0
        return self.invalid_sequences / self.generated_sequences


def evaluate_generative_ranking(
    *,
    model: GenerativeRetriever,
    item_to_index: Mapping[int, int],
    codec: SemanticIdCodec,
    eval_frame: pd.DataFrame,
    split_name: str,
    cutoffs: Sequence[int],
    beam_size: int,
    device: torch.device,
    static_decoding_index_path: str | Path | None = None,
    inference_batch_size: int = 1,
) -> GenerativeRankingEvaluation:
    """Generative model + STATIC decoder를 baseline과 같은 ranking metric으로 평가한다."""
    _validate_eval_frame(eval_frame)
    _validate_cutoffs(cutoffs)
    if beam_size < 1:
        msg = "beam_size는 1 이상이어야 합니다."
        raise ValueError(msg)
    if inference_batch_size < 1:
        msg = "inference_batch_size는 1 이상이어야 합니다."
        raise ValueError(msg)

    max_k = max(cutoffs)
    effective_beam_size = max(beam_size, max_k)
    decoder = load_or_build_static_decoding_index(
        codec=codec,
        static_decoding_index_path=static_decoding_index_path,
    )
    static_decoding_torch_index = decoder.to_torch(device)
    recommendations: list[tuple[int, ...]] = []
    relevant_items: list[tuple[int]] = []
    unknown_target_examples = 0
    generated_sequences = 0
    invalid_sequences = 0
    history_filtered_items = 0
    duplicate_items = 0
    started_at = perf_counter()

    rows = list(eval_frame.itertuples(index=False))
    for start in range(0, len(rows), inference_batch_size):
        row_batch = rows[start : start + inference_batch_size]
        histories = [coerce_item_ids(row.history_item_ids) for row in row_batch]
        target_item_ids = [int(cast(Any, row.target_item_id)) for row in row_batch]
        unknown_target_examples += sum(
            1 for target_item_id in target_item_ids if not codec.has_item(target_item_id)
        )

        if inference_batch_size == 1:
            batch_results: tuple[GenerativeRecommendationResult, ...] = (
                recommend_with_constrained_generation(
                    model=model,
                    decoder=decoder,
                    codec=codec,
                    history_item_ids=histories[0],
                    item_to_index=item_to_index,
                    k=max_k,
                    beam_size=effective_beam_size,
                    device=device,
                    static_decoding_torch_index=static_decoding_torch_index,
                ),
            )
        else:
            batch_results = recommend_batch_with_constrained_generation(
                model=model,
                decoder=decoder,
                codec=codec,
                history_item_ids_batch=histories,
                item_to_index=item_to_index,
                k=max_k,
                beam_size=effective_beam_size,
                device=device,
                static_decoding_torch_index=static_decoding_torch_index,
            )

        for target_item_id, result in zip(target_item_ids, batch_results, strict=True):
            recommendations.append(result.item_ids)
            relevant_items.append((target_item_id,))
            generated_sequences += result.generated_sequences
            invalid_sequences += result.invalid_sequences
            history_filtered_items += result.history_filtered_items
            duplicate_items += result.duplicate_items

    elapsed_ms = (perf_counter() - started_at) * 1000
    metrics_by_k = {
        cutoff: evaluate_ranking_at_k(recommendations, relevant_items, cutoff) for cutoff in cutoffs
    }
    num_examples = len(recommendations)
    total_recommendations = sum(len(item_ids) for item_ids in recommendations)

    return GenerativeRankingEvaluation(
        split_name=split_name,
        metrics_by_k=metrics_by_k,
        num_examples=num_examples,
        unknown_target_examples=unknown_target_examples,
        generated_sequences=generated_sequences,
        invalid_sequences=invalid_sequences,
        history_filtered_items=history_filtered_items,
        duplicate_items=duplicate_items,
        avg_recommendations=total_recommendations / max(num_examples, 1),
        elapsed_ms=elapsed_ms,
    )


def recommend_with_constrained_generation(
    *,
    model: GenerativeRetriever,
    decoder: StaticDecodingIndex,
    codec: SemanticIdCodec,
    history_item_ids: Sequence[int],
    item_to_index: Mapping[int, int],
    k: int,
    beam_size: int,
    device: torch.device,
    static_decoding_torch_index: StaticDecodingTorchIndex | None = None,
) -> GenerativeRecommendationResult:
    """STATIC constrained beam search 결과를 item ranking으로 변환한다."""
    if k < 1:
        msg = "k는 1 이상이어야 합니다."
        raise ValueError(msg)
    if beam_size < 1:
        msg = "beam_size는 1 이상이어야 합니다."
        raise ValueError(msg)

    beam_results = generate_semantic_ids_with_static_decoding(
        model=model,
        index=decoder,
        torch_index=static_decoding_torch_index,
        history_item_ids=history_item_ids,
        item_to_index=item_to_index,
        beam_size=max(beam_size, k),
        max_results=max(beam_size, k),
        device=device,
    )
    return _beam_results_to_recommendation_result(
        beam_results=beam_results,
        codec=codec,
        history_item_ids=history_item_ids,
        k=k,
    )


def recommend_batch_with_constrained_generation(
    *,
    model: GenerativeRetriever,
    decoder: StaticDecodingIndex,
    codec: SemanticIdCodec,
    history_item_ids_batch: Sequence[Sequence[int]],
    item_to_index: Mapping[int, int],
    k: int,
    beam_size: int,
    device: torch.device,
    static_decoding_torch_index: StaticDecodingTorchIndex | None = None,
) -> tuple[GenerativeRecommendationResult, ...]:
    """Batched STATIC constrained beam search 결과를 item ranking으로 변환한다."""
    if k < 1:
        msg = "k는 1 이상이어야 합니다."
        raise ValueError(msg)
    if beam_size < 1:
        msg = "beam_size는 1 이상이어야 합니다."
        raise ValueError(msg)

    batch_beam_results = generate_semantic_ids_batch_with_static_decoding(
        model=model,
        index=decoder,
        torch_index=static_decoding_torch_index,
        history_item_ids_batch=history_item_ids_batch,
        item_to_index=item_to_index,
        beam_size=max(beam_size, k),
        max_results=max(beam_size, k),
        device=device,
    )
    results: list[GenerativeRecommendationResult] = []
    for history_item_ids, beam_results in zip(
        history_item_ids_batch,
        batch_beam_results,
        strict=True,
    ):
        results.append(
            _beam_results_to_recommendation_result(
                beam_results=beam_results,
                codec=codec,
                history_item_ids=history_item_ids,
                k=k,
            )
        )
    return tuple(results)


def _beam_results_to_recommendation_result(
    *,
    beam_results: Sequence[Any],
    codec: SemanticIdCodec,
    history_item_ids: Sequence[int],
    k: int,
) -> GenerativeRecommendationResult:
    history_items = {int(item_id) for item_id in history_item_ids}
    seen_items: set[int] = set()
    item_ids: list[int] = []
    invalid_sequences = 0
    history_filtered_items = 0
    duplicate_items = 0

    for beam_result in beam_results:
        try:
            item_id = codec.decode_semantic_id(beam_result.semantic_id)
        except UnknownSemanticIdError:
            invalid_sequences += 1
            continue

        if item_id in history_items:
            history_filtered_items += 1
            continue
        if item_id in seen_items:
            duplicate_items += 1
            continue

        item_ids.append(item_id)
        seen_items.add(item_id)
        if len(item_ids) >= k:
            break

    return GenerativeRecommendationResult(
        item_ids=tuple(item_ids),
        generated_sequences=len(beam_results),
        invalid_sequences=invalid_sequences,
        history_filtered_items=history_filtered_items,
        duplicate_items=duplicate_items,
    )


def write_generative_ranking_report(
    report_path: str | Path,
    evaluations: Sequence[GenerativeRankingEvaluation],
    *,
    checkpoint_path: str | Path,
    semantic_id_path: str | Path,
    beam_size: int,
    cutoffs: Sequence[int],
    inference_batch_size: int = 1,
    decoder_name: str = STATIC_DECODING_DECODER_NAME,
    static_decoding_index_path: str | Path | None = None,
) -> Path:
    """Generative Retrieval ranking 평가 결과를 한국어 markdown 리포트로 저장한다."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generative Retrieval 추천 평가 리포트",
        "",
        "## 설정",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- Semantic ID artifact: `{semantic_id_path}`",
        f"- decoder: `{decoder_name}`",
        *(
            [f"- STATIC index artifact: `{static_decoding_index_path}`"]
            if static_decoding_index_path is not None
            else []
        ),
        f"- beam size: {beam_size}",
        f"- inference batch size: {inference_batch_size}",
        f"- 평가 cutoff: {', '.join(str(cutoff) for cutoff in cutoffs)}",
        "- 추천 ranking 생성 후 사용자 history에 이미 포함된 item은 제외합니다.",
        "",
        "## 추천 성능",
        "",
        "| split | k | Recall@K | NDCG@K | MRR@K |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for evaluation in evaluations:
        for k, metrics in sorted(evaluation.metrics_by_k.items()):
            lines.append(
                f"| {evaluation.split_name} | {k} | "
                f"{metrics.recall:.6f} | {metrics.ndcg:.6f} | {metrics.mrr:.6f} |"
            )

    lines.extend(
        [
            "",
            "## 생성 품질",
            "",
            "| split | examples | unknown targets | generated sequences | invalid sequences | "
            "invalid generation rate | history filtered | duplicate filtered | avg recs/query | "
            "elapsed ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for evaluation in evaluations:
        lines.append(
            f"| {evaluation.split_name} | {evaluation.num_examples:,} | "
            f"{evaluation.unknown_target_examples:,} | {evaluation.generated_sequences:,} | "
            f"{evaluation.invalid_sequences:,} | {evaluation.invalid_generation_rate:.6f} | "
            f"{evaluation.history_filtered_items:,} | {evaluation.duplicate_items:,} | "
            f"{evaluation.avg_recommendations:.2f} | {evaluation.elapsed_ms:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 해석",
            "",
            "이 평가는 teacher-forcing token accuracy가 아니라 실제 추천 ranking 품질을 봅니다.",
            "모델 logits에 선택한 constrained decoder를 적용해 존재하는 Semantic ID만 생성한 뒤 "
            "item_id로 복원하고, baseline과 같은 Recall/NDCG/MRR 기준으로 비교합니다.",
            "unknown target은 Semantic ID catalog에 없는 cold-start target이므로 추천 가능 "
            "후보에는 없지만, 평가 denominator에는 남겨 실제 추천 실패로 반영합니다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _validate_eval_frame(eval_frame: pd.DataFrame) -> None:
    required_columns = {"history_item_ids", "target_item_id"}
    missing_columns = sorted(required_columns - set(eval_frame.columns))
    if missing_columns:
        msg = f"eval 데이터에 필요한 컬럼이 없습니다: {missing_columns}"
        raise ValueError(msg)


def _validate_cutoffs(cutoffs: Sequence[int]) -> None:
    if not cutoffs:
        msg = "cutoffs는 비어 있을 수 없습니다."
        raise ValueError(msg)
    if any(cutoff < 1 for cutoff in cutoffs):
        msg = f"모든 cutoff는 1 이상이어야 합니다: {list(cutoffs)}"
        raise ValueError(msg)

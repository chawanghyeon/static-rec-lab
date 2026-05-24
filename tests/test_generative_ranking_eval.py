from pathlib import Path
from typing import cast

import pandas as pd
import pytest
import torch

from recsys.decoding.static_decoding import StaticDecodingIndex
from recsys.evaluation import (
    GenerativeRankingEvaluation,
    evaluate_generative_ranking,
    evaluate_generative_ranking_from_parquet,
    recommend_batch_with_constrained_generation,
    recommend_with_constrained_generation,
    write_generative_ranking_report,
)
from recsys.models import (
    SEMANTIC_TOKEN_OFFSET,
    GenerativeRetriever,
    GenerativeRetrieverConfig,
    build_item_index_from_codec,
)
from recsys.semantic_id import SemanticIdCodec


def test_evaluate_generative_ranking_computes_recommendation_metrics() -> None:
    codec = _codec()
    frame = pd.DataFrame(
        {
            "history_item_ids": [[10], [30]],
            "history_feedback_ids": [[3], [1]],
            "target_item_id": [20, 999],
        }
    )
    item_to_index = build_item_index_from_codec(codec)

    evaluation = evaluate_generative_ranking(
        model=cast(GenerativeRetriever, _FakeGenerativeModel()),
        item_to_index=item_to_index,
        codec=codec,
        eval_frame=frame,
        split_name="valid",
        cutoffs=(1, 2),
        beam_size=3,
        device=torch.device("cpu"),
    )

    assert isinstance(evaluation, GenerativeRankingEvaluation)
    assert evaluation.num_examples == 2
    assert evaluation.unknown_target_examples == 1
    assert evaluation.invalid_generation_rate == 0.0
    assert evaluation.metrics_by_k[1].recall == pytest.approx(0.5)
    assert evaluation.metrics_by_k[2].recall == pytest.approx(0.5)
    assert evaluation.generated_sequences > 0


def test_evaluate_generative_ranking_loads_static_decoding_index_artifact(
    tmp_path: Path,
) -> None:
    codec = _codec()
    static_decoding_index_path = StaticDecodingIndex.from_codec(
        codec,
        dense_lookup_layers=1,
    ).save_npz(tmp_path / "static_decoding_index.npz")
    frame = pd.DataFrame(
        {
            "history_item_ids": [[10]],
            "history_feedback_ids": [[3]],
            "target_item_id": [20],
        }
    )
    item_to_index = build_item_index_from_codec(codec)

    evaluation = evaluate_generative_ranking(
        model=cast(GenerativeRetriever, _FakeGenerativeModel()),
        item_to_index=item_to_index,
        codec=codec,
        eval_frame=frame,
        split_name="valid",
        cutoffs=(1,),
        beam_size=3,
        device=torch.device("cpu"),
        static_decoding_index_path=static_decoding_index_path,
    )

    assert evaluation.metrics_by_k[1].recall == pytest.approx(1.0)


def test_evaluate_generative_ranking_supports_batched_inference() -> None:
    codec = _codec()
    frame = pd.DataFrame(
        {
            "history_item_ids": [[10], [30]],
            "history_feedback_ids": [[3], [1]],
            "target_item_id": [20, 999],
        }
    )
    item_to_index = build_item_index_from_codec(codec)

    single_evaluation = evaluate_generative_ranking(
        model=cast(GenerativeRetriever, _FakeGenerativeModel()),
        item_to_index=item_to_index,
        codec=codec,
        eval_frame=frame,
        split_name="valid",
        cutoffs=(1, 2),
        beam_size=3,
        device=torch.device("cpu"),
        inference_batch_size=1,
    )
    batched_evaluation = evaluate_generative_ranking(
        model=cast(GenerativeRetriever, _FakeGenerativeModel()),
        item_to_index=item_to_index,
        codec=codec,
        eval_frame=frame,
        split_name="valid",
        cutoffs=(1, 2),
        beam_size=3,
        device=torch.device("cpu"),
        inference_batch_size=2,
    )

    assert batched_evaluation.num_examples == single_evaluation.num_examples
    assert batched_evaluation.generated_sequences == single_evaluation.generated_sequences
    assert batched_evaluation.invalid_generation_rate == pytest.approx(
        single_evaluation.invalid_generation_rate
    )
    assert batched_evaluation.metrics_by_k[1].recall == pytest.approx(
        single_evaluation.metrics_by_k[1].recall
    )
    assert batched_evaluation.metrics_by_k[2].ndcg == pytest.approx(
        single_evaluation.metrics_by_k[2].ndcg
    )


def test_evaluate_generative_ranking_from_parquet_streams_rows(tmp_path: Path) -> None:
    codec = _codec()
    item_to_index = build_item_index_from_codec(codec)
    frame = pd.DataFrame(
        {
            "history_item_ids": [[10], [30]],
            "history_feedback_ids": [[3], [1]],
            "target_item_id": [20, 999],
        }
    )
    eval_path = tmp_path / "valid.parquet"
    frame.to_parquet(eval_path, index=False)

    evaluation = evaluate_generative_ranking_from_parquet(
        model=cast(GenerativeRetriever, _FakeGenerativeModel()),
        item_to_index=item_to_index,
        codec=codec,
        eval_parquet=eval_path,
        split_name="valid",
        cutoffs=(1, 2),
        beam_size=3,
        device=torch.device("cpu"),
        inference_batch_size=2,
        parquet_batch_size=1,
    )

    assert evaluation.num_examples == 2
    assert evaluation.unknown_target_examples == 1
    assert evaluation.metrics_by_k[1].recall == pytest.approx(0.5)


def test_recommend_with_constrained_generation_filters_history_items() -> None:
    codec = _codec()
    item_to_index = build_item_index_from_codec(codec)

    result = recommend_with_constrained_generation(
        model=cast(GenerativeRetriever, _FakeGenerativeModel()),
        decoder=StaticDecodingIndex.from_codec(codec, dense_lookup_layers=1),
        codec=codec,
        history_item_ids=[20],
        item_to_index=item_to_index,
        k=2,
        beam_size=3,
        device=torch.device("cpu"),
    )

    assert result.item_ids[0] == 30
    assert 20 not in result.item_ids
    assert result.history_filtered_items == 1


def test_recommend_batch_with_constrained_generation_matches_single_results() -> None:
    codec = _codec()
    item_to_index = build_item_index_from_codec(codec)
    decoder = StaticDecodingIndex.from_codec(codec, dense_lookup_layers=1)
    model = cast(GenerativeRetriever, _FakeGenerativeModel())

    single_results = [
        recommend_with_constrained_generation(
            model=model,
            decoder=decoder,
            codec=codec,
            history_item_ids=history,
            item_to_index=item_to_index,
            k=2,
            beam_size=3,
            device=torch.device("cpu"),
        )
        for history in ([20], [10])
    ]
    batch_results = recommend_batch_with_constrained_generation(
        model=model,
        decoder=decoder,
        codec=codec,
        history_item_ids_batch=([20], [10]),
        item_to_index=item_to_index,
        k=2,
        beam_size=3,
        device=torch.device("cpu"),
    )

    assert [result.item_ids for result in batch_results] == [
        result.item_ids for result in single_results
    ]
    assert [result.generated_sequences for result in batch_results] == [
        result.generated_sequences for result in single_results
    ]


def test_write_generative_ranking_report(tmp_path: Path) -> None:
    codec = _codec()
    frame = pd.DataFrame(
        {
            "history_item_ids": [[10]],
            "history_feedback_ids": [[3]],
            "target_item_id": [20],
        }
    )
    item_to_index = build_item_index_from_codec(codec)
    evaluation = evaluate_generative_ranking(
        model=cast(GenerativeRetriever, _FakeGenerativeModel()),
        item_to_index=item_to_index,
        codec=codec,
        eval_frame=frame,
        split_name="valid",
        cutoffs=(1,),
        beam_size=3,
        device=torch.device("cpu"),
    )

    report_path = write_generative_ranking_report(
        tmp_path / "generative_eval.md",
        [evaluation],
        checkpoint_path="artifacts/generative/model.pt",
        semantic_id_path="artifacts/semantic_id/semantic_ids.json",
        beam_size=3,
        cutoffs=(1,),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "# Generative Retrieval 추천 평가 리포트" in report
    assert "Recall@K" in report
    assert "invalid generation rate" in report
    assert "unknown target" in report
    assert "static_decoding_pt" in report


def test_evaluate_generative_ranking_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="target_item_id"):
        evaluate_generative_ranking(
            model=cast(GenerativeRetriever, _FakeGenerativeModel()),
            item_to_index={20: 2},
            codec=_codec(),
            eval_frame=pd.DataFrame({"history_item_ids": [[10]], "history_feedback_ids": [[3]]}),
            split_name="valid",
            cutoffs=(1,),
            beam_size=3,
            device=torch.device("cpu"),
        )


class _FakeGenerativeModel:
    def __init__(self) -> None:
        self.config = GenerativeRetrieverConfig(
            item_vocab_size=16,
            semantic_vocab_size=6,
            semantic_id_length=2,
            max_history_length=4,
            d_model=4,
            num_heads=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=8,
            dropout=0.0,
        )

    def eval(self) -> None:
        return None

    def __call__(
        self,
        history_item_ids: torch.Tensor,
        _history_feedback_ids: torch.Tensor,
        _history_padding_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = history_item_ids.shape[0]
        logits = torch.full(
            (batch_size, self.config.semantic_id_length, self.config.semantic_vocab_size),
            -100.0,
        )
        for row_index in range(batch_size):
            logits[row_index, 0, SEMANTIC_TOKEN_OFFSET + 0] = 5.0
            logits[row_index, 0, SEMANTIC_TOKEN_OFFSET + 1] = 4.0
            logits[row_index, 0, SEMANTIC_TOKEN_OFFSET + 2] = 3.0

            previous_raw_token = int(decoder_input_ids[row_index, 1].item()) - (
                SEMANTIC_TOKEN_OFFSET
            )
            preferred_next_token = {0: 1, 1: 2, 2: 3}.get(previous_raw_token, 1)
            logits[row_index, 1, SEMANTIC_TOKEN_OFFSET + preferred_next_token] = 5.0
        return logits


def _codec() -> SemanticIdCodec:
    return SemanticIdCodec(
        {
            20: [0, 1],
            30: [1, 2],
            40: [2, 3],
        }
    )

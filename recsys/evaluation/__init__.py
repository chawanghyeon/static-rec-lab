"""추천 평가 패키지."""

from recsys.decoding import STATIC_DECODING_DECODER_NAME
from recsys.evaluation.generative import (
    GenerativeRankingEvaluation,
    GenerativeRecommendationResult,
    evaluate_generative_ranking,
    evaluate_generative_ranking_from_parquet,
    recommend_batch_with_constrained_generation,
    recommend_with_constrained_generation,
    write_generative_ranking_report,
)
from recsys.evaluation.metrics import (
    RankingMetrics,
    evaluate_ranking_at_k,
    mean_ndcg_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "STATIC_DECODING_DECODER_NAME",
    "GenerativeRankingEvaluation",
    "GenerativeRecommendationResult",
    "RankingMetrics",
    "evaluate_generative_ranking",
    "evaluate_generative_ranking_from_parquet",
    "evaluate_ranking_at_k",
    "mean_ndcg_at_k",
    "mean_recall_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "recommend_batch_with_constrained_generation",
    "recommend_with_constrained_generation",
    "write_generative_ranking_report",
]

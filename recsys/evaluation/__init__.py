"""추천 평가 패키지."""

from recsys.evaluation.generative import (
    STATIC_DECODING_DECODER_NAME,
    GenerativeRankingEvaluation,
    GenerativeRecommendationResult,
    build_static_decoding_index,
    evaluate_generative_ranking,
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
    "build_static_decoding_index",
    "evaluate_generative_ranking",
    "evaluate_ranking_at_k",
    "mean_ndcg_at_k",
    "mean_recall_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "recommend_with_constrained_generation",
    "write_generative_ranking_report",
]

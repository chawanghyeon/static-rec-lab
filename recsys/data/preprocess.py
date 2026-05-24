"""MovieLens sequential recommendation 전처리 로직."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import pandas as pd

from recsys.data.feedback import POSITIVE_FEEDBACK_ID, feedback_id_for_rating

SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "valid", "test")
STANDARD_COLUMNS: Final[tuple[str, str, str, str]] = (
    "user_id",
    "item_id",
    "rating",
    "timestamp",
)
MOVIELENS_COLUMN_MAP: Final[dict[str, str]] = {
    "userId": "user_id",
    "movieId": "item_id",
}
EXAMPLE_COLUMNS: Final[tuple[str, ...]] = (
    "user_id",
    "history_item_ids",
    "history_feedback_ids",
    "positive_history_item_ids",
    "target_item_id",
    "target_timestamp",
    "history_length",
)


@dataclass(frozen=True)
class PreprocessConfig:
    """MovieLens 전처리 설정."""

    min_interactions: int = 5
    max_history_length: int | None = 50
    min_rating: float = 4.0
    neutral_rating: float = 3.0

    def __post_init__(self) -> None:
        if self.min_interactions < 3:
            msg = "min_interactions는 valid/test split을 위해 3 이상이어야 합니다."
            raise ValueError(msg)
        if self.max_history_length is not None and self.max_history_length < 1:
            msg = "max_history_length는 None이거나 1 이상이어야 합니다."
            raise ValueError(msg)
        if self.min_rating < 0:
            msg = "min_rating은 0 이상이어야 합니다."
            raise ValueError(msg)
        if self.neutral_rating < 0:
            msg = "neutral_rating은 0 이상이어야 합니다."
            raise ValueError(msg)
        if self.neutral_rating > self.min_rating:
            msg = "neutral_rating은 min_rating보다 클 수 없습니다."
            raise ValueError(msg)


@dataclass(frozen=True)
class SequenceExample:
    """한 개의 user history -> target item 학습 예제."""

    user_id: int
    history_item_ids: tuple[int, ...]
    history_feedback_ids: tuple[int, ...]
    positive_history_item_ids: tuple[int, ...]
    target_item_id: int
    target_timestamp: int

    @property
    def history_length(self) -> int:
        return len(self.history_item_ids)


@dataclass(frozen=True)
class PreprocessResult:
    """전처리 실행 결과 요약."""

    paths: dict[str, Path]
    split_counts: dict[str, int]
    num_users: int
    num_interactions: int
    num_items: int
    raw_num_users: int
    raw_num_interactions: int
    raw_num_items: int
    rating_filtered_num_users: int
    rating_filtered_num_interactions: int
    rating_filtered_num_items: int
    min_interactions: int
    max_history_length: int | None
    min_rating: float
    neutral_rating: float


def load_ratings_csv(path: str | Path) -> pd.DataFrame:
    """MovieLens ratings.csv를 표준 컬럼명으로 로드한다."""
    return normalize_ratings_frame(pd.read_csv(path))


def normalize_ratings_frame(ratings: pd.DataFrame) -> pd.DataFrame:
    """MovieLens 컬럼명을 내부 표준 컬럼명으로 변환한다."""
    frame = ratings.rename(columns=MOVIELENS_COLUMN_MAP).copy()
    missing_columns = sorted(set(STANDARD_COLUMNS) - set(frame.columns))
    if missing_columns:
        msg = f"ratings 데이터에 필요한 컬럼이 없습니다: {missing_columns}"
        raise ValueError(msg)

    normalized = frame.loc[:, list(STANDARD_COLUMNS)].copy()
    normalized["user_id"] = pd.to_numeric(normalized["user_id"], downcast=None).astype("int64")
    normalized["item_id"] = pd.to_numeric(normalized["item_id"], downcast=None).astype("int64")
    normalized["rating"] = pd.to_numeric(normalized["rating"], downcast=None).astype("float64")
    normalized["timestamp"] = pd.to_numeric(normalized["timestamp"], downcast=None).astype("int64")
    return normalized


def sort_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    """interaction을 user_id, timestamp, item_id 순서로 정렬한다."""
    return interactions.sort_values(
        ["user_id", "timestamp", "item_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def filter_users_by_min_interactions(
    interactions: pd.DataFrame,
    min_interactions: int,
) -> pd.DataFrame:
    """interaction 수가 기준보다 적은 사용자를 제거한다."""
    if min_interactions < 1:
        msg = "min_interactions는 1 이상이어야 합니다."
        raise ValueError(msg)

    interaction_counts = interactions.groupby("user_id")["item_id"].transform("size")
    return interactions.loc[interaction_counts >= min_interactions].reset_index(drop=True)


def filter_interactions_by_min_rating(
    interactions: pd.DataFrame,
    min_rating: float,
) -> pd.DataFrame:
    """positive interaction으로 사용할 최소 rating 이상만 남긴다."""
    if min_rating < 0:
        msg = "min_rating은 0 이상이어야 합니다."
        raise ValueError(msg)
    return interactions.loc[interactions["rating"] >= min_rating].reset_index(drop=True)


def add_feedback_ids(
    interactions: pd.DataFrame,
    *,
    positive_rating: float,
    neutral_rating: float,
) -> pd.DataFrame:
    """rating을 explicit feedback id 컬럼으로 변환한다."""
    frame = interactions.copy()
    frame["feedback_id"] = [
        feedback_id_for_rating(
            float(rating),
            positive_rating=positive_rating,
            neutral_rating=neutral_rating,
        )
        for rating in frame["rating"].tolist()
    ]
    return frame


def filter_users_by_min_positive_interactions(
    interactions: pd.DataFrame,
    min_interactions: int,
) -> pd.DataFrame:
    """positive target 후보가 기준보다 적은 사용자를 제거한다."""
    if min_interactions < 1:
        msg = "min_interactions는 1 이상이어야 합니다."
        raise ValueError(msg)

    positive_counts = (
        interactions["feedback_id"]
        .eq(POSITIVE_FEEDBACK_ID)
        .groupby(interactions["user_id"])
        .transform("sum")
    )
    return interactions.loc[positive_counts >= min_interactions].reset_index(drop=True)


def make_sequential_splits(
    ratings: pd.DataFrame,
    config: PreprocessConfig,
) -> dict[str, pd.DataFrame]:
    """ratings DataFrame을 train/valid/test prefix-target DataFrame으로 변환한다."""
    interactions = _prepare_interactions(ratings, config)
    examples = build_sequence_examples(interactions, config.max_history_length)
    return {split_name: examples_to_frame(examples[split_name]) for split_name in SPLIT_NAMES}


def build_sequence_examples(
    interactions: pd.DataFrame,
    max_history_length: int | None,
) -> dict[str, list[SequenceExample]]:
    """정렬된 interaction에서 positive target split별 SequenceExample을 만든다."""
    examples: dict[str, list[SequenceExample]] = {split_name: [] for split_name in SPLIT_NAMES}

    for _, user_frame in interactions.groupby("user_id", sort=False):
        user_id = int(cast(int, user_frame["user_id"].iloc[0]))
        item_ids = [int(item_id) for item_id in user_frame["item_id"].tolist()]
        feedback_ids = [int(feedback_id) for feedback_id in user_frame["feedback_id"].tolist()]
        timestamps = [int(timestamp) for timestamp in user_frame["timestamp"].tolist()]
        positive_target_indices = [
            index
            for index, feedback_id in enumerate(feedback_ids)
            if feedback_id == POSITIVE_FEEDBACK_ID
        ]

        if len(positive_target_indices) < 3:
            continue

        for target_index in positive_target_indices[:-2]:
            if target_index == 0:
                continue
            examples["train"].append(
                _make_example(
                    user_id=user_id,
                    item_ids=item_ids,
                    feedback_ids=feedback_ids,
                    timestamps=timestamps,
                    target_index=target_index,
                    max_history_length=max_history_length,
                )
            )

        valid_target_index = positive_target_indices[-2]
        test_target_index = positive_target_indices[-1]
        examples["valid"].append(
            _make_example(
                user_id=user_id,
                item_ids=item_ids,
                feedback_ids=feedback_ids,
                timestamps=timestamps,
                target_index=valid_target_index,
                max_history_length=max_history_length,
            )
        )
        examples["test"].append(
            _make_example(
                user_id=user_id,
                item_ids=item_ids,
                feedback_ids=feedback_ids,
                timestamps=timestamps,
                target_index=test_target_index,
                max_history_length=max_history_length,
            )
        )

    return examples


def examples_to_frame(examples: list[SequenceExample]) -> pd.DataFrame:
    """SequenceExample 목록을 parquet 저장에 적합한 DataFrame으로 변환한다."""
    records = [
        {
            "user_id": example.user_id,
            "history_item_ids": list(example.history_item_ids),
            "history_feedback_ids": list(example.history_feedback_ids),
            "positive_history_item_ids": list(example.positive_history_item_ids),
            "target_item_id": example.target_item_id,
            "target_timestamp": example.target_timestamp,
            "history_length": example.history_length,
        }
        for example in examples
    ]
    frame = pd.DataFrame.from_records(records, columns=EXAMPLE_COLUMNS)

    if frame.empty:
        return frame

    frame["user_id"] = frame["user_id"].astype("int64")
    frame["target_item_id"] = frame["target_item_id"].astype("int64")
    frame["target_timestamp"] = frame["target_timestamp"].astype("int64")
    frame["history_length"] = frame["history_length"].astype("int64")
    return frame


def write_split_parquets(
    split_frames: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """split별 DataFrame을 parquet 파일로 저장한다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for split_name in SPLIT_NAMES:
        parquet_path = output_path / f"{split_name}.parquet"
        split_frames[split_name].to_parquet(parquet_path, index=False)
        paths[split_name] = parquet_path

    return paths


def preprocess_ratings_csv(
    ratings_csv: str | Path,
    output_dir: str | Path,
    config: PreprocessConfig,
) -> PreprocessResult:
    """ratings.csv를 전처리하고 train/valid/test parquet 파일을 생성한다."""
    ratings = load_ratings_csv(ratings_csv)
    sorted_ratings = sort_interactions(ratings)
    with_feedback = add_feedback_ids(
        sorted_ratings,
        positive_rating=config.min_rating,
        neutral_rating=config.neutral_rating,
    )
    rating_filtered = with_feedback.loc[
        with_feedback["feedback_id"] == POSITIVE_FEEDBACK_ID
    ].reset_index(drop=True)
    interactions = filter_users_by_min_positive_interactions(
        with_feedback,
        config.min_interactions,
    )
    examples = build_sequence_examples(interactions, config.max_history_length)
    split_frames = {
        split_name: examples_to_frame(examples[split_name]) for split_name in SPLIT_NAMES
    }
    paths = write_split_parquets(split_frames, output_dir)

    return PreprocessResult(
        paths=paths,
        split_counts={split_name: len(split_frames[split_name]) for split_name in SPLIT_NAMES},
        num_users=int(interactions["user_id"].nunique()),
        num_interactions=len(interactions),
        num_items=int(interactions["item_id"].nunique()),
        raw_num_users=int(ratings["user_id"].nunique()),
        raw_num_interactions=len(ratings),
        raw_num_items=int(ratings["item_id"].nunique()),
        rating_filtered_num_users=int(rating_filtered["user_id"].nunique()),
        rating_filtered_num_interactions=len(rating_filtered),
        rating_filtered_num_items=int(rating_filtered["item_id"].nunique()),
        min_interactions=config.min_interactions,
        max_history_length=config.max_history_length,
        min_rating=config.min_rating,
        neutral_rating=config.neutral_rating,
    )


def write_preprocess_report(report_path: str | Path, result: PreprocessResult) -> Path:
    """전처리 결과와 필터링 통계를 markdown report로 저장한다."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MovieLens feedback-aware 전처리 리포트",
        "",
        "## 설정",
        "",
        f"- positive interaction 기준: rating >= {result.min_rating:g}",
        f"- neutral interaction 기준: rating >= {result.neutral_rating:g}",
        f"- 사용자별 최소 positive interaction 수: {result.min_interactions}",
        f"- 최대 history 길이: {_format_optional_int(result.max_history_length)}",
        "",
        "## 필터링 요약",
        "",
        "| 단계 | users | items | interactions |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| raw ratings | {result.raw_num_users:,} | {result.raw_num_items:,} | "
            f"{result.raw_num_interactions:,} |"
        ),
        (
            f"| positive target candidates | {result.rating_filtered_num_users:,} | "
            f"{result.rating_filtered_num_items:,} | "
            f"{result.rating_filtered_num_interactions:,} |"
        ),
        (
            f"| eligible full histories | {result.num_users:,} | {result.num_items:,} | "
            f"{result.num_interactions:,} |"
        ),
        "",
        "## Split 예제 수",
        "",
        "| split | examples | output |",
        "| --- | ---: | --- |",
    ]
    for split_name in SPLIT_NAMES:
        lines.append(
            f"| {split_name} | {result.split_counts[split_name]:,} | `{result.paths[split_name]}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _make_example(
    user_id: int,
    item_ids: list[int],
    feedback_ids: list[int],
    timestamps: list[int],
    target_index: int,
    max_history_length: int | None,
) -> SequenceExample:
    history = item_ids[:target_index]
    history_feedback = feedback_ids[:target_index]
    if max_history_length is not None:
        history = history[-max_history_length:]
        history_feedback = history_feedback[-max_history_length:]
    positive_history = [
        item_id
        for item_id, feedback_id in zip(history, history_feedback, strict=True)
        if feedback_id == POSITIVE_FEEDBACK_ID
    ]

    return SequenceExample(
        user_id=user_id,
        history_item_ids=tuple(history),
        history_feedback_ids=tuple(history_feedback),
        positive_history_item_ids=tuple(positive_history),
        target_item_id=item_ids[target_index],
        target_timestamp=timestamps[target_index],
    )


def _prepare_interactions(ratings: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    normalized = normalize_ratings_frame(ratings)
    sorted_ratings = sort_interactions(normalized)
    with_feedback = add_feedback_ids(
        sorted_ratings,
        positive_rating=config.min_rating,
        neutral_rating=config.neutral_rating,
    )
    return filter_users_by_min_positive_interactions(with_feedback, config.min_interactions)


def _format_optional_int(value: int | None) -> str:
    return "unlimited" if value is None else str(value)

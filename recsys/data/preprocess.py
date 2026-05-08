"""MovieLens sequential recommendation 전처리 로직."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import pandas as pd

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
EXAMPLE_COLUMNS: Final[tuple[str, str, str, str, str]] = (
    "user_id",
    "history_item_ids",
    "target_item_id",
    "target_timestamp",
    "history_length",
)


@dataclass(frozen=True)
class PreprocessConfig:
    """MovieLens 전처리 설정."""

    min_interactions: int = 5
    max_history_length: int | None = 50

    def __post_init__(self) -> None:
        if self.min_interactions < 3:
            msg = "min_interactions는 valid/test split을 위해 3 이상이어야 합니다."
            raise ValueError(msg)
        if self.max_history_length is not None and self.max_history_length < 1:
            msg = "max_history_length는 None이거나 1 이상이어야 합니다."
            raise ValueError(msg)


@dataclass(frozen=True)
class SequenceExample:
    """한 개의 user history -> target item 학습 예제."""

    user_id: int
    history_item_ids: tuple[int, ...]
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


def make_sequential_splits(
    ratings: pd.DataFrame,
    config: PreprocessConfig,
) -> dict[str, pd.DataFrame]:
    """ratings DataFrame을 train/valid/test prefix-target DataFrame으로 변환한다."""
    interactions = filter_users_by_min_interactions(
        sort_interactions(normalize_ratings_frame(ratings)),
        config.min_interactions,
    )
    examples = build_sequence_examples(interactions, config.max_history_length)
    return {split_name: examples_to_frame(examples[split_name]) for split_name in SPLIT_NAMES}


def build_sequence_examples(
    interactions: pd.DataFrame,
    max_history_length: int | None,
) -> dict[str, list[SequenceExample]]:
    """정렬된 interaction에서 split별 SequenceExample을 만든다."""
    examples: dict[str, list[SequenceExample]] = {split_name: [] for split_name in SPLIT_NAMES}

    for _, user_frame in interactions.groupby("user_id", sort=False):
        user_id = int(cast(int, user_frame["user_id"].iloc[0]))
        item_ids = [int(item_id) for item_id in user_frame["item_id"].tolist()]
        timestamps = [int(timestamp) for timestamp in user_frame["timestamp"].tolist()]

        if len(item_ids) < 3:
            continue

        for target_index in range(1, len(item_ids) - 2):
            examples["train"].append(
                _make_example(
                    user_id=user_id,
                    item_ids=item_ids,
                    timestamps=timestamps,
                    target_index=target_index,
                    max_history_length=max_history_length,
                )
            )

        valid_target_index = len(item_ids) - 2
        test_target_index = len(item_ids) - 1
        examples["valid"].append(
            _make_example(
                user_id=user_id,
                item_ids=item_ids,
                timestamps=timestamps,
                target_index=valid_target_index,
                max_history_length=max_history_length,
            )
        )
        examples["test"].append(
            _make_example(
                user_id=user_id,
                item_ids=item_ids,
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
    interactions = filter_users_by_min_interactions(
        sort_interactions(ratings),
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
    )


def _make_example(
    user_id: int,
    item_ids: list[int],
    timestamps: list[int],
    target_index: int,
    max_history_length: int | None,
) -> SequenceExample:
    history = item_ids[:target_index]
    if max_history_length is not None:
        history = history[-max_history_length:]

    return SequenceExample(
        user_id=user_id,
        history_item_ids=tuple(history),
        target_item_id=item_ids[target_index],
        target_timestamp=timestamps[target_index],
    )

"""MovieLens 데이터 다운로드 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

DEFAULT_MOVIELENS_DATASET = "ml-latest-small"
DEFAULT_MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


@dataclass(frozen=True)
class MovieLensDownloadResult:
    """MovieLens 다운로드 및 압축 해제 결과."""

    dataset_name: str
    zip_path: Path
    extracted_dir: Path
    ratings_csv: Path


def download_file(url: str, destination: str | Path, force: bool = False) -> Path:
    """URL의 파일을 destination에 저장한다."""
    destination_path = Path(destination)
    if destination_path.exists() and not force:
        return destination_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, destination_path)
    return destination_path


def extract_movielens_zip(
    zip_path: str | Path,
    output_dir: str | Path,
    dataset_name: str = DEFAULT_MOVIELENS_DATASET,
    force: bool = False,
) -> MovieLensDownloadResult:
    """MovieLens zip 파일을 output_dir에 압축 해제한다."""
    zip_file_path = Path(zip_path)
    output_path = Path(output_dir)
    extracted_dir = output_path / dataset_name
    ratings_csv = extracted_dir / "ratings.csv"

    if ratings_csv.exists() and not force:
        return MovieLensDownloadResult(
            dataset_name=dataset_name,
            zip_path=zip_file_path,
            extracted_dir=extracted_dir,
            ratings_csv=ratings_csv,
        )

    output_path.mkdir(parents=True, exist_ok=True)
    expected_ratings_member = f"{dataset_name}/ratings.csv"

    with ZipFile(zip_file_path) as archive:
        names = set(archive.namelist())
        if expected_ratings_member not in names:
            msg = f"zip 파일 안에서 {expected_ratings_member}를 찾을 수 없습니다."
            raise ValueError(msg)

        _validate_zip_members(archive, output_path)
        archive.extractall(output_path)

    return MovieLensDownloadResult(
        dataset_name=dataset_name,
        zip_path=zip_file_path,
        extracted_dir=extracted_dir,
        ratings_csv=ratings_csv,
    )


def download_movielens(
    output_dir: str | Path,
    url: str = DEFAULT_MOVIELENS_URL,
    dataset_name: str = DEFAULT_MOVIELENS_DATASET,
    force: bool = False,
) -> MovieLensDownloadResult:
    """MovieLens zip 파일을 다운로드하고 압축 해제한다."""
    output_path = Path(output_dir)
    zip_path = output_path / f"{dataset_name}.zip"
    downloaded_zip = download_file(url, zip_path, force=force)
    return extract_movielens_zip(
        zip_path=downloaded_zip,
        output_dir=output_path,
        dataset_name=dataset_name,
        force=force,
    )


def _validate_zip_members(archive: ZipFile, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    for member in archive.infolist():
        target_path = (output_dir / member.filename).resolve()
        if not target_path.is_relative_to(output_root):
            msg = f"zip 파일에 안전하지 않은 경로가 있습니다: {member.filename}"
            raise ValueError(msg)

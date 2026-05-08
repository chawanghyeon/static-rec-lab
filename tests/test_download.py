from pathlib import Path
from zipfile import ZipFile

import pytest

from recsys.data import download_file, extract_movielens_zip


def test_download_file_reuses_existing_file_without_force(tmp_path: Path) -> None:
    destination = tmp_path / "already-downloaded.zip"
    destination.write_text("cached", encoding="utf-8")

    downloaded = download_file("https://example.invalid/file.zip", destination)

    assert downloaded == destination
    assert destination.read_text(encoding="utf-8") == "cached"


def test_extract_movielens_zip_extracts_ratings_csv(tmp_path: Path) -> None:
    zip_path = tmp_path / "ml-latest-small.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("ml-latest-small/ratings.csv", "userId,movieId,rating,timestamp\n")
        archive.writestr("ml-latest-small/movies.csv", "movieId,title,genres\n")

    result = extract_movielens_zip(zip_path=zip_path, output_dir=tmp_path)

    assert result.dataset_name == "ml-latest-small"
    assert result.ratings_csv == tmp_path / "ml-latest-small" / "ratings.csv"
    assert result.ratings_csv.read_text(encoding="utf-8") == "userId,movieId,rating,timestamp\n"


def test_extract_movielens_zip_requires_ratings_csv(tmp_path: Path) -> None:
    zip_path = tmp_path / "broken.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("ml-latest-small/movies.csv", "movieId,title,genres\n")

    with pytest.raises(ValueError, match=r"ratings\.csv"):
        extract_movielens_zip(zip_path=zip_path, output_dir=tmp_path)


def test_extract_movielens_zip_rejects_unsafe_paths(tmp_path: Path) -> None:
    zip_path = tmp_path / "unsafe.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("ml-latest-small/ratings.csv", "userId,movieId,rating,timestamp\n")
        archive.writestr("../outside.txt", "unsafe")

    with pytest.raises(ValueError, match="안전하지 않은 경로"):
        extract_movielens_zip(zip_path=zip_path, output_dir=tmp_path)

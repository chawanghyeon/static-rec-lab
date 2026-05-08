from pathlib import Path

import recsys


def test_required_project_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]

    required_files = [
        "Dockerfile",
        "Makefile",
        "README.md",
        "docker-compose.yml",
        "pyproject.toml",
    ]

    for relative_path in required_files:
        assert (root / relative_path).is_file()


def test_required_project_directories_exist() -> None:
    root = Path(__file__).resolve().parents[1]

    required_directories = [
        "apps/api/routes",
        "apps/api/schemas",
        "data/processed",
        "data/raw",
        "recsys/baseline",
        "recsys/benchmark",
        "recsys/data",
        "recsys/decoding",
        "recsys/evaluation",
        "recsys/models",
        "recsys/semantic_id",
        "reports",
        "scripts",
        "tests",
    ]

    for relative_path in required_directories:
        assert (root / relative_path).is_dir()


def test_package_version_is_defined() -> None:
    assert recsys.__version__ == "0.1.0"

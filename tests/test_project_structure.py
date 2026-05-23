from pathlib import Path

import recsys
import recsys.decoding as decoding
import recsys.models.inference as model_inference


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


def test_static_decoding_dependency_is_pinned_to_expected_git_commit() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    lock_text = (root / "uv.lock").read_text(encoding="utf-8")
    repository = "https://github.com/youtube/static-constraint-decoding.git"
    commit = "c24f9dc8b9b8045716fff7ef750a1f0cb31c6f57"

    assert f"static-decoding @ git+{repository}@{commit}" in pyproject_text
    assert f"{repository}?rev={commit}#{commit}" in lock_text


def test_decoding_public_api_exposes_static_decoding_only() -> None:
    assert hasattr(decoding, "StaticDecodingIndex")
    assert hasattr(decoding, "static_decoding_constrained_beam_search")
    assert hasattr(decoding, "static_decoding_generate_and_apply_logprobs_mask")

    validation_only_names = [
        "INVALID_STATE",
        "SemanticIdTrie",
        "StaticTransitionMatrixDecoder",
        "constrained_beam_search",
    ]
    for name in validation_only_names:
        assert not hasattr(decoding, name)


def test_model_inference_exposes_static_decoding_entrypoint_only() -> None:
    assert hasattr(model_inference, "generate_semantic_ids_with_static_decoding")
    assert not hasattr(model_inference, "generate_semantic_ids")

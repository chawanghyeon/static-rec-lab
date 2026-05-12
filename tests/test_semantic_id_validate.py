from pathlib import Path

from recsys.semantic_id import SemanticIdCodec, validate_semantic_id_file


def test_validate_semantic_id_file_loads_and_validates_codec(tmp_path: Path) -> None:
    path = tmp_path / "semantic_ids.json"
    SemanticIdCodec({1: [0, 1], 2: [1, 0]}).save_json(path)

    result = validate_semantic_id_file(path)

    assert result.num_items == 2
    assert result.semantic_id_length == 2

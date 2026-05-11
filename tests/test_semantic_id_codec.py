import json
from pathlib import Path
from typing import Any, cast

import pytest

from recsys.semantic_id import (
    DuplicateSemanticIdError,
    SemanticIdCodec,
    SemanticIdCodecError,
    UnknownItemIdError,
    UnknownSemanticIdError,
    find_duplicate_semantic_ids,
    validate_semantic_id_mapping,
)


def test_semantic_id_codec_encodes_and_decodes_items() -> None:
    codec = SemanticIdCodec({2959: [12, 4, 81, 7], 4993: [12, 5, 10, 3]})

    assert codec.encode_item(2959) == (12, 4, 81, 7)
    assert codec.decode_semantic_id([12, 5, 10, 3]) == 4993
    assert codec.semantic_id_length == 4
    assert codec.num_items == 2


def test_semantic_id_codec_exposes_read_only_mapping() -> None:
    codec = SemanticIdCodec({1: [1, 2, 3]})

    with pytest.raises(TypeError):
        cast(Any, codec.item_to_semantic_id)[1] = (9, 9, 9)

    assert codec.encode_item(1) == (1, 2, 3)


def test_semantic_id_codec_accepts_generator_semantic_id() -> None:
    codec = SemanticIdCodec({1: [1, 2, 3]})

    assert codec.decode_semantic_id(token for token in [1, 2, 3]) == 1
    assert codec.has_semantic_id(token for token in [1, 2, 3])


def test_semantic_id_codec_raises_for_unknown_item_id() -> None:
    codec = SemanticIdCodec({1: [1, 2, 3]})

    with pytest.raises(UnknownItemIdError, match="item_id"):
        codec.encode_item(2)


def test_semantic_id_codec_raises_for_unknown_semantic_id() -> None:
    codec = SemanticIdCodec({1: [1, 2, 3]})

    with pytest.raises(UnknownSemanticIdError, match="Semantic ID"):
        codec.decode_semantic_id([9, 9, 9])


def test_find_duplicate_semantic_ids_returns_duplicate_groups() -> None:
    duplicates = find_duplicate_semantic_ids(
        {
            1: [1, 2, 3],
            2: [1, 2, 3],
            3: [3, 2, 1],
        }
    )

    assert duplicates == {(1, 2, 3): [1, 2]}


def test_semantic_id_codec_rejects_duplicate_semantic_ids() -> None:
    with pytest.raises(DuplicateSemanticIdError, match="중복 Semantic ID"):
        SemanticIdCodec({1: [1, 2, 3], 2: [1, 2, 3]})


def test_semantic_id_codec_rejects_variable_length_semantic_ids() -> None:
    with pytest.raises(SemanticIdCodecError, match="고정 길이"):
        SemanticIdCodec({1: [1, 2, 3], 2: [1, 2]})


def test_semantic_id_codec_rejects_empty_mapping_and_empty_semantic_id() -> None:
    with pytest.raises(SemanticIdCodecError, match="비어"):
        SemanticIdCodec({})

    with pytest.raises(SemanticIdCodecError, match="비어"):
        SemanticIdCodec({1: []})


def test_validate_semantic_id_mapping_accepts_valid_mapping() -> None:
    validate_semantic_id_mapping({1: [1, 2], 2: [2, 1]})


def test_semantic_id_codec_round_trips_json(tmp_path: Path) -> None:
    codec = SemanticIdCodec({2959: [12, 4, 81, 7], 4993: [12, 5, 10, 3]})
    path = tmp_path / "semantic_ids.json"

    codec.save_json(path)
    loaded = SemanticIdCodec.load_json(path)

    assert loaded == codec
    assert loaded.to_records() == [
        {"item_id": 2959, "semantic_id": [12, 4, 81, 7]},
        {"item_id": 4993, "semantic_id": [12, 5, 10, 3]},
    ]


def test_semantic_id_codec_load_json_rejects_schema_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "semantic_ids.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 999,
                "semantic_id_length": 2,
                "mappings": [{"item_id": 1, "semantic_id": [1, 2]}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SemanticIdCodecError, match="schema_version"):
        SemanticIdCodec.load_json(path)


def test_semantic_id_codec_from_records_rejects_missing_fields() -> None:
    with pytest.raises(SemanticIdCodecError, match="item_id"):
        SemanticIdCodec.from_records([{"item_id": 1}])


def test_semantic_id_codec_rejects_string_semantic_id() -> None:
    with pytest.raises(SemanticIdCodecError, match="문자열"):
        SemanticIdCodec({1: "123"})


def test_semantic_id_codec_rejects_non_integer_item_ids() -> None:
    with pytest.raises(SemanticIdCodecError, match="item_id"):
        SemanticIdCodec(cast(Any, {1.2: [1, 2, 3]}))

    with pytest.raises(SemanticIdCodecError, match="item_id"):
        SemanticIdCodec(cast(Any, {"1": [1, 2, 3]}))

    with pytest.raises(SemanticIdCodecError, match="item_id"):
        SemanticIdCodec(cast(Any, {True: [1, 2, 3]}))


def test_semantic_id_codec_rejects_non_integer_tokens() -> None:
    with pytest.raises(SemanticIdCodecError, match="정수"):
        SemanticIdCodec({1: [1, 2.5, 3]})

    with pytest.raises(SemanticIdCodecError, match="정수"):
        SemanticIdCodec({1: [1, "2", 3]})

    with pytest.raises(SemanticIdCodecError, match="bool"):
        SemanticIdCodec({1: [1, True, 3]})

"""item_id와 Semantic ID 간 양방향 codec."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from operator import index
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

SemanticId = tuple[int, ...]
SCHEMA_VERSION: Final[int] = 1


class SemanticIdCodecError(ValueError):
    """Semantic ID codec 구성 오류."""


class DuplicateSemanticIdError(SemanticIdCodecError):
    """서로 다른 item이 같은 Semantic ID를 사용하는 경우."""


class UnknownItemIdError(KeyError):
    """알 수 없는 item_id 조회 오류."""


class UnknownSemanticIdError(KeyError):
    """알 수 없는 Semantic ID 조회 오류."""


@dataclass(frozen=True)
class SemanticIdCodec:
    """item_id와 고정 길이 Semantic ID token sequence를 변환한다."""

    item_to_semantic_id: Mapping[int, SemanticId] = field(init=False)
    _semantic_id_to_item_id: Mapping[SemanticId, int] = field(init=False, repr=False)
    semantic_id_length: int = field(init=False)

    def __init__(self, item_to_semantic_id: Mapping[int, Iterable[Any]]) -> None:
        normalized = _normalize_mapping(item_to_semantic_id)
        if not normalized:
            msg = "Semantic ID mapping은 비어 있을 수 없습니다."
            raise SemanticIdCodecError(msg)

        lengths = {len(semantic_id) for semantic_id in normalized.values()}
        if len(lengths) != 1:
            msg = f"Semantic ID는 고정 길이어야 합니다. 발견된 길이: {sorted(lengths)}"
            raise SemanticIdCodecError(msg)

        duplicates = find_duplicate_semantic_ids(normalized)
        if duplicates:
            formatted = {
                str(list(semantic_id)): item_ids for semantic_id, item_ids in duplicates.items()
            }
            msg = f"중복 Semantic ID가 있습니다: {formatted}"
            raise DuplicateSemanticIdError(msg)

        reverse = {semantic_id: item_id for item_id, semantic_id in normalized.items()}
        object.__setattr__(self, "item_to_semantic_id", MappingProxyType(normalized))
        object.__setattr__(self, "_semantic_id_to_item_id", MappingProxyType(reverse))
        object.__setattr__(self, "semantic_id_length", lengths.pop())

    @property
    def num_items(self) -> int:
        return len(self.item_to_semantic_id)

    def encode_item(self, item_id: int) -> SemanticId:
        """item_id를 Semantic ID로 변환한다."""
        normalized_item_id = _normalize_item_id(item_id)
        try:
            return self.item_to_semantic_id[normalized_item_id]
        except KeyError as exc:
            msg = f"알 수 없는 item_id입니다: {normalized_item_id}"
            raise UnknownItemIdError(msg) from exc

    def decode_semantic_id(self, semantic_id: Iterable[int]) -> int:
        """Semantic ID를 item_id로 변환한다."""
        normalized_semantic_id = _normalize_semantic_id(semantic_id)
        try:
            return self._semantic_id_to_item_id[normalized_semantic_id]
        except KeyError as exc:
            msg = f"알 수 없는 Semantic ID입니다: {list(normalized_semantic_id)}"
            raise UnknownSemanticIdError(msg) from exc

    def has_item(self, item_id: int) -> bool:
        """item_id가 codec에 포함되어 있는지 확인한다."""
        return _normalize_item_id(item_id) in self.item_to_semantic_id

    def has_semantic_id(self, semantic_id: Iterable[int]) -> bool:
        """Semantic ID가 codec에 포함되어 있는지 확인한다."""
        return _normalize_semantic_id(semantic_id) in self._semantic_id_to_item_id

    def to_records(self) -> list[dict[str, Any]]:
        """JSON 저장에 적합한 record 목록으로 변환한다."""
        return [
            {"item_id": item_id, "semantic_id": list(semantic_id)}
            for item_id, semantic_id in sorted(self.item_to_semantic_id.items())
        ]

    def save_json(self, path: str | Path) -> Path:
        """codec mapping을 JSON 파일로 저장한다."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "semantic_id_length": self.semantic_id_length,
            "num_items": self.num_items,
            "mappings": self.to_records(),
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> SemanticIdCodec:
        """record 목록에서 codec을 생성한다."""
        mapping: dict[int, SemanticId] = {}
        for record in records:
            if "item_id" not in record or "semantic_id" not in record:
                msg = "record에는 item_id와 semantic_id가 필요합니다."
                raise SemanticIdCodecError(msg)
            item_id = _normalize_item_id(record["item_id"])
            if item_id in mapping:
                msg = f"중복 item_id가 있습니다: {item_id}"
                raise SemanticIdCodecError(msg)
            mapping[item_id] = _normalize_semantic_id(record["semantic_id"])
        return cls(mapping)

    @classmethod
    def load_json(cls, path: str | Path) -> SemanticIdCodec:
        """JSON 파일에서 codec을 로드한다."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            msg = "Semantic ID codec JSON은 object여야 합니다."
            raise SemanticIdCodecError(msg)
        if payload.get("schema_version") != SCHEMA_VERSION:
            msg = f"지원하지 않는 schema_version입니다: {payload.get('schema_version')}"
            raise SemanticIdCodecError(msg)
        if "mappings" not in payload:
            msg = "Semantic ID codec JSON에 mappings 필드가 없습니다."
            raise SemanticIdCodecError(msg)
        codec = cls.from_records(payload["mappings"])
        expected_length = payload.get("semantic_id_length")
        if expected_length is not None and int(expected_length) != codec.semantic_id_length:
            msg = "JSON의 semantic_id_length와 실제 mapping 길이가 다릅니다."
            raise SemanticIdCodecError(msg)
        return codec


def find_duplicate_semantic_ids(
    item_to_semantic_id: Mapping[int, Iterable[Any]],
) -> dict[SemanticId, list[int]]:
    """같은 Semantic ID를 공유하는 item_id 목록을 찾는다."""
    semantic_to_items: dict[SemanticId, list[int]] = defaultdict(list)
    for item_id, semantic_id in item_to_semantic_id.items():
        semantic_to_items[_normalize_semantic_id(semantic_id)].append(_normalize_item_id(item_id))

    return {
        semantic_id: sorted(item_ids)
        for semantic_id, item_ids in semantic_to_items.items()
        if len(item_ids) > 1
    }


def validate_semantic_id_mapping(item_to_semantic_id: Mapping[int, Iterable[Any]]) -> None:
    """Semantic ID mapping이 codec 생성 조건을 만족하는지 검증한다."""
    SemanticIdCodec(item_to_semantic_id)


def _normalize_mapping(item_to_semantic_id: Mapping[int, Iterable[Any]]) -> dict[int, SemanticId]:
    normalized: dict[int, SemanticId] = {}
    for item_id, semantic_id in item_to_semantic_id.items():
        normalized_item_id = _normalize_item_id(item_id)
        if normalized_item_id in normalized:
            msg = f"중복 item_id가 있습니다: {normalized_item_id}"
            raise SemanticIdCodecError(msg)
        normalized[normalized_item_id] = _normalize_semantic_id(semantic_id)
    return normalized


def _normalize_semantic_id(semantic_id: Iterable[Any]) -> SemanticId:
    if isinstance(semantic_id, str):
        msg = "semantic_id는 문자열이 아니라 정수 token sequence여야 합니다."
        raise SemanticIdCodecError(msg)

    tokens = tuple(_normalize_token(token) for token in semantic_id)
    if not tokens:
        msg = "semantic_id는 비어 있을 수 없습니다."
        raise SemanticIdCodecError(msg)
    return tokens


def _normalize_item_id(item_id: Any) -> int:
    if isinstance(item_id, bool):
        msg = "item_id는 bool이 아니라 정수여야 합니다."
        raise SemanticIdCodecError(msg)
    try:
        return index(item_id)
    except TypeError as exc:
        msg = f"item_id는 정수여야 합니다: {item_id!r}"
        raise SemanticIdCodecError(msg) from exc


def _normalize_token(token: Any) -> int:
    if isinstance(token, bool):
        msg = "semantic_id token은 bool이 아니라 정수여야 합니다."
        raise SemanticIdCodecError(msg)
    try:
        return index(token)
    except TypeError as exc:
        msg = f"semantic_id token은 정수여야 합니다: {token!r}"
        raise SemanticIdCodecError(msg) from exc

"""Immutable knowledge cards and their bounded YAML loading boundary.

Only curated card files enter this module. Raw notes remain source material
and are never discovered or loaded as runtime knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Union

import yaml


KNOWLEDGE_CARD_SCHEMA = "chemtools.knowledge-card/1"
MAX_CARD_BYTES = 256 * 1024
MAX_CARDS = 512

CardStatus = Literal[
    "draft",
    "accepted",
    "exploratory",
    "shelved",
    "rejected",
]
CardConfidence = Literal["low", "medium", "high"]
CardKind = Literal[
    "validation",
    "provenance",
    "interpretation",
    "workflow",
    "optimization",
    "safety",
]
JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[
    JsonScalar,
    tuple["JsonValue", ...],
    Mapping[str, "JsonValue"],
]

_CARD_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_SCOPE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CARD_STATUSES = frozenset({
    "draft",
    "accepted",
    "exploratory",
    "shelved",
    "rejected",
})
_CARD_CONFIDENCES = frozenset({"low", "medium", "high"})
_CARD_KINDS = frozenset({
    "validation",
    "provenance",
    "interpretation",
    "workflow",
    "optimization",
    "safety",
})
_CARD_FIELDS = frozenset({
    "schema_version",
    "id",
    "programs",
    "workflows",
    "kind",
    "status",
    "confidence",
    "applies_when",
    "claim",
    "check",
    "failure",
    "sources",
    "tests",
})


class KnowledgeCardLoadError(ValueError):
    """A card file could not be read or did not satisfy the card contract."""


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{field_name} contains a NUL character")
    return value


def _scope_values(
    values: Any,
    field_name: str,
    *,
    wildcard: bool = False,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{field_name} must be a non-empty list")
    normalized: list[str] = []
    for index, value in enumerate(values):
        value = _text(value, f"{field_name}[{index}]")
        if value != "*" or not wildcard:
            if not _SCOPE_RE.fullmatch(value):
                raise ValueError(
                    f"{field_name}[{index}] must be a lowercase identifier"
                )
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must not contain duplicates")
    if wildcard and "*" in normalized and len(normalized) != 1:
        raise ValueError("programs '*' cannot be combined with named programs")
    return tuple(normalized)


def _freeze_json(value: Any, field_name: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            key = _text(key, f"{field_name} key")
            frozen[key] = _freeze_json(item, f"{field_name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{field_name} contains unsupported YAML type "
        f"{type(value).__name__}"
    )


def _mapping(value: Any, field_name: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    frozen = _freeze_json(value, field_name)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return frozen


def _optional_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, JsonValue] | None:
    if value is None:
        return None
    return _mapping(value, field_name)


def _plain_json(value: JsonValue) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _repo_references(
    values: Any,
    field_name: str,
    prefixes: tuple[str, ...],
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{field_name} must be a list")
    normalized: list[str] = []
    for index, value in enumerate(values):
        value = _text(value, f"{field_name}[{index}]")
        path_text = value.split("#", 1)[0].split("::", 1)[0]
        path = PurePosixPath(path_text)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"{field_name}[{index}] must stay inside the repository"
            )
        if not any(path_text.startswith(prefix) for prefix in prefixes):
            expected = ", ".join(prefixes)
            raise ValueError(
                f"{field_name}[{index}] must start with one of: {expected}"
            )
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return tuple(normalized)


@dataclass(frozen=True)
class KnowledgeCard:
    id: str
    programs: tuple[str, ...]
    workflows: tuple[str, ...]
    kind: CardKind
    status: CardStatus
    confidence: CardConfidence
    applies_when: Mapping[str, JsonValue]
    claim: str
    sources: tuple[str, ...]
    tests: tuple[str, ...]
    check: Mapping[str, JsonValue] | None = None
    failure: Mapping[str, JsonValue] | None = None
    schema_version: str = KNOWLEDGE_CARD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != KNOWLEDGE_CARD_SCHEMA:
            raise ValueError(
                f"unsupported knowledge-card schema {self.schema_version!r}"
            )
        _text(self.id, "id")
        if not _CARD_ID_RE.fullmatch(self.id):
            raise ValueError("id must be a lowercase dotted identifier")
        object.__setattr__(
            self,
            "programs",
            _scope_values(self.programs, "programs", wildcard=True),
        )
        object.__setattr__(
            self,
            "workflows",
            _scope_values(self.workflows, "workflows"),
        )
        if self.kind not in _CARD_KINDS:
            raise ValueError(f"unsupported knowledge-card kind {self.kind!r}")
        if self.status not in _CARD_STATUSES:
            raise ValueError(
                f"unsupported knowledge-card status {self.status!r}"
            )
        if self.confidence not in _CARD_CONFIDENCES:
            raise ValueError(
                f"unsupported knowledge-card confidence {self.confidence!r}"
            )
        object.__setattr__(
            self,
            "applies_when",
            _mapping(self.applies_when, "applies_when"),
        )
        if not self.applies_when:
            raise ValueError("applies_when must state at least one condition")
        object.__setattr__(self, "claim", _text(self.claim, "claim"))
        object.__setattr__(
            self,
            "sources",
            _repo_references(
                self.sources,
                "sources",
                ("notes/", "docs/", "references/"),
            ),
        )
        object.__setattr__(
            self,
            "tests",
            _repo_references(self.tests, "tests", ("tests/",)),
        )
        object.__setattr__(
            self,
            "check",
            _optional_mapping(self.check, "check"),
        )
        object.__setattr__(
            self,
            "failure",
            _optional_mapping(self.failure, "failure"),
        )
        if self.status == "accepted" and not self.sources:
            raise ValueError("accepted cards must cite at least one source")
        if self.status == "accepted" and not self.tests:
            raise ValueError("accepted cards must cite at least one test")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "programs": list(self.programs),
            "workflows": list(self.workflows),
            "kind": self.kind,
            "status": self.status,
            "confidence": self.confidence,
            "applies_when": _plain_json(self.applies_when),
            "claim": self.claim,
            "check": (
                _plain_json(self.check) if self.check is not None else None
            ),
            "failure": (
                _plain_json(self.failure)
                if self.failure is not None
                else None
            ),
            "sources": list(self.sources),
            "tests": list(self.tests),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KnowledgeCard:
        if not isinstance(value, Mapping):
            raise TypeError("knowledge card must be a mapping")
        unknown_fields = sorted(set(value) - _CARD_FIELDS)
        if unknown_fields:
            raise ValueError(
                f"unknown knowledge-card fields: {unknown_fields}"
            )
        required_fields = _CARD_FIELDS - {"check", "failure"}
        missing_fields = sorted(required_fields - set(value))
        if missing_fields:
            raise ValueError(
                f"missing knowledge-card fields: {missing_fields}"
            )
        return cls(
            schema_version=value["schema_version"],
            id=value["id"],
            programs=value["programs"],
            workflows=value["workflows"],
            kind=value["kind"],
            status=value["status"],
            confidence=value["confidence"],
            applies_when=value["applies_when"],
            claim=value["claim"],
            check=value.get("check"),
            failure=value.get("failure"),
            sources=value["sources"],
            tests=value["tests"],
        )


def bundled_card_directory() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "knowledge" / "cards"


def load_knowledge_card(path: str | Path) -> KnowledgeCard:
    card_path = Path(path)
    if not card_path.is_file():
        raise KnowledgeCardLoadError(
            f"knowledge card is not a readable file: {card_path}"
        )
    size = card_path.stat().st_size
    if size > MAX_CARD_BYTES:
        raise KnowledgeCardLoadError(
            f"knowledge card exceeds {MAX_CARD_BYTES} bytes: {card_path}"
        )
    try:
        document = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        return KnowledgeCard.from_dict(document)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise KnowledgeCardLoadError(
            f"invalid knowledge card {card_path}: {exc}"
        ) from exc


def load_knowledge_cards(
    directory: str | Path | None = None,
) -> tuple[KnowledgeCard, ...]:
    card_directory = (
        Path(directory)
        if directory is not None
        else bundled_card_directory()
    )
    if not card_directory.is_dir():
        raise KnowledgeCardLoadError(
            f"knowledge-card directory does not exist: {card_directory}"
        )
    paths = sorted(card_directory.glob("*.yaml"))
    if len(paths) > MAX_CARDS:
        raise KnowledgeCardLoadError(
            f"knowledge-card directory exceeds {MAX_CARDS} cards"
        )
    cards = tuple(load_knowledge_card(path) for path in paths)
    identifiers = [card.id for card in cards]
    if len(identifiers) != len(set(identifiers)):
        raise KnowledgeCardLoadError("duplicate knowledge-card IDs are not allowed")
    for path, card in zip(paths, cards):
        if path.stem != card.id:
            raise KnowledgeCardLoadError(
                f"knowledge card {path} must be named {card.id}.yaml"
            )
    return cards


__all__ = [
    "CardConfidence",
    "CardKind",
    "CardStatus",
    "KNOWLEDGE_CARD_SCHEMA",
    "MAX_CARD_BYTES",
    "MAX_CARDS",
    "KnowledgeCard",
    "KnowledgeCardLoadError",
    "bundled_card_directory",
    "load_knowledge_card",
    "load_knowledge_cards",
]

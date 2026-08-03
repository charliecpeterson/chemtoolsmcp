"""Deterministic search over validated knowledge cards.

Accepted cards are the default search scope. Other curation states require an
explicit status so incomplete or rejected claims cannot surface accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Iterable, get_args

from chemtools.knowledge.cards import (
    CardConfidence,
    CardKind,
    CardStatus,
    KnowledgeCard,
    load_knowledge_cards,
)


KNOWLEDGE_SEARCH_SCHEMA = "chemtools.knowledge-search/1"
MAX_QUERY_CHARS = 256
MAX_SEARCH_RESULTS = 50
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class KnowledgeSearchResult:
    query: str | None
    program: str | None
    workflow: str | None
    kind: CardKind | None
    status: CardStatus
    confidence: CardConfidence | None
    limit: int
    total_matches: int
    cards: tuple[KnowledgeCard, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": KNOWLEDGE_SEARCH_SCHEMA,
            "filters": {
                "query": self.query,
                "program": self.program,
                "workflow": self.workflow,
                "kind": self.kind,
                "status": self.status,
                "confidence": self.confidence,
                "limit": self.limit,
            },
            "total_matches": self.total_matches,
            "returned_count": len(self.cards),
            "truncated": self.total_matches > len(self.cards),
            "curation_notice": (
                "Only accepted cards are eligible recommendations. Other "
                "statuses are returned for inspection only."
            ),
            "cards": [self._card_dict(card) for card in self.cards],
        }

    @staticmethod
    def _card_dict(card: KnowledgeCard) -> dict[str, object]:
        payload = card.to_dict()
        payload["recommendation_eligible"] = card.status == "accepted"
        return payload


def search_knowledge_cards(
    *,
    query: str | None = None,
    program: str | None = None,
    workflow: str | None = None,
    kind: CardKind | None = None,
    status: CardStatus = "accepted",
    confidence: CardConfidence | None = None,
    limit: int = 10,
    cards: Iterable[KnowledgeCard] | None = None,
) -> KnowledgeSearchResult:
    normalized_query = _query(query)
    normalized_program = _identifier(program, "program")
    normalized_workflow = _identifier(workflow, "workflow")
    normalized_kind = _literal(kind, "kind", CardKind, optional=True)
    normalized_status = _literal(status, "status", CardStatus)
    normalized_confidence = _literal(
        confidence,
        "confidence",
        CardConfidence,
        optional=True,
    )
    bounded_limit = _limit(limit)
    available_cards = (
        tuple(cards) if cards is not None else load_knowledge_cards()
    )
    if not all(isinstance(card, KnowledgeCard) for card in available_cards):
        raise TypeError("cards must contain only KnowledgeCard values")

    matches = tuple(
        card
        for card in sorted(available_cards, key=lambda item: item.id)
        if card.status == normalized_status
        and (
            normalized_program is None
            or "*" in card.programs
            or normalized_program in card.programs
        )
        and (
            normalized_workflow is None
            or normalized_workflow in card.workflows
        )
        and (normalized_kind is None or card.kind == normalized_kind)
        and (
            normalized_confidence is None
            or card.confidence == normalized_confidence
        )
        and _matches_query(card, normalized_query)
    )
    return KnowledgeSearchResult(
        query=normalized_query,
        program=normalized_program,
        workflow=normalized_workflow,
        kind=normalized_kind,
        status=normalized_status,
        confidence=normalized_confidence,
        limit=bounded_limit,
        total_matches=len(matches),
        cards=matches[:bounded_limit],
    )


def _query(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("query must be a non-empty string when provided")
    normalized = " ".join(value.split())
    if len(normalized) > MAX_QUERY_CHARS:
        raise ValueError(f"query must not exceed {MAX_QUERY_CHARS} characters")
    if not _TOKEN_RE.search(normalized.casefold()):
        raise ValueError("query must contain at least one letter or number")
    return normalized


def _identifier(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string when provided")
    normalized = value.strip().lower()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(f"{field} must be a lowercase identifier")
    return normalized


def _literal(value, field: str, literal_type, *, optional: bool = False):
    if value is None and optional:
        return None
    allowed = get_args(literal_type)
    if value not in allowed:
        raise ValueError(f"{field} must be one of {list(allowed)}")
    return value


def _limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("limit must be an integer")
    if not 1 <= value <= MAX_SEARCH_RESULTS:
        raise ValueError(f"limit must be between 1 and {MAX_SEARCH_RESULTS}")
    return value


def _matches_query(card: KnowledgeCard, query: str | None) -> bool:
    if query is None:
        return True
    query_tokens = _TOKEN_RE.findall(query.casefold())
    searchable = json.dumps(
        card.to_dict(),
        ensure_ascii=True,
        sort_keys=True,
    ).casefold()
    searchable_tokens = set(_TOKEN_RE.findall(searchable))
    return bool(query_tokens) and all(
        token in searchable_tokens for token in query_tokens
    )


__all__ = [
    "KNOWLEDGE_SEARCH_SCHEMA",
    "MAX_QUERY_CHARS",
    "MAX_SEARCH_RESULTS",
    "KnowledgeSearchResult",
    "search_knowledge_cards",
]

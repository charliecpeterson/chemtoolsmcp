"""Read-only MCP access to curated computational-chemistry knowledge."""

from __future__ import annotations

from typing import Any, get_args

from chemtools.knowledge.cards import CardConfidence, CardKind, CardStatus
from chemtools.knowledge.search import (
    MAX_QUERY_CHARS,
    MAX_SEARCH_RESULTS,
    search_knowledge_cards,
)
from chemtools.mcp.decorator import _tool


_SEARCH_ARGUMENTS = frozenset({
    "query",
    "program",
    "workflow",
    "kind",
    "status",
    "confidence",
    "limit",
})


@_tool("search_knowledge_cards", program="generic")
def _handle_search_knowledge_cards(
    arguments: dict[str, Any],
) -> dict[str, object]:
    unknown = sorted(set(arguments) - _SEARCH_ARGUMENTS)
    if unknown:
        raise ValueError(f"unknown search arguments: {unknown}")
    return search_knowledge_cards(
        query=arguments.get("query"),
        program=arguments.get("program"),
        workflow=arguments.get("workflow"),
        kind=arguments.get("kind"),
        status=arguments.get("status", "accepted"),
        confidence=arguments.get("confidence"),
        limit=arguments.get("limit", 10),
    ).to_dict()


def knowledge_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "search_knowledge_cards",
            "description": (
                "Search validated Chemtools knowledge cards by text, program, "
                "workflow, claim kind, curation status, and confidence. "
                "Accepted cards are returned by default. Draft, exploratory, "
                "shelved, and rejected cards require an explicit status. "
                "Every result includes its scope, curation status, sources, "
                "checks, and regression tests."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_QUERY_CHARS,
                        "description": (
                            "Optional text matched against complete card "
                            "metadata using case-insensitive whole tokens."
                        ),
                    },
                    "program": {
                        "type": "string",
                        "pattern": "^[A-Za-z][A-Za-z0-9_]*$",
                        "description": (
                            "Optional program scope. Cross-program cards also "
                            "match a named program."
                        ),
                    },
                    "workflow": {
                        "type": "string",
                        "pattern": "^[A-Za-z][A-Za-z0-9_]*$",
                    },
                    "kind": {
                        "type": "string",
                        "enum": list(get_args(CardKind)),
                    },
                    "status": {
                        "type": "string",
                        "enum": list(get_args(CardStatus)),
                        "default": "accepted",
                        "description": (
                            "One explicit curation state. Omit to search only "
                            "accepted cards."
                        ),
                    },
                    "confidence": {
                        "type": "string",
                        "enum": list(get_args(CardConfidence)),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_SEARCH_RESULTS,
                        "default": 10,
                    },
                },
                "additionalProperties": False,
            },
        }
    ]


__all__ = ["knowledge_tool_definitions"]

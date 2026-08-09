"""Small JSON Schema builders shared by guided MCP result contracts."""

from __future__ import annotations

from typing import Any, Mapping


OBJECT = {"type": "object"}
ARRAY = {"type": "array"}
STRING = {"type": "string"}
INTEGER = {"type": "integer"}
BOOLEAN = {"type": "boolean"}


def versioned_output_schema(
    schema_version: str,
    properties: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """Describe stable top-level result fields without freezing nested detail."""
    success = {
        "type": "object",
        "properties": {
            "schema_version": {"const": schema_version},
            **properties,
        },
        "required": ["schema_version", *properties],
    }
    return {
        "type": "object",
        "oneOf": [
            success,
            {
                "type": "object",
                "properties": {
                    "error": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["error"],
            },
        ],
    }


__all__ = [
    "ARRAY",
    "BOOLEAN",
    "INTEGER",
    "OBJECT",
    "STRING",
    "versioned_output_schema",
]

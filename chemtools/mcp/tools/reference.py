"""Read-only MCP access to curated scientific reference datasets."""

from __future__ import annotations

from typing import Any

from chemtools.mcp.decorator import _tool
from chemtools.reference.fblock_lookup import lookup_grasp_fblock_state
from chemtools.reference.fblock_plan import plan_fblock_atomic_state


_LOOKUP_ARGUMENTS = frozenset({"element", "state"})


@_tool("lookup_grasp_fblock_state", program="grasp")
def _handle_lookup_grasp_fblock_state(
    arguments: dict[str, Any],
) -> dict[str, object]:
    unknown = sorted(set(arguments) - _LOOKUP_ARGUMENTS)
    if unknown:
        raise ValueError(f"unknown f-block lookup arguments: {unknown}")
    if "element" not in arguments:
        raise ValueError("element is required")
    return lookup_grasp_fblock_state(
        element=arguments["element"],
        state=arguments.get("state"),
    ).to_dict()


@_tool("plan_fblock_atomic_state", program="grasp")
def _handle_plan_fblock_atomic_state(
    arguments: dict[str, Any],
) -> dict[str, object]:
    unknown = sorted(set(arguments) - _LOOKUP_ARGUMENTS)
    if unknown:
        raise ValueError(f"unknown f-block plan arguments: {unknown}")
    missing = sorted(_LOOKUP_ARGUMENTS - set(arguments))
    if missing:
        raise ValueError(f"missing f-block plan arguments: {missing}")
    return plan_fblock_atomic_state(
        element=arguments["element"],
        state=arguments["state"],
    ).to_dict()


def reference_tool_definitions() -> list[dict[str, Any]]:
    identifier_properties = {
        "element": {
            "type": "string",
            "minLength": 1,
            "maxLength": 2,
            "pattern": "^[A-Za-z]{1,2}$",
            "description": (
                "Exact element symbol in the catalog, such as Th, U, or Y. "
                "Matching is case-insensitive."
            ),
        },
        "state": {
            "type": "string",
            "minLength": 8,
            "maxLength": 32,
            "pattern": "^ion[0-9]+_[a-z0-9]+$",
            "description": (
                "Exact catalog state slug, such as ion0_6d27s2."
            ),
        },
    }
    return [
        {
            "name": "lookup_grasp_fblock_state",
            "description": (
                "Look up an exact element or state in the versioned "
                "GRASP2018 f-block atomic catalog. Omit state to list the "
                "available state slugs for an element. An exact state result "
                "includes its configuration, J blocks, CSF counts, seed "
                "class, donor lineage, staged orbital birth, and DC and "
                "DC+Breit configuration-average energies. Every response "
                "includes the review status, catalog hash, method scope, and "
                "known limitations. These values are method-scoped "
                "references, not experimental levels."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "element": identifier_properties["element"],
                    "state": {
                        **identifier_properties["state"],
                        "description": (
                            "Optional exact catalog state slug, such as "
                            "ion0_6d27s2. Omit it to list the element's "
                            "available states."
                        ),
                    },
                },
                "required": ["element"],
                "additionalProperties": False,
            },
        },
        {
            "name": "plan_fblock_atomic_state",
            "description": (
                "Build the recorded ATSP2K seed and GRASP2018 DC+Breit "
                "reference plan for one exact f-block catalog state. The "
                "result includes the 13-line ATSP2K input, GRASP interactive "
                "inputs, J and CSF expectations, energies, ordered donor "
                "prerequisites, staged births, and validation checks. "
                "Unresolved donor aliases are checked against a "
                "consumer-scoped review ledger, and unsupported orbital "
                "merging remains an explicit manual requirement. The tool "
                "does not run programs or invent donor mappings."
            ),
            "inputSchema": {
                "type": "object",
                "properties": identifier_properties,
                "required": ["element", "state"],
                "additionalProperties": False,
            },
        },
    ]


__all__ = ["reference_tool_definitions"]

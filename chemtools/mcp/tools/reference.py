"""Read-only MCP access to curated scientific reference datasets."""

from __future__ import annotations

from typing import Any

from chemtools.mcp.decorator import _tool
from chemtools.reference.atomic_multiplets import analyze_atomic_multiplets
from chemtools.reference.fblock_lookup import lookup_grasp_fblock_state
from chemtools.reference.fblock_grasp import validate_grasp_fblock_artifacts
from chemtools.reference.fblock_plan import plan_fblock_atomic_state
from chemtools.reference.grasp_angular_census import (
    validate_grasp_csf_angular_census,
)


_LOOKUP_ARGUMENTS = frozenset({"element", "state"})
_MULTIPLET_ARGUMENTS = frozenset({"configuration"})
_ANGULAR_CENSUS_ARGUMENTS = frozenset({"csf_path"})
_VALIDATE_ARGUMENTS = frozenset({
    "element",
    "state",
    "csf_path",
    "mixing_path",
    "level_limit",
    "component_limit",
})


@_tool("analyze_atomic_multiplets")
def _handle_analyze_atomic_multiplets(
    arguments: dict[str, Any],
) -> dict[str, object]:
    unknown = sorted(set(arguments) - _MULTIPLET_ARGUMENTS)
    if unknown:
        raise ValueError(f"unknown atomic multiplet arguments: {unknown}")
    if "configuration" not in arguments:
        raise ValueError("configuration is required")
    return analyze_atomic_multiplets(arguments["configuration"])


@_tool("validate_grasp_csf_angular_census", program="grasp")
def _handle_validate_grasp_csf_angular_census(
    arguments: dict[str, Any],
) -> dict[str, object]:
    unknown = sorted(set(arguments) - _ANGULAR_CENSUS_ARGUMENTS)
    if unknown:
        raise ValueError(f"unknown GRASP angular census arguments: {unknown}")
    if "csf_path" not in arguments:
        raise ValueError("csf_path is required")
    return validate_grasp_csf_angular_census(arguments["csf_path"])


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


@_tool("validate_grasp_fblock_artifacts", program="grasp")
def _handle_validate_grasp_fblock_artifacts(
    arguments: dict[str, Any],
) -> dict[str, object]:
    unknown = sorted(set(arguments) - _VALIDATE_ARGUMENTS)
    if unknown:
        raise ValueError(f"unknown f-block validation arguments: {unknown}")
    required = {"element", "state", "csf_path"}
    missing = sorted(required - set(arguments))
    if missing:
        raise ValueError(f"missing f-block validation arguments: {missing}")
    return validate_grasp_fblock_artifacts(
        element=arguments["element"],
        state=arguments["state"],
        csf_path=arguments["csf_path"],
        mixing_path=arguments.get("mixing_path"),
        level_limit=arguments.get("level_limit", 64),
        component_limit=arguments.get("component_limit", 3),
    )


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
            "name": "analyze_atomic_multiplets",
            "description": (
                "Enumerate LS terms, repeated-term counts, allowed J/parity "
                "levels, pure-LS Lande factors, and the relativistic jj "
                "occupation/CSF census for a compact atomic configuration "
                "such as 4f7 6s2. Independently reconciles determinant, LS, "
                "J-level, and jj state counts. Hund guidance is limited to "
                "one open subshell. This is a symmetry preflight calculation; "
                "it does not calculate radial integrals, energies, SOC "
                "splittings, mixing, or unique LS labels for relativistic ASFs."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "configuration": {
                        "type": "string",
                        "minLength": 2,
                        "maxLength": 256,
                        "description": (
                            "Compact nonrelativistic shell occupations, for "
                            "example '2p2', '4f7 6s2', or '5f2 6d1 7s2'. "
                            "Principal quantum numbers may be omitted when only "
                            "angular structure matters."
                        ),
                    },
                },
                "required": ["configuration"],
                "additionalProperties": False,
            },
        },
        {
            "name": "validate_grasp_csf_angular_census",
            "description": (
                "Independently validate the jj-coupled CSF multiplicity of "
                "every relativistic occupation and J/parity pair represented "
                "in a GRASP .c file. Reports each configuration's complete "
                "allowed J census and whether the file contains its full J "
                "manifold. This catches missing or duplicate coupling paths "
                "inside represented configurations. It does not prove that "
                "the requested excitation space was generated, assign LS "
                "terms to ASFs, or validate energies and SOC mixing."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "csf_path": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Path to a generated GRASP .c file.",
                    },
                },
                "required": ["csf_path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "lookup_grasp_fblock_state",
            "description": (
                "Look up an exact element or state in the versioned "
                "GRASP2018 f-block atomic catalog. Omit state to list the "
                "available state slugs for an element. An exact state result "
                "includes its configuration, J blocks, CSF counts, seed "
                "class, donor lineage, staged orbital birth, and DC and "
                "DC+Breit configuration-average energies. It also reports "
                "explicit shell populations, unconstrained-SCF transfer risk, "
                "the D2h p/f limitation, and signed f/d separation when the "
                "paired states exist. Every response includes the review "
                "status, GRASP-only recommendation scope, catalog hash, method "
                "scope, and known limitations. These values are method-scoped "
                "references, not experimental levels or generic SCF inputs."
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
                "inputs, electron/J/parity/CSF expectations, energies, ordered donor "
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
        {
            "name": "validate_grasp_fblock_artifacts",
            "description": (
                "Validate a generated GRASP2018 CSF list against one exact "
                "f-block catalog state. Checks the ion electron count and "
                "every J, parity, and CSF-count block. Optionally inspect a "
                "matching RMCDHF or RCI mixing file, validate its ASF block "
                "labels and counts against the CSFs, and resolve bounded "
                "leading components. The catalog block contract is also "
                "derived independently from LS/jj state counting, and each "
                "represented relativistic occupation is checked for complete "
                "coupling multiplicity at every present J."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    **identifier_properties,
                    "csf_path": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Path to the generated GRASP .c file.",
                    },
                    "mixing_path": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Optional matching RMCDHF .m or RCI .cm file."
                        ),
                    },
                    "level_limit": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2000,
                        "default": 64,
                    },
                    "component_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 3,
                    },
                },
                "required": ["element", "state", "csf_path"],
                "additionalProperties": False,
            },
        },
    ]


__all__ = ["reference_tool_definitions"]

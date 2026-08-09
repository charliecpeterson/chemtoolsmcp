"""Metadata and pure argument adapters for hidden MCP tool aliases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from chemtools.mcp.decorator import SERVER_VERSION


ArgumentAdapter = Callable[[dict[str, Any]], dict[str, Any]]
ResultAdapter = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class CompatibilityAvailability:
    program: str
    capability: str


@dataclass(frozen=True)
class ToolEffects:
    reads_local_files: bool
    writes_local_files: bool
    executes_processes: bool
    cancels_processes: bool
    network_access: bool


@dataclass(frozen=True)
class ToolAlias:
    name: str
    target: str
    input_schema: Mapping[str, Any] | None
    translate_arguments: ArgumentAdapter
    translate_result: ResultAdapter | None
    availability: CompatibilityAvailability
    effects: ToolEffects | None
    contract_status: str
    deprecated_since: str | None
    remove_after: str | None
    reason: str
    state: str = "callable_deprecated"


def _identity(arguments: dict[str, Any]) -> dict[str, Any]:
    return dict(arguments)


def _scf_fix_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    translated = dict(arguments)
    translated["mode"] = "scf"
    return translated


def _state_recovery_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    translated = dict(arguments)
    translated["mode"] = "state"
    return translated


def _compact_to_detail(arguments: dict[str, Any]) -> dict[str, Any]:
    translated = dict(arguments)
    if translated.pop("compact", False):
        translated["detail"] = "compact"
    return translated


def _summarize_cube_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {**arguments, "summarize": True}


def _unverified_alias(
    name: str,
    target: str,
    *,
    translate_arguments: ArgumentAdapter = _identity,
    program: str,
    capability: str,
    reason: str,
) -> ToolAlias:
    return ToolAlias(
        name=name,
        target=target,
        input_schema=None,
        translate_arguments=translate_arguments,
        translate_result=None,
        availability=CompatibilityAvailability(
            program=program,
            capability=capability,
        ),
        effects=None,
        contract_status="unverified",
        deprecated_since=SERVER_VERSION,
        remove_after=None,
        reason=reason,
    )


HIDDEN_TOOL_ALIASES = (
    _unverified_alias(
        "render_with_orbitron",
        "visualize",
        program="generic",
        capability="none",
        reason="The guided surface renamed the fixed Orbitron operation.",
    ),
    _unverified_alias(
        "search_knowledge_cards",
        "search_knowledge",
        program="generic",
        capability="none",
        reason="The guided surface adopted an intent-level knowledge name.",
    ),
    _unverified_alias(
        "diagnose_nwchem_output",
        "analyze_nwchem_case",
        program="nwchem",
        capability="none",
        reason="NWChem case analysis replaced the narrower diagnosis name.",
    ),
    _unverified_alias(
        "summarize_nwchem_case",
        "analyze_nwchem_case",
        translate_arguments=_compact_to_detail,
        program="nwchem",
        capability="none",
        reason="NWChem case analysis consolidated summary operations.",
    ),
    _unverified_alias(
        "review_nwchem_case",
        "analyze_nwchem_case",
        translate_arguments=_compact_to_detail,
        program="nwchem",
        capability="none",
        reason="NWChem case analysis consolidated review operations.",
    ),
    _unverified_alias(
        "check_nwchem_run_status",
        "get_nwchem_run_status",
        program="nwchem",
        capability="executable",
        reason="The canonical run-status operation uses the get prefix.",
    ),
    _unverified_alias(
        "review_nwchem_followup_outcome",
        "compare_nwchem_runs",
        program="nwchem",
        capability="none",
        reason="Run comparison replaced the narrower follow-up review name.",
    ),
    _unverified_alias(
        "suggest_nwchem_scf_fix_strategy",
        "suggest_nwchem_recovery",
        translate_arguments=_scf_fix_arguments,
        program="nwchem",
        capability="none",
        reason="The canonical recovery operation selects SCF mode explicitly.",
    ),
    _unverified_alias(
        "suggest_nwchem_state_recovery_strategy",
        "suggest_nwchem_recovery",
        translate_arguments=_state_recovery_arguments,
        program="nwchem",
        capability="none",
        reason="The canonical recovery operation selects state mode explicitly.",
    ),
    _unverified_alias(
        "prepare_nwchem_run",
        "launch_nwchem_run",
        program="nwchem",
        capability="executable",
        reason="The canonical NWChem execution operation uses the launch name.",
    ),
    _unverified_alias(
        "render_nwchem_basis_from_input",
        "render_nwchem_basis_block",
        program="nwchem",
        capability="none",
        reason="The canonical name identifies the rendered basis-block result.",
    ),
    _unverified_alias(
        "summarize_cube_file",
        "parse_cube_file",
        translate_arguments=_summarize_cube_arguments,
        program="generic",
        capability="none",
        reason="Cube summary is a fixed mode of the canonical parser.",
    ),
    _unverified_alias(
        "resolve_nwchem_ecp",
        "render_nwchem_ecp_block",
        program="nwchem",
        capability="none",
        reason="The canonical name identifies the rendered ECP-block result.",
    ),
    _unverified_alias(
        "render_nwchem_ecp_from_elements",
        "render_nwchem_ecp_block",
        program="nwchem",
        capability="none",
        reason="Element-based ECP rendering moved under one canonical operation.",
    ),
    _unverified_alias(
        "resolve_nwchem_basis_setup",
        "render_nwchem_basis_setup",
        program="nwchem",
        capability="none",
        reason="The canonical basis-setup operation uses the render name.",
    ),
)


def validate_tool_aliases(
    aliases: Iterable[ToolAlias],
    *,
    canonical_names: Iterable[str],
    capabilities: Mapping[str, str],
    programs: Mapping[str, str],
    mode_capabilities: Mapping[str, frozenset[str]],
) -> tuple[ToolAlias, ...]:
    validated = tuple(aliases)
    names = [alias.name for alias in validated]
    canonical = set(canonical_names)
    alias_names = set(names)

    if len(names) != len(alias_names):
        raise ValueError("duplicate MCP compatibility alias name")

    for alias in validated:
        if not alias.name or not alias.target:
            raise ValueError("MCP compatibility aliases require names and targets")
        if alias.name in canonical:
            raise ValueError(
                f"MCP compatibility alias {alias.name!r} collides with a canonical tool"
            )
        if alias.target in alias_names:
            raise ValueError(
                f"MCP compatibility alias {alias.name!r} targets alias {alias.target!r}"
            )
        if alias.target not in canonical:
            raise ValueError(
                f"MCP compatibility alias {alias.name!r} has missing target {alias.target!r}"
            )
        if alias.state != "callable_deprecated":
            raise ValueError(
                f"MCP compatibility alias {alias.name!r} has unsupported state {alias.state!r}"
            )
        if alias.contract_status not in {"unverified", "verified_equivalent"}:
            raise ValueError(
                f"MCP compatibility alias {alias.name!r} has invalid contract status"
            )
        if alias.contract_status == "verified_equivalent" and (
            alias.input_schema is None or alias.effects is None
        ):
            raise ValueError(
                f"verified MCP compatibility alias {alias.name!r} requires schema and effects"
            )
        if alias.input_schema is not None:
            _validate_input_schema(alias)
        if alias.remove_after is not None and alias.deprecated_since is None:
            raise ValueError(
                f"MCP compatibility alias {alias.name!r} cannot set remove_after without deprecated_since"
            )
        if not alias.reason.strip():
            raise ValueError(
                f"MCP compatibility alias {alias.name!r} requires a reason"
            )
        _validate_availability(
            alias,
            capabilities=capabilities,
            programs=programs,
            mode_capabilities=mode_capabilities,
        )

    return validated


def alias_dispatch_map(
    aliases: Iterable[ToolAlias],
) -> dict[str, tuple[str, ArgumentAdapter]]:
    return {
        alias.name: (alias.target, alias.translate_arguments)
        for alias in aliases
    }


def is_alias_available(
    alias: ToolAlias,
    *,
    mode: str,
    programs: Iterable[str] | None,
    mode_capabilities: Mapping[str, frozenset[str]],
) -> bool:
    if alias.availability.capability not in mode_capabilities[mode]:
        return False
    if programs is None or alias.availability.program == "generic":
        return True
    program_set = {
        program.strip().lower()
        for program in programs
        if program and program.strip()
    }
    return alias.availability.program in program_set


def _validate_input_schema(alias: ToolAlias) -> None:
    schema = alias.input_schema
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        raise ValueError(
            f"MCP compatibility alias {alias.name!r} requires an object input schema"
        )
    if not isinstance(schema.get("properties", {}), Mapping):
        raise ValueError(
            f"MCP compatibility alias {alias.name!r} has invalid schema properties"
        )


def _validate_availability(
    alias: ToolAlias,
    *,
    capabilities: Mapping[str, str],
    programs: Mapping[str, str],
    mode_capabilities: Mapping[str, frozenset[str]],
) -> None:
    target_capability = capabilities.get(alias.target)
    if target_capability is None:
        raise ValueError(
            f"MCP compatibility alias {alias.name!r} target lacks capability metadata"
        )
    alias_modes = {
        mode
        for mode, allowed in mode_capabilities.items()
        if alias.availability.capability in allowed
    }
    target_modes = {
        mode
        for mode, allowed in mode_capabilities.items()
        if target_capability in allowed
    }
    if not alias_modes or not alias_modes <= target_modes:
        raise ValueError(
            f"MCP compatibility alias {alias.name!r} is broader than its target by mode"
        )

    target_program = programs.get(alias.target)
    if target_program is None:
        raise ValueError(
            f"MCP compatibility alias {alias.name!r} target lacks program metadata"
        )
    if target_program != "generic" and alias.availability.program != target_program:
        raise ValueError(
            f"MCP compatibility alias {alias.name!r} is broader than its target by program"
        )


__all__ = [
    "CompatibilityAvailability",
    "HIDDEN_TOOL_ALIASES",
    "ToolAlias",
    "ToolEffects",
    "alias_dispatch_map",
    "is_alias_available",
    "validate_tool_aliases",
]

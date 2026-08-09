"""Validated metadata and behavior for hidden MCP compatibility names."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from chemtools.mcp import dispatch
from chemtools.mcp.compatibility import (
    CompatibilityAvailability,
    ToolAlias,
    validate_tool_aliases,
)
from chemtools.mcp.decorator import (
    _TOOL_CAPABILITIES,
    _TOOL_PROGRAMS,
    set_active_mode,
    set_active_programs,
    set_active_toolset,
)
from chemtools.mcp.modes import MODE_CAPABILITIES


EXPECTED_TARGETS = {
    "render_with_orbitron": "visualize",
    "search_knowledge_cards": "search_knowledge",
    "diagnose_nwchem_output": "analyze_nwchem_case",
    "summarize_nwchem_case": "analyze_nwchem_case",
    "review_nwchem_case": "analyze_nwchem_case",
    "check_nwchem_run_status": "get_nwchem_run_status",
    "review_nwchem_followup_outcome": "compare_nwchem_runs",
    "suggest_nwchem_scf_fix_strategy": "suggest_nwchem_recovery",
    "suggest_nwchem_state_recovery_strategy": "suggest_nwchem_recovery",
    "prepare_nwchem_run": "launch_nwchem_run",
    "render_nwchem_basis_from_input": "render_nwchem_basis_block",
    "summarize_cube_file": "parse_cube_file",
    "resolve_nwchem_ecp": "render_nwchem_ecp_block",
    "render_nwchem_ecp_from_elements": "render_nwchem_ecp_block",
    "resolve_nwchem_basis_setup": "render_nwchem_basis_setup",
}


def test_legacy_nwchem_module_still_imports_server_state_names():
    from chemtools.mcp import nwchem

    assert nwchem.ACTIVE_MODE == "analysis"
    assert nwchem.ACTIVE_PROGRAMS is None
    assert nwchem.handle_request is dispatch.handle_request


@pytest.fixture(autouse=True)
def analysis_surface():
    set_active_mode("analysis")
    set_active_programs(None)
    set_active_toolset(None)
    yield
    set_active_mode("analysis")
    set_active_programs(None)
    set_active_toolset(None)


def test_hidden_alias_registry_pins_names_targets_and_availability():
    aliases = {
        alias.name: alias
        for alias in dispatch._TOOL_ALIAS_REGISTRY
    }

    assert {name: alias.target for name, alias in aliases.items()} == (
        EXPECTED_TARGETS
    )
    assert set(dispatch._TOOL_ALIASES) == set(EXPECTED_TARGETS)
    assert all(alias.contract_status == "unverified" for alias in aliases.values())
    assert all(alias.input_schema is None for alias in aliases.values())
    assert all(alias.effects is None for alias in aliases.values())
    assert all(alias.deprecated_since == "0.1.0" for alias in aliases.values())
    assert all(alias.remove_after is None for alias in aliases.values())
    for alias in aliases.values():
        assert alias.availability.program == _TOOL_PROGRAMS[alias.target]
        assert alias.availability.capability == _TOOL_CAPABILITIES[alias.target]
        assert alias.reason


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        (
            "summarize_nwchem_case",
            {"output_path": "run.out", "compact": True},
            {"output_path": "run.out", "detail": "compact"},
        ),
        (
            "review_nwchem_case",
            {"output_path": "run.out", "compact": False},
            {"output_path": "run.out"},
        ),
        (
            "suggest_nwchem_scf_fix_strategy",
            {"output_path": "run.out", "mode": "state"},
            {"output_path": "run.out", "mode": "scf"},
        ),
        (
            "suggest_nwchem_state_recovery_strategy",
            {"output_path": "run.out", "mode": "scf"},
            {"output_path": "run.out", "mode": "state"},
        ),
        (
            "summarize_cube_file",
            {"path": "density.cube", "summarize": False},
            {"path": "density.cube", "summarize": True},
        ),
    ],
)
def test_argument_adapters_pin_historical_defaults(name, arguments, expected):
    alias = next(
        item
        for item in dispatch._TOOL_ALIAS_REGISTRY
        if item.name == name
    )
    original = dict(arguments)

    assert alias.translate_arguments(arguments) == expected
    assert alias.translate_arguments(arguments) == expected
    assert arguments == original


def test_identity_adapters_copy_arguments_without_changing_values():
    translated_aliases = {
        "summarize_nwchem_case",
        "review_nwchem_case",
        "suggest_nwchem_scf_fix_strategy",
        "suggest_nwchem_state_recovery_strategy",
        "summarize_cube_file",
    }
    arguments = {"path": "run.out"}

    for alias in dispatch._TOOL_ALIAS_REGISTRY:
        if alias.name in translated_aliases:
            continue
        translated = alias.translate_arguments(arguments)
        assert translated == arguments
        assert translated is not arguments


def test_tools_call_uses_registry_argument_translation(monkeypatch):
    monkeypatch.setitem(
        dispatch._TOOL_REGISTRY,
        "suggest_nwchem_recovery",
        lambda arguments: {"received": arguments},
    )

    response, should_exit = dispatch.handle_request({
        "jsonrpc": "2.0",
        "id": 17,
        "method": "tools/call",
        "params": {
            "name": "suggest_nwchem_scf_fix_strategy",
            "arguments": {"output_path": "run.out"},
        },
    })

    assert should_exit is False
    assert response == {
        "jsonrpc": "2.0",
        "id": 17,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "received": {
                                "output_path": "run.out",
                                "mode": "scf",
                            }
                        },
                        separators=(",", ":"),
                    ),
                }
            ],
            "isError": False,
        },
    }


def test_tools_call_intersects_alias_and_target_program_scope(monkeypatch):
    alias = next(
        item
        for item in dispatch._TOOL_ALIAS_REGISTRY
        if item.name == "search_knowledge_cards"
    )
    restricted = replace(
        alias,
        availability=replace(alias.availability, program="nwchem"),
    )
    monkeypatch.setitem(
        dispatch._TOOL_ALIAS_REGISTRY_BY_NAME,
        alias.name,
        restricted,
    )
    set_active_programs({"molcas"})

    response, should_exit = dispatch.handle_request({
        "jsonrpc": "2.0",
        "id": 23,
        "method": "tools/call",
        "params": {
            "name": "search_knowledge_cards",
            "arguments": {"query": "failure"},
        },
    })

    assert should_exit is False
    assert response == {
        "jsonrpc": "2.0",
        "id": 23,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "tool 'search_knowledge_cards' (program=nwchem) is not "
                        "in the active program filter ['molcas']. Restart with "
                        "CHEMTOOLS_PROGRAMS including 'nwchem' to enable it."
                    ),
                }
            ],
            "isError": True,
        },
    }


def test_alias_validation_rejects_collisions_chains_and_missing_targets():
    valid = _test_alias()

    with pytest.raises(ValueError, match="collides with a canonical tool"):
        _validate(replace(valid, name="canonical"))
    with pytest.raises(ValueError, match="targets alias"):
        _validate((replace(valid, target="middle"), _test_alias(name="middle")))
    with pytest.raises(ValueError, match="missing target"):
        _validate(replace(valid, target="missing"))


def test_alias_validation_rejects_broader_mode_or_program_scope():
    valid = _test_alias(capability="executable")

    with pytest.raises(ValueError, match="broader than its target by mode"):
        _validate(valid, target_capability="executable", alias_capability="none")
    with pytest.raises(ValueError, match="broader than its target by program"):
        _validate(replace(valid, availability=CompatibilityAvailability(
            program="generic",
            capability="executable",
        )))


def test_verified_alias_requires_recovered_schema_and_effects():
    alias = replace(_test_alias(), contract_status="verified_equivalent")

    with pytest.raises(ValueError, match="requires schema and effects"):
        _validate(alias)


def _test_alias(
    *,
    name: str = "old_name",
    capability: str = "none",
) -> ToolAlias:
    return ToolAlias(
        name=name,
        target="canonical",
        input_schema=None,
        translate_arguments=dict,
        translate_result=None,
        availability=CompatibilityAvailability(
            program="nwchem",
            capability=capability,
        ),
        effects=None,
        contract_status="unverified",
        deprecated_since=None,
        remove_after=None,
        reason="renamed",
    )


def _validate(
    aliases,
    *,
    target_capability: str = "none",
    alias_capability: str | None = None,
):
    if isinstance(aliases, ToolAlias):
        aliases = (aliases,)
    if alias_capability is not None:
        aliases = tuple(
            replace(
                alias,
                availability=replace(
                    alias.availability,
                    capability=alias_capability,
                ),
            )
            for alias in aliases
        )
    return validate_tool_aliases(
        aliases,
        canonical_names={"canonical"},
        capabilities={"canonical": target_capability},
        programs={"canonical": "nwchem"},
        mode_capabilities=MODE_CAPABILITIES,
    )

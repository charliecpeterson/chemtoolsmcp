"""The guided preset exposes only the stable high-level workflow tools."""

import pytest
from jsonschema.validators import validator_for

from chemtools.mcp import modes
from chemtools.mcp.decorator import (
    _TOOL_CAPABILITIES,
    _TOOL_PROGRAMS,
    _TOOL_REGISTRY,
)
from chemtools.mcp.dispatch import tool_definitions
from chemtools.mcp.tools import guided


def _contract(name):
    return {
        "name": name,
        "description": f"Contract for {name}",
        "inputSchema": {"type": "object"},
    }


def _handler(name):
    def implementation(arguments):
        return arguments

    implementation.__name__ = f"_handle_{name}"
    return implementation


def test_guided_contracts_are_bound_to_the_exact_handlers():
    definitions = guided.guided_tool_definitions()

    assert [definition["name"] for definition in definitions] == [
        "review_input",
        "inspect_run",
        "compare_runs",
        "plan_recovery",
        "plan_calculation",
        "launch_run",
        "monitor_run",
        "draft_input",
    ]
    for definition in definitions:
        name = definition["name"]
        assert _TOOL_REGISTRY[name] is getattr(guided, f"_handle_{name}")


def test_guided_binding_rejects_duplicate_definitions():
    with pytest.raises(ValueError, match="duplicate guided tool definition"):
        guided._GuidedToolBindings([_contract("review"), _contract("review")])


def test_guided_binding_rejects_a_handler_without_a_definition(monkeypatch):
    bindings = guided._GuidedToolBindings([_contract("review")])
    monkeypatch.setattr(
        guided,
        "_tool",
        lambda name, program: lambda function: function,
    )

    with pytest.raises(ValueError, match="has no tool definition"):
        bindings.handler(_handler("inspect"))


def test_guided_binding_rejects_missing_and_duplicate_handlers(monkeypatch):
    bindings = guided._GuidedToolBindings([
        _contract("review"),
        _contract("inspect"),
    ])
    monkeypatch.setattr(
        guided,
        "_tool",
        lambda name, program: lambda function: function,
    )
    bindings.handler(_handler("review"))

    with pytest.raises(ValueError, match="definitions have no handlers"):
        bindings.definitions()
    with pytest.raises(ValueError, match="duplicate guided tool handler"):
        bindings.handler(_handler("review"))


def test_guided_toolset_resolves_to_exact_public_names():
    names, reason = modes.resolve_toolset("guided", env={})

    assert names == frozenset({
        "review_input",
        "inspect_run",
        "compare_runs",
        "plan_recovery",
        "plan_calculation",
        "launch_run",
        "monitor_run",
        "draft_input",
        "search_knowledge",
        "find_reference_case",
        "visualize",
    })
    assert reason == "preset 'guided' (11 tools)"


def test_guided_toolset_is_the_command_line_default():
    names, reason = modes.resolve_toolset(None, env={})

    assert names == modes.TOOLSETS["guided"]
    assert reason == "default preset 'guided' (11 tools)"


@pytest.mark.parametrize("name", ["developer", "full"])
def test_developer_toolset_explicitly_selects_the_full_surface(name):
    names, reason = modes.resolve_toolset(name, env={})

    assert names is None
    assert reason == f"preset {name!r} (full developer tool surface)"


def test_guided_toolset_filters_analysis_surface_to_eleven_tools():
    names = modes.TOOLSETS["guided"]

    visible = modes.filter_tools(
        tool_definitions(),
        _TOOL_CAPABILITIES,
        "analysis",
        program_tags=_TOOL_PROGRAMS,
        toolset=names,
    )

    assert [definition["name"] for definition in visible] == [
        "review_input",
        "inspect_run",
        "compare_runs",
        "plan_recovery",
        "plan_calculation",
        "launch_run",
        "monitor_run",
        "draft_input",
        "visualize",
        "search_knowledge",
        "find_reference_case",
    ]


def test_guided_output_schemas_are_valid_and_reject_empty_results():
    definitions = {
        definition["name"]: definition
        for definition in tool_definitions()
        if definition["name"] in modes.TOOLSETS["guided"]
    }

    assert set(definitions) == modes.TOOLSETS["guided"]
    for definition in definitions.values():
        schema = definition["outputSchema"]
        validator_class = validator_for(schema)
        validator_class.check_schema(schema)
        assert list(validator_class(schema).iter_errors({}))


def test_guided_error_results_conform_to_advertised_alternatives():
    definitions = {
        definition["name"]: definition
        for definition in tool_definitions()
        if definition["name"] in modes.TOOLSETS["guided"]
    }

    for name, definition in definitions.items():
        schema = definition["outputSchema"]
        validator_for(schema)(schema).validate({
            "error": "representative_error",
            "message": "Representative boundary failure.",
        })


def test_custom_toolset_normalizes_hidden_knowledge_alias():
    names, reason = modes.resolve_toolset(
        "search_knowledge_cards",
        env={},
        aliases={"search_knowledge_cards": "search_knowledge"},
    )

    assert names == frozenset({"search_knowledge"})
    assert reason == (
        "custom list (1 tool); normalized 1 compatibility alias"
    )


def test_custom_toolset_normalizes_hidden_visualize_alias():
    names, reason = modes.resolve_toolset(
        "render_with_orbitron",
        env={},
        aliases={"render_with_orbitron": "visualize"},
    )

    assert names == frozenset({"visualize"})
    assert reason == (
        "custom list (1 tool); normalized 1 compatibility alias"
    )


def test_inspect_run_schema_bounds_explicit_artifact_paths():
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "inspect_run"
    )

    assert definition["inputSchema"]["properties"]["artifact_files"] == {
        "type": "array",
        "items": {
            "type": "string",
            "minLength": 1,
        },
        "maxItems": 64,
        "description": (
            "Optional paths to related inputs, stderr, checkpoints, orbitals, "
            "or other run artifacts. Paths are classified and observed in "
            "the supplied order. Directories are never scanned."
        ),
    }


def test_draft_input_schema_requires_complete_inline_geometry():
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "draft_input"
    )

    assert definition["inputSchema"]["required"] == [
        "program",
        "atoms",
        "charge",
        "multiplicity",
        "method",
        "basis",
        "task",
    ]
    atoms = definition["inputSchema"]["properties"]["atoms"]
    assert atoms["maxItems"] == 2048
    assert atoms["items"]["required"] == ["element", "x", "y", "z"]
    assert "solvent" not in definition["inputSchema"]["properties"]
    assert "ecp" not in definition["inputSchema"]["properties"]


def test_plan_recovery_schema_is_read_only_and_bounds_target_state():
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "plan_recovery"
    )
    schema = definition["inputSchema"]

    assert schema["required"] == ["output_file"]
    assert schema["properties"]["program"]["enum"] == ["nwchem"]
    assert schema["properties"]["expected_multiplicity"]["minimum"] == 1
    assert schema["properties"]["expected_somo_count"]["minimum"] == 0
    assert schema["properties"]["expected_metal_elements"]["maxItems"] == 32
    assert "write_file" not in schema["properties"]
    assert "output_dir" not in schema["properties"]
    assert schema["additionalProperties"] is False


def test_plan_calculation_schema_is_bounded_and_read_only():
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "plan_calculation"
    )
    schema = definition["inputSchema"]

    assert schema["required"] == [
        "program",
        "system",
        "elements",
        "charge",
        "multiplicity",
        "stages",
    ]
    assert schema["properties"]["elements"]["maxItems"] == 32
    assert schema["properties"]["stages"]["maxItems"] == 8
    assert schema["properties"]["stages"]["items"]["enum"] == [
        "energy",
        "optimize",
        "frequency",
    ]
    assert "input_file" not in schema["properties"]
    assert "output_file" not in schema["properties"]
    assert "write_file" not in schema["properties"]
    assert schema["additionalProperties"] is False


def test_launch_run_schema_pins_approval_and_target_selection():
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "launch_run"
    )
    schema = definition["inputSchema"]

    assert schema["required"] == ["program", "input_file"]
    assert schema["properties"]["profile"]["minLength"] == 1
    assert schema["properties"]["target"]["minLength"] == 1
    assert schema["properties"]["approval_token"]["pattern"] == (
        "^sha256:[0-9a-f]{64}$"
    )
    assert schema["properties"]["resources"]["additionalProperties"] is False
    assert schema["properties"]["initialization_only"] == {
        "type": "boolean",
        "default": False,
        "description": (
            "QMCPACK only: append --dryrun so QMCPACK initializes the input "
            "but skips QMC sections."
        ),
    }
    assert "env_overrides" not in schema["properties"]
    assert definition["annotations"] == {
        "title": "Prepare or launch an approved calculation",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    }


def test_monitor_run_schema_accepts_only_owned_launch_identity():
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "monitor_run"
    )
    schema = definition["inputSchema"]

    assert schema["required"] == ["launch_id"]
    assert set(schema["properties"]) == {"launch_id"}
    assert schema["additionalProperties"] is False
    assert definition["annotations"] == {
        "title": "Monitor an owned calculation",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }


def test_other_guided_tools_are_accurately_annotated_read_only():
    definitions = {
        item["name"]: item
        for item in tool_definitions()
    }

    for name in modes.TOOLSETS["guided"] - {"launch_run", "monitor_run"}:
        annotations = definitions[name]["annotations"]
        assert annotations["readOnlyHint"] is True
        assert annotations["destructiveHint"] is False
        assert annotations["idempotentHint"] is True
        assert annotations["openWorldHint"] is False

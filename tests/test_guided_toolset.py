"""The guided preset exposes only the stable high-level workflow tools."""

from chemtools.mcp import modes
from chemtools.mcp.decorator import (
    _TOOL_CAPABILITIES,
    _TOOL_PROGRAMS,
)
from chemtools.mcp.dispatch import tool_definitions


def test_guided_toolset_resolves_to_exact_public_names():
    names, reason = modes.resolve_toolset("guided", env={})

    assert names == frozenset({
        "review_input",
        "inspect_run",
        "search_knowledge_cards",
    })
    assert reason == "preset 'guided' (3 tools)"


def test_guided_toolset_filters_analysis_surface_to_two_tools():
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
        "search_knowledge_cards",
    ]


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

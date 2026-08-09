"""Tool-registration integrity.

Self-contained (no external fixtures). These catch the failure modes a handler
refactor or a bad tool addition introduces: an import-time crash in a tool
module, a schema with no handler, or a duplicate tool name.
"""
from chemtools.mcp.dispatch import tool_definitions, _TOOL_ALIASES
from chemtools.mcp.decorator import _TOOL_REGISTRY


def test_tool_modules_import():
    # Importing each program's tool module runs its @_tool decorators; an
    # import-time NameError/ImportError surfaces here.
    import chemtools.mcp.tools.nwchem   # noqa: F401
    import chemtools.mcp.tools.molcas   # noqa: F401
    import chemtools.mcp.tools.dirac    # noqa: F401
    import chemtools.mcp.tools.grasp    # noqa: F401
    import chemtools.mcp.tools.generic  # noqa: F401


def test_definitions_wellformed_and_unique():
    defs = tool_definitions()
    assert len(defs) >= 250
    names = [d["name"] for d in defs]
    assert len(names) == len(set(names)), "duplicate tool names"
    for d in defs:
        assert d["name"] and d["description"]
        assert isinstance(d["inputSchema"], dict)


def test_every_definition_has_a_handler():
    dispatchable = set(_TOOL_REGISTRY) | set(_TOOL_ALIASES)
    missing = [d["name"] for d in tool_definitions() if d["name"] not in dispatchable]
    assert not missing, f"schemas with no handler: {missing}"


def test_legacy_handler_imports_resolve():
    # External callers may still use the old aggregator import path. The
    # __getattr__ shim must keep resolving it during the compatibility window.
    from chemtools.mcp.tools.nwchem import (  # noqa: F401
        _handle_analyze_nwchem_case,
        _handle_summarize_nwchem_output,
        _handle_suggest_nwchem_recovery,
        _do_create_campaign,
    )

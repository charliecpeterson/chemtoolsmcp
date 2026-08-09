"""Tool-registration integrity.

Self-contained (no external fixtures). These catch the failure modes a handler
refactor or a bad tool addition introduces: an import-time crash in a tool
module, a schema with no handler, or a duplicate tool name.
"""
import importlib.util

from chemtools.mcp.dispatch import tool_definitions, _TOOL_ALIASES
from chemtools.mcp.decorator import _TOOL_REGISTRY


def test_tool_modules_import():
    # Importing each program's tool module runs its @_tool decorators; an
    # import-time NameError/ImportError surfaces here.
    import chemtools.mcp.tools._nwchem_provider  # noqa: F401
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


def test_removed_nwchem_aggregator_is_not_importable():
    assert importlib.util.find_spec("chemtools.mcp.tools.nwchem") is None
    assert importlib.util.find_spec("chemtools.mcp.tools._nwchem_base") is None

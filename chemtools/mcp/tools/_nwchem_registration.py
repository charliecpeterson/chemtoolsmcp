"""NWChem-specific registration for MCP tool handlers."""

from __future__ import annotations

from chemtools.mcp.decorator import _tool as _register_tool


def _tool(name: str, *, needs: str = "none", program: str = "nwchem"):
    return _register_tool(name, needs=needs, program=program)

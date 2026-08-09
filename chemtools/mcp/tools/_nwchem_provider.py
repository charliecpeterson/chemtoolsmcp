"""Register focused NWChem MCP handlers and expose their tool definitions."""

from __future__ import annotations

from chemtools.mcp.tools import (
    nwchem_analysis,
    nwchem_docs,
    nwchem_input,
    nwchem_jobs,
    nwchem_parse,
)
from chemtools.mcp.tools._nwchem_schemas import _nwchem_tool_definitions


__all__ = ["_nwchem_tool_definitions"]

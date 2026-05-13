"""Back-compat shim for the renamed CLI entry point.

The chemtools MCP CLI now lives at ``chemtools.mcp.cli``. This module
re-exports the public surface so existing tests and external code that
still imports ``from chemtools.mcp.nwchem import ...`` keeps working.
"""
from __future__ import annotations

from chemtools.mcp.cli import (  # noqa: F401
    ACTIVE_MODE,
    ACTIVE_PROGRAMS,
    main,
    main_legacy_nwchem,
    serve,
    tool_definitions,
    dispatch_tool,
    handle_request,
    _TOOL_ALIASES,
)
# Tests reach into the decorator registry via ``mcp.nwchem._TOOL_CAPABILITIES``
# so keep that path live too.
from chemtools.mcp.decorator import (  # noqa: F401
    _TOOL_REGISTRY,
    _TOOL_CAPABILITIES,
    _TOOL_PROGRAMS,
)

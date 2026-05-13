"""Per-program MCP tool registration modules.

Each program's tool definitions and ``@_tool`` handlers live in
``chemtools.mcp.tools.<program>``. Importing one of these modules causes
its handlers to register with the shared registry in
``chemtools.mcp.decorator._TOOL_REGISTRY``.

This package eagerly imports every program module + the generic module so
``@_tool`` registrations are guaranteed to have run before
``chemtools.mcp.dispatch.tool_definitions()`` is first called.
"""
from chemtools.mcp.tools import generic   # noqa: F401
from chemtools.mcp.tools import nwchem    # noqa: F401
from chemtools.mcp.tools import molcas    # noqa: F401
from chemtools.mcp.tools import dirac     # noqa: F401
from chemtools.mcp.tools import grasp     # noqa: F401

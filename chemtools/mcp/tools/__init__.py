"""Per-program MCP tool registration modules.

Each program's tool definitions and @_tool handlers live in
`chemtools.mcp.tools.<program>`. Importing one of these modules causes
its handlers to register with the shared registry in
`chemtools.mcp.decorator._TOOL_REGISTRY`.

Per-program CLI entry points (chemtools-<program>) import only their
own tool module so the tool list is scoped to one program.
"""

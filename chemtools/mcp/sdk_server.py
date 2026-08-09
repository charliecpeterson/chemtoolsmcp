"""Official MCP SDK adapter for the Chemtools registry and server state."""

from __future__ import annotations

import base64
import json
from typing import Any

import anyio
from mcp import types
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server

from chemtools.mcp.decorator import SERVER_NAME, SERVER_VERSION, log_event
from chemtools.mcp.dispatch import (
    dispatch_tool,
    tool_unavailable_message,
    visible_tool_definitions,
)
from chemtools.mcp.server import ImageToolResult
from chemtools.mcp.state import ServerState


def _success_result(
    payload: dict[str, Any] | ImageToolResult,
) -> types.CallToolResult:
    if isinstance(payload, ImageToolResult):
        content: list[types.TextContent | types.ImageContent] = [
            types.TextContent(
                text=json.dumps(payload.payload, separators=(",", ":")),
            ),
            types.ImageContent(
                data=base64.b64encode(payload.image).decode("ascii"),
                mimeType="image/png",
            ),
        ]
        structured_content = payload.payload
    else:
        content = [
            types.TextContent(
                text=json.dumps(payload, separators=(",", ":")),
            ),
        ]
        structured_content = payload
    return types.CallToolResult(
        content=content,
        structuredContent=structured_content,
        isError=False,
    )


def _error_result(message: str, *, code: str) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(text=message)],
        structuredContent={"error": code, "message": message},
        isError=True,
    )


def create_server(state: ServerState) -> Server[Any]:
    """Build one low-level SDK server bound to explicit runtime state."""

    async def list_tools(
        context: ServerRequestContext[Any],
        params: types.PaginatedRequestParams | None,
    ) -> types.ListToolsResult:
        del context, params
        definitions = visible_tool_definitions(state)
        log_event(
            f"tools/list requested mode={state.mode} programs={state.programs} "
            f"toolset={'set' if state.toolset else None}"
        )
        return types.ListToolsResult(
            tools=[types.Tool.model_validate(item) for item in definitions],
        )

    async def call_tool(
        context: ServerRequestContext[Any],
        params: types.CallToolRequestParams,
    ) -> types.CallToolResult:
        del context
        arguments = params.arguments or {}
        log_event(f"tools/call name={params.name} mode={state.mode}")
        unavailable = tool_unavailable_message(params.name, state)
        if unavailable is not None:
            return _error_result(unavailable, code="tool_unavailable")
        try:
            return _success_result(
                dispatch_tool(params.name, arguments, state=state),
            )
        except Exception as exc:
            log_event(f"tools/call error name={params.name} error={exc}")
            return _error_result(str(exc), code="tool_execution_error")

    return Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


def serve_stdio(state: ServerState) -> None:
    """Run one Chemtools SDK server over standard input and output."""
    server = create_server(state)

    async def run() -> None:
        async with stdio_server() as streams:
            await server.run(
                streams[0],
                streams[1],
                server.create_initialization_options(),
            )

    anyio.run(run)


__all__ = ["create_server", "serve_stdio"]

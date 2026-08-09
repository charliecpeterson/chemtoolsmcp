"""Official SDK and legacy docs-server protocol regressions."""

from __future__ import annotations

import anyio
from mcp import Client
from mcp_types.version import (
    HANDSHAKE_PROTOCOL_VERSIONS,
    MODERN_PROTOCOL_VERSIONS,
)

from chemtools.mcp.nwchem_docs import handle_request as handle_docs_request
from chemtools.mcp.sdk_server import create_server
from chemtools.mcp.server import (
    DEFAULT_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
)
from chemtools.mcp.state import ServerState


def _initialize(version: str | None) -> dict:
    params = {
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "1.0"},
    }
    if version is not None:
        params["protocolVersion"] = version
    return {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "initialize",
        "params": params,
    }


def test_supported_protocol_versions_are_pinned():
    assert SUPPORTED_PROTOCOL_VERSIONS == (
        *HANDSHAKE_PROTOCOL_VERSIONS,
        *MODERN_PROTOCOL_VERSIONS,
    )
    assert DEFAULT_PROTOCOL_VERSION == HANDSHAKE_PROTOCOL_VERSIONS[-1]


def test_main_server_negotiates_with_the_official_client():
    async def connect() -> tuple[str, str, list[str]]:
        state = ServerState.create(toolset={"get_server_mode"})
        async with Client(create_server(state)) as client:
            listed = await client.list_tools()
            return (
                client.protocol_version,
                client.server_info.name,
                [tool.name for tool in listed.tools],
            )

    protocol_version, server_name, tools = anyio.run(connect)

    assert protocol_version == MODERN_PROTOCOL_VERSIONS[-1]
    assert server_name == "chemtools"
    assert tools == ["get_server_mode"]


def test_structured_tool_errors_conform_to_the_advertised_output_schema():
    async def call_invalid_tool():
        state = ServerState.create(toolset={"search_knowledge"})
        async with Client(create_server(state)) as client:
            await client.list_tools()
            return await client.call_tool(
                "search_knowledge",
                {"unknown_filter": "value"},
            )

    result = anyio.run(call_invalid_tool)

    assert result.is_error is True
    assert result.structured_content == {
        "error": "tool_execution_error",
        "message": "unknown search arguments: ['unknown_filter']",
    }
    assert result.content[0].text == result.structured_content["message"]


def test_legacy_docs_server_uses_the_same_negotiation_policy():
    response, should_exit = handle_docs_request(_initialize("2099-01-01"))

    assert should_exit is False
    assert response["result"]["protocolVersion"] == DEFAULT_PROTOCOL_VERSION
    assert response["result"]["serverInfo"] == {
        "name": "nwchem-docs-mcp",
        "version": "0.1.0",
    }

"""Explicit MCP server state keeps filters and launch ownership together."""

import json

from chemtools.mcp import dispatch
from chemtools.mcp.decorator import get_execution_service
from chemtools.mcp.state import ServerState


def test_explicit_state_filters_tools_without_mutating_other_servers():
    guided = ServerState.create(
        mode="analysis",
        programs={"nwchem"},
        toolset={"review_input", "draft_input"},
    )
    unfiltered = ServerState.create(mode="analysis")
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
    }

    guided_response, _ = dispatch.handle_request(request, state=guided)
    unfiltered_response, _ = dispatch.handle_request(request, state=unfiltered)

    assert [
        definition["name"]
        for definition in guided_response["result"]["tools"]
    ] == ["review_input", "draft_input"]
    assert len(unfiltered_response["result"]["tools"]) > 2


def test_dispatch_binds_the_state_execution_service_for_handlers(monkeypatch):
    local_state = ServerState.create(mode="local")
    analysis_state = ServerState.create(mode="analysis")
    monkeypatch.setitem(
        dispatch._TOOL_REGISTRY,
        "state_probe",
        lambda arguments: {
            "instance_id": get_execution_service().instance_id,
            "enabled": get_execution_service().enable_execution,
        },
    )

    local = dispatch.dispatch_tool("state_probe", {}, state=local_state)
    analysis = dispatch.dispatch_tool("state_probe", {}, state=analysis_state)

    assert local == {
        "instance_id": local_state.execution_service.instance_id,
        "enabled": True,
    }
    assert analysis == {
        "instance_id": analysis_state.execution_service.instance_id,
        "enabled": False,
    }
    assert local["instance_id"] != analysis["instance_id"]


def test_get_server_mode_reads_the_request_state():
    state = ServerState.create(
        mode="hpc",
        programs={"molcas"},
        toolset={"get_server_mode"},
    )
    response, should_exit = dispatch.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "get_server_mode", "arguments": {}},
        },
        state=state,
    )

    assert should_exit is False
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["mode"] == "hpc"
    assert payload["programs"] == ["molcas"]
    assert payload["toolset"] == ["get_server_mode"]

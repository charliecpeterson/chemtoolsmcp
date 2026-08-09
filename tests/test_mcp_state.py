"""Explicit MCP server state keeps filters and launch ownership together."""

import json
from pathlib import Path

from chemtools.mcp import dispatch
from chemtools.mcp.decorator import get_execution_service
from chemtools.mcp.state import ServerState
from chemtools.execution.targets import parse_target_catalog


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
    assert payload["execution_enabled"] is True
    assert payload["default_target"] is None
    assert payload["targets"] == []


def test_state_binds_named_targets_and_explicit_permission(tmp_path):
    catalog = parse_target_catalog(
        {
            "schema_version": "2.0",
            "chemtools": {
                "enable_execution": True,
                "default_target": "workstation",
            },
            "targets": {
                "workstation": {
                    "executor": "local",
                    "allowed_work_roots": [str(tmp_path)],
                    "programs": {
                        "nwchem": {
                            "executable_argv": ["nwchem"],
                        },
                    },
                },
            },
        },
        source=Path(tmp_path) / "targets.yaml",
    )

    state = ServerState.create(
        mode="analysis",
        target_catalog=catalog,
    )

    assert state.mode == "analysis"
    assert state.execution_service.enable_execution is True
    assert state.execution_service.resolve_target(
        program="nwchem"
    ).name == "workstation"

    response, _ = dispatch.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "get_server_mode",
                "arguments": {},
            },
        },
        state=state,
    )
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["execution_enabled"] is True
    assert payload["default_target"] == "workstation"
    assert payload["targets"] == [{
        "name": "workstation",
        "executor": "local",
        "programs": ["nwchem"],
    }]


def test_dispatch_preserves_qmcpack_initialization_only(tmp_path):
    input_path = tmp_path / "hydrogen.xml"
    input_path.write_text(
        '<simulation><qmc method="vmc"/></simulation>\n',
        encoding="utf-8",
    )
    catalog = parse_target_catalog(
        {
            "schema_version": "2.0",
            "chemtools": {
                "enable_execution": False,
                "default_target": "workstation",
            },
            "targets": {
                "workstation": {
                    "executor": "local",
                    "allowed_work_roots": [str(tmp_path)],
                    "programs": {
                        "qmcpack": {
                            "executable_argv": ["qmcpack"],
                        },
                    },
                },
            },
        },
        source=tmp_path / "targets.yaml",
    )
    state = ServerState.create(mode="analysis", target_catalog=catalog)

    prepared = dispatch.dispatch_tool(
        "launch_run",
        {
            "program": "qmcpack",
            "input_file": str(input_path),
            "initialization_only": True,
        },
        state=state,
    )

    assert prepared["status"] == "awaiting_approval"
    assert prepared["evidence"]["plan"]["argv"] == [
        "qmcpack",
        "hydrogen.xml",
        "--dryrun",
    ]

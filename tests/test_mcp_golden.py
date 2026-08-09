"""Golden JSON-RPC requests for each current program and generic dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
from mcp import Client
import pytest

from chemtools.mcp.sdk_server import create_server
from chemtools.mcp.state import ServerState

GOLDEN_SCHEMA = "chemtools.mcp-golden-case/1"
GOLDEN_ROOT = Path(__file__).parent / "golden" / "mcp"
CASE_PATHS = sorted(GOLDEN_ROOT.glob("*.case.json"))


@pytest.mark.parametrize("case_path", CASE_PATHS, ids=lambda path: path.stem)
def test_mcp_golden_case(case_path):
    case = json.loads(case_path.read_text(encoding="utf-8"))
    assert case["schema"] == GOLDEN_SCHEMA
    fixture = (case_path.parent / case["fixture"]).resolve()
    assert fixture.is_file()
    request = _replace_fixture(case["request"], fixture)

    async def call_tool():
        state = ServerState.create(mode="analysis", toolset=None)
        async with Client(create_server(state)) as client:
            await client.list_tools()
            return await client.call_tool(
                request["params"]["name"],
                request["params"].get("arguments", {}),
            )

    result = anyio.run(call_tool)

    assert result.is_error is False
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    payload = json.loads(result.content[0].text)
    assert result.structured_content == payload
    assert set(payload) == set(case["expected_payload_keys"])
    _assert_subset(case["expected_payload"], payload)


def test_golden_set_covers_each_current_program_and_generic():
    programs = {
        json.loads(path.read_text(encoding="utf-8"))["program"]
        for path in CASE_PATHS
    }

    assert programs == {
        "generic", "nwchem", "molcas", "dirac", "grasp", "qe"
    }
    assert len(CASE_PATHS) == 11


def _replace_fixture(value: Any, fixture: Path) -> Any:
    if value == "$FIXTURE":
        return str(fixture)
    if isinstance(value, dict):
        return {
            key: _replace_fixture(item, fixture)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_fixture(item, fixture) for item in value]
    return value


def _assert_subset(expected: Any, actual: Any, path: str = "payload") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path} is not an object"
        missing = expected.keys() - actual.keys()
        assert not missing, f"{path} is missing {sorted(missing)}"
        for key, value in expected.items():
            _assert_subset(value, actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path} is not a list"
        assert len(actual) == len(expected), f"{path} length changed"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _assert_subset(expected_item, actual_item, f"{path}[{index}]")
        return
    assert actual == expected, f"{path} changed"

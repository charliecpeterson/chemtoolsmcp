"""Golden JSON-RPC requests for each current program and generic dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chemtools.mcp.decorator import (
    set_active_mode,
    set_active_programs,
    set_active_toolset,
)
from chemtools.mcp.dispatch import handle_request

GOLDEN_SCHEMA = "chemtools.mcp-golden-case/1"
GOLDEN_ROOT = Path(__file__).parent / "golden" / "mcp"
CASE_PATHS = sorted(GOLDEN_ROOT.glob("*.case.json"))


@pytest.fixture(autouse=True)
def analysis_mode():
    set_active_mode("analysis")
    set_active_programs(None)
    set_active_toolset(None)
    yield
    set_active_mode("analysis")
    set_active_programs(None)
    set_active_toolset(None)


@pytest.mark.parametrize("case_path", CASE_PATHS, ids=lambda path: path.stem)
def test_mcp_golden_case(case_path):
    case = json.loads(case_path.read_text(encoding="utf-8"))
    assert case["schema"] == GOLDEN_SCHEMA
    fixture = (case_path.parent / case["fixture"]).resolve()
    assert fixture.is_file()
    request = _replace_fixture(case["request"], fixture)

    response, should_exit = handle_request(request)

    assert should_exit is False
    assert set(response) == {"jsonrpc", "id", "result"}
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == request["id"]
    assert set(response["result"]) == {"content", "isError"}
    assert response["result"]["isError"] is False
    assert len(response["result"]["content"]) == 1
    content = response["result"]["content"][0]
    assert set(content) == {"type", "text"}
    assert content["type"] == "text"

    payload = json.loads(content["text"])
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
    assert len(CASE_PATHS) == 8


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

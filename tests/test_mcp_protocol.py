"""MCP initialization and protocol-version negotiation regressions."""

from __future__ import annotations

from chemtools.mcp.dispatch import handle_request
from chemtools.mcp.nwchem_docs import handle_request as handle_docs_request
from chemtools.mcp.server import (
    DEFAULT_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
)


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
    assert SUPPORTED_PROTOCOL_VERSIONS == ("2024-11-05",)
    assert DEFAULT_PROTOCOL_VERSION == SUPPORTED_PROTOCOL_VERSIONS[-1]


def test_main_server_echoes_a_supported_protocol_version():
    response, should_exit = handle_request(_initialize(DEFAULT_PROTOCOL_VERSION))

    assert should_exit is False
    assert response["result"]["protocolVersion"] == DEFAULT_PROTOCOL_VERSION


def test_main_server_negotiates_an_unknown_protocol_version():
    response, should_exit = handle_request(_initialize("2099-01-01"))

    assert should_exit is False
    assert response["result"]["protocolVersion"] == DEFAULT_PROTOCOL_VERSION


def test_main_server_defaults_a_missing_protocol_version():
    response, should_exit = handle_request(_initialize(None))

    assert should_exit is False
    assert response["result"]["protocolVersion"] == DEFAULT_PROTOCOL_VERSION


def test_legacy_docs_server_uses_the_same_negotiation_policy():
    response, should_exit = handle_docs_request(_initialize("2099-01-01"))

    assert should_exit is False
    assert response["result"]["protocolVersion"] == DEFAULT_PROTOCOL_VERSION

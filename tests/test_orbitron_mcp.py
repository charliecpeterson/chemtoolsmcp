"""MCP contracts for fixed, read-only Orbitron operations."""

from __future__ import annotations

import base64
import json

import pytest

from chemtools.integrations.orbitron import (
    OrbitronCommandError,
    OrbitronProtocolError,
    OrbitronRender,
    OrbitronResponse,
    OrbitronUnavailableError,
    OrbitronVersion,
)
from chemtools.mcp.decorator import _TOOL_PROGRAMS
from chemtools.mcp.dispatch import dispatch_tool, tool_definitions
from chemtools.mcp.dispatch import handle_request
from chemtools.mcp.tools import orbitron as orbitron_tools


def test_mcp_returns_versioned_orbitron_evidence(tmp_path, monkeypatch):
    source = tmp_path / "run.out"
    source.write_text("output\n")
    version = OrbitronVersion(
        version="0.4.0",
        commit="58aa65b3f280",
        raw="orbitron-cli 0.4.0 (58aa65b3f280)",
    )
    response = OrbitronResponse(
        operation="inspect",
        source=str(source.resolve()),
        schema="orbitron.inspect/2",
        producer={
            "name": "orbitron",
            "version": "0.4.0",
            "commit": "58aa65b3f280",
        },
        warnings=(
            {"source": "loader", "code": "parser:notice"},
        ),
        payload={
            "schema": "orbitron.inspect/2",
            "producer": {
                "name": "orbitron",
                "version": "0.4.0",
                "commit": "58aa65b3f280",
            },
            "warnings": [{"source": "loader", "code": "parser:notice"}],
            "subject": "output",
            "program": "qe",
            "detected": "scene",
        },
        stderr="",
        version=version,
    )

    class Client:
        def inspect(self, path):
            assert path == str(source)
            return response

    monkeypatch.setattr(orbitron_tools, "OrbitronClient", Client)

    inspected = dispatch_tool("inspect_with_orbitron", {"path": str(source)})

    assert inspected == {
        "schema_version": "chemtools.orbitron-inspection/1",
        "status": "ok",
        "operation": "inspect",
        "source": str(source.resolve()),
        "orbitron_schema": "orbitron.inspect/2",
        "producer": {
            "name": "orbitron",
            "version": "0.4.0",
            "commit": "58aa65b3f280",
        },
        "warnings": [{"source": "loader", "code": "parser:notice"}],
        "evidence": {
            "subject": "output",
            "program": "qe",
            "detected": "scene",
        },
        "canonical_mapping": {
            "producer": {
                "producer_type": "external_tool",
                "name": "orbitron",
                "version": "0.4.0",
                "commit": "58aa65b3f280",
            },
            "artifact": {
                "program": "qe",
                "path": str(source.resolve()),
                "status": "matched",
                "candidates": [
                    {
                        "kind": "qe.output",
                        "roles": ["primary_output"],
                        "content_kind": "text",
                        "evidence": "inferred",
                        "matched_by": "extension",
                        "matched_value": ".out",
                        "producing_step": None,
                        "expectation_id": None,
                    }
                ],
                "orbitron_subject": "output",
            },
            "scientific_system": {
                "status": "insufficient_evidence",
                "detected": "scene",
                "reason": (
                    "orbitron.inspect/2 reports atom and bond counts, but not "
                    "atom coordinates or a complete periodic-system "
                    "specification."
                ),
            },
        },
    }


def test_mcp_returns_orbitron_render_as_png_content(tmp_path, monkeypatch):
    source = tmp_path / "molecule.xyz"
    source.write_text("1\nH\nH 0 0 0\n", encoding="utf-8")
    image = b"\x89PNG\r\n\x1a\nrendered"
    version = OrbitronVersion(
        version="0.4.0",
        commit="58aa65b3f280",
        raw="orbitron-cli 0.4.0 (58aa65b3f280)",
    )
    response = OrbitronRender(
        source=str(source.resolve()),
        image=image,
        width=1024,
        height=768,
        stderr="",
        version=version,
    )

    class Client:
        def render(self, path):
            assert path == str(source)
            return response

    monkeypatch.setattr(orbitron_tools, "OrbitronClient", Client)

    rpc_response, should_exit = handle_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "render_with_orbitron",
            "arguments": {"path": str(source)},
        },
    })

    assert should_exit is False
    assert rpc_response["result"] == {
        "content": [
            {
                "type": "text",
                "text": json.dumps({
                    "schema_version": "chemtools.orbitron-render/1",
                    "status": "ok",
                    "operation": "render",
                    "source": str(source.resolve()),
                    "producer": {
                        "name": "orbitron",
                        "version": "0.4.0",
                        "commit": "58aa65b3f280",
                    },
                    "image": {
                        "mime_type": "image/png",
                        "width": 1024,
                        "height": 768,
                        "size_bytes": len(image),
                    },
                }, separators=(",", ":")),
            },
            {
                "type": "image",
                "data": base64.b64encode(image).decode("ascii"),
                "mimeType": "image/png",
            },
        ],
        "isError": False,
    }


def test_mcp_maps_supported_output_to_canonical_artifact(tmp_path, monkeypatch):
    source = tmp_path / "run.out"
    source.write_text("output\n")
    response = _response(
        source,
        {
            "subject": "output",
            "program": "molcas",
            "detected": "unparsed",
            "parse_error": "no geometry",
        },
    )

    class Client:
        def inspect(self, path):
            assert path == str(source)
            return response

    monkeypatch.setattr(orbitron_tools, "OrbitronClient", Client)

    inspected = dispatch_tool("inspect_with_orbitron", {"path": str(source)})

    assert inspected["canonical_mapping"] == {
        "producer": {
            "producer_type": "external_tool",
            "name": "orbitron",
            "version": "0.4.0",
            "commit": "58aa65b3f280",
        },
        "artifact": {
            "program": "molcas",
            "path": str(source.resolve()),
            "status": "matched",
            "candidates": [
                {
                    "kind": "molcas.output",
                    "roles": ["primary_output"],
                    "content_kind": "text",
                    "evidence": "inferred",
                    "matched_by": "extension",
                    "matched_value": ".out",
                    "producing_step": None,
                    "expectation_id": None,
                }
            ],
            "orbitron_subject": "output",
        },
        "scientific_system": {
            "status": "unavailable",
            "detected": "unparsed",
            "reason": "no geometry",
        },
    }


@pytest.mark.parametrize(
    ("filename", "subject", "program", "kind", "roles", "content_kind"),
    (
        (
            "run.movecs",
            "movecs",
            "nwchem",
            "nwchem.movecs",
            ["checkpoint", "orbital"],
            "binary",
        ),
        (
            "run.hess",
            "hessian",
            "nwchem",
            "nwchem.hessian",
            ["auxiliary_output"],
            "text",
        ),
        (
            "run.h5",
            "dirac_checkpoint",
            "dirac",
            "dirac.checkpoint",
            ["checkpoint", "wavefunction"],
            "binary",
        ),
    ),
)
def test_mcp_maps_supported_companion_artifacts(
    tmp_path,
    monkeypatch,
    filename,
    subject,
    program,
    kind,
    roles,
    content_kind,
):
    source = tmp_path / filename
    source.write_bytes(b"artifact")
    response = _response(source, {"subject": subject})

    class Client:
        def inspect(self, path):
            assert path == str(source)
            return response

    monkeypatch.setattr(orbitron_tools, "OrbitronClient", Client)

    inspected = dispatch_tool("inspect_with_orbitron", {"path": str(source)})

    assert inspected["canonical_mapping"]["artifact"] == {
        "program": program,
        "path": str(source.resolve()),
        "status": "matched",
        "candidates": [
            {
                "kind": kind,
                "roles": roles,
                "content_kind": content_kind,
                "evidence": "inferred",
                "matched_by": "extension",
                "matched_value": source.suffix,
                "producing_step": None,
                "expectation_id": None,
            }
        ],
        "orbitron_subject": subject,
    }
    assert inspected["canonical_mapping"]["scientific_system"] == {
        "status": "not_applicable",
        "reason": (
            "Orbitron inspected a companion artifact rather than a complete "
            "scientific system."
        ),
    }


@pytest.mark.parametrize(
    ("filename", "subject", "expected_artifact"),
    (
        (
            "run.civecs_singlet",
            "civecs",
            {
                "program": "nwchem",
                "status": "unmatched",
                "candidates": [],
                "orbitron_subject": "civecs",
            },
        ),
        (
            "run.molden",
            "molden",
            {
                "status": "unsupported_subject",
                "orbitron_subject": "molden",
                "reason": (
                    "Chemtools has no canonical artifact owner for this "
                    "Orbitron subject."
                ),
            },
        ),
    ),
)
def test_mcp_keeps_undeclared_companion_kinds_unresolved(
    tmp_path,
    monkeypatch,
    filename,
    subject,
    expected_artifact,
):
    source = tmp_path / filename
    source.write_bytes(b"artifact")
    response = _response(source, {"subject": subject})

    class Client:
        def inspect(self, path):
            assert path == str(source)
            return response

    monkeypatch.setattr(orbitron_tools, "OrbitronClient", Client)

    inspected = dispatch_tool("inspect_with_orbitron", {"path": str(source)})
    artifact = inspected["canonical_mapping"]["artifact"]

    if subject == "civecs":
        expected_artifact = {
            "path": str(source.resolve()),
            **expected_artifact,
        }
    assert artifact == expected_artifact


def _response(source, evidence):
    version = OrbitronVersion(
        version="0.4.0",
        commit="58aa65b3f280",
        raw="orbitron-cli 0.4.0 (58aa65b3f280)",
    )
    producer = {
        "name": "orbitron",
        "version": "0.4.0",
        "commit": "58aa65b3f280",
    }
    payload = {
        "schema": "orbitron.inspect/2",
        "producer": producer,
        "warnings": [],
        **evidence,
    }
    return OrbitronResponse(
        operation="inspect",
        source=str(source.resolve()),
        schema="orbitron.inspect/2",
        producer=producer,
        warnings=(),
        payload=payload,
        stderr="",
        version=version,
    )


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (
            OrbitronUnavailableError("missing"),
            "unavailable",
            "orbitron_unavailable",
        ),
        (
            OrbitronProtocolError("schema changed"),
            "incompatible",
            "orbitron_protocol_error",
        ),
    ],
)
def test_mcp_reports_optional_integration_failures(
    monkeypatch,
    error,
    status,
    code,
):
    class Client:
        def __init__(self):
            raise error

    monkeypatch.setattr(orbitron_tools, "OrbitronClient", Client)

    inspected = dispatch_tool("inspect_with_orbitron", {"path": "run.out"})

    assert inspected == {
        "schema_version": "chemtools.orbitron-inspection/1",
        "status": status,
        "error": code,
        "message": str(error),
    }


def test_mcp_preserves_bounded_refusal_details(monkeypatch):
    error = OrbitronCommandError(
        "Orbitron command exited with status 7",
        argv=("orbitron", "inspect"),
        returncode=7,
        stderr="x" * 5_000,
    )

    class Client:
        def __init__(self):
            raise error

    monkeypatch.setattr(orbitron_tools, "OrbitronClient", Client)

    inspected = dispatch_tool("inspect_with_orbitron", {"path": "run.out"})

    assert inspected["status"] == "tool_refused"
    assert inspected["error"] == "orbitron_command_error"
    assert inspected["returncode"] == 7
    assert len(inspected["stderr"]) == 4_121
    assert inspected["stderr"].endswith("\n[truncated by Chemtools]")


def test_mcp_reports_orbitron_render_refusal(monkeypatch):
    error = OrbitronCommandError(
        "Orbitron render refused the source",
        argv=("orbitron", "render"),
        returncode=3,
        stderr="renderer unavailable",
    )

    class Client:
        def render(self, path):
            assert path == "molecule.xyz"
            raise error

    monkeypatch.setattr(orbitron_tools, "OrbitronClient", Client)

    rendered = dispatch_tool("render_with_orbitron", {"path": "molecule.xyz"})

    assert rendered == {
        "schema_version": "chemtools.orbitron-render/1",
        "status": "tool_refused",
        "error": "orbitron_command_error",
        "message": "Orbitron render refused the source",
        "returncode": 3,
        "stderr": "renderer unavailable",
    }


def test_mcp_definition_exposes_no_command_or_remote_arguments():
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "inspect_with_orbitron"
    )

    assert _TOOL_PROGRAMS["inspect_with_orbitron"] == "generic"
    assert definition["inputSchema"] == {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Path to one local file supported by Orbitron. "
                    "Directories and remote targets are not accepted."
                ),
            }
        },
        "required": ["path"],
        "additionalProperties": False,
    }


def test_mcp_render_definition_exposes_only_a_source_path():
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "render_with_orbitron"
    )

    assert _TOOL_PROGRAMS["render_with_orbitron"] == "generic"
    assert definition["inputSchema"] == {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Path to one local file supported by Orbitron. "
                    "Directories, remote targets, output paths, and "
                    "render arguments are not accepted."
                ),
            }
        },
        "required": ["path"],
        "additionalProperties": False,
    }

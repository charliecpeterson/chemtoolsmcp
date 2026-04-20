#!/usr/bin/env python3
"""Standalone NWChem docs MCP server (backward-compat entry point).

The docs tools are now bundled into the main chemtools-nwchem server.
This module is kept so the `chemtools-nwchem-docs` entry point still works,
but the preferred setup is a single `chemtools-nwchem` server.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if not any("chemtools" in p for p in sys.path):
    sys.path.insert(0, str(_REPO_ROOT))

from chemtools.nwchem_docs import (  # noqa: E402
    find_examples,
    get_topic_guide,
    list_docs,
    lookup_block_syntax,
    read_doc_excerpt,
    search_docs,
)


SERVER_NAME = "nwchem-docs-mcp"
SERVER_VERSION = "0.2.0"
DEFAULT_PROTOCOL_VERSION = "2024-11-05"
LOG_PATH = os.environ.get("NWCHEM_DOCS_MCP_LOG")


def log_event(message: str) -> None:
    if not LOG_PATH:
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with Path(LOG_PATH).open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "list_nwchem_docs",
            "description": "List available bundled NWChem documentation files.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "get_nwchem_topic_guide",
            "description": "Get a curated documentation guide for a common NWChem topic: scf_open_shell, mcscf, fragment_guess, or tce.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
        {
            "name": "search_nwchem_docs",
            "description": "Search bundled NWChem docs for syntax, keywords, or option details. Returns ranked excerpts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 8},
                    "context_lines": {"type": "integer", "default": 2},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "lookup_nwchem_block_syntax",
            "description": "Look up NWChem input block syntax from bundled docs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "block_name": {"type": "string"},
                    "max_results": {"type": "integer", "default": 6},
                },
                "required": ["block_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "find_nwchem_examples",
            "description": "Search bundled NWChem example/tutorial docs for a topic.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "max_results": {"type": "integer", "default": 6},
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
        {
            "name": "read_nwchem_doc_excerpt",
            "description": "Read an excerpt from a bundled NWChem doc file by filename and line range or query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "doc_name": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "query": {"type": "string"},
                    "context_lines": {"type": "integer", "default": 8},
                },
                "required": ["doc_name"],
                "additionalProperties": False,
            },
        },
    ]


def dispatch_tool(name: str, arguments: dict[str, Any]) -> Any:
    if name == "list_nwchem_docs":
        return {"files": list_docs()}
    if name == "search_nwchem_docs":
        return search_docs(
            arguments["query"],
            max_results=int(arguments.get("max_results", 8)),
            context_lines=int(arguments.get("context_lines", 2)),
        )
    if name == "get_nwchem_topic_guide":
        return get_topic_guide(arguments["topic"])
    if name == "lookup_nwchem_block_syntax":
        return lookup_block_syntax(
            arguments["block_name"],
            max_results=int(arguments.get("max_results", 6)),
        )
    if name == "find_nwchem_examples":
        return find_examples(
            arguments["topic"],
            max_results=int(arguments.get("max_results", 6)),
        )
    if name == "read_nwchem_doc_excerpt":
        return read_doc_excerpt(
            arguments["doc_name"],
            start_line=arguments.get("start_line"),
            end_line=arguments.get("end_line"),
            query=arguments.get("query"),
            context_lines=int(arguments.get("context_lines", 8)),
        )
    raise ValueError(f"Unknown tool: {name}")


def make_response(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def make_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle_request(message: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    method = message.get("method")
    request_id = message.get("id")
    log_event(f"handle_request method={method} id={request_id}")

    if method == "initialize":
        return (
            make_response(
                request_id,
                {
                    "protocolVersion": DEFAULT_PROTOCOL_VERSION,
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "capabilities": {"tools": {}},
                },
            ),
            False,
        )
    if method == "notifications/initialized":
        return None, False
    if method == "tools/list":
        log_event("tools/list requested")
        return make_response(request_id, {"tools": tool_definitions()}), False
    if method == "resources/list":
        return make_response(request_id, {"resources": []}), False
    if method == "prompts/list":
        return make_response(request_id, {"prompts": []}), False
    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        log_event(f"tools/call name={name}")
        try:
            payload = dispatch_tool(name, arguments)
        except Exception as exc:  # noqa: BLE001
            log_event(f"dispatch_tool error name={name}: {exc}")
            return make_error(request_id, -32000, str(exc)), False
        log_event(f"dispatch_tool done name={name}")
        return make_response(
            request_id,
            {
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                "structuredContent": payload,
                "isError": False,
            },
        ), False
    if method == "shutdown":
        return make_response(request_id, {}), False
    if method == "exit":
        return None, True
    return make_error(request_id, -32601, f"Method not found: {method}"), False


def _read_message(stdin: Any) -> dict[str, Any] | None:
    first_line = stdin.readline()
    if not first_line:
        return None

    stripped = first_line.strip()
    if stripped.startswith(b"{"):
        try:
            return json.loads(first_line.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    headers: dict[str, str] = {}
    line = first_line
    while line:
        stripped_line = line.strip()
        if not stripped_line:
            break
        decoded = line.decode("utf-8").strip()
        if ":" in decoded:
            key, value = decoded.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        line = stdin.readline()

    content_length = headers.get("content-length")
    if not content_length:
        return None
    body = stdin.read(int(content_length))
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return None


def _write_message(stdout: Any, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    stdout.write(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8"))
    stdout.write(body)
    stdout.flush()


def main() -> None:
    log_event("server start")
    while True:
        message = _read_message(sys.stdin.buffer)
        if message is None:
            log_event("server stop: no message")
            break
        response, should_exit = handle_request(message)
        if response is not None:
            _write_message(sys.stdout.buffer, response)
        if should_exit:
            log_event("server stop: exit")
            break


if __name__ == "__main__":
    main()

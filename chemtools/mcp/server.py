"""MCP server protocol I/O + argparse helpers.

Program-neutral framework code for the MCP server. Per-program entry
points (currently `chemtools.mcp.nwchem:main`) own:

  * `tool_definitions()` — the registered tool list for that program
  * Any program-specific aliases (`_TOOL_ALIASES`)
  * `handle_request()` and `dispatch_tool()` — they reference the
    per-program tool surface
  * `serve()` and `main()` — the JSON-RPC loop and entry point

This module owns what doesn't differ per program:

  * `read_message` / `write_message`         JSON-RPC transport read/write
                                              (supports both LSP-style
                                              `Content-Length` headers and
                                              JSON-lines).
  * `make_response`                          Build a JSON-RPC response envelope.
  * `make_success_result` / `make_error_result`
                                              Wrap a tool payload (or error)
                                              in the MCP `content` schema.
  * `build_arg_parser`                        Argparse setup shared by every
                                              chemtools-<program> entry.

Extracted from `chemtools/mcp/nwchem.py` as the second step of the MCP
split (after the @_tool decorator + registries moved to
`chemtools.mcp.decorator`).
"""

from __future__ import annotations
import argparse
import json
from typing import Any

from chemtools.mcp import modes as _modes


# Transport mode: "content-length" (LSP-style) or "jsonl" (newline-delimited).
# read_message infers from the first byte and sets this so write_message
# uses the matching framing.
TRANSPORT_MODE = "content-length"


def read_message(stream: Any) -> dict[str, Any] | None:
    """Read one JSON-RPC message from a binary input stream.

    Supports two framings:

      * Content-Length / JSON header + body — LSP-style framing used by
        most MCP clients.
      * JSON-lines — one JSON object per line. Easier to test by hand.

    Returns None at EOF.
    """
    global TRANSPORT_MODE
    headers: dict[str, str] = {}

    while True:
        line = stream.readline()
        if not line:
            return None
        if line.lstrip().startswith(b"{"):
            TRANSPORT_MODE = "jsonl"
            return json.loads(line.decode("utf-8"))
        if line in (b"\r\n", b"\n"):
            break
        decoded = line.decode("utf-8")
        if ":" not in decoded:
            continue
        key, value = decoded.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    content_length = headers.get("content-length")
    if content_length is None:
        return None
    body = stream.read(int(content_length))
    if not body:
        return None
    TRANSPORT_MODE = "content-length"
    return json.loads(body.decode("utf-8"))


def write_message(stream: Any, payload: dict[str, Any]) -> None:
    """Write a JSON-RPC message using the framing that read_message inferred."""
    body = json.dumps(payload).encode("utf-8")
    if TRANSPORT_MODE == "jsonl":
        stream.write(body + b"\n")
        stream.flush()
        return
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    stream.write(header)
    stream.write(body)
    stream.flush()


def make_success_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a tool's payload in the MCP `content` schema (success)."""
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, separators=(",", ":")),
            }
        ],
        "isError": False,
    }


def make_error_result(message: str) -> dict[str, Any]:
    """Wrap an error message in the MCP `content` schema (failure)."""
    return {
        "content": [
            {
                "type": "text",
                "text": message,
            }
        ],
        "isError": True,
    }


def make_response(
    request_id: Any,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 response envelope."""
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result if result is not None else {}
    return payload


def build_arg_parser(prog: str, description: str) -> argparse.ArgumentParser:
    """Standard --mode / --programs / --list-tools / --show-mode arg parser.

    Per-program entry points should use this to keep their CLI surface
    consistent.
    """
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "--mode",
        choices=_modes.VALID_MODES,
        default=None,
        help=(
            "Server mode: 'analysis' (no executable, no scheduler), "
            "'local' (direct subprocess launcher), or 'hpc' (scheduler "
            "submission). Default: read CHEMTOOLS_MODE, else auto-detect "
            "from CHEMTOOLS_RUNNER_PROFILES."
        ),
    )
    parser.add_argument(
        "--programs",
        default=None,
        help=(
            "Comma-separated list of programs whose tools should be loaded "
            "(e.g. 'molcas' or 'nwchem,molcas'). Generic tools are always "
            "visible. Default: read CHEMTOOLS_PROGRAMS, else no filter "
            "(all programs)."
        ),
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print the tool names visible in the resolved mode + programs and exit.",
    )
    parser.add_argument(
        "--show-mode",
        action="store_true",
        help="Print the resolved mode, program filter, and reasons and exit.",
    )
    return parser


__all__ = [
    "TRANSPORT_MODE",
    "read_message",
    "write_message",
    "make_response",
    "make_success_result",
    "make_error_result",
    "build_arg_parser",
]

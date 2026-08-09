"""Compatibility response helpers, image results, and shared CLI arguments.

The main ``chemtools`` command uses the official SDK transport in
``chemtools.mcp.sdk_server``. The response helpers remain temporarily for
direct callers of the former dictionary-based request dispatcher.

This module owns what doesn't differ per program:

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
import base64
import json
from dataclasses import dataclass
from typing import Any

from mcp_types.version import (
    HANDSHAKE_PROTOCOL_VERSIONS,
    LATEST_HANDSHAKE_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
)

from chemtools.mcp import modes as _modes


DEFAULT_PROTOCOL_VERSION = LATEST_HANDSHAKE_VERSION

_MAX_IMAGE_CONTENT_BYTES = 8 * 1024 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class ImageToolResult:
    """A JSON payload accompanied by one validated PNG tool-result block."""

    payload: dict[str, Any]
    image: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.payload, dict):
            raise TypeError("image tool payload must be a dictionary")
        if not isinstance(self.image, bytes):
            raise TypeError("image tool content must be bytes")
        if len(self.image) > _MAX_IMAGE_CONTENT_BYTES:
            raise ValueError(
                f"image tool content exceeds {_MAX_IMAGE_CONTENT_BYTES} bytes"
            )
        if not self.image.startswith(_PNG_SIGNATURE):
            raise ValueError("image tool content is not a PNG")


def negotiate_protocol_version(requested_version: Any) -> str:
    """Return the requested version when supported, otherwise our latest."""
    if requested_version in HANDSHAKE_PROTOCOL_VERSIONS:
        return requested_version
    return DEFAULT_PROTOCOL_VERSION


def make_success_result(
    payload: dict[str, Any] | ImageToolResult,
) -> dict[str, Any]:
    """Wrap a tool's payload in the MCP `content` schema (success)."""
    if isinstance(payload, ImageToolResult):
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload.payload, separators=(",", ":")),
                },
                {
                    "type": "image",
                    "data": base64.b64encode(payload.image).decode("ascii"),
                    "mimeType": "image/png",
                },
            ],
            "isError": False,
        }
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
    """Build the shared MCP server and installed-resource argument parser.

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
        "--toolset",
        default=None,
        help=(
            "Tool surface: a preset name, 'developer'/'full', or a "
            "comma-separated list of tool names. Default: read "
            "CHEMTOOLS_TOOLSET, else use the guided preset."
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
    parser.add_argument(
        "--print-profile-example",
        choices=("local", "slurm"),
        help="Print a bundled runner-profile example and exit.",
    )
    parser.add_argument(
        "--targets",
        help=(
            "Schema-2 target YAML or JSON path. Default: read "
            "CHEMTOOLS_TARGETS."
        ),
    )
    parser.add_argument(
        "--target",
        help="Default named target for this server process.",
    )
    permission = parser.add_mutually_exclusive_group()
    permission.add_argument(
        "--enable-execution",
        action="store_true",
        default=None,
        help="Allow approved launches and owned cancellation.",
    )
    permission.add_argument(
        "--disable-execution",
        action="store_false",
        dest="enable_execution",
        default=None,
        help="Disable launches and cancellation even if configured otherwise.",
    )
    parser.add_argument(
        "--print-target-example",
        action="store_true",
        help="Print the bundled schema-2 target example and exit.",
    )
    return parser


__all__ = [
    "SUPPORTED_PROTOCOL_VERSIONS",
    "DEFAULT_PROTOCOL_VERSION",
    "ImageToolResult",
    "negotiate_protocol_version",
    "make_response",
    "make_success_result",
    "make_error_result",
    "build_arg_parser",
]

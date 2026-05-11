#!/usr/bin/env python3
"""chemtools-nwchem MCP server entry point.

Slim entry-point module. Most of the work happens in:

  * `chemtools/mcp/decorator.py`        @_tool decorator + shared registries
  * `chemtools/mcp/server.py`           JSON-RPC transport + arg parser
  * `chemtools/mcp/tools/nwchem.py`     tool_definitions + all @_tool handlers
                                         + dispatch_tool + handle_request

Importing `chemtools.mcp.tools.nwchem` triggers all handler registrations,
so by the time `serve()` is called the shared `_TOOL_REGISTRY` is populated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Development fallback: if running directly from the source tree, add repo
# root to path so `chemtools` can be imported without `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parents[2]
if not any("chemtools" in p for p in sys.path):
    sys.path.insert(0, str(_REPO_ROOT))

# Framework
from chemtools.mcp.decorator import (  # noqa: E402
    _TOOL_REGISTRY,
    _TOOL_CAPABILITIES,
    log_event,
    set_active_mode,
)
from chemtools.mcp.server import (  # noqa: E402
    read_message,
    write_message,
    build_arg_parser,
)
from chemtools.mcp import modes as _modes  # noqa: E402

# Importing this module triggers all @_tool handler registrations and pulls
# tool_definitions / dispatch_tool / handle_request into our namespace for
# serve() and back-compat for tests.
from chemtools.mcp.tools.nwchem import (  # noqa: E402, F401
    tool_definitions,
    dispatch_tool,
    handle_request,
    _TOOL_ALIASES,
)

# Active mode mirror — main() updates the decorator-module canonical copy.
ACTIVE_MODE: str = "analysis"


def serve() -> None:
    input_stream = sys.stdin.buffer
    output_stream = sys.stdout.buffer
    log_event(f"server start mode={ACTIVE_MODE}")

    while True:
        message = read_message(input_stream)
        if message is None:
            log_event("server stop: no message")
            break
        response, should_exit = handle_request(message)
        if response is not None:
            write_message(output_stream, response)
        if should_exit:
            log_event("server stop: exit requested")
            break


def _build_arg_parser():
    return build_arg_parser(
        prog="chemtools-nwchem",
        description="NWChem MCP server. Tool exposure depends on --mode.",
    )


def main() -> None:
    """Entry point registered by pyproject.toml — `chemtools-nwchem` command."""
    global ACTIVE_MODE
    args = _build_arg_parser().parse_args()
    mode, reason = _modes.resolve_mode(args.mode)
    ACTIVE_MODE = mode
    set_active_mode(mode)  # Keep canonical mcp.decorator copy in sync.

    summary = _modes.summarize_mode(mode, _TOOL_CAPABILITIES, tool_definitions())
    log_event(
        f"resolved mode={mode} reason={reason} "
        f"tools={summary['available_tool_count']}/{summary['total_tool_count']}"
    )

    if args.show_mode:
        print(json.dumps({"mode": mode, "reason": reason, **summary}, indent=2))
        return
    if args.list_tools:
        visible = _modes.filter_tools(tool_definitions(), _TOOL_CAPABILITIES, mode)
        for d in visible:
            print(d["name"])
        return

    # Tell stderr (visible in the client transcript on stderr capture) what mode
    # we are running in. Stdout is reserved for the JSON-RPC stream.
    sys.stderr.write(
        f"chemtools-nwchem: mode={mode} ({reason}); "
        f"{summary['available_tool_count']}/{summary['total_tool_count']} tools exposed\n"
    )
    sys.stderr.flush()
    serve()


if __name__ == "__main__":
    main()

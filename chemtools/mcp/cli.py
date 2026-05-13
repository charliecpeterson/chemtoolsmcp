#!/usr/bin/env python3
"""chemtools MCP server entry point.

Slim CLI module — owns argument parsing, mode/program resolution, and the
``serve()`` JSON-RPC loop. The heavy lifting happens in:

  * `chemtools/mcp/decorator.py`     @_tool decorator + shared registries
  * `chemtools/mcp/server.py`        JSON-RPC transport + arg parser
  * `chemtools/mcp/dispatch.py`      tool_definitions aggregator +
                                     dispatch_tool + handle_request +
                                     _TOOL_ALIASES
  * `chemtools/mcp/tools/__init__.py`  Eagerly imports every program's
                                       handler module so @_tool decorators
                                       run before tool_definitions() is
                                       called.

Renamed from ``chemtools/mcp/nwchem.py`` — that path now lives as a
back-compat shim re-exporting from this module.
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
    _TOOL_PROGRAMS,
    log_event,
    set_active_mode,
    set_active_programs,
)
from chemtools.mcp.server import (  # noqa: E402
    read_message,
    write_message,
    build_arg_parser,
)
from chemtools.mcp import modes as _modes  # noqa: E402

# Importing dispatch triggers eager imports of every program's tool module
# (via chemtools.mcp.tools.__init__), so @_tool registrations are guaranteed
# to have run before tool_definitions() is first called.
from chemtools.mcp.dispatch import (  # noqa: E402, F401
    tool_definitions,
    dispatch_tool,
    handle_request,
    _TOOL_ALIASES,
)

# Active mode + program filter mirrors — main() updates the decorator-module
# canonical copies via set_active_mode + set_active_programs.
ACTIVE_MODE: str = "analysis"
ACTIVE_PROGRAMS: set[str] | None = None


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


def _build_arg_parser(prog: str = "chemtools"):
    return build_arg_parser(
        prog=prog,
        description=(
            "chemtools MCP server (NWChem + Molcas + future). Tool exposure "
            "depends on --mode and --programs."
        ),
    )


def main(prog: str = "chemtools") -> None:
    """Entry point registered by pyproject.toml.

    Default binary name is ``chemtools``; the legacy ``chemtools-nwchem``
    binary aliases to ``main(prog="chemtools-nwchem")`` for backward compat.
    """
    global ACTIVE_MODE, ACTIVE_PROGRAMS
    args = _build_arg_parser(prog=prog).parse_args()

    mode, mode_reason = _modes.resolve_mode(args.mode)
    ACTIVE_MODE = mode
    set_active_mode(mode)  # Keep canonical mcp.decorator copy in sync.

    programs, programs_reason = _modes.resolve_programs(args.programs)
    ACTIVE_PROGRAMS = programs
    set_active_programs(programs)  # Keep canonical mcp.decorator copy in sync.

    summary = _modes.summarize_mode(
        mode, _TOOL_CAPABILITIES, tool_definitions(),
        programs=programs, program_tags=_TOOL_PROGRAMS,
    )
    log_event(
        f"resolved mode={mode} ({mode_reason}) "
        f"programs={programs or 'all'} ({programs_reason}) "
        f"tools={summary['available_tool_count']}/{summary['total_tool_count']}"
    )

    if args.show_mode:
        print(json.dumps({
            "mode": mode, "mode_reason": mode_reason,
            "programs": sorted(programs) if programs else None,
            "programs_reason": programs_reason,
            **summary,
        }, indent=2))
        return
    if args.list_tools:
        visible = _modes.filter_tools(
            tool_definitions(), _TOOL_CAPABILITIES, mode,
            programs=programs, program_tags=_TOOL_PROGRAMS,
        )
        for d in visible:
            print(d["name"])
        return

    # Tell stderr (visible in the client transcript on stderr capture) what mode
    # we are running in. Stdout is reserved for the JSON-RPC stream.
    sys.stderr.write(
        f"{prog}: mode={mode} ({mode_reason}); "
        f"programs={sorted(programs) if programs else 'all'} ({programs_reason}); "
        f"{summary['available_tool_count']}/{summary['total_tool_count']} tools exposed\n"
    )
    sys.stderr.flush()
    serve()


def main_legacy_nwchem() -> None:
    """Backward-compat alias for the renamed ``chemtools-nwchem`` binary.

    Identical behavior to ``main()`` — just keeps the old script name working
    for one release. Emits a stderr deprecation hint.
    """
    sys.stderr.write(
        "chemtools-nwchem: this binary will be renamed to 'chemtools' in a "
        "future release. Update your MCP configs.\n"
    )
    sys.stderr.flush()
    main(prog="chemtools-nwchem")


if __name__ == "__main__":
    main()

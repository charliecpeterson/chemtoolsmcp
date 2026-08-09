#!/usr/bin/env python3
"""chemtools MCP server entry point.

Slim CLI module that owns argument parsing and mode/program resolution. The
heavy lifting happens in:

  * `chemtools/mcp/decorator.py`     @_tool decorator + shared registries
  * `chemtools/mcp/sdk_server.py`    official MCP SDK adapter
  * `chemtools/mcp/server.py`        shared result type + arg parser
  * `chemtools/mcp/dispatch.py`      tool_definitions aggregator +
                                     dispatch_tool + handle_request +
                                     _TOOL_ALIASES
  * `chemtools/mcp/catalog.py`       Built-in program and MCP tool providers.

Renamed from ``chemtools/mcp/nwchem.py`` — that path now lives as a
back-compat shim re-exporting from this module.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from importlib.resources import files

from chemtools.mcp.decorator import (
    _TOOL_REGISTRY,
    _TOOL_CAPABILITIES,
    _TOOL_PROGRAMS,
    log_event,
)
from chemtools.mcp.state import ServerState
from chemtools.mcp.server import build_arg_parser
from chemtools.mcp.sdk_server import serve_stdio
from chemtools.mcp import modes as _modes
from chemtools.execution.targets import (
    TARGET_CONFIG_ENV,
    load_target_catalog,
)

# Importing dispatch registers built-in backends and loads tool modules in
# catalog order before the server accepts requests.
from chemtools.mcp.dispatch import (
    tool_definitions,
    dispatch_tool,
    handle_request,
    _TOOL_ALIASES,
)

_PROFILE_EXAMPLE_FILES = {
    "local": "runner_profiles.local.example.json",
    "slurm": "runner_profiles.slurm.example.yaml",
}


def _profile_example_text(name: str) -> str:
    filename = _PROFILE_EXAMPLE_FILES.get(name)
    if filename is None:
        raise ValueError(f"unknown runner-profile example: {name!r}")
    return files("chemtools").joinpath(filename).read_text(encoding="utf-8")


def _target_example_text() -> str:
    return files("chemtools").joinpath(
        "execution_targets.example.yaml"
    ).read_text(encoding="utf-8")


def serve(state: ServerState | None = None) -> None:
    state = state or ServerState.create()
    log_event(f"server start mode={state.mode}")
    serve_stdio(state)
    log_event("server stop")


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
    parser = _build_arg_parser(prog=prog)
    args = parser.parse_args()

    if args.print_profile_example:
        sys.stdout.write(_profile_example_text(args.print_profile_example))
        return
    if args.print_target_example:
        sys.stdout.write(_target_example_text())
        return

    target_path = args.targets or os.environ.get(TARGET_CONFIG_ENV)
    try:
        target_catalog = (
            load_target_catalog(target_path)
            if target_path is not None
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))
    if target_catalog is None and args.target is not None:
        parser.error("--target requires --targets or CHEMTOOLS_TARGETS")
    if target_catalog is None and args.enable_execution is not None:
        parser.error(
            "--enable-execution and --disable-execution require configured targets"
        )
    if target_catalog is not None:
        try:
            target_catalog = replace(
                target_catalog,
                default_target=(
                    args.target
                    if args.target is not None
                    else target_catalog.default_target
                ),
                enable_execution=(
                    args.enable_execution
                    if args.enable_execution is not None
                    else target_catalog.enable_execution
                ),
            )
        except ValueError as exc:
            parser.error(str(exc))

    if target_catalog is not None and args.mode is None:
        mode = "analysis"
        mode_reason = "named targets keep tool visibility independent"
    else:
        mode, mode_reason = _modes.resolve_mode(args.mode)
    if (
        target_catalog is not None
        and args.mode is not None
        and args.enable_execution is None
        and (mode in {"local", "hpc"})
        != target_catalog.enable_execution
    ):
        parser.error(
            "legacy --mode conflicts with schema-2 execution permission; "
            "set --enable-execution or --disable-execution explicitly"
        )
    programs, programs_reason = _modes.resolve_programs(args.programs)
    toolset, toolset_reason = _modes.resolve_toolset(
        args.toolset,
        aliases={
            alias: target
            for alias, (target, _translator) in _TOOL_ALIASES.items()
        },
    )
    state = ServerState.create(
        mode=mode,
        programs=programs,
        toolset=toolset,
        target_catalog=target_catalog,
    )

    summary = _modes.summarize_mode(
        mode, _TOOL_CAPABILITIES, tool_definitions(),
        programs=programs, program_tags=_TOOL_PROGRAMS, toolset=toolset,
    )
    log_event(
        f"resolved mode={mode} ({mode_reason}) "
        f"programs={programs or 'all'} ({programs_reason}) "
        f"toolset={toolset_reason} "
        f"tools={summary['available_tool_count']}/{summary['total_tool_count']}"
    )

    if args.show_mode:
        print(json.dumps({
            "mode": mode, "mode_reason": mode_reason,
            "programs": sorted(programs) if programs else None,
            "programs_reason": programs_reason,
            "toolset": sorted(toolset) if toolset else None,
            "toolset_reason": toolset_reason,
            "execution_enabled": state.execution_service.enable_execution,
            "default_target": state.execution_service.default_target,
            "targets": sorted(state.execution_service.configured_targets),
            **summary,
        }, indent=2))
        return
    if args.list_tools:
        visible = _modes.filter_tools(
            tool_definitions(), _TOOL_CAPABILITIES, mode,
            programs=programs, program_tags=_TOOL_PROGRAMS, toolset=toolset,
        )
        for d in visible:
            print(d["name"])
        return

    # Tell stderr (visible in the client transcript on stderr capture) what mode
    # we are running in. Stdout is reserved for the JSON-RPC stream.
    sys.stderr.write(
        f"{prog}: mode={mode} ({mode_reason}); "
        f"programs={sorted(programs) if programs else 'all'} ({programs_reason}); "
        "execution="
        f"{'enabled' if state.execution_service.enable_execution else 'disabled'}; "
        f"target={state.execution_service.default_target or 'none'}; "
        f"{summary['available_tool_count']}/"
        f"{summary['total_tool_count']} tools exposed\n"
    )
    sys.stderr.flush()
    serve(state)


def main_legacy_nwchem() -> None:
    """Backward-compat alias for the renamed ``chemtools-nwchem`` binary.

    Identical behavior to ``main()`` — just keeps the old script name working
    for one release. Emits a stderr deprecation hint.
    """
    sys.stderr.write(
        "chemtools-nwchem: deprecated compatibility command; use 'chemtools'. "
        "Update your MCP configs before the compatibility command is removed.\n"
    )
    sys.stderr.flush()
    main(prog="chemtools-nwchem")


if __name__ == "__main__":
    main()

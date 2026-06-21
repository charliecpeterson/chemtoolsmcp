"""MCP tool decorator and shared registries.

This module owns the cross-tool state that the JSON-RPC server framework
and the per-tool handlers both touch:

  * `_TOOL_REGISTRY`     handler-by-name lookup
  * `_TOOL_CAPABILITIES` capability tag per tool (drives mode filtering)
  * `_tool(name, needs=...)` decorator that registers a handler

It also owns module-level state useful to handlers:

  * `LOG_PATH`           optional log file (set via env CHEMTOOLS_MCP_LOG)
  * `log_event(message)` writes a timestamped line to LOG_PATH if configured

Extracted from `chemtools/mcp/nwchem.py` as the first step of the
multi-program MCP split. The next step extracts the JSON-RPC `serve()`
loop into `mcp/server.py`; tool definitions and handlers eventually
move into `mcp/tools/<program>.py`.
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Any, Callable


# Server constants — shared between any per-program MCP entry point.
SERVER_NAME = "chemtools-nwchem"
SERVER_VERSION = "0.1.0"
DEFAULT_PROTOCOL_VERSION = "2024-11-05"
TRANSPORT_MODE = "content-length"

# Optional log file. Set CHEMTOOLS_MCP_LOG=/path/to/log.txt to enable.
LOG_PATH = os.environ.get("CHEMTOOLS_MCP_LOG")

# Active server mode — resolved at startup in `serve()` / `main()`.
# Default to analysis so any caller that imports the module without going
# through main() still gets a consistent answer (only pure tools visible).
ACTIVE_MODE: str = "analysis"

# Active program filter — None means "no filter, all programs visible".
# Otherwise it's a set of program tag strings (e.g. {"nwchem", "molcas"}).
ACTIVE_PROGRAMS: set[str] | None = None

# Active tool-name allowlist — None means "no filter". Otherwise an exact set of
# tool names (a preset like "triage" or a custom list) to expose.
ACTIVE_TOOLSET: frozenset[str] | None = None


# Per-tool state. Decorator below populates these dicts.
_TOOL_REGISTRY: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
_TOOL_CAPABILITIES: dict[str, str] = {}
# Program ownership tag per tool. "generic" means no program affiliation
# (always visible). Used by the --programs filter to select which subset of
# tools is exposed at tools/list time.
_TOOL_PROGRAMS: dict[str, str] = {}

_VALID_CAPABILITIES = {
    "none",
    "registry",
    "runner_profile",
    "executable_or_scheduler",
    "executable",
    "scheduler",
}


def _tool(
    name: str,
    *,
    needs: str = "none",
    program: str = "generic",
) -> Callable:
    """Decorator that registers a handler function under *name*.

    Parameters
    ----------
    needs
        Capability tag. Tools with ``needs="none"`` (the default) are visible
        in every server mode; other tags (``registry``, ``runner_profile``,
        ``executable_or_scheduler``, ``executable``, ``scheduler``) drive
        mode-based filtering in ``chemtools/mcp/modes.py``.
    program
        Program affiliation tag. ``"generic"`` (the default) means the tool
        works across programs and is always visible. Otherwise, the tool is
        only exposed when its program is in the active ``--programs`` filter
        (or no filter is set).
    """
    if needs not in _VALID_CAPABILITIES:
        raise ValueError(
            f"_tool({name!r}): unknown needs={needs!r}; "
            f"expected one of {sorted(_VALID_CAPABILITIES)}"
        )

    def decorator(fn: Callable) -> Callable:
        _TOOL_REGISTRY[name] = fn
        _TOOL_CAPABILITIES[name] = needs
        _TOOL_PROGRAMS[name] = program
        return fn

    return decorator


def log_event(message: str) -> None:
    """Append a timestamped line to LOG_PATH if it's configured."""
    if not LOG_PATH:
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with Path(LOG_PATH).open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def set_active_mode(mode: str) -> None:
    """Set the active server mode from a CLI entry point."""
    global ACTIVE_MODE
    ACTIVE_MODE = mode


def set_active_programs(programs: set[str] | None) -> None:
    """Set the active program filter from a CLI entry point."""
    global ACTIVE_PROGRAMS
    ACTIVE_PROGRAMS = programs


def set_active_toolset(toolset: frozenset[str] | None) -> None:
    """Set the active tool-name allowlist from a CLI entry point."""
    global ACTIVE_TOOLSET
    ACTIVE_TOOLSET = toolset


__all__ = [
    "SERVER_NAME",
    "SERVER_VERSION",
    "DEFAULT_PROTOCOL_VERSION",
    "TRANSPORT_MODE",
    "LOG_PATH",
    "ACTIVE_MODE",
    "ACTIVE_PROGRAMS",
    "ACTIVE_TOOLSET",
    "set_active_mode",
    "set_active_programs",
    "set_active_toolset",
    "_TOOL_REGISTRY",
    "_TOOL_CAPABILITIES",
    "_TOOL_PROGRAMS",
    "_VALID_CAPABILITIES",
    "_tool",
    "log_event",
]

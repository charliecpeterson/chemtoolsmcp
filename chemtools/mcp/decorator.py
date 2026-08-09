"""MCP tool decorator and shared registries.

This module owns the cross-tool state that the JSON-RPC server framework
and the per-tool handlers both touch:

  * `_TOOL_REGISTRY`     handler-by-name lookup
  * `_TOOL_CAPABILITIES` capability tag per tool (drives mode filtering)
  * `_tool(name, needs=...)` decorator that registers a handler

It also exposes request-bound state useful to handlers:

  * `get_server_state()` current filters and execution ownership
  * `LOG_PATH`           optional log file (set via env CHEMTOOLS_MCP_LOG)
  * `log_event(message)` writes a timestamped line to LOG_PATH if configured

The stdio server lives in ``mcp/server.py``. Focused modules under
``mcp/tools`` own tool definitions and handlers.
"""

from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from chemtools.application.execution import ExecutionService
from chemtools.mcp.server import (
    DEFAULT_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
)
from chemtools.mcp.state import ServerState


# Server constants shared by MCP entry points.
SERVER_NAME = "chemtools"
SERVER_VERSION = "0.1.0"
TRANSPORT_MODE = "content-length"

# Optional log file. Set CHEMTOOLS_MCP_LOG=/path/to/log.txt to enable.
LOG_PATH = os.environ.get("CHEMTOOLS_MCP_LOG")

# Direct Python callers historically configured process-wide state through the
# setters below. Request dispatch binds an explicit CLI-owned state instead.
_COMPATIBILITY_STATE = ServerState.create()
_REQUEST_STATE: ContextVar[Optional[ServerState]] = ContextVar(
    "chemtools_mcp_request_state",
    default=None,
)


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
    """Set mode for direct Python callers using the compatibility state."""
    global _COMPATIBILITY_STATE
    _COMPATIBILITY_STATE = ServerState.create(
        mode=mode,
        programs=_COMPATIBILITY_STATE.programs,
        toolset=_COMPATIBILITY_STATE.toolset,
    )


def get_server_state() -> ServerState:
    """Return request-bound state or the direct-call compatibility state."""
    return _REQUEST_STATE.get() or _COMPATIBILITY_STATE


@contextmanager
def bind_server_state(state: ServerState) -> Iterator[None]:
    """Make one server's state visible to its registered tool handler."""
    token = _REQUEST_STATE.set(state)
    try:
        yield
    finally:
        _REQUEST_STATE.reset(token)


def get_execution_service() -> ExecutionService:
    """Return the service that owns launches for this MCP process."""
    return get_server_state().execution_service


def set_active_programs(programs: set[str] | None) -> None:
    """Set the program filter for direct Python compatibility callers."""
    global _COMPATIBILITY_STATE
    normalized = frozenset(programs) if programs is not None else None
    _COMPATIBILITY_STATE = replace(_COMPATIBILITY_STATE, programs=normalized)


def set_active_toolset(toolset: frozenset[str] | None) -> None:
    """Set the tool allowlist for direct Python compatibility callers."""
    global _COMPATIBILITY_STATE
    _COMPATIBILITY_STATE = replace(_COMPATIBILITY_STATE, toolset=toolset)


def __getattr__(name: str) -> Any:
    """Retain the former read-only module attributes during migration."""
    state = get_server_state()
    if name == "ACTIVE_MODE":
        return state.mode
    if name == "ACTIVE_PROGRAMS":
        return set(state.programs) if state.programs is not None else None
    if name == "ACTIVE_TOOLSET":
        return state.toolset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SERVER_NAME",
    "SERVER_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "DEFAULT_PROTOCOL_VERSION",
    "TRANSPORT_MODE",
    "LOG_PATH",
    "ACTIVE_MODE",
    "ACTIVE_PROGRAMS",
    "ACTIVE_TOOLSET",
    "ServerState",
    "bind_server_state",
    "get_execution_service",
    "get_server_state",
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

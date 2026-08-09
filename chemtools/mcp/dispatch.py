"""Tool dispatch and cross-program definition aggregation.

Owns the multi-program glue that used to live at the bottom of
``chemtools/mcp/tools/nwchem.py``:

  * ``tool_definitions()``   Aggregator that concatenates each program's
                             tool definitions + generics.
  * ``dispatch_tool(name, arguments)``
                             Per-tool dispatcher (resolves aliases via
                             ``_TOOL_ALIASES`` + invokes the registered
                             handler).
  * ``handle_request(message)``
                             Compatibility JSON-RPC dispatcher retained for
                             direct Python callers during migration.
  * ``_TOOL_ALIAS_REGISTRY`` Validated metadata for renamed tools.
  * ``_TOOL_ALIASES``        Legacy tuple-map view of that registry.

The CLI passes one ``ServerState`` through request dispatch. Direct Python
callers fall back to the compatibility state in ``chemtools.mcp.decorator``.
"""
from __future__ import annotations

from typing import Any

from chemtools.mcp.decorator import (
    _TOOL_REGISTRY,
    _TOOL_CAPABILITIES,
    _TOOL_PROGRAMS,
    bind_server_state,
    get_server_state,
    log_event,
    SERVER_NAME,
    SERVER_VERSION,
)
from chemtools.mcp.state import ServerState
from chemtools.mcp.server import (
    ImageToolResult,
    make_response,
    make_success_result,
    make_error_result,
    negotiate_protocol_version,
)
from chemtools.mcp import modes as _modes
from chemtools.mcp.catalog import (
    catalog_tool_definitions,
    load_tool_modules,
    register_builtin_backends,
)
from chemtools.mcp.compatibility import (
    HIDDEN_TOOL_ALIASES,
    alias_dispatch_map,
    is_alias_available,
    validate_tool_aliases,
)

# Dispatch composes backends and tools for callers that bypass the CLI.
register_builtin_backends()
load_tool_modules()


def tool_definitions() -> list[dict[str, Any]]:
    """Return generic and program tool definitions in catalog order."""
    return catalog_tool_definitions()


_TOOL_ALIAS_REGISTRY = validate_tool_aliases(
    HIDDEN_TOOL_ALIASES,
    canonical_names=(definition["name"] for definition in tool_definitions()),
    capabilities=_TOOL_CAPABILITIES,
    programs=_TOOL_PROGRAMS,
    mode_capabilities=_modes.MODE_CAPABILITIES,
)
_TOOL_ALIAS_REGISTRY_BY_NAME = {
    alias.name: alias
    for alias in _TOOL_ALIAS_REGISTRY
}

# Preserve the old Python import shape while callers migrate to metadata.
_TOOL_ALIASES = alias_dispatch_map(_TOOL_ALIAS_REGISTRY)


def dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    state: ServerState | None = None,
) -> dict[str, Any] | ImageToolResult:
    log_event(f"dispatch_tool start name={name}")
    alias = _TOOL_ALIAS_REGISTRY_BY_NAME.get(name)
    if alias:
        resolved = alias.target
        arguments = alias.translate_arguments(arguments)
    else:
        resolved = name
    handler = _TOOL_REGISTRY.get(resolved)
    if handler is None:
        raise ValueError(f"unknown tool: {name}")
    with bind_server_state(state or get_server_state()):
        payload = handler(arguments)
    if alias and alias.translate_result is not None:
        payload = alias.translate_result(payload)
    log_event(f"dispatch_tool done name={name}")
    return payload


def visible_tool_definitions(state: ServerState) -> list[dict[str, Any]]:
    """Return definitions exposed by one server's active filters."""
    return _modes.filter_tools(
        tool_definitions(),
        _TOOL_CAPABILITIES,
        state.mode,
        programs=state.programs,
        program_tags=_TOOL_PROGRAMS,
        toolset=state.toolset,
    )


def tool_unavailable_message(name: str, state: ServerState) -> str | None:
    """Explain why a registered tool is hidden from one server state."""
    alias = _TOOL_ALIAS_REGISTRY_BY_NAME.get(name)
    resolved = alias.target if alias else name
    if resolved not in _TOOL_REGISTRY:
        return None
    target_allowed = _modes.is_tool_allowed(
        resolved,
        _TOOL_CAPABILITIES,
        state.mode,
        programs=state.programs,
        program_tags=_TOOL_PROGRAMS,
        toolset=state.toolset,
    )
    alias_allowed = alias is None or is_alias_available(
        alias,
        mode=state.mode,
        programs=state.programs,
        mode_capabilities=_modes.MODE_CAPABILITIES,
    )
    if target_allowed and alias_allowed:
        return None

    capability = (
        alias.availability.capability
        if alias is not None
        else _TOOL_CAPABILITIES.get(resolved, "none")
    )
    program = (
        alias.availability.program
        if alias is not None
        else _TOOL_PROGRAMS.get(resolved, "generic")
    )
    if state.toolset is not None and resolved not in state.toolset:
        message = (
            f"tool {name!r} is outside the active toolset filter "
            f"(CHEMTOOLS_TOOLSET). Unset it or add the tool to expose it."
        )
    elif (
        state.programs is not None
        and program != "generic"
        and program not in state.programs
    ):
        message = (
            f"tool {name!r} (program={program}) is not in the active program "
            f"filter {sorted(state.programs)}. Restart with CHEMTOOLS_PROGRAMS "
            f"including {program!r} to enable it."
        )
    else:
        message = (
            f"tool {name!r} (capability={capability}) is not available in "
            f"server mode {state.mode!r}. Restart with --mode=local or "
            f"--mode=hpc to enable it."
        )
    log_event(
        f"tools/call blocked name={name} mode={state.mode} "
        f"prog={program} tag={capability}"
    )
    return message


def handle_request(
    message: dict[str, Any],
    *,
    state: ServerState | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    state = state or get_server_state()
    active_mode = state.mode
    active_programs = state.programs
    active_toolset = state.toolset
    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params", {})
    log_event(f"handle_request method={method} id={request_id}")

    if method == "notifications/initialized":
        return None, False
    if method == "exit":
        return None, True
    if method == "initialize":
        log_event("initialize requested")
        protocol_version = negotiate_protocol_version(params.get("protocolVersion"))
        return (
            make_response(
                request_id,
                {
                    "protocolVersion": protocol_version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                    },
                },
            ),
            False,
        )
    if method == "ping":
        return make_response(request_id, {}), False
    if method == "shutdown":
        return make_response(request_id, {}), True
    if method == "tools/list":
        log_event(f"tools/list requested mode={active_mode} programs={active_programs} toolset={'set' if active_toolset else None}")
        visible = visible_tool_definitions(state)
        return make_response(request_id, {"tools": visible}), False
    if method == "tools/call":
        try:
            tool_name = params["name"]
            arguments = params.get("arguments", {})
            log_event(f"tools/call name={tool_name} mode={active_mode}")
            unavailable = tool_unavailable_message(tool_name, state)
            if unavailable is not None:
                return make_response(
                    request_id,
                    make_error_result(unavailable),
                ), False
            payload = dispatch_tool(tool_name, arguments, state=state)
            return make_response(request_id, make_success_result(payload)), False
        except Exception as exc:
            log_event(f"tools/call error name={params.get('name')} error={exc}")
            return make_response(request_id, make_error_result(str(exc))), False

    if request_id is None:
        return None, False
    return (
        make_response(
            request_id,
            error={
                "code": -32601,
                "message": f"Method not found: {method}",
            },
        ),
        False,
    )


__all__ = [
    "tool_definitions",
    "visible_tool_definitions",
    "tool_unavailable_message",
    "dispatch_tool",
    "handle_request",
    "_TOOL_ALIAS_REGISTRY",
    "_TOOL_ALIASES",
]

"""JSON-RPC dispatch + cross-program tool aggregator.

Owns the multi-program glue that used to live at the bottom of
``chemtools/mcp/tools/nwchem.py``:

  * ``tool_definitions()``   Aggregator that concatenates each program's
                             tool definitions + generics.
  * ``dispatch_tool(name, arguments)``
                             Per-tool dispatcher (resolves aliases via
                             ``_TOOL_ALIASES`` + invokes the registered
                             handler).
  * ``handle_request(message)``
                             JSON-RPC method dispatcher used by the CLI's
                             ``serve()`` loop. Honors the active mode +
                             program filter.
  * ``_TOOL_ALIASES``        Back-compat alias map (renamed tools).

The active server mode + program filter are read live from
``chemtools.mcp.decorator`` so the CLI's ``main()`` can update them
without circular imports.
"""
from __future__ import annotations

from typing import Any

from chemtools.mcp.decorator import (
    _TOOL_REGISTRY,
    _TOOL_CAPABILITIES,
    _TOOL_PROGRAMS,
    log_event,
    SERVER_NAME,
    SERVER_VERSION,
    DEFAULT_PROTOCOL_VERSION,
)
from chemtools.mcp.server import (
    make_response,
    make_success_result,
    make_error_result,
)
from chemtools.mcp import modes as _modes
# Eager registration imports — ensure every program's @_tool decorators
# have run before tool_definitions() is called.
from chemtools.mcp.tools import generic as _generic_tools  # noqa: F401
from chemtools.mcp.tools import nwchem as _nwchem_tools  # noqa: F401
from chemtools.mcp.tools import molcas as _molcas_tools  # noqa: F401
from chemtools.mcp.tools import dirac as _dirac_tools  # noqa: F401
from chemtools.mcp.tools import grasp as _grasp_tools  # noqa: F401


def tool_definitions() -> list[dict[str, Any]]:
    """Concatenate every program's tool-definition list + generics."""
    from chemtools.mcp.tools.generic import generic_tool_definitions
    from chemtools.mcp.tools.nwchem import _nwchem_tool_definitions
    from chemtools.mcp.tools.molcas import molcas_tool_definitions
    from chemtools.mcp.tools.dirac import dirac_tool_definitions
    from chemtools.mcp.tools.grasp import grasp_tool_definitions
    return (
        generic_tool_definitions()
        + _nwchem_tool_definitions()
        + molcas_tool_definitions()
        + dirac_tool_definitions()
        + grasp_tool_definitions()
    )


# ---------------------------------------------------------------------------
# Backward-compat alias map: old tool names → (current name, arg translator).
# These are NOT in tool_definitions() so models see only the current names.
# ---------------------------------------------------------------------------

def _identity(args: dict[str, Any]) -> dict[str, Any]:
    return args


def _scf_fix_args(args: dict[str, Any]) -> dict[str, Any]:
    args = dict(args)
    args["mode"] = "scf"
    return args


def _state_recovery_args(args: dict[str, Any]) -> dict[str, Any]:
    args = dict(args)
    args["mode"] = "state"
    return args


def _compact_to_detail(args: dict[str, Any]) -> dict[str, Any]:
    args = dict(args)
    if args.pop("compact", False):
        args["detail"] = "compact"
    return args


_TOOL_ALIASES: dict[str, tuple[str, Any]] = {
    "diagnose_nwchem_output": ("analyze_nwchem_case", _identity),
    "summarize_nwchem_case": ("analyze_nwchem_case", _compact_to_detail),
    "review_nwchem_case": ("analyze_nwchem_case", _compact_to_detail),
    "check_nwchem_run_status": ("get_nwchem_run_status", _identity),
    "review_nwchem_followup_outcome": ("compare_nwchem_runs", _identity),
    "suggest_nwchem_scf_fix_strategy": ("suggest_nwchem_recovery", _scf_fix_args),
    "suggest_nwchem_state_recovery_strategy": ("suggest_nwchem_recovery", _state_recovery_args),
    "prepare_nwchem_run": ("launch_nwchem_run", _identity),
    "render_nwchem_basis_from_input": ("render_nwchem_basis_block", _identity),
    "summarize_cube_file": ("parse_cube_file", lambda args: {**args, "summarize": True}),
    "resolve_nwchem_ecp": ("render_nwchem_ecp_block", _identity),
    "render_nwchem_ecp_from_elements": ("render_nwchem_ecp_block", _identity),
    "resolve_nwchem_basis_setup": ("render_nwchem_basis_setup", _identity),
}


def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    log_event(f"dispatch_tool start name={name}")
    alias = _TOOL_ALIASES.get(name)
    if alias:
        resolved, translate = alias
        arguments = translate(arguments)
    else:
        resolved = name
    handler = _TOOL_REGISTRY.get(resolved)
    if handler is None:
        raise ValueError(f"unknown tool: {name}")
    payload = handler(arguments)
    log_event(f"dispatch_tool done name={name}")
    return payload


def handle_request(message: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    # Read mode live from the decorator module so the CLI's main() can update
    # it without import cycles.
    from chemtools.mcp.decorator import ACTIVE_MODE

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
        requested_version = params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION
        return (
            make_response(
                request_id,
                {
                    "protocolVersion": requested_version,
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
        log_event(f"tools/list requested mode={ACTIVE_MODE}")
        visible = _modes.filter_tools(tool_definitions(), _TOOL_CAPABILITIES, ACTIVE_MODE)
        return make_response(request_id, {"tools": visible}), False
    if method == "tools/call":
        try:
            tool_name = params["name"]
            arguments = params.get("arguments", {})
            log_event(f"tools/call name={tool_name} mode={ACTIVE_MODE}")
            # Gate against tools that the active mode does not expose. Resolves
            # aliases to the canonical name so blocked tools cannot be reached
            # via a back-compat alias either.
            resolved_for_check = _TOOL_ALIASES.get(tool_name, (tool_name, None))[0]
            if not _modes.is_tool_allowed(resolved_for_check, _TOOL_CAPABILITIES, ACTIVE_MODE):
                tag = _TOOL_CAPABILITIES.get(resolved_for_check, "none")
                msg = (
                    f"tool {tool_name!r} (capability={tag}) is not available in "
                    f"server mode {ACTIVE_MODE!r}. Restart with --mode=local or "
                    f"--mode=hpc to enable it."
                )
                log_event(f"tools/call blocked name={tool_name} mode={ACTIVE_MODE} tag={tag}")
                return make_response(request_id, make_error_result(msg)), False
            payload = dispatch_tool(tool_name, arguments)
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
    "dispatch_tool",
    "handle_request",
    "_TOOL_ALIASES",
]

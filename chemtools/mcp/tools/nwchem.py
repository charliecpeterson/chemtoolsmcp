"""NWChem MCP tool surface — entry module.

Thin aggregator: re-exports the shared base namespace, keeps the back-compat
__getattr__ shim, and imports the per-category handler modules so their @_tool
decorators register. Schemas -> _nwchem_schemas, shared imports/helpers ->
_nwchem_base, next_actions builder -> _nwchem_next_actions, handlers ->
nwchem_<category>.
"""
from __future__ import annotations

from chemtools.mcp.tools._nwchem_base import *  # noqa: F401,F403
from chemtools.mcp.tools._nwchem_base import (  # noqa: F401
    _tool,
    _build_next_actions,
    _nwchem_tool_definitions,
    _resolve_plugin_or_error,
    _dispatch_to_per_program_tool,
    basis_library_path,
    tool_definitions,
)


# ---------------------------------------------------------------------------
# Back-compat re-exports.
#
# The JSON-RPC dispatcher, tool-alias map, and the generic (cross-program)
# handlers all moved out of this file:
#
#     dispatch_tool / handle_request / _TOOL_ALIASES   → chemtools.mcp.dispatch
#     generic @_tool handlers (parse_output, etc.)     → chemtools.mcp.tools.generic
#
# Older tests and external callers still import these symbols from
# ``chemtools.mcp.tools.nwchem``. We re-export them via ``__getattr__`` to
# avoid circular-import problems (dispatch.py eagerly imports this module
# via tools/__init__.py and would not yet exist at this point in startup).
# ---------------------------------------------------------------------------


_BACKCOMPAT_FROM_DISPATCH = {
    "dispatch_tool",
    "handle_request",
    "_TOOL_ALIASES",
}

_BACKCOMPAT_FROM_GENERIC = {
    "_handle_get_server_mode",
    "_handle_summarize_run",
    "_handle_parse_output_generic",
    "_handle_extract_geometry",
    "_handle_parse_thermochem_generic",
    "_handle_parse_frequencies_generic",
    "_handle_parse_trajectory_generic",
    "_handle_inspect_geometry_generic",
    "_handle_summarize_output_generic",
    "_handle_analyze_case_generic",
    "_handle_suggest_recovery_generic",
    "_handle_apply_recovery_generic",
    "_handle_draft_initial_geometry",
    "_handle_compute_reaction_energy",
    "_handle_suggest_relativistic_correction",
    "_handle_suggest_spin_state",
    "_handle_suggest_basis_set",
    "_handle_suggest_memory",
    "_handle_suggest_resources",
    "_handle_parse_cube_file",
    "_handle_preflight_check",
    "_handle_render_job_script",
    "_handle_watch_multiple_runs",
    "_handle_init_session_log",
    "_handle_append_session_log",
    "_handle_next_versioned_path",
    "_handle_basis_library_summary",
    "_handle_register_run_generic",
    "_handle_update_run_status_generic",
    "_handle_list_runs_generic",
    "_handle_get_run_summary_generic",
    "_handle_create_campaign_generic",
    "_handle_get_campaign_status_generic",
    "_handle_get_campaign_energies_generic",
    "_handle_create_workflow_generic",
    "_handle_advance_workflow_generic",
    "_resolve_plugin_or_error",
    "_dispatch_to_per_program_tool",
}


def __getattr__(name: str):  # pragma: no cover — straightforward re-export shim
    if name in _BACKCOMPAT_FROM_DISPATCH:
        from chemtools.mcp import dispatch
        return getattr(dispatch, name)
    if name in _BACKCOMPAT_FROM_GENERIC:
        from chemtools.mcp.tools import generic
        return getattr(generic, name)
    # Handlers moved into nwchem_<category> modules; resolve them here so legacy
    # `from chemtools.mcp.tools.nwchem import _handle_x` / `_do_x` imports work.
    if name.startswith(("_handle_", "_do_")):
        from chemtools.mcp.tools import (
            nwchem_input, nwchem_parse, nwchem_analysis, nwchem_jobs, nwchem_docs,
        )
        for mod in (nwchem_input, nwchem_parse, nwchem_analysis, nwchem_jobs, nwchem_docs):
            if hasattr(mod, name):
                return getattr(mod, name)
    raise AttributeError(f"module 'chemtools.mcp.tools.nwchem' has no attribute {name!r}")


# Import the per-category handler modules so their @_tool decorators register.
from chemtools.mcp.tools import (  # noqa: F401,E402
    nwchem_input,
    nwchem_parse,
    nwchem_analysis,
    nwchem_jobs,
    nwchem_docs,
)

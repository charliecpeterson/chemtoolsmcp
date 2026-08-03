"""Shared imports + helpers for the NWChem MCP handler modules.

Holds the public chemtools surface, the @_tool decorator, basis_library_path,
and the two plugin-dispatch helpers. The per-category nwchem_<cat> handler
modules pull this in via `from _nwchem_base import *`; nwchem.py re-exports it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

# Repo-root fallback for source-tree runs (mirrors mcp/nwchem.py).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if not any("chemtools" in p for p in sys.path):
    sys.path.insert(0, str(_REPO_ROOT))

# Public chemtools surface — all the functions the handlers below call.
from chemtools import (  # noqa: E402
    analyze_frontier_orbitals,
    draft_initial_geometry,
    plan_nwchem_workflow,
    suggest_basis_set,
    suggest_memory,
    suggest_resources,
    suggest_relativistic_correction,
    suggest_spin_state,
    recommend_multiplicity_scan,
    validate_nwchem_tce_setup,
    analyze_imaginary_modes,
    check_nwchem_geometry_plausibility,
    check_nwchem_freq_plausibility,
    check_nwchem_run_status,
    compare_nwchem_runs,
    create_nwchem_input,
    create_nwchem_input_variant,
    create_nwchem_dft_workflow_input,
    displace_geometry_along_mode,
    extract_nwchem_geometry,
    draft_nwchem_cube_input,
    draft_nwchem_frontier_cube_input,
    draft_nwchem_imaginary_mode_inputs,
    draft_nwchem_mcscf_input,
    draft_nwchem_mcscf_retry_input,
    draft_nwchem_optimization_followup_input,
    draft_nwchem_property_check_input,
    draft_nwchem_scf_stabilization_input,
    draft_nwchem_tce_input,
    draft_nwchem_tce_restart_input,
    draft_nwchem_atom_input,
    draft_nwchem_vectors_swap_input,
    compute_reaction_energy,
    parse_nwchem_thermochem,
    summarize_electronic_structure,
    track_spin_state_across_optimization,
    find_restart_assets,
    inspect_input,
    inspect_runner_profiles,
    lint_nwchem_input,
    parse_cube,
    parse_freq_progress,
    parse_mcscf_output,
    parse_mos,
    parse_nwchem_movecs,
    parse_output,
    parse_population_analysis,
    parse_scf,
    parse_tce_amplitudes,
    parse_tce_output,
    preflight_check,
    get_nwchem_workflow_state,
    plan_calculation,
    list_protocols,
    prepare_freq_restart,
    prepare_nwchem_next_step,
    render_basis_block,
    render_basis_block_from_geometry,
    render_ecp_block,
    render_nwchem_basis_setup,
    resolve_basis,
    resolve_ecp,
    review_nwchem_followup_outcome,
    review_nwchem_mcscf_case,
    review_nwchem_mcscf_followup_outcome,
    review_nwchem_progress,
    suggest_nwchem_mcscf_active_space,
    suggest_nwchem_scf_fix_strategy,
    suggest_nwchem_state_recovery_strategy,
    suggest_tce_freeze_count,
    suggest_vectors_swaps,
    summarize_cube,
    summarize_nwchem_case,
    swap_nwchem_movecs,
    tail_nwchem_output,
    render_job_script,
    watch_nwchem_run,
    watch_multiple_nwchem_runs,
    init_session_log,
    append_session_log,
    next_versioned_path,
    register_run,
    update_run_status,
    list_runs,
    get_run_summary,
    create_campaign,
    get_campaign_status,
    get_campaign_energies,
    create_workflow,
    advance_workflow,
    generate_input_batch,
    create_nwchem_dft_input_from_request,
    basis_library_summary,
    check_spin_charge_state,
    inspect_nwchem_geometry,
    parse_tasks,
    parse_trajectory,
    review_nwchem_input_request,
    summarize_output,
    summarize_nwchem_outputs,
    check_memory_fit,
    estimate_freq_walltime,
    suggest_hpc_resources,
    detect_hpc_accounts,
    suggest_partition,
    draft_nwchem_pyscf_reference,
)
from chemtools.core.eval import evaluate_case, evaluate_cases  # noqa: E402
from chemtools.programs.nwchem.docs import (  # noqa: E402
    find_examples as docs_find_examples,
    get_topic_guide as docs_get_topic_guide,
    list_docs as docs_list_docs,
    lookup_block_syntax as docs_lookup_block_syntax,
    read_doc_excerpt as docs_read_doc_excerpt,
    search_docs as docs_search_docs,
)
from chemtools.programs.nwchem.forum import search_forum as forum_search  # noqa: E402

# MCP framework — registries, decorator, server-side helpers.
from chemtools.mcp.decorator import (  # noqa: E402
    _TOOL_REGISTRY,
    _TOOL_CAPABILITIES,
    _TOOL_PROGRAMS,
    _tool as _raw_tool,
    log_event,
    ACTIVE_MODE,
    SERVER_NAME,
    SERVER_VERSION,
    DEFAULT_PROTOCOL_VERSION,
)


def _tool(name: str, *, needs: str = "none", program: str = "nwchem"):
    """Program-scoped @_tool wrapper for NWChem. All tools in this module
    are tagged with program='nwchem' by default — pass program='generic'
    on tools that work across programs (basis advisors, session log,
    reaction energy, etc.)."""
    return _raw_tool(name, needs=needs, program=program)
from chemtools.mcp.server import (  # noqa: E402
    make_response,
    make_success_result,
    make_error_result,
)
from chemtools.mcp import modes as _modes  # noqa: E402

# Basis library: bundled inside the package at chemtools/data/nwchem/basis_library/
# Can be overridden at runtime with CHEMTOOLS_BASIS_LIBRARY env var.
import os  # noqa: E402
try:
    from importlib.resources import files as _pkg_files  # noqa: E402
    DEFAULT_BASIS_LIBRARY = Path(str(_pkg_files("chemtools").joinpath("data/nwchem/basis_library")))
except Exception:
    DEFAULT_BASIS_LIBRARY = _REPO_ROOT / "chemtools" / "data" / "nwchem" / "basis_library"


def basis_library_path(path: str | None = None) -> str:
    if path:
        return path
    return os.environ.get("CHEMTOOLS_BASIS_LIBRARY", str(DEFAULT_BASIS_LIBRARY))


def tool_definitions() -> list[dict[str, Any]]:
    """Back-compat shim — delegates to ``chemtools.mcp.dispatch.tool_definitions``.

    The aggregator now lives in ``chemtools/mcp/dispatch.py`` so the
    multi-program glue is in one place. This shim preserves the legacy
    import path ``from chemtools.mcp.tools.nwchem import tool_definitions``
    used by older tests and external callers.
    """
    from chemtools.mcp.dispatch import tool_definitions as _td
    return _td()


# Schema definitions live in _nwchem_schemas.py (pure data); re-exported here
# so dispatch's `from chemtools.mcp.tools.nwchem import _nwchem_tool_definitions`
# keeps working.
from chemtools.mcp.tools._nwchem_schemas import _nwchem_tool_definitions  # noqa: F401,E402

# next_actions builder (~900-line pure-data helper) lives in its own module.
from chemtools.mcp.tools._nwchem_next_actions import _build_next_actions  # noqa: F401,E402


# ---------------------------------------------------------------------------
# Handlers — frontier orbital / vectors-swap workflow
# ---------------------------------------------------------------------------



def _resolve_plugin_or_error(arguments: dict[str, Any]):
    """Shared dispatch logic: resolve plugin from `program` override or auto-
    detect from `output_file`. Returns (plugin, None) on success or
    (None, error_dict) on failure."""
    from chemtools.core import registry as _registry
    output_file = arguments["output_file"]
    program = arguments.get("program")
    try:
        plugin = _registry.resolve(program=program, path=output_file)
        return plugin, None
    except _registry.ProgramDetectionAmbiguous as e:
        return None, {
            "error": "program_detection_ambiguous",
            "message": str(e),
            "candidates": list(e.candidates),
        }
    except _registry.ProgramContentMismatch as e:
        return None, {
            "error": "program_content_mismatch",
            "message": str(e),
            "program": e.program,
            "detected_programs": list(e.candidates),
        }
    except _registry.ProgramDetectorError as e:
        return None, {
            "error": "program_detector_error",
            "message": str(e),
            "candidates": list(e.candidates),
            "detector_failures": [
                {
                    "program": failure.program,
                    "error_type": failure.error_type,
                    "message": failure.message,
                }
                for failure in e.failures
            ],
        }
    except _registry.ProgramDetectionSourceError as e:
        return None, {
            "error": "program_source_error",
            "message": str(e),
            "path": e.path,
            "source_failure": {
                "error_type": e.failure.error_type,
                "message": e.failure.message,
                "errno": e.failure.errno,
            },
        }
    except _registry.ProgramDetectionFailed as e:
        return None, {
            "error": "program_detection_failed",
            "message": str(e),
            "registered_programs": _registry.list_programs(),
        }
    except _registry.ProgramNotRegistered as e:
        return None, {
            "error": "program_not_registered",
            "message": str(e),
            "registered_programs": _registry.list_programs(),
        }

# ---------------------------------------------------------------------------
# Generic case-analysis / recovery dispatchers (Phase 6a)
# ---------------------------------------------------------------------------
# Auto-detect program and dispatch to the appropriate program-specific
# tool. Each returns the per-program shape tagged with "program" so the
# agent can dispatch its own follow-up logic — the alternative (forcing
# both programs to a unified shape) would be a much larger refactor and
# the per-program rich data is genuinely useful where it differs.




def _dispatch_to_per_program_tool(
    arguments: dict[str, Any],
    handler_by_program: dict[str, "Callable[[dict[str, Any]], dict[str, Any]]"],
) -> dict[str, Any]:
    plugin, err = _resolve_plugin_or_error(arguments)
    if err is not None:
        return err
    handler = handler_by_program.get(plugin.name)
    if handler is None:
        return {
            "error": "no_handler_for_program",
            "message": (
                f"No {arguments.get('_tool_label', 'handler')} for program "
                f"{plugin.name!r}. Available programs: "
                f"{sorted(handler_by_program)}"
            ),
        }
    result = handler(arguments)
    if isinstance(result, dict):
        return {"program": plugin.name, **result}
    return {"program": plugin.name, "result": result}

"""NWChem tool definitions and @_tool handlers for the chemtools-nwchem MCP server.

Extracted from chemtools/mcp/nwchem.py as the final step of the MCP
split. This module owns the tool surface:

  * `tool_definitions()`    Returns the list of tool spec dicts.
  * `@_tool` handlers       All ~112 NWChem tool handler functions.
  * `_TOOL_ALIASES`         Back-compat name aliases.
  * `dispatch_tool`         Per-tool dispatcher (resolves aliases + runs handler).
  * `handle_request`        JSON-RPC method dispatcher.

The CLI entry point in `chemtools/mcp/nwchem.py` imports `tool_definitions`,
`dispatch_tool`, `handle_request`, and `_TOOL_ALIASES` from here, plus the
mcp framework (`mcp.decorator`, `mcp.server`).
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
    launch_nwchem_run,
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
    terminate_nwchem_run,
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

# Eagerly import Molcas + GRASP tool handlers so their @_tool decorators register
# with _TOOL_REGISTRY before serve() starts dispatching.
from chemtools.mcp.tools import molcas as _molcas_tools  # noqa: F401, E402
from chemtools.mcp.tools import grasp as _grasp_tools  # noqa: F401, E402

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

@_tool("prepare_nwchem_mcscf_setup")
def _handle_prepare_nwchem_mcscf_setup(arguments: dict[str, Any]) -> dict[str, Any]:
    from chemtools.programs.nwchem.strategy.active_space import prepare_nwchem_mcscf_setup
    return prepare_nwchem_mcscf_setup(
        scf_output_path=arguments["scf_output_path"],
        input_path=arguments.get("input_path"),
        expected_metal_elements=arguments.get("expected_metal_elements"),
        expected_somo_count=arguments.get("expected_somo_count"),
        prefer_expanded=arguments.get("prefer_expanded", False),
    )


@_tool("prepare_nwchem_tce_setup")
def _handle_prepare_nwchem_tce_setup(arguments: dict[str, Any]) -> dict[str, Any]:
    """Thick orchestrator — parses MOs, computes freeze count, checks ordering,
    suggests swaps, and returns a Diagnosis with next_actions."""
    from chemtools.programs.nwchem.strategy.active_space import prepare_nwchem_tce_setup
    return prepare_nwchem_tce_setup(
        scf_output_path=arguments["scf_output_path"],
        target_method=arguments.get("target_method", "ccsd(t)"),
        elements=arguments.get("elements"),
        charge=arguments.get("charge", 0),
        multiplicity=arguments.get("multiplicity", 1),
        expected_metal_elements=arguments.get("expected_metal_elements"),
        expected_somo_count=arguments.get("expected_somo_count"),
        ecp_core_electrons=arguments.get("ecp_core_electrons"),
    )

# ---------------------------------------------------------------------------
# Generic auto-detect tool dispatchers (Phase 4)
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

@_tool("parse_nwchem_output")
def _handle_parse_nwchem_output(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_output(
        arguments["output_file"],
        sections=arguments.get("sections"),
        top_n=arguments.get("top_n", 5),
        include_coefficients=arguments.get("include_coefficients", False),
        include_displacements=arguments.get("include_displacements", False),
        include_positions=arguments.get("include_positions", False),
    )


@_tool("plan_nwchem_workflow")
def _handle_plan_nwchem_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    return plan_nwchem_workflow(
        goal=arguments["goal"],
        elements=arguments["elements"],
        charge=arguments["charge"],
        multiplicity=arguments["multiplicity"],
        basis=arguments.get("basis"),
        method=arguments.get("method", "ccsd"),
        xc_functional=arguments.get("xc_functional", "b3lyp"),
        has_geometry_file=arguments.get("has_geometry_file", False),
        has_dft_output=arguments.get("has_dft_output", False),
        has_scf_output=arguments.get("has_scf_output", False),
    )

@_tool("draft_nwchem_atom_input")
def _handle_draft_nwchem_atom_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_atom_input(
        element=arguments["element"],
        basis=arguments["basis"],
        method=arguments.get("method", "scf"),
        charge=arguments.get("charge", 0),
        multiplicity=arguments.get("multiplicity"),
        xc_functional=arguments.get("xc_functional", "m06"),
        memory=arguments.get("memory"),
        start_name=arguments.get("start_name"),
        output_dir=arguments.get("output_dir"),
        write_file=arguments.get("write_file", False),
        basis_library=basis_library_path(arguments.get("basis_library")),
    )

@_tool("parse_nwchem_thermochem")
def _handle_parse_nwchem_thermochem(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_nwchem_thermochem(
        path=arguments["output_file"],
        T=arguments.get("T", 298.15),
        P=arguments.get("P", 1.0),
    )


@_tool("summarize_nwchem_electronic_structure")
def _handle_summarize_electronic_structure(arguments: dict[str, Any]) -> dict[str, Any]:
    result = summarize_electronic_structure(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
    )
    result["next_actions"] = _build_next_actions(
        "electronic_structure", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("track_nwchem_spin_state")
def _handle_track_spin_state(arguments: dict[str, Any]) -> dict[str, Any]:
    result = track_spin_state_across_optimization(
        output_path=arguments["output_file"],
    )
    result["next_actions"] = _build_next_actions(
        "track_spin_state", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result

@_tool("analyze_nwchem_frontier_orbitals")
def _handle_analyze_nwchem_frontier_orbitals(arguments: dict[str, Any]) -> dict[str, Any]:
    result = analyze_frontier_orbitals(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
    )
    result["next_actions"] = _build_next_actions(
        "frontier_orbitals", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("suggest_nwchem_vectors_swaps")
def _handle_suggest_nwchem_vectors_swaps(arguments: dict[str, Any]) -> dict[str, Any]:
    return suggest_vectors_swaps(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
        vectors_input=arguments.get("vectors_input"),
        vectors_output=arguments.get("vectors_output"),
    )


@_tool("draft_nwchem_vectors_swap_input")
def _handle_draft_nwchem_vectors_swap_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_vectors_swap_input(
        output_path=arguments["output_file"],
        input_path=arguments["input_file"],
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
        vectors_input=arguments.get("vectors_input"),
        vectors_output=arguments.get("vectors_output"),
        task_operation=arguments.get("task_operation", "energy"),
        iterations=arguments.get("iterations", 500),
        smear=arguments.get("smear", 0.001),
        convergence_damp=arguments.get("convergence_damp", 30),
        convergence_ncydp=arguments.get("convergence_ncydp", 30),
        population_print=arguments.get("population_print", "mulliken"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        title=arguments.get("title"),
        write_file=arguments.get("write_file", False),
    )


@_tool("draft_nwchem_property_check_input")
def _handle_draft_nwchem_property_check_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_property_check_input(
        input_path=arguments["input_file"],
        reference_output_path=arguments.get("reference_output_file"),
        vectors_input=arguments.get("vectors_input"),
        vectors_output=arguments.get("vectors_output"),
        property_keywords=arguments.get("property_keywords"),
        task_strategy=arguments.get("task_strategy", "auto"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
        iterations=arguments.get("iterations", 1),
        convergence_energy=arguments.get("convergence_energy", "1e-3"),
        smear=arguments.get("smear"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        title=arguments.get("title"),
        write_file=arguments.get("write_file", False),
    )


@_tool("draft_nwchem_scf_stabilization_input")
def _handle_draft_nwchem_scf_stabilization_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_scf_stabilization_input(
        input_path=arguments["input_file"],
        reference_output_path=arguments.get("reference_output_file"),
        vectors_input=arguments.get("vectors_input"),
        vectors_output=arguments.get("vectors_output"),
        task_operation=arguments.get("task_operation", "energy"),
        iterations=arguments.get("iterations"),
        smear=arguments.get("smear"),
        convergence_damp=arguments.get("convergence_damp"),
        convergence_ncydp=arguments.get("convergence_ncydp"),
        population_print=arguments.get("population_print"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        title=arguments.get("title"),
        write_file=arguments.get("write_file", False),
    )


@_tool("draft_nwchem_optimization_followup_input")
def _handle_draft_nwchem_optimization_followup_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_optimization_followup_input(
        output_path=arguments["output_file"],
        input_path=arguments["input_file"],
        task_strategy=arguments.get("task_strategy", "auto"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        title=arguments.get("title"),
        write_file=arguments.get("write_file", False),
    )


@_tool("extract_nwchem_geometry")
def _handle_extract_nwchem_geometry(arguments: dict[str, Any]) -> dict[str, Any]:
    frame_arg = arguments.get("frame", "best")
    try:
        frame_arg = int(frame_arg)
    except (TypeError, ValueError):
        pass
    result = extract_nwchem_geometry(
        output_path=arguments["output_file"],
        frame=frame_arg,
        input_path=arguments.get("input_file"),
    )
    return result


@_tool("draft_nwchem_cube_input")
def _handle_draft_nwchem_cube_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_cube_input(
        input_path=arguments["input_file"],
        vectors_input=arguments["vectors_input"],
        orbital_vectors=arguments.get("orbital_vectors"),
        density_modes=arguments.get("density_modes"),
        orbital_spin=arguments.get("orbital_spin", "total"),
        extent_angstrom=arguments.get("extent_angstrom", 6.0),
        grid_points=arguments.get("grid_points", 120),
        gaussian=arguments.get("gaussian", True),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        title=arguments.get("title"),
        write_file=arguments.get("write_file", False),
    )


@_tool("draft_nwchem_frontier_cube_input")
def _handle_draft_nwchem_frontier_cube_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_frontier_cube_input(
        output_path=arguments["output_file"],
        input_path=arguments["input_file"],
        vectors_input=arguments.get("vectors_input"),
        include_somos=arguments.get("include_somos", True),
        include_homo=arguments.get("include_homo", True),
        include_lumo=arguments.get("include_lumo", True),
        include_density_modes=arguments.get("include_density_modes"),
        extent_angstrom=arguments.get("extent_angstrom", 6.0),
        grid_points=arguments.get("grid_points", 120),
        gaussian=arguments.get("gaussian", True),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        title=arguments.get("title"),
        write_file=arguments.get("write_file", False),
    )


# ---------------------------------------------------------------------------
# Handlers — output parsers
# ---------------------------------------------------------------------------

@_tool("parse_nwchem_scf")
def _handle_parse_nwchem_scf(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_scf(arguments["file_path"])

@_tool("parse_nwchem_mos")
def _handle_parse_nwchem_mos(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_mos(
        arguments["file_path"],
        top_n=arguments.get("top_n", 5),
        include_coefficients=arguments.get("include_coefficients", False),
        include_all_orbitals=arguments.get("include_all_orbitals", False),
    )


@_tool("parse_nwchem_mcscf_output")
def _handle_parse_nwchem_mcscf_output(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_mcscf_output(arguments["file_path"])


@_tool("parse_nwchem_population_analysis")
def _handle_parse_nwchem_population_analysis(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_population_analysis(arguments["file_path"])


# ---------------------------------------------------------------------------
# Handlers — input inspection and linting
# ---------------------------------------------------------------------------

@_tool("inspect_nwchem_input")
def _handle_inspect_nwchem_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return inspect_input(arguments["input_file"])


@_tool("inspect_nwchem_runner_profiles", needs="runner_profile")
def _handle_inspect_nwchem_runner_profiles(arguments: dict[str, Any]) -> dict[str, Any]:
    return inspect_runner_profiles(arguments.get("profiles_path"))

@_tool("lint_nwchem_input")
def _handle_lint_nwchem_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return lint_nwchem_input(
        input_path=arguments["input_file"],
        library_path=basis_library_path(arguments.get("library_path")),
    )


@_tool("find_nwchem_restart_assets")
def _handle_find_nwchem_restart_assets(arguments: dict[str, Any]) -> dict[str, Any]:
    return find_restart_assets(arguments["path"])


# ---------------------------------------------------------------------------
# Handlers — runner / job management
# ---------------------------------------------------------------------------

@_tool("launch_nwchem_run", needs="executable")
def _handle_launch_nwchem_run(arguments: dict[str, Any]) -> dict[str, Any]:
    dry_run = arguments.get("dry_run", False)
    auto_watch = arguments.get("auto_watch", True)
    auto_register = arguments.get("auto_register", True)
    result = launch_nwchem_run(
        input_path=arguments["input_file"],
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path"),
        job_name=arguments.get("job_name"),
        resource_overrides=arguments.get("resource_overrides"),
        env_overrides=arguments.get("env_overrides"),
        write_script=arguments.get("write_script", True),
        dry_run=dry_run,
    )
    # Auto-register in the run registry
    if not dry_run and auto_register:
        try:
            reg = register_run(
                job_name=result.get("job_name", arguments.get("job_name", "")),
                input_file=arguments["input_file"],
                output_file=result.get("output_file"),
                profile=arguments["profile"],
                campaign_id=arguments.get("campaign_id"),
                workflow_id=arguments.get("workflow_id"),
                workflow_step_id=arguments.get("workflow_step_id"),
                parent_run_id=arguments.get("parent_run_id"),
                mpi_ranks=arguments.get("resource_overrides", {}).get("mpi_ranks") if arguments.get("resource_overrides") else None,
            )
            result["registry"] = reg
        except Exception as exc:
            result["registry_error"] = str(exc)

    # For scheduler jobs: automatically watch until terminal unless opted out
    if (
        not dry_run
        and auto_watch
        and result.get("launcher_kind") == "scheduler"
        and result.get("job_id")
    ):
        out_file = result.get("output_file")
        in_file = arguments["input_file"]
        profiles_path = arguments.get("profiles_path")
        profile = arguments["profile"]
        watch_result = watch_nwchem_run(
            output_path=out_file,
            input_path=in_file,
            profile=profile,
            job_id=result["job_id"],
            profiles_path=profiles_path,
            poll_interval_seconds=30.0,
            adaptive_polling=True,
            max_poll_interval_seconds=120.0,
            timeout_seconds=None,   # no timeout — let the scheduler walltime govern
        )
        result["watch"] = watch_result

        # Auto-update registry with final status
        if auto_register and result.get("registry", {}).get("run_id"):
            try:
                run_id = result["registry"]["run_id"]
                overall = watch_result.get("overall_status", "")
                status_map = {
                    "completed": "completed",
                    "failed": "failed",
                    "error": "failed",
                    "timelimit": "timelimited",
                    "cancelled": "cancelled",
                }
                reg_status = status_map.get(overall, overall)
                if reg_status:
                    update_kwargs: dict[str, Any] = {"run_id": run_id, "status": reg_status}
                    # Extract energy from watch result if available
                    prog = watch_result.get("progress_summary", {})
                    tasks = prog.get("tasks", []) if prog else []
                    if tasks:
                        last_task = tasks[-1]
                        if last_task.get("energy") is not None:
                            update_kwargs["energy_hartree"] = last_task["energy"]
                    update_run_status(**update_kwargs)
            except Exception:
                pass  # best-effort
    return result


@_tool("get_nwchem_run_status", needs="executable")
def _handle_get_nwchem_run_status(arguments: dict[str, Any]) -> dict[str, Any]:
    status = check_nwchem_run_status(
        output_path=arguments.get("output_file"),
        input_path=arguments.get("input_file"),
        error_path=arguments.get("error_file"),
        process_id=arguments.get("process_id"),
        profile=arguments.get("profile"),
        job_id=arguments.get("job_id"),
        profiles_path=arguments.get("profiles_path"),
    )
    # Add compact progress summary when output file is available
    if arguments.get("output_file"):
        try:
            progress = review_nwchem_progress(
                output_path=arguments["output_file"],
                input_path=arguments.get("input_file"),
                error_path=arguments.get("error_file"),
                process_id=arguments.get("process_id"),
                profile=arguments.get("profile"),
                job_id=arguments.get("job_id"),
                profiles_path=arguments.get("profiles_path"),
            )
            status["progress"] = progress
        except Exception as exc:
            status["progress_error"] = str(exc)
    status["next_actions"] = _build_next_actions(
        "run_status", status,
        output_file=arguments.get("output_file", ""),
        input_file=arguments.get("input_file", ""),
        profile=arguments.get("profile", ""),
    )
    return status


@_tool("tail_nwchem_output", needs="executable")
def _handle_tail_nwchem_output(arguments: dict[str, Any]) -> dict[str, Any]:
    return tail_nwchem_output(
        arguments["output_file"],
        lines=arguments.get("lines", 30),
        max_characters=min(arguments.get("max_characters", 4000), 10000),
    )


@_tool("terminate_nwchem_run", needs="executable")
def _handle_terminate_nwchem_run(arguments: dict[str, Any]) -> dict[str, Any]:
    profiles_path = arguments.get("profiles_path") or os.environ.get("CHEMTOOLS_RUNNER_PROFILES")
    return terminate_nwchem_run(
        process_id=arguments.get("process_id"),
        signal_name=arguments.get("signal_name", "term"),
        job_id=arguments.get("job_id"),
        profile=arguments.get("profile"),
        profiles_path=profiles_path,
    )


@_tool("watch_nwchem_run", needs="executable")
def _handle_watch_nwchem_run(arguments: dict[str, Any]) -> dict[str, Any]:
    result = watch_nwchem_run(
        output_path=arguments.get("output_file"),
        input_path=arguments.get("input_file"),
        error_path=arguments.get("error_file"),
        process_id=arguments.get("process_id"),
        profile=arguments.get("profile"),
        job_id=arguments.get("job_id"),
        profiles_path=arguments.get("profiles_path"),
        poll_interval_seconds=arguments.get("poll_interval_seconds", 10.0),
        adaptive_polling=arguments.get("adaptive_polling", True),
        max_poll_interval_seconds=arguments.get("max_poll_interval_seconds", 60.0),
        timeout_seconds=arguments.get("timeout_seconds", 3600.0),
        max_polls=arguments.get("max_polls"),
        history_limit=arguments.get("history_limit", 8),
    )
    result["next_actions"] = _build_next_actions(
        "watch_run", result,
        output_file=arguments.get("output_file", ""),
        input_file=arguments.get("input_file", ""),
        profile=arguments.get("profile", ""),
    )
    return result


# ---------------------------------------------------------------------------
# Handlers — run comparison and follow-up
# ---------------------------------------------------------------------------

@_tool("compare_nwchem_runs")
def _handle_compare_nwchem_runs(arguments: dict[str, Any]) -> dict[str, Any]:
    if arguments.get("output_dir") or arguments.get("base_name"):
        result = review_nwchem_followup_outcome(
            reference_output_path=arguments["reference_output_file"],
            candidate_output_path=arguments["candidate_output_file"],
            reference_input_path=arguments.get("reference_input_file"),
            candidate_input_path=arguments.get("candidate_input_file"),
            expected_metal_elements=arguments.get("expected_metals"),
            expected_somo_count=arguments.get("expected_somos"),
            output_dir=arguments.get("output_dir"),
            base_name=arguments.get("base_name"),
        )
    else:
        result = compare_nwchem_runs(
            reference_output_path=arguments["reference_output_file"],
            candidate_output_path=arguments["candidate_output_file"],
            reference_input_path=arguments.get("reference_input_file"),
            candidate_input_path=arguments.get("candidate_input_file"),
            expected_metal_elements=arguments.get("expected_metals"),
            expected_somo_count=arguments.get("expected_somos"),
        )
    result["next_actions"] = _build_next_actions(
        "compare_runs", result,
        output_file=arguments["candidate_output_file"],
        input_file=arguments.get("candidate_input_file", ""),
    )
    return result


@_tool("review_nwchem_mcscf_case")
def _handle_review_nwchem_mcscf_case(arguments: dict[str, Any]) -> dict[str, Any]:
    return review_nwchem_mcscf_case(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
    )


@_tool("review_nwchem_mcscf_followup_outcome")
def _handle_review_nwchem_mcscf_followup_outcome(arguments: dict[str, Any]) -> dict[str, Any]:
    return review_nwchem_mcscf_followup_outcome(
        reference_output_path=arguments["reference_output_file"],
        candidate_output_path=arguments["candidate_output_file"],
        reference_input_path=arguments.get("reference_input_file"),
        candidate_input_path=arguments.get("candidate_input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
    )


@_tool("prepare_nwchem_next_step")
def _handle_prepare_nwchem_next_step(arguments: dict[str, Any]) -> dict[str, Any]:
    return prepare_nwchem_next_step(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        write_files=arguments.get("write_files", False),
        include_property_check=arguments.get("include_property_check", True),
        include_frontier_cubes=arguments.get("include_frontier_cubes", False),
        include_density_modes=arguments.get("include_density_modes"),
        cube_extent_angstrom=arguments.get("cube_extent_angstrom", 6.0),
        cube_grid_points=arguments.get("cube_grid_points", 120),
    )


# ---------------------------------------------------------------------------
# Handlers — basis and ECP
# ---------------------------------------------------------------------------

@_tool("render_nwchem_basis_block")
def _handle_render_nwchem_basis_block(arguments: dict[str, Any]) -> dict[str, Any]:
    basis_name = arguments["basis_name"]
    library_path = basis_library_path(arguments.get("library_path"))
    if arguments.get("check_only", False):
        elements = arguments.get("elements") or []
        return resolve_basis(basis_name, elements, library_path)
    if arguments.get("input_file"):
        return render_basis_block_from_geometry(
            basis_name,
            arguments["input_file"],
            library_path,
            block_name=arguments.get("block_name", "ao basis"),
            mode=arguments.get("mode"),
        )
    return render_basis_block(
        basis_name,
        arguments.get("elements", []),
        library_path,
        block_name=arguments.get("block_name", "ao basis"),
        mode=arguments.get("mode"),
    )


@_tool("render_nwchem_ecp_block")
def _handle_render_nwchem_ecp_block(arguments: dict[str, Any]) -> dict[str, Any]:
    ecp_name = arguments["ecp_name"]
    elements = arguments["elements"]
    library_path = basis_library_path(arguments.get("library_path"))
    if arguments.get("check_only", False):
        return resolve_ecp(ecp_name, elements, library_path)
    return render_ecp_block(ecp_name, elements, library_path)


@_tool("render_nwchem_basis_setup")
def _handle_render_nwchem_basis_setup(arguments: dict[str, Any]) -> dict[str, Any]:
    return render_nwchem_basis_setup(
        geometry_path=arguments["geometry_file"],
        library_path=basis_library_path(arguments.get("library_path")),
        basis_assignments=arguments["basis_assignments"],
        ecp_assignments=arguments.get("ecp_assignments"),
        default_basis=arguments.get("default_basis"),
        default_ecp=arguments.get("default_ecp"),
        basis_block_name=arguments.get("block_name", "ao basis"),
        basis_mode=arguments.get("basis_mode"),
    )


# ---------------------------------------------------------------------------
# Handlers — input creation
# ---------------------------------------------------------------------------

@_tool("create_nwchem_input")
def _handle_create_nwchem_input(arguments: dict[str, Any]) -> dict[str, Any]:
    # Translate explicit SCF params into module_settings lines
    module = arguments.get("module", "").strip().lower()
    module_settings: list[str] = []
    if module == "scf":
        scf_type = arguments.get("scf_type")
        nopen = arguments.get("nopen")
        maxiter = arguments.get("maxiter")
        thresh = arguments.get("thresh")
        if scf_type:
            module_settings.append(scf_type)
        if nopen is not None:
            module_settings.append(f"nopen {nopen}")
        if maxiter is not None:
            module_settings.append(f"maxiter {maxiter}")
        if thresh is not None:
            module_settings.append(f"thresh {thresh:.2e}")
    return create_nwchem_input(
        geometry_path=arguments["geometry_file"],
        library_path=basis_library_path(arguments.get("library_path")),
        basis_assignments=arguments["basis_assignments"],
        ecp_assignments=arguments.get("ecp_assignments"),
        default_basis=arguments.get("default_basis"),
        default_ecp=arguments.get("default_ecp"),
        basis_block_name=arguments.get("block_name", "ao basis"),
        basis_mode=arguments.get("basis_mode"),
        module=arguments["module"],
        task_operation=arguments.get("task_operation"),
        charge=arguments.get("charge"),
        multiplicity=arguments.get("multiplicity"),
        module_settings=module_settings or None,
        extra_blocks=arguments.get("extra_blocks"),
        memory=arguments.get("memory"),
        title=arguments.get("title"),
        start_name=arguments.get("start_name"),
        vectors_input=arguments.get("vectors_input"),
        vectors_output=arguments.get("vectors_output"),
        output_dir=arguments.get("output_dir"),
        write_file=arguments.get("write_file", False),
    )


@_tool("create_nwchem_dft_workflow_input")
def _handle_create_nwchem_dft_workflow_input(arguments: dict[str, Any]) -> dict[str, Any]:
    result = create_nwchem_dft_workflow_input(
        geometry_path=arguments["geometry_file"],
        library_path=basis_library_path(arguments.get("library_path")) if arguments.get("library_path") else basis_library_path(),
        basis_assignments=arguments["basis_assignments"],
        ecp_assignments=arguments.get("ecp_assignments"),
        default_basis=arguments.get("default_basis"),
        default_ecp=arguments.get("default_ecp"),
        basis_block_name=arguments.get("block_name", "ao basis"),
        basis_mode=arguments.get("basis_mode"),
        xc_functional=arguments["xc_functional"],
        task_operations=arguments["task_operations"],
        charge=arguments.get("charge"),
        multiplicity=arguments.get("multiplicity"),
        dft_settings=arguments.get("dft_settings"),
        extra_blocks=arguments.get("extra_blocks"),
        geometry_options=arguments.get("geometry_options"),
        memory=arguments.get("memory"),
        title=arguments.get("title"),
        start_name=arguments.get("start_name"),
        vectors_input=arguments.get("vectors_input"),
        vectors_output=arguments.get("vectors_output"),
        output_dir=arguments.get("output_dir"),
        write_file=arguments.get("write_file", False),
    )
    # Strip large basis/ECP text from response when file was written — saves tokens
    if arguments.get("write_file") and result.get("written_file"):
        result.pop("input_text", None)
        bs = result.get("basis_setup")
        if isinstance(bs, dict):
            bs = dict(bs)
            if isinstance(bs.get("basis_block"), dict):
                bb = dict(bs["basis_block"])
                bb.pop("text", None)
                bs["basis_block"] = bb
            if isinstance(bs.get("ecp_block"), dict):
                eb = dict(bs["ecp_block"])
                eb.pop("text", None)
                bs["ecp_block"] = eb
            result["basis_setup"] = bs
    return result


# ---------------------------------------------------------------------------
# Handlers — case analysis and recovery
# ---------------------------------------------------------------------------



@_tool("analyze_nwchem_case")
def _handle_analyze_nwchem_case(arguments: dict[str, Any]) -> dict[str, Any]:
    compact = arguments.get("detail", "compact") == "compact"
    result = summarize_nwchem_case(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        err_file=arguments.get("err_file"),
        library_path=basis_library_path(arguments.get("library_path")),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        compact=compact,
    )
    result["next_actions"] = _build_next_actions(
        "analyze_case", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
        profile=arguments.get("profile", ""),
    )
    return result


@_tool("suggest_nwchem_recovery")
def _handle_suggest_nwchem_recovery(arguments: dict[str, Any]) -> dict[str, Any]:
    mode = arguments.get("mode", "auto")
    kwargs = dict(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
    )
    if mode == "scf":
        return suggest_nwchem_scf_fix_strategy(**kwargs)
    if mode == "state":
        return suggest_nwchem_state_recovery_strategy(**kwargs)
    # auto: return both
    result: dict[str, Any] = {}
    try:
        result["scf_strategies"] = suggest_nwchem_scf_fix_strategy(**kwargs)
    except Exception as exc:
        result["scf_strategies"] = {"error": str(exc)}
    try:
        result["state_strategies"] = suggest_nwchem_state_recovery_strategy(**kwargs)
    except Exception as exc:
        result["state_strategies"] = {"error": str(exc)}
    return result


# ---------------------------------------------------------------------------
# Handlers — MCSCF
# ---------------------------------------------------------------------------

@_tool("suggest_nwchem_mcscf_active_space")
def _handle_suggest_nwchem_mcscf_active_space(arguments: dict[str, Any]) -> dict[str, Any]:
    return suggest_nwchem_mcscf_active_space(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
    )


@_tool("draft_nwchem_mcscf_input")
def _handle_draft_nwchem_mcscf_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_mcscf_input(
        reference_output_path=arguments["reference_output_file"],
        input_path=arguments["input_file"],
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
        active_space_mode=arguments.get("active_space_mode", "minimal"),
        vectors_input=arguments.get("vectors_input"),
        vectors_output=arguments.get("vectors_output"),
        state_label=arguments.get("state_label"),
        symmetry=arguments.get("symmetry"),
        hessian=arguments.get("hessian", "exact"),
        maxiter=arguments.get("maxiter", 80),
        thresh=arguments.get("thresh", 1.0e-5),
        level=arguments.get("level", 0.6),
        lock_vectors=arguments.get("lock_vectors", True),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        title=arguments.get("title"),
        write_file=arguments.get("write_file", False),
    )


@_tool("draft_nwchem_mcscf_retry_input")
def _handle_draft_nwchem_mcscf_retry_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_mcscf_retry_input(
        output_path=arguments["output_file"],
        input_path=arguments["input_file"],
        expected_metal_elements=arguments.get("expected_metals"),
        active_space_mode=arguments.get("active_space_mode", "auto"),
        vectors_input=arguments.get("vectors_input"),
        vectors_output=arguments.get("vectors_output"),
        state_label=arguments.get("state_label"),
        symmetry=arguments.get("symmetry"),
        hessian=arguments.get("hessian"),
        maxiter=arguments.get("maxiter"),
        thresh=arguments.get("thresh"),
        level=arguments.get("level"),
        lock_vectors=arguments.get("lock_vectors", True),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        title=arguments.get("title"),
        write_file=arguments.get("write_file", False),
    )


# ---------------------------------------------------------------------------
# Handlers — geometry and frequency plausibility
# ---------------------------------------------------------------------------

@_tool("check_nwchem_geometry_plausibility")
def _handle_check_nwchem_geometry_plausibility(arguments: dict[str, Any]) -> dict[str, Any]:
    frame_arg = arguments.get("frame", "best")
    try:
        frame_arg = int(frame_arg)
    except (TypeError, ValueError):
        pass
    result = check_nwchem_geometry_plausibility(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        frame=frame_arg,
    )
    result["next_actions"] = _build_next_actions(
        "geometry_plausibility", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("check_nwchem_freq_plausibility")
def _handle_check_nwchem_freq_plausibility(arguments: dict[str, Any]) -> dict[str, Any]:
    result = check_nwchem_freq_plausibility(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expect_minimum=arguments.get("expect_minimum", True),
    )
    result["next_actions"] = _build_next_actions(
        "freq_plausibility", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("parse_nwchem_freq_progress")
def _handle_parse_nwchem_freq_progress(arguments: dict[str, Any]) -> dict[str, Any]:
    result = parse_freq_progress(arguments["output_file"])
    result["next_actions"] = _build_next_actions(
        "freq_progress", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("prepare_nwchem_freq_restart")
def _handle_prepare_nwchem_freq_restart(arguments: dict[str, Any]) -> dict[str, Any]:
    return prepare_freq_restart(
        input_file=arguments["input_file"],
        output_file=arguments["output_file"],
        profile=arguments.get("profile"),
    )


# ---------------------------------------------------------------------------
# Handlers — imaginary modes
# ---------------------------------------------------------------------------

@_tool("analyze_nwchem_imaginary_modes")
def _handle_analyze_nwchem_imaginary_modes(arguments: dict[str, Any]) -> dict[str, Any]:
    result = analyze_imaginary_modes(
        arguments["output_file"],
        significant_threshold_cm1=arguments.get("significant_threshold_cm1", 20.0),
        top_atoms=arguments.get("top_atoms", 4),
        detail=arguments.get("detail", "compact"),
    )
    result["next_actions"] = _build_next_actions(
        "imaginary_modes", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("displace_nwchem_geometry_along_mode")
def _handle_displace_nwchem_geometry_along_mode(arguments: dict[str, Any]) -> dict[str, Any]:
    return displace_geometry_along_mode(
        arguments["output_file"],
        mode_number=arguments.get("mode_number"),
        amplitude_angstrom=arguments.get("amplitude_angstrom", 0.15),
        significant_threshold_cm1=arguments.get("significant_threshold_cm1", 20.0),
    )


@_tool("draft_nwchem_imaginary_mode_inputs")
def _handle_draft_nwchem_imaginary_mode_inputs(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_imaginary_mode_inputs(
        output_path=arguments["output_file"],
        input_path=arguments["input_file"],
        mode_number=arguments.get("mode_number"),
        amplitude_angstrom=arguments.get("amplitude_angstrom", 0.15),
        significant_threshold_cm1=arguments.get("significant_threshold_cm1", 20.0),
        task_strategy=arguments.get("task_strategy", "auto"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        write_files=arguments.get("write_files", False),
        add_noautosym=arguments.get("noautosym", True),
        enforce_symmetry_c1=arguments.get("symmetry_c1", True),
    )


# ---------------------------------------------------------------------------
# TCE (Tensor Contraction Engine) handlers
# ---------------------------------------------------------------------------

@_tool("parse_nwchem_tce_output")
def _handle_parse_nwchem_tce_output(arguments: dict[str, Any]) -> dict[str, Any]:
    output_file = arguments["output_file"]
    result = parse_tce_output(output_file)
    # Auto-include T1/D1 multireference diagnostics when amplitude files exist
    try:
        amp = parse_tce_amplitudes(arguments["output_file"])
        if amp.get("available"):
            result["multireference_diagnostics"] = {
                "t1_diagnostic": amp.get("t1_diagnostic"),
                "d1_diagnostic": amp.get("d1_diagnostic"),
                "t2_frobenius_norm": amp.get("t2_frobenius_norm"),
                "mr_assessment": amp.get("mr_assessment"),
                "mr_flags": amp.get("mr_flags", []),
                "top_t2_amplitudes": amp.get("top_t2_amplitudes", []),
                "amplitude_files": amp.get("amplitude_files", []),
                "note": (
                    "T1 > 0.02: moderate MR character; > 0.05: strong MR — CCSD unreliable. "
                    "D1 > 0.05: significant orbital relaxation."
                ),
            }
        else:
            result["multireference_diagnostics"] = {
                "available": False,
                "reason": amp.get("reason", "amplitude files not found"),
                "note": "Rerun with 'set tce:save_t T T' to enable T1/D1 diagnostics.",
            }
    except Exception as exc:
        result["multireference_diagnostics"] = {"available": False, "error": str(exc)}
    result["next_actions"] = _build_next_actions(
        "tce_output", result,
        output_file=output_file,
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("parse_nwchem_tce_amplitudes")
def _handle_parse_nwchem_tce_amplitudes(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_tce_amplitudes(arguments["output_file"])


@_tool("draft_nwchem_tce_input")
def _handle_draft_nwchem_tce_input(arguments: dict[str, Any]) -> dict[str, Any]:
    # swap_pairs comes in as list of [i, j] arrays from JSON
    raw_swaps = arguments.get("swap_pairs")
    swap_pairs = [tuple(pair) for pair in raw_swaps] if raw_swaps else None
    result = draft_nwchem_tce_input(
        scf_output_file=arguments["scf_output_file"],
        input_file=arguments["input_file"],
        method=arguments.get("method", "mp2"),
        freeze_count=arguments.get("freeze_count"),
        swap_pairs=swap_pairs,
        movecs_file=arguments.get("movecs_file"),
        ecp_core_electrons=arguments.get("ecp_core_electrons"),
        basis_library=arguments.get("basis_library"),
        start_name=arguments.get("start_name"),
        memory=arguments.get("memory"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        write_file=arguments.get("write_file", False),
    )
    return result


@_tool("draft_nwchem_tce_restart_input")
def _handle_draft_nwchem_tce_restart_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_tce_restart_input(
        tce_output_file=arguments["tce_output_file"],
        tce_input_file=arguments.get("tce_input_file"),
        max_iterations=arguments.get("max_iterations", 200),
        thresh=arguments.get("thresh", 1e-5),
        copy_amplitudes=arguments.get("copy_amplitudes", True),
        output_dir=arguments.get("output_dir"),
        write_file=arguments.get("write_file", False),
    )


@_tool("validate_nwchem_tce_setup")
def _handle_validate_nwchem_tce_setup(arguments: dict[str, Any]) -> dict[str, Any]:
    result = validate_nwchem_tce_setup(
        tce_input_path=arguments["tce_input_file"],
        scf_output_path=arguments.get("scf_output_file"),
    )
    result["next_actions"] = _build_next_actions(
        "tce_validation", result,
        output_file=arguments.get("scf_output_file", ""),
        input_file=arguments["tce_input_file"],
    )
    return result


@_tool("parse_nwchem_movecs")
def _handle_parse_nwchem_movecs(arguments: dict[str, Any]) -> dict[str, Any]:
    movecs_file = arguments["movecs_file"]
    result = parse_nwchem_movecs(movecs_file)
    # The natural sibling .out file usually shares the same stem, e.g.
    # water.movecs ↔ water.out — emit that as the next-action target.
    from pathlib import Path
    output_file = str(Path(movecs_file).with_suffix(".out"))
    result["next_actions"] = _build_next_actions(
        "movecs", result,
        output_file=output_file,
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("parse_nwchem_hessian")
def _handle_parse_nwchem_hessian(arguments: dict[str, Any]) -> dict[str, Any]:
    from chemtools.programs.nwchem.binary.hessian import parse_nwchem_hessian
    return parse_nwchem_hessian(
        arguments["hessian_file"],
        return_matrix=arguments.get("return_matrix", True),
    )


@_tool("compute_nwchem_harmonic_frequencies")
def _handle_compute_nwchem_harmonic_frequencies(arguments: dict[str, Any]) -> dict[str, Any]:
    from chemtools.programs.nwchem.binary.hessian import compute_nwchem_harmonic_frequencies
    return compute_nwchem_harmonic_frequencies(
        hessian_path=arguments["hessian_file"],
        elements=arguments["elements"],
        masses_amu=arguments.get("masses_amu"),
    )


@_tool("swap_nwchem_movecs")
def _handle_swap_nwchem_movecs(arguments: dict[str, Any]) -> dict[str, Any]:
    return swap_nwchem_movecs(
        movecs_path=arguments["movecs_file"],
        i=arguments["i"],
        j=arguments["j"],
        output_path=arguments.get("output_file"),
    )


@_tool("suggest_nwchem_tce_freeze")
def _handle_suggest_nwchem_tce_freeze(arguments: dict[str, Any]) -> dict[str, Any]:
    return suggest_tce_freeze_count(
        elements=arguments["elements"],
        ecp_core_electrons=arguments.get("ecp_core_electrons"),
    )


# ---------------------------------------------------------------------------
# Handlers — parallel job monitoring, session log, input versioning
# ---------------------------------------------------------------------------

@_tool("get_nwchem_workflow_state")
def _handle_get_nwchem_workflow_state(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_nwchem_workflow_state(
        input_file=arguments.get("input_file"),
        output_file=arguments["output_file"],
        profile=arguments.get("profile", ""),
        error_file=arguments.get("error_file"),
    )


@_tool("plan_nwchem_calculation")
def _handle_plan_nwchem_calculation(arguments: dict[str, Any]) -> dict[str, Any]:
    return plan_calculation(
        input_file=arguments["input_file"],
        protocol=arguments["protocol"],
        profile=arguments.get("profile", ""),
        output_dir=arguments.get("output_dir"),
        overrides=arguments.get("overrides"),
    )


@_tool("list_nwchem_protocols")
def _handle_list_nwchem_protocols(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"protocols": list_protocols()}


@_tool("create_nwchem_input_variant")
def _handle_create_nwchem_input_variant(arguments: dict[str, Any]) -> dict[str, Any]:
    result = create_nwchem_input_variant(
        source_input=arguments["source_input"],
        changes=arguments["changes"],
        reason=arguments.get("reason", ""),
        output_path=arguments.get("output_path"),
    )
    result.pop("input_text", None)
    return result


# ---------------------------------------------------------------------------
# Handlers — eval + smart input creation (Phase 6)
# ---------------------------------------------------------------------------

@_tool("evaluate_nwchem_case")
def _handle_evaluate_case(arguments: dict[str, Any]) -> dict[str, Any]:
    return evaluate_case(arguments["case_path"])


@_tool("evaluate_nwchem_cases")
def _handle_evaluate_cases(arguments: dict[str, Any]) -> dict[str, Any]:
    return evaluate_cases(arguments["path"])


@_tool("create_nwchem_dft_input_from_request")
def _handle_create_nwchem_dft_input_from_request(arguments: dict[str, Any]) -> dict[str, Any]:
    result = create_nwchem_dft_input_from_request(
        formula=arguments.get("formula"),
        geometry_path=arguments.get("geometry_file"),
        library_path=basis_library_path(arguments.get("library_path")),
        basis_assignments=arguments.get("basis_assignments"),
        ecp_assignments=arguments.get("ecp_assignments"),
        default_basis=arguments.get("default_basis"),
        default_ecp=arguments.get("default_ecp"),
        xc_functional=arguments.get("xc_functional"),
        task_operations=arguments.get("task_operations"),
        charge=arguments.get("charge"),
        multiplicity=arguments.get("multiplicity"),
        dft_settings=arguments.get("dft_settings"),
        extra_blocks=arguments.get("extra_blocks"),
        geometry_options=arguments.get("geometry_options"),
        memory=arguments.get("memory"),
        title=arguments.get("title"),
        start_name=arguments.get("start_name"),
        output_dir=arguments.get("output_dir"),
        write_file=arguments.get("write_file", False),
    )
    # Don't send full input text through MCP — it can be huge with explicit basis blocks
    if result.get("input_text") and len(result["input_text"]) > 5000:
        result["input_text_truncated"] = result["input_text"][:2000] + "\n... (truncated, see written_file)"
        del result["input_text"]
    return result


# ---------------------------------------------------------------------------
# Handlers — gap-fill tools (Phase 5)
# ---------------------------------------------------------------------------

@_tool("check_nwchem_spin_charge_state")
def _handle_check_spin_charge_state(arguments: dict[str, Any]) -> dict[str, Any]:
    result = check_spin_charge_state(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
    )
    result["next_actions"] = _build_next_actions(
        "spin_charge_state", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("suggest_nwchem_multiplicity_scan")
def _handle_suggest_multiplicity_scan(arguments: dict[str, Any]) -> dict[str, Any]:
    elements = arguments.get("elements")
    charge = arguments.get("charge")
    multiplicity = arguments.get("multiplicity")
    input_file = arguments.get("input_file")
    if input_file and (elements is None or charge is None or multiplicity is None):
        summary = inspect_input(input_file)
        if elements is None:
            # Full atom multiset, not unique elements — the electron-count parity
            # that fixes the scan's multiplicities depends on every atom.
            elements = summary.get("all_elements") or summary.get("elements")
        if charge is None:
            charge = summary.get("charge")
        if multiplicity is None:
            multiplicity = summary.get("multiplicity")
    if not elements:
        return {
            "error": "Provide input_file (to read elements/charge/multiplicity) or an explicit elements list.",
        }
    result = recommend_multiplicity_scan(
        elements=elements,
        charge=charge or 0,
        current_multiplicity=multiplicity,
        metal_oxidation_states=arguments.get("metal_oxidation_states"),
    )
    if result["scan_warranted"] and input_file:
        result["next_actions"] = [{
            "priority": 1,
            "tool": "generate_nwchem_input_batch",
            "params": {
                "template_input": input_file,
                "vary": {"mult": result["recommended_multiplicities"]},
                "output_dir": arguments.get("output_dir") or str(Path(input_file).parent),
            },
            "reason": "Generate one input per candidate multiplicity at the same geometry "
                      "and basis; run them, then take the lowest total energy.",
            "confidence": 0.9,
        }]
    return result


@_tool("inspect_nwchem_geometry")
def _handle_inspect_nwchem_geometry(arguments: dict[str, Any]) -> dict[str, Any]:
    return inspect_nwchem_geometry(
        input_path=arguments["input_file"],
    )


@_tool("parse_nwchem_tasks")
def _handle_parse_nwchem_tasks(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_tasks(arguments["output_file"])


@_tool("parse_nwchem_trajectory")
def _handle_parse_nwchem_trajectory(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_trajectory(
        path=arguments["output_file"],
        include_positions=arguments.get("include_positions", False),
    )


@_tool("review_nwchem_input_request")
def _handle_review_nwchem_input_request(arguments: dict[str, Any]) -> dict[str, Any]:
    return review_nwchem_input_request(
        formula=arguments.get("formula"),
        geometry_path=arguments.get("geometry_file"),
        library_path=basis_library_path(arguments.get("library_path")),
        basis_assignments=arguments.get("basis_assignments"),
        ecp_assignments=arguments.get("ecp_assignments"),
        default_basis=arguments.get("default_basis"),
        default_ecp=arguments.get("default_ecp"),
        module=arguments.get("module", "dft"),
        task_operations=arguments.get("task_operations"),
        functional=arguments.get("functional"),
        charge=arguments.get("charge"),
        multiplicity=arguments.get("multiplicity"),
    )


@_tool("review_nwchem_progress")
def _handle_review_nwchem_progress(arguments: dict[str, Any]) -> dict[str, Any]:
    result = review_nwchem_progress(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        error_path=arguments.get("error_file"),
        process_id=arguments.get("process_id"),
        profile=arguments.get("profile"),
        job_id=arguments.get("job_id"),
    )
    result["next_actions"] = _build_next_actions(
        "review_progress", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("summarize_nwchem_output")
def _handle_summarize_nwchem_output(arguments: dict[str, Any]) -> dict[str, Any]:
    result = summarize_output(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
        detail_level=arguments.get("detail", "summary"),
    )
    result["next_actions"] = _build_next_actions(
        "summarize_output", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("summarize_nwchem_outputs")
def _handle_summarize_nwchem_outputs(arguments: dict[str, Any]) -> dict[str, Any]:
    target = arguments.get("paths") or arguments.get("path")
    if not target:
        return {"error": "Provide 'path' (a directory, glob, or file) or 'paths' (a list)."}
    return summarize_nwchem_outputs(
        paths=target,
        pattern=arguments.get("pattern", "*.out"),
        recursive=arguments.get("recursive", False),
        limit=arguments.get("limit"),
    )


# ---------------------------------------------------------------------------
# Handlers — run registry, campaigns, workflows, batch generation
# ---------------------------------------------------------------------------

@_tool("register_nwchem_run", needs="registry")
def _handle_register_run(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy NWChem-tagged registry — pre-fills program='nwchem'.

    Equivalent to ``register_run(..., program='nwchem')``. Kept for one
    release so older agents/tests don't break.
    """
    return register_run(
        program="nwchem",
        job_name=arguments["job_name"],
        input_file=arguments.get("input_file"),
        output_file=arguments.get("output_file"),
        profile=arguments.get("profile"),
        method=arguments.get("method"),
        functional=arguments.get("functional"),
        basis=arguments.get("basis"),
        n_atoms=arguments.get("n_atoms"),
        elements=arguments.get("elements"),
        charge=arguments.get("charge"),
        multiplicity=arguments.get("multiplicity"),
        mpi_ranks=arguments.get("mpi_ranks"),
        campaign_id=arguments.get("campaign_id"),
        workflow_id=arguments.get("workflow_id"),
        workflow_step_id=arguments.get("workflow_step_id"),
        parent_run_id=arguments.get("parent_run_id"),
        tags=arguments.get("tags"),
    )


# --- Registry: per-run status + lookup ---
def _do_update_run_status(arguments: dict[str, Any]) -> dict[str, Any]:
    return update_run_status(
        run_id=arguments["run_id"],
        status=arguments["status"],
        energy_hartree=arguments.get("energy_hartree"),
        h_hartree=arguments.get("h_hartree"),
        g_hartree=arguments.get("g_hartree"),
        imaginary_modes=arguments.get("imaginary_modes"),
        walltime_used_sec=arguments.get("walltime_used_sec"),
        sec_per_gradient=arguments.get("sec_per_gradient"),
        output_file=arguments.get("output_file"),
    )

@_tool("update_nwchem_run_status", needs="registry")
def _handle_update_run_status(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy alias for update_run_status. Run-status updates aren't
    program-specific; the run_id selects the row to modify."""
    return _do_update_run_status(arguments)


def _do_list_runs(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"runs": list_runs(
        campaign_id=arguments.get("campaign_id"),
        workflow_id=arguments.get("workflow_id"),
        status=arguments.get("status"),
        method=arguments.get("method"),
        program=arguments.get("program"),
        limit=arguments.get("limit", 50),
    )}

@_tool("list_nwchem_runs", needs="registry")
def _handle_list_runs(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy: list runs. Pre-fills program='nwchem' if no program filter
    is given (preserves the historical NWChem-only return for callers
    that haven't migrated)."""
    args = dict(arguments)
    if args.get("program") is None:
        args["program"] = "nwchem"
    return _do_list_runs(args)


def _do_get_run_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_run_summary(
        run_id=arguments.get("run_id"),
        job_name=arguments.get("job_name"),
    )

@_tool("get_nwchem_run_summary", needs="registry")
def _handle_get_run_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy alias — fetches by run_id or job_name; program-agnostic."""
    return _do_get_run_summary(arguments)


# --- Registry: campaigns ---
def _do_create_campaign(arguments: dict[str, Any]) -> dict[str, Any]:
    return create_campaign(
        name=arguments["name"],
        description=arguments.get("description"),
        tags=arguments.get("tags"),
    )

@_tool("create_nwchem_campaign", needs="registry")
def _handle_create_campaign(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy alias. Campaigns are cross-program by design — no program tag
    on campaigns themselves; runs inside the campaign carry their own."""
    return _do_create_campaign(arguments)


def _do_get_campaign_status(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_campaign_status(
        campaign_id=arguments.get("campaign_id"),
        name=arguments.get("name"),
    )

@_tool("get_nwchem_campaign_status", needs="registry")
def _handle_get_campaign_status(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy alias."""
    return _do_get_campaign_status(arguments)


def _do_get_campaign_energies(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_campaign_energies(
        campaign_id=arguments.get("campaign_id"),
        name=arguments.get("name"),
    )

@_tool("get_nwchem_campaign_energies", needs="registry")
def _handle_get_campaign_energies(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy alias. Each returned row now includes the run's program tag."""
    return _do_get_campaign_energies(arguments)


# --- Registry: workflows ---
def _do_create_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    return create_workflow(
        name=arguments["name"],
        steps=arguments["steps"],
        protocol=arguments.get("protocol"),
        campaign_id=arguments.get("campaign_id"),
    )

@_tool("create_nwchem_workflow", needs="registry")
def _handle_create_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy alias. Workflows themselves are cross-program; the per-step
    `program` field inside the steps_json controls each run."""
    return _do_create_workflow(arguments)


def _do_advance_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    return advance_workflow(workflow_id=arguments["workflow_id"])

@_tool("advance_nwchem_workflow", needs="registry")
def _handle_advance_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy alias."""
    return _do_advance_workflow(arguments)


@_tool("generate_nwchem_input_batch", needs="executable_or_scheduler")
def _handle_generate_input_batch(arguments: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        template_input=arguments["template_input"],
        vary=arguments["vary"],
        output_dir=arguments["output_dir"],
    )
    if arguments.get("naming_pattern"):
        kwargs["naming_pattern"] = arguments["naming_pattern"]
    if arguments.get("campaign_id") is not None:
        kwargs["campaign_id"] = arguments["campaign_id"]
    return generate_input_batch(**kwargs)


@_tool("check_nwchem_memory_fit", needs="executable_or_scheduler")
def _handle_check_memory_fit(arguments: dict[str, Any]) -> dict[str, Any]:
    profile_resources = None
    if arguments.get("profile"):
        from chemtools.core.runner import load_runner_profiles, _resolve_profile
        profiles_path = arguments.get("profiles_path")
        loaded = load_runner_profiles(profiles_path)
        resolved = _resolve_profile(loaded, arguments["profile"])
        profile_resources = resolved.get("resources", {})
        # Merge resource_overrides if present
        if arguments.get("resource_overrides"):
            profile_resources = {**profile_resources, **arguments["resource_overrides"]}
    kwargs: dict[str, Any] = {
        "input_file": arguments["input_file"],
        "profile_resources": profile_resources,
    }
    if "nodes" in arguments:
        kwargs["nodes"] = arguments["nodes"]
    if "mpi_ranks" in arguments:
        kwargs["mpi_ranks"] = arguments["mpi_ranks"]
    if "node_memory_mb" in arguments:
        kwargs["node_memory_mb"] = arguments["node_memory_mb"]
    return check_memory_fit(**kwargs)


@_tool("estimate_nwchem_freq_walltime", needs="executable_or_scheduler")
def _handle_estimate_freq_walltime(arguments: dict[str, Any]) -> dict[str, Any]:
    return estimate_freq_walltime(
        n_atoms=arguments["n_atoms"],
        seconds_per_displacement=arguments.get("seconds_per_displacement"),
        n_displacements=arguments.get("n_displacements"),
        mpi_ranks=arguments.get("mpi_ranks", 1),
        nodes=arguments.get("nodes", 1),
        max_walltime_hours=arguments.get("max_walltime_hours", 48.0),
    )


@_tool("suggest_nwchem_resources", needs="executable_or_scheduler")
def _handle_suggest_hpc_resources(arguments: dict[str, Any]) -> dict[str, Any]:
    return suggest_hpc_resources(
        input_file=arguments["input_file"],
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path"),
    )


@_tool("detect_nwchem_hpc_accounts", needs="scheduler")
def _handle_detect_hpc_accounts(arguments: dict[str, Any]) -> dict[str, Any]:
    return detect_hpc_accounts(
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path"),
    )


@_tool("suggest_nwchem_partition", needs="scheduler")
def _handle_suggest_partition(arguments: dict[str, Any]) -> dict[str, Any]:
    return suggest_partition(
        input_file=arguments["input_file"],
        profiles_path=arguments.get("profiles_path"),
        check_queue=arguments.get("check_queue", True),
    )


# ---------------------------------------------------------------------------
# Handlers — NWChem documentation (bundled)
# ---------------------------------------------------------------------------

@_tool("list_nwchem_docs")
def _handle_list_nwchem_docs(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"files": docs_list_docs()}


@_tool("search_nwchem_docs")
def _handle_search_nwchem_docs(arguments: dict[str, Any]) -> dict[str, Any]:
    return docs_search_docs(
        arguments["query"],
        max_results=int(arguments.get("max_results", 8)),
        context_lines=int(arguments.get("context_lines", 2)),
    )


@_tool("lookup_nwchem_block_syntax")
def _handle_lookup_nwchem_block_syntax(arguments: dict[str, Any]) -> dict[str, Any]:
    return docs_lookup_block_syntax(
        arguments["block_name"],
        max_results=int(arguments.get("max_results", 6)),
    )


@_tool("find_nwchem_examples")
def _handle_find_nwchem_examples(arguments: dict[str, Any]) -> dict[str, Any]:
    return docs_find_examples(
        arguments["topic"],
        max_results=int(arguments.get("max_results", 6)),
    )


@_tool("read_nwchem_doc_excerpt")
def _handle_read_nwchem_doc_excerpt(arguments: dict[str, Any]) -> dict[str, Any]:
    return docs_read_doc_excerpt(
        arguments["doc_name"],
        start_line=arguments.get("start_line"),
        end_line=arguments.get("end_line"),
        query=arguments.get("query"),
        context_lines=int(arguments.get("context_lines", 8)),
    )


@_tool("get_nwchem_topic_guide")
def _handle_get_nwchem_topic_guide(arguments: dict[str, Any]) -> dict[str, Any]:
    return docs_get_topic_guide(arguments["topic"])


# ---------------------------------------------------------------------------
# Handlers — NWChem community forum search
# ---------------------------------------------------------------------------

@_tool("search_nwchem_forum")
def _handle_search_nwchem_forum(arguments: dict[str, Any]) -> dict[str, Any]:
    return forum_search(
        arguments["query"],
        max_results=int(arguments.get("max_results", 5)),
        fetch_content=arguments.get("fetch_content", True),
        subforums=arguments.get("subforums"),
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
    raise AttributeError(f"module 'chemtools.mcp.tools.nwchem' has no attribute {name!r}")


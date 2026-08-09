"""Generic (cross-program) MCP tool handlers + tool definitions.

These tools are program-agnostic. Some auto-detect the program via
``chemtools.core.registry.resolve(path=output_file)`` and route to the
appropriate per-program plugin; others (basis advisors, session log,
reaction-energy calc, registry / campaign / workflow tools) operate on
arguments rather than a specific output file.

All ``@_tool`` decorators here pass ``program="generic"`` so the
``--programs`` filter never hides them — generics are always visible.

Extracted from ``chemtools/mcp/tools/nwchem.py`` during the
multi-program MCP split.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from chemtools.core.basis_advisor import suggest_basis_set
from chemtools.core.cube import (
    compare_cube_densities,
    compare_cube_orbitals,
    parse_cube_file as parse_cube,
    summarize_cube_file as summarize_cube,
)
from chemtools.core.cube_subspace import compare_cube_orbital_subspaces
from chemtools.persistence.runs import (
    get_run_summary,
    list_runs,
    register_run,
    update_run_status,
)
from chemtools.application.run_registry import (
    advance_workflow,
    create_campaign,
    create_workflow,
    get_campaign_energies,
    get_campaign_status,
)
from chemtools.application.nwchem_execution import render_nwchem_job_script
from chemtools.core.session import (
    append_session_log,
    init_session_log,
    next_versioned_path,
)
from chemtools.programs.nwchem.input.basis import basis_library_summary
from chemtools.programs.nwchem.input.geometry import draft_initial_geometry
from chemtools.programs.nwchem.output import compute_reaction_energy
from chemtools.programs.nwchem.runner import watch_multiple_nwchem_runs
from chemtools.programs.nwchem.strategy.input_advisors import (
    suggest_relativistic_correction,
    suggest_spin_state,
)
from chemtools.programs.nwchem.strategy.resources import (
    suggest_memory,
    suggest_resources,
)
from chemtools.programs.nwchem.strategy.workflow_state import preflight_check

from chemtools.mcp.decorator import (
    _TOOL_CAPABILITIES,
    _TOOL_PROGRAMS,
    _tool as _raw_tool,
)
from chemtools.mcp import modes as _modes
from chemtools.core.program import (
    ProgramBackend,
    ProgramCapability,
    UnsupportedCapabilityError,
)
from chemtools.application.run_inspection import inspect_run_geometry
from chemtools.application.input_review import InputReviewError
from chemtools.application.recovery_planning import (
    ApplyRecoveryResolutionError,
    resolve_apply_recovery_program,
)


def _tool(name: str, *, needs: str = "none"):
    """Generic @_tool wrapper — every handler in this module is
    program-tagged 'generic' so it's always visible."""
    return _raw_tool(name, needs=needs, program="generic")


def _require_capability_or_error(
    plugin: Any,
    capability: ProgramCapability,
) -> dict[str, Any] | None:
    # Legacy third-party Program implementations remain callable until the
    # registry compatibility union is removed.
    if not isinstance(plugin, ProgramBackend):
        return None
    try:
        plugin.require(capability)
    except UnsupportedCapabilityError as error:
        return {
            "error": "unsupported_capability",
            "program": error.program,
            "capability": error.capability.value,
            "available_capabilities": list(error.available_capabilities),
        }
    return None


def _program_detector_error_payload(error: Any) -> dict[str, Any]:
    return {
        "error": "program_detector_error",
        "message": str(error),
        "candidates": list(error.candidates),
        "detector_failures": [
            {
                "program": failure.program,
                "error_type": failure.error_type,
                "message": failure.message,
            }
            for failure in error.failures
        ],
    }


def _program_source_error_payload(error: Any) -> dict[str, Any]:
    return {
        "error": "program_source_error",
        "message": str(error),
        "path": error.path,
        "source_failure": {
            "error_type": error.failure.error_type,
            "message": error.failure.message,
            "errno": error.failure.errno,
        },
    }


# ---------------------------------------------------------------------------
# Server introspection
# ---------------------------------------------------------------------------

@_tool("get_server_mode")
def _handle_get_server_mode(arguments: dict[str, Any]) -> dict[str, Any]:
    from chemtools.mcp.decorator import get_server_state
    from chemtools.mcp.dispatch import tool_definitions
    state = get_server_state()
    summary = _modes.summarize_mode(
        state.mode, _TOOL_CAPABILITIES, tool_definitions(),
        programs=state.programs, program_tags=_TOOL_PROGRAMS, toolset=state.toolset,
    )
    summary["programs"] = sorted(state.programs) if state.programs else None
    summary["toolset"] = sorted(state.toolset) if state.toolset else None
    summary["execution_enabled"] = (
        state.execution_service.enable_execution
    )
    summary["default_target"] = state.execution_service.default_target
    summary["targets"] = [
        {
            "name": target.name,
            "executor": target.executor,
            "programs": sorted(target.programs),
        }
        for target in state.execution_service.configured_targets.values()
    ]
    return summary


# ---------------------------------------------------------------------------
# summarize_run — flagship plugin-dispatch tool
# ---------------------------------------------------------------------------

@_tool("summarize_run")
def _handle_summarize_run(arguments: dict[str, Any]) -> dict[str, Any]:
    """Flagship plugin-dispatch tool — combines parser + strategist for any program."""
    output_file = arguments["output_file"]
    plugin, err = _resolve_plugin_or_error(
        arguments, capability=ProgramCapability.OUTPUT_PARSE
    )
    if err is not None:
        return err

    parsed = plugin.parser.parse_output(output_file)
    diagnosis: dict[str, Any] | None = None
    if isinstance(plugin, ProgramBackend):
        strategist = (
            plugin.require(ProgramCapability.DIAGNOSIS_RUN).diagnostics
            if plugin.supports(ProgramCapability.DIAGNOSIS_RUN)
            else None
        )
    else:
        strategist = plugin.strategist
    if strategist is not None:
        try:
            diagnosis = strategist.diagnose(parsed)
        except NotImplementedError:
            diagnosis = None
    return {
        "program": plugin.name,
        "parsed": parsed,
        "diagnosis": diagnosis,
    }


# ---------------------------------------------------------------------------
# Generic parser and geometry dispatchers
# ---------------------------------------------------------------------------

def _resolve_plugin_or_error(
    arguments: dict[str, Any],
    *,
    capability: ProgramCapability | None = None,
):
    """Shared dispatch logic: resolve plugin from `program` override or auto-
    detect from `output_file`. Returns (plugin, None) on success or
    (None, error_dict) on failure."""
    from chemtools.core import registry as _registry
    output_file = arguments["output_file"]
    program = arguments.get("program")
    try:
        plugin = _registry.resolve(program=program, path=output_file)
        if capability is None:
            return plugin, None
        capability_error = _require_capability_or_error(plugin, capability)
        if capability_error is not None:
            return None, capability_error
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
        return None, _program_detector_error_payload(e)
    except _registry.ProgramDetectionSourceError as e:
        return None, _program_source_error_payload(e)
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


@_tool("parse_output")
def _handle_parse_output_generic(arguments: dict[str, Any]) -> dict[str, Any]:
    plugin, err = _resolve_plugin_or_error(
        arguments, capability=ProgramCapability.OUTPUT_PARSE
    )
    if err is not None:
        return err
    parsed = plugin.parser.parse_output(arguments["output_file"])
    # ParsedRun is a TypedDict; cast to plain dict for the JSON return.
    return dict(parsed)


@_tool("extract_geometry")
def _handle_extract_geometry(arguments: dict[str, Any]) -> dict[str, Any]:
    plugin, err = _resolve_plugin_or_error(
        arguments, capability=ProgramCapability.OUTPUT_GEOMETRY
    )
    if err is not None:
        return err
    geom = plugin.parser.get_geometry(
        arguments["output_file"], task_index=arguments.get("task_index"),
    )
    return {"program": plugin.name, "atoms": geom} if isinstance(geom, list) else {"program": plugin.name, **(geom or {})}


@_tool("parse_thermochem")
def _handle_parse_thermochem_generic(arguments: dict[str, Any]) -> dict[str, Any]:
    plugin, err = _resolve_plugin_or_error(
        arguments, capability=ProgramCapability.OUTPUT_THERMOCHEMISTRY
    )
    if err is not None:
        return err
    payload = plugin.parser.get_thermochem(
        arguments["output_file"], task_index=arguments.get("task_index"),
    )
    return {"program": plugin.name, **(payload or {})}


@_tool("parse_frequencies")
def _handle_parse_frequencies_generic(arguments: dict[str, Any]) -> dict[str, Any]:
    plugin, err = _resolve_plugin_or_error(
        arguments, capability=ProgramCapability.OUTPUT_FREQUENCIES
    )
    if err is not None:
        return err
    payload = plugin.parser.get_frequency(
        arguments["output_file"], task_index=arguments.get("task_index"),
    )
    return {"program": plugin.name, **(payload or {})}


@_tool("parse_trajectory")
def _handle_parse_trajectory_generic(arguments: dict[str, Any]) -> dict[str, Any]:
    plugin, err = _resolve_plugin_or_error(
        arguments, capability=ProgramCapability.OUTPUT_TRAJECTORY
    )
    if err is not None:
        return err
    payload = plugin.parser.get_trajectory(
        arguments["output_file"], task_index=arguments.get("task_index"),
    )
    return {"program": plugin.name, **(payload or {})}


@_tool("inspect_geometry")
def _handle_inspect_geometry_generic(arguments: dict[str, Any]) -> dict[str, Any]:
    plugin, err = _resolve_plugin_or_error(
        arguments, capability=ProgramCapability.OUTPUT_GEOMETRY
    )
    if err is not None:
        return err
    return inspect_run_geometry(
        plugin,
        arguments["output_file"],
        max_bond_length=float(arguments.get("max_bond_length", 2.5)),
        min_safe_distance=float(arguments.get("min_safe_distance", 0.6)),
        covalent_tolerance=float(arguments.get("covalent_tolerance", 1.20)),
        measurements=arguments.get("measurements"),
    )


# ---------------------------------------------------------------------------
# Generic case-analysis and recovery dispatchers
# ---------------------------------------------------------------------------

def _dispatch_to_per_program_tool(
    arguments: dict[str, Any],
    handler_by_program: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
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


@_tool("summarize_output")
def _handle_summarize_output_generic(arguments: dict[str, Any]) -> dict[str, Any]:
    """Auto-detecting summarize_output. Routes to summarize_nwchem_output
    or summarize_molcas_output. Returns the program-specific shape tagged
    with `program`."""
    from chemtools.mcp.tools.molcas import _handle_summarize_molcas_output
    from chemtools.mcp.tools.nwchem_parse import (
        _handle_summarize_nwchem_output,
    )
    return _dispatch_to_per_program_tool(
        {**arguments, "_tool_label": "summarize_output"},
        {
            "nwchem": _handle_summarize_nwchem_output,
            "molcas": _handle_summarize_molcas_output,
        },
    )


@_tool("analyze_case")
def _handle_analyze_case_generic(arguments: dict[str, Any]) -> dict[str, Any]:
    """Auto-detecting analyze_case. Routes to analyze_nwchem_case or
    analyze_molcas_case. Returns the program-specific shape tagged with
    `program`."""
    from chemtools.mcp.tools.molcas import _handle_analyze_molcas_case
    from chemtools.mcp.tools.nwchem_analysis import _handle_analyze_nwchem_case
    return _dispatch_to_per_program_tool(
        {**arguments, "_tool_label": "analyze_case"},
        {
            "nwchem": _handle_analyze_nwchem_case,
            "molcas": _handle_analyze_molcas_case,
        },
    )


@_tool("suggest_recovery")
def _handle_suggest_recovery_generic(arguments: dict[str, Any]) -> dict[str, Any]:
    """Auto-detecting recovery suggester. Routes to suggest_nwchem_recovery
    or suggest_molcas_recovery."""
    from chemtools.mcp.tools.molcas import _handle_suggest_molcas_recovery
    from chemtools.mcp.tools.nwchem_analysis import (
        _handle_suggest_nwchem_recovery,
    )
    return _dispatch_to_per_program_tool(
        {**arguments, "_tool_label": "suggest_recovery"},
        {
            "nwchem": _handle_suggest_nwchem_recovery,
            "molcas": _handle_suggest_molcas_recovery,
        },
    )


@_tool("apply_recovery")
def _handle_apply_recovery_generic(arguments: dict[str, Any]) -> dict[str, Any]:
    """Auto-detecting apply_recovery. Currently only Molcas has a
    mechanical-fix patcher. NWChem returns a not_implemented envelope."""
    from chemtools.mcp.tools.molcas import _handle_apply_molcas_recovery
    from chemtools.core import registry as _registry

    selected_program = arguments.get("program")
    if arguments.get("output_file"):
        backend, resolution_error = _resolve_plugin_or_error(arguments)
        if resolution_error is not None:
            if resolution_error["error"] != "program_detection_failed":
                return resolution_error
        else:
            selected_program = backend.name
    try:
        program = resolve_apply_recovery_program(
            (
                backend
                for backend in _registry.iter_programs()
                if isinstance(backend, ProgramBackend)
            ),
            input_file=arguments.get("input_file"),
            selected_program=selected_program,
        )
    except (ApplyRecoveryResolutionError, InputReviewError) as error:
        return error.as_dict()

    def _nwchem_not_implemented(_args: dict[str, Any]) -> dict[str, Any]:
        return {
            "verdict": "not_implemented_for_program",
            "message": (
                "apply_recovery does not currently have an NWChem patcher. "
                "Use suggest_recovery to get the strategy bundle and apply "
                "the recommended changes manually (or via a draft_nwchem_* "
                "tool with the suggested overrides)."
            ),
        }
    dispatch = {
        "nwchem": _nwchem_not_implemented,
        "molcas": _handle_apply_molcas_recovery,
    }
    handler = dispatch.get(program)
    if handler is None:
        return {
            "error": "no_handler_for_program",
            "message": f"No apply_recovery handler for program {program!r}.",
        }
    result = handler(arguments)
    return {"program": program, **(result if isinstance(result, dict) else {"result": result})}


# ---------------------------------------------------------------------------
# Geometry / draft helpers
# ---------------------------------------------------------------------------

@_tool("draft_initial_geometry")
def _handle_draft_initial_geometry(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_initial_geometry(
        atoms=arguments["atoms"],
        output_path=arguments["output_path"],
        comment=arguments.get("comment"),
        central_atom=arguments.get("central_atom"),
    )


# ---------------------------------------------------------------------------
# Reaction energy
# ---------------------------------------------------------------------------

@_tool("compute_reaction_energy")
def _handle_compute_reaction_energy(arguments: dict[str, Any]) -> dict[str, Any]:
    return compute_reaction_energy(
        species=arguments["species"],
        reactants=arguments["reactants"],
        products=arguments["products"],
        method=arguments.get("method"),
        include_thermochem=arguments.get("include_thermochem", False),
    )


# ---------------------------------------------------------------------------
# Strategy / suggestions
# ---------------------------------------------------------------------------

@_tool("suggest_relativistic_correction")
def _handle_suggest_relativistic_correction(arguments: dict[str, Any]) -> dict[str, Any]:
    return suggest_relativistic_correction(
        elements=arguments["elements"],
        basis_assignments=arguments.get("basis_assignments"),
        ecp_assignments=arguments.get("ecp_assignments"),
        purpose=arguments.get("purpose", "dft"),
    )


@_tool("suggest_spin_state")
def _handle_suggest_spin_state(arguments: dict[str, Any]) -> dict[str, Any]:
    return suggest_spin_state(
        elements=arguments["elements"],
        charge=arguments.get("charge", 0),
        metal_oxidation_states=arguments.get("metal_oxidation_states"),
    )


@_tool("suggest_basis_set")
def _handle_suggest_basis_set(arguments: dict[str, Any]) -> dict[str, Any]:
    return suggest_basis_set(
        elements=arguments["elements"],
        purpose=arguments.get("purpose", "geometry"),
    )


@_tool("suggest_memory")
def _handle_suggest_memory(arguments: dict[str, Any]) -> dict[str, Any]:
    return suggest_memory(
        n_atoms=arguments["n_atoms"],
        basis=arguments["basis"],
        method=arguments["method"],
        n_heavy_atoms=arguments.get("n_heavy_atoms"),
    )


@_tool("suggest_resources", needs="executable_or_scheduler")
def _handle_suggest_resources(arguments: dict[str, Any]) -> dict[str, Any]:
    hw = arguments.get("hw_specs")
    if not hw and arguments.get("profile"):
        from chemtools.execution.resource_inspection import (
            get_local_resource_budget,
            query_partition_specs,
        )
        from chemtools.execution.profiles import (
            _resolve_profile,
            load_runner_profiles,
        )
        profiles_path = arguments.get("profiles_path") or os.environ.get("CHEMTOOLS_RUNNER_PROFILES")
        profiles = load_runner_profiles(profiles_path)
        profile_payload = _resolve_profile(profiles, arguments["profile"])
        launcher = profile_payload.get("launcher", {})
        if launcher.get("kind") == "scheduler":
            partition = profile_payload.get("resources", {}).get("partition")
            scheduler_type = (
                profile_payload.get("scheduler", {}).get("system")
                or launcher.get("scheduler_type", "slurm")
            ).lower()
            hw = query_partition_specs(partition, scheduler_type) if partition else {}
            hw.setdefault("cpus_per_node", profile_payload.get("resources", {}).get("mpi_ranks"))
        else:
            hw = get_local_resource_budget()
    if not hw:
        from chemtools.execution.resource_inspection import (
            get_local_resource_budget,
        )
        hw = get_local_resource_budget()
    return suggest_resources(input_file=arguments["input_file"], hw_specs=hw)


# ---------------------------------------------------------------------------
# Cube file parser
# ---------------------------------------------------------------------------

@_tool("parse_cube_file")
def _handle_parse_cube_file(arguments: dict[str, Any]) -> dict[str, Any]:
    if arguments.get("summarize", False):
        return summarize_cube(
            arguments["file_path"],
            top_atoms=arguments.get("top_atoms", 5),
        )
    return parse_cube(
        arguments["file_path"],
        include_values=arguments.get("include_values", False),
    )


@_tool("compare_cube_densities")
def _handle_compare_cube_densities(arguments: dict[str, Any]) -> dict[str, Any]:
    return compare_cube_densities(
        arguments["reference_cube"],
        arguments["candidate_cube"],
        reference_density_unit=arguments["reference_density_unit"],
        candidate_density_unit=arguments["candidate_density_unit"],
    )


@_tool("compare_cube_orbitals")
def _handle_compare_cube_orbitals(arguments: dict[str, Any]) -> dict[str, Any]:
    return compare_cube_orbitals(
        arguments["reference_cube"],
        arguments["candidate_cube"],
        reference_orbital_label=arguments["reference_orbital_label"],
        candidate_orbital_label=arguments["candidate_orbital_label"],
    )


@_tool("compare_cube_orbital_subspaces")
def _handle_compare_cube_orbital_subspaces(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return compare_cube_orbital_subspaces(
        arguments["reference_orbitals"],
        arguments["candidate_orbitals"],
    )


# ---------------------------------------------------------------------------
# Preflight / job-script tools
# ---------------------------------------------------------------------------

@_tool("preflight_check", needs="runner_profile")
def _handle_preflight_check(arguments: dict[str, Any]) -> dict[str, Any]:
    return preflight_check(
        input_file=arguments["input_file"],
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path") or os.environ.get("CHEMTOOLS_RUNNER_PROFILES"),
    )


@_tool("render_job_script", needs="scheduler")
def _handle_render_job_script(arguments: dict[str, Any]) -> dict[str, Any]:
    return render_nwchem_job_script(
        input_path=arguments["input_file"],
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path") or os.environ.get("CHEMTOOLS_RUNNER_PROFILES"),
        job_name=arguments.get("job_name"),
        resource_overrides=arguments.get("resource_overrides"),
    )


# ---------------------------------------------------------------------------
# Parallel job monitoring, session log, input versioning
# ---------------------------------------------------------------------------

@_tool("watch_multiple_runs", needs="executable")
def _handle_watch_multiple_runs(arguments: dict[str, Any]) -> dict[str, Any]:
    return watch_multiple_nwchem_runs(
        jobs=arguments["jobs"],
        profile=arguments.get("profile"),
        profiles_path=arguments.get("profiles_path"),
        poll_interval_seconds=arguments.get("poll_interval_seconds", 30.0),
        timeout_seconds=arguments.get("timeout_seconds"),
    )


@_tool("init_session_log")
def _handle_init_session_log(arguments: dict[str, Any]) -> dict[str, Any]:
    return init_session_log(
        log_path=arguments["log_path"],
        session_title=arguments["session_title"],
        working_dir=arguments.get("working_dir"),
    )


@_tool("append_session_log")
def _handle_append_session_log(arguments: dict[str, Any]) -> dict[str, Any]:
    return append_session_log(
        log_path=arguments["log_path"],
        entry_type=arguments["entry_type"],
        content=arguments["content"],
    )


@_tool("next_versioned_path")
def _handle_next_versioned_path(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"path": next_versioned_path(arguments["path"])}


# ---------------------------------------------------------------------------
# Basis library summary
# ---------------------------------------------------------------------------

def _generic_basis_library_path(path: str | None = None) -> str:
    """Resolve the bundled NWChem basis library — shared with tools/nwchem.py."""
    if path:
        return path
    try:
        from importlib.resources import files as _pkg_files
        default = Path(str(_pkg_files("chemtools").joinpath("data/nwchem/basis_library")))
    except Exception:
        default = _REPO_ROOT / "chemtools" / "data" / "nwchem" / "basis_library"
    return os.environ.get("CHEMTOOLS_BASIS_LIBRARY", str(default))


@_tool("basis_library_summary")
def _handle_basis_library_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    return basis_library_summary(
        library_path=_generic_basis_library_path(arguments.get("library_path")),
    )


# ---------------------------------------------------------------------------
# Registry (cross-program runs)
# ---------------------------------------------------------------------------

@_tool("register_run", needs="registry")
def _handle_register_run_generic(arguments: dict[str, Any]) -> dict[str, Any]:
    """Generic registry — tags the run with whichever program is passed."""
    return register_run(
        program=arguments.get("program"),
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
        run_uid=arguments.get("run_uid"),
    )


# Shared helpers for the per-program legacy aliases living in tools/nwchem.py.
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


@_tool("update_run_status", needs="registry")
def _handle_update_run_status_generic(arguments: dict[str, Any]) -> dict[str, Any]:
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


@_tool("list_runs", needs="registry")
def _handle_list_runs_generic(arguments: dict[str, Any]) -> dict[str, Any]:
    return _do_list_runs(arguments)


def _do_get_run_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_run_summary(
        run_id=arguments.get("run_id"),
        run_uid=arguments.get("run_uid"),
        job_name=arguments.get("job_name"),
    )


@_tool("get_run_summary", needs="registry")
def _handle_get_run_summary_generic(arguments: dict[str, Any]) -> dict[str, Any]:
    return _do_get_run_summary(arguments)


# --- Registry: campaigns ---
def _do_create_campaign(arguments: dict[str, Any]) -> dict[str, Any]:
    return create_campaign(
        name=arguments["name"],
        description=arguments.get("description"),
        tags=arguments.get("tags"),
    )


@_tool("create_campaign", needs="registry")
def _handle_create_campaign_generic(arguments: dict[str, Any]) -> dict[str, Any]:
    return _do_create_campaign(arguments)


def _do_get_campaign_status(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_campaign_status(
        campaign_id=arguments.get("campaign_id"),
        name=arguments.get("name"),
    )


@_tool("get_campaign_status", needs="registry")
def _handle_get_campaign_status_generic(arguments: dict[str, Any]) -> dict[str, Any]:
    return _do_get_campaign_status(arguments)


def _do_get_campaign_energies(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_campaign_energies(
        campaign_id=arguments.get("campaign_id"),
        name=arguments.get("name"),
    )


@_tool("get_campaign_energies", needs="registry")
def _handle_get_campaign_energies_generic(arguments: dict[str, Any]) -> dict[str, Any]:
    return _do_get_campaign_energies(arguments)


# --- Registry: workflows ---
def _do_create_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    return create_workflow(
        name=arguments["name"],
        steps=arguments["steps"],
        protocol=arguments.get("protocol"),
        campaign_id=arguments.get("campaign_id"),
    )


@_tool("create_workflow", needs="registry")
def _handle_create_workflow_generic(arguments: dict[str, Any]) -> dict[str, Any]:
    return _do_create_workflow(arguments)


def _do_advance_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    return advance_workflow(workflow_id=arguments["workflow_id"])


@_tool("advance_workflow", needs="registry")
def _handle_advance_workflow_generic(arguments: dict[str, Any]) -> dict[str, Any]:
    return _do_advance_workflow(arguments)


# ---------------------------------------------------------------------------
# Generic tool definitions — physically live in this module (formerly
# inlined in ``_nwchem_tool_definitions()``). Aggregated into the global
# ``tool_definitions()`` in ``chemtools/mcp/dispatch.py``.
# ---------------------------------------------------------------------------


def generic_tool_definitions() -> list[dict[str, Any]]:
    """Tool-definition dicts for all generic (cross-program) tools.

    Physically live here (as of the multi-program MCP split — formerly
    in ``_nwchem_tool_definitions()``). Aggregated into the global
    ``tool_definitions()`` in ``chemtools/mcp/dispatch.py``.
    """
    return [
        {
            "name": "get_server_mode",
            "description": (
                "Report tool-visibility mode, explicit execution permission, "
                "configured target names, executors, installed programs, and "
                "blocked tools. Commands and allowed roots are not returned."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "summarize_run",
            "description": (
                "One-call thick summary of any registered program's output file. Auto-detects "
                "the program (NWChem / Molpro / Molcas) and returns a compact ParsedRun "
                "(tasks list with kind/method/basis/energy/outcome, auto-picked "
                "primary_task_index, pre-computed derived scalars like final_energy_hartree "
                "and n_imaginary_modes) plus a Diagnosis (verdict label + ready-to-execute "
                "next_actions). Designed for small-LLM workflows: read the verdict, "
                "execute next_actions[0]. For heavy sections (full MO coefficients, "
                "trajectories, eigenvectors) use the section-specific tools instead."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {
                        "type": "string",
                        "description": "Path to the output file. Program is auto-detected.",
                    },
                    "program": {
                        "type": "string",
                        "enum": ["nwchem", "molcas", "dirac", "grasp", "qe", "qmcpack"],
                        "description": "Optional program override; if omitted, auto-detect from the file head.",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_output",
            "description": (
                "Auto-detecting cross-program parser. Returns the unified "
                "ParsedRun shape: program, program_version, file, "
                "file_size_bytes, tasks (list of TaskSummary dicts with "
                "kind/method/basis/energy/outcome), primary_task_index, "
                "derived (scalars: final_energy_hartree, primary_energy_hartree, "
                "n_tasks, n_imaginary_modes, ...), diagnostics, diagnosis. "
                "Compact by design — fits in context for huge outputs. For "
                "program-specific rich data (per-module SCF/CASPT2/etc. "
                "details, MO coefficients, active-space summary), call the "
                "program-prefixed parse_nwchem_output / parse_molcas_output."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "program": {"type": "string", "enum": ["nwchem", "molcas", "dirac", "grasp", "qe", "qmcpack"],
                                "description": "Optional program override; auto-detected from file head if omitted."},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "extract_geometry",
            "description": (
                "Auto-detecting geometry extractor. Dispatches to the program "
                "plugin's parser.get_geometry() — for NWChem this returns the "
                "last converged geometry, for Molcas the SLAPAF converged or "
                "last 'Cartesian coordinates' block. Atoms list with "
                "{symbol, x, y, z} dicts. Pass `program` to override the auto-"
                "detect from the file head."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "program": {"type": "string", "enum": ["nwchem", "molcas", "dirac", "grasp", "qe", "qmcpack"]},
                    "task_index": {"type": ["integer", "null"], "default": None,
                        "description": "0-indexed task. None = primary task (final geometry for opt, input for energy/freq)."},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_thermochem",
            "description": (
                "Auto-detecting thermochemistry parser. Returns ZPE, thermal "
                "corrections, S, Cv, H, G at per-temperature granularity. "
                "Routes to the plugin's parser.get_thermochem(). NWChem returns "
                "a single converged thermochem block; Molcas returns per-"
                "temperature blocks with a 'standard_298_15' shortcut."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "program": {"type": "string", "enum": ["nwchem", "molcas", "dirac", "grasp", "qe", "qmcpack"]},
                    "task_index": {"type": ["integer", "null"], "default": None},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_frequencies",
            "description": (
                "Auto-detecting harmonic-frequency parser. Returns normal "
                "modes (cm⁻¹, IR intensities, reduced mass) and ZPVE. Dispatches "
                "to the plugin's parser.get_frequency(). Imaginary modes encoded "
                "as negative cm⁻¹ values."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "program": {"type": "string", "enum": ["nwchem", "molcas", "dirac", "grasp", "qe", "qmcpack"]},
                    "task_index": {"type": ["integer", "null"], "default": None},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_trajectory",
            "description": (
                "Auto-detecting trajectory parser for geometry optimizations "
                "and molecular dynamics. Returns the backend trajectory "
                "contract with per-frame geometry, provenance, available "
                "optimization evidence, and bounded structural analysis where "
                "supported. Dispatches to the plugin's parser.get_trajectory()."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "program": {"type": "string", "enum": ["nwchem", "molcas", "dirac", "grasp", "qe", "qmcpack"]},
                    "task_index": {"type": ["integer", "null"], "default": None},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "inspect_geometry",
            "description": (
                "Auto-detecting geometry inspector. Pulls the geometry via the "
                "plugin's parser.get_geometry(), normalizes to Å, then computes "
                "formula + bond lengths (≤2.5 Å) + bond angles (through bonded "
                "triples) + close contacts (<0.6 Å) + fragments + center of "
                "mass. Optional `measurements` dict {distances, angles, "
                "dihedrals} for explicit 1-based-index measurements. Uses "
                "core.geometry.inspect_geometry under the hood — same math as "
                "the program-specific inspect_*_geometry tools."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "program": {"type": "string", "enum": ["nwchem", "molcas", "dirac", "grasp", "qe", "qmcpack"]},
                    "max_bond_length": {"type": "number", "default": 2.5},
                    "min_safe_distance": {"type": "number", "default": 0.6},
                    "covalent_tolerance": {"type": "number", "default": 1.20},
                    "measurements": {
                        "type": ["object", "null"], "default": None,
                        "properties": {
                            "distances": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                            "angles": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                            "dihedrals": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                        },
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "summarize_output",
            "description": (
                "Auto-detecting summary tool. Routes to summarize_nwchem_output "
                "(compact case summary including method/energy/imaginary modes "
                "and bullets per task) or summarize_molcas_output (flat "
                "structured summary with method/energy/active_space/freqs/"
                "thermochem). Returns the program-specific shape tagged with "
                "`program`. For a unified ParsedRun, use `parse_output`."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "program": {"type": "string", "enum": ["nwchem", "molcas"]},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "analyze_case",
            "description": (
                "Auto-detecting quality analyzer. Routes to analyze_nwchem_case "
                "(rich review payload with state checks + active-space density) "
                "or analyze_molcas_case (verdict + issues + next_actions). Use "
                "this when you want to know 'is this run healthy?' regardless "
                "of program. Branch on the `program` field in the response to "
                "interpret the rest of the shape."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "program": {"type": "string", "enum": ["nwchem", "molcas"]},
                    "input_file": {"type": "string", "description": "NWChem only — matching input file."},
                    "err_file": {"type": "string", "description": "NWChem only — slurm/scheduler error file."},
                    "expected_metals": {"type": "array", "items": {"type": "string"}, "description": "NWChem only."},
                    "expected_somos": {"type": "integer", "description": "NWChem only."},
                    "detail": {"type": "string", "enum": ["compact", "full"], "default": "compact", "description": "NWChem only."},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_recovery",
            "description": (
                "Auto-detecting recovery suggester. Routes to "
                "suggest_nwchem_recovery (rich SCF + state-recovery strategy "
                "payloads) or suggest_molcas_recovery (failure-class rule "
                "engine with fix_recipe + next_actions). Both serve the same "
                "goal — tell the agent how to recover from a failed run — but "
                "shapes differ. Branch on the `program` field."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "program": {"type": "string", "enum": ["nwchem", "molcas"]},
                    "input_file": {"type": "string", "description": "NWChem only."},
                    "expected_metals": {"type": "array", "items": {"type": "string"}, "description": "NWChem only."},
                    "expected_somos": {"type": "integer", "description": "NWChem only."},
                    "mode": {"type": "string", "enum": ["auto", "scf", "state"], "default": "auto", "description": "NWChem only — which strategy to return."},
                    "return_all_matches": {"type": "boolean", "default": False, "description": "Molcas only — return all rule matches."},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "apply_recovery",
            "description": (
                "Auto-detecting apply_recovery. Currently only Molcas has a "
                "mechanical-fix patcher (regex edits on .input files for known "
                "failure classes). NWChem runs return verdict="
                "'not_implemented_for_program' — use suggest_recovery for the "
                "strategy bundle and apply changes via a draft_nwchem_* tool."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "output_file": {"type": "string", "description": "Either output_file (auto-classify) or recovery dict required for Molcas."},
                    "recovery": {"type": ["object", "null"], "default": None, "description": "Molcas only — pre-computed recovery dict from suggest_molcas_recovery."},
                    "write_to": {"type": ["string", "null"], "default": None, "description": "Molcas only — output path for the fixed input."},
                    "program": {"type": "string", "enum": ["nwchem", "molcas"]},
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_initial_geometry",
            "description": (
                "Create an initial geometry XYZ file from an element list using covalent-radii estimates. "
                "Always use this instead of writing XYZ files manually. "
                "Handles diatomics, MXn complexes (n=1..6), and linear chains automatically."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "atoms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Flat list of element symbols, e.g. ['Fe', 'Cl'] or ['Fe', 'Cl', 'Cl', 'Cl', 'Cl']. Repeats allowed.",
                    },
                    "output_path": {"type": "string", "description": "Where to write the XYZ file."},
                    "comment": {"type": "string", "description": "Optional XYZ comment line."},
                    "central_atom": {"type": "string",
                                     "description": "Hint for which element is the central atom (auto-detected if omitted)."},
                },
                "required": ["atoms", "output_path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "compute_reaction_energy",
            "description": (
                "Compute reaction energy ΔE from NWChem output files. "
                "Uses best energy per species (CCSD(T)>CCSD>MP2>DFT>SCF). "
                "Returns ΔE in Hartree/kcal/eV. Set include_thermochem=true for ZPE/ΔH/ΔG."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "species": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Dict mapping label → output file path. E.g. {'FeO2-': 'feo2.out', 'Fe': 'fe.out', 'O': 'o.out'}.",
                    },
                    "reactants": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "Stoichiometric coefficients for reactants (positive). E.g. {'FeO2-': 1}.",
                    },
                    "products": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "Stoichiometric coefficients for products (positive). E.g. {'Fe': 1, 'O': 2}.",
                    },
                    "method": {
                        "type": "string",
                        "description": "If set, only use energies from this method level (e.g. 'CCSD'). Default: auto (highest available).",
                    },
                    "include_thermochem": {
                        "type": "boolean",
                        "description": "If true, extract ZPE/H(T)/G(T) from frequency outputs and compute ΔE+ZPE, ΔH, ΔG. Default: false.",
                    },
                },
                "required": ["species", "reactants", "products"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_spin_state",
            "description": (
                "Suggest likely spin multiplicities for a molecule given its elements and charge. "
                "For transition-metal systems computes d-electron counts and returns high-spin "
                "and low-spin multiplicity candidates. Call this before drafting any input to "
                "determine the correct 'multiplicity' (and 'nopen') value."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "elements": {
                        "type": "array", "items": {"type": "string"},
                        "description": "All element symbols in the molecule, e.g. ['Fe', 'Cl', 'Cl'].  Repeats are fine.",
                    },
                    "charge": {"type": "integer", "default": 0, "description": "Total molecular charge."},
                    "metal_oxidation_states": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                        "description": "Optional: formal oxidation states for metal(s), e.g. {'Fe': 2}. If omitted, common states are enumerated.",
                    },
                },
                "required": ["elements"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_basis_set",
            "description": (
                "Suggest an appropriate basis set (and ECP when needed) for a molecule. "
                "Returns 'basis_assignments' and 'ecp_assignments' ready to pass directly "
                "to create_nwchem_input. Call this before drafting any input."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "elements": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Element symbols in the molecule.",
                    },
                    "purpose": {
                        "type": "string",
                        "enum": ["geometry", "single_point", "correlated", "heavy_elements"],
                        "default": "geometry",
                        "description": "'geometry' for opt, 'single_point' for DFT energy, 'correlated' for MP2/CCSD, 'heavy_elements' for post-Kr.",
                    },
                },
                "required": ["elements"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_memory",
            "description": (
                "Suggest NWChem memory settings for a calculation. "
                "Returns a 'memory_string' ready to use as the 'memory' parameter "
                "in create_nwchem_input or create_nwchem_dft_workflow_input. "
                "Pass the same basis that will be used in the calculation — use the "
                "'basis' field returned by suggest_basis_set, not a guess."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_atoms": {"type": "integer", "description": "Total number of atoms."},
                    "basis": {"type": "string", "description": "Basis set name, e.g. 'def2-svp'."},
                    "method": {
                        "type": "string",
                        "description": "Computational method: 'scf', 'dft', 'mp2', 'ccsd', or 'ccsd(t)'.",
                    },
                    "n_heavy_atoms": {"type": "integer", "description": "Number of non-hydrogen atoms (optional)."},
                },
                "required": ["n_atoms", "basis", "method"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_resources",
            "description": (
                "Low-level: recommend MPI rank count and memory per rank for a single node. "
                "For HPC jobs, prefer suggest_nwchem_resources instead — it is profile-aware, "
                "multi-node capable, and handles task-type-specific walltime and memory. "
                "This tool only handles single-node rank/memory selection using a BF/rank scaling model."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the NWChem .nw input file."},
                    "profile": {
                        "type": "string",
                        "description": "Runner profile name. If provided, hw_specs are queried automatically from the partition.",
                    },
                    "profiles_path": {"type": "string"},
                    "hw_specs": {
                        "type": "object",
                        "description": "Hardware specs override. Keys: cpus_per_node, node_memory_mb, cpu_arch.",
                    },
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_relativistic_correction",
            "description": (
                "Advise on relativistic corrections (X2C/DKH2/none) for given elements. "
                "Returns the relativistic block and basis compatibility warnings. "
                "Call before drafting inputs with 4d/5d metals or Z > 36."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "All element symbols in the molecule.",
                    },
                    "basis_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Dict of element → basis name. Used to detect DK-quality bases.",
                    },
                    "ecp_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Dict of element → ECP name. If present, warns about X2C/DKH incompatibility.",
                    },
                    "purpose": {
                        "type": "string",
                        "enum": ["dft", "scf", "ccsd", "property"],
                        "default": "dft",
                        "description": "Type of calculation.",
                    },
                },
                "required": ["elements"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_cube_file",
            "description": "Parse or summarize a Gaussian/NWChem cube file. By default returns header and grid metadata. Set summarize=true for grid statistics and approximate atom-localized density/orbital lobes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "include_values": {"type": "boolean", "default": False},
                    "summarize": {"type": "boolean", "default": False},
                    "top_atoms": {"type": "integer", "default": 5},
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "compare_cube_densities",
            "description": (
                "Compare two explicitly declared electron-density CUBE files "
                "only when they have the same nuclear geometry and exact "
                "real-space grid. Density values are converted from their "
                "declared units before electron-count, L1, L2, RMS, and "
                "maximum-difference metrics are calculated. The tool refuses "
                "grid or geometry mismatch and does not resample fields."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_cube": {"type": "string", "minLength": 1},
                    "candidate_cube": {"type": "string", "minLength": 1},
                    "reference_density_unit": {
                        "type": "string",
                        "enum": [
                            "electron_per_bohr3",
                            "electron_per_angstrom3",
                        ],
                        "description": "Scalar density unit declared for the reference CUBE values.",
                    },
                    "candidate_density_unit": {
                        "type": "string",
                        "enum": [
                            "electron_per_bohr3",
                            "electron_per_angstrom3",
                        ],
                        "description": "Scalar density unit declared for the candidate CUBE values.",
                    },
                },
                "required": [
                    "reference_cube",
                    "candidate_cube",
                    "reference_density_unit",
                    "candidate_density_unit",
                ],
                "additionalProperties": False,
            },
        },
        {
            "name": "compare_cube_orbitals",
            "description": (
                "Compare one caller-matched non-degenerate orbital CUBE pair "
                "on the same nuclear geometry and exact real-space grid. "
                "Reports signed normalized overlap, the candidate phase flip "
                "when needed, phase-aligned overlap, and aligned L2 distance. "
                "The tool refuses density, spin-density, and potential fields; "
                "it does not choose orbital correspondence, resample grids, or "
                "compare degenerate subspaces."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_cube": {"type": "string", "minLength": 1},
                    "candidate_cube": {"type": "string", "minLength": 1},
                    "reference_orbital_label": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Caller-declared identity of the reference non-degenerate orbital.",
                    },
                    "candidate_orbital_label": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Caller-declared identity of the candidate non-degenerate orbital.",
                    },
                },
                "required": [
                    "reference_cube",
                    "candidate_cube",
                    "reference_orbital_label",
                    "candidate_orbital_label",
                ],
                "additionalProperties": False,
            },
        },
        {
            "name": "compare_cube_orbital_subspaces",
            "description": (
                "Compare two caller-declared equal-size orbital CUBE "
                "subspaces on one exact grid and nuclear geometry. The tool "
                "orthonormalizes the sampled fields, then reports the raw "
                "cross-overlap matrix plus phase- and rotation-invariant "
                "principal overlaps and angles. It refuses rank-deficient "
                "sets, density/potential fields, mismatched grids, and does "
                "not choose the subspace membership."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_orbitals": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "minLength": 1},
                                "orbital_label": {"type": "string", "minLength": 1},
                            },
                            "required": ["path", "orbital_label"],
                            "additionalProperties": False,
                        },
                    },
                    "candidate_orbitals": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "minLength": 1},
                                "orbital_label": {"type": "string", "minLength": 1},
                            },
                            "required": ["path", "orbital_label"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["reference_orbitals", "candidate_orbitals"],
                "additionalProperties": False,
            },
        },
        {
            "name": "preflight_check",
            "description": (
                "Run all pre-submission checks on a NWChem input and return a pass/fail report. "
                "Combines: lint (syntax/consistency), movecs input file existence, and memory vs node RAM ceiling. "
                "Call before launch_nwchem_run to catch errors before wasting queue time."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the NWChem .nw input file."},
                    "profile": {"type": "string", "description": "Runner profile name (for memory ceiling check)."},
                    "profiles_path": {"type": "string"},
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "render_job_script",
            "description": (
                "Preview the HPC job submission script (SLURM .job, PBS .job, etc.) that would be created "
                "for a given input file and scheduler profile. Does not write or submit. "
                "Use before launch_nwchem_run to verify the script, resource settings, and output file paths."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "profile": {"type": "string"},
                    "profiles_path": {"type": "string"},
                    "job_name": {"type": "string"},
                    "resource_overrides": {
                        "type": "object",
                        "description": "Override specific resource fields, e.g. {\"walltime\": \"48:00:00\", \"mpi_ranks\": 96}",
                    },
                },
                "required": ["input_file", "profile"],
                "additionalProperties": False,
            },
        },
        {
            "name": "watch_multiple_runs",
            "description": (
                "Monitor multiple NWChem jobs simultaneously until all reach a terminal state "
                "(completed, failed, or cancelled). Use this after submitting several jobs in "
                "parallel with auto_watch=false — call this once and it will block until all "
                "jobs finish, then return a consolidated status table. "
                "Each job entry requires output_file. External Slurm jobs must "
                "also provide profile and job_id together."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "jobs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "output_file": {"type": "string", "description": "Path to the .out file for this job."},
                                "job_id": {"type": "string", "description": "Explicit external Slurm job ID; requires profile."},
                                "profile": {"type": "string", "description": "Runner profile name (required for HPC scheduler jobs)."},
                                "label": {"type": "string", "description": "Human-readable label for this job in the summary table."},
                            },
                            "required": ["output_file"],
                            "additionalProperties": False,
                        },
                        "description": "List of jobs to watch.",
                    },
                    "profile": {"type": "string", "description": "Default runner profile for all jobs (overridden per-job if set)."},
                    "profiles_path": {"type": "string"},
                    "poll_interval_seconds": {"type": "number", "default": 30, "description": "How often to poll scheduler status."},
                    "timeout_seconds": {"type": "number", "description": "Give up after this many seconds (null = no timeout)."},
                },
                "required": ["jobs"],
                "additionalProperties": False,
            },
        },
        {
            "name": "init_session_log",
            "description": (
                "Create a new session log Markdown file. Call this at the START of every "
                "multi-step NWChem workflow to establish a running record. The log captures "
                "what was done, what was found, and what the next steps are — preserving "
                "context across long sessions and providing a summary the user can review. "
                "Returns the log path; save it and pass to append_session_log throughout the session."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "log_path": {"type": "string", "description": "Where to write the Markdown log (e.g. /path/to/session.md)."},
                    "session_title": {"type": "string", "description": "Short title describing this session's goal."},
                    "working_dir": {"type": "string", "description": "Working directory for this session (for context)."},
                },
                "required": ["log_path", "session_title"],
                "additionalProperties": False,
            },
        },
        {
            "name": "append_session_log",
            "description": (
                "Append a timestamped entry to the session log. Call this frequently throughout "
                "a workflow: after each major action (job launch, parse, fix), after finding "
                "errors or making decisions, and at the end to write a final summary. "
                "Entry types: 'step' (action taken), 'result' (what was found), "
                "'error' (problems encountered), 'note' (decisions/reasoning), 'summary' (final recap)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "log_path": {"type": "string", "description": "Path to the log file (from init_session_log)."},
                    "entry_type": {
                        "type": "string",
                        "enum": ["step", "result", "error", "note", "summary"],
                        "description": "Category of this log entry.",
                    },
                    "content": {"type": "string", "description": "Markdown content for this entry."},
                },
                "required": ["log_path", "entry_type", "content"],
                "additionalProperties": False,
            },
        },
        {
            "name": "next_versioned_path",
            "description": (
                "Return the next available versioned path for a NWChem input file, "
                "avoiding overwrites. Given 'fe.nw', returns 'fe_v2.nw' if that file "
                "does not exist, or 'fe_v3.nw' if _v2 already exists, etc. "
                "ALWAYS call this before creating or modifying an input file — never "
                "overwrite existing .nw files so the user can track the progression."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Existing or planned input file path."},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "basis_library_summary",
            "description": (
                "List all basis sets and ECPs available in the bundled library. "
                "Returns counts and names grouped by category (orbital, ECP, auxiliary). "
                "Use this to check what basis sets are available before drafting inputs."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "register_run",
            "description": (
                "Register a run in the persistent registry with a program tag "
                "(nwchem / molcas / dirac / grasp / ...). Generic version of "
                "register_nwchem_run — same schema plus a ``program`` field. "
                "Use this when registering Molcas runs or building cross-"
                "program campaigns (e.g. CrO atomization with Cr at NWChem "
                "CCSD(T) + CrO at Molcas CASPT2). Returns the local run_id "
                "and portable run_uid."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "program": {"type": "string", "enum": ["nwchem", "molcas", "dirac", "grasp", "qe", "qmcpack"],
                                "description": "QC program that produced this run."},
                    "job_name": {"type": "string"},
                    "input_file": {"type": "string"},
                    "output_file": {"type": "string"},
                    "profile": {"type": "string"},
                    "method": {"type": "string"},
                    "functional": {"type": "string"},
                    "basis": {"type": "string"},
                    "n_atoms": {"type": "integer"},
                    "elements": {"type": "array", "items": {"type": "string"}},
                    "charge": {"type": "integer"},
                    "multiplicity": {"type": "integer"},
                    "mpi_ranks": {"type": "integer"},
                    "campaign_id": {"type": "integer"},
                    "workflow_id": {"type": "integer"},
                    "workflow_step_id": {"type": "string"},
                    "parent_run_id": {"type": "integer"},
                    "tags": {"type": "object"},
                    "run_uid": {
                        "type": "string",
                        "format": "uuid",
                        "description": (
                            "Portable ID to preserve when importing a run. "
                            "Generated when omitted."
                        ),
                    },
                },
                "required": ["job_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "update_run_status",
            "description": (
                "Update a registered run's status and optionally its results "
                "(energy, H, G, imaginary modes, walltime). Generic version — "
                "works on any registered run regardless of program. Call after "
                "a job completes, fails, or is cancelled."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer", "description": "The run_id from register_run."},
                    "status": {"type": "string", "enum": ["submitted", "running", "completed", "failed", "timelimited", "oom", "cancelled"]},
                    "energy_hartree": {"type": "number"},
                    "h_hartree": {"type": "number", "description": "Enthalpy H(T) in Hartree."},
                    "g_hartree": {"type": "number", "description": "Gibbs G(T) in Hartree."},
                    "imaginary_modes": {"type": "integer"},
                    "walltime_used_sec": {"type": "number"},
                    "sec_per_gradient": {"type": "number"},
                    "output_file": {"type": "string"},
                },
                "required": ["run_id", "status"],
                "additionalProperties": False,
            },
        },
        {
            "name": "list_runs",
            "description": (
                "List registered runs, optionally filtered by campaign, "
                "workflow, status, method, or **program** (nwchem / molcas / "
                "grasp / ...). Generic version. Returns most recent first."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "campaign_id": {"type": "integer"},
                    "workflow_id": {"type": "integer"},
                    "status": {"type": "string"},
                    "method": {"type": "string"},
                    "program": {"type": "string", "description": "Filter by QC program tag."},
                    "limit": {"type": "integer", "description": "Max results (default 50)."},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "get_run_summary",
            "description": (
                "Get detailed info for a single registered run, including its "
                "restart chain. Generic runs are addressable by "
                "run_id, run_uid, or job_name regardless of program."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer"},
                    "run_uid": {"type": "string", "format": "uuid"},
                    "job_name": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "create_campaign",
            "description": (
                "Create a campaign to group related runs across one or more "
                "QC programs. E.g. a cross-program atomization study with Cr "
                "atom at NWChem CCSD(T) and CrO at Molcas CASPT2 — both share "
                "the campaign_id and `get_campaign_energies` returns a "
                "sortable program-tagged energy table. Returns a campaign_id."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "tags": {"type": "object"},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_campaign_status",
            "description": (
                "Get aggregate status for a campaign: total/completed/running/"
                "failed counts, completion percentage, estimated remaining "
                "time. Generic version."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "campaign_id": {"type": "integer"},
                    "name": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "get_campaign_energies",
            "description": (
                "Energy table for all completed runs in a campaign, sorted by "
                "energy with relative energies in kcal/mol. Each row carries "
                "the run's program tag so cross-program campaigns are "
                "self-labeled. Includes H(T) and G(T) if available."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "campaign_id": {"type": "integer"},
                    "name": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "create_workflow",
            "description": (
                "Create a workflow DAG with step dependencies. Generic version. "
                "Workflows can span programs — each step's auto_input dict can "
                "target whichever QC program is appropriate. Use "
                "`advance_workflow` to find ready-to-launch steps."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "depends_on": {"type": "string"},
                                "input_file": {"type": "string"},
                                "profile": {"type": "string"},
                                "auto_input": {"type": "object"},
                            },
                            "required": ["id"],
                        },
                    },
                    "protocol": {"type": "string"},
                    "campaign_id": {"type": "integer"},
                },
                "required": ["name", "steps"],
                "additionalProperties": False,
            },
        },
        {
            "name": "advance_workflow",
            "description": (
                "Check a workflow's progress and return which steps are ready "
                "to launch. Does not launch jobs — the caller decides. Generic "
                "version. Returns workflow state, completed/running/failed "
                "steps, and unblocked steps."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "integer"},
                },
                "required": ["workflow_id"],
                "additionalProperties": False,
            },
        },
    ]

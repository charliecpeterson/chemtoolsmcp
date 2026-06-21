"""GRASP2018 MCP tool definitions and handlers.

GRASP runs as a chain of small executables in an apptainer container.
Each MCP tool here either:
  * **runs** an executable directly (executable capability — local/hpc only)
  * **plans** the heredoc input + command sequence (analysis-safe, no exec)
  * **parses** a GRASP output file (rlevels, .sum, .lsj.lbl, rmcdhf log)
  * **analyzes** a level table (term grouping, splittings, comparisons)

All run_* tools auto-append a `grasp_session.md` markdown log to the working
directory recording the command, stdin, key stdout, and elapsed time. This
gives the user a replayable trace of every interactive debug step.

Container path resolution:
  * default: ``~/mycontainers/grasp2018.sif``
  * override: ``CHEMTOOLS_GRASP_CONTAINER`` env var

Workflow tools (planners) work in any mode. Run tools require the
``executable`` capability (i.e. ``local`` or ``hpc`` mode).
"""

from __future__ import annotations

from typing import Any

from chemtools.mcp.decorator import _tool

from chemtools.programs.grasp.runtime import (
    run_grasp_exe as _run_grasp_exe,
    container_available as _container_available,
    resolve_container as _resolve_container,
    append_session_note as _append_session_note,
    read_session_log as _read_session_log,
)
from chemtools.programs.grasp.input.heredoc import (
    rnucleus_input as _rnucleus_input,
    rcsfgenerate_input as _rcsfgenerate_input,
    rangular_input as _rangular_input,
    rwfnestimate_input as _rwfnestimate_input,
    rmcdhf_input as _rmcdhf_input,
    jj2lsj_input as _jj2lsj_input,
    hf_input as _hf_input,
    rwfnmchfmcdf_input as _rwfnmchfmcdf_input,
    rsave_args as _rsave_args,
    rci_input as _rci_input,
    rhfs_input as _rhfs_input,
    rhfs_lsj_input as _rhfs_lsj_input,
    ris4_input as _ris4_input,
    rbiotransform_input as _rbiotransform_input,
    rtransition_input as _rtransition_input,
)
from chemtools.programs.grasp.parse.hfs import parse_hfs as _parse_hfs
from chemtools.programs.grasp.parse.ris import parse_ris as _parse_ris
from chemtools.programs.grasp.parse.transition import parse_transition as _parse_transition
from chemtools.programs.grasp.parse.rlevels import (
    parse_rlevels as _parse_rlevels,
    summarize_terms as _summarize_terms,
    compare_rlevels as _compare_rlevels,
)
from chemtools.programs.grasp.parse.lsjlbl import parse_lsjlbl as _parse_lsjlbl
from chemtools.programs.grasp.parse.sum_file import parse_sum as _parse_sum
from chemtools.programs.grasp.parse.rmcdhf_log import parse_rmcdhf_log as _parse_rmcdhf_log
from chemtools.programs.grasp.strategy.workflows import (
    plan_dhf_workflow as _plan_dhf_workflow,
    plan_nonrel_limit_workflow as _plan_nonrel_limit_workflow,
    plan_restart_from_workflow as _plan_restart_from_workflow,
    plan_hf_bootstrap_workflow as _plan_hf_bootstrap_workflow,
)
from chemtools.programs.grasp.strategy.runner import run_workflow as _run_workflow
from chemtools.programs.grasp.docs import (
    list_docs as _list_grasp_docs,
    search_docs as _search_grasp_docs,
    lookup_section as _lookup_grasp_section,
    read_doc_excerpt as _read_grasp_doc_excerpt,
    list_topics as _list_grasp_topics,
    get_topic_guide as _get_grasp_topic_guide,
)
from chemtools.programs.grasp.strategy.diagnose import (
    analyze_grasp_case as _analyze_grasp_case,
    suggest_grasp_recovery as _suggest_grasp_recovery,
)
from chemtools.programs.grasp.scheduler import (
    launch_grasp_workflow_run as _launch_grasp_workflow_run,
    get_grasp_run_status as _get_grasp_run_status,
    watch_grasp_run as _watch_grasp_run,
    terminate_grasp_run as _terminate_grasp_run,
)


def grasp_tool_definitions() -> list[dict[str, Any]]:
    """Return GRASP tool definitions for tools/list."""
    return _DEFS


# =============================================================================
# Per-exe runners (executable capability — hidden in analysis mode)
# =============================================================================

@_tool("run_grasp_exe", needs="executable", program="grasp")
def _handle_run_grasp_exe(arguments: dict[str, Any]) -> dict[str, Any]:
    """Generic GRASP exe runner. Use the typed builders for input safety."""
    return _run_grasp_exe(
        arguments["exe"],
        working_dir=arguments["working_dir"],
        stdin_lines=arguments.get("stdin_lines", ""),
        args=arguments.get("args"),
        timeout_seconds=arguments.get("timeout_seconds", 600.0),
        capture_log_file=arguments.get("capture_log_file"),
        log_to_session=arguments.get("log_to_session", True),
    )


@_tool("run_grasp_rnucleus", needs="executable", program="grasp")
def _handle_run_rnucleus(arguments: dict[str, Any]) -> dict[str, Any]:
    return _run_grasp_exe(
        "rnucleus",
        working_dir=arguments["working_dir"],
        stdin_lines=_rnucleus_input(
            z=arguments["z"], a=arguments["a"],
            point_source=arguments.get("point_source", False),
            revise_radius=arguments.get("revise_radius", False),
            nuclear_mass_amu=arguments.get("nuclear_mass_amu"),
            nuclear_spin=arguments.get("nuclear_spin", 0),
            dipole_moment=arguments.get("dipole_moment", 0),
            quadrupole_moment=arguments.get("quadrupole_moment", 0),
        ),
    )


@_tool("run_grasp_rcsfgenerate", needs="executable", program="grasp")
def _handle_run_rcsfgenerate(arguments: dict[str, Any]) -> dict[str, Any]:
    result = _run_grasp_exe(
        "rcsfgenerate",
        working_dir=arguments["working_dir"],
        stdin_lines=_rcsfgenerate_input(
            core=arguments.get("core", 0),
            configurations=arguments["configurations"],
            active_orbitals=arguments["active_orbitals"],
            twoj_min=arguments["twoj_min"], twoj_max=arguments["twoj_max"],
            excitations=arguments.get("excitations", 0),
            ordering=arguments.get("ordering", "*"),
        ),
    )
    # Convenience: copy rcsf.out to rcsf.inp for the next step
    if result["ok"] and arguments.get("copy_to_inp", True):
        from pathlib import Path
        work = Path(arguments["working_dir"])
        if (work / "rcsf.out").exists():
            (work / "rcsf.inp").write_bytes((work / "rcsf.out").read_bytes())
            result["copied_rcsf_out_to_inp"] = True
    return result


@_tool("run_grasp_rangular", needs="executable", program="grasp")
def _handle_run_rangular(arguments: dict[str, Any]) -> dict[str, Any]:
    return _run_grasp_exe(
        "rangular",
        working_dir=arguments["working_dir"],
        stdin_lines=_rangular_input(default_settings=arguments.get("default_settings", True)),
    )


@_tool("run_grasp_rwfnestimate", needs="executable", program="grasp")
def _handle_run_rwfnestimate(arguments: dict[str, Any]) -> dict[str, Any]:
    return _run_grasp_exe(
        "rwfnestimate",
        working_dir=arguments["working_dir"],
        stdin_lines=_rwfnestimate_input(
            default_settings=arguments.get("default_settings", True),
            speed_of_light_au=arguments.get("speed_of_light_au"),
            sources=arguments.get("sources", ["2"]),
        ),
    )


@_tool("run_grasp_rmcdhf", needs="executable", program="grasp")
def _handle_run_rmcdhf(arguments: dict[str, Any]) -> dict[str, Any]:
    return _run_grasp_exe(
        "rmcdhf",
        working_dir=arguments["working_dir"],
        stdin_lines=_rmcdhf_input(
            default_settings=arguments.get("default_settings", True),
            speed_of_light_au=arguments.get("speed_of_light_au"),
            block_level_selections=arguments["block_level_selections"],
            orbitals_to_optimize=arguments.get("orbitals_to_optimize", "*"),
            weighting=arguments.get("weighting", "5"),
            spectroscopic_orbitals=arguments.get("spectroscopic_orbitals", "*"),
            max_scf_cycles=arguments.get("max_scf_cycles", 100),
        ),
        timeout_seconds=arguments.get("timeout_seconds", 600.0),
    )


@_tool("run_grasp_rsave", needs="executable", program="grasp")
def _handle_run_rsave(arguments: dict[str, Any]) -> dict[str, Any]:
    return _run_grasp_exe(
        "rsave",
        working_dir=arguments["working_dir"],
        stdin_lines="",
        args=_rsave_args(arguments["name"]),
    )


@_tool("run_grasp_jj2lsj", needs="executable", program="grasp")
def _handle_run_jj2lsj(arguments: dict[str, Any]) -> dict[str, Any]:
    return _run_grasp_exe(
        "jj2lsj",
        working_dir=arguments["working_dir"],
        stdin_lines=_jj2lsj_input(
            name=arguments["name"],
            mixing_coefficients=arguments.get("mixing_coefficients", False),
            unique_labeling=arguments.get("unique_labeling", True),
            default_settings=arguments.get("default_settings", True),
        ),
    )


@_tool("run_grasp_rhfs", needs="executable", program="grasp")
def _handle_run_rhfs(arguments: dict[str, Any]) -> dict[str, Any]:
    return _run_grasp_exe(
        "rhfs",
        working_dir=arguments["working_dir"],
        stdin_lines=_rhfs_input(
            name=arguments["name"],
            ci_mixing=arguments.get("ci_mixing", False),
            default_settings=arguments.get("default_settings", True),
        ),
    )


@_tool("run_grasp_rhfs_lsj", needs="executable", program="grasp")
def _handle_run_rhfs_lsj(arguments: dict[str, Any]) -> dict[str, Any]:
    return _run_grasp_exe(
        "rhfs_lsj",
        working_dir=arguments["working_dir"],
        stdin_lines=_rhfs_lsj_input(
            name=arguments["name"],
            ci_mixing=arguments.get("ci_mixing", False),
            energy_sorted=arguments.get("energy_sorted", True),
        ),
    )


@_tool("run_grasp_rbiotransform", needs="executable", program="grasp")
def _handle_run_rbiotransform(arguments: dict[str, Any]) -> dict[str, Any]:
    return _run_grasp_exe(
        "rbiotransform",
        working_dir=arguments["working_dir"],
        stdin_lines=_rbiotransform_input(
            initial=arguments["initial"],
            final=arguments["final"],
            ci_mixing=arguments.get("ci_mixing", False),
            all_symmetries=arguments.get("all_symmetries", True),
            default_settings=arguments.get("default_settings", True),
        ),
    )


@_tool("run_grasp_rtransition", needs="executable", program="grasp")
def _handle_run_rtransition(arguments: dict[str, Any]) -> dict[str, Any]:
    return _run_grasp_exe(
        "rtransition",
        working_dir=arguments["working_dir"],
        stdin_lines=_rtransition_input(
            initial=arguments["initial"],
            final=arguments["final"],
            transition_types=arguments.get("transition_types", "E1"),
            ci_mixing=arguments.get("ci_mixing", False),
            default_settings=arguments.get("default_settings", True),
        ),
    )


@_tool("run_grasp_ris4", needs="executable", program="grasp")
def _handle_run_ris4(arguments: dict[str, Any]) -> dict[str, Any]:
    return _run_grasp_exe(
        "ris4",
        working_dir=arguments["working_dir"],
        stdin_lines=_ris4_input(
            name=arguments["name"],
            ci_mixing=arguments.get("ci_mixing", False),
            higher_order_field_shift=arguments.get("higher_order_field_shift", False),
            save_angular=arguments.get("save_angular", False),
            default_settings=arguments.get("default_settings", True),
        ),
    )


@_tool("run_grasp_rlevels", needs="executable", program="grasp")
def _handle_run_rlevels(arguments: dict[str, Any]) -> dict[str, Any]:
    """Run rlevels and return the parsed energy-level table."""
    files = arguments["files"] if isinstance(arguments["files"], list) else [arguments["files"]]
    result = _run_grasp_exe(
        "rlevels",
        working_dir=arguments["working_dir"],
        stdin_lines="",
        args=files,
    )
    if result["ok"]:
        result["levels"] = _parse_rlevels(result["stdout"])
    return result


@_tool("run_grasp_hf", needs="executable", program="grasp")
def _handle_run_hf(arguments: dict[str, Any]) -> dict[str, Any]:
    """Run the non-relativistic Hartree-Fock code that ships with GRASP.

    Used as a starting-orbital generator for high-Z atoms where the
    Thomas-Fermi guess fails. Follow with rwfnmchfmcdf to convert.
    """
    return _run_grasp_exe(
        "hf",
        working_dir=arguments["working_dir"],
        stdin_lines=_hf_input(
            element_av_z=arguments["element_av_z"],
            orbital_list=arguments["orbital_list"],
            open_shell=arguments["open_shell"],
            estimate_orbitals=arguments.get("estimate_orbitals", "ALL"),
            full_breit=arguments.get("full_breit", True),
            relativistic_corrections=arguments.get("relativistic_corrections", True),
            qed_corrections=arguments.get("qed_corrections", False),
            finite_nucleus=arguments.get("finite_nucleus", False),
        ),
        timeout_seconds=arguments.get("timeout_seconds", 300.0),
    )


@_tool("run_grasp_rwfnmchfmcdf", needs="executable", program="grasp")
def _handle_run_rwfnmchfmcdf(arguments: dict[str, Any]) -> dict[str, Any]:
    """Convert hf wfn.inp → grasp rwfn.out. No prompts."""
    return _run_grasp_exe(
        "rwfnmchfmcdf",
        working_dir=arguments["working_dir"],
        stdin_lines=_rwfnmchfmcdf_input(),
    )


@_tool("run_grasp_rci", needs="executable", program="grasp")
def _handle_run_rci(arguments: dict[str, Any]) -> dict[str, Any]:
    """Relativistic CI with Breit + QED corrections on top of rmcdhf."""
    return _run_grasp_exe(
        "rci",
        working_dir=arguments["working_dir"],
        stdin_lines=_rci_input(
            name=arguments["name"],
            transverse=arguments.get("transverse", True),
            photon_freq_scale=arguments.get("photon_freq_scale", 1e-6),
            vacuum_polarization=arguments.get("vacuum_polarization", True),
            normal_mass_shift=arguments.get("normal_mass_shift", False),
            specific_mass_shift=arguments.get("specific_mass_shift", False),
            self_energy=arguments.get("self_energy", True),
            max_n_self_energy=arguments.get("max_n_self_energy", 3),
            block_level_selections=arguments.get("block_level_selections"),
            default_settings=arguments.get("default_settings", True),
        ),
        timeout_seconds=arguments.get("timeout_seconds", 600.0),
    )


# =============================================================================
# Workflow planners (no execution — analysis-safe)
# =============================================================================

@_tool("plan_grasp_dhf_workflow", program="grasp")
def _handle_plan_dhf(arguments: dict[str, Any]) -> dict[str, Any]:
    return _plan_dhf_workflow(**arguments)


@_tool("plan_grasp_nonrel_limit_workflow", program="grasp")
def _handle_plan_nonrel(arguments: dict[str, Any]) -> dict[str, Any]:
    return _plan_nonrel_limit_workflow(**arguments)


@_tool("plan_grasp_restart_from_workflow", program="grasp")
def _handle_plan_restart(arguments: dict[str, Any]) -> dict[str, Any]:
    return _plan_restart_from_workflow(**arguments)


@_tool("plan_grasp_hf_bootstrap_workflow", program="grasp")
def _handle_plan_hf_bootstrap(arguments: dict[str, Any]) -> dict[str, Any]:
    return _plan_hf_bootstrap_workflow(**arguments)


# =============================================================================
# Workflow orchestrator (executes a planned workflow end-to-end)
# =============================================================================

@_tool("run_grasp_workflow", needs="executable", program="grasp")
def _handle_run_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a workflow plan (from any plan_*) end-to-end. Local mode only."""
    plan = arguments["plan"]
    return _run_workflow(
        plan,
        working_dir=arguments["working_dir"],
        stop_on_failure=arguments.get("stop_on_failure", True),
        timeout_per_step=arguments.get("timeout_per_step", 600.0),
    )


# =============================================================================
# Parsers (analysis-safe)
# =============================================================================

@_tool("parse_grasp_levels", program="grasp")
def _handle_parse_levels(arguments: dict[str, Any]) -> dict[str, Any]:
    """Parse rlevels stdout (or a saved file) into a structured level table."""
    return _parse_rlevels(arguments["text_or_path"])


@_tool("summarize_grasp_terms", program="grasp")
def _handle_summarize_terms(arguments: dict[str, Any]) -> dict[str, Any]:
    """Group rlevels output by LSJ term, report per-term J values + spread."""
    parsed = _parse_rlevels(arguments["text_or_path"])
    return {**parsed, "term_summary": _summarize_terms(parsed)}


@_tool("compare_grasp_levels", program="grasp")
def _handle_compare_levels(arguments: dict[str, Any]) -> dict[str, Any]:
    """Compare two rlevels parses (e.g. relativistic vs non-rel limit)."""
    a = _parse_rlevels(arguments["a"])
    b = _parse_rlevels(arguments["b"])
    return _compare_rlevels(
        a, b,
        label_a=arguments.get("label_a", "A"),
        label_b=arguments.get("label_b", "B"),
    )


@_tool("parse_grasp_lsjlbl", program="grasp")
def _handle_parse_lsjlbl(arguments: dict[str, Any]) -> dict[str, Any]:
    return _parse_lsjlbl(arguments["path"])


@_tool("parse_grasp_sum", program="grasp")
def _handle_parse_sum(arguments: dict[str, Any]) -> dict[str, Any]:
    return _parse_sum(arguments["path"])


@_tool("parse_grasp_rmcdhf_log", program="grasp")
def _handle_parse_rmcdhf_log(arguments: dict[str, Any]) -> dict[str, Any]:
    return _parse_rmcdhf_log(arguments["path"])


# =============================================================================
# Diagnosis / recovery tools (analysis-safe)
# =============================================================================

@_tool("analyze_grasp_case", program="grasp")
def _handle_analyze_grasp_case(arguments: dict[str, Any]) -> dict[str, Any]:
    return _analyze_grasp_case(arguments["working_dir"])


@_tool("summarize_grasp_runs", program="grasp")
def _handle_summarize_grasp_runs(arguments: dict[str, Any]) -> dict[str, Any]:
    from chemtools.programs.grasp.strategy.triage import summarize_grasp_runs
    return summarize_grasp_runs(
        arguments["path"],
        recursive=arguments.get("recursive", False),
        limit=arguments.get("limit"),
    )


@_tool("parse_grasp_hfs", program="grasp")
def _handle_parse_grasp_hfs(arguments: dict[str, Any]) -> dict[str, Any]:
    return _parse_hfs(arguments["path"])


@_tool("parse_grasp_ris", program="grasp")
def _handle_parse_grasp_ris(arguments: dict[str, Any]) -> dict[str, Any]:
    return _parse_ris(arguments["path"])


@_tool("parse_grasp_transition", program="grasp")
def _handle_parse_grasp_transition(arguments: dict[str, Any]) -> dict[str, Any]:
    return _parse_transition(arguments["path"])


@_tool("suggest_grasp_recovery", program="grasp")
def _handle_suggest_grasp_recovery(arguments: dict[str, Any]) -> dict[str, Any]:
    return _suggest_grasp_recovery(
        working_dir=arguments.get("working_dir"),
        error_text=arguments.get("error_text"),
    )


# =============================================================================
# Documentation tools (analysis-safe)
# =============================================================================

@_tool("list_grasp_docs", program="grasp")
def _handle_list_grasp_docs(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"docs": _list_grasp_docs()}


@_tool("search_grasp_docs", program="grasp")
def _handle_search_grasp_docs(arguments: dict[str, Any]) -> dict[str, Any]:
    return _search_grasp_docs(
        arguments["query"],
        max_hits=arguments.get("max_hits", 8),
        context_lines=arguments.get("context_lines", 2),
    )


@_tool("lookup_grasp_section", program="grasp")
def _handle_lookup_grasp_section(arguments: dict[str, Any]) -> dict[str, Any]:
    return _lookup_grasp_section(
        arguments["section"],
        max_results=arguments.get("max_results", 5),
    )


@_tool("read_grasp_doc_excerpt", program="grasp")
def _handle_read_grasp_doc_excerpt(arguments: dict[str, Any]) -> dict[str, Any]:
    return _read_grasp_doc_excerpt(
        arguments["name"],
        start_line=arguments.get("start_line", 1),
        end_line=arguments.get("end_line"),
    )


@_tool("get_grasp_topic_guide", program="grasp")
def _handle_get_grasp_topic_guide(arguments: dict[str, Any]) -> dict[str, Any]:
    topic = arguments.get("topic")
    if topic is None:
        return {"available_topics": _list_grasp_topics()}
    return _get_grasp_topic_guide(topic)


# =============================================================================
# Session log + container introspection
# =============================================================================

@_tool("get_grasp_container", program="grasp")
def _handle_get_container(arguments: dict[str, Any]) -> dict[str, Any]:
    """Show the resolved container path and whether the file exists."""
    path = _resolve_container()
    return {
        "container_path": path,
        "exists": _container_available(),
        "env_var": "CHEMTOOLS_GRASP_CONTAINER",
        "default": "~/mycontainers/grasp2018.sif",
    }


@_tool("read_grasp_session_log", program="grasp")
def _handle_read_session(arguments: dict[str, Any]) -> dict[str, Any]:
    return _read_session_log(arguments["working_dir"])


@_tool("append_grasp_session_note", needs="executable", program="grasp")
def _handle_append_note(arguments: dict[str, Any]) -> dict[str, Any]:
    return _append_session_note(
        arguments["working_dir"],
        arguments["note"],
        title=arguments.get("title"),
    )


# =============================================================================
# Scheduler runner handlers (HPC / local)
# =============================================================================

@_tool("launch_grasp_workflow_run", needs="executable", program="grasp")
def _handle_launch_grasp_workflow_run(arguments: dict[str, Any]) -> dict[str, Any]:
    return _launch_grasp_workflow_run(
        workflow_script_path=arguments["workflow_script_path"],
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path"),
        job_name=arguments.get("job_name"),
        resource_overrides=arguments.get("resource_overrides"),
        env_overrides=arguments.get("env_overrides"),
        write_script=arguments.get("write_script", True),
        dry_run=arguments.get("dry_run", False),
    )


@_tool("get_grasp_run_status", needs="executable", program="grasp")
def _handle_get_grasp_run_status(arguments: dict[str, Any]) -> dict[str, Any]:
    return _get_grasp_run_status(
        output_path=arguments.get("output_file"),
        input_path=arguments.get("input_file"),
        error_path=arguments.get("error_file"),
        process_id=arguments.get("process_id"),
        profile=arguments.get("profile"),
        job_id=arguments.get("job_id"),
        profiles_path=arguments.get("profiles_path"),
    )


@_tool("watch_grasp_run", needs="executable", program="grasp")
def _handle_watch_grasp_run(arguments: dict[str, Any]) -> dict[str, Any]:
    return _watch_grasp_run(
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


@_tool("terminate_grasp_run", needs="executable", program="grasp")
def _handle_terminate_grasp_run(arguments: dict[str, Any]) -> dict[str, Any]:
    import os
    profiles_path = arguments.get("profiles_path") or os.environ.get("CHEMTOOLS_RUNNER_PROFILES")
    return _terminate_grasp_run(
        job_id=arguments["job_id"],
        profile=arguments["profile"],
        profiles_path=profiles_path,
    )


# =============================================================================
# Tool definitions (JSONSchema)
# =============================================================================

_DEFS: list[dict[str, Any]] = [
    # ----- Per-exe runners ---------------------------------------------------
    {
        "name": "run_grasp_exe",
        "description": (
            "Run any GRASP executable in the apptainer container. Generic "
            "escape-hatch — prefer the typed run_grasp_<exe> tools when available."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "exe": {"type": "string", "description": "Executable name (e.g. rmcdhf)"},
                "working_dir": {"type": "string"},
                "stdin_lines": {"type": ["string", "array"], "items": {"type": "string"}},
                "args": {"type": "array", "items": {"type": "string"}},
                "timeout_seconds": {"type": "number", "default": 600},
                "capture_log_file": {"type": "string"},
                "log_to_session": {"type": "boolean", "default": True},
            },
            "required": ["exe", "working_dir"],
        },
    },
    {
        "name": "run_grasp_rnucleus",
        "description": "Build nuclear data (writes isodata file). Z + A required.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "z": {"type": "integer"},
                "a": {"type": "integer", "description": "Mass number; 0 = point nucleus"},
                "point_source": {"type": "boolean", "default": False},
                "revise_radius": {"type": "boolean", "default": False},
                "nuclear_mass_amu": {"type": "number"},
                "nuclear_spin": {"type": "number", "default": 0},
                "dipole_moment": {"type": "number", "default": 0},
                "quadrupole_moment": {"type": "number", "default": 0},
            },
            "required": ["working_dir", "z", "a"],
        },
    },
    {
        "name": "run_grasp_rcsfgenerate",
        "description": (
            "Generate CSF list. Configurations use spectroscopic notation with "
            "(occ,inactive/active/min) marker, e.g. 1s(2,i)2p(1,*)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "core": {
                    "type": "integer", "default": 0,
                    "description": "0=none, 1=He, 2=Ne, 3=Ar, 4=Kr, 5=Xe, 6=Rn",
                },
                "configurations": {"type": "array", "items": {"type": "string"}},
                "active_orbitals": {"type": "string", "description": "e.g. '7s,6p,5d,5f'"},
                "twoj_min": {"type": "integer"},
                "twoj_max": {"type": "integer"},
                "excitations": {
                    "type": "integer", "default": 0,
                    "description": "0=none; negative N = always doubly occupied",
                },
                "ordering": {"type": "string", "default": "*"},
                "copy_to_inp": {
                    "type": "boolean", "default": True,
                    "description": "Auto-copy rcsf.out → rcsf.inp for the next step",
                },
            },
            "required": ["working_dir", "configurations", "active_orbitals", "twoj_min", "twoj_max"],
        },
    },
    {
        "name": "run_grasp_rangular",
        "description": "Angular integration. Reads rcsf.inp + isodata; writes mcp.30..mcp.39.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "default_settings": {"type": "boolean", "default": True},
            },
            "required": ["working_dir"],
        },
    },
    {
        "name": "run_grasp_rwfnestimate",
        "description": (
            "Generate initial radial orbitals. sources is the source-number sequence "
            "to try: '1' (file), '2' (Thomas-Fermi), '3' (screened hydrogenic), "
            "or 'file:<path>' as syntactic sugar for '1' + path."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "default_settings": {"type": "boolean", "default": True},
                "speed_of_light_au": {
                    "type": "number",
                    "description": "Override c (default 137.036). Set 2000 for non-rel limit.",
                },
                "sources": {
                    "type": "array", "items": {"type": "string"},
                    "default": ["2"],
                    "description": "Source-number sequence (e.g. ['2'] for Thomas-Fermi)",
                },
            },
            "required": ["working_dir"],
        },
    },
    {
        "name": "run_grasp_rmcdhf",
        "description": (
            "Self-consistent Dirac-Fock SCF. block_level_selections has one entry "
            "per block (e.g. ['1', '1-2'] for 2 blocks; first picks level 1 only, "
            "second picks levels 1-2)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "default_settings": {"type": "boolean", "default": True},
                "speed_of_light_au": {"type": "number"},
                "block_level_selections": {"type": "array", "items": {"type": "string"}},
                "orbitals_to_optimize": {"type": "string", "default": "*"},
                "weighting": {
                    "type": "string", "default": "5",
                    "description": "5 = stat weight 2J+1, 1 = equal, etc.",
                },
                "spectroscopic_orbitals": {"type": "string", "default": "*"},
                "max_scf_cycles": {"type": "integer", "default": 100},
                "timeout_seconds": {"type": "number", "default": 600},
            },
            "required": ["working_dir", "block_level_selections"],
        },
    },
    {
        "name": "run_grasp_rsave",
        "description": "Save converged results with the given name prefix (writes name.{w,c,m,sum,alog,log}).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["working_dir", "name"],
        },
    },
    {
        "name": "run_grasp_jj2lsj",
        "description": "Transform jj-coupled CSFs to LSJ representation. Writes name.lsj.lbl.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "name": {"type": "string"},
                "mixing_coefficients": {
                    "type": "boolean", "default": False,
                    "description": "Set True if a CI-mixed name.cm file is present",
                },
                "unique_labeling": {"type": "boolean", "default": True},
                "default_settings": {"type": "boolean", "default": True},
            },
            "required": ["working_dir", "name"],
        },
    },
    {
        "name": "run_grasp_rhfs",
        "description": (
            "Run rhfs (hyperfine structure): computes magnetic-dipole A(MHz) + "
            "electric-quadrupole B(MHz) constants and Landé g_J from isodata "
            "(nuclear spin + moments) + name.c/.w/.(c)m. Writes name.(c)h. "
            "Needs a nucleus with spin I!=0 (set via rnucleus); A,B vanish for I=0."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "name": {"type": "string"},
                "ci_mixing": {
                    "type": "boolean", "default": False,
                    "description": "Set True to read CI-mixed name.cm (from rci) instead of name.m",
                },
                "default_settings": {"type": "boolean", "default": True},
            },
            "required": ["working_dir", "name"],
        },
    },
    {
        "name": "run_grasp_rhfs_lsj",
        "description": (
            "Run rhfs_lsj: relabel rhfs output (name.(c)h) with LSJ terms + level "
            "energies, energy-sortable. Writes name.(c)hlsj. Run rhfs first."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "name": {"type": "string"},
                "ci_mixing": {"type": "boolean", "default": False},
                "energy_sorted": {"type": "boolean", "default": True},
            },
            "required": ["working_dir", "name"],
        },
    },
    {
        "name": "run_grasp_ris4",
        "description": (
            "Run ris4 (isotope shift): computes the electronic normal + specific "
            "mass-shift parameters and the electron density at the nucleus "
            "(first-order field-shift factor) per level. Writes name.i. Factors are "
            "isotope-independent, so any isotope (including spin-0 like Th-232) works."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "name": {"type": "string"},
                "ci_mixing": {"type": "boolean", "default": False,
                              "description": "Set True to read CI-mixed name.cm (from rci)"},
                "higher_order_field_shift": {"type": "boolean", "default": False},
                "save_angular": {"type": "boolean", "default": False},
                "default_settings": {"type": "boolean", "default": True},
            },
            "required": ["working_dir", "name"],
        },
    },
    {
        "name": "run_grasp_rbiotransform",
        "description": (
            "Run rbiotransform: biorthogonalise two states' wavefunctions so "
            "standard tensor algebra applies. Required before run_grasp_rtransition. "
            "Reads initial/final name.c/.w/.(c)m; writes name.bw + name.(c)bm."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "initial": {"type": "string", "description": "Initial state name"},
                "final": {"type": "string", "description": "Final state name"},
                "ci_mixing": {"type": "boolean", "default": False},
                "all_symmetries": {"type": "boolean", "default": True},
                "default_settings": {"type": "boolean", "default": True},
            },
            "required": ["working_dir", "initial", "final"],
        },
    },
    {
        "name": "run_grasp_rtransition",
        "description": (
            "Run rtransition: radiative-transition parameters (line strength S, "
            "oscillator strength gf, rate A_ki, lifetimes) between two states. Run "
            "run_grasp_rbiotransform on the same pair first. transition_types is a "
            "GRASP spec, e.g. 'E1' or 'E1,M2'. Writes name1.name2.(c)t(.lsj)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "initial": {"type": "string"},
                "final": {"type": "string"},
                "transition_types": {"type": "string", "default": "E1",
                                     "description": "e.g. 'E1', 'E1,M2', 'E2'"},
                "ci_mixing": {"type": "boolean", "default": False},
                "default_settings": {"type": "boolean", "default": True},
            },
            "required": ["working_dir", "initial", "final"],
        },
    },
    {
        "name": "run_grasp_rlevels",
        "description": (
            "Run rlevels and parse the energy-level table. Pass one or more *.m or "
            "*.cm files. Returns the parsed levels along with raw stdout."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "files": {
                    "type": ["array", "string"], "items": {"type": "string"},
                    "description": "e.g. '5f10.m' or ['2s_3.m', '2p_3.m']",
                },
            },
            "required": ["working_dir", "files"],
        },
    },
    {
        "name": "run_grasp_hf",
        "description": (
            "Run the non-relativistic HF code (ships with GRASP). Used as a "
            "starting-orbital generator for high-Z atoms where Thomas-Fermi fails. "
            "Follow with run_grasp_rwfnmchfmcdf to convert wfn → rwfn."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "element_av_z": {
                    "type": "string",
                    "description": "Element symbol + ',AV,' + Z, e.g. 'Cf,AV,98'",
                },
                "orbital_list": {
                    "type": "string",
                    "description": "Closed orbitals, space-separated, e.g. ' 1s  2s  2p  3s ...'",
                },
                "open_shell": {"type": "string", "description": "e.g. '5f(10)'"},
                "estimate_orbitals": {"type": "string", "default": "ALL"},
                "full_breit": {"type": "boolean", "default": True},
                "relativistic_corrections": {"type": "boolean", "default": True},
                "qed_corrections": {"type": "boolean", "default": False},
                "finite_nucleus": {"type": "boolean", "default": False},
                "timeout_seconds": {"type": "number", "default": 300},
            },
            "required": ["working_dir", "element_av_z", "orbital_list", "open_shell"],
        },
    },
    {
        "name": "run_grasp_rwfnmchfmcdf",
        "description": "Convert hf wfn.inp → grasp rwfn.out. No prompts; reads/writes files only.",
        "inputSchema": {
            "type": "object",
            "properties": {"working_dir": {"type": "string"}},
            "required": ["working_dir"],
        },
    },
    {
        "name": "run_grasp_rci",
        "description": (
            "Relativistic CI with Breit + QED on top of rmcdhf. Adds vacuum "
            "polarization, self-energy, and the transverse-photon (Breit) "
            "interaction at the configured photon frequency."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "name": {"type": "string", "description": "Same prefix used in rsave"},
                "transverse": {"type": "boolean", "default": True},
                "photon_freq_scale": {"type": "number", "default": 1e-6},
                "vacuum_polarization": {"type": "boolean", "default": True},
                "normal_mass_shift": {"type": "boolean", "default": False},
                "specific_mass_shift": {"type": "boolean", "default": False},
                "self_energy": {"type": "boolean", "default": True},
                "max_n_self_energy": {"type": "integer", "default": 3},
                "block_level_selections": {"type": "array", "items": {"type": "string"}},
                "default_settings": {"type": "boolean", "default": True},
                "timeout_seconds": {"type": "number", "default": 600},
            },
            "required": ["working_dir", "name"],
        },
    },
    # ----- Workflow planners (analysis-safe) ---------------------------------
    {
        "name": "plan_grasp_dhf_workflow",
        "description": (
            "Plan a full DHF workflow (rnucleus → rcsfgenerate → rangular → "
            "rwfnestimate → rmcdhf → rsave → jj2lsj → rlevels). Returns the "
            "ordered step list with stdin heredocs ready for run_grasp_workflow."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "z": {"type": "integer"}, "a": {"type": "integer"},
                "nuclear_mass_amu": {"type": "number"},
                "nuclear_spin": {"type": "number", "default": 0},
                "dipole_moment": {"type": "number", "default": 0},
                "quadrupole_moment": {"type": "number", "default": 0},
                "core": {"type": "integer", "default": 0},
                "configurations": {"type": "array", "items": {"type": "string"}},
                "active_orbitals": {"type": "string"},
                "twoj_min": {"type": "integer"}, "twoj_max": {"type": "integer"},
                "excitations": {"type": "integer", "default": 0},
                "block_level_selections": {"type": "array", "items": {"type": "string"}},
                "orbitals_to_optimize": {"type": "string", "default": "*"},
                "weighting": {"type": "string", "default": "5"},
                "spectroscopic_orbitals": {"type": "string", "default": "*"},
                "max_scf_cycles": {"type": "integer", "default": 100},
                "name": {"type": "string"},
                "speed_of_light_au": {"type": "number"},
                "rwfnestimate_sources": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["z", "a", "configurations", "active_orbitals",
                         "twoj_min", "twoj_max", "block_level_selections", "name"],
        },
    },
    {
        "name": "plan_grasp_nonrel_limit_workflow",
        "description": (
            "Plan a DHF workflow with c set to a large value (default 2000 au) "
            "to access the nonrelativistic limit. All other arguments same as "
            "plan_grasp_dhf_workflow."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "speed_of_light_au": {"type": "number", "default": 2000},
                "z": {"type": "integer"}, "a": {"type": "integer"},
                "nuclear_mass_amu": {"type": "number"},
                "nuclear_spin": {"type": "number", "default": 0},
                "dipole_moment": {"type": "number", "default": 0},
                "quadrupole_moment": {"type": "number", "default": 0},
                "core": {"type": "integer", "default": 0},
                "configurations": {"type": "array", "items": {"type": "string"}},
                "active_orbitals": {"type": "string"},
                "twoj_min": {"type": "integer"}, "twoj_max": {"type": "integer"},
                "excitations": {"type": "integer", "default": 0},
                "block_level_selections": {"type": "array", "items": {"type": "string"}},
                "orbitals_to_optimize": {"type": "string", "default": "*"},
                "weighting": {"type": "string", "default": "5"},
                "spectroscopic_orbitals": {"type": "string", "default": "*"},
                "max_scf_cycles": {"type": "integer", "default": 100},
                "name": {"type": "string"},
            },
            "required": ["z", "a", "configurations", "active_orbitals",
                         "twoj_min", "twoj_max", "block_level_selections", "name"],
        },
    },
    {
        "name": "plan_grasp_restart_from_workflow",
        "description": (
            "Plan a DHF workflow that uses a previous run's *.w as starting "
            "orbitals. Useful for n-shell expansion: optimize n=3 first, then "
            "restart n=4 by reusing the n=3 orbitals."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "previous_w_file": {
                    "type": "string",
                    "description": "Absolute path to the previous run's *.w file",
                },
                "z": {"type": "integer"}, "a": {"type": "integer"},
                "nuclear_mass_amu": {"type": "number"},
                "nuclear_spin": {"type": "number", "default": 0},
                "dipole_moment": {"type": "number", "default": 0},
                "quadrupole_moment": {"type": "number", "default": 0},
                "core": {"type": "integer", "default": 0},
                "configurations": {"type": "array", "items": {"type": "string"}},
                "active_orbitals": {"type": "string"},
                "twoj_min": {"type": "integer"}, "twoj_max": {"type": "integer"},
                "excitations": {"type": "integer", "default": 0},
                "block_level_selections": {"type": "array", "items": {"type": "string"}},
                "orbitals_to_optimize": {"type": "string", "default": "*"},
                "weighting": {"type": "string", "default": "5"},
                "spectroscopic_orbitals": {"type": "string", "default": "*"},
                "max_scf_cycles": {"type": "integer", "default": 100},
                "name": {"type": "string"},
            },
            "required": ["previous_w_file", "z", "a", "configurations",
                         "active_orbitals", "twoj_min", "twoj_max",
                         "block_level_selections", "name"],
        },
    },
    {
        "name": "plan_grasp_hf_bootstrap_workflow",
        "description": (
            "Plan: hf (non-rel) → rwfnmchfmcdf → DHF using hf orbitals as guess. "
            "Required for high-Z atoms (Z≥80) where Thomas-Fermi diverges. "
            "Adds 2 extra steps before rwfnestimate."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "z": {"type": "integer"}, "a": {"type": "integer"},
                "element_symbol": {"type": "string", "description": "e.g. 'Cf'"},
                "hf_orbital_list": {
                    "type": "string",
                    "description": "Space-separated closed orbitals for hf (e.g. ' 1s  2s  2p  3s ...')",
                },
                "hf_open_shell": {"type": "string", "description": "e.g. '5f(10)'"},
                "nuclear_mass_amu": {"type": "number"},
                "nuclear_spin": {"type": "number", "default": 0},
                "dipole_moment": {"type": "number", "default": 0},
                "quadrupole_moment": {"type": "number", "default": 0},
                "core": {"type": "integer", "default": 0},
                "configurations": {"type": "array", "items": {"type": "string"}},
                "active_orbitals": {"type": "string"},
                "twoj_min": {"type": "integer"}, "twoj_max": {"type": "integer"},
                "excitations": {"type": "integer", "default": 0},
                "block_level_selections": {"type": "array", "items": {"type": "string"}},
                "orbitals_to_optimize": {"type": "string", "default": "*"},
                "weighting": {"type": "string", "default": "5"},
                "spectroscopic_orbitals": {"type": "string", "default": "*"},
                "max_scf_cycles": {"type": "integer", "default": 100},
                "name": {"type": "string"},
            },
            "required": ["z", "a", "element_symbol", "hf_orbital_list", "hf_open_shell",
                         "configurations", "active_orbitals", "twoj_min", "twoj_max",
                         "block_level_selections", "name"],
        },
    },
    {
        "name": "run_grasp_workflow",
        "description": (
            "Execute a workflow plan end-to-end. Each step writes to grasp_session.md. "
            "Stops on first failure unless stop_on_failure=False."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan": {"type": "object", "description": "From any plan_grasp_*"},
                "working_dir": {"type": "string"},
                "stop_on_failure": {"type": "boolean", "default": True},
                "timeout_per_step": {"type": "number", "default": 600},
            },
            "required": ["plan", "working_dir"],
        },
    },
    # ----- Parsers + analysis -----------------------------------------------
    {
        "name": "parse_grasp_levels",
        "description": (
            "Parse rlevels stdout (or a saved file containing it). Returns each "
            "level: no, pos, j, parity, energy_hartree, energy_cm1, splitting_cm1, "
            "configuration. Plus rydberg_constant + ground_state_au summary."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text_or_path": {"type": "string", "description": "Raw rlevels text or a file path"},
            },
            "required": ["text_or_path"],
        },
    },
    {
        "name": "summarize_grasp_terms",
        "description": (
            "Parse rlevels output and group levels by their LSJ term label. "
            "Reports per-term J values, parity, and energy spread within multiplet."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"text_or_path": {"type": "string"}},
            "required": ["text_or_path"],
        },
    },
    {
        "name": "compare_grasp_levels",
        "description": (
            "Compare two rlevels parses pairwise. Useful for relativistic vs "
            "non-rel-limit comparisons or before/after-CI shift analysis."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "string", "description": "First rlevels stdout/file"},
                "b": {"type": "string", "description": "Second rlevels stdout/file"},
                "label_a": {"type": "string", "default": "A"},
                "label_b": {"type": "string", "default": "B"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "parse_grasp_lsjlbl",
        "description": "Parse a name.lsj.lbl file into per-level LSJ-component compositions.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "parse_grasp_sum",
        "description": (
            "Parse a name.sum (rmcdhf summary) file: nuclear params, speed of "
            "light, radial grid, per-subshell wfn summary, eigenenergies."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "parse_grasp_rmcdhf_log",
        "description": "Parse the rmcdhf.log file: per-iteration mean energy + convergence flag.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    # ----- Diagnosis / recovery ---------------------------------------------
    {
        "name": "analyze_grasp_case",
        "description": (
            "Inspect a GRASP working directory and report status of each "
            "workflow step (rnucleus, rcsfgenerate, rangular, rwfnestimate, "
            "rmcdhf, ...). Returns verdict (healthy/partial/failed/"
            "not_started) + per-step artifact audit + SCF convergence "
            "summary + issues + next_actions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"working_dir": {"type": "string"}},
            "required": ["working_dir"],
        },
    },
    {
        "name": "summarize_grasp_runs",
        "description": (
            "Triage MANY GRASP working directories in one call (the GRASP unit "
            "is a directory per atom/term, not a single file). Give a parent "
            "directory; returns one compact row per run (element, Z, n_csfs, "
            "non-rel-limit flag, ground energy + term, level count, max "
            "splitting, verdict) plus roll-up counts by verdict and element. "
            "Use this for atom/term screens; drill into a flagged run with "
            "analyze_grasp_case afterward."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "A GRASP working dir, or a parent containing several."},
                "recursive": {"type": "boolean", "default": False, "description": "Recurse into nested subdirectories."},
                "limit": {"type": "integer", "description": "Cap runs processed (response flags truncation)."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "parse_grasp_hfs",
        "description": (
            "Parse hyperfine output from rhfs (name.(c)h) or rhfs_lsj (name.(c)hlsj): "
            "nuclear spin + dipole/quadrupole moments, and per-level A(MHz), B(MHz), "
            "Landé g_J (LSJ label + energy when from rhfs_lsj)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to a .h/.ch or .hlsj/.chlsj file."}},
            "required": ["path"],
        },
    },
    {
        "name": "parse_grasp_ris",
        "description": (
            "Parse isotope-shift output from ris4 (name.i): per-level normal + "
            "specific mass-shift parameters (<K^1>, <K^2+K^3>, total) and the "
            "electron density at the nucleus (first-order field-shift factor)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to a ris4 .i file."}},
            "required": ["path"],
        },
    },
    {
        "name": "parse_grasp_transition",
        "description": (
            "Parse radiative-transition output from rtransition (name1.name2.(c)t.lsj): "
            "per-transition lower/upper state (label + energy), energy in cm-1, vacuum "
            "+ air wavelengths, multipole type, and length + velocity gauge line "
            "strength / gf / A_ki (s-1) / dT gauge agreement."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to a .t.lsj / .ct.lsj file."}},
            "required": ["path"],
        },
    },
    {
        "name": "suggest_grasp_recovery",
        "description": (
            "Classify a GRASP failure and suggest recovery. Provide either "
            "`working_dir` (auto-reads session log + rmcdhf output) or "
            "`error_text` (raw stderr/stdout chunk). Recognized failure "
            "classes: tfwave_divergence (→ hf-bootstrap), "
            "block_level_mismatch (→ fix block selections), "
            "premature_eof (→ stdin missing prompts), missing_input_file, "
            "orbital_divergence (→ non-rel warm-up), max_iter_exhausted."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "error_text": {"type": "string"},
            },
        },
    },
    # ----- Documentation tools ----------------------------------------------
    {
        "name": "list_grasp_docs",
        "description": "List the 15 bundled GRASP2018 manual files (4 parts: overview, CSF generation, sample runs, convergence).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_grasp_docs",
        "description": (
            "Full-text search across the bundled GRASP2018 manual. Returns "
            "up to max_hits matches with surrounding context lines."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_hits": {"type": "integer", "default": 8},
                "context_lines": {"type": "integer", "default": 2},
            },
            "required": ["query"],
        },
    },
    {
        "name": "lookup_grasp_section",
        "description": (
            "Look up a GRASP exe / section / keyword (e.g. 'rmcdhf', "
            "'rcsfgenerate', 'Breit', 'convergence', 'csf'). Returns top "
            "doc files most likely to document it (filename + heading match)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "section": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["section"],
        },
    },
    {
        "name": "read_grasp_doc_excerpt",
        "description": (
            "Read a slice of a bundled GRASP doc. `name` is the relative "
            "path returned by list_grasp_docs (e.g. "
            "'part_iii_sample_runs/01_Running_the_application_programs.md')."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "start_line": {"type": "integer", "default": 1},
                "end_line": {"type": "integer"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_grasp_topic_guide",
        "description": (
            "Curated cheatsheet for high-value GRASP topics. Pass `topic` "
            "as one of: 'csf_generation', 'convergence_debugging', "
            "'nonrel_limit', 'hf_bootstrap', 'level_interpretation'. "
            "Omit `topic` to list available topics."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
            },
        },
    },
    # ----- Session log + container ------------------------------------------
    {
        "name": "get_grasp_container",
        "description": "Resolve the GRASP container path (env var or default) and check it exists.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_grasp_session_log",
        "description": "Read the grasp_session.md from a working directory.",
        "inputSchema": {
            "type": "object",
            "properties": {"working_dir": {"type": "string"}},
            "required": ["working_dir"],
        },
    },
    {
        "name": "append_grasp_session_note",
        "description": "Append a free-form note to grasp_session.md (manual debugging trail).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "note": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["working_dir", "note"],
        },
    },
    # ----- Scheduler runner tools (HPC / local) -----
    {
        "name": "launch_grasp_workflow_run",
        "description": (
            "Submit a GRASP workflow shell script to the scheduler defined by a runner "
            "profile. The script is the file the profile's script_template invokes via "
            "`bash {input_file}` and typically chains rnucleus -> rcsfgenerate -> ... -> "
            "rlevels in one apptainer-wrapped pass. Generate the script via "
            "plan_grasp_dhf_workflow + the heredoc input builders before calling this. "
            "Set dry_run=true to preview."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_script_path": {"type": "string", "description": "Path to the GRASP workflow shell script."},
                "profile": {"type": "string"},
                "profiles_path": {"type": "string"},
                "job_name": {"type": "string"},
                "resource_overrides": {"type": "object"},
                "env_overrides": {"type": "object"},
                "write_script": {"type": "boolean", "default": True},
                "dry_run": {"type": "boolean", "default": False},
            },
            "required": ["workflow_script_path", "profile"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_grasp_run_status",
        "description": (
            "Check the status of a GRASP run. For HPC jobs the scheduler job ID is "
            "auto-detected from {job_name}.jobid alongside the workflow script."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "output_file": {"type": "string"},
                "input_file": {"type": "string"},
                "error_file": {"type": "string"},
                "process_id": {"type": "integer"},
                "profile": {"type": "string"},
                "job_id": {"type": "string"},
                "profiles_path": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "watch_grasp_run",
        "description": (
            "Poll GRASP status until terminal state or timeout. For HPC jobs omit "
            "timeout_seconds to block until scheduler completion."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "output_file": {"type": "string"},
                "input_file": {"type": "string"},
                "error_file": {"type": "string"},
                "process_id": {"type": "integer"},
                "profile": {"type": "string"},
                "job_id": {"type": "string"},
                "profiles_path": {"type": "string"},
                "poll_interval_seconds": {"type": "number", "default": 10.0},
                "adaptive_polling": {"type": "boolean", "default": True},
                "max_poll_interval_seconds": {"type": "number", "default": 60.0},
                "timeout_seconds": {"type": ["number", "null"], "default": 3600.0},
                "max_polls": {"type": "integer"},
                "history_limit": {"type": "integer", "default": 8},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "terminate_grasp_run",
        "description": (
            "Cancel a running GRASP scheduler job. Provide job_id + profile "
            "(profile resolves the scancel/qdel/bkill command)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "profile": {"type": "string"},
                "profiles_path": {"type": "string"},
            },
            "required": ["job_id", "profile"],
            "additionalProperties": False,
        },
    },
]

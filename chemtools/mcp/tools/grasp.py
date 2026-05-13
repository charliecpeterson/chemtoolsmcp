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
)
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
]

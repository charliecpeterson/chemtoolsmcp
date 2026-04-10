#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any


# Development fallback: if running directly from the source tree, add repo root to path
# so `chemtools` can be imported without `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parents[2]
if not any("chemtools" in p for p in sys.path):
    sys.path.insert(0, str(_REPO_ROOT))

from chemtools import (  # noqa: E402
    analyze_frontier_orbitals,
    draft_initial_geometry,
    plan_nwchem_workflow,
    suggest_basis_set,
    suggest_memory,
    suggest_spin_state,
    validate_nwchem_tce_setup,
    analyze_imaginary_modes,
    check_nwchem_geometry_plausibility,
    check_nwchem_freq_plausibility,
    check_nwchem_run_status,
    compare_nwchem_runs,
    create_nwchem_input,
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
    draft_nwchem_vectors_swap_input,
    find_restart_assets,
    inspect_input,
    inspect_runner_profiles,
    lint_nwchem_input,
    launch_nwchem_run,
    parse_cube,
    parse_mcscf_output,
    parse_mos,
    parse_nwchem_movecs,
    parse_output,
    parse_population_analysis,
    parse_scf,
    parse_tce_amplitudes,
    parse_tce_output,
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
    terminate_nwchem_run,
    watch_nwchem_run,
)


SERVER_NAME = "chemtools-nwchem"
SERVER_VERSION = "0.1.0"
DEFAULT_PROTOCOL_VERSION = "2024-11-05"

# Basis library: bundled inside the package at chemtools/data/nwchem/basis_library/
# Can be overridden at runtime with CHEMTOOLS_BASIS_LIBRARY env var.
try:
    from importlib.resources import files as _pkg_files
    DEFAULT_BASIS_LIBRARY = Path(str(_pkg_files("chemtools").joinpath("data/nwchem/basis_library")))
except Exception:
    DEFAULT_BASIS_LIBRARY = _REPO_ROOT / "chemtools" / "data" / "nwchem" / "basis_library"
LOG_PATH = os.environ.get("CHEMTOOLS_MCP_LOG")
TRANSPORT_MODE = "content-length"

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
from typing import Callable  # noqa: E402

_TOOL_REGISTRY: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def _tool(name: str) -> Callable:
    """Decorator that registers a handler function under *name*."""
    def decorator(fn: Callable) -> Callable:
        _TOOL_REGISTRY[name] = fn
        return fn
    return decorator


def log_event(message: str) -> None:
    if not LOG_PATH:
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with Path(LOG_PATH).open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def basis_library_path(path: str | None = None) -> str:
    if path:
        return path
    return os.environ.get("CHEMTOOLS_BASIS_LIBRARY", str(DEFAULT_BASIS_LIBRARY))


def tool_definitions() -> list[dict[str, Any]]:
    return [
        # ------------------------------------------------------------------
        # Workflow planning and geometry setup (start here for new jobs)
        # ------------------------------------------------------------------
        {
            "name": "plan_nwchem_workflow",
            "description": (
                "Return a concrete step-by-step tool call plan for a NWChem workflow. "
                "Call this FIRST when starting any new calculation to get the exact sequence "
                "of tools and parameters to use. Eliminates guesswork about workflow order."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "enum": ["opt_freq", "opt_freq_ccsd", "opt_freq_mp2",
                                 "single_point_dft", "single_point_ccsd", "single_point_mp2"],
                        "description": "What you want to compute.",
                    },
                    "elements": {"type": "array", "items": {"type": "string"},
                                 "description": "Element symbols in the molecule, e.g. ['Fe', 'Cl']."},
                    "charge": {"type": "integer"},
                    "multiplicity": {"type": "integer"},
                    "basis": {"type": "string", "description": "Basis set name, e.g. '6-31gs'."},
                    "method": {"type": "string", "default": "ccsd",
                               "description": "TCE method: 'ccsd', 'mp2', or 'ccsd(t)'."},
                    "xc_functional": {"type": "string", "default": "b3lyp",
                                      "description": "DFT exchange-correlation functional for opt/freq step, e.g. 'b3lyp', 'pbe0', 'm06', 'tpss'. Default: 'b3lyp'."},
                    "has_geometry_file": {"type": "boolean", "default": False},
                    "has_dft_output": {"type": "boolean", "default": False},
                    "has_scf_output": {"type": "boolean", "default": False},
                },
                "required": ["goal", "elements", "charge", "multiplicity", "basis"],
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
                "Returns a 'nwchem_directive' string ready to use as the 'memory' parameter "
                "in create_nwchem_input."
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
        # ------------------------------------------------------------------
        # Frontier orbital / vectors-swap workflow
        # ------------------------------------------------------------------
        {
            "name": "analyze_nwchem_frontier_orbitals",
            "description": "Analyze NWChem frontier orbitals and SOMOs to estimate metal-centered vs ligand-centered open-shell character.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_nwchem_vectors_swaps",
            "description": "Suggest NWChem vectors swap operations to move buried metal-centered orbitals into the SOMO window.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                    "vectors_input": {"type": "string"},
                    "vectors_output": {"type": "string"},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_vectors_swap_input",
            "description": "Draft a NWChem restart input that applies explicit vectors swaps to move buried metal-centered orbitals into the SOMO window.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                    "vectors_input": {"type": "string"},
                    "vectors_output": {"type": "string"},
                    "task_operation": {"type": "string", "default": "energy"},
                    "iterations": {"type": "integer", "default": 500},
                    "smear": {"type": "number", "default": 0.001},
                    "convergence_damp": {"type": "integer", "default": 30},
                    "convergence_ncydp": {"type": "integer", "default": 30},
                    "population_print": {"type": "string", "default": "mulliken"},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "title": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["output_file", "input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_property_check_input",
            "description": "Draft a one-step NWChem property input around a chosen movecs file for Mulliken or Lowdin inspection.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "reference_output_file": {"type": "string"},
                    "vectors_input": {"type": "string"},
                    "vectors_output": {"type": "string"},
                    "property_keywords": {"type": "array", "items": {"type": "string"}},
                    "task_strategy": {"type": "string", "enum": ["auto", "property", "energy"], "default": "auto"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                    "iterations": {"type": "integer", "default": 1},
                    "convergence_energy": {"type": "string", "default": "1e3"},
                    "smear": {"type": "number"},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "title": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_scf_stabilization_input",
            "description": "Draft a safer SCF stabilization restart input from an existing movecs path when a state-check or SCF retry still fails.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "reference_output_file": {"type": "string"},
                    "vectors_input": {"type": "string"},
                    "vectors_output": {"type": "string"},
                    "task_operation": {"type": "string", "default": "energy"},
                    "iterations": {"type": "integer"},
                    "smear": {"type": "number"},
                    "convergence_damp": {"type": "integer"},
                    "convergence_ncydp": {"type": "integer"},
                    "population_print": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "title": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_optimization_followup_input",
            "description": "Draft a NWChem follow-up input from the last optimized geometry, either to continue optimization or to run frequency only.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "task_strategy": {
                        "type": "string",
                        "enum": ["auto", "optimize_only", "freq_only", "optimize_then_freq"],
                        "default": "auto",
                    },
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "title": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["output_file", "input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "extract_nwchem_geometry",
            "description": (
                "Extract a geometry from a NWChem optimization output as XYZ and NWChem geometry block text. "
                "For converged or incomplete runs returns the last frame; for failed/diverged runs returns the "
                "lowest-energy frame as the best restart guess. Use frame='best' (default) for automatic smart "
                "selection, or 'last', 'first', 'min_energy', or an integer step number for explicit control."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string", "description": "Optional: used to preserve geometry block header/directives"},
                    "frame": {
                        "type": "string",
                        "description": "'best' (smart selection), 'last', 'first', 'min_energy', or integer step",
                        "default": "best",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_cube_input",
            "description": "Draft a NWChem dplot input for orbital, density, or spin-density cube generation from a chosen movecs file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "vectors_input": {"type": "string"},
                    "orbital_vectors": {"type": "array", "items": {"type": "integer"}},
                    "density_modes": {"type": "array", "items": {"type": "string"}},
                    "orbital_spin": {"type": "string", "default": "total"},
                    "extent_angstrom": {"type": "number", "default": 6.0},
                    "grid_points": {"type": "integer", "default": 120},
                    "gaussian": {"type": "boolean", "default": True},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "title": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["input_file", "vectors_input"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_frontier_cube_input",
            "description": "Draft a NWChem dplot input for SOMO, HOMO, and LUMO cubes inferred from parsed frontier orbitals.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "vectors_input": {"type": "string"},
                    "include_somos": {"type": "boolean", "default": True},
                    "include_homo": {"type": "boolean", "default": True},
                    "include_lumo": {"type": "boolean", "default": True},
                    "include_density_modes": {"type": "array", "items": {"type": "string"}},
                    "extent_angstrom": {"type": "number", "default": 6.0},
                    "grid_points": {"type": "integer", "default": 120},
                    "gaussian": {"type": "boolean", "default": True},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "title": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["output_file", "input_file"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Output parsers
        # ------------------------------------------------------------------
        {
            "name": "parse_nwchem_output",
            "description": "Parse a NWChem output file into structured sections. Defaults to tasks section only. Use sections=['tasks','mos'] for SCF review, add 'freq' for frequency, 'trajectory' for optimization path.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["tasks", "mos", "freq", "mcscf", "population", "trajectory"],
                        },
                        "description": "Sections to parse. Omit for tasks-only (default). Each section adds to response size.",
                    },
                    "top_n": {"type": "integer", "default": 5, "description": "Frontier orbitals to return per spin channel (mos section)."},
                    "include_coefficients": {"type": "boolean", "default": False, "description": "Include MO coefficients (mos section). Only use for small systems (<50 basis functions)."},
                    "include_displacements": {"type": "boolean", "default": False},
                    "include_positions": {"type": "boolean", "default": False},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_nwchem_mos",
            "description": "Parse NWChem molecular orbitals. Returns frontier orbital window (SOMOs + top_n around HOMO/LUMO) by default. Set include_all_orbitals=true only when you need to inspect the full orbital spectrum (e.g. to verify core ordering for TCE freeze count — prefer parse_nwchem_movecs for that instead).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "top_n": {"type": "integer", "default": 5},
                    "include_coefficients": {"type": "boolean", "default": False, "description": "Include MO coefficient vectors. Only use for small systems (<50 basis functions) — large systems produce very large responses."},
                    "include_all_orbitals": {"type": "boolean", "default": False},
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_nwchem_mcscf_output",
            "description": "Parse a NWChem MCSCF output for settings, iteration energies, CI convergence, natural occupations, and active-space Mulliken summaries.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_nwchem_population_analysis",
            "description": "Parse NWChem Mulliken and Lowdin population analysis blocks, including total and spin density tables.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
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
            "name": "parse_nwchem_scf",
            "description": "Parse the NWChem SCF/DFT iteration table and identify the convergence pattern.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Input inspection and linting
        # ------------------------------------------------------------------
        {
            "name": "inspect_nwchem_input",
            "description": "Inspect a NWChem input file for geometry elements, transition metals, charge, multiplicity, and tasks.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "lint_nwchem_input",
            "description": "Lint a NWChem input for task/module consistency, basis/ECP coverage, and movecs output policy issues.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "library_path": {"type": "string"},
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "find_nwchem_restart_assets",
            "description": "Discover restart-relevant files in a job directory, including movecs, db, xyz, cubes, inputs, and outputs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Runner / job management
        # ------------------------------------------------------------------
        {
            "name": "inspect_nwchem_runner_profiles",
            "description": "List available NWChem runner profiles and their launcher kinds.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "profiles_path": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "launch_nwchem_run",
            "description": "Launch a NWChem input through a named runner profile. Use dry_run=true for a preview without executing.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "profile": {"type": "string"},
                    "profiles_path": {"type": "string"},
                    "job_name": {"type": "string"},
                    "resource_overrides": {"type": "object"},
                    "env_overrides": {"type": "object"},
                    "write_script": {"type": "boolean", "default": True},
                    "dry_run": {"type": "boolean", "default": False},
                },
                "required": ["input_file", "profile"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_nwchem_run_status",
            "description": "Check the status of a NWChem run. Returns process/scheduler state, parsed output outcome, and a compact progress summary for running jobs.",
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
            "name": "watch_nwchem_run",
            "description": "Poll NWChem status until the run reaches a terminal state or a timeout/max-poll limit.",
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
                    "timeout_seconds": {"type": "number", "default": 3600.0},
                    "max_polls": {"type": "integer"},
                    "history_limit": {"type": "integer", "default": 8},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "tail_nwchem_output",
            "description": "Return the tail of a NWChem output file for quick inspection. Capped at 10000 characters.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "lines": {"type": "integer", "default": 30},
                    "max_characters": {"type": "integer", "default": 4000, "maximum": 10000, "description": "Maximum characters to return (hard cap: 10000)."},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "terminate_nwchem_run",
            "description": "Send SIGTERM or SIGKILL to a local NWChem process after intervention review has determined the run should stop.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "process_id": {"type": "integer"},
                    "signal_name": {"type": "string", "default": "term"},
                },
                "required": ["process_id"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Run comparison and follow-up
        # ------------------------------------------------------------------
        {
            "name": "compare_nwchem_runs",
            "description": "Compare two NWChem runs by diagnosis, task outcome, and energy change. Optionally writes a follow-up artifact when output_dir is provided.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_output_file": {"type": "string"},
                    "candidate_output_file": {"type": "string"},
                    "reference_input_file": {"type": "string"},
                    "candidate_input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                },
                "required": ["reference_output_file", "candidate_output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "review_nwchem_mcscf_case",
            "description": "Review a NWChem MCSCF run for convergence quality, active-space health, and likely next action.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "review_nwchem_mcscf_followup_outcome",
            "description": "Compare a follow-up MCSCF run against a reference MCSCF run and summarize whether convergence or active-space quality improved.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_output_file": {"type": "string"},
                    "candidate_output_file": {"type": "string"},
                    "reference_input_file": {"type": "string"},
                    "candidate_input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                },
                "required": ["reference_output_file", "candidate_output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "prepare_nwchem_next_step",
            "description": "Diagnose a NWChem output and prepare the most likely next artifact, such as a swap restart or imaginary-mode follow-up inputs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "write_files": {"type": "boolean", "default": False},
                    "include_property_check": {"type": "boolean", "default": True},
                    "include_frontier_cubes": {"type": "boolean", "default": False},
                    "include_density_modes": {"type": "array", "items": {"type": "string"}},
                    "cube_extent_angstrom": {"type": "number", "default": 6.0},
                    "cube_grid_points": {"type": "integer", "default": 120},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Basis and ECP
        # ------------------------------------------------------------------
        {
            "name": "render_nwchem_basis_block",
            "description": "Render an explicit per-element NWChem basis block from the local library. Provide elements list or input_file (geometry source). Set check_only=true to validate existence without rendering.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "basis_name": {"type": "string"},
                    "elements": {"type": "array", "items": {"type": "string"}},
                    "input_file": {"type": "string"},
                    "library_path": {"type": "string"},
                    "block_name": {"type": "string", "default": "ao basis"},
                    "mode": {"type": "string"},
                    "check_only": {"type": "boolean", "default": False},
                },
                "required": ["basis_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "render_nwchem_ecp_block",
            "description": "Render an explicit per-element NWChem ECP block from the local library. Set check_only=true to validate existence without rendering.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ecp_name": {"type": "string"},
                    "elements": {"type": "array", "items": {"type": "string"}},
                    "library_path": {"type": "string"},
                    "check_only": {"type": "boolean", "default": False},
                },
                "required": ["ecp_name", "elements"],
                "additionalProperties": False,
            },
        },
        {
            "name": "render_nwchem_basis_setup",
            "description": "Render mixed per-element NWChem basis and ECP blocks from the local basis library.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "geometry_file": {"type": "string"},
                    "basis_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Per-element basis set assignments, e.g. {\"Fe\": \"def2-svp\", \"Cl\": \"def2-svp\"}. Use suggest_basis_set to get these.",
                    },
                    "ecp_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Per-element ECP assignments for heavy elements (Z>36), e.g. {\"U\": \"def2-ecp\"}. Only needed for post-Kr elements.",
                    },
                    "default_basis": {"type": "string"},
                    "default_ecp": {"type": "string"},
                    "block_name": {"type": "string", "default": "ao basis"},
                    "basis_mode": {"type": "string"},
                    "library_path": {"type": "string"},
                },
                "required": ["geometry_file", "basis_assignments"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Input creation
        # ------------------------------------------------------------------
        {
            "name": "create_nwchem_input",
            "description": "Create a new NWChem input with mixed explicit basis/ECP assignments and automatic movecs output for SCF/DFT.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "geometry_file": {"type": "string"},
                    "basis_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Per-element basis set assignments, e.g. {\"Fe\": \"def2-svp\", \"Cl\": \"def2-svp\"}. Use suggest_basis_set to generate these.",
                    },
                    "ecp_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Per-element ECP assignments for heavy elements (Z>36), e.g. {\"U\": \"def2-ecp\"}.",
                    },
                    "default_basis": {"type": "string"},
                    "default_ecp": {"type": "string"},
                    "block_name": {"type": "string", "default": "ao basis"},
                    "basis_mode": {"type": "string"},
                    "module": {"type": "string"},
                    "task_operation": {"type": "string"},
                    "charge": {"type": "integer"},
                    "multiplicity": {"type": "integer"},
                    "scf_type": {
                        "type": "string",
                        "enum": ["rhf", "uhf", "rohf"],
                        "description": "SCF wavefunction type for module='scf'. 'rohf' required for open-shell (multiplicity>1). Omit for DFT.",
                    },
                    "nopen": {
                        "type": "integer",
                        "description": "Number of open-shell (singly occupied) orbitals for ROHF. Equal to multiplicity-1. Required when scf_type='rohf'.",
                    },
                    "maxiter": {
                        "type": "integer",
                        "description": "Maximum SCF iterations (default: NWChem uses 30). Increase to 100+ for difficult convergence.",
                    },
                    "thresh": {
                        "type": "number",
                        "description": "SCF convergence threshold (e.g. 1e-6). Leave unset for NWChem default.",
                    },
                    "extra_blocks": {"type": "array", "items": {"type": "string"}},
                    "memory": {"type": "string"},
                    "title": {"type": "string"},
                    "start_name": {"type": "string"},
                    "vectors_input": {"type": "string"},
                    "vectors_output": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                    "library_path": {"type": "string"},
                },
                "required": ["geometry_file", "basis_assignments", "module"],
                "additionalProperties": False,
            },
        },
        {
            "name": "create_nwchem_dft_workflow_input",
            "description": "Create a standard NWChem DFT workflow input, such as optimize+freq, with explicit basis/ECP blocks and automatic movecs output.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "geometry_file": {"type": "string"},
                    "basis_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Per-element basis set assignments, e.g. {\"Fe\": \"def2-svp\", \"Cl\": \"def2-svp\"}. Use suggest_basis_set to generate these.",
                    },
                    "ecp_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Per-element ECP assignments for heavy elements (Z>36), e.g. {\"U\": \"def2-ecp\"}.",
                    },
                    "default_basis": {"type": "string"},
                    "default_ecp": {"type": "string"},
                    "block_name": {"type": "string", "default": "ao basis"},
                    "basis_mode": {"type": "string"},
                    "xc_functional": {"type": "string"},
                    "task_operations": {"type": "array", "items": {"type": "string"}},
                    "charge": {"type": "integer"},
                    "multiplicity": {"type": "integer"},
                    "dft_settings": {"type": "array", "items": {"type": "string"}},
                    "extra_blocks": {"type": "array", "items": {"type": "string"}},
                    "memory": {"type": "string"},
                    "title": {"type": "string"},
                    "start_name": {"type": "string"},
                    "vectors_input": {"type": "string"},
                    "vectors_output": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                    "library_path": {"type": "string"},
                },
                "required": ["geometry_file", "basis_assignments", "xc_functional", "task_operations"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Case analysis and recovery
        # ------------------------------------------------------------------
        {
            "name": "analyze_nwchem_case",
            "description": "One-shot NWChem case analysis: diagnosis, input lint, restart assets, spin-state review, and next-step planning. Use detail='compact' for the agent-facing triage payload, 'full' for the human-readable summary.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "library_path": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "detail": {"type": "string", "enum": ["compact", "full"], "default": "compact"},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_nwchem_recovery",
            "description": "Suggest ranked recovery strategies for a failed or suspicious NWChem run. Use mode='scf' for convergence failures, 'state' for wrong electronic state, 'auto' to check both.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                    "mode": {"type": "string", "enum": ["scf", "state", "auto"], "default": "auto"},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # MCSCF
        # ------------------------------------------------------------------
        {
            "name": "suggest_nwchem_mcscf_active_space",
            "description": "Suggest minimal and expanded MCSCF active spaces from the current MO and spin-state picture.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_mcscf_input",
            "description": "Draft a NWChem MCSCF input using the recommended active space and a vectors reordering plan.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                    "active_space_mode": {"type": "string", "enum": ["minimal", "expanded"], "default": "minimal"},
                    "vectors_input": {"type": "string"},
                    "vectors_output": {"type": "string"},
                    "state_label": {"type": "string"},
                    "symmetry": {"type": "integer"},
                    "hessian": {"type": "string", "enum": ["exact", "onel"], "default": "exact"},
                    "maxiter": {"type": "integer", "default": 80},
                    "thresh": {"type": "number", "default": 1.0e-5},
                    "level": {"type": "number", "default": 0.6},
                    "lock_vectors": {"type": "boolean", "default": True},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "title": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["reference_output_file", "input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_mcscf_retry_input",
            "description": "Draft a refined NWChem MCSCF retry input after reviewing a failed or stiff prior MCSCF run.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "active_space_mode": {"type": "string", "enum": ["auto", "minimal", "expanded"], "default": "auto"},
                    "vectors_input": {"type": "string"},
                    "vectors_output": {"type": "string"},
                    "state_label": {"type": "string"},
                    "symmetry": {"type": "integer"},
                    "hessian": {"type": "string", "enum": ["exact", "onel"]},
                    "maxiter": {"type": "integer"},
                    "thresh": {"type": "number"},
                    "level": {"type": "number"},
                    "lock_vectors": {"type": "boolean", "default": True},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "title": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["output_file", "input_file"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Geometry and frequency plausibility checks
        # ------------------------------------------------------------------
        {
            "name": "check_nwchem_geometry_plausibility",
            "description": (
                "Check whether an optimised NWChem geometry is chemically plausible. "
                "Run this after any optimisation — even a 'successful' one — to catch "
                "wrong minima, clashes, broken bonds, and coordination errors that NWChem "
                "itself does not report. Checks: bond lengths (clashes, unusually long bonds), "
                "coordination numbers per element, extreme bond angles (<55° or >175°), "
                "ring planarity for 5–7-membered rings, and metal coordination. "
                "Returns plausible=True/False, red_flags, warnings, and full bond/coord detail."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the NWChem optimisation output."},
                    "input_file": {"type": "string", "description": "Optional: input file for element labels."},
                    "frame": {
                        "type": "string",
                        "description": "'best' (smart selection), 'last', 'first', 'min_energy', or integer step.",
                        "default": "best",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "check_nwchem_freq_plausibility",
            "description": (
                "Check whether NWChem frequency results are chemically plausible. "
                "Run after a successful frequency job to catch: wrong number of imaginary modes "
                "(TS vs. minimum), very large imaginary modes indicating a bad structure, "
                "extra near-zero modes suggesting flat PES or incomplete optimisation, "
                "missing expected X-H stretch bands given the elements present, abnormal ZPE, "
                "and suspiciously high frequencies. Returns plausible=True/False, red_flags, "
                "warnings, mode band distribution, and ZPE check."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the NWChem frequency output."},
                    "input_file": {"type": "string", "description": "Optional: input file for element list."},
                    "expect_minimum": {
                        "type": "boolean",
                        "description": "True (default) if expecting a local minimum (0 imaginary modes). False for TS.",
                        "default": True,
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Imaginary modes
        # ------------------------------------------------------------------
        {
            "name": "analyze_nwchem_imaginary_modes",
            "description": "Analyze significant imaginary modes in a NWChem frequency output and identify the dominant moving atoms.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "significant_threshold_cm1": {"type": "number", "default": 20.0},
                    "top_atoms": {"type": "integer", "default": 4},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "displace_nwchem_geometry_along_mode",
            "description": "Generate plus/minus displaced geometries along an imaginary mode from a NWChem output.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "mode_number": {"type": "integer"},
                    "amplitude_angstrom": {"type": "number", "default": 0.15},
                    "significant_threshold_cm1": {"type": "number", "default": 20.0},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_imaginary_mode_inputs",
            "description": "Create plus/minus displaced NWChem input texts by replacing the input geometry along an imaginary mode.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "mode_number": {"type": "integer"},
                    "amplitude_angstrom": {"type": "number", "default": 0.15},
                    "significant_threshold_cm1": {"type": "number", "default": 20.0},
                    "task_strategy": {
                        "type": "string",
                        "enum": ["auto", "optimize_only", "optimize_then_freq"],
                        "default": "auto",
                    },
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "write_files": {"type": "boolean", "default": False},
                    "noautosym": {"type": "boolean", "default": True},
                    "symmetry_c1": {"type": "boolean", "default": True},
                },
                "required": ["output_file", "input_file"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # TCE (Tensor Contraction Engine) tools
        # ------------------------------------------------------------------
        {
            "name": "parse_nwchem_tce_output",
            "description": (
                "Parse a NWChem output file for TCE (Tensor Contraction Engine) results. "
                "Extracts: method (MP2/CCSD/CCSD(T)), correlation energy, total energy, "
                "frozen core count, convergence status, and per-section details. "
                "Works for any TCE method run via 'task tce energy'."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_nwchem_tce_amplitudes",
            "description": (
                "Compute multireference diagnostics from saved TCE amplitude files "
                "(*.t1_copy.*, *.t2_copy.*). Requires the TCE input to have included "
                "'set tce:save_t T T' — draft_nwchem_tce_input adds this by default. "
                "Returns: T1 diagnostic, D1 (nosym runs only), T2 Frobenius norm, "
                "max|t2|, top-10 T2 amplitudes, T2 dominance fraction, singles/doubles "
                "balance, triples fraction (for CCSD(T)), and a combined MR verdict "
                "('single_reference_ok', 'moderate_mr_character', 'strong_mr_character', "
                "or 'unreliable_ccsd'). Call after parse_nwchem_tce_output when you need "
                "to assess whether the single-reference wavefunction is adequate."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {
                        "type": "string",
                        "description": "Path to the NWChem TCE .out file.",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_tce_input",
            "description": (
                "Design a NWChem TCE correlated wavefunction input (MP2, CCSD, or CCSD(T)). "
                "ALWAYS call this AFTER an SCF calculation: it reads the SCF orbital output to "
                "determine the correct number of frozen core orbitals, auto-detects ECP nelec "
                "from the input file (no need to pass ecp_core_electrons manually), checks "
                "orbital ordering for anomalies (e.g. ligand 1s lower than metal 3s/3p), and "
                "warns if swap_nwchem_movecs must be run first. Never uses 'freeze atomic' — "
                "always emits an explicit 'freeze N' count. Returns n_electrons, n_correlated, "
                "and consistency warnings if the freeze/ECP setup looks wrong."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scf_output_file": {
                        "type": "string",
                        "description": "Path to the completed SCF output file (must contain MO analysis).",
                    },
                    "input_file": {
                        "type": "string",
                        "description": "Path to the SCF input file (for geometry/ECP metadata).",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["mp2", "ccsd", "ccsd(t)"],
                        "default": "mp2",
                        "description": "TCE correlation method.",
                    },
                    "freeze_count": {
                        "type": "integer",
                        "description": "Override freeze count. If omitted, computed from chemistry + ECP.",
                    },
                    "ecp_core_electrons": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                        "description": (
                            "Override ECP nelec values per element, e.g. {\"Zn\": 10, \"I\": 28}. "
                            "If omitted, nelec is auto-detected from the input file ECP block."
                        ),
                    },
                    "basis_library": {
                        "type": "string",
                        "description": "Path to basis library for ECP nelec lookup (library-assigned ECPs).",
                    },
                    "movecs_file": {
                        "type": "string",
                        "description": "Path to the movecs file. Inferred from SCF output if omitted.",
                    },
                    "swap_pairs": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "description": "List of [i,j] MO swap pairs already applied via swap_nwchem_movecs.",
                    },
                    "start_name": {"type": "string"},
                    "memory": {"type": "string", "default": "2000 mb"},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["scf_output_file", "input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "validate_nwchem_tce_setup",
            "description": (
                "Validate a NWChem TCE input file before submitting. "
                "Catches common errors: missing symmetry c1, wrong SCF reference for open-shell, "
                "'freeze atomic' (forbidden), missing vectors file, freeze count out of range. "
                "Call after draft_nwchem_tce_input and lint_nwchem_input, before launch_nwchem_run."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tce_input_file": {
                        "type": "string",
                        "description": "Path to the NWChem TCE input file to validate.",
                    },
                    "scf_output_file": {
                        "type": "string",
                        "description": "Optional: path to the SCF output that will be used as reference. Used to verify the vectors file exists.",
                    },
                },
                "required": ["tce_input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_nwchem_movecs",
            "description": (
                "Read orbital eigenvalues and occupations from a binary NWChem movecs file. "
                "Use this to inspect orbital ordering BEFORE designing a TCE freeze count. "
                "Returns all MO indices (1-based), energies in Hartree, and occupancies. "
                "Identifies which orbitals are occupied vs virtual."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "movecs_file": {
                        "type": "string",
                        "description": "Path to the binary NWChem .movecs file.",
                    },
                },
                "required": ["movecs_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "swap_nwchem_movecs",
            "description": (
                "Swap two MOs in a binary NWChem movecs file. This is the solution when "
                "orbital ordering is wrong for TCE freezing (e.g., O 1s at position 1 "
                "but Zn 3s/3p at positions 2-5). The RTDB is NOT modified, so if SCF "
                "was already converged, NWChem will use the swapped vectors directly on "
                "restart without re-running SCF. Always call parse_nwchem_movecs before "
                "and after to verify the swap worked correctly."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "movecs_file": {
                        "type": "string",
                        "description": "Path to the binary NWChem .movecs file to modify.",
                    },
                    "i": {
                        "type": "integer",
                        "description": "1-based index of the first MO to swap.",
                    },
                    "j": {
                        "type": "integer",
                        "description": "1-based index of the second MO to swap.",
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Output path. If omitted, overwrites the input file in-place.",
                    },
                },
                "required": ["movecs_file", "i", "j"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_nwchem_tce_freeze",
            "description": (
                "Suggest a freeze count for a NWChem TCE calculation from element list and ECP info. "
                "Returns a per-element breakdown and the total freeze count. "
                "This is a starting estimate — the agent must always verify against the actual "
                "SCF orbital eigenvalues using parse_nwchem_movecs or parse_nwchem_mos."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of element symbols (repeats allowed), e.g. [\"C\",\"C\",\"H\",\"H\"].",
                    },
                    "ecp_core_electrons": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                        "description": "ECP nelec values per element, e.g. {\"Zn\": 10}.",
                    },
                },
                "required": ["elements"],
                "additionalProperties": False,
            },
        },
    ]


# ---------------------------------------------------------------------------
# Handlers — frontier orbital / vectors-swap workflow
# ---------------------------------------------------------------------------

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
        basis=arguments["basis"],
        method=arguments.get("method", "ccsd"),
        xc_functional=arguments.get("xc_functional", "b3lyp"),
        has_geometry_file=arguments.get("has_geometry_file", False),
        has_dft_output=arguments.get("has_dft_output", False),
        has_scf_output=arguments.get("has_scf_output", False),
    )


@_tool("draft_initial_geometry")
def _handle_draft_initial_geometry(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_initial_geometry(
        atoms=arguments["atoms"],
        output_path=arguments["output_path"],
        comment=arguments.get("comment"),
        central_atom=arguments.get("central_atom"),
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


@_tool("analyze_nwchem_frontier_orbitals")
def _handle_analyze_nwchem_frontier_orbitals(arguments: dict[str, Any]) -> dict[str, Any]:
    return analyze_frontier_orbitals(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
    )


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
        convergence_energy=arguments.get("convergence_energy", "1e3"),
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
    if result.get("available"):
        xyz = result.get("xyz_file") or result.get("written_file", "<xyz_file>")
        result["next_steps"] = [
            f"Write the geometry to an XYZ file, then call create_nwchem_input or create_nwchem_dft_workflow_input with geometry_file='{xyz}'.",
            "For a TCE calculation: call suggest_nwchem_tce_freeze(elements=[...]) and draft_nwchem_tce_input after running an SCF on this geometry.",
        ]
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


@_tool("inspect_nwchem_runner_profiles")
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

@_tool("launch_nwchem_run")
def _handle_launch_nwchem_run(arguments: dict[str, Any]) -> dict[str, Any]:
    return launch_nwchem_run(
        input_path=arguments["input_file"],
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path"),
        job_name=arguments.get("job_name"),
        resource_overrides=arguments.get("resource_overrides"),
        env_overrides=arguments.get("env_overrides"),
        write_script=arguments.get("write_script", True),
        dry_run=arguments.get("dry_run", False),
    )


@_tool("get_nwchem_run_status")
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
    return status


@_tool("tail_nwchem_output")
def _handle_tail_nwchem_output(arguments: dict[str, Any]) -> dict[str, Any]:
    return tail_nwchem_output(
        arguments["output_file"],
        lines=arguments.get("lines", 30),
        max_characters=min(arguments.get("max_characters", 4000), 10000),
    )


@_tool("terminate_nwchem_run")
def _handle_terminate_nwchem_run(arguments: dict[str, Any]) -> dict[str, Any]:
    return terminate_nwchem_run(
        process_id=arguments["process_id"],
        signal_name=arguments.get("signal_name", "term"),
    )


@_tool("watch_nwchem_run")
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
    out_file = arguments.get("output_file", "")
    in_file = arguments.get("input_file", "")
    overall = result.get("overall_status", "")
    tasks = result.get("progress_summary", {}).get("tasks", []) if result.get("progress_summary") else []
    has_tce = any((t.get("module") or "").lower() == "tce" for t in tasks)
    has_freq = any((t.get("module") or "").lower() in {"freq", "frequency"} for t in tasks)
    has_opt = any((t.get("operation") or "").lower() == "optimize" for t in tasks)

    next_steps: list[str] = []
    if overall == "completed":
        if has_tce:
            next_steps.append(f"Call parse_nwchem_tce_output(output_file='{out_file}') to extract correlation energies and T1/D1 diagnostics.")
        elif has_freq and has_opt:
            next_steps.append(f"Call extract_nwchem_geometry(output_file='{out_file}', frame='best') to get the converged geometry for the next calculation.")
            next_steps.append(f"Call parse_nwchem_output(output_file='{out_file}', sections=['tasks','freq']) to review frequencies.")
        elif has_freq:
            next_steps.append(f"Call parse_nwchem_output(output_file='{out_file}', sections=['freq']) to review frequency results.")
        else:
            next_steps.append(f"Call analyze_nwchem_case(output_file='{out_file}', input_file='{in_file}') to review the result and get next-step guidance.")
            has_scf_or_dft = any((t.get("module") or "").lower() in {"scf", "dft"} for t in tasks)
            if has_scf_or_dft:
                next_steps.append(f"Call suggest_nwchem_vectors_swaps(output_file='{out_file}', input_file='{in_file}') to verify the correct spin state converged (important for open-shell metals).")
    elif overall in {"failed", "error"}:
        next_steps.append(f"Call analyze_nwchem_case(output_file='{out_file}', input_file='{in_file}') to diagnose the failure.")
        next_steps.append("Call suggest_nwchem_recovery(output_file=..., input_file=..., mode='auto') if recovery is needed.")
    elif overall == "running":
        next_steps.append("Job is still running. Call watch_nwchem_run again to continue monitoring.")

    if next_steps:
        result["next_steps"] = next_steps
    return result


# ---------------------------------------------------------------------------
# Handlers — run comparison and follow-up
# ---------------------------------------------------------------------------

@_tool("compare_nwchem_runs")
def _handle_compare_nwchem_runs(arguments: dict[str, Any]) -> dict[str, Any]:
    if arguments.get("output_dir") or arguments.get("base_name"):
        return review_nwchem_followup_outcome(
            reference_output_path=arguments["reference_output_file"],
            candidate_output_path=arguments["candidate_output_file"],
            reference_input_path=arguments.get("reference_input_file"),
            candidate_input_path=arguments.get("candidate_input_file"),
            expected_metal_elements=arguments.get("expected_metals"),
            expected_somo_count=arguments.get("expected_somos"),
            output_dir=arguments.get("output_dir"),
            base_name=arguments.get("base_name"),
        )
    return compare_nwchem_runs(
        reference_output_path=arguments["reference_output_file"],
        candidate_output_path=arguments["candidate_output_file"],
        reference_input_path=arguments.get("reference_input_file"),
        candidate_input_path=arguments.get("candidate_input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
    )


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
    return create_nwchem_dft_workflow_input(
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
        memory=arguments.get("memory"),
        title=arguments.get("title"),
        start_name=arguments.get("start_name"),
        vectors_input=arguments.get("vectors_input"),
        vectors_output=arguments.get("vectors_output"),
        output_dir=arguments.get("output_dir"),
        write_file=arguments.get("write_file", False),
    )


# ---------------------------------------------------------------------------
# Handlers — case analysis and recovery
# ---------------------------------------------------------------------------

def _derive_recommended_next_tool(result: dict[str, Any]) -> dict[str, Any]:
    """Derive a structured next-tool recommendation from summarize_nwchem_case output."""
    diagnosis = result.get("diagnosis") or {}
    next_step = result.get("next_step") or {}
    state = result.get("spin_charge_state") or {}
    output_file = result.get("output_file", "")
    input_file = result.get("input_file") or ""

    task_outcome = diagnosis.get("task_outcome", "")
    failure_class = diagnosis.get("failure_class", "")
    selected_workflow = next_step.get("selected_workflow", "")
    state_action = state.get("recommended_next_action", "") if state else ""

    if task_outcome == "success":
        wf = selected_workflow.lower()
        if "tce" in wf or "ccsd" in wf or "mp2" in wf:
            return {
                "tool": "parse_nwchem_tce_output",
                "parameters": {"output_file": output_file},
                "reason": "Correlated calculation completed — extract energies and T1/D1 diagnostics.",
            }
        if "freq" in wf and ("opt" in wf or "geometry" in wf):
            return {
                "tool": "extract_nwchem_geometry",
                "parameters": {"output_file": output_file, "frame": "best"},
                "reason": "Opt+freq completed — extract converged geometry for the next calculation.",
            }
        if "freq" in wf:
            return {
                "tool": "parse_nwchem_output",
                "parameters": {"output_file": output_file, "sections": ["freq", "tasks"]},
                "reason": "Frequency calculation completed — review normal modes.",
            }
        if state_action and "swap" in state_action:
            return {
                "tool": "suggest_nwchem_vectors_swaps",
                "parameters": {"output_file": output_file, "input_file": input_file},
                "reason": "State check flagged spin inconsistency — verify orbital ordering before proceeding.",
            }
        return {
            "tool": "draft_nwchem_tce_input",
            "parameters": {"output_file": output_file, "input_file": input_file},
            "reason": "SCF/DFT complete and state looks OK — draft correlated follow-up if needed.",
        }
    elif task_outcome in ("failed", "error", "scf_failed"):
        if failure_class == "scf_convergence":
            return {
                "tool": "suggest_nwchem_recovery",
                "parameters": {"output_file": output_file, "input_file": input_file, "mode": "scf"},
                "reason": "SCF convergence failure — get targeted strategies.",
            }
        if failure_class in ("bad_state", "wrong_state", "state_mismatch"):
            return {
                "tool": "suggest_nwchem_recovery",
                "parameters": {"output_file": output_file, "input_file": input_file, "mode": "state"},
                "reason": "Spin/state error — recover with state correction strategies.",
            }
        return {
            "tool": "suggest_nwchem_recovery",
            "parameters": {"output_file": output_file, "input_file": input_file, "mode": "auto"},
            "reason": f"Calculation failed ({failure_class or 'unknown'}) — get recovery recommendations.",
        }
    return {
        "tool": "analyze_nwchem_case",
        "parameters": {"output_file": output_file, "input_file": input_file},
        "reason": "Outcome unclear — re-run analysis with more context.",
    }


@_tool("analyze_nwchem_case")
def _handle_analyze_nwchem_case(arguments: dict[str, Any]) -> dict[str, Any]:
    compact = arguments.get("detail", "compact") == "compact"
    result = summarize_nwchem_case(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        library_path=basis_library_path(arguments.get("library_path")),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        compact=compact,
    )
    result["recommended_next_tool"] = _derive_recommended_next_tool(result)
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
    return check_nwchem_geometry_plausibility(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        frame=frame_arg,
    )


@_tool("check_nwchem_freq_plausibility")
def _handle_check_nwchem_freq_plausibility(arguments: dict[str, Any]) -> dict[str, Any]:
    return check_nwchem_freq_plausibility(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expect_minimum=arguments.get("expect_minimum", True),
    )


# ---------------------------------------------------------------------------
# Handlers — imaginary modes
# ---------------------------------------------------------------------------

@_tool("analyze_nwchem_imaginary_modes")
def _handle_analyze_nwchem_imaginary_modes(arguments: dict[str, Any]) -> dict[str, Any]:
    return analyze_imaginary_modes(
        arguments["output_file"],
        significant_threshold_cm1=arguments.get("significant_threshold_cm1", 20.0),
        top_atoms=arguments.get("top_atoms", 4),
    )


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
    return parse_tce_output(arguments["output_file"])


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
    nw_file = result.get("written_file") or result.get("planned_output_file", "<tce.nw>")
    has_swap_warnings = bool(result.get("ordering_warnings"))
    next_steps = []
    if has_swap_warnings:
        next_steps.append("WARNING: orbital ordering issues detected. Call swap_nwchem_movecs for each flagged pair, verify with parse_nwchem_movecs, then re-run draft_nwchem_tce_input.")
    next_steps += [
        f"Call lint_nwchem_input(input_file='{nw_file}') to validate before launching.",
        f"Call launch_nwchem_run(input_file='{nw_file}', profile='<your_profile>') to start the job.",
        "After completion: call parse_nwchem_tce_output to extract energies and T1/D1 diagnostics.",
    ]
    result["next_steps"] = next_steps
    return result


@_tool("validate_nwchem_tce_setup")
def _handle_validate_nwchem_tce_setup(arguments: dict[str, Any]) -> dict[str, Any]:
    return validate_nwchem_tce_setup(
        tce_input_path=arguments["tce_input_file"],
        scf_output_path=arguments.get("scf_output_file"),
    )


@_tool("parse_nwchem_movecs")
def _handle_parse_nwchem_movecs(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_nwchem_movecs(arguments["movecs_file"])


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


def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    log_event(f"dispatch_tool start name={name}")
    handler = _TOOL_REGISTRY.get(name)
    if handler is None:
        raise ValueError(f"unknown tool: {name}")
    payload = handler(arguments)
    log_event(f"dispatch_tool done name={name}")
    return payload


def make_success_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, indent=2),
            }
        ],
        "structuredContent": payload,
        "isError": False,
    }


def make_error_result(message: str) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": message,
            }
        ],
        "isError": True,
    }


def make_response(request_id: Any, result: dict[str, Any] | None = None, error: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result if result is not None else {}
    return payload


def handle_request(message: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
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
        log_event("tools/list requested")
        return make_response(request_id, {"tools": tool_definitions()}), False
    if method == "tools/call":
        try:
            tool_name = params["name"]
            arguments = params.get("arguments", {})
            log_event(f"tools/call name={tool_name}")
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


def read_message(stream: Any) -> dict[str, Any] | None:
    global TRANSPORT_MODE
    headers: dict[str, str] = {}

    while True:
        line = stream.readline()
        if not line:
            return None
        if line.lstrip().startswith(b"{"):
            TRANSPORT_MODE = "jsonl"
            return json.loads(line.decode("utf-8"))
        if line in (b"\r\n", b"\n"):
            break
        decoded = line.decode("utf-8")
        if ":" not in decoded:
            continue
        key, value = decoded.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    content_length = headers.get("content-length")
    if content_length is None:
        return None
    body = stream.read(int(content_length))
    if not body:
        return None
    TRANSPORT_MODE = "content-length"
    return json.loads(body.decode("utf-8"))


def write_message(stream: Any, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    if TRANSPORT_MODE == "jsonl":
        stream.write(body + b"\n")
        stream.flush()
        return
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    stream.write(header)
    stream.write(body)
    stream.flush()


def serve() -> None:
    input_stream = sys.stdin.buffer
    output_stream = sys.stdout.buffer
    log_event("server start")

    while True:
        message = read_message(input_stream)
        if message is None:
            log_event("server stop: no message")
            break
        response, should_exit = handle_request(message)
        if response is not None:
            write_message(output_stream, response)
        if should_exit:
            log_event("server stop: exit requested")
            break


def main() -> None:
    """Entry point registered by pyproject.toml — `chemtools-nwchem` command."""
    serve()


if __name__ == "__main__":
    serve()

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
    suggest_resources,
    suggest_relativistic_correction,
    suggest_spin_state,
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
    check_memory_fit,
    estimate_freq_walltime,
    suggest_hpc_resources,
    detect_hpc_accounts,
)
from chemtools.eval import evaluate_case, evaluate_cases
from chemtools.nwchem_docs import (
    find_examples as docs_find_examples,
    get_topic_guide as docs_get_topic_guide,
    list_docs as docs_list_docs,
    lookup_block_syntax as docs_lookup_block_syntax,
    read_doc_excerpt as docs_read_doc_excerpt,
    search_docs as docs_search_docs,
)
from chemtools.nwchem_forum import search_forum as forum_search


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
                    "basis": {"type": "string", "description": "Basis set name, e.g. '6-31gs'. If omitted, suggest_basis_set is included as step 1 of the plan."},
                    "method": {"type": "string", "default": "ccsd",
                               "description": "TCE method: 'ccsd', 'mp2', or 'ccsd(t)'."},
                    "xc_functional": {"type": "string", "default": "b3lyp",
                                      "description": "DFT exchange-correlation functional for opt/freq step, e.g. 'b3lyp', 'pbe0', 'm06', 'tpss'. Default: 'b3lyp'."},
                    "has_geometry_file": {"type": "boolean", "default": False},
                    "has_dft_output": {"type": "boolean", "default": False},
                    "has_scf_output": {"type": "boolean", "default": False},
                },
                "required": ["goal", "elements", "charge", "multiplicity"],
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
            "name": "draft_nwchem_atom_input",
            "description": (
                "Generate a NWChem input for a single atom (for atomization energies, ionization "
                "potentials, electron affinities). Automatically looks up the neutral ground-state "
                "multiplicity for common elements (H–Xe plus 5d metals). Use before "
                "compute_reaction_energy to run each atom at the same level of theory as the molecule. "
                "Always uses symmetry c1 and places the atom at the origin."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "element": {
                        "type": "string",
                        "description": "Element symbol, e.g. 'Fe', 'O', 'C'.",
                    },
                    "basis": {
                        "type": "string",
                        "description": "Basis set name, e.g. '6-31gs', 'def2-tzvp', 'cc-pvtz'.",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["scf", "dft", "mp2"],
                        "default": "scf",
                        "description": "NWChem module to use.",
                    },
                    "charge": {
                        "type": "integer",
                        "default": 0,
                        "description": "Total charge (0 = neutral atom).",
                    },
                    "multiplicity": {
                        "type": "integer",
                        "description": "Spin multiplicity. Auto-looked-up from ground-state table if omitted.",
                    },
                    "xc_functional": {
                        "type": "string",
                        "default": "m06",
                        "description": "XC functional when method=dft.",
                    },
                    "memory": {"type": "string", "description": "NWChem memory directive, e.g. '2000 mb'."},
                    "start_name": {"type": "string", "description": "NWChem start name. Defaults to '{element}_atom'."},
                    "output_dir": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                    "basis_library": {"type": "string"},
                },
                "required": ["element", "basis"],
                "additionalProperties": False,
            },
        },
        {
            "name": "compute_reaction_energy",
            "description": (
                "Compute a reaction energy ΔE from a set of NWChem output files. "
                "Collects the best available energy per species (CCSD(T) > CCSD > MP2 > DFT > SCF) "
                "and returns ΔE in Hartree, kcal/mol, and eV. "
                "Use for atomization energies (FeO2⁻ → Fe + 2O), binding energies, "
                "reaction enthalpies, or isomerization energies. "
                "Set include_thermochem=true to also compute ΔE+ZPE, ΔH(T), ΔG(T) "
                "from frequency outputs (requires all species to have thermochemistry data). "
                "Example: species={'mol': 'mol.out', 'A': 'a.out', 'B': 'b.out'}, "
                "reactants={'mol': 1}, products={'A': 1, 'B': 1}."
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
            "name": "parse_nwchem_thermochem",
            "description": (
                "Extract complete thermochemistry from an NWChem frequency output: "
                "electronic energy (E_scf), zero-point energy (ZPE), enthalpy H(T), "
                "Gibbs free energy G(T), entropy S, and heat capacity Cv. "
                "Combines the SCF/DFT energy with frequency-derived corrections. "
                "Returns all quantities in both Hartree and kcal/mol. "
                "Warns if imaginary modes are present (saddle point, not a minimum). "
                "Call this after check_nwchem_freq_plausibility confirms no significant imaginary modes."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {
                        "type": "string",
                        "description": "Path to the NWChem frequency output file.",
                    },
                    "T": {
                        "type": "number",
                        "description": "Temperature in Kelvin (default 298.15). Note: NWChem computes corrections at the temperature in the input; this is for reporting only.",
                    },
                    "P": {
                        "type": "number",
                        "description": "Pressure in atm (default 1.0). For reporting only.",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "summarize_nwchem_electronic_structure",
            "description": (
                "Produce a compact electronic structure summary from an NWChem output: "
                "HOMO-LUMO gap (Hartree and eV), frontier orbital character, SOMO count, "
                "Mulliken charges and spin densities on metal centers, "
                "spin-state consistency check, and top charge/spin sites. "
                "Use after a DFT or SCF calculation to verify the electronic state is "
                "physically reasonable before proceeding to the next step."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {
                        "type": "string",
                        "description": "Path to the NWChem output file.",
                    },
                    "input_file": {
                        "type": "string",
                        "description": "Optional path to the NWChem input file (used to read charge/multiplicity).",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "track_nwchem_spin_state",
            "description": (
                "Track <S²> and energy across optimization steps to detect spin-state "
                "changes during geometry optimization. Parses per-step DFT energies and "
                "<S²> values, detects discontinuities that suggest spin flips, state "
                "crossings, or broken-symmetry collapse. Reports spin contamination. "
                "Call this after an optimization completes (especially for open-shell "
                "transition-metal or f-element systems) to verify the electronic state "
                "remained consistent throughout."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {
                        "type": "string",
                        "description": "Path to the NWChem optimization output file.",
                    },
                },
                "required": ["output_file"],
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
        # ------------------------------------------------------------------
        {
            "name": "suggest_relativistic_correction",
            "description": (
                "Advise on relativistic corrections for a calculation given the elements present. "
                "Returns the recommended method (X2C, DKH2, or none), the NWChem 'relativistic...end' "
                "block to add, and warnings about incompatibilities. "
                "KEY RULES: (1) X2C/DKH are ALL-ELECTRON methods — do NOT use with ECPs on the same element. "
                "(2) Use DK-family basis sets (cc-pVDZ-DK, cc-pVTZ-DK, x2c-SVPall) with X2C/DKH. "
                "(3) X2C SAD guess for heavy transition metals can take 30–120+ min with no output — this is normal. "
                "Call before drafting any input that contains 4d/5d transition metals or Z > 36 elements."
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
                    "convergence_energy": {"type": "string", "default": "1e-3"},
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
            "description": (
                "Launch a NWChem input through a named runner profile. For scheduler (HPC) profiles "
                "the job is submitted via sbatch/qsub and the scheduler job ID is written to "
                "{job_name}.jobid. With auto_watch=true (default) the tool immediately transitions "
                "into polling mode and blocks until the job reaches a terminal state — the agent does "
                "not need to call watch_nwchem_run separately. Set auto_watch=false to return "
                "immediately after submission (use when you want to submit multiple jobs in parallel). "
                "Use dry_run=true for a preview without executing."
            ),
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
                    "auto_watch": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "When true (default), automatically polls until the job completes. "
                            "For scheduler jobs this blocks until squeue reports a terminal state. "
                            "Set false to return immediately after submission."
                        ),
                    },
                    "auto_register": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "When true (default), auto-registers the run in the SQLite registry. "
                            "If auto_watch is also true, auto-updates status on completion."
                        ),
                    },
                    "campaign_id": {"type": "integer", "description": "Link this run to a campaign (requires auto_register)."},
                    "workflow_id": {"type": "integer", "description": "Link this run to a workflow."},
                    "workflow_step_id": {"type": "string", "description": "Workflow step ID."},
                    "parent_run_id": {"type": "integer", "description": "Previous run in a restart chain."},
                },
                "required": ["input_file", "profile"],
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
            "name": "get_nwchem_run_status",
            "description": (
                "Check the status of a NWChem run. For HPC jobs the scheduler job ID is auto-detected "
                "from {job_name}.jobid alongside the input/output file. Returns scheduler state "
                "(queued/running/completed/failed/cancelled), parsed output outcome, and a compact "
                "progress summary for running jobs."
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
            "name": "watch_nwchem_run",
            "description": (
                "Poll NWChem status until the run reaches a terminal state or a timeout/max-poll limit. "
                "For HPC scheduler jobs omit timeout_seconds (or set to null) so the tool blocks until "
                "the job reaches a terminal scheduler state — the job's own walltime governs the limit. "
                "For local jobs the default 3600 s timeout applies. "
                "Detects known output-silent phases and reports them as 'expected slow' rather than 'hung': "
                "SAD guess (always silent), X2C/DKH SAD atomic solves (30–120+ min for heavy TMs), "
                "DFT grid generation, frequency Hessian displacements, TCE AO→MO transformation. "
                "When slow_phase is set in final_status, do NOT interpret output silence as a crash. "
                "NOTE: launch_nwchem_run already calls this automatically for scheduler jobs unless "
                "auto_watch=false was set — only call this directly for local processes or for jobs "
                "launched with auto_watch=false."
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
                    "timeout_seconds": {
                        "type": ["number", "null"],
                        "default": 3600.0,
                        "description": "Seconds before timing out. Set null for HPC jobs to wait indefinitely.",
                    },
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
            "description": (
                "Stop a running NWChem job. "
                "For local runs: provide process_id and optionally signal_name (term or kill). "
                "For HPC scheduler jobs: provide job_id and profile (calls scancel/qdel/bkill). "
                "Only call after intervention review has determined the run should stop."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "process_id": {"type": "integer", "description": "Local process ID (direct/local runs)."},
                    "signal_name": {"type": "string", "default": "term", "description": "term or kill (local only)."},
                    "job_id": {"type": "string", "description": "Scheduler job ID (HPC runs)."},
                    "profile": {"type": "string", "description": "Runner profile name (required with job_id)."},
                    "profiles_path": {"type": "string"},
                },
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
            "description": "One-shot NWChem case analysis: diagnosis, input lint, restart assets, spin-state review, and next-step planning. Automatically reads the .err file (same basename) for crash classification. Use detail='compact' for the agent-facing triage payload, 'full' for the human-readable summary.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "err_file": {"type": "string", "description": "Path to the .err file. Auto-detected from output_file if omitted."},
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
        {
            "name": "parse_nwchem_freq_progress",
            "description": (
                "Report progress of a finite-difference frequency (Hessian) job. "
                "Returns: displacements done vs total, percentage complete, pace (sec/gradient), "
                "estimated remaining time, number of additional 48h runs needed, and fdrst checkpoint info. "
                "Essential for multi-restart freq jobs on HPC with walltime limits."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the NWChem freq output file."},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "prepare_nwchem_freq_restart",
            "description": (
                "Validate that a frequency restart is ready and return a submit-ready report. "
                "Checks: 'restart' keyword in input, .fdrst checkpoint exists, .db exists, "
                "and reports freq progress from the previous output. "
                "Does NOT submit — use launch_nwchem_run after confirming ready_to_restart=true."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the NWChem .nw input file."},
                    "output_file": {"type": "string", "description": "Path to the previous NWChem output file."},
                    "profile": {"type": "string", "description": "Runner profile name for resubmission."},
                },
                "required": ["input_file", "output_file"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Imaginary modes
        # ------------------------------------------------------------------
        {
            "name": "analyze_nwchem_imaginary_modes",
            "description": (
                "Analyze significant imaginary modes in a NWChem frequency output and identify the dominant moving atoms. "
                "Default detail='compact' strips displacement vectors (~3 KB output). Use detail='full' to include "
                "full Cartesian displacements (needed for displace_nwchem_geometry_along_mode)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "significant_threshold_cm1": {"type": "number", "default": 20.0},
                    "top_atoms": {"type": "integer", "default": 4},
                    "detail": {
                        "type": "string",
                        "enum": ["compact", "full"],
                        "default": "compact",
                        "description": "compact: omit displacement vectors (default, much smaller). full: include all Cartesian displacements.",
                    },
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
                "Also automatically attempts to read saved amplitude files (*.t1_copy.*, *.t2_copy.*) "
                "and includes multireference diagnostics: T1, D1, T2 norm, mr_assessment, and mr_flags. "
                "T1 > 0.02 = moderate MR; > 0.05 = strong MR / CCSD unreliable. "
                "Amplitude files are written when draft_nwchem_tce_input includes 'set tce:save_t T T' (default). "
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
            "name": "draft_nwchem_tce_restart_input",
            "description": (
                "Generate a NWChem TCE restart input for a stalled or timed-out CCSD/MP2 run. "
                "Finds saved amplitude files (.t1amp.* or .t1_copy.*), copies them to the "
                "{start_name}.t1/.t2 restart names, and builds a 'restart' input with "
                "'set tce:read_ta .true.' and 'set tce:save_t T T'. Use when CCSD iterations "
                "stall before convergence (e.g. residual ~0.001 at iter 100). "
                "Returns the restart input text and a report of which amplitude files were copied."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tce_output_file": {
                        "type": "string",
                        "description": "Path to the incomplete TCE output (.out) file.",
                    },
                    "tce_input_file": {
                        "type": "string",
                        "description": "Path to the previous TCE input (.nw). Auto-inferred if omitted.",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "default": 200,
                        "description": "Max CCSD iterations for the restart run.",
                    },
                    "thresh": {
                        "type": "number",
                        "default": 1e-5,
                        "description": "CCSD residual threshold (default 1e-5, 10× looser than NWChem default).",
                    },
                    "copy_amplitudes": {
                        "type": "boolean",
                        "default": True,
                        "description": "If true, copy .t1amp/.t1_copy files to the restart names automatically.",
                    },
                    "output_dir": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["tce_output_file"],
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
        # ------------------------------------------------------------------
        # Parallel job monitoring
        # ------------------------------------------------------------------
        {
            "name": "watch_multiple_runs",
            "description": (
                "Monitor multiple NWChem jobs simultaneously until all reach a terminal state "
                "(completed, failed, or cancelled). Use this after submitting several jobs in "
                "parallel with auto_watch=false — call this once and it will block until all "
                "jobs finish, then return a consolidated status table. "
                "Each job entry requires output_file and optionally profile and job_id "
                "(job_id auto-detected from <output_file>.jobid if omitted)."
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
                                "job_id": {"type": "string", "description": "Scheduler job ID (auto-detected from .jobid file if omitted)."},
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
        # ------------------------------------------------------------------
        # Session log — running Markdown doc for context preservation
        # ------------------------------------------------------------------
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
        # ------------------------------------------------------------------
        # Input file versioning
        # ------------------------------------------------------------------
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
            "name": "create_nwchem_input_variant",
            "description": (
                "Create a versioned copy of an NWChem input file with specified changes, "
                "recording a structured diff of what changed and why. The original file is "
                "never overwritten — output goes to _v2.nw, _v3.nw, etc. "
                "Use this when resubmitting a failed job with modifications (e.g. reducing "
                "memory after OOM, changing iterations, switching functional). "
                "Supported change keys: 'memory', 'charge', 'mult', 'task', and "
                "'block.keyword' patterns like 'dft.iterations', 'dft.xc', 'scf.maxiter'."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_input": {"type": "string", "description": "Path to the original .nw input file."},
                    "changes": {
                        "type": "object",
                        "description": (
                            "Key-value pairs of changes to apply. Keys are directive names "
                            "like 'memory', 'charge', 'mult', 'task', or 'block.keyword' "
                            "patterns like 'dft.iterations', 'dft.xc'."
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why these changes are being made (e.g. 'OOM at 2000 mb on SPR nodes').",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Explicit output path. If omitted, auto-versioned from source.",
                    },
                },
                "required": ["source_input", "changes"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_nwchem_workflow_state",
            "description": (
                "Determine the current workflow state of an NWChem calculation and return "
                "the exact next tool call to advance it. Returns an explicit state enum "
                "(pending, running_scf, running_freq, freq_timelimited, scf_failed, "
                "imaginary_modes, opt_converged, completed, oom, etc.) plus a pre-filled "
                "next_action with all parameters ready. A model can drive the full NWChem "
                "workflow by looping: call this tool → execute next_action → repeat. "
                "Domain logic is encoded in the tool, not expected from the model. "
                "input_file is optional — when only the .out file is available, the tool "
                "parses the NWChem input echo from the output. The tool also checks squeue "
                "for related running jobs and reports them in related_jobs (never assumes — "
                "asks the model to confirm with the user). Missing companion files (.nw, "
                ".fdrst, .err, .db, .movecs) are listed in missing_files."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the NWChem .nw input file. Optional — parsed from output echo if missing."},
                    "output_file": {"type": "string", "description": "Path to the NWChem .out output file."},
                    "profile": {"type": "string", "description": "Runner profile name (e.g. 'stampede3_skx')."},
                    "error_file": {"type": "string", "description": "Path to .err file. Auto-derived from output_file if omitted."},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "plan_nwchem_calculation",
            "description": (
                "Generate a step-by-step calculation plan from a pre-baked protocol. "
                "Available protocols: single_point_dft, geometry_opt_dft, thermochem_dft, "
                "opt_then_tce, basis_set_convergence, spin_state_scan. "
                "Returns step IDs, dependencies, and the exact tool calls for each step. "
                "The model follows the plan — no NWChem expertise needed."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the NWChem .nw input file."},
                    "protocol": {
                        "type": "string",
                        "description": "Protocol name.",
                        "enum": ["single_point_dft", "geometry_opt_dft", "thermochem_dft",
                                 "opt_then_tce", "basis_set_convergence", "spin_state_scan"],
                    },
                    "profile": {"type": "string", "description": "Runner profile name."},
                    "output_dir": {"type": "string", "description": "Directory for output files. Defaults to input file directory."},
                    "overrides": {
                        "type": "object",
                        "description": "Optional overrides (e.g. multiplicities for spin_state_scan).",
                        "additionalProperties": True,
                    },
                },
                "required": ["input_file", "protocol"],
                "additionalProperties": False,
            },
        },
        {
            "name": "list_nwchem_protocols",
            "description": (
                "List all available pre-baked calculation protocols with descriptions. "
                "Protocols encode multi-step NWChem workflows (opt→freq, opt→TCE, etc.) "
                "so the model can plan calculations without NWChem expertise."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        # --- Phase 6: Eval + smart input creation ---
        {
            "name": "evaluate_nwchem_case",
            "description": (
                "Evaluate an NWChem test case against expected outcomes. "
                "Reads a case.json file that defines input/output paths and expectations "
                "(failure_class, recommended_next_action, workflow). Returns pass/fail checks. "
                "Use for automated validation of tool quality and regression testing."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string", "description": "Path to a case.json or *.case.json file."},
                },
                "required": ["case_path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "evaluate_nwchem_cases",
            "description": (
                "Batch-evaluate all NWChem test cases in a directory. "
                "Recursively finds case.json files and evaluates each one. "
                "Returns aggregate pass/fail counts and per-case results."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to a directory containing case.json files."},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "create_nwchem_dft_input_from_request",
            "description": (
                "Create an NWChem DFT input from a high-level request, with automatic validation. "
                "Runs review_nwchem_input_request first to check readiness (basis availability, "
                "charge/multiplicity consistency, etc.). If ready, creates the full input with "
                "explicit basis blocks and geometry. Returns ready_to_create=false with guidance "
                "if requirements are missing. Simpler than create_nwchem_input — fewer required params."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "formula": {"type": "string", "description": "Molecular formula (e.g. 'C6H6'). Used to detect elements."},
                    "geometry_file": {"type": "string", "description": "Path to geometry file (.xyz)."},
                    "basis_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Element → basis name mapping.",
                    },
                    "ecp_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "default_basis": {"type": "string", "description": "Default basis for all elements."},
                    "default_ecp": {"type": "string"},
                    "xc_functional": {"type": "string", "description": "DFT functional (e.g. 'b3lyp', 'm06')."},
                    "task_operations": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Operations: ['optimize'], ['optimize', 'freq'], ['energy'].",
                    },
                    "charge": {"type": "integer"},
                    "multiplicity": {"type": "integer"},
                    "dft_settings": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Extra DFT block lines (e.g. ['grid fine', 'convergence energy 1e-8']).",
                    },
                    "extra_blocks": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Extra NWChem blocks to append.",
                    },
                    "memory": {"type": "string", "description": "Memory directive (e.g. 'total 2000 mb')."},
                    "title": {"type": "string"},
                    "start_name": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
        },
        # --- Phase 5: Gap-fill tools ---
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
            "name": "check_nwchem_spin_charge_state",
            "description": (
                "Validate the spin/charge state from a completed NWChem output. "
                "Checks <S²> vs expected, SOMO count, Mulliken spin density on metals, "
                "and flags spin contamination or wrong-state convergence. "
                "Essential after any open-shell SCF/DFT calculation."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the NWChem output file."},
                    "input_file": {"type": "string", "description": "Path to the NWChem input file."},
                    "expected_metals": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Metal element symbols to check spin density on (e.g. ['Fe', 'Ru']).",
                    },
                    "expected_somos": {
                        "type": "integer",
                        "description": "Expected number of singly-occupied MOs.",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "inspect_nwchem_geometry",
            "description": (
                "Inspect the geometry from an NWChem input file. Returns atom count, "
                "elements, coordinate format (xyz/zmatrix), symmetry, bond distances, "
                "and detects potential issues (close contacts, missing atoms). "
                "Use before running a calculation to verify the geometry is reasonable."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the NWChem input file."},
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_nwchem_tasks",
            "description": (
                "Parse task boundaries and energies from NWChem output. "
                "Returns each task's module, operation, energy, status, and timing. "
                "Useful for multi-task outputs (e.g. opt followed by freq)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the NWChem output file."},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_nwchem_trajectory",
            "description": (
                "Parse the optimization trajectory from NWChem output. "
                "Returns per-step energies, gradients, step sizes, and convergence criteria. "
                "Optionally includes atomic positions at each step. "
                "Use to understand optimization progress and convergence behavior."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the NWChem output file."},
                    "include_positions": {
                        "type": "boolean", "default": False,
                        "description": "Include atomic positions at each step (verbose).",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "review_nwchem_input_request",
            "description": (
                "Pre-flight review of input parameters before creating an NWChem input. "
                "Validates basis/element compatibility, checks charge/multiplicity, "
                "suggests corrections. Call this before create_nwchem_input to catch errors early."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "formula": {"type": "string", "description": "Molecular formula (e.g. 'C6H6')."},
                    "geometry_file": {"type": "string", "description": "Path to geometry file (.xyz)."},
                    "basis_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Element → basis name mapping.",
                    },
                    "ecp_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "default_basis": {"type": "string"},
                    "default_ecp": {"type": "string"},
                    "module": {"type": "string", "default": "dft", "description": "Calculation module (scf, dft, tce)."},
                    "task_operations": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Operations to perform (e.g. ['optimize', 'freq']).",
                    },
                    "functional": {"type": "string", "description": "DFT functional (e.g. 'b3lyp')."},
                    "charge": {"type": "integer"},
                    "multiplicity": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "review_nwchem_progress",
            "description": (
                "Check the progress of a running or recently completed NWChem job. "
                "Parses the output file, detects slow phases, reports convergence progress, "
                "and estimates remaining time. Works with both local and HPC jobs."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the NWChem output file."},
                    "input_file": {"type": "string"},
                    "error_file": {"type": "string", "description": "Path to stderr file (.err)."},
                    "process_id": {"type": "integer", "description": "PID for local jobs."},
                    "profile": {"type": "string", "description": "Runner profile name for HPC jobs."},
                    "job_id": {"type": "string", "description": "Scheduler job ID."},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "summarize_nwchem_output",
            "description": (
                "Generate a compact summary of an NWChem output file. "
                "Returns key results: final energy, convergence status, "
                "spin state, geometry quality, and any warnings. "
                "Lighter than analyze_nwchem_case — good for quick checks."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the NWChem output file."},
                    "input_file": {"type": "string"},
                    "expected_metals": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "expected_somos": {"type": "integer"},
                    "detail": {
                        "type": "string", "enum": ["summary", "full"], "default": "summary",
                        "description": "Level of detail.",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        # --- Phase 3: Campaign / scale management ---
        {
            "name": "register_nwchem_run",
            "description": (
                "Register a new run in the persistent run registry. "
                "Call this when submitting a job to track it across sessions. "
                "Returns a run_id for later updates. Optionally link to a campaign or workflow."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string", "description": "Job name (e.g. 'mol_opt')."},
                    "input_file": {"type": "string"},
                    "output_file": {"type": "string"},
                    "profile": {"type": "string", "description": "Runner profile name."},
                    "method": {"type": "string", "description": "E.g. 'DFT', 'CCSD(T)'."},
                    "functional": {"type": "string"},
                    "basis": {"type": "string"},
                    "n_atoms": {"type": "integer"},
                    "elements": {"type": "array", "items": {"type": "string"}},
                    "charge": {"type": "integer"},
                    "multiplicity": {"type": "integer"},
                    "mpi_ranks": {"type": "integer"},
                    "campaign_id": {"type": "integer", "description": "Link to a campaign."},
                    "workflow_id": {"type": "integer", "description": "Link to a workflow."},
                    "workflow_step_id": {"type": "string", "description": "Step ID within a workflow."},
                    "parent_run_id": {"type": "integer", "description": "Previous run in a restart chain."},
                    "tags": {"type": "object", "description": "Arbitrary metadata."},
                },
                "required": ["job_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "update_nwchem_run_status",
            "description": (
                "Update a registered run's status and optionally its results "
                "(energy, H, G, imaginary modes, walltime). "
                "Call after a job completes, fails, or is cancelled."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer", "description": "The run_id from register_nwchem_run."},
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
            "name": "list_nwchem_runs",
            "description": (
                "List registered runs, optionally filtered by campaign, workflow, status, or method. "
                "Returns the most recent runs first."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "campaign_id": {"type": "integer"},
                    "workflow_id": {"type": "integer"},
                    "status": {"type": "string"},
                    "method": {"type": "string"},
                    "limit": {"type": "integer", "description": "Max results (default 50)."},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "get_nwchem_run_summary",
            "description": (
                "Get detailed info for a single registered run, including its restart chain. "
                "Look up by run_id or job_name."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer"},
                    "job_name": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "create_nwchem_campaign",
            "description": (
                "Create a campaign to group related runs (e.g. a ligand screen, "
                "basis set convergence study, or conformer scan). "
                "Returns a campaign_id to pass when registering runs."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Unique campaign name."},
                    "description": {"type": "string"},
                    "tags": {"type": "object"},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_nwchem_campaign_status",
            "description": (
                "Get aggregate status for a campaign: total/completed/running/failed counts, "
                "completion percentage, and estimated remaining time."
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
            "name": "get_nwchem_campaign_energies",
            "description": (
                "Get an energy table for all completed runs in a campaign, "
                "sorted by energy with relative energies in kcal/mol. "
                "Includes H(T) and G(T) if available."
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
            "name": "create_nwchem_workflow",
            "description": (
                "Create a workflow DAG with step dependencies. Each step can depend on "
                "a previous step. Use advance_nwchem_workflow to check which steps are "
                "ready to launch. Steps: [{id, depends_on, input_file, profile, auto_input}]."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Workflow name."},
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
                        "description": "Workflow steps with dependencies.",
                    },
                    "protocol": {"type": "string"},
                    "campaign_id": {"type": "integer"},
                },
                "required": ["name", "steps"],
                "additionalProperties": False,
            },
        },
        {
            "name": "advance_nwchem_workflow",
            "description": (
                "Check a workflow's progress and return which steps are ready to launch. "
                "Does not launch jobs — the caller decides. Returns the workflow state, "
                "completed/running/failed steps, and a list of unblocked steps."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "integer", "description": "The workflow ID."},
                },
                "required": ["workflow_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "generate_nwchem_input_batch",
            "description": (
                "Generate multiple NWChem inputs by varying parameters from a template. "
                "Supports scanning over charge, multiplicity, task, memory, or any block.keyword "
                "(e.g. dft.xc for functionals). Generates all combinations (Cartesian product). "
                "Optionally registers all generated inputs in a campaign."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "template_input": {"type": "string", "description": "Path to the base .nw file."},
                    "vary": {
                        "type": "object",
                        "additionalProperties": {"type": "array"},
                        "description": "Parameters to vary. Keys are param names, values are lists. E.g. {'charge': [0,1,2], 'mult': [1,3,5]}.",
                    },
                    "output_dir": {"type": "string", "description": "Directory to write generated inputs."},
                    "naming_pattern": {"type": "string", "description": "Filename pattern. Placeholders: {stem}, {idx}, plus vary keys. Default: {stem}_{key}_{value}"},
                    "campaign_id": {"type": "integer", "description": "Register generated inputs in this campaign."},
                },
                "required": ["template_input", "vary", "output_dir"],
                "additionalProperties": False,
            },
        },
        {
            "name": "check_nwchem_memory_fit",
            "description": (
                "Check whether an NWChem input's memory directive fits the target node. "
                "Reads the 'memory total' line from the input file, multiplies by MPI ranks, "
                "and compares against the node's physical RAM (from the runner profile). "
                "Returns warnings if the job would crash with MA_init out-of-memory errors, "
                "and suggests a safe memory value. IMPORTANT: call this before launching, "
                "especially when switching profiles or changing node counts."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the .nw input file."},
                    "profile": {"type": "string", "description": "Runner profile name (reads node_memory_mb from it)."},
                    "profiles_path": {"type": "string", "description": "Custom profiles file path."},
                    "nodes": {"type": "integer", "description": "Override number of nodes."},
                    "mpi_ranks": {"type": "integer", "description": "Override total MPI ranks."},
                    "node_memory_mb": {"type": "integer", "description": "Override node RAM in MB (bypasses profile lookup)."},
                    "resource_overrides": {"type": "object", "description": "Resource overrides to merge into profile resources (e.g. from suggest_nwchem_resources)."},
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "estimate_nwchem_freq_walltime",
            "description": (
                "Estimate walltime needed for a numerical frequency calculation. "
                "NWChem numerical frequencies require 6*N_atoms gradient evaluations. "
                "CRITICAL: NWChem CANNOT checkpoint mid-frequency — if the job exceeds "
                "walltime, ALL progress is lost. This tool estimates total time and "
                "recommends multi-node scaling if the job won't fit in the walltime limit. "
                "Call this BEFORE launching any frequency job to avoid wasting compute time."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_atoms": {"type": "integer", "description": "Number of atoms in the molecule."},
                    "seconds_per_displacement": {"type": "number", "description": "Measured seconds per displacement (from a prior run). If omitted, uses heuristic estimate."},
                    "n_displacements": {"type": "integer", "description": "Total displacements (default: 6 * n_atoms for central differences)."},
                    "mpi_ranks": {"type": "integer", "description": "MPI ranks per node (default: 1)."},
                    "nodes": {"type": "integer", "description": "Number of nodes (default: 1)."},
                    "max_walltime_hours": {"type": "number", "description": "Maximum walltime in hours (default: 48)."},
                },
                "required": ["n_atoms"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_nwchem_resources",
            "description": (
                "Recommend optimal HPC resources for a NWChem job. Analyzes the input "
                "file (atoms, basis set, method, task type) and the runner profile's "
                "hardware specs (cores/node, memory/node, max nodes, max walltime) to "
                "recommend nodes, MPI ranks, walltime, and NWChem memory directive. "
                "Returns resource_overrides ready to pass to launch_nwchem_run. "
                "Call this BEFORE launching jobs on HPC to avoid wasting queue time "
                "with suboptimal resources. Requires the profile to have hardware "
                "fields populated (cores_per_node, node_memory_mb, etc.)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the NWChem .nw input file."},
                    "profile": {"type": "string", "description": "Runner profile name (e.g. 'stampede3_skx')."},
                    "profiles_path": {"type": "string", "description": "Optional path to runner profiles YAML/JSON."},
                },
                "required": ["input_file", "profile"],
                "additionalProperties": False,
            },
        },
        {
            "name": "detect_nwchem_hpc_accounts",
            "description": (
                "Detect available HPC allocation accounts for a runner profile. "
                "Runs the profile's account_command (e.g. /usr/local/etc/taccinfo on TACC) "
                "to discover project names, available SUs, and expiration dates. "
                "Returns the recommended account (most SUs available) ready to use in "
                "resource_overrides. Automatically called by suggest_nwchem_resources "
                "when account is not set, but can also be called standalone to check "
                "allocation status."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "profile": {"type": "string", "description": "Runner profile name (e.g. 'stampede3_skx')."},
                    "profiles_path": {"type": "string", "description": "Optional path to runner profiles YAML/JSON."},
                },
                "required": ["profile"],
                "additionalProperties": False,
            },
        },
        # ----- NWChem documentation tools (bundled docs) --------------------
        {
            "name": "list_nwchem_docs",
            "description": "List available bundled NWChem documentation files.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "search_nwchem_docs",
            "description": "Search the bundled NWChem documentation for syntax, keywords, or option details. Returns ranked excerpts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (keywords, directives, options)."},
                    "max_results": {"type": "integer", "default": 8, "description": "Maximum results to return."},
                    "context_lines": {"type": "integer", "default": 2, "description": "Lines of context around each match."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "lookup_nwchem_block_syntax",
            "description": "Look up NWChem input block syntax (e.g. scf, dft, mcscf, tce, vectors, geometry) from bundled docs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "block_name": {"type": "string", "description": "Block name to look up (e.g. 'scf', 'dft', 'tce')."},
                    "max_results": {"type": "integer", "default": 6},
                },
                "required": ["block_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "find_nwchem_examples",
            "description": "Search bundled NWChem example/tutorial documentation for a topic (e.g. fragment guess, mcscf, tce, dft).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to find examples for."},
                    "max_results": {"type": "integer", "default": 6},
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
        {
            "name": "read_nwchem_doc_excerpt",
            "description": "Read an excerpt from a bundled NWChem doc file by filename and line range, or around the first match for a query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "doc_name": {"type": "string", "description": "Doc filename (e.g. '11_QuantumMechanicalMethods.pdf.txt')."},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "query": {"type": "string", "description": "Find first occurrence of this text and show context around it."},
                    "context_lines": {"type": "integer", "default": 8},
                },
                "required": ["doc_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_nwchem_topic_guide",
            "description": "Get a curated documentation guide for a common NWChem topic: scf_open_shell, mcscf, fragment_guess, or tce.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic name (scf_open_shell, mcscf, fragment_guess, tce)."},
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
        # ----- NWChem community forum search --------------------------------
        {
            "name": "search_nwchem_forum",
            "description": (
                "Search the archived NWChem community forums for threads matching a query. "
                "Use this when encountering unusual NWChem errors, edge-case behavior, or "
                "issues that may have been discussed by the community. Fetches forum pages "
                "at runtime (requires internet). Returns matching thread titles, URLs, and "
                "optionally the thread content."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms (e.g. 'CCSD convergence', 'DFT grid error', 'segfault GA').",
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum threads to return (default 5).",
                    },
                    "fetch_content": {
                        "type": "boolean",
                        "default": True,
                        "description": "If true, fetch and include thread content (slower but more useful). If false, return titles and URLs only.",
                    },
                    "subforums": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict search to specific subforums. Options: 'Running NWChem', 'NWChem functionality', 'General Topics', 'Compiling NWChem', 'QM/MM'. Default: all.",
                    },
                },
                "required": ["query"],
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
        basis=arguments.get("basis"),
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


@_tool("compute_reaction_energy")
def _handle_compute_reaction_energy(arguments: dict[str, Any]) -> dict[str, Any]:
    return compute_reaction_energy(
        species=arguments["species"],
        reactants=arguments["reactants"],
        products=arguments["products"],
        method=arguments.get("method"),
        include_thermochem=arguments.get("include_thermochem", False),
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
    return summarize_electronic_structure(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
    )


@_tool("track_nwchem_spin_state")
def _handle_track_spin_state(arguments: dict[str, Any]) -> dict[str, Any]:
    return track_spin_state_across_optimization(
        output_path=arguments["output_file"],
    )


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


@_tool("suggest_resources")
def _handle_suggest_resources(arguments: dict[str, Any]) -> dict[str, Any]:
    hw = arguments.get("hw_specs")
    if not hw and arguments.get("profile"):
        from chemtools.runner import load_runner_profiles, _resolve_profile, query_partition_specs, get_local_resource_budget
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
            # Supplement with profile-level resources
            hw.setdefault("cpus_per_node", profile_payload.get("resources", {}).get("mpi_ranks"))
        else:
            hw = get_local_resource_budget()
    if not hw:
        from chemtools.runner import get_local_resource_budget
        hw = get_local_resource_budget()
    return suggest_resources(input_file=arguments["input_file"], hw_specs=hw)


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


@_tool("preflight_check")
def _handle_preflight_check(arguments: dict[str, Any]) -> dict[str, Any]:
    return preflight_check(
        input_file=arguments["input_file"],
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path") or os.environ.get("CHEMTOOLS_RUNNER_PROFILES"),
    )


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

@_tool("render_job_script")
def _handle_render_job_script(arguments: dict[str, Any]) -> dict[str, Any]:
    return render_job_script(
        input_path=arguments["input_file"],
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path") or os.environ.get("CHEMTOOLS_RUNNER_PROFILES"),
        job_name=arguments.get("job_name"),
        resource_overrides=arguments.get("resource_overrides"),
    )


@_tool("launch_nwchem_run")
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
    status["next_actions"] = _build_next_actions(
        "run_status", status,
        output_file=arguments.get("output_file", ""),
        input_file=arguments.get("input_file", ""),
        profile=arguments.get("profile", ""),
    )
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
    profiles_path = arguments.get("profiles_path") or os.environ.get("CHEMTOOLS_RUNNER_PROFILES")
    return terminate_nwchem_run(
        process_id=arguments.get("process_id"),
        signal_name=arguments.get("signal_name", "term"),
        job_id=arguments.get("job_id"),
        profile=arguments.get("profile"),
        profiles_path=profiles_path,
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

def _build_next_actions(
    context: str,
    result: dict[str, Any],
    output_file: str = "",
    input_file: str = "",
    profile: str = "",
) -> list[dict[str, Any]]:
    """Build a structured next_actions list from analysis results.

    Each action is a dict with: priority, tool, params, reason, confidence.
    The model can execute actions[0] without understanding NWChem internals.
    """
    actions: list[dict[str, Any]] = []

    if context == "analyze_case":
        diagnosis = result.get("diagnosis") or {}
        task_outcome = diagnosis.get("task_outcome", "")
        failure_class = diagnosis.get("failure_class", "")
        next_step = result.get("next_step") or {}
        selected_workflow = next_step.get("selected_workflow", "")

        if task_outcome == "success":
            wf = selected_workflow.lower()
            if "tce" in wf or "ccsd" in wf or "mp2" in wf:
                actions.append({
                    "priority": 1,
                    "tool": "parse_nwchem_tce_output",
                    "params": {"output_file": output_file},
                    "reason": "Correlated calculation completed — extract energies and T1/D1 diagnostics.",
                    "confidence": 0.95,
                })
            elif "freq" in wf:
                actions.append({
                    "priority": 1,
                    "tool": "check_nwchem_freq_plausibility",
                    "params": {"output_file": output_file, "input_file": input_file},
                    "reason": "Frequency calculation completed — verify plausibility before using results.",
                    "confidence": 0.95,
                })
            elif "opt" in wf or "geometry" in wf:
                actions.append({
                    "priority": 1,
                    "tool": "extract_nwchem_geometry",
                    "params": {"output_file": output_file, "frame": "best"},
                    "reason": "Optimization converged — extract geometry for next step.",
                    "confidence": 0.90,
                })
            else:
                actions.append({
                    "priority": 1,
                    "tool": "parse_nwchem_output",
                    "params": {"output_file": output_file, "sections": ["tasks"]},
                    "reason": "Calculation completed — review results.",
                    "confidence": 0.80,
                })
        elif task_outcome in ("failed", "error", "scf_failed"):
            if failure_class == "scf_convergence":
                actions.append({
                    "priority": 1,
                    "tool": "suggest_nwchem_recovery",
                    "params": {"output_file": output_file, "input_file": input_file, "mode": "scf"},
                    "reason": "SCF convergence failure — get targeted recovery strategies.",
                    "confidence": 0.90,
                })
            elif failure_class in ("bad_state", "wrong_state", "state_mismatch"):
                actions.append({
                    "priority": 1,
                    "tool": "suggest_nwchem_recovery",
                    "params": {"output_file": output_file, "input_file": input_file, "mode": "state"},
                    "reason": "Spin/state error — recover with state correction strategies.",
                    "confidence": 0.85,
                })
            elif failure_class in ("memory", "oom", "ma_init"):
                actions.append({
                    "priority": 1,
                    "tool": "create_nwchem_input_variant",
                    "params": {
                        "source_input": input_file,
                        "changes": {"memory": "800 mb"},
                        "reason": f"OOM failure ({failure_class}) — reduce memory",
                    },
                    "reason": "Out of memory — reduce memory directive and resubmit.",
                    "confidence": 0.80,
                })
            else:
                actions.append({
                    "priority": 1,
                    "tool": "suggest_nwchem_recovery",
                    "params": {"output_file": output_file, "input_file": input_file, "mode": "auto"},
                    "reason": f"Calculation failed ({failure_class or 'unknown'}) — get recovery recommendations.",
                    "confidence": 0.75,
                })
        elif task_outcome == "timelimit":
            actions.append({
                "priority": 1,
                "tool": "prepare_nwchem_freq_restart",
                "params": {"input_file": input_file, "output_file": output_file, "profile": profile},
                "reason": "Hit walltime limit — check if freq restart is ready.",
                "confidence": 0.85,
            })

    elif context == "watch_run":
        overall = result.get("overall_status", "")
        tasks = result.get("progress_summary", {}).get("tasks", []) if result.get("progress_summary") else []
        has_tce = any((t.get("module") or "").lower() == "tce" for t in tasks)
        has_freq = any((t.get("module") or "").lower() in {"freq", "frequency"} for t in tasks)
        has_opt = any((t.get("operation") or "").lower() == "optimize" for t in tasks)

        if overall == "completed":
            actions.append({
                "priority": 1,
                "tool": "analyze_nwchem_case",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "Job completed — run full analysis to determine next steps.",
                "confidence": 0.95,
            })
        elif overall in ("failed", "error"):
            actions.append({
                "priority": 1,
                "tool": "analyze_nwchem_case",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "Job failed — diagnose the failure.",
                "confidence": 0.90,
            })
        elif overall == "running":
            actions.append({
                "priority": 1,
                "tool": "watch_nwchem_run",
                "params": {"output_file": output_file, "input_file": input_file, "profile": profile},
                "reason": "Job is still running — continue monitoring.",
                "confidence": 0.95,
            })

    elif context == "freq_plausibility":
        assessment = result.get("overall_assessment", "")
        imag_count = result.get("imaginary_mode_count", 0)
        if assessment == "suspicious" and imag_count and imag_count > 0:
            actions.append({
                "priority": 1,
                "tool": "analyze_nwchem_imaginary_modes",
                "params": {"output_file": output_file},
                "reason": f"Found {imag_count} imaginary mode(s) — analyze which atoms are involved.",
                "confidence": 0.90,
            })
            actions.append({
                "priority": 2,
                "tool": "draft_nwchem_imaginary_mode_inputs",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "Generate displaced geometries for re-optimization.",
                "confidence": 0.80,
            })
        elif assessment in ("ok", "plausible"):
            actions.append({
                "priority": 1,
                "tool": "parse_nwchem_output",
                "params": {"output_file": output_file, "sections": ["freq", "tasks"]},
                "reason": "Frequencies look reasonable — extract thermochemistry data.",
                "confidence": 0.90,
            })

    elif context == "run_status":
        status = result.get("status", "")
        if status in ("completed", "done"):
            actions.append({
                "priority": 1,
                "tool": "analyze_nwchem_case",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "Job completed — run full analysis.",
                "confidence": 0.95,
            })
        elif status in ("failed", "error"):
            actions.append({
                "priority": 1,
                "tool": "analyze_nwchem_case",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "Job failed — diagnose the failure.",
                "confidence": 0.90,
            })
        elif status in ("cancelled", "timelimit"):
            actions.append({
                "priority": 1,
                "tool": "prepare_nwchem_freq_restart",
                "params": {"input_file": input_file, "output_file": output_file, "profile": profile},
                "reason": "Job cancelled/timelimit — check if restart is possible.",
                "confidence": 0.80,
            })

    elif context == "imaginary_modes":
        sig_count = result.get("significant_imaginary_mode_count", 0)
        if sig_count > 0:
            actions.append({
                "priority": 1,
                "tool": "draft_nwchem_imaginary_mode_inputs",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": f"{sig_count} significant imaginary mode(s) — generate displaced inputs for re-optimization.",
                "confidence": 0.85,
            })
        else:
            actions.append({
                "priority": 1,
                "tool": "parse_nwchem_output",
                "params": {"output_file": output_file, "sections": ["freq", "tasks"]},
                "reason": "No significant imaginary modes — extract thermochemistry data.",
                "confidence": 0.90,
            })

    return actions


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
        err_file=arguments.get("err_file"),
        library_path=basis_library_path(arguments.get("library_path")),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        compact=compact,
    )
    result["recommended_next_tool"] = _derive_recommended_next_tool(result)
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
    return check_nwchem_geometry_plausibility(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        frame=frame_arg,
    )


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
    return parse_freq_progress(arguments["output_file"])


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
    result = parse_tce_output(arguments["output_file"])
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


@_tool("draft_nwchem_tce_restart_input")
def _handle_draft_nwchem_tce_restart_input(arguments: dict[str, Any]) -> dict[str, Any]:
    result = draft_nwchem_tce_restart_input(
        tce_output_file=arguments["tce_output_file"],
        tce_input_file=arguments.get("tce_input_file"),
        max_iterations=arguments.get("max_iterations", 200),
        thresh=arguments.get("thresh", 1e-5),
        copy_amplitudes=arguments.get("copy_amplitudes", True),
        output_dir=arguments.get("output_dir"),
        write_file=arguments.get("write_file", False),
    )
    nw_file = result.get("written_file") or result.get("planned_output_file", "<restart.nw>")
    next_steps = []
    if result.get("copy_errors"):
        next_steps.append(
            "WARNING: amplitude file copy failed — check copy_errors. "
            "Manually copy .t1amp.* → {start_name}.t1 and .t2amp.* → {start_name}.t2 before running."
        )
    if result.get("can_read_amplitudes"):
        next_steps.append(f"Amplitude files are in place. Call lint_nwchem_input(input_file='{nw_file}') then launch_nwchem_run to continue.")
    else:
        next_steps.append(f"Amplitude files NOT found. Input has 'set tce:read_ta' commented out — NWChem will restart from scratch with higher maxiter.")
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


# ---------------------------------------------------------------------------
# Handlers — parallel job monitoring, session log, input versioning
# ---------------------------------------------------------------------------

@_tool("watch_multiple_runs")
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

@_tool("basis_library_summary")
def _handle_basis_library_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    return basis_library_summary(
        library_path=basis_library_path(arguments.get("library_path")),
    )


@_tool("check_nwchem_spin_charge_state")
def _handle_check_spin_charge_state(arguments: dict[str, Any]) -> dict[str, Any]:
    return check_spin_charge_state(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
    )


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
    return review_nwchem_progress(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        error_path=arguments.get("error_file"),
        process_id=arguments.get("process_id"),
        profile=arguments.get("profile"),
        job_id=arguments.get("job_id"),
    )


@_tool("summarize_nwchem_output")
def _handle_summarize_nwchem_output(arguments: dict[str, Any]) -> dict[str, Any]:
    return summarize_output(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
        detail_level=arguments.get("detail", "summary"),
    )


# ---------------------------------------------------------------------------
# Handlers — run registry, campaigns, workflows, batch generation
# ---------------------------------------------------------------------------

@_tool("register_nwchem_run")
def _handle_register_run(arguments: dict[str, Any]) -> dict[str, Any]:
    return register_run(
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


@_tool("update_nwchem_run_status")
def _handle_update_run_status(arguments: dict[str, Any]) -> dict[str, Any]:
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


@_tool("list_nwchem_runs")
def _handle_list_runs(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"runs": list_runs(
        campaign_id=arguments.get("campaign_id"),
        workflow_id=arguments.get("workflow_id"),
        status=arguments.get("status"),
        method=arguments.get("method"),
        limit=arguments.get("limit", 50),
    )}


@_tool("get_nwchem_run_summary")
def _handle_get_run_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_run_summary(
        run_id=arguments.get("run_id"),
        job_name=arguments.get("job_name"),
    )


@_tool("create_nwchem_campaign")
def _handle_create_campaign(arguments: dict[str, Any]) -> dict[str, Any]:
    return create_campaign(
        name=arguments["name"],
        description=arguments.get("description"),
        tags=arguments.get("tags"),
    )


@_tool("get_nwchem_campaign_status")
def _handle_get_campaign_status(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_campaign_status(
        campaign_id=arguments.get("campaign_id"),
        name=arguments.get("name"),
    )


@_tool("get_nwchem_campaign_energies")
def _handle_get_campaign_energies(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_campaign_energies(
        campaign_id=arguments.get("campaign_id"),
        name=arguments.get("name"),
    )


@_tool("create_nwchem_workflow")
def _handle_create_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    return create_workflow(
        name=arguments["name"],
        steps=arguments["steps"],
        protocol=arguments.get("protocol"),
        campaign_id=arguments.get("campaign_id"),
    )


@_tool("advance_nwchem_workflow")
def _handle_advance_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    return advance_workflow(
        workflow_id=arguments["workflow_id"],
    )


@_tool("generate_nwchem_input_batch")
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


@_tool("check_nwchem_memory_fit")
def _handle_check_memory_fit(arguments: dict[str, Any]) -> dict[str, Any]:
    profile_resources = None
    if arguments.get("profile"):
        from chemtools.runner import load_runner_profiles, _resolve_profile
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


@_tool("estimate_nwchem_freq_walltime")
def _handle_estimate_freq_walltime(arguments: dict[str, Any]) -> dict[str, Any]:
    return estimate_freq_walltime(
        n_atoms=arguments["n_atoms"],
        seconds_per_displacement=arguments.get("seconds_per_displacement"),
        n_displacements=arguments.get("n_displacements"),
        mpi_ranks=arguments.get("mpi_ranks", 1),
        nodes=arguments.get("nodes", 1),
        max_walltime_hours=arguments.get("max_walltime_hours", 48.0),
    )


@_tool("suggest_nwchem_resources")
def _handle_suggest_hpc_resources(arguments: dict[str, Any]) -> dict[str, Any]:
    return suggest_hpc_resources(
        input_file=arguments["input_file"],
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path"),
    )


@_tool("detect_nwchem_hpc_accounts")
def _handle_detect_hpc_accounts(arguments: dict[str, Any]) -> dict[str, Any]:
    return detect_hpc_accounts(
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path"),
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


# Backward-compat aliases: old tool names → (current name, arg translator).
# These are NOT in tool_definitions() so models see only the current names.
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

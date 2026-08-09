"""NWChem protocol library + tool-name bindings for the generic DAG engine.

The PROTOCOLS dict below is the NWChem-specific recipe library; the
engine that walks it (``plan_calculation``, ``list_protocols``) lives in
``chemtools.core.workflow`` and is program-agnostic. This module just
hands the engine the NWChem protocol library, the NWChem tool-name
mapping, and the NWChem dynamic-step generators.
"""
from __future__ import annotations

from typing import Any

from chemtools.core.workflow import (
    list_protocols as _core_list_protocols,
    plan_calculation as _core_plan_calculation,
)


# ---------------------------------------------------------------------------
# Protocol definitions
# ---------------------------------------------------------------------------

PROTOCOLS: dict[str, dict[str, Any]] = {
    "single_point_dft": {
        "description": "Single-point DFT energy evaluation",
        "steps": [
            {"id": "energy", "task": "dft energy", "depends_on": None},
        ],
        "method": "dft",
        "functional": "b3lyp",
        "basis_rule": "suggest_basis_set(purpose='energy')",
    },
    "geometry_opt_dft": {
        "description": "Standard DFT geometry optimization",
        "steps": [
            {"id": "opt", "task": "dft optimize", "depends_on": None},
        ],
        "method": "dft",
        "functional": "b3lyp",
        "basis_rule": "suggest_basis_set(purpose='geometry')",
    },
    "thermochem_dft": {
        "description": "Full thermochemistry: optimize then frequency analysis",
        "steps": [
            {"id": "opt", "task": "dft optimize", "depends_on": None},
            {"id": "freq", "task": "dft freq", "depends_on": "opt",
             "auto_input": "extract_geometry_and_switch_to_freq"},
        ],
        "method": "dft",
        "functional": "b3lyp",
        "post_process": ["check_nwchem_freq_plausibility"],
        "checks": ["no_imaginary_modes"],
        "on_imaginary_modes": "displace_and_reopt",
    },
    "opt_then_tce": {
        "description": "DFT geometry optimization followed by correlated single point (CCSD(T))",
        "steps": [
            {"id": "opt", "task": "dft optimize", "depends_on": None},
            {"id": "sp_tce", "task": "tce energy", "depends_on": "opt",
             "auto_input": "extract_geometry_for_tce"},
        ],
        "method": "tce",
        "dft_functional": "b3lyp",
        "tce_method": "ccsd(t)",
    },
    "basis_set_convergence": {
        "description": "Run the same calculation with progressively larger basis sets",
        "steps": [
            {"id": "small", "task": "dft energy", "basis_override": "6-31G*", "depends_on": None},
            {"id": "medium", "task": "dft energy", "basis_override": "cc-pVDZ", "depends_on": None},
            {"id": "large", "task": "dft energy", "basis_override": "cc-pVTZ", "depends_on": None},
        ],
        "method": "dft",
        "functional": "b3lyp",
        "parallel_independent": True,
    },
    "spin_state_scan": {
        "description": "Optimize at multiple spin multiplicities to find ground state",
        "steps": [
            # Steps are generated dynamically by plan_calculation based on
            # the element/charge combination
        ],
        "method": "dft",
        "functional": "b3lyp",
        "dynamic": True,
        "dynamic_generator": "spin_states",
    },
    "freq_only": {
        "description": "Frequency analysis at a previously optimized geometry",
        "steps": [
            {"id": "freq", "task": "dft freq", "depends_on": None},
        ],
        "method": "dft",
        "functional": "b3lyp",
        "post_process": ["check_nwchem_freq_plausibility"],
        "checks": ["no_imaginary_modes"],
        "on_imaginary_modes": "displace_and_reopt",
    },
    "tce_single_point": {
        "description": "Correlated single-point energy (CCSD or CCSD(T)) at existing geometry",
        "steps": [
            {"id": "scf", "task": "scf energy", "depends_on": None},
            {"id": "tce", "task": "tce energy", "depends_on": "scf",
             "auto_input": "reuse_vectors_for_tce"},
        ],
        "method": "tce",
        "tce_method": "ccsd(t)",
        "post_process": ["parse_nwchem_tce_output"],
    },
    "relativistic_dft": {
        "description": "DFT with scalar relativistic corrections (DKH2 or X2C) for heavy elements",
        "steps": [
            {"id": "energy", "task": "dft energy", "depends_on": None},
        ],
        "method": "dft",
        "functional": "b3lyp",
        "relativistic": "dkh2",
        "basis_rule": "suggest_basis_set(purpose='energy', relativistic=True)",
    },
    "thermochem_opt_freq_reopt": {
        "description": "Optimize, frequency, check for imaginary modes, re-optimize if needed",
        "steps": [
            {"id": "opt", "task": "dft optimize", "depends_on": None},
            {"id": "freq", "task": "dft freq", "depends_on": "opt",
             "auto_input": "extract_geometry_and_switch_to_freq"},
            {"id": "reopt", "task": "dft optimize", "depends_on": "freq",
             "auto_input": "displace_along_imaginary_mode",
             "conditional": "has_imaginary_modes"},
            {"id": "freq2", "task": "dft freq", "depends_on": "reopt",
             "auto_input": "extract_geometry_and_switch_to_freq",
             "conditional": "reopt_ran"},
        ],
        "method": "dft",
        "functional": "b3lyp",
        "post_process": ["check_nwchem_freq_plausibility"],
        "checks": ["no_imaginary_modes"],
    },
    "solvation_comparison": {
        "description": "Compare gas-phase and COSMO solvation energies",
        "steps": [
            {"id": "gas", "task": "dft energy", "depends_on": None},
            {"id": "solvent", "task": "dft energy", "depends_on": None,
             "cosmo": True, "cosmo_solvent": "water"},
        ],
        "method": "dft",
        "functional": "b3lyp",
        "parallel_independent": True,
    },
    "vertical_excitation_tddft": {
        "description": "TDDFT vertical excitation energies at a ground-state geometry",
        "steps": [
            {"id": "gs", "task": "dft energy", "depends_on": None},
            {"id": "tddft", "task": "tddft energy", "depends_on": "gs",
             "auto_input": "reuse_vectors_for_tddft"},
        ],
        "method": "tddft",
        "functional": "b3lyp",
        "n_roots": 10,
    },
    "reaction_energy": {
        "description": "Compute reaction energy from reactant and product single points",
        "steps": [
            # Steps are generated dynamically from a list of species
        ],
        "method": "dft",
        "functional": "b3lyp",
        "dynamic": True,
        "dynamic_generator": "reaction_species",
        "post_process": ["compute_reaction_energy"],
    },
}


# NWChem tool-name mapping for the core/workflow engine.
_NWCHEM_TOOL_NAMES: dict[str, str] = {
    "check_freq": "check_nwchem_freq_plausibility",
    "check_geom": "check_nwchem_geometry_plausibility",
    "workflow_state": "get_nwchem_workflow_state",
    "extract_geom": "extract_nwchem_geometry",
    "input_variant": "create_nwchem_input_variant",
    "launch": "launch_nwchem_run",
}


def list_protocols() -> list[dict[str, str]]:
    """Return a summary of all available NWChem protocols."""
    return _core_list_protocols(PROTOCOLS)


def plan_calculation(
    input_file: str,
    protocol: str,
    profile: str = "",
    output_dir: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Given a molecule input and a protocol name, build a step-by-step plan.

    Thin wrapper over ``chemtools.core.workflow.plan_calculation`` that
    injects the NWChem protocol library, tool-name mapping, and dynamic
    step generators.
    """
    return _core_plan_calculation(
        PROTOCOLS,
        input_file=input_file,
        protocol=protocol,
        profile=profile,
        output_dir=output_dir,
        overrides=overrides,
        tool_names=_NWCHEM_TOOL_NAMES,
        dynamic_generators={
            "spin_states": _generate_spin_scan_steps,
            "reaction_species": _generate_reaction_steps,
        },
    )


def _generate_spin_scan_steps(
    input_file: str,
    overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate optimization steps for each plausible spin multiplicity."""
    mults = overrides.get("multiplicities")
    if not mults:
        # Default: try a few common multiplicities
        mults = [1, 3, 5]
    steps = []
    for mult in mults:
        steps.append({
            "id": f"mult{mult}",
            "task": "dft optimize",
            "depends_on": None,
            "mult_override": mult,
        })
    return steps


def _generate_reaction_steps(
    input_file: str,
    overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate single-point steps for each species in a reaction."""
    species = overrides.get("species", [])
    if not species:
        return [{"id": "species_0", "task": "dft energy", "depends_on": None}]
    steps = []
    for i, sp in enumerate(species):
        label = sp.get("label", f"species_{i}")
        steps.append({
            "id": label,
            "task": "dft energy",
            "depends_on": None,
            "species_info": sp,
        })
    return steps

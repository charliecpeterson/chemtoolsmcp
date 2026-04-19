"""Pre-baked calculation protocols for NWChem workflows.

Protocols encode multi-step calculation recipes so a model (even a cheap one)
can drive the full workflow without understanding NWChem internals.  The model
calls ``plan_calculation`` to get the step list, then executes each step.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


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


def list_protocols() -> list[dict[str, str]]:
    """Return a summary of all available protocols."""
    return [
        {"name": name, "description": proto["description"]}
        for name, proto in PROTOCOLS.items()
    ]


def plan_calculation(
    input_file: str,
    protocol: str,
    profile: str = "",
    output_dir: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Given a molecule input and a protocol name, generate a step-by-step plan.

    Returns a workflow plan with step descriptions and the tool calls needed
    to execute each step.  The model follows the plan sequentially (or in
    parallel where ``parallel_independent`` is True).
    """
    if protocol not in PROTOCOLS:
        available = ", ".join(sorted(PROTOCOLS.keys()))
        raise ValueError(f"Unknown protocol '{protocol}'. Available: {available}")

    proto = PROTOCOLS[protocol]
    inp = Path(input_file)
    if output_dir is None:
        output_dir = str(inp.parent)
    base_stem = re.sub(r"_v\d+$", "", inp.stem)
    overrides = overrides or {}

    # Build steps
    steps = proto.get("steps", [])
    if proto.get("dynamic"):
        gen = proto.get("dynamic_generator")
        if gen == "spin_states":
            steps = _generate_spin_scan_steps(input_file, overrides)
        elif gen == "reaction_species":
            steps = _generate_reaction_steps(input_file, overrides)
        # else: keep static steps

    plan_steps: list[dict[str, Any]] = []
    for step in steps:
        step_id = step["id"]
        task_str = step.get("task", "dft energy")
        depends = step.get("depends_on")

        step_input = input_file if depends is None else f"<from step '{depends}'>"
        step_output = str(Path(output_dir) / f"{base_stem}_{step_id}.out")

        tool_params: dict[str, Any] = {
            "input_file": step_input,
            "profile": profile,
        }
        if step.get("basis_override"):
            tool_params["basis_override"] = step["basis_override"]

        auto_input_action = step.get("auto_input")
        pre_actions: list[dict[str, Any]] = []
        if auto_input_action and depends:
            if "extract_geometry" in auto_input_action:
                pre_actions.append({
                    "tool": "extract_nwchem_geometry",
                    "params": {"output_file": f"<output of step '{depends}'>", "frame": "best"},
                    "purpose": "Get optimized geometry for next step",
                })
            if "freq" in auto_input_action:
                pre_actions.append({
                    "tool": "create_nwchem_input_variant",
                    "params": {
                        "source_input": input_file,
                        "changes": {"task": task_str},
                        "reason": f"Switch task to {task_str} for protocol step '{step_id}'",
                    },
                    "purpose": f"Create input for {task_str}",
                })

        plan_steps.append({
            "step_id": step_id,
            "task": task_str,
            "depends_on": depends,
            "expected_output": step_output,
            "pre_actions": pre_actions,
            "launch_action": {
                "tool": "launch_nwchem_run",
                "params": tool_params,
            },
            "post_actions": _post_actions_for_step(step, proto, step_output, input_file),
        })

    # Post-process checks
    post_checks: list[dict[str, Any]] = []
    for check_name in proto.get("post_process", []):
        post_checks.append({
            "tool": check_name,
            "params": {"output_file": "<final output>"},
        })

    return {
        "protocol": protocol,
        "description": proto["description"],
        "input_file": input_file,
        "output_dir": output_dir,
        "profile": profile,
        "n_steps": len(plan_steps),
        "parallel_independent": proto.get("parallel_independent", False),
        "steps": plan_steps,
        "post_checks": post_checks,
        "on_imaginary_modes": proto.get("on_imaginary_modes"),
    }


def _post_actions_for_step(
    step: dict[str, Any],
    proto: dict[str, Any],
    output_file: str,
    input_file: str,
) -> list[dict[str, Any]]:
    """Build post-completion actions for a step."""
    actions: list[dict[str, Any]] = []
    task = step.get("task", "")
    if "freq" in task:
        actions.append({
            "tool": "check_nwchem_freq_plausibility",
            "params": {"output_file": output_file},
            "purpose": "Verify frequencies are physically reasonable",
        })
    elif "optimize" in task:
        actions.append({
            "tool": "check_nwchem_geometry_plausibility",
            "params": {"output_file": output_file},
            "purpose": "Verify optimized geometry is reasonable",
        })
    actions.append({
        "tool": "get_nwchem_workflow_state",
        "params": {"input_file": input_file, "output_file": output_file},
        "purpose": "Check workflow state and determine if step succeeded",
    })
    return actions


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

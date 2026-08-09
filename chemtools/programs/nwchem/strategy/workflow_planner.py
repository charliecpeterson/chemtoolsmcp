"""NWChem workflow planning and next-step orchestration.

Two strategy-level entry points that compose the per-family drafters into
high-level workflows:

  * prepare_nwchem_next_step    Given a finished or failed NWChem run,
                                 produce a "what to do next" payload —
                                 typically a drafted follow-up input
                                 (recovery, property check, frontier
                                 cubes, etc.) plus a Diagnosis envelope.
                                 The orchestrator behind multiple MCP
                                 tools.

  * plan_nwchem_workflow         Given a goal (opt_freq, single_point_ccsd,
                                 ...), elements, charge, and multiplicity,
                                 return a concrete step-by-step plan of
                                 tool calls and parameters. Used by
                                 agents starting fresh calculations.

Both functions compose across `programs/nwchem/input/*` drafter modules,
which is why they live under `strategy/` rather than `input/`: they
orchestrate the input layer rather than producing inputs themselves.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Any

from chemtools.core.common import read_text
from chemtools.programs.nwchem.parse.input import inspect_nwchem_input
from chemtools.programs.nwchem.strategy.diagnose import summarize_nwchem_output
from chemtools.programs.nwchem.scf_quality import (
    find_converged_scf_excursion,
)
from chemtools.programs.nwchem.input._utils import _summarize_prepared_artifact

# Drafters used by prepare_nwchem_next_step and plan_nwchem_workflow.
from chemtools.programs.nwchem.input.geometry import (
    extract_nwchem_geometry,
    draft_initial_geometry,
)
from chemtools.programs.nwchem.input.general import (
    create_nwchem_input,
    review_nwchem_input_request,
)
from chemtools.programs.nwchem.input.dft import (
    create_nwchem_dft_workflow_input,
)
from chemtools.programs.nwchem.input.scf_recovery import (
    draft_nwchem_vectors_swap_input,
    draft_nwchem_property_check_input,
    draft_nwchem_scf_stabilization_input,
)
from chemtools.programs.nwchem.input.imaginary_modes import (
    draft_nwchem_imaginary_mode_inputs,
)
from chemtools.programs.nwchem.input.opt_followup import (
    draft_nwchem_optimization_followup_input,
)
from chemtools.programs.nwchem.input.cube import (
    draft_nwchem_frontier_cube_input,
)
from chemtools.programs.nwchem.input.tce import (
    draft_nwchem_tce_input,
)
from chemtools.programs.nwchem.input.mcscf import (
    draft_nwchem_mcscf_input,
)


def prepare_nwchem_next_step(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
    write_files: bool = False,
    include_property_check: bool = True,
    include_frontier_cubes: bool = False,
    include_density_modes: list[str] | None = None,
    cube_extent_angstrom: float = 6.0,
    cube_grid_points: int = 120,
    _precomputed_summary: "dict | None" = None,
) -> dict[str, Any]:
    if _precomputed_summary is not None:
        summary = _precomputed_summary
    else:
        summary = summarize_nwchem_output(
            output_path=output_path,
            input_path=input_path,
            expected_metal_elements=expected_metal_elements,
            expected_somo_count=expected_somo_count,
            detail_level="full",
        )
    diagnosis = summary["diagnosis"]
    failure_class = diagnosis["failure_class"]

    prepared_artifacts: dict[str, Any] = {}
    artifact_order: list[str] = []
    notes: list[str] = []
    can_auto_prepare = False
    selected_workflow = "manual_review"
    stage = diagnosis["stage"]
    trajectory = diagnosis.get("trajectory") or {}
    converged_excursion = find_converged_scf_excursion(
        diagnosis.get("scf") or {}
    )
    trigger_evidence: dict[str, Any] = {}

    if (
        stage == "optimization"
        and trajectory.get("optimization_status") is not None
    ):
        selected_workflow = "optimization_follow_up"
        if not input_path:
            notes.append("input_file_required_for_optimization_follow_up")
        else:
            can_auto_prepare = True
            optimization_base_name = base_name or None
            prepared_artifacts["optimization_follow_up"] = draft_nwchem_optimization_followup_input(
                output_path=output_path,
                input_path=input_path,
                output_dir=output_dir,
                base_name=optimization_base_name,
                write_file=write_files,
            )
            artifact_order.append("optimization_follow_up")
            if failure_class == "wrong_state_convergence":
                notes.append("optimization_follow_up_prioritized_over_wrong_state_for_optimization_stage")
    elif (
        stage == "frequency"
        and diagnosis["task_outcome"] != "success"
        and trajectory.get("optimization_status") == "converged"
    ):
        selected_workflow = "post_optimization_frequency_follow_up"
        if not input_path:
            notes.append("input_file_required_for_frequency_follow_up")
        else:
            can_auto_prepare = True
            optimization_base_name = base_name or None
            prepared_artifacts["optimization_follow_up"] = draft_nwchem_optimization_followup_input(
                output_path=output_path,
                input_path=input_path,
                task_strategy="freq_only",
                output_dir=output_dir,
                base_name=optimization_base_name,
                write_file=write_files,
            )
            artifact_order.append("optimization_follow_up")
    elif failure_class == "wrong_state_convergence":
        selected_workflow = "wrong_state_swap_recovery"
        if not input_path:
            notes.append("input_file_required_for_swap_restart")
        elif not diagnosis.get("swap_suggestion", {}).get("available"):
            notes.append("no_actionable_vectors_swap_identified")
        else:
            can_auto_prepare = True
            swap_base_name = base_name or f"{Path(input_path).stem}_swap"
            swap_restart = draft_nwchem_vectors_swap_input(
                output_path=output_path,
                input_path=input_path,
                expected_metal_elements=expected_metal_elements,
                expected_somo_count=expected_somo_count,
                output_dir=output_dir,
                base_name=swap_base_name,
                write_file=write_files,
            )
            prepared_artifacts["swap_restart"] = swap_restart
            artifact_order.append("swap_restart")

            if include_property_check:
                property_base_name = f"{swap_base_name}_prop"
                property_check = draft_nwchem_property_check_input(
                    input_path=input_path,
                    reference_output_path=output_path,
                    vectors_input=swap_restart["vectors_output"],
                    vectors_output=f"{property_base_name}.movecs",
                    expected_metal_elements=expected_metal_elements,
                    expected_somo_count=expected_somo_count,
                    output_dir=output_dir,
                    base_name=property_base_name,
                    write_file=write_files,
                )
                prepared_artifacts["property_check"] = property_check
                artifact_order.append("property_check")

            if include_frontier_cubes:
                cube_base_name = f"{swap_base_name}_frontier"
                frontier_cubes = draft_nwchem_frontier_cube_input(
                    output_path=output_path,
                    input_path=input_path,
                    vectors_input=swap_restart["vectors_output"],
                    include_density_modes=include_density_modes,
                    extent_angstrom=cube_extent_angstrom,
                    grid_points=cube_grid_points,
                    output_dir=output_dir,
                    base_name=cube_base_name,
                    write_file=write_files,
                )
                prepared_artifacts["frontier_cubes"] = frontier_cubes
                artifact_order.append("frontier_cubes")
    elif failure_class == "frequency_interpretation_required":
        selected_workflow = "imaginary_mode_follow_up"
        if not input_path:
            notes.append("input_file_required_for_imaginary_mode_restart")
        else:
            can_auto_prepare = True
            imaginary_base_name = base_name or f"{Path(input_path).stem}_imaginary_followup"
            prepared_artifacts["imaginary_mode_restarts"] = draft_nwchem_imaginary_mode_inputs(
                output_path=output_path,
                input_path=input_path,
                output_dir=output_dir,
                base_name=imaginary_base_name,
                write_files=write_files,
            )
            artifact_order.append("imaginary_mode_restarts")
    elif failure_class == "scf_nonconvergence":
        selected_workflow = "scf_stabilization_restart"
        if not input_path:
            notes.append("input_file_required_for_scf_stabilization_restart")
        else:
            can_auto_prepare = True
            stabilize_base_name = base_name or f"{Path(input_path).stem}_stabilize"
            prepared_artifacts["scf_stabilization"] = draft_nwchem_scf_stabilization_input(
                input_path=input_path,
                reference_output_path=output_path,
                output_dir=output_dir,
                base_name=stabilize_base_name,
                write_file=write_files,
            )
            artifact_order.append("scf_stabilization")
    elif (
        failure_class == "no_clear_failure_detected"
        and converged_excursion is not None
    ):
        selected_workflow = "scf_stability_hardening"
        multi_stage_input = False
        if input_path:
            input_summary = inspect_nwchem_input(input_path)
            multi_stage_input = (
                len(input_summary.get("task_states") or []) > 1
                or any(
                    block.get("fragment_inputs")
                    for block in input_summary.get("start_blocks") or []
                )
            )
        prepared_status = (
            "input_required"
            if not input_path
            else "manual_edit_required"
            if multi_stage_input
            else "prepared"
        )
        trigger_evidence = {
            "scf_instability": converged_excursion,
            "multi_stage_input": multi_stage_input,
            "strategy_options": [
                {
                    "name": "reuse_converged_vectors_with_damping",
                    "status": prepared_status,
                    "rationale": (
                        "The completed run already produced target-basis "
                        "orbitals, so reusing them is the shortest controlled "
                        "repeat when the checkpoint is available."
                    ),
                },
                {
                    "name": "smaller_basis_projection",
                    "status": "manual_fallback",
                    "rationale": (
                        "Use a reviewed smaller-basis seed when no trustworthy "
                        "target-basis checkpoint exists. Inspect that seed's "
                        "own SCF path before projecting it."
                    ),
                },
            ],
        }
        notes.extend([
            "completed_result_does_not_require_automatic_recovery",
            "current_converged_vectors_preferred_for_controlled_repeat",
            "smaller_basis_seed_requires_independent_scf_review",
        ])
        if not input_path:
            notes.append("input_file_required_for_scf_stability_hardening")
        elif multi_stage_input:
            notes.append(
                "multi_stage_or_fragment_input_requires_manual_hardening_edit"
            )
        else:
            can_auto_prepare = True
            stabilize_base_name = base_name or f"{Path(input_path).stem}_stable"
            hardening = draft_nwchem_scf_stabilization_input(
                input_path=input_path,
                reference_output_path=output_path,
                output_dir=output_dir,
                base_name=stabilize_base_name,
                write_file=write_files,
            )
            hardening["input_text"] = (
                "\n".join(
                    line
                    for line in hardening["input_text"].splitlines()
                    if not line.lstrip().startswith("#")
                ).rstrip()
                + "\n"
            )
            prepared_artifacts["scf_stability_hardening"] = hardening
            artifact_order.append("scf_stability_hardening")
    elif failure_class == "no_clear_failure_detected":
        selected_workflow = "verification_only"
        notes.append("no_automatic_repair_needed")
    else:
        notes.append("no_matching_automatic_workflow")

    if can_auto_prepare:
        notes.append("prepared_artifacts_ready_for_local_review")

    return {
        "output_file": output_path,
        "input_file": input_path,
        "selected_workflow": selected_workflow,
        "can_auto_prepare": can_auto_prepare,
        "artifact_order": artifact_order,
        "prepared_artifacts": prepared_artifacts,
        "prepared_artifact_summaries": {
            name: _summarize_prepared_artifact(name, payload) for name, payload in prepared_artifacts.items()
        },
        "trigger_evidence": trigger_evidence,
        "notes": notes,
        "summary_text": summary["summary_text"],
        "summary_bullets": summary["summary_bullets"],
        "diagnosis": {
            "stage": diagnosis["stage"],
            "task_outcome": diagnosis["task_outcome"],
            "failure_class": diagnosis["failure_class"],
            "likely_cause": diagnosis["likely_cause"],
            "recommended_next_action": diagnosis["recommended_next_action"],
            "confidence": diagnosis["confidence"],
        },
    }



def plan_nwchem_workflow(
    goal: str,
    elements: list[str],
    charge: int,
    multiplicity: int,
    basis: str | None = None,
    method: str = "ccsd",
    xc_functional: str = "b3lyp",
    has_geometry_file: bool = False,
    has_dft_output: bool = False,
    has_scf_output: bool = False,
) -> dict[str, Any]:
    """Return a concrete step-by-step tool call plan for a NWChem workflow.

    Parameters
    ----------
    goal:
        One of: ``"opt_freq"``, ``"opt_freq_ccsd"``, ``"single_point_dft"``,
        ``"single_point_ccsd"``, ``"opt_freq_mp2"``.
    elements:
        Element symbols present in the molecule, e.g. ``["Fe", "Cl"]``.
    charge, multiplicity:
        Electronic state parameters.
    basis:
        Basis set name, e.g. ``"6-31gs"``.
    method:
        Correlated method for TCE steps: ``"ccsd"`` or ``"mp2"`` or ``"ccsd(t)"``.
    has_geometry_file, has_dft_output, has_scf_output:
        Set True if those artifacts already exist to skip earlier steps.
    """
    goal_norm = goal.lower().replace("-", "_").replace(" ", "_")
    open_shell = multiplicity > 1
    scf_ref = "rohf" if open_shell else "rhf"
    nopen_note = f"nopen={multiplicity - 1}" if open_shell else ""
    tce_method = method.lower()
    basis_placeholder = basis or "<basis from suggest_basis_set>"

    steps: list[dict[str, Any]] = []
    step = 0

    def _s(desc: str, tool: str, params: dict[str, Any], notes: list[str] | None = None) -> None:
        nonlocal step
        step += 1
        entry: dict[str, Any] = {"step": step, "description": desc, "tool": tool, "key_parameters": params}
        if notes:
            entry["notes"] = notes
        steps.append(entry)

    needs_tce = goal_norm in {"opt_freq_ccsd", "single_point_ccsd", "opt_freq_mp2", "single_point_mp2"}

    # Step 0: If no basis given, suggest one first
    if not basis:
        purpose = "correlation" if needs_tce else "geometry"
        _s(
            "Choose basis set",
            "suggest_basis_set",
            {"elements": elements, "purpose": purpose},
            ["Use the returned 'basis' field in all subsequent steps"],
        )

    # Step 1: Geometry
    if not has_geometry_file:
        _s(
            "Create initial geometry XYZ file",
            "draft_initial_geometry",
            {"atoms": elements, "output_path": "<job_dir>/<name>.xyz"},
            ["Use covalent-radii guess — do not skip geometry optimization"],
        )

    # Step 2: DFT opt+freq (always, unless goal is pure single-point)
    if goal_norm not in {"single_point_dft", "single_point_ccsd", "single_point_mp2"}:
        if not has_dft_output:
            _s(
                "Build DFT optimization + frequency input",
                "create_nwchem_dft_workflow_input",
                {
                    "geometry_file": "<xyz from step 1>",
                    "basis_assignments": {e: basis_placeholder for e in elements},
                    "xc_functional": xc_functional,
                    "task_operations": ["optimize", "freq"],
                    "charge": charge,
                    "multiplicity": multiplicity,
                    "write_file": True,
                },
            )
            _s("Lint the DFT input", "lint_nwchem_input", {"input_file": "<nw from previous step>"})
            _s(
                "Launch and watch the DFT job (auto_watch=true blocks until done)",
                "launch_nwchem_run",
                {"input_file": "<nw file>", "profile": "<your profile>", "auto_watch": True},
                ["launch_nwchem_run automatically polls until terminal — no separate watch call needed"],
            )
            _s("Extract converged geometry", "extract_nwchem_geometry",
               {"output_file": "<dft.out>", "frame": "best"})
    else:
        if not has_geometry_file:
            pass  # geometry already added above

    # TCE steps
    if needs_tce:
        _s(
            "Build SCF reference input for TCE",
            "create_nwchem_input",
            {
                "geometry_file": "<converged xyz>",
                "basis_assignments": {e: basis_placeholder for e in elements},
                "module": "scf",
                "scf_type": scf_ref,
                "nopen": multiplicity - 1,
                "charge": charge,
                "multiplicity": multiplicity,
                "vectors_output": "<name>_scf.movecs",
                "write_file": True,
            },
            [f"Must use {scf_ref} reference for open-shell TCE{(' (' + nopen_note + ')') if nopen_note else ''}"],
        )
        _s("Lint the SCF input", "lint_nwchem_input", {"input_file": "<scf.nw>"})
        _s("Launch and watch the SCF job (auto_watch=true blocks until done)", "launch_nwchem_run",
           {"input_file": "<scf.nw>", "profile": "<your profile>", "auto_watch": True},
           ["launch_nwchem_run automatically polls until terminal — no separate watch call needed"])
        _s(
            "Inspect orbital ordering before freezing",
            "parse_nwchem_movecs",
            {"movecs_file": "<scf.movecs>"},
            ["Check that core orbitals (metal 1s/2s/2p) are lowest-index; swap if not"],
        )
        _s(
            "Suggest freeze count",
            "suggest_nwchem_tce_freeze",
            {"elements": elements, "charge": charge, "multiplicity": multiplicity},
        )
        _s(
            "Build TCE input",
            "draft_nwchem_tce_input",
            {
                "scf_output_file": "<scf.out>",
                "input_file": "<scf.nw>",
                "method": tce_method,
                "movecs_file": "<scf.movecs>",
                "write_file": True,
            },
            ["symmetry c1 is added automatically", "geometry block is included automatically"],
        )
        _s("Lint the TCE input", "lint_nwchem_input", {"input_file": "<tce.nw>"})
        _s("Launch and watch the TCE job (auto_watch=true blocks until done)", "launch_nwchem_run",
           {"input_file": "<tce.nw>", "profile": "<your profile>", "auto_watch": True},
           ["launch_nwchem_run automatically polls until terminal — no separate watch call needed"])
        _s("Parse TCE results", "parse_nwchem_tce_output", {"output_file": "<tce.out>"})

    warnings: list[str] = []
    if open_shell:
        warnings.append(
            f"Open-shell system (mult={multiplicity}): SCF may converge to wrong electronic state. "
            "After SCF, use suggest_nwchem_vectors_swaps to verify the correct SOMOs are occupied."
        )
    if any(e in {"Fe", "Co", "Ni", "Cu", "Mn", "Cr", "Mo", "W", "Ru", "Rh"} for e in elements):
        warnings.append(
            "Transition metal present: verify spin state carefully. "
            "Use suggest_nwchem_recovery(mode='state') if occupations look wrong after SCF."
        )

    return {
        "goal": goal_norm,
        "total_steps": len(steps),
        "steps": steps,
        "warnings": warnings,
        "note": "Replace <placeholders> with actual file paths as you complete each step.",
    }

__all__ = [
    "prepare_nwchem_next_step",
    "plan_nwchem_workflow",
]

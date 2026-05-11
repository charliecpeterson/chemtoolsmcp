from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from chemtools.core.common import detect_program, make_metadata, read_text, ELEMENT_TO_Z
from chemtools.programs.nwchem.input.basis_library import (
    extract_basis_blocks,
    extract_nwchem_geometry_elements,
    list_basis_sets,
    render_mixed_nwchem_basis_block,
    render_mixed_nwchem_ecp_block,
    render_nwchem_ecp_block,
    render_nwchem_basis_block,
    render_nwchem_basis_block_from_geometry,
    resolve_ecp_set,
    resolve_mixed_basis_assignments,
    resolve_mixed_ecp_assignments,
    resolve_basis_set,
)
from chemtools.programs.nwchem.strategy.diagnose import (
    analyze_frontier_orbitals as analyze_nwchem_frontier_orbitals,
    diagnose_nwchem_output,
    parse_scf,
    suggest_vectors_swaps as suggest_nwchem_vectors_swaps,
    summarize_nwchem_output,
)
from chemtools.programs.nwchem.parse.input import (
    extract_nwchem_geometry_block,
    extract_nwchem_module_block,
    inspect_all_nwchem_basis_blocks,
    inspect_nwchem_basis_block,
    inspect_nwchem_ecp_block,
    inspect_nwchem_input,
    inspect_nwchem_module_vectors,
    load_geometry_source,
    parse_start_blocks,
    render_nwchem_module_block,
    render_nwchem_geometry_block,
    replace_nwchem_geometry_block,
    replace_nwchem_module_block,
)
# Raw-signature versions of functions that have same-name MCP wrappers
# in this file or in chemtools.programs.nwchem.output. The wrappers below
# take a single `path` argument; the raw versions take `(path, contents, ...)`.
from chemtools.programs.nwchem.parse.mos import parse_mos as _parse_mos_raw
from chemtools.programs.nwchem.parse.freq import (
    parse_trajectory as _parse_trajectory_raw,
    analyze_imaginary_modes as _analyze_imaginary_modes_raw,
    displace_geometry_along_mode as _displace_geometry_along_mode_raw,
)
from chemtools.programs.nwchem.input._utils import (
    _TRANSITION_METALS,
    _COVALENT_RADII,
    _coerce_api_int,
    _coerce_api_float,
    _strategy_entry,
    _summarize_prepared_artifact,
    KEYWORD_LINE_RE,
    CONVERGENCE_DAMP_RE,
    CONVERGENCE_NCYDP_RE,
    ITERATIONS_RE,
    SMEAR_RE,
    PRINT_RE,
    CONVERGENCE_ENERGY_RE,
    VECTORS_RE,
    VECTORS_INPUT_TOKEN_RE,
    VECTORS_OUTPUT_TOKEN_RE,
    _select_primary_task_module,
    _select_scf_stabilization_strategy,
    _select_optimization_follow_up_strategy,
    _build_optimization_follow_up_plan,
    _rewrite_module_body_for_vectors_swap,
    _rewrite_module_body_for_property_check,
    _rewrite_module_body_for_scf_stabilization,
    _extract_vectors_io_from_lines,
    _rewrite_module_body_for_vectors_output,
    _indent_vectors_block_lines,
    _replace_module_block_in_text,
    _ensure_module_vectors_output_in_text,
    _default_optimization_follow_up_base_name,
    _default_optimization_follow_up_title,
    _build_simple_input_file_plan,
    _apply_default_dft_settings,
    _ensure_driver_block,
    _parse_formula_elements,
    _normalize_nwchem_task_operation,
    _replace_or_insert_keyword_line,
    _remove_keyword_blocks,
    _render_named_block,
    _replace_or_insert_named_block,
    _append_named_blocks_before_tasks,
    _render_limitxyz_lines,
    _render_dplot_density_block,
    _render_dplot_orbital_block,
    _build_vectors_swap_file_plan,
    _build_mcscf_reorder_plan,
    _render_mcscf_block,
    _build_cube_file_plan,
    _write_text_file,
    _build_imaginary_follow_up_plan,
    _auto_task_strategy,
    _replace_tasks_in_text,
    _build_imaginary_output_file_plan,
    _write_imaginary_input_files,
)
from chemtools.programs.nwchem.input.basis import render_nwchem_basis_setup
from chemtools.programs.nwchem.input.opt_followup import _select_best_optimization_frame  # used by extract_nwchem_geometry below
from chemtools.programs.nwchem.output import (
    parse_tasks,
    parse_mos,
    parse_trajectory,
    parse_mcscf_output,
    parse_population_analysis,
    summarize_output,
    diagnose_output,
    suggest_vectors_swaps,
    analyze_frontier_orbitals,
    parse_freq,
)
from .api_strategy import (
    check_spin_charge_state,
    suggest_nwchem_mcscf_active_space,
    review_nwchem_mcscf_case,
)


def _normalize_stem_for_match(stem: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def _stem_tokens(stem: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", stem.lower()) if token]


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
    elem_str = str(elements)
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


# Imaginary-mode handling drafters moved to programs/nwchem/input/imaginary_modes.py.
from chemtools.programs.nwchem.input.imaginary_modes import (  # noqa: F401, E402
    analyze_imaginary_modes,
    displace_geometry_along_mode,
    draft_nwchem_imaginary_mode_inputs,
)


# Optimization follow-up drafter moved to programs/nwchem/input/opt_followup.py.
from chemtools.programs.nwchem.input.opt_followup import (  # noqa: F401, E402
    draft_nwchem_optimization_followup_input,
)


# DFT workflow drafters moved to programs/nwchem/input/dft.py.
from chemtools.programs.nwchem.input.dft import (  # noqa: F401, E402
    create_nwchem_dft_workflow_input,
    create_nwchem_dft_input_from_request,
)


# Geometry helpers moved to programs/nwchem/input/geometry.py.
from chemtools.programs.nwchem.input.geometry import (  # noqa: F401, E402
    extract_nwchem_geometry,
    draft_initial_geometry,
)


# Re-exports for previously-carved drafter families (lost during the
# geometry carve-out's range delete). Restored for back-compat.
from chemtools.programs.nwchem.input.scf_recovery import (  # noqa: F401, E402
    draft_nwchem_vectors_swap_input,
    draft_nwchem_property_check_input,
    draft_nwchem_scf_stabilization_input,
)
from chemtools.programs.nwchem.input.mcscf import (  # noqa: F401, E402
    draft_nwchem_mcscf_input,
    draft_nwchem_mcscf_retry_input,
)
from chemtools.programs.nwchem.input.cube import (  # noqa: F401, E402
    draft_nwchem_cube_input,
    draft_nwchem_frontier_cube_input,
)


# Lint + restart helpers moved to programs/nwchem/input/lint_restart.py.
from chemtools.programs.nwchem.input.lint_restart import (  # noqa: F401, E402
    inspect_input,
    lint_nwchem_input,
    find_restart_assets,
)
from chemtools.programs.nwchem.input.tce import (  # noqa: F401, E402
    draft_nwchem_tce_input,
    validate_nwchem_tce_setup,
    draft_nwchem_atom_input,
    draft_nwchem_tce_restart_input,
)


# General input drafters moved to programs/nwchem/input/general.py.
from chemtools.programs.nwchem.input.general import (  # noqa: F401, E402
    create_nwchem_input,
    review_nwchem_input_request,
    create_nwchem_input_variant,
)

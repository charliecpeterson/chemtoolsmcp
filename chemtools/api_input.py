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


def create_nwchem_input(
    geometry_path: str,
    library_path: str,
    basis_assignments: dict[str, str],
    module: str,
    task_operation: str | None = None,
    ecp_assignments: dict[str, str] | None = None,
    default_basis: str | None = None,
    default_ecp: str | None = None,
    basis_block_name: str = "ao basis",
    basis_mode: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    module_settings: list[str] | None = None,
    extra_blocks: list[str] | None = None,
    memory: str | None = None,
    title: str | None = None,
    start_name: str | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    geometry_block_index: int = 0,
    output_dir: str | None = None,
    write_file: bool = False,
    inline_blocks: bool = True,
) -> dict[str, Any]:
    resolved_module = module.strip().lower()
    if not resolved_module:
        raise ValueError("module is required")

    geometry = load_geometry_source(geometry_path, block_index=geometry_block_index)
    geometry_block = render_nwchem_geometry_block(
        geometry["header_line"],
        geometry["atoms"],
        directives=geometry["directives"],
    )
    basis_setup = render_nwchem_basis_setup(
        geometry_path=geometry_path,
        library_path=library_path,
        basis_assignments=basis_assignments,
        ecp_assignments=ecp_assignments,
        default_basis=default_basis,
        default_ecp=default_ecp,
        basis_block_name=basis_block_name,
        basis_mode=basis_mode,
        geometry_block_index=geometry_block_index,
        inline_blocks=inline_blocks,
    )

    resolved_start_name = start_name or Path(geometry_path).stem or "nwchem_job"
    resolved_title = title or f"{resolved_start_name}: {resolved_module} {task_operation or 'run'}"
    resolved_vectors_output = vectors_output or (
        f"{resolved_start_name}.movecs" if resolved_module in {"scf", "dft"} else None
    )

    rendered_module_settings = [
        line.rstrip() if line[:1].isspace() else f"  {line.rstrip()}"
        for line in (module_settings or [])
        if str(line).strip()
    ]
    stripped_lower = [line.strip().lower() for line in rendered_module_settings]

    if resolved_module == "dft":
        rendered_module_settings = _apply_default_dft_settings(
            rendered_module_settings,
            xc_functional=None,
            multiplicity=multiplicity,
            vectors_input=vectors_input,
            vectors_output=resolved_vectors_output,
        )
        stripped_lower = [line.strip().lower() for line in rendered_module_settings]
    elif resolved_module in {"scf"} and resolved_vectors_output:
        if not any(line.startswith("vectors ") for line in stripped_lower):
            if vectors_input:
                rendered_module_settings.append(f"  vectors input {vectors_input} output {resolved_vectors_output}")
            else:
                rendered_module_settings.append(f"  vectors output {resolved_vectors_output}")

    module_block = render_nwchem_module_block(resolved_module, rendered_module_settings)

    sections: list[str] = [f"start {resolved_start_name}", f'title "{resolved_title}"', "echo"]
    if memory:
        sections.append(f"memory {memory}")
    sections.append(geometry_block)
    sections.append(basis_setup["basis_block"]["text"])
    if basis_setup["ecp_block"]:
        sections.append(basis_setup["ecp_block"]["text"])
    if charge is not None:
        sections.append(f"charge {charge}")
    rendered_extra_blocks = [block.strip("\n") for block in (extra_blocks or []) if str(block).strip()]
    if resolved_module == "dft" and task_operation == "optimize":
        _ensure_driver_block(rendered_extra_blocks)
    sections.extend(rendered_extra_blocks)
    sections.append(module_block)

    task_line = f"task {resolved_module}"
    if task_operation:
        task_line = f"{task_line} {task_operation}"
    sections.append(task_line)

    input_text = "\n\n".join(sections).rstrip() + "\n"
    file_plan = _build_simple_input_file_plan(
        input_path=geometry_path,
        output_dir=output_dir,
        base_name=resolved_start_name,
    )
    written_file: str | None = None
    if write_file:
        written_file = _write_text_file(input_text, file_plan["input_file"])

    return {
        "geometry_source": geometry["file"],
        "geometry_source_kind": geometry.get("source_kind"),
        "module": resolved_module,
        "task_operation": task_operation,
        "charge": charge,
        "multiplicity": multiplicity,
        "basis_setup": basis_setup,
        "module_settings": [line.strip() for line in rendered_module_settings],
        "vectors_input": vectors_input,
        "vectors_output": resolved_vectors_output,
        "input_text": input_text,
        "file_plan": file_plan,
        "written_file": written_file,
        "inline_blocks": inline_blocks,
    }


def review_nwchem_input_request(
    *,
    formula: str | None = None,
    geometry_path: str | None = None,
    library_path: str | None = None,
    basis_assignments: dict[str, str] | None = None,
    ecp_assignments: dict[str, str] | None = None,
    default_basis: str | None = None,
    default_ecp: str | None = None,
    module: str = "dft",
    task_operations: list[str] | None = None,
    functional: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
) -> dict[str, Any]:
    normalized_module = module.strip().lower() or "dft"
    normalized_tasks = [_normalize_nwchem_task_operation(task) for task in (task_operations or ["energy"])]
    formula_elements = _parse_formula_elements(formula) if formula else []
    geometry_summary = inspect_nwchem_input(geometry_path) if geometry_path and Path(geometry_path).suffix.lower() != ".xyz" else None
    geometry = load_geometry_source(geometry_path) if geometry_path else None
    geometry_elements = list(dict.fromkeys(atom["element"] for atom in geometry["atoms"])) if geometry else []
    elements = geometry_elements or formula_elements
    transition_metals = [element for element in elements if element in _TRANSITION_METALS]

    inferred_charge = charge
    assumptions: list[str] = []
    if inferred_charge is None and formula and elements:
        inferred_charge = 0
        assumptions.append("assumed_neutral_formula_charge")

    missing_requirements: list[dict[str, str]] = []
    warnings: list[str] = []

    if not geometry_path:
        missing_requirements.append(
            {
                "field": "geometry_source",
                "reason": "An NWChem input creator needs explicit coordinates from an .xyz or existing .nw file. Do not invent geometry silently.",
            }
        )
    if not elements:
        missing_requirements.append(
            {
                "field": "composition",
                "reason": "No geometry or parsable formula was provided, so element assignments cannot be validated.",
            }
        )
    if not basis_assignments and not default_basis:
        missing_requirements.append(
            {
                "field": "basis_assignment_policy",
                "reason": "At least one explicit basis assignment or a default basis is required.",
            }
        )
    if normalized_module in {"dft", "scf"} and transition_metals and multiplicity is None:
        missing_requirements.append(
            {
                "field": "multiplicity",
                "reason": "Transition-metal/open-shell systems should not have multiplicity guessed automatically.",
            }
        )
    if formula_elements and geometry_elements and formula_elements != geometry_elements:
        warnings.append("formula_elements_do_not_match_geometry_elements")

    basis_preview = None
    if geometry_path and library_path and (basis_assignments or default_basis):
        try:
            basis_preview = resolve_basis_setup(
                geometry_path=geometry_path,
                library_path=library_path,
                basis_assignments=basis_assignments or {},
                ecp_assignments=ecp_assignments,
                default_basis=default_basis,
                default_ecp=default_ecp,
            )
            if not basis_preview["basis"]["all_elements_covered"]:
                missing_requirements.append(
                    {
                        "field": "basis_assignment_policy",
                        "reason": "Current basis assignments do not cover all elements in the geometry.",
                    }
                )
        except Exception as exc:
            warnings.append(f"basis_preview_failed: {exc}")

    ready_to_create = not missing_requirements and geometry_path is not None
    recommended_tool = None
    if ready_to_create:
        if normalized_module == "dft":
            recommended_tool = "create_nwchem_dft_workflow_input"
        else:
            recommended_tool = "create_nwchem_input"

    next_questions = [item["field"] for item in missing_requirements]
    return {
        "formula": formula,
        "geometry_file": geometry_path,
        "elements": elements,
        "transition_metals": transition_metals,
        "module": normalized_module,
        "task_operations": normalized_tasks,
        "functional": functional,
        "charge": inferred_charge,
        "multiplicity": multiplicity,
        "assumptions": assumptions,
        "ready_to_create": ready_to_create,
        "recommended_tool": recommended_tool,
        "missing_requirements": missing_requirements,
        "next_questions": next_questions,
        "warnings": warnings,
        "basis_preview": basis_preview,
        "input_summary": geometry_summary,
    }


def inspect_input(input_path: str) -> dict[str, Any]:
    return inspect_nwchem_input(input_path)



def _lint_fragment_guess(
    path: str,
    add_issue: Any,
) -> None:
    """Check that fragment Nα/Nβ sums match the molecular Nα/Nβ for every
    'vectors input fragment' block found in the file."""
    blocks = parse_start_blocks(path)

    if not any(b["fragment_inputs"] for b in blocks):
        return  # no fragment guess in this file

    # Build lookup: vectors_output filename → block
    output_map: dict[str, dict[str, Any]] = {
        b["vectors_output"]: b for b in blocks if b["vectors_output"]
    }

    for mol in blocks:
        if not mol["fragment_inputs"]:
            continue

        if mol["multiplicity"] is None:
            add_issue(
                "warning",
                "fragment_mult_unknown",
                "A 'vectors input fragment' block is present but the molecular DFT "
                "multiplicity (mult N) is not set; cannot validate Nα/Nβ balance.",
            )
            continue

        missing_sources = [f for f in mol["fragment_inputs"] if f not in output_map]
        if missing_sources:
            add_issue(
                "warning",
                "fragment_source_not_found",
                "Some fragment movecs files are not produced by a 'vectors output' "
                "in any start block in this file; Nα/Nβ balance cannot be checked.",
                {"missing": missing_sources},
            )
            continue

        mol_electrons = (
            sum(ELEMENT_TO_Z.get(e.capitalize(), 0) for e in mol["elements"])
            - mol["charge"]
        )
        mol_mult = mol["multiplicity"]
        mol_nalpha = (mol_electrons + (mol_mult - 1)) // 2
        mol_nbeta = mol_electrons - mol_nalpha

        frag_nalpha_sum = 0
        frag_nbeta_sum = 0
        incomplete = False
        for frag_file in mol["fragment_inputs"]:
            fb = output_map[frag_file]
            if fb["multiplicity"] is None:
                add_issue(
                    "warning",
                    "fragment_mult_unknown",
                    f"Fragment block producing '{frag_file}' has no multiplicity set; "
                    "cannot validate Nα/Nβ balance.",
                )
                incomplete = True
                break
            frag_electrons = (
                sum(ELEMENT_TO_Z.get(e.capitalize(), 0) for e in fb["elements"])
                - fb["charge"]
            )
            frag_mult = fb["multiplicity"]
            frag_nalpha = (frag_electrons + (frag_mult - 1)) // 2
            frag_nbeta = frag_electrons - frag_nalpha
            frag_nalpha_sum += frag_nalpha
            frag_nbeta_sum += frag_nbeta

        if incomplete:
            continue

        if frag_nalpha_sum == mol_nalpha and frag_nbeta_sum == mol_nbeta:
            add_issue(
                "info",
                "fragment_electron_balance_ok",
                f"Fragment Nα/Nβ sums ({frag_nalpha_sum}/{frag_nbeta_sum}) match "
                f"the molecular Nα/Nβ ({mol_nalpha}/{mol_nbeta}). "
                "Fragment guess electron counts are consistent.",
            )
        else:
            add_issue(
                "error",
                "fragment_electron_mismatch",
                f"Fragment Nα/Nβ sums ({frag_nalpha_sum}/{frag_nbeta_sum}) do not "
                f"match the molecular Nα/Nβ ({mol_nalpha}/{mol_nbeta}). "
                "NWChem will abort with 'movecs_fragment: open shell mismatch'. "
                "Adjust fragment multiplicities so their Nα and Nβ sum exactly to "
                "the molecular values.",
                {
                    "molecular": {
                        "nalpha": mol_nalpha,
                        "nbeta": mol_nbeta,
                        "mult": mol_mult,
                        "electrons": mol_electrons,
                    },
                    "fragments": {
                        "nalpha_sum": frag_nalpha_sum,
                        "nbeta_sum": frag_nbeta_sum,
                        "files": mol["fragment_inputs"],
                    },
                },
            )


def lint_nwchem_input(
    input_path: str,
    library_path: str | None = None,
) -> dict[str, Any]:
    input_summary = inspect_nwchem_input(input_path)
    issues: list[dict[str, Any]] = []

    def add_issue(level: str, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        payload = {"level": level, "code": code, "message": message}
        if details:
            payload["details"] = details
        issues.append(payload)

    if not input_summary["tasks"]:
        add_issue("error", "missing_tasks", "No task lines were found in the input.")
    if not input_summary["start_present"]:
        add_issue("warning", "missing_start", "No explicit start line was found in the input.")
    if input_summary["charge"] is None:
        add_issue("info", "charge_not_set", "Charge is not explicitly set; NWChem will assume the default.")
    if input_summary["multiplicity"] is None:
        add_issue("info", "multiplicity_not_set", "Multiplicity is not explicitly set in the input.")

    all_basis_blocks = inspect_all_nwchem_basis_blocks(input_path)
    basis_block = all_basis_blocks[0] if all_basis_blocks else None

    if not all_basis_blocks:
        add_issue("warning", "missing_basis_block", "No explicit basis block was found in the input.")
    else:
        for blk in all_basis_blocks:
            block_idx = blk["block_index"]
            blk_details_base: dict[str, Any] = {"block_index": block_idx}
            if blk["has_manual_content"] and not blk["has_library_lines"]:
                add_issue(
                    "info",
                    "manual_basis_content",
                    "Basis block contains manual basis data; library validation was skipped.",
                    {**blk_details_base, "elements": blk["explicit_elements"]},
                )
            elif blk["has_library_lines"]:
                if library_path:
                    resolved_basis = resolve_mixed_basis_assignments(
                        assignments=blk["library_assignments"],
                        elements=input_summary["elements"],
                        library_path=library_path,
                        default_basis=blk["default_library"],
                    )
                    if resolved_basis["missing_assignments"]:
                        add_issue(
                            "error",
                            "basis_assignment_missing",
                            "Some geometry elements do not have basis assignments.",
                            {**blk_details_base, "elements": resolved_basis["missing_assignments"]},
                        )
                    if resolved_basis["missing_coverage"]:
                        add_issue(
                            "error",
                            "basis_library_missing_coverage",
                            "The chosen basis library entries do not cover all assigned elements.",
                            {**blk_details_base, "elements": resolved_basis["missing_coverage"]},
                        )
                    if resolved_basis["all_elements_covered"]:
                        add_issue(
                            "info",
                            "basis_validated",
                            "Basis assignments were validated against the local basis library.",
                            blk_details_base,
                        )
                else:
                    add_issue(
                        "info",
                        "basis_library_not_checked",
                        "Basis block uses library entries, but no library path was provided for validation.",
                        blk_details_base,
                    )

    try:
        ecp_block = inspect_nwchem_ecp_block(input_path)
    except ValueError:
        ecp_block = None
    else:
        if ecp_block["has_manual_content"] and not ecp_block["has_library_lines"]:
            add_issue(
                "info",
                "manual_ecp_content",
                "ECP block contains manual ECP data; library validation was skipped.",
                {"elements": ecp_block["explicit_elements"]},
            )
        elif ecp_block["has_library_lines"]:
            if library_path:
                resolved_ecp = resolve_mixed_ecp_assignments(
                    assignments=ecp_block["library_assignments"],
                    elements=input_summary["elements"],
                    library_path=library_path,
                    default_ecp=ecp_block["default_library"],
                )
                if resolved_ecp["missing_coverage"]:
                    add_issue(
                        "error",
                        "ecp_library_missing_coverage",
                        "The chosen ECP library entries do not cover all assigned elements.",
                        {"elements": resolved_ecp["missing_coverage"]},
                    )
                if resolved_ecp["elements_with_ecp"]:
                    add_issue(
                        "info",
                        "ecp_validated",
                        "ECP assignments were validated against the local basis library.",
                        {"elements": resolved_ecp["elements_with_ecp"]},
                    )
            else:
                add_issue(
                    "info",
                    "ecp_library_not_checked",
                    "ECP block uses library entries, but no library path was provided for validation.",
                )

    if basis_block and basis_block["has_library_lines"] and not ecp_block:
        assigned_families = set(basis_block["library_assignments"].values())
        if basis_block["default_library"]:
            assigned_families.add(basis_block["default_library"])
        if any(("ecp" in family.lower()) or family.lower().endswith("-pp") for family in assigned_families):
            add_issue(
                "warning",
                "possible_missing_ecp_block",
                "Basis assignments look pseudopotential-based, but no explicit ECP block was found.",
            )

    task_modules = []
    seen_modules: set[str] = set()
    for task in input_summary["tasks"]:
        module_name = (task.get("module") or "").lower()
        operation_name = (task.get("operation") or "").lower()

        if module_name in {"optimize", "frequency", "freq", "energy", "property", "gradient", "hessian", "raman"} and not operation_name:
            suggested_module = "dft" if any(
                block_name in {module_name, "dft"}
                for block_name in [module_name, "dft"]
            ) else "dft"
            suggested_operation = "freq" if module_name in {"frequency", "freq"} else module_name
            add_issue(
                "error",
                "invalid_task_syntax",
                f"Task line 'task {module_name}' is not valid NWChem syntax for this workflow.",
                {
                    "task_module": module_name,
                    "suggested_task_line": f"task {suggested_module} {suggested_operation}",
                },
            )
            continue

        if module_name and module_name not in seen_modules:
            seen_modules.add(module_name)
            task_modules.append(module_name)

    for module_name in task_modules:
        try:
            module_vectors = inspect_nwchem_module_vectors(input_path, module=module_name)
        except ValueError:
            add_issue(
                "error",
                "missing_module_block",
                f"Task module '{module_name}' is referenced, but no matching module block was found.",
                {"module": module_name},
            )
            continue

        if module_name in {"scf", "dft"} and not module_vectors["has_vectors_output"]:
            add_issue(
                "warning",
                "missing_vectors_output",
                f"Module '{module_name}' does not explicitly write a movecs file.",
                {"module": module_name},
            )

    _lint_fragment_guess(input_path, add_issue)

    # --- Relativistic + ECP conflict check, and relativistic + SP-shell incompatibility ---
    try:
        import re as _re2
        from chemtools.core.common import read_text as _rt2
        _rc = _rt2(input_path)
        _has_rel = bool(_re2.search(r"^\s*relativistic\b", _rc, _re2.IGNORECASE | _re2.MULTILINE))
        _has_ecp = bool(_re2.search(r"^\s*ecp\b", _rc, _re2.IGNORECASE | _re2.MULTILINE))
        if _has_rel and _has_ecp:
            add_issue(
                "error",
                "relativistic_ecp_conflict",
                "Both a 'relativistic' block and an 'ecp' block are present. "
                "X2C and DKH are all-electron methods — they are incompatible with ECPs. "
                "Choose one: (a) all-electron basis + relativistic block, OR "
                "(b) ECP basis (no relativistic block needed — ECP implicitly encodes scalar relativistic effects).",
            )
        if _has_rel:
            # SP-contracted shells (Pople style) are incompatible with X2C/DKH.
            # NWChem builds an uncontracted auxiliary basis for the relativistic
            # one-electron operator; SP shells cause a dimension mismatch and
            # crash with "dimensions not the same" / MPI_Abort.
            _sp_elements = sorted({
                m.group(1)
                for m in _re2.finditer(
                    r"^\s*([A-Za-z][a-z]?)\s+SP\s*$", _rc, _re2.MULTILINE
                )
            })
            if _sp_elements:
                add_issue(
                    "error",
                    "relativistic_sp_shell_incompatibility",
                    f"SP-contracted basis shells detected for element(s) {_sp_elements} "
                    "while a relativistic block (X2C or DKH) is present. "
                    "NWChem X2C/DKH builds an uncontracted auxiliary basis internally; "
                    "Pople-style SP shells (6-31G*, 6-311G**, etc.) cause a 'dimensions not the same' "
                    "crash during this step. "
                    "Fix: replace the Pople basis with a Dunning basis (cc-pVDZ, cc-pVTZ, etc.) "
                    "or a def2 basis (def2-SVP, def2-TZVP) — both use separate S and P contractions "
                    "and are fully compatible with X2C/DKH.",
                )
    except Exception:
        pass

    # --- TCE-specific checks ---
    tce_tasks = [t for t in input_summary["tasks"] if (t.get("module") or "").lower() == "tce"]
    if tce_tasks:
        from chemtools.programs.nwchem.parse.input import extract_nwchem_geometry_block
        from chemtools.core.common import read_text as _read_text
        import re as _re

        # NWChem TCE accepts any of these Abelian point groups
        _TCE_ABELIAN = {"c1", "ci", "cs", "c2", "c2v", "c2h", "d2", "d2h"}

        # Check that geometry specifies an Abelian symmetry group
        try:
            geo = extract_nwchem_geometry_block(input_path)
            directives = [d.strip().lower() for d in (geo.get("directives") or [])]
            sym_group = None
            for d in directives:
                parts = d.split()
                if parts and parts[0] == "symmetry" and len(parts) > 1:
                    sym_group = parts[1]
                    break
            if sym_group not in _TCE_ABELIAN:
                if sym_group is None:
                    msg = (
                        "TCE requires Abelian symmetry. Add 'symmetry c1' (or d2h, c2v, etc.) "
                        "as a line inside the geometry block. "
                        "NWChem will abort with 'non-Abelian symmetry not permitted' otherwise."
                    )
                else:
                    msg = (
                        f"TCE requires Abelian symmetry; '{sym_group}' is non-Abelian. "
                        f"Use one of: {', '.join(sorted(_TCE_ABELIAN))}."
                    )
                add_issue("error", "tce_missing_symmetry_c1", msg)
        except Exception:
            pass

        # Check for symmetry placed on the geometry header line (wrong syntax)
        _lint_contents = _read_text(input_path)
        for _line in _lint_contents.splitlines():
            if _re.match(r"^\s*geometry\b.*\bsymmetry\b", _line, _re.IGNORECASE):
                add_issue(
                    "error",
                    "symmetry_on_geometry_header",
                    "'symmetry' must appear as its own line inside the geometry block, not on the "
                    "'geometry ...' header line. Correct form:\n"
                    "  geometry units angstrom\n"
                    "    symmetry c1\n"
                    "    ...\n"
                    "  end",
                )
                break

    # --- autoz + symmetric TM complex warning ---
    _tm_elements_in_input = {e for e in (input_summary.get("elements") or []) if e in _TRANSITION_METALS}
    if _tm_elements_in_input:
        _has_optimize_task = any(
            (t.get("operation") or "").lower() == "optimize" for t in (input_summary.get("tasks") or [])
        )
        if _has_optimize_task:
            _rc_full = open(input_path, encoding="utf-8", errors="replace").read()
            _has_driver = bool(_re2.search(r"^\s*driver\b", _rc_full, _re2.IGNORECASE | _re2.MULTILINE))
            _has_xyz = bool(_re2.search(r"^\s*xyz\b", _rc_full, _re2.IGNORECASE | _re2.MULTILINE))
            if not _has_driver or not _has_xyz:
                _n_heavy_in_input = sum(
                    1 for e in (input_summary.get("elements") or []) if e not in {"H", "D"}
                )
                if _n_heavy_in_input >= 4:
                    add_issue(
                        "warning",
                        "autoz_symmetric_tm_complex",
                        "Optimization of a TM complex without explicit 'driver; xyz; end' may produce "
                        "degenerate Z-matrix coordinates for symmetric geometries (e.g. octahedral, "
                        "tetrahedral), causing the optimizer to walk uphill. "
                        "Add 'driver\\n  xyz\\n  maxiter 300\\nend' before the DFT/SCF block.",
                    )

    # --- Memory directive consistency check ---
    try:
        _rc2 = open(input_path, encoding="utf-8", errors="replace").read()
        _mem_m = _re2.search(
            r"^\s*memory\s+total\s+(\d+)\s*mb\b.*stack\s+(\d+)\s*mb\b.*heap\s+(\d+)\s*mb\b.*global\s+(\d+)\s*mb\b",
            _rc2, _re2.IGNORECASE | _re2.MULTILINE,
        )
        if _mem_m:
            _total, _stack, _heap, _glob = (int(_mem_m.group(i)) for i in range(1, 5))
            if _stack + _heap + _glob > _total:
                add_issue(
                    "error",
                    "memory_subcomponents_exceed_total",
                    f"Memory sub-components (stack {_stack} + heap {_heap} + global {_glob} = "
                    f"{_stack + _heap + _glob} MB) exceed the declared total of {_total} MB. "
                    "NWChem will abort on startup with 'Memory_Defaults: Inconsistent memory specification'. "
                    f"Set total to at least {_stack + _heap + _glob} MB.",
                )
    except OSError:
        pass

    severity_order = {"error": 3, "warning": 2, "info": 1}
    highest = max((severity_order[item["level"]] for item in issues), default=0)
    status = "ok"
    if highest >= 3:
        status = "error"
    elif highest == 2:
        status = "warning"

    return {
        "input_file": input_path,
        "library_path": library_path,
        "status": status,
        "issue_count": len(issues),
        "issues": issues,
        "counts": {
            "error": sum(1 for item in issues if item["level"] == "error"),
            "warning": sum(1 for item in issues if item["level"] == "warning"),
            "info": sum(1 for item in issues if item["level"] == "info"),
        },
        "input_summary": input_summary,
        "basis_block": basis_block,
        "ecp_block": ecp_block,
    }



def find_restart_assets(path: str) -> dict[str, Any]:
    target = Path(path).resolve()
    job_dir = target if target.is_dir() else target.parent
    focus_stem = None if target.is_dir() else target.stem

    relevant_suffixes = {
        ".nw": "inputs",
        ".out": "outputs",
        ".err": "errors",
        ".movecs": "movecs",
        ".db": "databases",
        ".xyz": "xyz",
        ".zmat": "zmat",
        ".cube": "cubes",
        ".nmode": "nmodes",
        ".normal": "normal_modes",
        ".hess": "hessians",
    }
    collections: dict[str, list[str]] = {label: [] for label in relevant_suffixes.values()}

    for child in sorted(job_dir.iterdir()):
        if not child.is_file():
            continue
        suffix = child.suffix.lower()
        label = relevant_suffixes.get(suffix)
        if label:
            collections[label].append(str(child.resolve()))

    related_files = sorted(
        str(child.resolve())
        for child in job_dir.iterdir()
        if child.is_file() and (focus_stem is None or child.name.startswith(focus_stem))
    )

    def choose_exact(suffix: str) -> str | None:
        if focus_stem is None:
            return None
        candidate = job_dir / f"{focus_stem}{suffix}"
        return str(candidate.resolve()) if candidate.exists() else None

    def newest(label: str) -> str | None:
        files = [Path(item) for item in collections[label]]
        if not files:
            return None
        return str(max(files, key=lambda candidate: candidate.stat().st_mtime).resolve())

    def best_related(label: str) -> str | None:
        files = [Path(item) for item in collections[label]]
        if not files:
            return None
        if focus_stem is None:
            return str(max(files, key=lambda candidate: candidate.stat().st_mtime).resolve())

        exact = choose_exact(files[0].suffix.lower())
        if exact:
            return exact

        normalized_focus = _normalize_stem_for_match(focus_stem)
        focus_tokens = set(_stem_tokens(focus_stem))
        scored: list[tuple[tuple[int, int, int, float], Path]] = []
        for candidate in files:
            stem = candidate.stem
            normalized = _normalize_stem_for_match(stem)
            candidate_tokens = set(_stem_tokens(stem))
            score = (
                1 if normalized == normalized_focus else 0,
                len(focus_tokens & candidate_tokens),
                1 if focus_tokens and (focus_tokens <= candidate_tokens or candidate_tokens <= focus_tokens) else 0,
                candidate.stat().st_mtime,
            )
            scored.append((score, candidate))

        best_score, best_path = max(scored, key=lambda item: item[0])
        if best_score[1] > 0 or best_score[0] > 0:
            return str(best_path.resolve())
        return str(max(files, key=lambda candidate: candidate.stat().st_mtime).resolve())

    preferred = {
        "input_file": choose_exact(".nw") or best_related("inputs"),
        "output_file": choose_exact(".out") or best_related("outputs"),
        "error_file": choose_exact(".err") or best_related("errors"),
        "vectors_file": choose_exact(".movecs") or best_related("movecs"),
        "database_file": choose_exact(".db") or best_related("databases"),
        "xyz_file": choose_exact(".xyz") or best_related("xyz"),
        "zmat_file": choose_exact(".zmat") or best_related("zmat"),
    }

    restart_candidates: list[dict[str, Any]] = []
    for key, label in (
        ("vectors_file", "movecs"),
        ("database_file", "database"),
        ("xyz_file", "xyz"),
        ("input_file", "input"),
    ):
        if preferred[key]:
            restart_candidates.append({"kind": label, "path": preferred[key]})

    return {
        "query_path": str(target),
        "job_dir": str(job_dir),
        "focus_stem": focus_stem,
        "preferred": preferred,
        "collections": collections,
        "related_files": related_files,
        "restart_candidates": restart_candidates,
    }




# TCE drafters moved to programs/nwchem/input/tce.py.
from chemtools.programs.nwchem.input.tce import (  # noqa: F401
    draft_nwchem_tce_input,
    validate_nwchem_tce_setup,
    draft_nwchem_atom_input,
    draft_nwchem_tce_restart_input,
)


def create_nwchem_input_variant(
    source_input: str,
    changes: dict[str, str],
    reason: str = "",
    output_path: str | None = None,
    write_file: bool = True,
) -> dict[str, Any]:
    """Create a versioned copy of an NWChem input with specified keyword changes.

    ``changes`` maps directive keys to new values.  Supported keys:

    * ``"memory"`` – e.g. ``"800 mb"``  (replaces the ``memory`` line)
    * ``"charge"`` – e.g. ``"2"``
    * ``"mult"`` – e.g. ``"6"``
    * ``"dft.iterations"`` – e.g. ``"200"``
    * ``"dft.xc"`` – e.g. ``"pbe0"``
    * ``"dft.convergence energy"`` – e.g. ``"1e-7"``
    * ``"scf.maxiter"`` – e.g. ``"200"``
    * ``"task"`` – e.g. ``"dft optimize"``  (replaces the last task line)
    * any other ``"block.keyword"`` pair – best-effort replacement inside
      ``block ... end``

    If *output_path* is ``None``, ``next_versioned_path`` is called on the
    source to generate ``_v2.nw``, ``_v3.nw``, etc.  The original file is
    never overwritten.
    """
    from chemtools.programs.nwchem.runner import next_versioned_path as _next_versioned_path

    src = Path(source_input)
    if not src.exists():
        raise FileNotFoundError(f"Source input not found: {source_input}")

    text = src.read_text(encoding="utf-8")
    original_text = text
    diff_summary: list[dict[str, str | None]] = []

    for key, new_value in changes.items():
        old_value: str | None = None

        if key == "memory":
            m = re.search(r"^(\s*memory\s+)(.+)$", text, re.MULTILINE | re.IGNORECASE)
            if m:
                old_value = m.group(2).strip()
                text = text[: m.start()] + m.group(1) + new_value + text[m.end() :]
            else:
                old_value = None
                text = f"memory {new_value}\n" + text

        elif key == "charge":
            m = re.search(r"^(\s*charge\s+)(\S+)", text, re.MULTILINE | re.IGNORECASE)
            if m:
                old_value = m.group(2)
                text = text[: m.start()] + m.group(1) + new_value + text[m.end() :]
            else:
                old_value = None
                # insert before first geometry block
                gm = re.search(r"^\s*geometry\b", text, re.MULTILINE | re.IGNORECASE)
                pos = gm.start() if gm else 0
                text = text[:pos] + f"charge {new_value}\n" + text[pos:]

        elif key == "mult":
            # NWChem: inside geometry block or as standalone keyword
            m = re.search(
                r"^(\s*(?:geometry\b[^\n]*\n(?:.*\n)*?\s*))?(mult(?:iplicity)?\s+)(\d+)",
                text, re.MULTILINE | re.IGNORECASE,
            )
            if not m:
                # Try as standalone
                m2 = re.search(r"^(\s*mult(?:iplicity)?\s+)(\d+)", text, re.MULTILINE | re.IGNORECASE)
                if m2:
                    old_value = m2.group(2)
                    text = text[: m2.start()] + m2.group(1) + new_value + text[m2.end() :]
                else:
                    old_value = None
                    gm = re.search(r"^\s*geometry\b", text, re.MULTILINE | re.IGNORECASE)
                    pos = gm.start() if gm else 0
                    text = text[:pos] + f"mult {new_value}\n" + text[pos:]
            else:
                old_value = m.group(3)
                text = text[: m.start(2)] + f"mult {new_value}" + text[m.end(3) :]

        elif key == "task":
            # Replace the last task line
            task_matches = list(re.finditer(r"^\s*task\s+.*$", text, re.MULTILINE | re.IGNORECASE))
            if task_matches:
                last = task_matches[-1]
                old_value = last.group(0).strip()
                text = text[: last.start()] + f"task {new_value}" + text[last.end() :]
            else:
                old_value = None
                text = text.rstrip() + f"\ntask {new_value}\n"

        elif "." in key:
            # block.keyword pattern, e.g. "dft.iterations" or "dft.xc"
            block_name, kw = key.split(".", 1)
            block_pat = re.compile(
                rf"^(\s*{re.escape(block_name)}\b[^\n]*\n)(.*?)(^\s*end\b)",
                re.MULTILINE | re.DOTALL | re.IGNORECASE,
            )
            bm = block_pat.search(text)
            if bm:
                block_body = bm.group(2)
                kw_pat = re.compile(
                    rf"^(\s*{re.escape(kw)}\s+)(.+)$",
                    re.MULTILINE | re.IGNORECASE,
                )
                km = kw_pat.search(block_body)
                if km:
                    old_value = km.group(2).strip()
                    new_body = block_body[: km.start()] + km.group(1) + new_value + block_body[km.end() :]
                else:
                    old_value = None
                    new_body = block_body + f"  {kw} {new_value}\n"
                text = text[: bm.start(2)] + new_body + text[bm.start(3) :]

        diff_summary.append({
            "key": key,
            "old": old_value,
            "new": new_value,
        })

    # Determine output path
    if output_path is None:
        output_path = _next_versioned_path(source_input)

    written_file: str | None = None
    if write_file:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text, encoding="utf-8")
        written_file = output_path

    return {
        "output_file": output_path,
        "written_file": written_file,
        "source_input": source_input,
        "diff_summary": diff_summary,
        "reason": reason,
        "input_text": text,
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

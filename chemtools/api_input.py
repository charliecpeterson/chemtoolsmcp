from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .common import detect_program, make_metadata, read_text, ELEMENT_TO_Z
from .basis import (
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
from .diagnostics import (
    analyze_frontier_orbitals as analyze_nwchem_frontier_orbitals,
    diagnose_nwchem_output,
    parse_scf,
    suggest_vectors_swaps as suggest_nwchem_vectors_swaps,
    summarize_nwchem_output,
)
from .nwchem_input import (
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
from . import nwchem
from ._api_utils import (
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
from .api_basis import render_nwchem_basis_setup
from .api_output import (
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


def analyze_imaginary_modes(
    path: str,
    significant_threshold_cm1: float = 20.0,
    top_atoms: int = 4,
) -> dict[str, Any]:
    contents = read_text(path)
    program = detect_program(contents)
    if program == "nwchem":
        return nwchem.analyze_imaginary_modes(
            path,
            contents,
            significant_threshold_cm1=significant_threshold_cm1,
            top_atoms=top_atoms,
        )
    raise ValueError(f"imaginary mode analysis is not implemented for {program or 'unknown'}")


def _select_best_optimization_frame(
    frames: list[dict],
    optimization_status: str,
) -> tuple[dict, str]:
    """Return (best_frame, reason) for restarting from an optimization output.

    Selection logic:
    - converged:  last frame — it IS the converged geometry.
    - incomplete: last frame — most recently optimized point; just needs more steps.
    - failed:     min-energy frame if the run diverged significantly (last energy
                  > 1 mHa above the minimum); otherwise last frame (failure was at
                  the best point — likely a trust-radius issue, not divergence).
    """
    last = frames[-1]

    if optimization_status in ("converged", "incomplete"):
        label = (
            "last_frame_is_converged_geometry"
            if optimization_status == "converged"
            else "last_frame_is_most_optimized_point_needs_more_steps"
        )
        return last, label

    # Failed run: check for divergence
    frames_with_e = [f for f in frames if f.get("energy_hartree") is not None]
    if not frames_with_e:
        return last, "failed_no_energy_data_using_last_frame"

    min_frame = min(frames_with_e, key=lambda f: f["energy_hartree"])
    last_e = last.get("energy_hartree")

    if min_frame["step"] == last["step"] or last_e is None:
        return last, "failed_last_frame_is_lowest_energy_trust_radius_issue_not_divergence"

    gap_mha = (last_e - min_frame["energy_hartree"]) * 1000  # mHa
    if gap_mha > 1.0:
        return (
            min_frame,
            f"failed_diverged_{gap_mha:.2f}mHa_above_minimum_at_step_{min_frame['step']}_using_min_energy_frame",
        )
    return last, f"failed_small_divergence_{gap_mha:.3f}mHa_using_last_frame"


def extract_nwchem_geometry(
    output_path: str,
    frame: "str | int" = "best",
    input_path: "str | None" = None,
) -> dict:
    """Extract a geometry from a NWChem optimization output as XYZ and NWChem block text.

    Works without the original input file.  When ``input_path`` is provided the
    NWChem geometry block preserves the original header (units, symmetry, etc.) and
    atom labels; otherwise a plain ``geometry units angstrom`` block is emitted.

    Parameters
    ----------
    output_path:
        Path to the NWChem ``.out`` file (must contain an optimization trajectory).
    frame:
        Which geometry to extract.  One of:

        - ``"best"``  — smart selection: converged/incomplete → last frame;
          failed → min-energy frame if divergence > 1 mHa, else last.
        - ``"last"``  — last frame regardless of status.
        - ``"first"`` — first frame (useful as a before/after comparison).
        - ``"min_energy"`` — frame with the lowest energy.
        - ``int``     — specific frame index (0-based).
    input_path:
        Optional path to the original ``.nw`` input; used to preserve geometry
        header/directives and atom labels.

    Returns
    -------
    dict with keys:
        xyz_text, nwchem_geometry_block, selected_frame (step/energy/metrics),
        selection_reason, optimization_status, frame_count, all_frames_summary.
    """
    import re as _re
    contents = read_text(output_path)
    trajectory = nwchem.parse_trajectory(output_path, contents, include_positions=True)
    frames = trajectory["frames"]

    if not frames:
        return {
            "available": False,
            "reason": "No optimization geometry frames found in output.",
            "optimization_status": trajectory.get("optimization_status", "unknown"),
        }

    opt_status = trajectory["optimization_status"]

    # --- Frame selection ---
    if frame == "best":
        chosen, reason = _select_best_optimization_frame(frames, opt_status)
    elif frame == "last":
        chosen, reason = frames[-1], "last_frame_requested"
    elif frame == "first":
        chosen, reason = frames[0], "first_frame_requested"
    elif frame == "min_energy":
        frames_with_e = [f for f in frames if f.get("energy_hartree") is not None]
        if frames_with_e:
            chosen = min(frames_with_e, key=lambda f: f["energy_hartree"])
            reason = f"min_energy_frame_at_step_{chosen['step']}"
        else:
            chosen, reason = frames[-1], "min_energy_requested_but_no_energy_data_using_last"
    elif isinstance(frame, int):
        if 0 <= frame < len(frames):
            chosen, reason = frames[frame], f"frame_index_{frame}_requested"
        else:
            raise ValueError(f"frame index {frame} out of range (0–{len(frames)-1})")
    else:
        raise ValueError(f"frame must be 'best', 'last', 'first', 'min_energy', or an int; got {frame!r}")

    positions = chosen.get("positions_angstrom")
    labels = chosen.get("labels", [])
    if not positions:
        return {
            "available": False,
            "reason": f"Frame {chosen.get('step')} has no position data (re-parse with include_positions=True).",
            "optimization_status": opt_status,
        }

    # Strip trailing digits from labels for element symbols (e.g. "C1" → "C")
    def _label_to_element(lbl: str) -> str:
        return _re.sub(r"\d+$", "", lbl).capitalize()

    elements = [_label_to_element(lbl) for lbl in labels]

    # --- XYZ text ---
    xyz_lines = [str(len(positions)), f"step={chosen.get('step')} E={chosen.get('energy_hartree')} ({reason})"]
    for elem, (x, y, z) in zip(elements, positions):
        xyz_lines.append(f"{elem:4s} {x:15.8f} {y:15.8f} {z:15.8f}")
    xyz_text = "\n".join(xyz_lines) + "\n"

    # --- NWChem geometry block ---
    # If input_path provided: preserve header+directives+original labels
    if input_path:
        try:
            orig_geom = extract_nwchem_geometry_block(input_path)
            orig_atoms = orig_geom.get("atoms", [])
            if len(orig_atoms) == len(positions):
                atom_dicts = [
                    {"label": a["label"], "element": a["element"],
                     "x": pos[0], "y": pos[1], "z": pos[2]}
                    for a, pos in zip(orig_atoms, positions)
                ]
                nw_block = render_nwchem_geometry_block(
                    orig_geom["header_line"], atom_dicts, directives=orig_geom["directives"]
                )
            else:
                input_path = None  # atom count mismatch, fall through to plain block
        except Exception:  # input file may be missing or have unreadable geometry block
            input_path = None

    if not input_path:
        nw_lines = ["geometry units angstrom"]
        for lbl, (x, y, z) in zip(labels, positions):
            nw_lines.append(f"  {lbl:6s} {x:15.8f} {y:15.8f} {z:15.8f}")
        nw_lines.append("end")
        nw_block = "\n".join(nw_lines)

    # --- All frames summary (without positions, for context) ---
    frames_summary = [
        {
            "index": f["index"],
            "step": f["step"],
            "energy_hartree": f.get("energy_hartree"),
            "gmax": f.get("gmax"),
            "grms": f.get("grms"),
        }
        for f in frames
    ]

    return {
        "available": True,
        "optimization_status": opt_status,
        "frame_count": len(frames),
        "selected_frame": {
            "index": chosen["index"],
            "step": chosen["step"],
            "energy_hartree": chosen.get("energy_hartree"),
            "gmax": chosen.get("gmax"),
            "grms": chosen.get("grms"),
            "xrms": chosen.get("xrms"),
            "xmax": chosen.get("xmax"),
        },
        "selection_reason": reason,
        "xyz_text": xyz_text,
        "nwchem_geometry_block": nw_block,
        "atom_count": len(positions),
        "elements": elements,
        "all_frames_summary": frames_summary,
    }


def draft_nwchem_optimization_followup_input(
    output_path: str,
    input_path: str,
    task_strategy: str = "auto",
    block_index: int = 0,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    if task_strategy not in {"auto", "optimize_only", "freq_only", "optimize_then_freq"}:
        raise ValueError("task_strategy must be one of: auto, optimize_only, freq_only, optimize_then_freq")

    contents = read_text(output_path)
    program = detect_program(contents)
    if program != "nwchem":
        raise ValueError(f"optimization follow-up drafting is not implemented for {program or 'unknown'}")

    trajectory = nwchem.parse_trajectory(output_path, contents, include_positions=True)
    if not trajectory["frames"]:
        raise ValueError("no optimization geometry frames were found in the output")

    diagnosis = diagnose_nwchem_output(output_path=output_path, input_path=input_path)
    input_summary = inspect_nwchem_input(input_path)
    geometry = extract_nwchem_geometry_block(input_path, block_index=block_index)
    best_frame, frame_selection_reason = _select_best_optimization_frame(
        frames=trajectory["frames"],
        optimization_status=trajectory["optimization_status"],
    )
    positions = best_frame.get("positions_angstrom")
    if not positions:
        raise ValueError("selected optimization frame does not include positions")
    if len(positions) != len(geometry["atoms"]):
        raise ValueError("input geometry atom count does not match last optimization frame atom count")

    selected_strategy = _select_optimization_follow_up_strategy(
        task_strategy=task_strategy,
        trajectory=trajectory,
        diagnosis=diagnosis,
    )
    follow_up_plan = _build_optimization_follow_up_plan(
        input_summary=input_summary,
        trajectory=trajectory,
        diagnosis=diagnosis,
        task_strategy=selected_strategy,
    )

    restarted_atoms = []
    for atom, coords in zip(geometry["atoms"], positions):
        restarted_atoms.append(
            {
                "label": atom["label"],
                "element": atom["element"],
                "x": coords[0],
                "y": coords[1],
                "z": coords[2],
            }
        )
    geometry_block = render_nwchem_geometry_block(
        geometry["header_line"],
        restarted_atoms,
        directives=geometry["directives"],
    )
    replaced_geometry = replace_nwchem_geometry_block(input_path, geometry_block, block_index=block_index)
    final_text = _replace_tasks_in_text(input_path, replaced_geometry["text"], follow_up_plan["task_lines"])["text"]
    if selected_strategy == "freq_only":
        final_text = _remove_keyword_blocks(final_text, {"driver"})

    resolved_base_name = base_name or _default_optimization_follow_up_base_name(
        input_path=input_path,
        strategy=selected_strategy,
    )
    resolved_title = title or _default_optimization_follow_up_title(selected_strategy)
    final_text = _replace_or_insert_keyword_line(final_text, "start", f"start {resolved_base_name}")
    final_text = _replace_or_insert_keyword_line(final_text, "title", f'title "{resolved_title}"', insert_after="start")
    final_text, vectors_output = _ensure_module_vectors_output_in_text(
        final_text,
        module=follow_up_plan["module"],
        vectors_output=f"{resolved_base_name}.movecs",
    )

    file_plan = _build_simple_input_file_plan(
        input_path=input_path,
        output_dir=output_dir,
        base_name=resolved_base_name,
    )
    written_file: str | None = None
    if write_file:
        written_file = _write_text_file(final_text, file_plan["input_file"])

    return {
        "metadata": make_metadata(output_path, contents, "nwchem"),
        "input_file": input_path,
        "input_summary": input_summary,
        "selected_frame": {
            "step": best_frame["step"],
            "energy_hartree": best_frame["energy_hartree"],
            "gmax": best_frame.get("gmax"),
            "grms": best_frame.get("grms"),
            "xrms": best_frame.get("xrms"),
            "xmax": best_frame.get("xmax"),
            "selection_reason": frame_selection_reason,
        },
        "trajectory_summary": {
            "optimization_status": trajectory["optimization_status"],
            "restart_recommended": trajectory["restart_recommended"],
            "step_count": trajectory["step_count"],
            "last_step": trajectory["last_step"],
            "final_energy_hartree": trajectory["final_energy_hartree"],
        },
        "follow_up_plan": follow_up_plan,
        "geometry_block_text": geometry_block,
        "vectors_output": vectors_output,
        "input_text": final_text,
        "file_plan": file_plan,
        "written_file": written_file,
    }


def displace_geometry_along_mode(
    path: str,
    mode_number: int | None = None,
    amplitude_angstrom: float = 0.15,
    significant_threshold_cm1: float = 20.0,
) -> dict[str, Any]:
    contents = read_text(path)
    program = detect_program(contents)
    if program == "nwchem":
        return nwchem.displace_geometry_along_mode(
            path,
            contents,
            mode_number=mode_number,
            amplitude_angstrom=amplitude_angstrom,
            significant_threshold_cm1=significant_threshold_cm1,
        )
    raise ValueError(f"mode displacement is not implemented for {program or 'unknown'}")


def draft_nwchem_imaginary_mode_inputs(
    output_path: str,
    input_path: str,
    mode_number: int | None = None,
    amplitude_angstrom: float = 0.15,
    significant_threshold_cm1: float = 20.0,
    add_noautosym: bool = True,
    enforce_symmetry_c1: bool = True,
    block_index: int = 0,
    task_strategy: str = "auto",
    output_dir: str | None = None,
    base_name: str | None = None,
    write_files: bool = False,
) -> dict[str, Any]:
    if task_strategy not in {"auto", "optimize_only", "optimize_then_freq"}:
        raise ValueError("task_strategy must be one of: auto, optimize_only, optimize_then_freq")

    displaced = displace_geometry_along_mode(
        output_path,
        mode_number=mode_number,
        amplitude_angstrom=amplitude_angstrom,
        significant_threshold_cm1=significant_threshold_cm1,
    )
    geometry = extract_nwchem_geometry_block(input_path, block_index=block_index)
    input_summary = inspect_nwchem_input(input_path)
    header_line = geometry["header_line"]
    if add_noautosym and "noautosym" not in header_line.lower():
        header_line = header_line.rstrip() + " noautosym"

    directives = list(geometry["directives"])
    if enforce_symmetry_c1:
        directives = [directive for directive in directives if not directive.lower().startswith("symmetry ")]
        directives.insert(0, "symmetry c1")

    plus_block = render_nwchem_geometry_block(header_line, displaced["plus_geometry"]["atoms"], directives=directives)
    minus_block = render_nwchem_geometry_block(
        header_line,
        displaced["minus_geometry"]["atoms"],
        directives=directives,
    )

    plus_geometry_replaced = replace_nwchem_geometry_block(input_path, plus_block, block_index=block_index)
    minus_geometry_replaced = replace_nwchem_geometry_block(input_path, minus_block, block_index=block_index)

    follow_up_plan = _build_imaginary_follow_up_plan(
        input_summary=input_summary,
        stability_assessment=displaced.get("stability_assessment"),
        selected_mode=displaced["selected_mode"],
        task_strategy=task_strategy,
    )
    plus_input = _replace_tasks_in_text(input_path, plus_geometry_replaced["text"], follow_up_plan["task_lines"])
    minus_input = _replace_tasks_in_text(input_path, minus_geometry_replaced["text"], follow_up_plan["task_lines"])

    file_plan = _build_imaginary_output_file_plan(
        input_path=input_path,
        selected_mode=displaced["selected_mode"],
        output_dir=output_dir,
        base_name=base_name,
    )
    plus_vectors_output = f"{Path(file_plan['plus_file']).stem}.movecs"
    minus_vectors_output = f"{Path(file_plan['minus_file']).stem}.movecs"
    plus_text, _ = _ensure_module_vectors_output_in_text(
        plus_input["text"],
        module=follow_up_plan["module"],
        vectors_output=plus_vectors_output,
    )
    minus_text, _ = _ensure_module_vectors_output_in_text(
        minus_input["text"],
        module=follow_up_plan["module"],
        vectors_output=minus_vectors_output,
    )
    written_files: dict[str, str] | None = None
    if write_files:
        written_files = _write_imaginary_input_files(
            plus_text=plus_text,
            minus_text=minus_text,
            plus_path=file_plan["plus_file"],
            minus_path=file_plan["minus_file"],
        )

    return {
        "metadata": make_metadata(output_path, read_text(output_path), "nwchem"),
        "input_file": input_path,
        "selected_mode": displaced["selected_mode"],
        "amplitude_angstrom": displaced["amplitude_angstrom"],
        "stability_assessment": displaced.get("stability_assessment"),
        "follow_up_plan": follow_up_plan,
        "file_plan": file_plan,
        "plus_vectors_output": plus_vectors_output,
        "minus_vectors_output": minus_vectors_output,
        "plus_geometry_block": plus_block,
        "minus_geometry_block": minus_block,
        "plus_input_text": plus_text,
        "minus_input_text": minus_text,
        "written_files": written_files,
    }


def draft_nwchem_vectors_swap_input(
    output_path: str,
    input_path: str,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    module: str | None = None,
    block_index: int = -1,
    task_operation: str = "energy",
    iterations: int | None = 500,
    smear: float | None = 0.001,
    convergence_damp: int | None = 30,
    convergence_ncydp: int | None = 30,
    population_print: str | None = "mulliken",
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    contents = read_text(output_path)
    program = detect_program(contents)
    if program != "nwchem":
        raise ValueError(f"vectors swap drafting is not implemented for {program or 'unknown'}")

    input_summary = inspect_nwchem_input(input_path)
    module_name = module or _select_primary_task_module(input_summary)
    resolved_base_name = base_name or f"{Path(input_path).stem}_swap"
    resolved_vectors_input = vectors_input or f"{Path(output_path).stem}.movecs"
    resolved_vectors_output = vectors_output or f"{resolved_base_name}.movecs"

    suggestion_payload = suggest_vectors_swaps(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
        vectors_input=resolved_vectors_input,
        vectors_output=resolved_vectors_output,
    )
    suggestion = suggestion_payload["suggestion"]
    if not suggestion.get("available"):
        raise ValueError("no actionable vectors swap suggestion was identified for this output")

    module_block = extract_nwchem_module_block(input_path, module=module_name, block_index=block_index)
    rewritten_body_lines = _rewrite_module_body_for_vectors_swap(
        module_block["body_lines"],
        suggestion["vectors_block"],
        iterations=iterations,
        smear=smear,
        convergence_damp=convergence_damp,
        convergence_ncydp=convergence_ncydp,
        population_print=population_print,
    )
    rewritten_module_block = render_nwchem_module_block(module_block["header_line"], rewritten_body_lines)
    replaced_module = replace_nwchem_module_block(
        input_path,
        rewritten_module_block,
        module=module_name,
        block_index=block_index,
    )

    task_lines = [f"task {module_name} {task_operation}"]
    replaced_tasks = _replace_tasks_in_text(input_path, replaced_module["text"], task_lines)

    final_text = replaced_tasks["text"]
    if task_operation in {"energy", "property", "freq"}:
        final_text = _remove_keyword_blocks(final_text, {"driver"})
    if task_operation != "property":
        final_text = _remove_keyword_blocks(final_text, {"property"})
    resolved_title = title or f'{resolved_base_name}: push metal-centered orbitals into SOMO positions'
    final_text = _replace_or_insert_keyword_line(final_text, "start", f"start {resolved_base_name}")
    final_text = _replace_or_insert_keyword_line(final_text, "title", f'title "{resolved_title}"', insert_after="start")

    file_plan = _build_vectors_swap_file_plan(
        input_path=input_path,
        output_dir=output_dir,
        base_name=resolved_base_name,
        vectors_output=resolved_vectors_output,
    )
    written_file: str | None = None
    if write_file:
        written_file = _write_text_file(final_text, file_plan["input_file"])

    return {
        "metadata": make_metadata(output_path, contents, "nwchem"),
        "input_file": input_path,
        "input_summary": input_summary,
        "module": module_name,
        "module_block_index": block_index,
        "task_lines": task_lines,
        "suggestion": suggestion,
        "vectors_input": resolved_vectors_input,
        "vectors_output": resolved_vectors_output,
        "module_block_text": rewritten_module_block,
        "input_text": final_text,
        "file_plan": file_plan,
        "written_file": written_file,
    }


def draft_nwchem_property_check_input(
    input_path: str,
    reference_output_path: str | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    module: str | None = None,
    block_index: int = -1,
    property_keywords: list[str] | None = None,
    task_strategy: str = "auto",
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    iterations: int | None = 1,
    convergence_energy: str | None = "1e3",
    smear: float | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    input_summary = inspect_nwchem_input(input_path)
    module_name = module or _select_primary_task_module(input_summary)
    input_stem = Path(input_path).stem
    resolved_base_name = base_name or f"{input_stem}_prop"
    resolved_vectors_input = vectors_input or f"{input_stem}.movecs"
    resolved_vectors_output = vectors_output or f"{resolved_base_name}.movecs"
    resolved_property_keywords = property_keywords or ["mulliken"]
    if task_strategy not in {"auto", "property", "energy"}:
        raise ValueError("task_strategy must be one of: auto, property, energy")

    selected_task_operation = "property"
    selected_iterations = iterations
    selected_convergence_energy = convergence_energy
    selected_smear = smear
    strategy_notes: list[str] = []
    reference_state_review: dict[str, Any] | None = None
    reference_diagnosis: dict[str, Any] | None = None

    if task_strategy == "energy":
        selected_task_operation = "energy"
        strategy_notes.append("explicit_energy_strategy_requested")
    elif task_strategy == "auto" and reference_output_path:
        reference_state_review = check_spin_charge_state(
            output_path=reference_output_path,
            input_path=input_path,
            expected_metal_elements=expected_metal_elements,
            expected_somo_count=expected_somo_count,
        )
        reference_diagnosis = diagnose_nwchem_output(
            output_path=reference_output_path,
            input_path=input_path,
            expected_metal_elements=expected_metal_elements,
            expected_somo_count=expected_somo_count,
        )
        if (
            reference_state_review.get("assessment") == "suspicious"
            or reference_diagnosis.get("failure_class") in {"wrong_state_convergence", "scf_nonconvergence"}
        ):
            selected_task_operation = "energy"
            strategy_notes.append("auto_strategy_downgraded_to_energy_due_to_unstable_or_suspicious_state")

    if selected_task_operation == "energy":
        if selected_iterations in {None, 1}:
            selected_iterations = 80
        if selected_convergence_energy == "1e3":
            selected_convergence_energy = None

    module_block = extract_nwchem_module_block(input_path, module=module_name, block_index=block_index)
    rewritten_body_lines = _rewrite_module_body_for_property_check(
        module_block["body_lines"],
        vectors_input=resolved_vectors_input,
        vectors_output=resolved_vectors_output,
        iterations=selected_iterations,
        convergence_energy=selected_convergence_energy,
        smear=selected_smear,
        include_mulliken_in_module=selected_task_operation == "energy"
        and "mulliken" in {keyword.strip().lower() for keyword in resolved_property_keywords},
    )
    rewritten_module_block = render_nwchem_module_block(module_block["header_line"], rewritten_body_lines)
    replaced_module = replace_nwchem_module_block(
        input_path,
        rewritten_module_block,
        module=module_name,
        block_index=block_index,
    )

    property_block = _render_named_block("property", [f"  {keyword}" for keyword in resolved_property_keywords])
    if selected_task_operation == "property":
        with_property = _replace_or_insert_named_block(
            replaced_module["text"],
            "property",
            property_block,
            insert_before_task=True,
        )
        with_property = _remove_keyword_blocks(with_property, {"driver"})
        task_lines = [f"task {module_name} property"]
    else:
        with_property = _remove_keyword_blocks(replaced_module["text"], {"property", "driver"})
        task_lines = [f"task {module_name} energy"]
    replaced_tasks = _replace_tasks_in_text(input_path, with_property, task_lines)

    final_text = replaced_tasks["text"]
    resolved_title = title or (
        f'{resolved_base_name}: property check from chosen vectors'
        if selected_task_operation == "property"
        else f'{resolved_base_name}: state-check energy run from chosen vectors'
    )
    final_text = _replace_or_insert_keyword_line(final_text, "start", f"start {resolved_base_name}")
    final_text = _replace_or_insert_keyword_line(final_text, "title", f'title "{resolved_title}"', insert_after="start")

    file_plan = _build_vectors_swap_file_plan(
        input_path=input_path,
        output_dir=output_dir,
        base_name=resolved_base_name,
        vectors_output=resolved_vectors_output,
    )
    written_file: str | None = None
    if write_file:
        written_file = _write_text_file(final_text, file_plan["input_file"])

    return {
        "input_file": input_path,
        "input_summary": input_summary,
        "module": module_name,
        "module_block_index": block_index,
        "property_keywords": resolved_property_keywords,
        "task_strategy": task_strategy,
        "selected_task_operation": selected_task_operation,
        "strategy_notes": strategy_notes,
        "reference_output_file": reference_output_path,
        "reference_state_review": reference_state_review,
        "reference_diagnosis": reference_diagnosis,
        "task_lines": task_lines,
        "vectors_input": resolved_vectors_input,
        "vectors_output": resolved_vectors_output,
        "module_block_text": rewritten_module_block,
        "property_block_text": property_block,
        "input_text": final_text,
        "file_plan": file_plan,
        "written_file": written_file,
    }


def draft_nwchem_scf_stabilization_input(
    input_path: str,
    reference_output_path: str | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    module: str | None = None,
    block_index: int = -1,
    task_operation: str = "energy",
    iterations: int | None = None,
    smear: float | None = None,
    convergence_damp: int | None = None,
    convergence_ncydp: int | None = None,
    population_print: str | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    input_summary = inspect_nwchem_input(input_path)
    module_name = module or _select_primary_task_module(input_summary)
    input_stem = Path(input_path).stem
    resolved_base_name = base_name or f"{input_stem}_stabilize"

    module_vectors = inspect_nwchem_module_vectors(input_path, module=module_name, block_index=block_index)
    existing_vectors_input, existing_vectors_output = _extract_vectors_io_from_lines(module_vectors.get("vectors_lines") or [])
    resolved_vectors_input = vectors_input or existing_vectors_output or existing_vectors_input or f"{input_stem}.movecs"
    resolved_vectors_output = vectors_output or f"{resolved_base_name}.movecs"
    reference_diagnosis = None
    if reference_output_path:
        try:
            reference_diagnosis = diagnose_nwchem_output(output_path=reference_output_path, input_path=input_path)
        except Exception:  # reference output may be incomplete or from a failed run
            reference_diagnosis = None

    stabilization_strategy = _select_scf_stabilization_strategy(
        reference_diagnosis=reference_diagnosis,
        iterations=iterations,
        smear=smear,
        convergence_damp=convergence_damp,
        convergence_ncydp=convergence_ncydp,
        population_print=population_print,
    )

    module_block = extract_nwchem_module_block(input_path, module=module_name, block_index=block_index)
    rewritten_body_lines = _rewrite_module_body_for_scf_stabilization(
        module_block["body_lines"],
        vectors_input=resolved_vectors_input,
        vectors_output=resolved_vectors_output,
        iterations=stabilization_strategy["iterations"],
        smear=stabilization_strategy["smear"],
        convergence_damp=stabilization_strategy["convergence_damp"],
        convergence_ncydp=stabilization_strategy["convergence_ncydp"],
        population_print=stabilization_strategy["population_print"],
    )
    rewritten_module_block = render_nwchem_module_block(module_block["header_line"], rewritten_body_lines)
    replaced_module = replace_nwchem_module_block(
        input_path,
        rewritten_module_block,
        module=module_name,
        block_index=block_index,
    )

    final_text = _remove_keyword_blocks(replaced_module["text"], {"driver", "property"})
    task_lines = [f"task {module_name} {task_operation}"]
    replaced_tasks = _replace_tasks_in_text(input_path, final_text, task_lines)
    final_text = replaced_tasks["text"]
    resolved_title = title or f'{resolved_base_name}: stabilize SCF/state from previous vectors'
    final_text = _replace_or_insert_keyword_line(final_text, "start", f"start {resolved_base_name}")
    final_text = _replace_or_insert_keyword_line(final_text, "title", f'title "{resolved_title}"', insert_after="start")

    file_plan = _build_vectors_swap_file_plan(
        input_path=input_path,
        output_dir=output_dir,
        base_name=resolved_base_name,
        vectors_output=resolved_vectors_output,
    )
    written_file: str | None = None
    if write_file:
        written_file = _write_text_file(final_text, file_plan["input_file"])

    return {
        "input_file": input_path,
        "reference_output_file": reference_output_path,
        "reference_diagnosis": reference_diagnosis,
        "input_summary": input_summary,
        "module": module_name,
        "module_block_index": block_index,
        "task_operation": task_operation,
        "stabilization_strategy": stabilization_strategy["strategy"],
        "strategy_notes": stabilization_strategy["notes"],
        "iterations": stabilization_strategy["iterations"],
        "smear": stabilization_strategy["smear"],
        "convergence_damp": stabilization_strategy["convergence_damp"],
        "convergence_ncydp": stabilization_strategy["convergence_ncydp"],
        "population_print": stabilization_strategy["population_print"],
        "vectors_input": resolved_vectors_input,
        "vectors_output": resolved_vectors_output,
        "module_block_text": rewritten_module_block,
        "task_lines": task_lines,
        "input_text": final_text,
        "file_plan": file_plan,
        "written_file": written_file,
    }


def draft_nwchem_mcscf_input(
    input_path: str,
    reference_output_path: str,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    active_space_mode: str = "minimal",
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    state_label: str | None = None,
    symmetry: int | None = None,
    hessian: str = "exact",
    maxiter: int = 80,
    thresh: float | None = 1.0e-5,
    level: float | None = 0.6,
    lock_vectors: bool = True,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    if active_space_mode not in {"minimal", "expanded"}:
        raise ValueError("active_space_mode must be 'minimal' or 'expanded'")
    if hessian not in {"exact", "onel"}:
        raise ValueError("hessian must be 'exact' or 'onel'")

    input_summary = inspect_nwchem_input(input_path)
    active_space_payload = suggest_nwchem_mcscf_active_space(
        output_path=reference_output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
    )
    active_space = active_space_payload[f"{active_space_mode}_active_space"]
    if not active_space.get("active_orbitals"):
        parsed_mcscf = parse_mcscf_output(reference_output_path)
        parsed_settings = parsed_mcscf.get("settings") or {}
        fallback_active_orbitals = _coerce_api_int(parsed_settings.get("active_orbitals"))
        fallback_active_electrons = _coerce_api_int(parsed_settings.get("active_electrons"))
        fallback_inactive_shells = _coerce_api_int(parsed_settings.get("inactive_shells")) or 0
        if fallback_active_orbitals and fallback_active_electrons is not None:
            active_space = {
                "active_electrons": fallback_active_electrons,
                "active_orbitals": fallback_active_orbitals,
                "occupied_like_count": 0,
                "virtual_like_count": 0,
                "closed_shell_count": fallback_inactive_shells,
                "vector_numbers": [],
                "orbitals": [],
            }

    resolved_base_name = base_name or f"{Path(input_path).stem}_mcscf"
    primary_module = _select_primary_task_module(input_summary)
    try:
        module_vectors = inspect_nwchem_module_vectors(input_path, module=primary_module, block_index=-1)
        existing_vectors_input, existing_vectors_output = _extract_vectors_io_from_lines(module_vectors.get("vectors_lines") or [])
    except Exception:  # module block may be absent (e.g. no dft block in a scf-only input)
        existing_vectors_input, existing_vectors_output = (None, None)

    resolved_vectors_input = (
        vectors_input
        or existing_vectors_output
        or existing_vectors_input
        or f"{Path(reference_output_path).stem}.movecs"
    )
    resolved_vectors_output = vectors_output or f"{resolved_base_name}.movecs"

    reorder_plan = _build_mcscf_reorder_plan(active_space)
    mcscf_block = _render_mcscf_block(
        active_space=active_space,
        multiplicity=input_summary.get("multiplicity"),
        vectors_input=resolved_vectors_input,
        vectors_output=resolved_vectors_output,
        state_label=state_label,
        symmetry=symmetry,
        hessian=hessian,
        maxiter=maxiter,
        thresh=thresh,
        level=level,
        lock_vectors=lock_vectors,
        swap_pairs=reorder_plan["swap_pairs"],
    )

    contents = read_text(input_path)
    cleaned = _remove_keyword_blocks(contents, {"dft", "scf", "property", "driver", "mcscf"})
    with_mcscf = _replace_or_insert_named_block(cleaned, "mcscf", mcscf_block, insert_before_task=True)
    replaced_tasks = _replace_tasks_in_text(input_path, with_mcscf, ["task mcscf"])
    final_text = replaced_tasks["text"]

    resolved_title = title or f'{resolved_base_name}: mcscf from recommended active space'
    final_text = _replace_or_insert_keyword_line(final_text, "start", f"start {resolved_base_name}")
    final_text = _replace_or_insert_keyword_line(final_text, "title", f'title "{resolved_title}"', insert_after="start")

    file_plan = _build_vectors_swap_file_plan(
        input_path=input_path,
        output_dir=output_dir,
        base_name=resolved_base_name,
        vectors_output=resolved_vectors_output,
    )
    written_file: str | None = None
    if write_file:
        written_file = _write_text_file(final_text, file_plan["input_file"])

    return {
        "input_file": input_path,
        "reference_output_file": reference_output_path,
        "input_summary": input_summary,
        "active_space_mode": active_space_mode,
        "active_space": active_space,
        "reorder_plan": reorder_plan,
        "vectors_input": resolved_vectors_input,
        "vectors_output": resolved_vectors_output,
        "state_label": state_label,
        "symmetry": symmetry,
        "hessian": hessian,
        "maxiter": maxiter,
        "thresh": thresh,
        "level": level,
        "lock_vectors": lock_vectors,
        "module_block_text": mcscf_block,
        "task_lines": ["task mcscf"],
        "input_text": final_text,
        "file_plan": file_plan,
        "written_file": written_file,
    }


def draft_nwchem_mcscf_retry_input(
    output_path: str,
    input_path: str,
    expected_metal_elements: list[str] | None = None,
    active_space_mode: str = "auto",
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    state_label: str | None = None,
    symmetry: int | None = None,
    hessian: str | None = None,
    maxiter: int | None = None,
    thresh: float | None = None,
    level: float | None = None,
    lock_vectors: bool = True,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    if active_space_mode not in {"auto", "minimal", "expanded"}:
        raise ValueError("active_space_mode must be 'auto', 'minimal', or 'expanded'")

    review = review_nwchem_mcscf_case(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
    )
    parsed = review["raw_mcscf"]
    settings = parsed.get("settings") or {}
    convergence_assessment = (review.get("convergence_review") or {}).get("assessment")
    occupation_assessment = (review.get("occupation_review") or {}).get("assessment")

    if active_space_mode == "auto":
        resolved_active_space_mode = "expanded" if occupation_assessment == "overly_pinned_active_space" else "minimal"
    else:
        resolved_active_space_mode = active_space_mode

    strategy_notes: list[str] = []
    if parsed["failure_mode"] == "input_parse_error":
        retry_strategy = "syntax_cleanup_retry"
        resolved_hessian = hessian or "exact"
        resolved_maxiter = maxiter or max(_coerce_api_int(settings.get("maxiter")) or 0, 80)
        resolved_thresh = thresh if thresh is not None else (_coerce_api_float(settings.get("thresh")) or 1.0e-5)
        resolved_level = level if level is not None else max(_coerce_api_float(settings.get("initial_levelshift")) or 0.0, 0.6)
        strategy_notes.append("removed problematic state syntax and rebuilt the mcscf block from parsed defaults")
    elif convergence_assessment in {"input_or_convergence_failure", "incomplete_mcscf_convergence"}:
        retry_strategy = "stronger_convergence_retry"
        resolved_hessian = hessian or "exact"
        resolved_maxiter = maxiter or max(_coerce_api_int(settings.get("maxiter")) or 0, 120)
        resolved_thresh = thresh if thresh is not None else (_coerce_api_float(settings.get("thresh")) or 1.0e-5)
        resolved_level = level if level is not None else max(_coerce_api_float(settings.get("initial_levelshift")) or 0.0, 0.6)
        strategy_notes.append("increased macroiteration budget for a previously incomplete or fragile mcscf run")
    elif convergence_assessment == "converged_with_stiff_orbital_optimization":
        retry_strategy = "stiff_but_converged_refinement"
        resolved_hessian = hessian or "exact"
        resolved_maxiter = maxiter or max(_coerce_api_int(settings.get("maxiter")) or 0, 120)
        resolved_thresh = thresh if thresh is not None else (_coerce_api_float(settings.get("thresh")) or 1.0e-5)
        resolved_level = level if level is not None else max(_coerce_api_float(settings.get("initial_levelshift")) or 0.0, 0.6)
        strategy_notes.append("kept exact hessian and a higher macroiteration budget because the previous mcscf converged stiffly")
    else:
        retry_strategy = "active_space_refinement_retry"
        resolved_hessian = hessian or (settings.get("hessian") or "exact")
        resolved_maxiter = maxiter or max(_coerce_api_int(settings.get("maxiter")) or 0, 80)
        resolved_thresh = thresh if thresh is not None else (_coerce_api_float(settings.get("thresh")) or 1.0e-5)
        resolved_level = level if level is not None else (_coerce_api_float(settings.get("initial_levelshift")) or 0.6)
        strategy_notes.append("reused the stable mcscf settings and focused the retry on active-space refinement")

    if resolved_active_space_mode == "expanded":
        strategy_notes.append("using expanded active space because the current active window looks too pinned or needs more flexibility")

    resolved_base_name = base_name or f"{Path(input_path).stem}_mcscf_retry"
    resolved_vectors_input = vectors_input or settings.get("vectors_output") or settings.get("vectors_input")
    resolved_vectors_output = vectors_output or f"{resolved_base_name}.movecs"
    resolved_title = title or f"{resolved_base_name}: {retry_strategy.replace('_', ' ')}"

    drafted = draft_nwchem_mcscf_input(
        input_path=input_path,
        reference_output_path=output_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=(review.get("input_summary") or {}).get("multiplicity", 1) - 1 if (review.get("input_summary") or {}).get("multiplicity") else None,
        active_space_mode=resolved_active_space_mode,
        vectors_input=resolved_vectors_input,
        vectors_output=resolved_vectors_output,
        state_label=state_label,
        symmetry=symmetry,
        hessian=resolved_hessian,
        maxiter=resolved_maxiter,
        thresh=resolved_thresh,
        level=resolved_level,
        lock_vectors=lock_vectors,
        output_dir=output_dir,
        base_name=resolved_base_name,
        title=resolved_title,
        write_file=write_file,
    )

    return {
        "output_file": output_path,
        "input_file": input_path,
        "retry_strategy": retry_strategy,
        "strategy_notes": strategy_notes,
        "mcscf_review": {
            "status": review["status"],
            "failure_mode": review["failure_mode"],
            "recommended_next_action": review["recommended_next_action"],
            "convergence_assessment": convergence_assessment,
            "occupation_assessment": occupation_assessment,
        },
        "resolved_settings": {
            "active_space_mode": resolved_active_space_mode,
            "vectors_input": resolved_vectors_input,
            "vectors_output": resolved_vectors_output,
            "hessian": resolved_hessian,
            "maxiter": resolved_maxiter,
            "thresh": resolved_thresh,
            "level": resolved_level,
            "lock_vectors": lock_vectors,
        },
        "drafted_input": drafted,
        "input_text": drafted["input_text"],
        "file_plan": drafted["file_plan"],
        "written_file": drafted["written_file"],
    }


def draft_nwchem_cube_input(
    input_path: str,
    vectors_input: str,
    orbital_vectors: list[int] | None = None,
    density_modes: list[str] | None = None,
    orbital_spin: str = "total",
    orbital_requests: list[dict[str, Any]] | None = None,
    extent_angstrom: float = 6.0,
    grid_points: int = 120,
    gaussian: bool = True,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    resolved_orbitals = orbital_vectors or []
    resolved_orbital_requests = orbital_requests or []
    resolved_density_modes = [mode.lower() for mode in (density_modes or [])]
    if not resolved_orbitals and not resolved_density_modes and not resolved_orbital_requests:
        raise ValueError("provide at least one orbital vector or density mode for cube drafting")

    input_summary = inspect_nwchem_input(input_path)
    input_stem = Path(input_path).stem
    resolved_base_name = base_name or f"{input_stem}_cubes"

    dplot_blocks: list[str] = []
    cube_outputs: list[str] = []

    for mode in resolved_density_modes:
        if mode not in {"total", "spindens"}:
            raise ValueError("density_modes entries must be 'total' or 'spindens'")
        output_name = f"{resolved_base_name}_{mode}.cube"
        cube_outputs.append(output_name)
        dplot_blocks.append(
            _render_dplot_density_block(
                vectors_input=vectors_input,
                output_name=output_name,
                density_mode=mode,
                extent_angstrom=extent_angstrom,
                grid_points=grid_points,
                gaussian=gaussian,
            )
        )

    for vector_number in resolved_orbitals:
        output_name = f"{resolved_base_name}_mo_{vector_number:03d}.cube"
        cube_outputs.append(output_name)
        dplot_blocks.append(
            _render_dplot_orbital_block(
                vectors_input=vectors_input,
                output_name=output_name,
                vector_number=vector_number,
                spin=orbital_spin,
                title=f"Orbital {vector_number}",
                extent_angstrom=extent_angstrom,
                grid_points=grid_points,
                gaussian=gaussian,
            )
        )

    for request in resolved_orbital_requests:
        vector_number = int(request["vector_number"])
        spin = str(request.get("spin") or orbital_spin)
        output_name = request.get("output_name") or f"{resolved_base_name}_{spin}_mo_{vector_number:03d}.cube"
        cube_outputs.append(output_name)
        dplot_blocks.append(
            _render_dplot_orbital_block(
                vectors_input=vectors_input,
                output_name=output_name,
                vector_number=vector_number,
                spin=spin,
                title=request.get("title") or f"{spin.capitalize()} orbital {vector_number}",
                extent_angstrom=extent_angstrom,
                grid_points=grid_points,
                gaussian=gaussian,
            )
        )

    contents = read_text(input_path)
    cleaned = _remove_keyword_blocks(contents, {"dplot", "property", "driver"})
    cleaned = _append_named_blocks_before_tasks(cleaned, dplot_blocks)
    replaced_tasks = _replace_tasks_in_text(input_path, cleaned, ["task dplot"])
    final_text = replaced_tasks["text"]
    resolved_title = title or f'{resolved_base_name}: cube generation from chosen vectors'
    final_text = _replace_or_insert_keyword_line(final_text, "start", f"start {resolved_base_name}")
    final_text = _replace_or_insert_keyword_line(final_text, "title", f'title "{resolved_title}"', insert_after="start")

    file_plan = _build_cube_file_plan(
        input_path=input_path,
        output_dir=output_dir,
        base_name=resolved_base_name,
        cube_outputs=cube_outputs,
    )
    written_file: str | None = None
    if write_file:
        written_file = _write_text_file(final_text, file_plan["input_file"])

    return {
        "input_file": input_path,
        "input_summary": input_summary,
        "vectors_input": vectors_input,
        "orbital_vectors": resolved_orbitals,
        "orbital_requests": resolved_orbital_requests,
        "density_modes": resolved_density_modes,
        "orbital_spin": orbital_spin,
        "extent_angstrom": extent_angstrom,
        "grid_points": grid_points,
        "dplot_block_count": len(dplot_blocks),
        "dplot_blocks": dplot_blocks,
        "input_text": final_text,
        "file_plan": file_plan,
        "written_file": written_file,
    }


def draft_nwchem_frontier_cube_input(
    output_path: str,
    input_path: str,
    vectors_input: str | None = None,
    include_somos: bool = True,
    include_homo: bool = True,
    include_lumo: bool = True,
    include_density_modes: list[str] | None = None,
    extent_angstrom: float = 6.0,
    grid_points: int = 120,
    gaussian: bool = True,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    mos = parse_mos(output_path, top_n=12)
    resolved_base_name = base_name or f"{Path(input_path).stem}_frontier_cubes"
    resolved_vectors_input = vectors_input or f"{Path(output_path).stem}.movecs"
    density_modes = include_density_modes or []

    orbital_requests: list[dict[str, Any]] = []
    seen_vectors: set[tuple[str, int]] = set()
    for spin, channel in mos.get("spin_channels", {}).items():
        if include_somos:
            for index, orbital in enumerate(channel.get("somos", []), start=1):
                vector_number = orbital["vector_number"]
                key = (spin, vector_number)
                if key in seen_vectors:
                    continue
                seen_vectors.add(key)
                orbital_requests.append(
                    {
                        "spin": spin,
                        "vector_number": vector_number,
                        "title": f"{spin.capitalize()} SOMO {index} (vector {vector_number})",
                        "output_name": f"{resolved_base_name}_{spin}_somo_{index}_v{vector_number:03d}.cube",
                    }
                )
        if include_homo and (orbital := channel.get("homo")) is not None:
            vector_number = orbital["vector_number"]
            key = (spin, vector_number)
            if key not in seen_vectors:
                seen_vectors.add(key)
                orbital_requests.append(
                    {
                        "spin": spin,
                        "vector_number": vector_number,
                        "title": f"{spin.capitalize()} HOMO (vector {vector_number})",
                        "output_name": f"{resolved_base_name}_{spin}_homo_v{vector_number:03d}.cube",
                    }
                )
        if include_lumo and (orbital := channel.get("lumo")) is not None:
            vector_number = orbital["vector_number"]
            key = (spin, vector_number)
            if key not in seen_vectors:
                seen_vectors.add(key)
                orbital_requests.append(
                    {
                        "spin": spin,
                        "vector_number": vector_number,
                        "title": f"{spin.capitalize()} LUMO (vector {vector_number})",
                        "output_name": f"{resolved_base_name}_{spin}_lumo_v{vector_number:03d}.cube",
                    }
                )

    drafted = draft_nwchem_cube_input(
        input_path=input_path,
        vectors_input=resolved_vectors_input,
        orbital_requests=orbital_requests,
        density_modes=density_modes,
        extent_angstrom=extent_angstrom,
        grid_points=grid_points,
        gaussian=gaussian,
        output_dir=output_dir,
        base_name=resolved_base_name,
        title=title,
        write_file=write_file,
    )
    drafted.update(
        {
            "output_file": output_path,
            "frontier_requests": orbital_requests,
            "metadata": make_metadata(output_path, read_text(output_path), "nwchem"),
        }
    )
    return drafted



def draft_initial_geometry(
    atoms: list[str],
    output_path: str,
    comment: str | None = None,
    central_atom: str | None = None,
) -> dict[str, Any]:
    """Generate a plausible initial geometry XYZ file from an atom list.

    Uses covalent radii sums for bond length estimates. Handles diatomics,
    MXn complexes (n=1..6) with symmetric placement, and linear chains.
    Never requires the caller to know bond lengths or 3-D coordinates.

    Parameters
    ----------
    atoms:
        Flat list of element symbols, e.g. ``["Fe", "Cl"]`` or
        ``["Fe", "Cl", "Cl", "Cl", "Cl"]``.  Repeats allowed.
    output_path:
        Where to write the XYZ file.
    comment:
        Optional comment line (second line of XYZ).  Auto-generated if None.
    central_atom:
        Hint for which element is the central/metal atom when building MXn
        geometry.  Inferred automatically if None (the element that appears
        fewest times, or the heaviest unique element).
    """
    from .nwchem_freq import COVALENT_RADII

    _FALLBACK_R = 1.20  # Å, used when element not in table

    def _r(elem: str) -> float:
        return COVALENT_RADII.get(elem, _FALLBACK_R)

    def _bond(a: str, b: str) -> float:
        return _r(a) + _r(b)

    n = len(atoms)
    if n == 0:
        raise ValueError("atoms list must not be empty")

    positions: list[tuple[float, float, float]] = []

    if n == 1:
        positions = [(0.0, 0.0, 0.0)]

    elif n == 2:
        # Diatomic along z
        bl = _bond(atoms[0], atoms[1])
        positions = [(0.0, 0.0, 0.0), (0.0, 0.0, bl)]

    else:
        # Try to identify central atom for MXn placement
        from collections import Counter
        counts = Counter(atoms)

        # Determine central atom: explicit hint, or fewest occurrences, or heaviest element
        from .common import ELEMENT_TO_Z
        if central_atom and central_atom in counts:
            center = central_atom
        elif len(counts) > 1:
            min_count = min(counts.values())
            candidates = [e for e, c in counts.items() if c == min_count]
            # Among candidates pick heaviest (most likely the metal)
            center = max(candidates, key=lambda e: ELEMENT_TO_Z.get(e, 0))
        else:
            center = atoms[0]

        ligands = [a for a in atoms if a != center]
        n_lig = len(ligands)

        # Unique ligand bond length (use first ligand type for MXn)
        r_ml = _bond(center, ligands[0]) if ligands else 2.0

        # Symmetric ligand positions around center at origin
        # Elements with lone pairs that produce bent AX2 geometries
        _BENT_ANGLE: dict[str, float] = {
            "O": 104.5, "S": 92.0, "Se": 91.0, "Te": 90.0,
            "N": 107.0, "P": 93.5, "As": 91.8, "Sb": 91.6,
            "C": 109.5, "Si": 109.5, "Ge": 91.0,
        }
        two_pi = 2.0 * math.pi
        if n_lig == 1:
            lig_coords = [(0.0, 0.0, r_ml)]
        elif n_lig == 2:
            if center in _BENT_ANGLE:
                angle_deg = _BENT_ANGLE[center]
                half = math.radians(angle_deg / 2)
                lig_coords = [
                    ( r_ml * math.sin(half),  r_ml * math.cos(half), 0.0),
                    (-r_ml * math.sin(half),  r_ml * math.cos(half), 0.0),
                ]
            else:
                lig_coords = [(0.0, 0.0, -r_ml), (0.0, 0.0, r_ml)]
        elif n_lig == 3:
            lig_coords = [
                (r_ml * math.cos(k * two_pi / 3), r_ml * math.sin(k * two_pi / 3), 0.0)
                for k in range(3)
            ]
        elif n_lig == 4:
            s = r_ml / math.sqrt(3)
            lig_coords = [(s, s, s), (s, -s, -s), (-s, s, -s), (-s, -s, s)]
        elif n_lig == 5:
            # Square pyramidal
            lig_coords = [
                (r_ml, 0.0, 0.0), (-r_ml, 0.0, 0.0),
                (0.0, r_ml, 0.0), (0.0, -r_ml, 0.0),
                (0.0, 0.0, r_ml),
            ]
        elif n_lig == 6:
            lig_coords = [
                (r_ml, 0.0, 0.0), (-r_ml, 0.0, 0.0),
                (0.0, r_ml, 0.0), (0.0, -r_ml, 0.0),
                (0.0, 0.0, r_ml), (0.0, 0.0, -r_ml),
            ]
        else:
            # Linear chain along z for anything larger
            lig_coords = [(0.0, 0.0, (k + 1) * r_ml) for k in range(n_lig)]

        # Reassemble: center first, then ligands in original order
        center_pos = (0.0, 0.0, 0.0)
        ordered_atoms: list[str] = [center]
        ordered_pos: list[tuple[float, float, float]] = [center_pos]
        lig_iter = iter(lig_coords)
        for a in atoms:
            if a == center and center not in ordered_atoms[1:]:
                continue  # already placed center
            else:
                ordered_atoms.append(a)
                ordered_pos.append(next(lig_iter))
        atoms = ordered_atoms
        positions = ordered_pos

    # Build XYZ content
    auto_comment = comment or f"Initial geometry guess — {' '.join(atoms)}"
    lines = [str(len(atoms)), auto_comment]
    bond_summary: list[dict[str, Any]] = []
    for elem, (x, y, z) in zip(atoms, positions):
        lines.append(f"{elem:4s}  {x:14.8f}  {y:14.8f}  {z:14.8f}")

    # Summarise bond lengths for the return value
    if len(atoms) == 2:
        bond_summary.append({
            "atoms": f"{atoms[0]}-{atoms[1]}",
            "length_angstrom": round(
                math.sqrt(sum((a - b) ** 2 for a, b in zip(positions[0], positions[1]))), 4
            ),
            "source": "covalent_radii_sum",
        })

    xyz_text = "\n".join(lines) + "\n"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xyz_text, encoding="utf-8")

    return {
        "output_file": str(out.resolve()),
        "n_atoms": len(atoms),
        "atoms": atoms,
        "positions_angstrom": [list(p) for p in positions],
        "bond_lengths_used": bond_summary,
        "comment": auto_comment,
        "note": (
            "Geometry is a covalent-radii guess — suitable for initial optimization only. "
            "Always run geometry optimization before any correlated calculation."
        ),
        "next_steps": [
            f"Call create_nwchem_dft_workflow_input with geometry_file='{out.resolve()}' "
            "to build the optimization input.",
            "Run lint_nwchem_input on the generated input before launching.",
        ],
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


def create_nwchem_dft_workflow_input(
    geometry_path: str,
    library_path: str,
    basis_assignments: dict[str, str],
    xc_functional: str,
    task_operations: list[str],
    *,
    ecp_assignments: dict[str, str] | None = None,
    default_basis: str | None = None,
    default_ecp: str | None = None,
    basis_block_name: str = "ao basis",
    basis_mode: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    dft_settings: list[str] | None = None,
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
    normalized_tasks = [_normalize_nwchem_task_operation(task) for task in task_operations]
    if not normalized_tasks:
        raise ValueError("at least one task operation is required")
    if not xc_functional.strip():
        raise ValueError("xc_functional is required")

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

    resolved_start_name = start_name or Path(geometry_path).stem or "nwchem_dft_job"
    resolved_title = title or f"{resolved_start_name}: dft {'+'.join(normalized_tasks)}"
    resolved_vectors_output = vectors_output or f"{resolved_start_name}.movecs"

    rendered_dft_settings = [
        line.rstrip() if line[:1].isspace() else f"  {line.rstrip()}"
        for line in (dft_settings or [])
        if str(line).strip()
    ]
    rendered_dft_settings = _apply_default_dft_settings(
        rendered_dft_settings,
        xc_functional=xc_functional,
        multiplicity=multiplicity,
        vectors_input=vectors_input,
        vectors_output=resolved_vectors_output,
    )

    module_block = render_nwchem_module_block("dft", rendered_dft_settings)
    task_lines = [f"task dft {task}" for task in normalized_tasks]

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
    if "optimize" in normalized_tasks:
        _ensure_driver_block(rendered_extra_blocks)
    sections.extend(rendered_extra_blocks)
    sections.append(module_block)
    sections.extend(task_lines)

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
        "module": "dft",
        "xc_functional": xc_functional,
        "task_operations": normalized_tasks,
        "charge": charge,
        "multiplicity": multiplicity,
        "basis_setup": basis_setup,
        "dft_settings": [line.strip() for line in rendered_dft_settings],
        "vectors_input": vectors_input,
        "vectors_output": resolved_vectors_output,
        "input_text": input_text,
        "file_plan": file_plan,
        "written_file": written_file,
        "inline_blocks": inline_blocks,
    }


def create_nwchem_dft_input_from_request(
    *,
    formula: str | None = None,
    geometry_path: str | None = None,
    library_path: str | None = None,
    basis_assignments: dict[str, str] | None = None,
    ecp_assignments: dict[str, str] | None = None,
    default_basis: str | None = None,
    default_ecp: str | None = None,
    xc_functional: str | None = None,
    task_operations: list[str] | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    dft_settings: list[str] | None = None,
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
    review = review_nwchem_input_request(
        formula=formula,
        geometry_path=geometry_path,
        library_path=library_path,
        basis_assignments=basis_assignments,
        ecp_assignments=ecp_assignments,
        default_basis=default_basis,
        default_ecp=default_ecp,
        module="dft",
        task_operations=task_operations,
        functional=xc_functional,
        charge=charge,
        multiplicity=multiplicity,
    )

    if not review["ready_to_create"]:
        return {
            "ready_to_create": False,
            "created": False,
            "review": review,
            "next_action": "provide_missing_requirements",
            "input_text": None,
            "written_file": None,
        }

    if not geometry_path:
        raise ValueError("geometry_path is required when the request is ready to create")
    if not library_path:
        raise ValueError("library_path is required when the request is ready to create")
    if not xc_functional or not xc_functional.strip():
        raise ValueError("xc_functional is required when the request is ready to create")

    created = create_nwchem_dft_workflow_input(
        geometry_path=geometry_path,
        library_path=library_path,
        basis_assignments=basis_assignments or {},
        ecp_assignments=ecp_assignments,
        default_basis=default_basis,
        default_ecp=default_ecp,
        xc_functional=xc_functional,
        task_operations=task_operations or ["energy"],
        charge=review["charge"],
        multiplicity=multiplicity,
        dft_settings=dft_settings,
        extra_blocks=extra_blocks,
        memory=memory,
        title=title,
        start_name=start_name,
        vectors_input=vectors_input,
        vectors_output=vectors_output,
        geometry_block_index=geometry_block_index,
        output_dir=output_dir,
        write_file=write_file,
        inline_blocks=inline_blocks,
    )
    # When the file was written, strip the large raw text fields from the response
    # (input_text, basis_block.text, ecp_block.text) — the file on disk is the source
    # of truth and returning those texts wastes tokens.
    if write_file and created.get("written_file"):
        created = dict(created)
        created.pop("input_text", None)
        bs = created.get("basis_setup")
        if isinstance(bs, dict):
            bs = dict(bs)
            if isinstance(bs.get("basis_block"), dict):
                bb = dict(bs["basis_block"])
                bb.pop("text", None)
                bs["basis_block"] = bb
            if isinstance(bs.get("ecp_block"), dict):
                eb = dict(bs["ecp_block"])
                eb.pop("text", None)
                bs["ecp_block"] = eb
            created["basis_setup"] = bs

    result: dict[str, Any] = {
        "ready_to_create": True,
        "created": True,
        "next_action": "input_created",
        **created,
    }
    # Include warnings from the review at the top level if present, but omit the full review dict
    if review.get("warnings"):
        result["warnings"] = review["warnings"]
    return result


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
        from .common import read_text as _rt2
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
        from .nwchem_input import extract_nwchem_geometry_block
        from .common import read_text as _read_text
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


def _normalize_stem_for_match(stem: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def _stem_tokens(stem: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", stem.lower()) if token]


# ---------------------------------------------------------------------------
# TCE input drafting
# ---------------------------------------------------------------------------

def _extract_ecp_nelec_from_input(
    input_path: str,
    basis_library_path: str | None = None,
) -> dict[str, int]:
    """Return {element: nelec} for all ECP-covered elements in an NWChem input.

    Tries inline ``nelec`` lines first (explicit ECP blocks), then falls back
    to the basis library for library-assigned ECPs.  Returns an empty dict if
    no ECP block exists.
    """
    from .nwchem_input import inspect_nwchem_ecp_block
    try:
        ecp_info = inspect_nwchem_ecp_block(input_path)
    except (ValueError, FileNotFoundError):
        # ValueError: no ECP block in file (expected for most inputs)
        # FileNotFoundError: input file doesn't exist
        return {}

    if not ecp_info.get("body_lines"):
        return {}

    # Inline nelec parsed directly from the ECP block body
    result: dict[str, int] = dict(ecp_info.get("nelec_by_element") or {})

    # Library-assigned elements: look up nelec from the basis library
    library_assignments = ecp_info.get("library_assignments") or {}
    if library_assignments and basis_library_path:
        from .api_basis import resolve_ecp
        for elem, ecp_name in library_assignments.items():
            if elem in result:
                continue
            try:
                resolved = resolve_ecp(ecp_name, [elem], basis_library_path)
                nelec = (resolved.get("nelec_by_element") or {}).get(elem)
                if nelec is not None:
                    result[elem] = nelec
            except Exception:  # ECP not in library for this element/name combination
                pass

    return result


def draft_nwchem_tce_input(
    scf_output_file: str,
    input_file: str,
    method: str = "mp2",
    freeze_count: int | None = None,
    swap_pairs: list[tuple[int, int]] | None = None,
    movecs_file: str | None = None,
    ecp_core_electrons: dict[str, int] | None = None,
    basis_library: str | None = None,
    start_name: str | None = None,
    title: str | None = None,
    memory: str | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    """Design a NWChem TCE input, inspecting SCF orbitals to determine freeze count.

    The agent MUST call this AFTER a completed SCF calculation so that the orbital
    ordering can be verified.  This function:

    1. Reads the SCF output and parses molecular orbitals.
    2. Computes a chemically-informed freeze count from the element list + ECP info.
    3. Checks the actual orbital ordering against the expected core pattern.
    4. Warns if swaps are needed (the agent should call swap_nwchem_movecs first).
    5. Returns a ready-to-use TCE input block with an explicit ``freeze N`` directive.

    Never uses ``freeze atomic`` — always emits an explicit integer.

    Parameters
    ----------
    scf_output_file:
        Path to the completed SCF/DFT output that contains the MO analysis.
    input_file:
        Path to the SCF input file (for geometry/basis/ECP metadata).
    method:
        TCE method: ``mp2``, ``ccsd``, or ``ccsd(t)``.  Default ``mp2``.
    freeze_count:
        Override the suggested freeze count.  If None, computed from chemistry.
    swap_pairs:
        List of (i, j) MO index pairs that have already been applied via
        swap_nwchem_movecs. If provided, the input notes that swaps were done.
    movecs_file:
        Path to the movecs file.  If None, inferred from the SCF output.
    ecp_core_electrons:
        ECP nelec values per element, e.g. ``{"Zn": 10, "I": 28}``.
    start_name, title, memory:
        NWChem input header directives.
    output_dir, base_name:
        Where to write the file (if write_file=True).
    write_file:
        If True, write the generated input to disk.
    """
    from .nwchem_tce import suggest_tce_freeze_count, analyze_tce_orbital_ordering
    from .common import read_text

    method_norm = method.strip().lower()
    valid_methods = {"mp2", "ccsd", "ccsd(t)", "ccsdt"}
    if method_norm not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}")
    tce_method_keyword = {
        "mp2": "mp2",
        "ccsd": "ccsd",
        "ccsd(t)": "ccsd(t)",
        "ccsdt": "ccsd(t)",
    }[method_norm]

    # --- Read SCF output and parse orbitals ---
    scf_contents = read_text(scf_output_file)
    mos_result = nwchem.parse_mos(scf_output_file, scf_contents)
    orbitals = mos_result.get("orbitals", [])

    # --- Infer elements from the input file ---
    input_summary = inspect_nwchem_input(input_file)
    elements = input_summary.get("elements", [])

    # --- Auto-detect ECP nelec from input file (if not provided by caller) ---
    ecp_auto_detected: bool = False
    if ecp_core_electrons is None:
        detected = _extract_ecp_nelec_from_input(input_file, basis_library_path=basis_library)
        if detected:
            ecp_core_electrons = detected
            ecp_auto_detected = True

    # --- Suggest freeze count ---
    freeze_suggestion = suggest_tce_freeze_count(
        elements,
        ecp_core_electrons=ecp_core_electrons,
        charge=input_summary.get("charge") or 0,
        multiplicity=input_summary.get("multiplicity") or 1,
    )
    suggested_freeze = freeze_suggestion["freeze_count"]
    effective_freeze = freeze_count if freeze_count is not None else suggested_freeze

    # --- Analyse orbital ordering ---
    ordering_analysis: dict[str, Any] = {}
    if orbitals and effective_freeze > 0:
        ordering_analysis = analyze_tce_orbital_ordering(orbitals, effective_freeze)

    # --- Determine start_name and movecs ---
    scf_stem = Path(scf_output_file).stem
    resolved_start = start_name or base_name or scf_stem

    # Try to infer movecs from the SCF output text
    resolved_movecs = movecs_file
    if resolved_movecs is None:
        for line in scf_contents.splitlines():
            if "output vectors" in line.lower() and "=" in line:
                candidate = line.split("=", 1)[-1].strip().strip("./")
                candidate_path = Path(scf_output_file).parent / candidate
                if candidate_path.exists():
                    resolved_movecs = str(candidate_path.resolve())
                break

    has_ordering_warnings = bool(ordering_analysis.get("warnings"))
    pending_swaps = ordering_analysis.get("swap_suggestions", [])
    swaps_applied = swap_pairs or []

    # --- Build tce block ---
    tce_lines: list[str] = [f"  {tce_method_keyword}"]
    tce_lines.append(f"  freeze {effective_freeze}")
    tce_block = "tce\n" + "\n".join(tce_lines) + "\nend"

    # Always save T1 and T2 amplitude files so parse_nwchem_tce_amplitudes can
    # compute T1/D1/T2 diagnostics after the run.
    save_t_directive = "set tce:save_t T T"

    # --- Build geometry block (always include with symmetry c1 for TCE) ---
    from .nwchem_input import extract_nwchem_geometry_block, render_nwchem_geometry_block
    geo_section: str | None = None
    charge_line: str | None = None
    try:
        geo = extract_nwchem_geometry_block(input_file)
        directives = [d for d in (geo.get("directives") or []) if not d.lower().startswith("symmetry")]
        directives.insert(0, "symmetry c1")
        geo_section = render_nwchem_geometry_block(geo["header_line"], geo["atoms"], directives=directives)
    except Exception:
        pass
    charge = input_summary.get("charge")
    if charge is not None:
        charge_line = f"charge {charge}"

    # --- Build scf block (reference type from multiplicity) ---
    mult = input_summary.get("multiplicity") or 1
    nopen = mult - 1
    if nopen > 0:
        scf_ref = "rohf"
        scf_lines = ["scf", f"  {scf_ref}", f"  nopen {nopen}", "  thresh 1e-8", "  maxiter 200"]
    else:
        scf_ref = "rhf"
        scf_lines = ["scf", f"  {scf_ref}", "  thresh 1e-8", "  maxiter 200"]
    if resolved_movecs:
        movecs_basename = Path(resolved_movecs).name
        tce_movecs_out = f"{resolved_start}.movecs"
        scf_lines.append(f"  vectors input {movecs_basename} output {tce_movecs_out}")
    scf_block = "\n".join(scf_lines) + "\nend"

    # --- Assemble explanatory comment ---
    freeze_comment_lines = [
        f"# TCE {tce_method_keyword.upper()} — freeze analysis",
        f"# Effective freeze count: {effective_freeze} orbitals",
    ]
    if freeze_suggestion["per_element"]:
        freeze_comment_lines.append("# Per-element core orbital counts:")
        for pe in freeze_suggestion["per_element"]:
            if pe.get("freeze_orbitals") is not None:
                n_atoms = pe.get("n_atoms", 1)
                ecp_note = (
                    f" (ECP removes {pe['ecp_orbitals_removed_per_atom']} orb/atom)"
                    if pe.get("ecp_electrons", 0) > 0
                    else ""
                )
                atom_str = f"{n_atoms}×" if n_atoms > 1 else ""
                freeze_comment_lines.append(
                    f"#   {pe['element']} ({n_atoms} atoms): "
                    f"{pe.get('all_electron_core_orbitals_per_atom', '?')} all-e core/atom"
                    f"{ecp_note} → {atom_str}{pe['freeze_orbitals_per_atom']}={pe['freeze_orbitals']} orbitals"
                )

    if ordering_analysis.get("proposed_freeze_orbitals"):
        freeze_comment_lines.append("# Proposed frozen MOs (from SCF output):")
        for orb_info in ordering_analysis["proposed_freeze_orbitals"]:
            char = orb_info.get("dominant_character") or "?"
            freeze_comment_lines.append(
                f"#   MO {orb_info['mo']:3d}: E={orb_info['energy_hartree']:10.4f} h  {char}"
            )

    if ordering_analysis.get("warnings"):
        freeze_comment_lines.append("#")
        freeze_comment_lines.append("# *** ORBITAL ORDERING WARNINGS ***")
        for w in ordering_analysis["warnings"]:
            freeze_comment_lines.append(f"# {w}")

    if pending_swaps and not swaps_applied:
        freeze_comment_lines.append("#")
        freeze_comment_lines.append(
            "# ACTION REQUIRED: run swap_nwchem_movecs for each pair BEFORE this input:"
        )
        for sw in pending_swaps:
            freeze_comment_lines.append(
                f"#   swap MO {sw['from_mo']} <-> MO {sw['to_mo']}: {sw['reason']}"
            )

    if swaps_applied:
        freeze_comment_lines.append("#")
        freeze_comment_lines.append("# Swaps already applied to movecs:")
        for s_i, s_j in swaps_applied:
            freeze_comment_lines.append(f"#   MO {s_i} <-> MO {s_j}")

    comment_block = "\n".join(freeze_comment_lines)

    # --- Extract basis/ECP blocks from SCF input file ---
    basis_section: str | None = None
    ecp_section: str | None = None
    try:
        basis_blocks = inspect_all_nwchem_basis_blocks(input_file)
        if basis_blocks:
            raw = basis_blocks[-1]
            basis_section = raw["header_line"] + "\n" + "\n".join(raw["body_lines"]) + "\nend"
    except Exception:
        pass
    try:
        ecp_info = inspect_nwchem_ecp_block(input_file)
        if ecp_info.get("body_lines"):
            ecp_section = ecp_info["header_line"] + "\n" + "\n".join(ecp_info["body_lines"]) + "\nend"
    except Exception:
        pass

    # --- Assemble input sections ---
    sections: list[str] = [
        f"start {resolved_start}",
        "echo",
    ]
    if memory:
        sections.append(f"memory {memory}")
    if geo_section:
        sections.append(geo_section)
    if charge_line:
        sections.append(charge_line)
    if basis_section:
        sections.append(basis_section)
    if ecp_section:
        sections.append(ecp_section)
    sections.append(comment_block)
    sections.append(scf_block)
    sections.append("task scf energy")
    sections.append(tce_block)
    sections.append(save_t_directive)
    sections.append("task tce energy")

    input_text = "\n\n".join(sections).rstrip() + "\n"

    # --- File plan ---
    method_tag = tce_method_keyword.replace("(", "").replace(")", "")
    out_stem = base_name or f"{resolved_start}_tce_{method_tag}"
    out_dir = Path(output_dir) if output_dir else Path(scf_output_file).parent
    out_path = out_dir / f"{out_stem}.nw"

    written_file: str | None = None
    if write_file:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(input_text, encoding="utf-8")
        written_file = str(out_path.resolve())

    return {
        "input_text": input_text,
        "written_file": written_file,
        "planned_output_file": str(out_path),
        "method": tce_method_keyword,
        "effective_freeze_count": effective_freeze,
        "suggested_freeze_count": suggested_freeze,
        "n_electrons": freeze_suggestion.get("n_electrons"),
        "n_correlated": freeze_suggestion.get("n_correlated"),
        "ecp_core_electrons": ecp_core_electrons or {},
        "ecp_auto_detected": ecp_auto_detected,
        "freeze_suggestion": freeze_suggestion,
        "orbital_ordering_analysis": ordering_analysis,
        "needs_orbital_swap": has_ordering_warnings,
        "pending_swap_suggestions": pending_swaps,
        "movecs_file": resolved_movecs,
        "elements": elements,
        "n_orbitals_parsed": len(orbitals),
        "warnings": freeze_suggestion.get("warnings", []) + ordering_analysis.get("warnings", []),
    }


def validate_nwchem_tce_setup(
    tce_input_path: str,
    scf_output_path: str | None = None,
) -> dict[str, Any]:
    """Cross-check a NWChem TCE input file for common setup errors.

    Catches issues before submitting the job:

    * Missing ``symmetry c1`` in the geometry block
    * ``freeze atomic`` (forbidden — always use explicit count)
    * No freeze directive (all electrons correlated — expensive and wrong)
    * Freeze count far outside the element-derived suggestion
    * Missing or unreachable vectors file
    * ROHF reference missing for open-shell system
    * No TCE method keyword found

    Parameters
    ----------
    tce_input_path:
        Path to the NWChem TCE input file to validate.
    scf_output_path:
        Optional path to the SCF output file.  If provided and the file does
        not yet exist, a warning is emitted.

    Returns
    -------
    dict with ``status`` ("ok" | "warnings" | "errors"), ``issues`` list,
    ``detected`` parsed fields, and ``summary`` text.
    """
    import re as _re
    from pathlib import Path
    from .nwchem_tce import suggest_tce_freeze_count as _suggest_freeze
    from .nwchem_input import inspect_nwchem_input

    issues: list[dict[str, Any]] = []

    def _err(code: str, message: str) -> None:
        issues.append({"level": "error", "code": code, "message": message})

    def _warn(code: str, message: str) -> None:
        issues.append({"level": "warning", "code": code, "message": message})

    # --- Read file content once ---
    contents = read_text(tce_input_path)
    contents_lower = contents.lower()

    # --- Parse high-level input summary ---
    tce_summary = inspect_nwchem_input(tce_input_path)
    elements: list[str] = tce_summary.get("elements") or []
    charge: int = tce_summary.get("charge") or 0
    multiplicity: int | None = tce_summary.get("multiplicity")
    open_shell = multiplicity is not None and multiplicity > 1

    # --- Extract raw tce block via regex ---
    tce_block_match = _re.search(r"\btce\b(.*?)\bend\b", contents, _re.IGNORECASE | _re.DOTALL)
    tce_block = tce_block_match.group(1) if tce_block_match else ""
    tce_block_lower = tce_block.lower()

    # --- Extract geometry block via regex ---
    geo_block_match = _re.search(r"\bgeometry\b(.*?)\bend\b", contents, _re.IGNORECASE | _re.DOTALL)
    geo_block = geo_block_match.group(1) if geo_block_match else ""

    # --- Extract SCF block via regex ---
    scf_block_match = _re.search(r"\bscf\b(.*?)\bend\b", contents, _re.IGNORECASE | _re.DOTALL)
    scf_block = scf_block_match.group(1) if scf_block_match else ""
    scf_block_lower = scf_block.lower()

    # --- Method check ---
    method_found: str | None = None
    for m in ("ccsd(t)", "ccsd", "mp2", "cisd", "mbpt2", "mbpt3"):
        if m in tce_block_lower:
            method_found = m
            break
    if not tce_block:
        _err("tce_block_missing", "No 'tce...end' block found in the input file.")
    elif not method_found:
        _err("tce_no_method",
             "No recognized TCE method keyword found in the tce block. Expected: ccsd, mp2, ccsd(t), etc.")

    # --- Symmetry check ---
    _TCE_ABELIAN = {"c1", "ci", "cs", "c2", "c2v", "c2h", "d2", "d2h"}
    _sym_m = _re.search(r"\bsymmetry\s+(\S+)\b", geo_block, _re.IGNORECASE)
    _sym_group = _sym_m.group(1).lower() if _sym_m else None
    if _sym_group not in _TCE_ABELIAN:
        if _sym_group is None:
            _err("tce_missing_symmetry_c1",
                 "The geometry block is missing a symmetry directive. "
                 "NWChem TCE requires Abelian symmetry — add 'symmetry c1' (or d2h, c2v, etc.) "
                 "inside the geometry...end block.")
        else:
            _err("tce_missing_symmetry_c1",
                 f"Symmetry '{_sym_group}' is non-Abelian and not supported by NWChem TCE. "
                 f"Use one of: {', '.join(sorted(_TCE_ABELIAN))}.")

    # --- SCF reference check ---
    scf_ref = None
    for ref in ("rohf", "rhf", "uhf"):
        if _re.search(rf"\b{ref}\b", scf_block_lower):
            scf_ref = ref
            break
    if open_shell and scf_ref and scf_ref != "rohf":
        _err("tce_wrong_scf_reference",
             f"Open-shell system (mult={multiplicity}) requires ROHF reference, "
             f"but '{scf_ref}' was found in the SCF block. "
             "Use scf_type='rohf' and nopen=multiplicity-1.")
    elif open_shell and not scf_ref:
        _warn("tce_scf_reference_undetected",
              f"Open-shell system (mult={multiplicity}) but no SCF reference keyword (rohf/rhf/uhf) "
              "was found. Ensure 'rohf' is present in the scf block.")
    if scf_ref == "uhf" and not open_shell:
        _warn("tce_uhf_closed_shell",
              "UHF reference for closed-shell system — RHF is preferred as the TCE reference.")

    # --- Vectors file check ---
    vectors_match = _re.search(r"\bvectors\s+input\s+(\S+)", tce_block, _re.IGNORECASE)
    vectors_path_str: str | None = None
    if vectors_match:
        vectors_path_str = vectors_match.group(1)
        candidate = Path(vectors_path_str)
        if not candidate.is_absolute():
            candidate = Path(tce_input_path).parent / vectors_path_str
        if not candidate.exists():
            _err("tce_vectors_file_missing",
                 f"Vectors file '{vectors_path_str}' does not exist at '{candidate}'. "
                 "Run the SCF job first and verify the path.")
    else:
        _warn("tce_no_vectors_input",
              "No 'vectors input ...' directive found in the tce block. NWChem will use default vectors — "
              "results may be wrong for open-shell or orbital-reordered systems.")

    # --- Freeze count check ---
    freeze_match = _re.search(r"\bfreeze\s+(\d+)\b", tce_block, _re.IGNORECASE)
    actual_freeze: int | None = None
    if freeze_match:
        actual_freeze = int(freeze_match.group(1))
        if elements:
            try:
                suggestion = _suggest_freeze(elements, charge=charge, multiplicity=multiplicity or 1)
                suggested = suggestion.get("freeze_count", 0)
                n_elec = suggestion.get("n_electrons") or 0
                min_correlated = 2
                max_freeze = max(0, n_elec // 2 - min_correlated)
                if max_freeze > 0 and actual_freeze > max_freeze:
                    _err("tce_freeze_too_large",
                         f"Freeze count {actual_freeze} leaves fewer than {min_correlated} correlated electrons "
                         f"({n_elec} total electrons). This is likely wrong.")
                elif actual_freeze < suggested - 2:
                    _warn("tce_freeze_too_small",
                          f"Freeze count {actual_freeze} is less than suggested {suggested}. "
                          "Under-freezing wastes compute and may affect energetics.")
                elif actual_freeze > suggested + 2:
                    _warn("tce_freeze_larger_than_suggested",
                          f"Freeze count {actual_freeze} exceeds suggested {suggested}. "
                          "Verify this is intentional and not over-freezing valence electrons.")
            except Exception:
                pass
    elif "freeze atomic" in tce_block_lower:
        _err("tce_freeze_atomic_forbidden",
             "'freeze atomic' is forbidden in TCE inputs — always specify an explicit count "
             "(e.g. 'freeze 10'). Use suggest_nwchem_tce_freeze to compute the correct value.")
    else:
        _warn("tce_no_freeze",
              "No 'freeze N' directive found. All electrons will be correlated — this is very expensive "
              "and almost always wrong. Use suggest_nwchem_tce_freeze to compute the correct freeze count.")

    # --- SCF output existence check ---
    if scf_output_path and not Path(scf_output_path).exists():
        _warn("tce_scf_output_missing",
              f"SCF output file '{scf_output_path}' does not exist yet. Run the SCF job first.")

    # --- Build summary ---
    n_errors = sum(1 for i in issues if i["level"] == "error")
    n_warnings = sum(1 for i in issues if i["level"] == "warning")
    status = "errors" if n_errors else ("warnings" if n_warnings else "ok")

    summary_lines = [f"TCE setup validation: {status}"]
    if n_errors:
        summary_lines.append(f"  {n_errors} error(s) must be fixed before submitting.")
    if n_warnings:
        summary_lines.append(f"  {n_warnings} warning(s) worth reviewing.")
    if status == "ok":
        summary_lines.append("  No issues found. Input looks correct for TCE.")
    for iss in issues:
        summary_lines.append(f"  [{iss['level'].upper()}] {iss['code']}: {iss['message']}")

    return {
        "tce_input_file": tce_input_path,
        "scf_output_file": scf_output_path,
        "status": status,
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "issues": issues,
        "detected": {
            "method": method_found,
            "elements": elements,
            "charge": charge,
            "multiplicity": multiplicity,
            "scf_reference": scf_ref,
            "has_freeze_directive": actual_freeze is not None,
            "freeze_count": actual_freeze,
            "has_vectors_input": bool(vectors_match),
            "vectors_file": vectors_path_str,
        },
        "summary": "\n".join(summary_lines),
    }


# ---------------------------------------------------------------------------
# Atomic ground-state multiplicities (neutral, lowest term)
# ---------------------------------------------------------------------------

_ATOM_GROUND_MULT: dict[str, int] = {
    "H": 2, "He": 1, "Li": 2, "Be": 1, "B": 2, "C": 3, "N": 4,
    "O": 3, "F": 2, "Ne": 1, "Na": 2, "Mg": 1, "Al": 2, "Si": 3,
    "P": 4, "S": 3, "Cl": 2, "Ar": 1, "K": 2, "Ca": 1,
    # 3d transition metals
    "Sc": 2, "Ti": 3, "V": 4, "Cr": 7, "Mn": 6, "Fe": 5,
    "Co": 4, "Ni": 3, "Cu": 2, "Zn": 1,
    # p-block period 4
    "Ga": 2, "Ge": 3, "As": 4, "Se": 3, "Br": 2, "Kr": 1,
    "Rb": 2, "Sr": 1,
    # 4d transition metals
    "Y": 2, "Zr": 3, "Nb": 6, "Mo": 7, "Tc": 6, "Ru": 5,
    "Rh": 4, "Pd": 1, "Ag": 2, "Cd": 1,
    # p-block period 5
    "In": 2, "Sn": 3, "Sb": 4, "Te": 3, "I": 2, "Xe": 1,
    "Cs": 2, "Ba": 1,
    # 5d transition metals (simplified; use ROHF + MCSCF for production)
    "La": 2, "Hf": 3, "Ta": 4, "W": 5, "Re": 6, "Os": 5,
    "Ir": 4, "Pt": 3, "Au": 2, "Hg": 1,
    # p-block period 6
    "Tl": 2, "Pb": 3, "Bi": 4, "Po": 3, "At": 2, "Rn": 1,
}


def draft_nwchem_atom_input(
    element: str,
    basis: str,
    method: str = "scf",
    charge: int = 0,
    multiplicity: int | None = None,
    xc_functional: str = "m06",
    basis_assignments: dict[str, str] | None = None,
    ecp_assignments: dict[str, str] | None = None,
    memory: str | None = None,
    start_name: str | None = None,
    output_dir: str | None = None,
    write_file: bool = False,
    basis_library: str | None = None,
) -> dict[str, Any]:
    """Generate a NWChem input for a single atom (for atomization energies, IPs, etc.).

    Automatically looks up the neutral ground-state multiplicity for common elements.
    For ions, provide ``multiplicity`` explicitly or it will be estimated from electron
    count parity (with a warning).

    Parameters
    ----------
    element:
        Element symbol, e.g. ``"Fe"``, ``"O"``.
    basis:
        Basis set name (resolved from the local library), e.g. ``"6-31gs"``.
    method:
        NWChem module: ``"scf"``, ``"dft"``, or ``"mp2"``.
    charge:
        Total charge.  0 for neutral atom.
    multiplicity:
        Spin multiplicity (2S+1).  If None, looked up from the ground-state table;
        for ions, estimated from electron-count parity.
    xc_functional:
        XC functional used when ``method="dft"``.
    basis_assignments / ecp_assignments:
        Override basis/ECP per element (passed to render_nwchem_basis_setup).
    memory:
        NWChem memory directive string, e.g. ``"total 2000 mb"``.
    start_name:
        NWChem start name.  Defaults to ``"{element}_atom"``.
    output_dir, write_file:
        Where to write the file if ``write_file=True``.
    basis_library:
        Path to the basis library.  Auto-detected if None.
    """
    from pathlib import Path as _Path

    sym = element[0].upper() + element[1:].lower()

    # --- Determine multiplicity ---
    mult_source = "provided"
    if multiplicity is None:
        neutral_mult = _ATOM_GROUND_MULT.get(sym)
        if neutral_mult is None:
            raise ValueError(
                f"Unknown element '{sym}' or no ground-state table entry; "
                "provide multiplicity explicitly."
            )
        if charge == 0:
            multiplicity = neutral_mult
            mult_source = "ground_state_table"
        else:
            from ._api_utils import ELEMENT_TO_Z
            z = ELEMENT_TO_Z.get(sym, 0)
            n_electrons = z - charge
            if n_electrons <= 0:
                raise ValueError(
                    f"Element {sym} (Z={z}) with charge {charge:+d} has {n_electrons} electrons."
                )
            # Parity rule: even electrons → even nopen, odd → odd nopen
            ion_nopen = n_electrons % 2
            multiplicity = ion_nopen + 1
            mult_source = "estimated_from_parity"

    nopen = multiplicity - 1

    # --- Resolve basis library path ---
    if basis_library is not None:
        lib_path = basis_library
    else:
        try:
            from importlib.resources import files as _pkg_files
            lib_path = str(_pkg_files("chemtools").joinpath("data/nwchem/basis_library"))
        except Exception:
            lib_path = None

    # Build basis block
    try:
        from .basis import render_nwchem_basis_block as _render_basis
        basis_info = _render_basis(
            basis_name=basis,
            elements=[sym],
            library_path=lib_path,
        )
        basis_block = basis_info["text"]
        ecp_block = basis_info.get("ecp_text") or None
    except Exception as exc:
        raise ValueError(f"Could not render basis '{basis}' for element '{sym}': {exc}") from exc

    # --- Build SCF/DFT block ---
    resolved_start = start_name or f"{sym.lower()}_atom"
    method_norm = method.strip().lower()

    if method_norm == "dft":
        if nopen > 0:
            scf_block = f"dft\n  odft\n  mult {multiplicity}\n  xc {xc_functional}\n  thresh 1e-8\n  maxiter 200\nend"
        else:
            scf_block = f"dft\n  xc {xc_functional}\n  thresh 1e-8\n  maxiter 200\nend"
        task_line = "task dft energy"
    elif method_norm in ("scf", "rohf", "rhf", "uhf"):
        if nopen > 0:
            scf_block = f"scf\n  rohf\n  nopen {nopen}\n  thresh 1e-8\n  maxiter 200\nend"
        else:
            scf_block = "scf\n  rhf\n  thresh 1e-8\n  maxiter 200\nend"
        task_line = "task scf energy"
    elif method_norm == "mp2":
        if nopen > 0:
            scf_block = f"scf\n  rohf\n  nopen {nopen}\n  thresh 1e-8\n  maxiter 200\nend"
        else:
            scf_block = "scf\n  rhf\n  thresh 1e-8\n  maxiter 200\nend"
        task_line = "task mp2 energy"
    else:
        raise ValueError(f"Unsupported method '{method}' for draft_nwchem_atom_input. Use scf, dft, or mp2.")

    # --- Geometry block (single atom at origin, always c1) ---
    geo_block = f"geometry units angstroms\n  symmetry c1\n  {sym}  0.00000  0.00000  0.00000\nend"

    # --- Assemble sections ---
    sections: list[str] = [f"start {resolved_start}", "echo"]
    if memory:
        sections.append(f"memory {memory}")
    sections.append(geo_block)
    if charge != 0:
        sections.append(f"charge {charge}")
    sections.append(basis_block)
    if ecp_block:
        sections.append(ecp_block)
    sections.append(scf_block)
    sections.append(task_line)

    input_text = "\n\n".join(sections).rstrip() + "\n"

    # --- Write file ---
    out_dir = _Path(output_dir) if output_dir else _Path(".")
    out_path = out_dir / f"{resolved_start}.nw"
    written_file: str | None = None
    if write_file:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(input_text, encoding="utf-8")
        written_file = str(out_path.resolve())

    warnings: list[str] = []
    if mult_source == "estimated_from_parity":
        warnings.append(
            f"Multiplicity {multiplicity} for {sym}{charge:+d} is estimated from electron-count parity only. "
            "Verify the correct ground state for this ion before running."
        )

    return {
        "input_text": input_text,
        "written_file": written_file,
        "planned_output_file": str(out_path),
        "element": sym,
        "charge": charge,
        "multiplicity": multiplicity,
        "nopen": nopen,
        "multiplicity_source": mult_source,
        "method": method_norm,
        "basis": basis,
        "start_name": resolved_start,
        "warnings": warnings,
    }


def draft_nwchem_tce_restart_input(
    tce_output_file: str,
    tce_input_file: str | None = None,
    max_iterations: int = 200,
    thresh: float = 1e-5,
    copy_amplitudes: bool = True,
    output_dir: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    """Generate a NWChem TCE restart input from a stalled or incomplete CCSD/MP2 run.

    This function:

    1. Parses the previous TCE output to determine method, freeze count, and
       convergence state (last iteration + residual).
    2. Locates saved amplitude files (``{stem}.t1amp.*`` or ``{stem}.t1_copy.*``).
    3. Optionally copies them to ``{start_name}.t1`` / ``{start_name}.t2`` so
       NWChem can read them via ``set tce:read_ta .true.``.
    4. Builds a ``restart`` input with ``set tce:read_ta .true.``,
       ``set tce:save_t T T``, the requested ``maxiter``, and ``thresh``.

    Parameters
    ----------
    tce_output_file:
        Path to the incomplete TCE output (``.out``) file.
    tce_input_file:
        Path to the previous TCE input (``.nw``) file.  Used to extract
        geometry, basis, ECP, and SCF blocks.  Auto-inferred from output stem
        if None.
    max_iterations:
        Maximum CCSD iterations for the restart run (default 200).
    thresh:
        CCSD residual convergence threshold (default 1e-5, 10× looser than
        the 1e-6 NWChem default — suitable for slowly-converging systems).
    copy_amplitudes:
        If True, copy the found amplitude files to the correct restart names
        (``{start_name}.t1`` / ``{start_name}.t2``).
    output_dir, write_file:
        Where to write the restart input file.
    """
    import glob as _glob
    import shutil as _shutil
    from pathlib import Path as _Path
    from .common import read_text as _read_text
    from .nwchem_tce import parse_tce_output as _parse_tce

    out_path = _Path(tce_output_file).resolve()
    out_dir = out_path.parent
    stem = out_path.stem

    # --- Parse previous TCE output ---
    contents = _read_text(tce_output_file)
    tce_result = _parse_tce(tce_output_file, contents)

    method = tce_result.get("method") or "CCSD"
    method_kw = method.lower().replace("(", "").replace(")", "")
    tce_method_kw = {"mp2": "mp2", "ccsd": "ccsd", "ccsdt": "ccsd(t)"}.get(method_kw, "ccsd")

    # Extract last iteration count and residual from output
    import re as _re
    iter_re = _re.compile(
        r"^\s*\d+\s+[-\d.Ee+]+\s+([\d.Ee+\-]+)\s*$", _re.IGNORECASE
    )
    last_iter: int | None = None
    last_residual: float | None = None
    iter_count = 0
    for line in contents.splitlines():
        m = iter_re.match(line)
        if m:
            iter_count += 1
            try:
                last_residual = float(m.group(1))
                last_iter = iter_count
            except ValueError:
                pass

    # Extract frozen core count from TCE section
    freeze_count: int | None = None
    for sec in tce_result.get("tce_sections", []):
        if sec.get("frozen_cores") is not None:
            freeze_count = sec["frozen_cores"]

    # --- Determine start_name from output stem ---
    # Output file might be named {start_name}.out or {start_name}_restart.out etc.
    start_name = stem
    # Try to read from the input file if available
    inferred_input = tce_input_file
    if inferred_input is None:
        for candidate in [
            out_dir / f"{stem}.nw",
            out_dir / f"{stem.replace('_restart', '')}.nw",
            out_dir / f"{stem.replace('_ccsd', '_ccsd_tce_ccsd')}.nw",
        ]:
            if candidate.exists():
                inferred_input = str(candidate)
                break

    if inferred_input and _Path(inferred_input).exists():
        try:
            from .nwchem_input import inspect_nwchem_input as _inspect
            summary = _inspect(inferred_input)
            blocks = summary.get("start_blocks", [])
            if blocks and blocks[0].get("start_name"):
                start_name = blocks[0]["start_name"]
            elif summary.get("start_present"):
                # Read the start line directly
                for line in _read_text(inferred_input).splitlines():
                    m2 = _re.match(r"^\s*(?:start|restart)\s+(\S+)", line, _re.IGNORECASE)
                    if m2:
                        start_name = m2.group(1)
                        break
        except Exception:
            pass

    # --- Locate amplitude files ---
    amp_patterns = [
        (f"{start_name}.t1amp.*", f"{start_name}.t2amp.*"),
        (f"{stem}.t1amp.*", f"{stem}.t2amp.*"),
        (f"{start_name}.t1_copy.*", f"{start_name}.t2_copy.*"),
        (f"{stem}.t1_copy.*", f"{stem}.t2_copy.*"),
    ]
    t1_src: str | None = None
    t2_src: str | None = None
    for t1_pat, t2_pat in amp_patterns:
        t1_candidates = sorted(_glob.glob(str(out_dir / t1_pat)))
        t2_candidates = sorted(_glob.glob(str(out_dir / t2_pat)))
        if t1_candidates:
            t1_src = t1_candidates[-1]  # take latest
        if t2_candidates:
            t2_src = t2_candidates[-1]
        if t1_src:
            break

    # --- Copy amplitude files if requested ---
    copied_files: list[str] = []
    t1_dest = str(out_dir / f"{start_name}.t1")
    t2_dest = str(out_dir / f"{start_name}.t2")
    copy_errors: list[str] = []

    if copy_amplitudes:
        if t1_src:
            try:
                _shutil.copy2(t1_src, t1_dest)
                copied_files.append(f"{t1_src} → {t1_dest}")
            except Exception as exc:
                copy_errors.append(f"Could not copy T1 file: {exc}")
        else:
            copy_errors.append(
                f"No T1 amplitude file found (tried patterns: "
                f"{', '.join(p[0] for p in amp_patterns)})."
            )
        if t2_src:
            try:
                _shutil.copy2(t2_src, t2_dest)
                copied_files.append(f"{t2_src} → {t2_dest}")
            except Exception as exc:
                copy_errors.append(f"Could not copy T2 file: {exc}")
        else:
            copy_errors.append(
                f"No T2 amplitude file found."
            )

    can_read_amplitudes = bool(
        _Path(t1_dest).exists() and _Path(t2_dest).exists()
    )

    # --- Extract geometry, basis, ECP, SCF blocks from previous input ---
    geo_section: str | None = None
    basis_section: str | None = None
    ecp_section: str | None = None
    scf_section: str | None = None
    charge_line: str | None = None
    movecs_line: str | None = None

    if inferred_input and _Path(inferred_input).exists():
        try:
            from .nwchem_input import (
                extract_nwchem_geometry_block,
                render_nwchem_geometry_block,
                inspect_all_nwchem_basis_blocks,
                inspect_nwchem_ecp_block,
            )
            geo = extract_nwchem_geometry_block(inferred_input)
            # Keep existing symmetry (already set correctly)
            geo_section = render_nwchem_geometry_block(
                geo["header_line"], geo["atoms"], directives=geo.get("directives", [])
            )
        except Exception:
            pass

        try:
            basis_blocks = inspect_all_nwchem_basis_blocks(inferred_input)
            if basis_blocks:
                raw = basis_blocks[-1]
                basis_section = raw["header_line"] + "\n" + "\n".join(raw["body_lines"]) + "\nend"
        except Exception:
            pass

        try:
            ecp_info = inspect_nwchem_ecp_block(inferred_input)
            if ecp_info.get("body_lines"):
                ecp_section = ecp_info["header_line"] + "\n" + "\n".join(ecp_info["body_lines"]) + "\nend"
        except Exception:
            pass

        try:
            from .nwchem_input import extract_nwchem_module_block
            # Try to find the last SCF block before the TCE block
            scf_blk = extract_nwchem_module_block(inferred_input, module="scf", block_index=-1)
            # Strip existing vectors lines; we'll use the restart movecs
            scf_lines = [l for l in scf_blk["body_lines"] if "vectors" not in l.lower()]
            movecs_file = f"{start_name}.movecs"
            restart_movecs = str(out_dir / movecs_file)
            if _Path(restart_movecs).exists():
                scf_lines.append(f"  vectors input {movecs_file} output {movecs_file}")
                movecs_line = movecs_file
            scf_section = scf_blk["header_line"] + "\n" + "\n".join(scf_lines) + "\nend"
        except Exception:
            pass

        try:
            from .nwchem_input import inspect_nwchem_input as _inspect
            summary = _inspect(inferred_input)
            charge = summary.get("charge")
            if charge:
                charge_line = f"charge {charge}"
        except Exception:
            pass

    # --- Build TCE block ---
    tce_lines = [f"  {tce_method_kw}"]
    if freeze_count is not None:
        tce_lines.append(f"  freeze {freeze_count}")
    tce_lines.append(f"  maxiter {max_iterations}")
    tce_lines.append(f"  thresh {thresh:.2e}")
    tce_block = "tce\n" + "\n".join(tce_lines) + "\nend"

    read_ta_line = "set tce:read_ta .true." if can_read_amplitudes else "# set tce:read_ta .true.  # (enable once .t1/.t2 files are in place)"
    save_t_line = "set tce:save_t T T"

    # --- Assemble input ---
    sections_list: list[str] = [f"restart {start_name}", "echo"]
    if geo_section:
        sections_list.append(geo_section)
    if charge_line:
        sections_list.append(charge_line)
    if basis_section:
        sections_list.append(basis_section)
    if ecp_section:
        sections_list.append(ecp_section)
    if scf_section:
        sections_list.append(scf_section)
        sections_list.append("task scf energy")
    sections_list.append(read_ta_line)
    sections_list.append(save_t_line)
    sections_list.append(tce_block)
    sections_list.append("task tce energy")

    input_text = "\n\n".join(sections_list).rstrip() + "\n"

    # --- Write file ---
    resolved_outdir = _Path(output_dir) if output_dir else out_dir
    out_nw = resolved_outdir / f"{start_name}_restart.nw"
    written_file: str | None = None
    if write_file:
        resolved_outdir.mkdir(parents=True, exist_ok=True)
        out_nw.write_text(input_text, encoding="utf-8")
        written_file = str(out_nw.resolve())

    return {
        "input_text": input_text,
        "written_file": written_file,
        "planned_output_file": str(out_nw),
        "restart_start_name": start_name,
        "method": tce_method_kw,
        "freeze_count": freeze_count,
        "max_iterations": max_iterations,
        "thresh": thresh,
        "previous_tce_last_iter": last_iter,
        "previous_tce_last_residual": last_residual,
        "amplitude_files_found": {
            "t1": t1_src,
            "t2": t2_src,
        },
        "amplitude_files_copied": copied_files,
        "can_read_amplitudes": can_read_amplitudes,
        "copy_errors": copy_errors,
        "movecs_file": movecs_line,
        "warnings": copy_errors,
    }

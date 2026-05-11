"""NWChem optimization follow-up drafter.

Two entry points:

  * _select_best_optimization_frame   Helper that picks the best frame
                                      to restart from when an opt run
                                      did not fully converge. Used by
                                      this module's draft_* function, by
                                      extract_nwchem_geometry, and by
                                      api_strategy. Exported under its
                                      original name so existing imports
                                      from any of those locations
                                      continue to work.
  * draft_nwchem_optimization_followup_input
                                      Render a follow-up input that
                                      retries the optimization (with
                                      adjusted strategy) starting from
                                      the best available frame in a
                                      previous run's trajectory.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

from chemtools.core.common import read_text, detect_program, make_metadata
from chemtools.programs.nwchem.parse.input import (
    inspect_nwchem_input,
    extract_nwchem_geometry_block,
    render_nwchem_geometry_block,
    replace_nwchem_geometry_block,
)
from chemtools.programs.nwchem.parse.freq import parse_trajectory as _parse_trajectory_raw
from chemtools.programs.nwchem.strategy.diagnose import diagnose_nwchem_output
from chemtools.programs.nwchem.input._utils import (
    _select_optimization_follow_up_strategy,
    _build_optimization_follow_up_plan,
    _default_optimization_follow_up_base_name,
    _default_optimization_follow_up_title,
    _replace_tasks_in_text,
    _replace_or_insert_keyword_line,
    _remove_keyword_blocks,
    _build_vectors_swap_file_plan,
    _build_simple_input_file_plan,
    _ensure_module_vectors_output_in_text,
    _write_text_file,
)


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

    trajectory = _parse_trajectory_raw(output_path, contents, include_positions=True)
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

__all__ = [
    "_select_best_optimization_frame",
    "draft_nwchem_optimization_followup_input",
]

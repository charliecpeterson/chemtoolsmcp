"""Assemble normalized geometry trajectories from PWSCF relaxation output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.programs.qe._coordinates import (
    bfgs_converged,
    geometry_record,
    initial_alat_bohr,
    initial_runtime_geometry,
    is_relaxation,
    job_done,
    normalize_atoms,
    normalize_cell,
    parse_cell_card,
    parse_positions_card,
)


def parse_pw_trajectory(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_pw_trajectory_text(text)


def parse_pw_trajectory_text(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    if not is_relaxation(lines):
        return {
            "status": "unavailable",
            "reason": "The PWSCF output is not a geometry relaxation.",
        }

    initial = initial_runtime_geometry(lines)
    if initial.get("status") != "available":
        return {
            "status": "unavailable",
            "reason": (
                "The relaxation output does not contain a complete initial "
                "runtime geometry."
            ),
        }

    frames = [_trajectory_frame(initial, index=0, role="initial")]
    warnings: list[dict[str, Any]] = []
    alat_bohr = initial_alat_bohr(lines)
    current_cell = initial["cell"]["vectors_angstrom"]
    current_cell_line = initial["source"]["cell_line"]
    atom_count = initial["atom_count"]
    in_final_coordinates = False

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "Begin final coordinates":
            in_final_coordinates = True
            continue
        if stripped == "End final coordinates":
            in_final_coordinates = False
            continue
        if stripped.upper().startswith("CELL_PARAMETERS"):
            cell_record = parse_cell_card(lines, index)
            normalized = normalize_cell(
                cell_record,
                fallback=None,
                fallback_alat_bohr=alat_bohr,
            )
            if normalized is None:
                current_cell = None
                warnings.append({
                    "line": index + 1,
                    "message": "CELL_PARAMETERS could not be normalized.",
                })
            else:
                current_cell = normalized
                current_cell_line = index + 1
            continue
        if not stripped.upper().startswith("ATOMIC_POSITIONS"):
            continue

        positions = parse_positions_card(lines, index, atom_count)
        atoms = normalize_atoms(
            positions,
            cell=current_cell,
            fallback_alat_bohr=alat_bohr,
        )
        if current_cell is None or len(atoms) != atom_count:
            warnings.append({
                "line": index + 1,
                "message": "ATOMIC_POSITIONS could not be normalized.",
            })
            continue
        geometry = geometry_record(
            atoms,
            current_cell,
            role=(
                "converged_final"
                if in_final_coordinates and bfgs_converged(lines)
                else "optimization_step"
            ),
            position_line=index + 1,
            cell_line=current_cell_line,
        )
        candidate = _trajectory_frame(
            geometry,
            index=len(frames),
            role=geometry["role"],
        )
        if _same_frame_geometry(frames[-1], candidate):
            if candidate["role"] == "converged_final":
                candidate["index"] = frames[-1]["index"]
                candidate["step"] = frames[-1]["step"]
                frames[-1] = candidate
            continue
        frames.append(candidate)

    converged = bfgs_converged(lines)
    if converged:
        optimization_status = "converged"
    elif job_done(lines):
        optimization_status = "not_converged"
    else:
        optimization_status = "incomplete"
    if frames[-1]["role"] != "converged_final":
        frames[-1]["role"] = "last_attempted"
    frame_count = len(frames)
    if frames[-1]["role"] == "converged_final":
        geometry_source = (
            f"step {frame_count} of {frame_count}, the converged geometry"
        )
    elif optimization_status == "not_converged":
        geometry_source = (
            f"step {frame_count} of {frame_count}; "
            "the run stopped without converging"
        )
    else:
        geometry_source = (
            f"step {frame_count} of {frame_count}; "
            "the output ended before convergence"
        )
    return {
        "status": "available",
        "optimization_status": optimization_status,
        "geometry_role": frames[-1]["role"],
        "geometry_source": geometry_source,
        "units": "angstrom",
        "frame_count": frame_count,
        "frames": frames,
        "energy_alignment": {
            "status": "not_assigned",
            "reason": (
                "PWSCF may print a separate final SCF energy at the relaxed "
                "geometry, so SCF records are not assigned to frames by index."
            ),
        },
        "warnings": warnings,
    }


def _trajectory_frame(
    geometry: dict[str, Any],
    *,
    index: int,
    role: str,
) -> dict[str, Any]:
    return {
        "index": index,
        "step": index,
        "role": role,
        "units": geometry["units"],
        "atoms": geometry["atoms"],
        "atom_count": geometry["atom_count"],
        "elements": geometry["elements"],
        "cell": geometry["cell"],
        "source": geometry["source"],
    }


def _same_frame_geometry(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    if left["elements"] != right["elements"]:
        return False
    left_values = [
        coordinate
        for atom in left["atoms"]
        for coordinate in (atom["x"], atom["y"], atom["z"])
    ]
    right_values = [
        coordinate
        for atom in right["atoms"]
        for coordinate in (atom["x"], atom["y"], atom["z"])
    ]
    left_values.extend(
        component
        for vector in left["cell"]["vectors_angstrom"]
        for component in vector
    )
    right_values.extend(
        component
        for vector in right["cell"]["vectors_angstrom"]
        for component in vector
    )
    return len(left_values) == len(right_values) and all(
        abs(left_value - right_value) <= 1e-8
        for left_value, right_value in zip(left_values, right_values)
    )


__all__ = ["parse_pw_trajectory", "parse_pw_trajectory_text"]

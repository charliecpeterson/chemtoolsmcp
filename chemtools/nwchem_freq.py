from __future__ import annotations

from collections import defaultdict, deque
import math
import re
from typing import Any

from .common import make_metadata, parse_float_after_delimiter, parse_scientific_float, split_tokens
from .nwchem_tasks import detect_energy_token, detect_method_token, detect_basis_token

METHOD_PATTERNS: list[tuple[int, str, tuple[str, ...]]] = [
    (5, "CCSD(T)", ("ccsd(t)",)),
    (4, "CCSD", ("ccsd total energy", " ccsd ")),
    (3, "MP2", ("total mp2 energy", " mp2 ")),
    (3, "MCSCF", ("total mcscf energy", " mcscf ")),
    (2, "DFT", ("total dft energy", " dft ", "b3lyp", "pbe0", "pbe ")),
    (1, "SCF", ("total scf energy", " scf ", " rhf", " uhf", " rohf")),
    (0, "TCE", ("tce",)),
]

BOHR_TO_ANGSTROM = 0.529177210903
FREQUENCY_ROW_RE = re.compile(
    r"^\s*(\d+)\s+([-\d.DEde+]+)\s+\|\|\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s*$"
)
THERMO_AU_RE = re.compile(r"\(\s*([-\d.DEde+]+)\s+au\s*\)", re.IGNORECASE)
OPT_PROGRESS_RE = re.compile(
    r"^\@\s+(\d+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.]+)\s*$"
)
POPULATION_HEADER_RE = re.compile(
    r"^\s*(Total Density|Spin Density)\s*-\s*(Mulliken|Lowdin|L[öo]wdin)\s+Population Analysis\s*$",
    re.IGNORECASE,
)
MCSCF_ENERGY_RE = re.compile(r">>>\|\s*MCSCF energy:\s*([-\d.DEde+]+)")
MCSCF_TOTAL_ENERGY_RE = re.compile(r"Total MCSCF energy\s*=\s*([-\d.DEde+]+)", re.IGNORECASE)
MCSCF_LEVELSHIFT_RE = re.compile(r"Increase level shift to\s+([-\d.DEde+]+)", re.IGNORECASE)
MCSCF_RESIDUE_RE = re.compile(
    r"Precondition failed to converge:Residue:\s*current=\s*([+-]?\d+(?:\.\d+)?(?:[DEde][+-]?\d+)?)\s*required=\s*([+-]?\d+(?:\.\d+)?(?:[DEde][+-]?\d+)?)",
    re.IGNORECASE,
)
MCSCF_NEGATIVE_CURVATURE_RE = re.compile(r"Negative curvature:\s*hessian=\s*([-\d.DEde+]+)", re.IGNORECASE)
MCSCF_SETTING_RE = re.compile(
    r"^\s*(active|actelec|multiplicity|state|hessian|maxiter|thresh|tol2e|level|symmetry)\s+(.+?)\s*$",
    re.IGNORECASE,
)
MCSCF_SUMMARY_VALUE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z\s]+?):\s+(.+?)\s*$")
TRANSITION_METALS = {
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
}
COVALENT_RADII = {
    "H": 0.31,
    "B": 0.85,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Mo": 1.54,
    "W": 1.62,
    "Re": 1.51,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "Pt": 1.36,
    "Au": 1.36,
    "U": 1.96,
}




def parse_freq(path: str, contents: str, include_displacements: bool = False) -> dict[str, Any]:
    preferred_kind = "raw"
    regular_modes, projected_modes = _extract_frequency_modes(contents)
    if projected_modes:
        modes = projected_modes
        preferred_kind = "projected"
    else:
        modes = regular_modes

    equilibrium_energy = None
    for line in contents.splitlines():
        token = detect_energy_token(line)
        if token is not None:
            equilibrium_energy = token[1]

    thermochemistry = _parse_thermochemistry(contents)
    near_zero_threshold = 20.0
    significant_imaginary_threshold = -20.0
    near_zero_modes = [mode for mode in modes if abs(mode["frequency_cm1"]) < near_zero_threshold]
    significant_imaginary_modes = [
        mode for mode in modes if mode["frequency_cm1"] <= significant_imaginary_threshold
    ]
    vibrational_modes = [mode for mode in modes if abs(mode["frequency_cm1"]) >= near_zero_threshold]
    if include_displacements:
        labels = _extract_last_geometry_labels(contents)
        displacement_sections = _extract_normal_mode_displacements(contents, labels)
        _attach_mode_displacements(regular_modes, displacement_sections["raw"])
        _attach_mode_displacements(projected_modes, displacement_sections["projected"])
        modes = projected_modes if projected_modes else regular_modes

    return {
        "metadata": make_metadata(path, contents, "nwchem"),
        "equilibrium_energy_hartree": equilibrium_energy,
        "atom_count": None,
        "preferred_kind": preferred_kind,
        "mode_count": len(modes),
        "imaginary_mode_count": sum(1 for mode in modes if mode["is_imaginary"]),
        "significant_imaginary_mode_count": len(significant_imaginary_modes),
        "significant_imaginary_frequencies_cm1": [
            mode["frequency_cm1"] for mode in significant_imaginary_modes
        ],
        "near_zero_mode_count": len(near_zero_modes),
        "near_zero_frequencies_cm1": [mode["frequency_cm1"] for mode in near_zero_modes[:10]],
        "vibrational_mode_count": len(vibrational_modes),
        "lowest_vibrational_frequencies_cm1": [mode["frequency_cm1"] for mode in vibrational_modes[:6]],
        "thermochemistry": thermochemistry,
        "raw_mode_count": len(regular_modes),
        "projected_mode_count": len(projected_modes),
        "raw_modes": regular_modes,
        "projected_modes": projected_modes,
        "modes": modes,
    }


def parse_trajectory(path: str, contents: str, include_positions: bool = False) -> dict[str, Any]:
    lines = contents.splitlines()
    in_geom = False
    in_optimization = False
    optimization_done = False
    optimization_status = "not_detected"
    current_step_header: int | None = None
    current_atoms: list[dict[str, Any]] = []
    geometries: list[tuple[int, list[dict[str, Any]]]] = []
    step_energies: dict[int, float] = {}
    step_headers: list[int] = []
    coordinate_unit = "angstrom"
    thresholds = {
        "gmax": None,
        "grms": None,
        "xmax": None,
        "xrms": None,
        "trust": None,
    }
    step_metrics: dict[int, dict[str, Any]] = {}

    for line in lines:
        if "NWChem Geometry Optimization" in line:
            in_optimization = True
            optimization_status = "in_progress"
        if "Optimization converged" in line:
            optimization_done = True
            optimization_status = "converged"
        elif "Optimization failed" in line:
            optimization_done = True
            optimization_status = "failed"
        if optimization_done and in_optimization and "NORMAL MODE EIGENVECTORS" in line:
            break

        trimmed = line.strip()
        if "maximum gradient threshold" in line and "(gmax)" in line:
            thresholds["gmax"] = parse_float_after_delimiter(line, "=")
        elif "rms gradient threshold" in line and "(grms)" in line:
            thresholds["grms"] = parse_float_after_delimiter(line, "=")
        elif "maximum cartesian step threshold" in line and "(xmax)" in line:
            thresholds["xmax"] = parse_float_after_delimiter(line, "=")
        elif "rms cartesian step threshold" in line and "(xrms)" in line:
            thresholds["xrms"] = parse_float_after_delimiter(line, "=")
        elif "fixed trust radius" in line and "(trust)" in line:
            thresholds["trust"] = parse_float_after_delimiter(line, "=")

        if in_optimization and trimmed.startswith("Step") and not trimmed.startswith("Step       Energy"):
            parts = trimmed.split()
            if len(parts) >= 2:
                try:
                    current_step_header = int(parts[1])
                    step_headers.append(current_step_header)
                except ValueError:
                    pass

        if in_optimization and trimmed.startswith("@") and "----" not in line:
            if match := OPT_PROGRESS_RE.match(line):
                step_num = int(match.group(1))
                metrics = {
                    "step": step_num,
                    "energy_hartree": parse_scientific_float(match.group(2)),
                    "delta_e_hartree": parse_scientific_float(match.group(3)),
                    "gmax": parse_scientific_float(match.group(4)),
                    "grms": parse_scientific_float(match.group(5)),
                    "xrms": parse_scientific_float(match.group(6)),
                    "xmax": parse_scientific_float(match.group(7)),
                    "walltime_seconds": parse_scientific_float(match.group(8)),
                }
                step_metrics[step_num] = metrics
                if metrics["energy_hartree"] is not None:
                    step_energies[step_num] = metrics["energy_hartree"]
            else:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        step_energies[int(parts[1])] = float(parts[2])
                    except ValueError:
                        pass

        if in_optimization and current_step_header is not None and "Output coordinates in angstroms" in line:
            in_geom = True
            coordinate_unit = "angstrom"
            current_atoms = []
            continue
        if in_optimization and current_step_header is not None and "Output coordinates in a.u." in line:
            in_geom = True
            coordinate_unit = "bohr"
            current_atoms = []
            continue

        if not in_geom:
            continue

        if not trimmed or line.startswith("----"):
            continue
        if "No." in line and "Tag" in line and "Charge" in line:
            continue
        if "Atomic Mass" in line:
            in_geom = False
            if current_atoms:
                geometries.append((current_step_header, current_atoms[:]))
            continue

        parts = line.split()
        if len(parts) >= 6:
            try:
                int(parts[0])
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
            except ValueError:
                continue
            current_atoms.append(
                {
                    "label": parts[1],
                    "position_angstrom": _to_angstrom([x, y, z], coordinate_unit),
                }
            )

    unique_steps: set[int] = set()
    frames = []
    for step_num, atoms in sorted(geometries, key=lambda item: item[0]):
        if step_num in unique_steps:
            continue
        unique_steps.add(step_num)
        frames.append(
            {
                "index": len(frames),
                "step": step_num,
                "energy_hartree": step_energies.get(step_num),
                "delta_e_hartree": step_metrics.get(step_num, {}).get("delta_e_hartree"),
                "gmax": step_metrics.get(step_num, {}).get("gmax"),
                "grms": step_metrics.get(step_num, {}).get("grms"),
                "xrms": step_metrics.get(step_num, {}).get("xrms"),
                "xmax": step_metrics.get(step_num, {}).get("xmax"),
                "walltime_seconds": step_metrics.get(step_num, {}).get("walltime_seconds"),
                "atom_count": len(atoms),
                "labels": [atom["label"] for atom in atoms],
                "positions_angstrom": [atom["position_angstrom"] for atom in atoms]
                if include_positions
                else None,
            }
        )

    step_sequence = sorted(set(step_headers))
    energies = [frame["energy_hartree"] for frame in frames if frame["energy_hartree"] is not None]
    final_energy = energies[-1] if energies else None
    initial_energy = energies[0] if energies else None
    if in_optimization and optimization_status == "in_progress":
        optimization_status = "incomplete"
    restart_recommended = bool(optimization_status in {"failed", "incomplete"} and frames)
    metric_sequence = [step_metrics[step] for step in step_sequence if step in step_metrics]
    final_metrics = metric_sequence[-1] if metric_sequence else None
    criteria_met = _evaluate_optimization_criteria(final_metrics, thresholds) if final_metrics else None
    unmet_criteria = [
        key
        for key, met in (criteria_met or {}).items()
        if key != "all_met" and met is False
    ]

    return {
        "metadata": make_metadata(path, contents, "nwchem"),
        "frame_count": len(frames),
        "atom_count": frames[0]["atom_count"] if frames else 0,
        "energies_hartree": [frame["energy_hartree"] for frame in frames],
        "optimization_status": optimization_status,
        "step_count": len(step_sequence) if step_sequence else len(frames),
        "steps": step_sequence,
        "first_step": step_sequence[0] if step_sequence else None,
        "last_step": step_sequence[-1] if step_sequence else None,
        "final_energy_hartree": final_energy,
        "energy_drop_hartree": (final_energy - initial_energy) if final_energy is not None and initial_energy is not None else None,
        "thresholds": thresholds,
        "step_metrics": metric_sequence,
        "final_metrics": final_metrics,
        "criteria_met": criteria_met,
        "unmet_criteria": unmet_criteria,
        "restart_recommended": restart_recommended,
        "frames": frames,
        "trajectory_extras": {
            "calculation_type": "nwchem_optimization",
            "optimization_status": optimization_status,
            "coordinate_unit_normalized": "angstrom",
        }
        if frames
        else None,
    }


def analyze_imaginary_modes(
    path: str,
    contents: str,
    significant_threshold_cm1: float = 20.0,
    top_atoms: int = 4,
    detail: str = "compact",
) -> dict[str, Any]:
    frequency = parse_freq(path, contents, include_displacements=True)
    geometry = _extract_last_geometry_atoms(contents)
    atoms = geometry["atoms"]
    bonds, adjacency = _infer_bonds(atoms)
    stability_assessment = _assess_frequency_stability(frequency)
    modes = frequency["modes"]
    significant = [
        mode
        for mode in modes
        if mode["frequency_cm1"] <= -abs(significant_threshold_cm1)
    ]

    analyses = [
        _analyze_single_mode(
            mode,
            top_atoms=top_atoms,
            geometry_atoms=atoms,
            bonds=bonds,
            adjacency=adjacency,
        )
        for mode in significant
    ]
    if detail == "compact":
        for analysis in analyses:
            analysis.pop("displacements_cartesian", None)
    return {
        "metadata": make_metadata(path, contents, "nwchem"),
        "preferred_kind": frequency["preferred_kind"],
        "significant_threshold_cm1": significant_threshold_cm1,
        "significant_imaginary_mode_count": len(analyses),
        "detail": detail,
        "geometry": geometry,
        "bond_count": len(bonds),
        "stability_assessment": stability_assessment,
        "modes": analyses,
        "selected_mode": analyses[0] if analyses else None,
    }


def parse_freq_progress(path: str, contents: str) -> dict[str, Any]:
    """Report finite-difference Hessian progress for a freq job.

    Returns displacement counts (done vs total), pace, ETA, fdrst age,
    and estimated additional runs needed at a given walltime limit.
    """
    # Get total atom count from geometry block
    geometry = _extract_last_geometry_atoms(contents)
    n_atoms = geometry["atom_count"] or None
    # Total displacements: ±displacement × 3 coordinates × N atoms
    n_total = 2 * 3 * n_atoms if n_atoms else None

    # Find where the current run started (gen_hess restart)
    restart_matches = list(re.finditer(
        r"\*\*\*\* gen_hess restart \*\*\*\*.*?"
        r"iatom_start\s*=\s*(\d+).*?"
        r"ixyz_start\s*=\s*(\d+)",
        contents, re.DOTALL
    ))
    if restart_matches:
        last_restart = restart_matches[-1]
        iatom_start = int(last_restart.group(1))
        ixyz_start = int(last_restart.group(2))
        # Each atom has 6 displacements (3 coords × 2 ±), ixyz is 1-indexed
        n_done_before_this_run = (iatom_start - 1) * 6 + (ixyz_start - 1) * 2
    else:
        n_done_before_this_run = 0

    # Count gradient evaluations in this run
    gradient_pattern = re.compile(
        r"(?:DFT ENERGY GRADIENTS|NWChem SCF Module Gradient|NWChem DFT Module Gradient)"
    )
    n_gradients_this_run = len(gradient_pattern.findall(contents))

    # Also try counting displacement lines with wall times
    wall_times: list[float] = []
    for m in re.finditer(r"wall:\s+([\d.]+)\s*s", contents):
        wall_times.append(float(m.group(1)))

    # Use gradient count as primary, wall_times as secondary
    n_done_this_run = n_gradients_this_run or len(wall_times)
    n_done_total = n_done_before_this_run + n_done_this_run

    # Pace from wall-time stamps in the output
    sec_per_gradient = None
    eta_hours = None
    runs_needed = None

    # Parse task-level wall times (NWChem prints wall time for each task)
    task_times: list[float] = []
    for m in re.finditer(
        r"(?:Total times|Task\s+times).*?wall:\s+([\d.]+)\s*s",
        contents,
    ):
        task_times.append(float(m.group(1)))

    if len(task_times) >= 2:
        # Compute inter-gradient wall time from consecutive task completions
        diffs = [task_times[i+1] - task_times[i]
                 for i in range(len(task_times) - 1)
                 if task_times[i+1] > task_times[i]]
        if diffs:
            # Use recent 20 for more stable estimate
            recent = diffs[-min(20, len(diffs)):]
            sec_per_gradient = sum(recent) / len(recent)

    if sec_per_gradient and n_total:
        remaining = max(0, n_total - n_done_total)
        eta_seconds = remaining * sec_per_gradient
        eta_hours = round(eta_seconds / 3600, 1)
        # Estimate additional 48h runs needed
        runs_needed = math.ceil(eta_seconds / (48 * 3600)) if eta_seconds > 0 else 0

    # Check fdrst file age
    fdrst_path = re.sub(r"\.(out|nw|log|nwout)$", ".fdrst", path, flags=re.IGNORECASE)
    fdrst_info: dict[str, Any] = {"path": fdrst_path, "exists": False}
    try:
        from pathlib import Path as _P
        fp = _P(fdrst_path)
        if fp.exists():
            from datetime import datetime, timezone as _tz
            stat = fp.stat()
            fdrst_info = {
                "path": fdrst_path,
                "exists": True,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=_tz.utc).isoformat(),
            }
    except Exception:
        pass

    return {
        "n_atoms": n_atoms,
        "n_total_displacements": n_total,
        "n_done_cumulative": n_done_total,
        "n_done_this_run": n_done_this_run,
        "n_done_before_restart": n_done_before_this_run,
        "pct_complete": round(100 * n_done_total / n_total, 1) if n_total and n_total > 0 else None,
        "sec_per_gradient_recent": round(sec_per_gradient, 1) if sec_per_gradient else None,
        "estimated_remaining_hours": eta_hours,
        "runs_needed_at_48h_walltime": runs_needed,
        "fdrst": fdrst_info,
    }


def displace_geometry_along_mode(
    path: str,
    contents: str,
    mode_number: int | None = None,
    amplitude_angstrom: float = 0.15,
    significant_threshold_cm1: float = 20.0,
) -> dict[str, Any]:
    frequency = parse_freq(path, contents, include_displacements=True)
    geometry = _extract_last_geometry_atoms(contents)
    if not geometry["atoms"]:
        raise ValueError(f"could not extract an equilibrium geometry from {path}")
    stability_assessment = _assess_frequency_stability(frequency)

    selected_mode = _select_mode_for_displacement(
        frequency["modes"],
        mode_number=mode_number,
        significant_threshold_cm1=significant_threshold_cm1,
    )
    displacements = selected_mode.get("displacements_cartesian") or []
    if not displacements:
        raise ValueError("selected mode does not contain cartesian displacements")
    if len(displacements) != len(geometry["atoms"]):
        raise ValueError("mode displacement count does not match geometry atom count")

    max_norm = max(
        math.sqrt(entry["x"] ** 2 + entry["y"] ** 2 + entry["z"] ** 2)
        for entry in displacements
    )
    if max_norm <= 0.0:
        raise ValueError("selected mode has zero displacement norm")

    scale_factor = amplitude_angstrom / max_norm
    plus_atoms = []
    minus_atoms = []
    for atom, disp in zip(geometry["atoms"], displacements):
        plus_atoms.append(
            {
                "label": atom["label"],
                "x": atom["x"] + scale_factor * disp["x"],
                "y": atom["y"] + scale_factor * disp["y"],
                "z": atom["z"] + scale_factor * disp["z"],
            }
        )
        minus_atoms.append(
            {
                "label": atom["label"],
                "x": atom["x"] - scale_factor * disp["x"],
                "y": atom["y"] - scale_factor * disp["y"],
                "z": atom["z"] - scale_factor * disp["z"],
            }
        )

    bonds, adjacency = _infer_bonds(geometry["atoms"])
    mode_analysis = _analyze_single_mode(
        selected_mode,
        top_atoms=4,
        geometry_atoms=geometry["atoms"],
        bonds=bonds,
        adjacency=adjacency,
    )

    return {
        "metadata": make_metadata(path, contents, "nwchem"),
        "selected_mode": mode_analysis,
        "stability_assessment": stability_assessment,
        "amplitude_angstrom": amplitude_angstrom,
        "scale_factor": scale_factor,
        "equilibrium_geometry": geometry,
        "plus_geometry": {
            "atoms": plus_atoms,
            "geometry_block": _render_output_geometry_block(plus_atoms),
        },
        "minus_geometry": {
            "atoms": minus_atoms,
            "geometry_block": _render_output_geometry_block(minus_atoms),
        },
    }


def _extract_frequency_modes(contents: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    regular_modes: list[dict[str, Any]] = []
    projected_modes: list[dict[str, Any]] = []
    active_ir_table: str | None = None
    saw_rows_in_table = False

    for line in contents.splitlines():
        trimmed = line.strip()

        if "Projected Infra Red Intensities" in line:
            active_ir_table = "projected"
            saw_rows_in_table = False
            continue
        if "Infra Red Intensities" in line and "Projected" not in line:
            active_ir_table = "raw"
            saw_rows_in_table = False
            continue

        if active_ir_table is not None:
            match = FREQUENCY_ROW_RE.match(line)
            if match:
                target = projected_modes if active_ir_table == "projected" else regular_modes
                frequency = parse_scientific_float(match.group(2))
                ir_km_mol = parse_scientific_float(match.group(5))
                if frequency is not None:
                    target.append(
                        {
                            "mode_index": len(target),
                            "mode_number": int(match.group(1)),
                            "frequency_cm1": frequency,
                            "is_imaginary": frequency < 0.0,
                            "ir_intensity_km_mol": ir_km_mol,
                            "metadata": {"table_kind": active_ir_table},
                            "displacements_angstrom": None,
                        }
                    )
                    saw_rows_in_table = True
                continue
            if saw_rows_in_table and trimmed and not trimmed.startswith("-"):
                active_ir_table = None
            elif saw_rows_in_table and not trimmed:
                active_ir_table = None

    if regular_modes or projected_modes:
        return regular_modes, projected_modes

    return _extract_frequency_modes_fallback(contents)


def _analyze_single_mode(
    mode: dict[str, Any],
    top_atoms: int = 4,
    geometry_atoms: list[dict[str, Any]] | None = None,
    bonds: list[dict[str, Any]] | None = None,
    adjacency: dict[int, list[int]] | None = None,
) -> dict[str, Any]:
    displacements = mode.get("displacements_cartesian") or []
    atom_summaries = []
    axis_totals = {"x": 0.0, "y": 0.0, "z": 0.0}

    for entry in displacements:
        norm = math.sqrt(entry["x"] ** 2 + entry["y"] ** 2 + entry["z"] ** 2)
        axis_totals["x"] += abs(entry["x"])
        axis_totals["y"] += abs(entry["y"])
        axis_totals["z"] += abs(entry["z"])
        atom_summaries.append(
            {
                "atom_index": entry["atom_index"],
                "label": entry.get("label"),
                "norm": norm,
                "x": entry["x"],
                "y": entry["y"],
                "z": entry["z"],
            }
        )

    total_norm = sum(atom["norm"] for atom in atom_summaries)
    ranked_atoms = sorted(atom_summaries, key=lambda atom: atom["norm"], reverse=True)
    dominant_atoms = []
    for atom in ranked_atoms[:top_atoms]:
        dominant_atoms.append(
            {
                **atom,
                "fraction_of_total_motion": (atom["norm"] / total_norm) if total_norm > 0 else None,
            }
        )

    axis_total = sum(axis_totals.values())
    axis_fractions = {
        axis: (value / axis_total if axis_total > 0 else None)
        for axis, value in axis_totals.items()
    }
    dominant_axis = max(axis_totals, key=axis_totals.get) if axis_total > 0 else None
    axis_character = "mixed"
    if dominant_axis is not None and axis_fractions[dominant_axis] is not None and axis_fractions[dominant_axis] >= 0.65:
        axis_character = dominant_axis

    leading_fraction = sum(atom["fraction_of_total_motion"] or 0.0 for atom in dominant_atoms[:2])
    locality = "localized" if leading_fraction >= 0.65 else "delocalized"
    motion_classification = {
        "type": "unclassified",
        "score": 0.0,
        "rationale": "insufficient geometry-aware evidence",
    }
    recommended_action = "displace_along_mode_plus_and_minus_then_reoptimize"
    if geometry_atoms is not None and bonds is not None and adjacency is not None and displacements:
        motion_classification = _classify_mode_motion(
            geometry_atoms,
            displacements,
            bonds,
            adjacency,
            top_atom_summaries=ranked_atoms,
        )
        recommended_action = _recommended_action_for_motion(
            motion_classification["type"],
            mode["frequency_cm1"],
        )

    return {
        "mode_number": mode.get("mode_number"),
        "frequency_cm1": mode["frequency_cm1"],
        "is_imaginary": mode["is_imaginary"],
        "ir_intensity_km_mol": mode.get("ir_intensity_km_mol"),
        "dominant_axis": dominant_axis,
        "axis_character": axis_character,
        "axis_fractions": axis_fractions,
        "locality": locality,
        "atom_count": len(displacements),
        "dominant_atoms": dominant_atoms,
        "max_atom_displacement": ranked_atoms[0]["norm"] if ranked_atoms else None,
        "motion_type": motion_classification["type"],
        "motion_score": motion_classification["score"],
        "motion_rationale": motion_classification["rationale"],
        "recommended_action": recommended_action,
        "displacements_cartesian": displacements,
    }


def _classify_mode_motion(
    geometry_atoms: list[dict[str, Any]],
    displacements: list[dict[str, Any]],
    bonds: list[dict[str, Any]],
    adjacency: dict[int, list[int]],
    top_atom_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    best = {
        "type": "unclassified",
        "score": 0.0,
        "rationale": "no strong stretch, bend, torsion, or metal-ligand signature detected",
    }
    atom_motion = {
        entry["atom_index"]: entry["norm"]
        for entry in top_atom_summaries
    }
    total_motion = sum(atom_motion.values()) or 1.0

    inversion_scores: list[tuple[float, str]] = []
    for center, neighbors in adjacency.items():
        if len(neighbors) < 3:
            continue
        if geometry_atoms[center]["label"] in TRANSITION_METALS:
            continue
        neighbor_motion = sum(atom_motion.get(neighbor, 0.0) for neighbor in neighbors)
        center_motion = atom_motion.get(center, 0.0)
        if neighbor_motion <= 0.0:
            continue
        same_label_neighbors = len({geometry_atoms[neighbor]["label"] for neighbor in neighbors}) == 1
        score = (neighbor_motion / total_motion) + 0.25 * (center_motion / total_motion)
        if same_label_neighbors:
            score += 0.1
        label = f"{geometry_atoms[center]['label']} with neighbors " + ",".join(
            geometry_atoms[neighbor]["label"] for neighbor in neighbors[:4]
        )
        inversion_scores.append((score, label))

    if inversion_scores:
        inversion_score, inversion_label = max(inversion_scores, key=lambda item: item[0])
        if inversion_score > best["score"] and inversion_score >= 0.75:
            best = {
                "type": "pyramidal_inversion",
                "score": inversion_score,
                "rationale": f"motion is dominated by multiple equivalent substituents around {inversion_label}",
            }

    bond_scores: list[tuple[float, str]] = []
    for bond in bonds:
        i = bond["i"]
        j = bond["j"]
        axis = _normalize(_vector_sub(_position(geometry_atoms[j]), _position(geometry_atoms[i])))
        if axis is None:
            continue
        di = _disp_vector(displacements[i])
        dj = _disp_vector(displacements[j])
        relative = _vector_sub(dj, di)
        stretch_score = abs(_dot(relative, axis))
        if geometry_atoms[i]["label"] in TRANSITION_METALS or geometry_atoms[j]["label"] in TRANSITION_METALS:
            if stretch_score > best["score"]:
                best = {
                    "type": "metal_ligand_distortion",
                    "score": stretch_score,
                    "rationale": f"largest bonded motion is along {geometry_atoms[i]['label']}-{geometry_atoms[j]['label']}",
                }
        bond_scores.append(
            (
                stretch_score,
                f"{geometry_atoms[i]['label']}-{geometry_atoms[j]['label']}",
            )
        )

    if bond_scores:
        stretch_score, stretch_label = max(bond_scores, key=lambda item: item[0])
        if stretch_score > best["score"] and stretch_score >= 0.20:
            best = {
                "type": "stretch",
                "score": stretch_score,
                "rationale": f"relative motion is strongest along bonded pair {stretch_label}",
            }

    angle_scores: list[tuple[float, str]] = []
    for center, neighbors in adjacency.items():
        if len(neighbors) < 2:
            continue
        for left_index in range(len(neighbors)):
            for right_index in range(left_index + 1, len(neighbors)):
                i = neighbors[left_index]
                k = neighbors[right_index]
                v1 = _normalize(_vector_sub(_position(geometry_atoms[i]), _position(geometry_atoms[center])))
                v2 = _normalize(_vector_sub(_position(geometry_atoms[k]), _position(geometry_atoms[center])))
                if v1 is None or v2 is None:
                    continue
                bend_score = _norm(_vector_sub(_disp_vector(displacements[i]), _disp_vector(displacements[k])))
                label = f"{geometry_atoms[i]['label']}-{geometry_atoms[center]['label']}-{geometry_atoms[k]['label']}"
                angle_scores.append((bend_score, label))

    if angle_scores:
        bend_score, bend_label = max(angle_scores, key=lambda item: item[0])
        if bend_score > best["score"] and bend_score >= 0.25:
            best = {
                "type": "bend",
                "score": bend_score,
                "rationale": f"opposing motion is strongest across angle {bend_label}",
            }

    torsion_scores: list[tuple[float, str]] = []
    for bond in bonds:
        b = bond["i"]
        c = bond["j"]
        left_neighbors = [neighbor for neighbor in adjacency[b] if neighbor != c]
        right_neighbors = [neighbor for neighbor in adjacency[c] if neighbor != b]
        if not left_neighbors or not right_neighbors:
            continue
        axis = _normalize(_vector_sub(_position(geometry_atoms[c]), _position(geometry_atoms[b])))
        if axis is None:
            continue
        for a in left_neighbors:
            for d in right_neighbors:
                endpoint_motion = _norm(_disp_vector(displacements[a])) + _norm(_disp_vector(displacements[d]))
                central_motion = _norm(_disp_vector(displacements[b])) + _norm(_disp_vector(displacements[c]))
                if endpoint_motion <= 0:
                    continue
                perp_score = (
                    _perpendicular_component(_disp_vector(displacements[a]), axis)
                    + _perpendicular_component(_disp_vector(displacements[d]), axis)
                )
                score = perp_score * (endpoint_motion / (endpoint_motion + central_motion + 1e-12))
                label = (
                    f"{geometry_atoms[a]['label']}-{geometry_atoms[b]['label']}-"
                    f"{geometry_atoms[c]['label']}-{geometry_atoms[d]['label']}"
                )
                torsion_scores.append((score, label))

    if torsion_scores:
        torsion_score, torsion_label = max(torsion_scores, key=lambda item: item[0])
        if torsion_score > best["score"] and torsion_score >= 0.20:
            best = {
                "type": "torsion",
                "score": torsion_score,
                "rationale": f"motion is concentrated on terminal atoms around dihedral {torsion_label}",
            }

    if best["type"] == "unclassified" and top_atom_summaries:
        top_labels = ", ".join(
            f"{geometry_atoms[atom['atom_index']]['label']}{atom['atom_index'] + 1}"
            for atom in top_atom_summaries[:2]
        )
        best["rationale"] = f"largest motion is on {top_labels}"

    return best


def _recommended_action_for_motion(motion_type: str, frequency_cm1: float) -> str:
    if motion_type == "torsion":
        return "displace_plus_minus_and_reoptimize_with_noautosym_or_c1"
    if motion_type == "pyramidal_inversion":
        return "displace_plus_minus_and_reoptimize_without_symmetry_constraints"
    if motion_type == "bend":
        return "displace_plus_minus_and_reoptimize_then_recheck_frequency"
    if motion_type == "stretch":
        return "inspect_bonding_then_displace_plus_minus_and_reoptimize"
    if motion_type == "metal_ligand_distortion":
        return "inspect_spin_state_and_metal_ligand_geometry_then_displace_and_reoptimize"
    if abs(frequency_cm1) < 50.0:
        return "likely_soft_mode_check_symmetry_and_tighten_optimization_before_major_changes"
    return "displace_along_mode_plus_and_minus_then_reoptimize"


def _assess_frequency_stability(frequency_payload: dict[str, Any]) -> dict[str, Any]:
    significant_count = frequency_payload["significant_imaginary_mode_count"]
    near_zero_count = frequency_payload["near_zero_mode_count"]
    projected = frequency_payload["preferred_kind"] == "projected"
    thermochemistry = frequency_payload.get("thermochemistry") or {}
    linear_molecule = bool(thermochemistry.get("linear_molecule"))
    significant_values = frequency_payload.get("significant_imaginary_frequencies_cm1", [])

    if significant_count > 0:
        strongest = min(significant_values) if significant_values else None
        if strongest is not None and abs(strongest) < 50.0 and projected and near_zero_count >= 5:
            return {
                "classification": "soft_instability_or_projection_sensitive_mode",
                "likely_noise": False,
                "recommended_follow_up": "reoptimize_then_recheck_frequency",
            }
        return {
            "classification": "likely_real_instability",
            "likely_noise": False,
            "recommended_follow_up": "displace_and_reoptimize",
        }

    if near_zero_count > 0:
        if projected and ((linear_molecule and near_zero_count >= 4) or near_zero_count >= 5):
            return {
                "classification": "likely_projection_or_symmetry_zero_modes",
                "likely_noise": True,
                "recommended_follow_up": "usually_safe_if_other_modes_real_but_verify_geometry_quality",
            }
        return {
            "classification": "soft_modes_present",
            "likely_noise": True,
            "recommended_follow_up": "tighten_optimization_and_recheck_if_chemically_suspicious",
        }

    return {
        "classification": "all_real_modes",
        "likely_noise": False,
        "recommended_follow_up": "no_imaginary_mode_fix_needed",
    }


def _select_mode_for_displacement(
    modes: list[dict[str, Any]],
    mode_number: int | None,
    significant_threshold_cm1: float,
) -> dict[str, Any]:
    if mode_number is not None:
        for mode in modes:
            if mode.get("mode_number") == mode_number:
                return mode
        raise ValueError(f"mode number {mode_number} was not found")

    significant = [
        mode
        for mode in modes
        if mode["frequency_cm1"] <= -abs(significant_threshold_cm1)
    ]
    if significant:
        return significant[0]

    with_displacements = [mode for mode in modes if mode.get("displacements_cartesian")]
    if with_displacements:
        return with_displacements[0]

    raise ValueError("no suitable mode was found for displacement")


def _extract_normal_mode_displacements(
    contents: str,
    labels: list[str] | None = None,
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    lines = contents.splitlines()
    sections: dict[str, dict[int, list[float]]] = {"raw": {}, "projected": {}}
    idx = 0

    while idx < len(lines):
        if "NORMAL MODE EIGENVECTORS IN CARTESIAN COORDINATES" not in lines[idx]:
            idx += 1
            continue

        kind = "raw"
        section_start = idx + 1
        lookahead = section_start
        while lookahead < len(lines) and lookahead <= idx + 6:
            if "Projected Frequencies expressed in cm-1" in lines[lookahead]:
                kind = "projected"
                break
            lookahead += 1

        idx = section_start
        while idx < len(lines):
            trimmed = lines[idx].strip()
            if not trimmed:
                idx += 1
                continue
            if trimmed.startswith("Normal Eigenvalue") or "Projected Infra Red Intensities" in lines[idx]:
                break
            header_tokens = trimmed.split()
            if header_tokens and all(token.isdigit() for token in header_tokens):
                mode_numbers = [int(token) for token in header_tokens]
                idx += 1
                while idx < len(lines) and not lines[idx].strip():
                    idx += 1
                if idx >= len(lines):
                    break
                frequency_line = lines[idx].strip()
                if not (frequency_line.startswith("Frequency") or frequency_line.startswith("P.Frequency")):
                    idx += 1
                    continue
                for mode_number in mode_numbers:
                    sections[kind].setdefault(mode_number, [])
                idx += 1
                while idx < len(lines) and not lines[idx].strip():
                    idx += 1
                while idx < len(lines):
                    row = lines[idx].strip()
                    if not row:
                        idx += 1
                        break
                    if row.startswith("Normal Eigenvalue") or row.startswith("P.Frequency") or row.startswith("Frequency"):
                        break
                    row_tokens = row.split()
                    if len(row_tokens) >= len(mode_numbers) + 1 and row_tokens[0].isdigit():
                        values = [parse_scientific_float(token) for token in row_tokens[1 : 1 + len(mode_numbers)]]
                        if all(value is not None for value in values):
                            for mode_number, value in zip(mode_numbers, values):
                                sections[kind][mode_number].append(value)
                            idx += 1
                            continue
                    break
                continue
            idx += 1
        continue

    return {
        "raw": {
            mode_number: _format_displacements(values, labels)
            for mode_number, values in sections["raw"].items()
        },
        "projected": {
            mode_number: _format_displacements(values, labels)
            for mode_number, values in sections["projected"].items()
        },
    }


def _extract_frequency_modes_fallback(contents: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    regular_modes: list[dict[str, Any]] = []
    projected_modes: list[dict[str, Any]] = []
    pending_indices: dict[str, deque[int]] = {"raw": deque(), "projected": deque()}

    for line in contents.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("P.Frequency") and "=" not in trimmed:
            for value in [parse_scientific_float(token) for token in trimmed.split()[1:]]:
                if value is None:
                    continue
                projected_modes.append(
                    {
                        "mode_index": len(projected_modes),
                        "mode_number": len(projected_modes) + 1,
                        "frequency_cm1": value,
                        "is_imaginary": value < 0.0,
                        "ir_intensity_km_mol": None,
                        "metadata": {"table_kind": "projected"},
                        "displacements_angstrom": None,
                    }
                )
                pending_indices["projected"].append(len(projected_modes) - 1)
        elif trimmed.startswith("Frequency") and "=" not in trimmed:
            for value in [parse_scientific_float(token) for token in trimmed.split()[1:]]:
                if value is None:
                    continue
                regular_modes.append(
                    {
                        "mode_index": len(regular_modes),
                        "mode_number": len(regular_modes) + 1,
                        "frequency_cm1": value,
                        "is_imaginary": value < 0.0,
                        "ir_intensity_km_mol": None,
                        "metadata": {"table_kind": "raw"},
                        "displacements_angstrom": None,
                    }
                )
                pending_indices["raw"].append(len(regular_modes) - 1)
        elif trimmed.startswith("IR Inten") or trimmed.lower().startswith("ir intensity"):
            for token in trimmed.split():
                value = parse_scientific_float(token)
                if value is None or not pending_indices["raw"]:
                    continue
                regular_modes[pending_indices["raw"].popleft()]["ir_intensity_km_mol"] = value

    return regular_modes, projected_modes


def _attach_mode_displacements(
    modes: list[dict[str, Any]],
    displacements: dict[int, list[dict[str, Any]]],
) -> None:
    for mode in modes:
        mode["displacements_cartesian"] = displacements.get(mode.get("mode_number"), None)


def _format_displacements(values: list[float], labels: list[str] | None = None) -> list[dict[str, Any]]:
    if not values:
        return []
    atom_count = len(values) // 3
    formatted: list[dict[str, Any]] = []
    for atom_index in range(atom_count):
        base = atom_index * 3
        formatted.append(
            {
                "atom_index": atom_index,
                "label": labels[atom_index] if labels and atom_index < len(labels) else None,
                "x": values[base],
                "y": values[base + 1],
                "z": values[base + 2],
            }
        )
    return formatted


def _parse_thermochemistry(contents: str) -> dict[str, Any] | None:
    temperature = None
    scaling_parameter = None
    zero_point = None
    thermal_energy = None
    thermal_enthalpy = None
    total_entropy = None
    trans_entropy = None
    rot_entropy = None
    vib_entropy = None
    cv_total = None
    cv_trans = None
    cv_rot = None
    cv_vib = None
    mol_weight = None
    symmetry_number = None
    linear_molecule = False

    cv_section = False
    for line in contents.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue

        if trimmed.startswith("Temperature") and "=" in trimmed:
            temperature = _parse_float_with_units(trimmed, "=")
            continue
        if trimmed.startswith("frequency scaling parameter") and "=" in trimmed:
            scaling_parameter = _parse_float_with_units(trimmed, "=")
            continue
        if trimmed.startswith("Linear Molecule"):
            linear_molecule = True
            continue
        if trimmed.startswith("Zero-Point correction to Energy"):
            zero_point = _parse_thermo_correction(trimmed)
            continue
        if trimmed.startswith("Thermal correction to Energy"):
            thermal_energy = _parse_thermo_correction(trimmed)
            continue
        if trimmed.startswith("Thermal correction to Enthalpy"):
            thermal_enthalpy = _parse_thermo_correction(trimmed)
            continue
        if trimmed.startswith("Total Entropy") and "=" in trimmed:
            total_entropy = parse_float_after_delimiter(trimmed, "=")
            cv_section = False
            continue
        if trimmed.startswith("Cv (constant volume heat capacity)") and "=" in trimmed:
            cv_total = parse_float_after_delimiter(trimmed, "=")
            cv_section = True
            continue
        if "- Translational" in trimmed and "cal/mol-K" in trimmed:
            value = parse_float_after_delimiter(trimmed, "=")
            if cv_section:
                cv_trans = value
            else:
                trans_entropy = value
                if weight_part := trimmed.split("mol. weight =", 1)[1:] :
                    mol_weight = parse_scientific_float(weight_part[0].strip().rstrip(")"))
            continue
        if "- Rotational" in trimmed and "cal/mol-K" in trimmed:
            value = parse_float_after_delimiter(trimmed, "=")
            if cv_section:
                cv_rot = value
            else:
                rot_entropy = value
                if sym_part := trimmed.split("symmetry #  =", 1)[1:] :
                    symmetry_number = int(sym_part[0].strip().rstrip(")"))
            continue
        if "- Vibrational" in trimmed and "cal/mol-K" in trimmed:
            value = parse_float_after_delimiter(trimmed, "=")
            if cv_section:
                cv_vib = value
            else:
                vib_entropy = value

    if temperature is None:
        return None
    if not any(
        value is not None
        for value in (zero_point, thermal_energy, thermal_enthalpy, total_entropy, cv_total)
    ):
        return None

    return {
        "temperature_kelvin": temperature,
        "frequency_scaling_parameter": scaling_parameter,
        "linear_molecule": linear_molecule,
        "zero_point_correction": zero_point,
        "thermal_correction_energy": thermal_energy,
        "thermal_correction_enthalpy": thermal_enthalpy,
        "total_entropy_cal_mol_k": total_entropy,
        "translational_entropy_cal_mol_k": trans_entropy,
        "rotational_entropy_cal_mol_k": rot_entropy,
        "vibrational_entropy_cal_mol_k": vib_entropy,
        "cv_total_cal_mol_k": cv_total,
        "cv_translational_cal_mol_k": cv_trans,
        "cv_rotational_cal_mol_k": cv_rot,
        "cv_vibrational_cal_mol_k": cv_vib,
        "molecular_weight": mol_weight,
        "symmetry_number": symmetry_number,
    }


def _extract_last_geometry_labels(contents: str) -> list[str]:
    return [atom["label"] for atom in _extract_last_geometry_atoms(contents)["atoms"]]


def _extract_last_geometry_atoms(contents: str) -> dict[str, Any]:
    labels: list[str] = []
    atoms: list[dict[str, Any]] = []
    current: list[str] = []
    current_atoms: list[dict[str, Any]] = []
    in_geom = False
    coordinate_unit = "angstrom"

    for line in contents.splitlines():
        if "Output coordinates in angstroms" in line:
            in_geom = True
            current = []
            current_atoms = []
            coordinate_unit = "angstrom"
            continue
        if "Output coordinates in a.u." in line:
            in_geom = True
            current = []
            current_atoms = []
            coordinate_unit = "bohr"
            continue
        if not in_geom:
            continue
        trimmed = line.strip()
        if not trimmed or line.startswith("----"):
            continue
        if "No." in line and "Tag" in line and "Charge" in line:
            continue
        if "Atomic Mass" in line:
            if current:
                labels = current[:]
                atoms = current_atoms[:]
            in_geom = False
            continue
        parts = line.split()
        if len(parts) >= 6:
            try:
                int(parts[0])
                coords = _to_angstrom(
                    [float(parts[3]), float(parts[4]), float(parts[5])],
                    coordinate_unit,
                )
            except ValueError:
                continue
            current.append(parts[1])
            current_atoms.append(
                {
                    "label": parts[1],
                    "x": coords[0],
                    "y": coords[1],
                    "z": coords[2],
                }
            )

    return {
        "labels": labels,
        "atoms": atoms,
        "atom_count": len(atoms),
    }


def _render_output_geometry_block(atoms: list[dict[str, Any]]) -> str:
    lines = ['geometry units angstrom']
    for atom in atoms:
        lines.append(
            f"  {atom['label']:<2} {atom['x']: .8f} {atom['y']: .8f} {atom['z']: .8f}"
        )
    lines.append("end")
    return "\n".join(lines)


def _parse_thermo_correction(line: str) -> dict[str, float] | None:
    kcal = parse_float_after_delimiter(line, "=")
    hartree = None
    if match := THERMO_AU_RE.search(line):
        hartree = parse_scientific_float(match.group(1))
    if kcal is None and hartree is None:
        return None
    return {
        "kcal_mol": kcal,
        "hartree": hartree,
    }


def _parse_float_with_units(line: str, delimiter: str) -> float | None:
    value = parse_float_after_delimiter(line, delimiter)
    if value is not None:
        return value
    if delimiter not in line:
        return None
    token = line.split(delimiter, 1)[1].split()[0]
    token = token.rstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return parse_scientific_float(token)


def _to_angstrom(position: list[float], unit: str) -> list[float]:
    if unit == "bohr":
        return [value * BOHR_TO_ANGSTROM for value in position]
    return position


def _evaluate_optimization_criteria(
    final_metrics: dict[str, Any],
    thresholds: dict[str, float | None],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    all_met = True
    for key in ("gmax", "grms", "xmax", "xrms"):
        threshold = thresholds.get(key)
        value = final_metrics.get(key)
        if threshold is None or value is None:
            checks[key] = None
            continue
        met = value <= threshold + 1e-12
        checks[key] = met
        all_met = all_met and met
    checks["all_met"] = all_met
    return checks


def _infer_bonds(atoms: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    bonds: list[dict[str, Any]] = []
    adjacency = {index: [] for index in range(len(atoms))}
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            ri = COVALENT_RADII.get(atoms[i]["label"], 0.77)
            rj = COVALENT_RADII.get(atoms[j]["label"], 0.77)
            cutoff = 1.25 * (ri + rj)
            distance = _distance(_position(atoms[i]), _position(atoms[j]))
            if 0.1 < distance <= cutoff:
                bonds.append({"i": i, "j": j, "distance_angstrom": distance})
                adjacency[i].append(j)
                adjacency[j].append(i)
    return bonds, adjacency


def _position(atom: dict[str, Any]) -> list[float]:
    return [atom["x"], atom["y"], atom["z"]]


def _disp_vector(entry: dict[str, Any]) -> list[float]:
    return [entry["x"], entry["y"], entry["z"]]


def _vector_sub(a: list[float], b: list[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _dot(a: list[float], b: list[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: list[float]) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: list[float]) -> list[float] | None:
    norm = _norm(a)
    if norm <= 0.0:
        return None
    return [value / norm for value in a]


def _distance(a: list[float], b: list[float]) -> float:
    return _norm(_vector_sub(a, b))


def _perpendicular_component(vector: list[float], axis: list[float]) -> float:
    parallel = _dot(vector, axis)
    squared = max(0.0, _norm(vector) ** 2 - parallel**2)
    return math.sqrt(squared)

"""Assess structural health across normalized PWSCF trajectory frames.

Bond-network judgments are restricted to isolated molecules with clear vacuum
padding. Extended periodic structures receive geometric metrics without a
molecular connectivity verdict.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from chemtools.core.common import COVALENT_RADII


_COVALENT_TOLERANCE = 1.20
_CLOSE_CONTACT_ANGSTROM = 0.60
_MINIMUM_VACUUM_GAP_ANGSTROM = 5.0
_MAXIMUM_IMAGE_CANDIDATES = 4096
_MAXIMUM_ANALYZED_FRAMES = 512
_MAXIMUM_PAIR_EVALUATIONS = 250_000
_MAIN_GROUP_COORDINATION_LIMITS = {
    "H": 1,
    "B": 4,
    "C": 4,
    "N": 4,
    "O": 3,
    "F": 1,
    "Si": 6,
    "P": 6,
    "S": 6,
    "Cl": 1,
    "Br": 1,
    "I": 1,
}


def analyze_pw_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    frames = trajectory.get("frames") or []
    if trajectory.get("status") != "available" or not frames:
        return {
            "schema": "qe-trajectory-structural-analysis/1",
            "scope": "not_assessed",
            "verdict": {
                "status": "not_assessed",
                "reasons": ["No normalized trajectory frames are available."],
            },
        }
    if len(frames) > _MAXIMUM_ANALYZED_FRAMES:
        return _limit_exceeded(
            "frames",
            len(frames),
            _MAXIMUM_ANALYZED_FRAMES,
        )
    pair_evaluations = sum(
        len(frame.get("atoms") or [])
        * (len(frame.get("atoms") or []) - 1)
        // 2
        for frame in frames
    )
    if pair_evaluations > _MAXIMUM_PAIR_EVALUATIONS:
        return _limit_exceeded(
            "pair evaluations",
            pair_evaluations,
            _MAXIMUM_PAIR_EVALUATIONS,
        )

    try:
        frame_summaries = [_analyze_frame(frame) for frame in frames]
    except (ValueError, np.linalg.LinAlgError) as error:
        return {
            "schema": "qe-trajectory-structural-analysis/1",
            "scope": "not_assessed",
            "verdict": {
                "status": "not_assessed",
                "reasons": [f"Periodic geometry metrics failed: {error}"],
            },
        }
    initial = frame_summaries[0]
    final = frame_summaries[-1]
    vacuum_isolated = all(
        summary["isolated_molecule_candidate"] for summary in frame_summaries
    )
    radii_complete = all(
        summary["covalent_radius_coverage"] == 1.0
        for summary in frame_summaries
    )
    molecular_assessment = vacuum_isolated and radii_complete
    scope = "isolated_molecule" if molecular_assessment else "metrics_only"
    evolution = _evolution(initial, final)
    verdict = (
        _molecular_verdict(frame_summaries, evolution)
        if molecular_assessment
        else _limited_verdict(vacuum_isolated, radii_complete)
    )
    observations = []
    if evolution["large_cell_volume_change"]:
        observations.append({
            "code": "large_cell_volume_change",
            "message": (
                "The cell volume changes by "
                f"{evolution['cell_volume_change_percent']:.1f} percent."
            ),
            "impact": (
                "Check the starting cell, pressure, stress, and convergence "
                "path before treating the final cell as routine."
            ),
        })
    return {
        "schema": "qe-trajectory-structural-analysis/1",
        "scope": scope,
        "method": {
            "distance_model": "periodic_minimum_image",
            "covalent_tolerance": _COVALENT_TOLERANCE,
            "close_contact_angstrom": _CLOSE_CONTACT_ANGSTROM,
            "minimum_vacuum_gap_angstrom": _MINIMUM_VACUUM_GAP_ANGSTROM,
        },
        "limits": {
            "maximum_frames": _MAXIMUM_ANALYZED_FRAMES,
            "maximum_pair_evaluations": _MAXIMUM_PAIR_EVALUATIONS,
            "maximum_image_candidates_per_pair": _MAXIMUM_IMAGE_CANDIDATES,
        },
        "initial": initial,
        "final": final,
        "evolution": evolution,
        "observations": observations,
        "verdict": verdict,
    }


def _limit_exceeded(
    measure: str,
    actual: int,
    limit: int,
) -> dict[str, Any]:
    return {
        "schema": "qe-trajectory-structural-analysis/1",
        "scope": "not_assessed",
        "verdict": {
            "status": "not_assessed",
            "reasons": [
                f"Structural analysis requires {actual} {measure}, exceeding "
                f"the bounded limit of {limit}."
            ],
        },
    }


def _limited_verdict(
    vacuum_isolated: bool,
    radii_complete: bool,
) -> dict[str, Any]:
    reasons = []
    if not vacuum_isolated:
        reasons.append(
            (
                "The structure lacks enough vacuum padding for molecular "
                "bond-network heuristics; use a periodic topology analysis."
            )
        )
    if not radii_complete:
        reasons.append(
            "The shared covalent-radius table does not cover every element."
        )
    return {
        "status": "not_assessed",
        "reasons": reasons,
    }


def _analyze_frame(frame: dict[str, Any]) -> dict[str, Any]:
    atoms = frame["atoms"]
    cell = np.asarray(frame["cell"]["vectors_angstrom"], dtype=float)
    coordinates = np.asarray(
        [[atom["x"], atom["y"], atom["z"]] for atom in atoms],
        dtype=float,
    )
    volume = abs(float(np.linalg.det(cell)))
    if not np.isfinite(cell).all() or not np.isfinite(coordinates).all():
        raise ValueError("the trajectory contains non-finite coordinates")
    if not np.isfinite(volume) or volume <= 1e-12:
        raise ValueError("the trajectory contains a singular periodic cell")
    inverse = np.linalg.inv(cell)
    smallest_stretch = float(np.linalg.svd(cell, compute_uv=False)[-1])
    fractional = coordinates @ inverse
    vacuum_gaps = _vacuum_gaps(cell, fractional, volume)
    isolated = min(vacuum_gaps) >= _MINIMUM_VACUUM_GAP_ANGSTROM

    coordination = [0] * len(atoms)
    bonds: set[tuple[int, int]] = set()
    close_contacts = 0
    minimum_distance: float | None = None
    covered_atoms = sum(atom["element"] in COVALENT_RADII for atom in atoms)
    for left in range(len(atoms)):
        for right in range(left + 1, len(atoms)):
            delta_fractional = fractional[right] - fractional[left]
            distance = _minimum_image_distance(
                delta_fractional,
                cell,
                smallest_stretch,
            )
            minimum_distance = (
                distance
                if minimum_distance is None
                else min(minimum_distance, distance)
            )
            if distance < _CLOSE_CONTACT_ANGSTROM:
                close_contacts += 1
            left_radius = COVALENT_RADII.get(atoms[left]["element"])
            right_radius = COVALENT_RADII.get(atoms[right]["element"])
            if left_radius is None or right_radius is None:
                continue
            if distance <= _COVALENT_TOLERANCE * (left_radius + right_radius):
                bonds.add((left, right))
                coordination[left] += 1
                coordination[right] += 1

    fragments = _fragment_count(len(atoms), bonds)
    overcoordinated = sum(
        coordination[index]
        > _MAIN_GROUP_COORDINATION_LIMITS.get(atom["element"], float("inf"))
        for index, atom in enumerate(atoms)
    )
    return {
        "frame_index": frame["index"],
        "role": frame["role"],
        "cell_volume_angstrom3": round(volume, 12),
        "minimum_pair_distance_angstrom": (
            round(minimum_distance, 12)
            if minimum_distance is not None
            else None
        ),
        "minimum_vacuum_gap_angstrom": round(min(vacuum_gaps), 12),
        "isolated_molecule_candidate": isolated,
        "covalent_radius_coverage": (
            covered_atoms / len(atoms) if atoms else 0.0
        ),
        "covalent_bond_count": len(bonds),
        "close_contact_count": close_contacts,
        "overcoordinated_atom_count": overcoordinated,
        "dangling_atom_count": sum(value == 0 for value in coordination),
        "fragment_count": fragments,
    }


def _minimum_image_distance(
    delta_fractional: np.ndarray,
    cell: np.ndarray,
    smallest_stretch: float,
) -> float:
    nearest_integer = np.rint(delta_fractional).astype(int)
    best_distance = float(
        np.linalg.norm((delta_fractional - nearest_integer) @ cell)
    )
    search_radius = best_distance / smallest_stretch
    image_ranges = [
        range(
            int(np.ceil(value - search_radius)),
            int(np.floor(value + search_radius)) + 1,
        )
        for value in delta_fractional
    ]
    candidate_count = int(np.prod([len(values) for values in image_ranges]))
    if candidate_count > _MAXIMUM_IMAGE_CANDIDATES:
        raise ValueError(
            "the periodic cell is too ill-conditioned for bounded "
            "minimum-image analysis"
        )

    for image in product(*image_ranges):
        displacement = delta_fractional - image
        if np.linalg.norm(displacement) > search_radius + 1e-12:
            continue
        best_distance = min(
            best_distance,
            float(np.linalg.norm(displacement @ cell)),
        )
    return best_distance


def _vacuum_gaps(
    cell: np.ndarray,
    fractional: np.ndarray,
    volume: float,
) -> list[float]:
    face_areas = [
        float(np.linalg.norm(np.cross(cell[1], cell[2]))),
        float(np.linalg.norm(np.cross(cell[2], cell[0]))),
        float(np.linalg.norm(np.cross(cell[0], cell[1]))),
    ]
    heights = [volume / area for area in face_areas]
    return [
        _largest_circular_gap(fractional[:, axis]) * heights[axis]
        for axis in range(3)
    ]


def _largest_circular_gap(values: np.ndarray) -> float:
    wrapped = sorted(float(value % 1.0) for value in values)
    if len(wrapped) < 2:
        return 1.0
    gaps = [
        wrapped[index + 1] - wrapped[index]
        for index in range(len(wrapped) - 1)
    ]
    gaps.append(wrapped[0] + 1.0 - wrapped[-1])
    return max(gaps)


def _fragment_count(atom_count: int, bonds: set[tuple[int, int]]) -> int:
    parents = list(range(atom_count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left, right in bonds:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[left_root] = right_root
    return len({find(index) for index in range(atom_count)})


def _evolution(
    initial: dict[str, Any],
    final: dict[str, Any]
) -> dict[str, Any]:
    initial_volume = initial["cell_volume_angstrom3"]
    final_volume = final["cell_volume_angstrom3"]
    return {
        "covalent_bond_count_change": (
            final["covalent_bond_count"] - initial["covalent_bond_count"]
        ),
        "close_contact_count_change": (
            final["close_contact_count"] - initial["close_contact_count"]
        ),
        "dangling_atom_count_change": (
            final["dangling_atom_count"] - initial["dangling_atom_count"]
        ),
        "fragment_count_change": (
            final["fragment_count"] - initial["fragment_count"]
        ),
        "cell_volume_change_percent": round(
            100.0 * (final_volume - initial_volume) / initial_volume,
            12,
        ),
        "large_cell_volume_change": (
            abs(final_volume - initial_volume) / initial_volume >= 0.20
        ),
    }


def _molecular_verdict(
    frames: list[dict[str, Any]],
    evolution: dict[str, Any],
) -> dict[str, Any]:
    initial = frames[0]
    final = frames[-1]
    findings = []
    if initial["close_contact_count"]:
        findings.append({
            "code": "initial_close_contact",
            "origin": "input_geometry",
            "message": (
                "The initial geometry has "
                f"{initial['close_contact_count']} pair(s) closer than "
                f"{_CLOSE_CONTACT_ANGSTROM:.2f} angstrom."
            ),
        })
    if initial["overcoordinated_atom_count"]:
        findings.append({
            "code": "initial_main_group_overcoordination",
            "origin": "input_geometry",
            "message": (
                "The initial covalent-radius graph has "
                f"{initial['overcoordinated_atom_count']} overcoordinated "
                "main-group atom(s)."
            ),
        })
    intermediate_problem = next(
        (
            frame
            for frame in frames[1:-1]
            if (
                frame["close_contact_count"]
                > initial["close_contact_count"]
                or frame["overcoordinated_atom_count"]
                > initial["overcoordinated_atom_count"]
            )
        ),
        None,
    )
    if intermediate_problem is not None:
        if (
            intermediate_problem["close_contact_count"]
            > initial["close_contact_count"]
        ):
            message = (
                f"Frame {intermediate_problem['frame_index']} increases the "
                "close-contact count from "
                f"{initial['close_contact_count']} to "
                f"{intermediate_problem['close_contact_count']}."
            )
        else:
            message = (
                f"Frame {intermediate_problem['frame_index']} increases the "
                "overcoordinated main-group atom count from "
                f"{initial['overcoordinated_atom_count']} to "
                f"{intermediate_problem['overcoordinated_atom_count']}."
            )
        findings.append({
            "code": "intermediate_structural_concern",
            "origin": "trajectory",
            "message": message,
        })
    if final["close_contact_count"] > initial["close_contact_count"]:
        findings.append({
            "code": "final_close_contact_increase",
            "origin": "trajectory",
            "message": (
                "The final frame increases the close-contact count from "
                f"{initial['close_contact_count']} to "
                f"{final['close_contact_count']}."
            ),
        })
    if final["overcoordinated_atom_count"] > initial[
        "overcoordinated_atom_count"
    ]:
        findings.append({
            "code": "final_main_group_overcoordination_increase",
            "origin": "trajectory",
            "message": (
                "The final frame increases the overcoordinated main-group "
                "atom count from "
                f"{initial['overcoordinated_atom_count']} to "
                f"{final['overcoordinated_atom_count']}."
            ),
        })
    if evolution["fragment_count_change"] > 0:
        findings.append({
            "code": "new_fragment",
            "origin": "trajectory",
            "message": (
                "The covalent-radius graph gains "
                f"{evolution['fragment_count_change']} disconnected "
                "fragment(s)."
            ),
        })
    if evolution["dangling_atom_count_change"] > 0:
        findings.append({
            "code": "new_dangling_atom",
            "origin": "trajectory",
            "message": (
                "The final frame has "
                f"{evolution['dangling_atom_count_change']} additional atom(s) "
                "without a covalent-radius neighbor."
            ),
        })
    if findings:
        origins = {finding["origin"] for finding in findings}
        return {
            "status": "concerning",
            "origin": next(iter(origins)) if len(origins) == 1 else "mixed",
            "reasons": [finding["message"] for finding in findings],
            "findings": findings,
        }
    return {
        "status": "no_obvious_issue",
        "origin": None,
        "reasons": [
            "No close contacts, new fragmentation, or main-group "
            "overcoordination were detected by this heuristic."
        ],
        "findings": [],
    }


__all__ = ["analyze_pw_trajectory"]

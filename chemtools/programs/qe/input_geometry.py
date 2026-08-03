"""Normalize explicit pw.x input geometries for structural review."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from chemtools.core.types import LintIssue
from chemtools.core.units import ANGSTROM_PER_BOHR
from chemtools.programs.qe._elements import element_from_label
from chemtools.programs.qe.trajectory_analysis import analyze_pw_trajectory


def analyze_pw_input_geometry(parsed: dict[str, Any]) -> dict[str, Any]:
    geometry = normalize_pw_input_geometry(parsed)
    if geometry["status"] != "available":
        return _not_assessed(geometry["reason"])
    analysis = analyze_pw_trajectory({
        "status": "available",
        "frames": [{
            "index": 0,
            "role": "input",
            "atoms": geometry["atoms"],
            "cell": {
                "vectors_angstrom": geometry["cell_vectors_angstrom"],
                "periodic": [True, True, True],
            },
        }],
    })
    return {
        **analysis,
        "schema": "qe-input-structural-analysis/1",
        "coordinate_contract": geometry["coordinate_contract"],
    }


def normalize_pw_input_geometry(parsed: dict[str, Any]) -> dict[str, Any]:
    cell_result = _normalized_cell(parsed)
    if isinstance(cell_result, str):
        return {"status": "unavailable", "reason": cell_result}
    cell, cell_units, alat_angstrom = cell_result
    positions_record = parsed.get("atomic_positions") or {}
    raw_position_units = str(positions_record.get("units") or "").lower()
    if raw_position_units == "crystal_sg":
        return {
            "status": "unavailable",
            "reason": (
                "Structural review does not expand ATOMIC_POSITIONS crystal_sg "
                "symmetry records."
            ),
        }
    position_units = raw_position_units or "implicit_alat"
    if position_units not in {
        "alat",
        "angstrom",
        "bohr",
        "crystal",
        "implicit_alat",
    }:
        return {
            "status": "unavailable",
            "reason": (
                "Structural review supports ATOMIC_POSITIONS in alat, angstrom, "
                "bohr, or crystal coordinates."
            ),
        }

    position_rows = positions_record.get("atoms") or []
    coordinates = np.asarray(
        [row.get("coordinates") or [] for row in position_rows],
        dtype=float,
    )
    if not position_rows or coordinates.shape != (len(position_rows), 3):
        return {
            "status": "unavailable",
            "reason": "Structural review requires complete coordinates for every atom.",
        }

    if position_units == "bohr":
        coordinates = coordinates * ANGSTROM_PER_BOHR
    elif position_units in {"alat", "implicit_alat"}:
        if alat_angstrom is None:
            return {
                "status": "unavailable",
                "reason": (
                    "Unitless ATOMIC_POSITIONS uses alat and requires celldm(1) "
                    "or A."
                    if position_units == "implicit_alat"
                    else "ATOMIC_POSITIONS alat requires celldm(1) or A."
                ),
            }
        coordinates = coordinates * alat_angstrom
    elif position_units == "crystal":
        coordinates = coordinates @ cell

    atoms = [
        {
            "element": element_from_label(str(row["label"])) or row["label"],
            "x": float(coordinates[index, 0]),
            "y": float(coordinates[index, 1]),
            "z": float(coordinates[index, 2]),
        }
        for index, row in enumerate(position_rows)
    ]
    return {
        "status": "available",
        "atoms": atoms,
        "cell_vectors_angstrom": cell.tolist(),
        "coordinate_contract": {
            "cell_input_units": cell_units,
            "position_input_units": position_units,
            "normalized_units": "angstrom",
        },
    }


def _normalized_cell(
    parsed: dict[str, Any],
) -> tuple[np.ndarray, str, float | None] | str:
    ibrav = parsed.get("system", {}).get("ibrav")
    if ibrav == 0:
        return _explicit_cell(parsed)
    if not isinstance(ibrav, int):
        return "Structural review requires a numeric ibrav value."
    return _bravais_cell(parsed, ibrav)


def _explicit_cell(
    parsed: dict[str, Any],
) -> tuple[np.ndarray, str, float | None] | str:
    cell_record = parsed.get("cell_parameters") or {}
    cell_units = str(cell_record.get("units") or "").lower()
    if cell_units not in {"", "alat", "angstrom", "bohr"}:
        return (
            "Structural review supports CELL_PARAMETERS in alat, angstrom, "
            "or bohr."
        )
    cell = np.asarray(cell_record.get("vectors") or [], dtype=float)
    if cell.shape != (3, 3):
        return "Structural review requires three complete cell vectors."
    if not cell_units:
        system = parsed.get("namelists", {}).get("system") or {}
        try:
            alat_angstrom = _alat_angstrom(parsed)
        except ValueError as error:
            return str(error)
        if alat_angstrom is not None:
            return cell * alat_angstrom, "implicit_alat", alat_angstrom
        if "celldm(1)" in system or "a" in system:
            return (
                "CELL_PARAMETERS without units requires one unambiguous "
                "positive celldm(1) or A."
            )
        return cell * ANGSTROM_PER_BOHR, "implicit_bohr", None
    if cell_units == "bohr":
        cell = cell * ANGSTROM_PER_BOHR
    if cell_units == "alat":
        try:
            alat_angstrom = _alat_angstrom(parsed)
        except ValueError as error:
            return str(error)
        if alat_angstrom is None:
            return (
                "CELL_PARAMETERS alat requires celldm(1) or A to normalize "
                "the cell."
            )
        return cell * alat_angstrom, cell_units, alat_angstrom
    return cell, cell_units, _alat_angstrom(parsed)


def _bravais_cell(
    parsed: dict[str, Any],
    ibrav: int,
) -> tuple[np.ndarray, str, float | None] | str:
    parameters = _bravais_lattice_parameters(parsed, ibrav)
    if isinstance(parameters, str):
        return parameters
    system, parameter_style, alat_angstrom = parameters
    try:
        if ibrav == 1:
            vectors = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        elif ibrav == 2:
            vectors = ((-0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (-0.5, 0.5, 0.0))
        elif ibrav == 3:
            vectors = ((0.5, 0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, -0.5, 0.5))
        elif ibrav == -3:
            vectors = ((-0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5))
        elif ibrav == 4:
            c_over_a = _length_ratio(
                system,
                parameter_style,
                "celldm(3)",
                "c",
                alat_angstrom,
            )
            vectors = (
                (1.0, 0.0, 0.0),
                (-0.5, math.sqrt(3.0) / 2.0, 0.0),
                (0.0, 0.0, c_over_a),
            )
        elif ibrav in {5, -5}:
            cos_gamma = _cosine_parameter(
                system,
                parameter_style,
                "celldm(4)",
                "cosab",
            )
            tx = _sqrt_positive((1.0 - cos_gamma) / 2.0, "rhombohedral tx")
            ty = _sqrt_positive((1.0 - cos_gamma) / 6.0, "rhombohedral ty")
            tz = _sqrt_positive((1.0 + 2.0 * cos_gamma) / 3.0, "rhombohedral tz")
            if ibrav == 5:
                vectors = (
                    (tx, -ty, tz),
                    (0.0, 2.0 * ty, tz),
                    (-tx, -ty, tz),
                )
            else:
                u = tz - 2.0 * math.sqrt(2.0) * ty
                v = tz + math.sqrt(2.0) * ty
                scale = 1.0 / math.sqrt(3.0)
                vectors = (
                    (scale * u, scale * v, scale * v),
                    (scale * v, scale * u, scale * v),
                    (scale * v, scale * v, scale * u),
                )
        elif ibrav == 6:
            c_over_a = _length_ratio(
                system,
                parameter_style,
                "celldm(3)",
                "c",
                alat_angstrom,
            )
            vectors = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, c_over_a))
        elif ibrav == 7:
            c_over_a = _length_ratio(
                system,
                parameter_style,
                "celldm(3)",
                "c",
                alat_angstrom,
            )
            vectors = (
                (0.5, -0.5, c_over_a / 2.0),
                (0.5, 0.5, c_over_a / 2.0),
                (-0.5, -0.5, c_over_a / 2.0),
            )
        elif ibrav == 8:
            vectors = (
                (1.0, 0.0, 0.0),
                (
                    0.0,
                    _length_ratio(
                        system,
                        parameter_style,
                        "celldm(2)",
                        "b",
                        alat_angstrom,
                    ),
                    0.0,
                ),
                (
                    0.0,
                    0.0,
                    _length_ratio(
                        system,
                        parameter_style,
                        "celldm(3)",
                        "c",
                        alat_angstrom,
                    ),
                ),
            )
        elif ibrav in {9, -9, 91, 10, 11}:
            b_over_a = _length_ratio(
                system,
                parameter_style,
                "celldm(2)",
                "b",
                alat_angstrom,
            )
            c_over_a = _length_ratio(
                system,
                parameter_style,
                "celldm(3)",
                "c",
                alat_angstrom,
            )
            if ibrav == 9:
                vectors = (
                    (0.5, b_over_a / 2.0, 0.0),
                    (-0.5, b_over_a / 2.0, 0.0),
                    (0.0, 0.0, c_over_a),
                )
            elif ibrav == -9:
                vectors = (
                    (0.5, -b_over_a / 2.0, 0.0),
                    (0.5, b_over_a / 2.0, 0.0),
                    (0.0, 0.0, c_over_a),
                )
            elif ibrav == 91:
                vectors = (
                    (1.0, 0.0, 0.0),
                    (0.0, b_over_a / 2.0, -c_over_a / 2.0),
                    (0.0, b_over_a / 2.0, c_over_a / 2.0),
                )
            elif ibrav == 10:
                vectors = (
                    (0.5, 0.0, c_over_a / 2.0),
                    (0.5, b_over_a / 2.0, 0.0),
                    (0.0, b_over_a / 2.0, c_over_a / 2.0),
                )
            else:
                vectors = (
                    (0.5, b_over_a / 2.0, c_over_a / 2.0),
                    (-0.5, b_over_a / 2.0, c_over_a / 2.0),
                    (-0.5, -b_over_a / 2.0, c_over_a / 2.0),
                )
        elif ibrav == 12:
            b_over_a = _length_ratio(
                system,
                parameter_style,
                "celldm(2)",
                "b",
                alat_angstrom,
            )
            c_over_a = _length_ratio(
                system,
                parameter_style,
                "celldm(3)",
                "c",
                alat_angstrom,
            )
            cos_gamma = _cosine_parameter(
                system,
                parameter_style,
                "celldm(4)",
                "cosab",
            )
            sin_gamma = _sqrt_positive(
                1.0 - cos_gamma**2,
                "monoclinic gamma",
            )
            vectors = (
                (1.0, 0.0, 0.0),
                (b_over_a * cos_gamma, b_over_a * sin_gamma, 0.0),
                (0.0, 0.0, c_over_a),
            )
        elif ibrav == 13:
            b_over_a = _length_ratio(
                system,
                parameter_style,
                "celldm(2)",
                "b",
                alat_angstrom,
            )
            c_over_a = _length_ratio(
                system,
                parameter_style,
                "celldm(3)",
                "c",
                alat_angstrom,
            )
            cos_gamma = _cosine_parameter(
                system,
                parameter_style,
                "celldm(4)",
                "cosab",
            )
            sin_gamma = _sqrt_positive(
                1.0 - cos_gamma**2,
                "monoclinic gamma",
            )
            vectors = (
                (0.5, 0.0, -c_over_a / 2.0),
                (b_over_a * cos_gamma, b_over_a * sin_gamma, 0.0),
                (0.5, 0.0, c_over_a / 2.0),
            )
        elif ibrav == -13:
            b_over_a = _length_ratio(
                system,
                parameter_style,
                "celldm(2)",
                "b",
                alat_angstrom,
            )
            c_over_a = _length_ratio(
                system,
                parameter_style,
                "celldm(3)",
                "c",
                alat_angstrom,
            )
            cos_beta = _cosine_parameter(
                system,
                parameter_style,
                "celldm(5)",
                "cosac",
            )
            sin_beta = _sqrt_positive(
                1.0 - cos_beta**2,
                "monoclinic beta",
            )
            vectors = (
                (0.5, b_over_a / 2.0, 0.0),
                (-0.5, b_over_a / 2.0, 0.0),
                (c_over_a * cos_beta, 0.0, c_over_a * sin_beta),
            )
        elif ibrav == -12:
            b_over_a = _length_ratio(
                system,
                parameter_style,
                "celldm(2)",
                "b",
                alat_angstrom,
            )
            c_over_a = _length_ratio(
                system,
                parameter_style,
                "celldm(3)",
                "c",
                alat_angstrom,
            )
            cos_beta = _cosine_parameter(
                system,
                parameter_style,
                "celldm(5)",
                "cosac",
            )
            vectors = (
                (1.0, 0.0, 0.0),
                (0.0, b_over_a, 0.0),
                (
                    c_over_a * cos_beta,
                    0.0,
                    c_over_a * math.sqrt(1.0 - cos_beta**2),
                ),
            )
        elif ibrav == 14:
            b_over_a = _length_ratio(
                system,
                parameter_style,
                "celldm(2)",
                "b",
                alat_angstrom,
            )
            c_over_a = _length_ratio(
                system,
                parameter_style,
                "celldm(3)",
                "c",
                alat_angstrom,
            )
            cos_alpha = _cosine_parameter(
                system,
                parameter_style,
                "celldm(4)",
                "cosbc",
            )
            cos_beta = _cosine_parameter(
                system,
                parameter_style,
                "celldm(5)",
                "cosac",
            )
            cos_gamma = _cosine_parameter(
                system,
                parameter_style,
                "celldm(6)",
                "cosab",
            )
            sin_gamma = _sqrt_positive(
                1.0 - cos_gamma**2,
                "triclinic gamma",
            )
            z_radicand = (
                1.0
                + 2.0 * cos_alpha * cos_beta * cos_gamma
                - cos_alpha**2
                - cos_beta**2
                - cos_gamma**2
            )
            vectors = (
                (1.0, 0.0, 0.0),
                (b_over_a * cos_gamma, b_over_a * sin_gamma, 0.0),
                (
                    c_over_a * cos_beta,
                    c_over_a * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
                    c_over_a * _sqrt_positive(z_radicand, "triclinic cell") / sin_gamma,
                ),
            )
        else:
            return (
                f"Structural review does not yet normalize ibrav={ibrav}."
            )
    except ValueError as error:
        return str(error)
    return (
        np.asarray(vectors, dtype=float) * alat_angstrom,
        "qe_bravais",
        alat_angstrom,
    )


def _bravais_lattice_parameters(
    parsed: dict[str, Any],
    ibrav: int,
) -> tuple[dict[str, Any], str, float] | str:
    system = parsed.get("namelists", {}).get("system") or {}
    has_celldm = any(key.startswith("celldm(") for key in system)
    conventional_keys = {"a", "b", "c", "cosab", "cosac", "cosbc"}
    has_conventional = bool(conventional_keys.intersection(system))
    if has_celldm and has_conventional:
        return (
            "Structural review cannot mix celldm parameters with A/B/C/cosine "
            "lattice parameters."
        )
    if has_celldm:
        try:
            alat_angstrom = (
                _positive_value(system, "celldm(1)") * ANGSTROM_PER_BOHR
            )
            return system, "celldm", alat_angstrom
        except ValueError as error:
            return str(error)
    if has_conventional:
        try:
            return system, "conventional", _positive_value(system, "a")
        except ValueError as error:
            return str(error)
    return f"Structural review requires celldm(1) or A for ibrav={ibrav}."


def _length_ratio(
    system: dict[str, Any],
    parameter_style: str,
    celldm_key: str,
    conventional_key: str,
    alat_angstrom: float,
) -> float:
    if parameter_style == "celldm":
        return _positive_value(system, celldm_key)
    return _positive_value(system, conventional_key) / alat_angstrom


def _cosine_parameter(
    system: dict[str, Any],
    parameter_style: str,
    celldm_key: str,
    conventional_key: str,
) -> float:
    key = celldm_key if parameter_style == "celldm" else conventional_key
    return _cosine_value(system, key)


def _sqrt_positive(value: float, context: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Structural review requires a non-degenerate {context}.")
    return math.sqrt(value)


def _alat_angstrom(parsed: dict[str, Any]) -> float | None:
    system = parsed.get("namelists", {}).get("system") or {}
    celldm = system.get("celldm(1)")
    a_angstrom = system.get("a")
    if celldm is not None and a_angstrom is not None:
        return None
    if celldm is not None:
        return _positive_value(system, "celldm(1)") * ANGSTROM_PER_BOHR
    if a_angstrom is not None:
        return _positive_value(system, "a")
    return None


def _positive_value(system: dict[str, Any], key: str) -> float:
    value = system.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Structural review requires numeric {key}.")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Structural review requires positive {key}.")
    return value


def _cosine_value(system: dict[str, Any], key: str) -> float:
    value = system.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Structural review requires numeric {key}.")
    value = float(value)
    if not math.isfinite(value) or not -1.0 < value < 1.0:
        raise ValueError(f"Structural review requires {key} between -1 and 1.")
    return value


def input_geometry_issues(
    parsed: dict[str, Any],
    analysis: dict[str, Any],
) -> list[LintIssue]:
    verdict = analysis.get("verdict") or {}
    if verdict.get("status") != "concerning":
        return []
    line = (parsed.get("card_lines") or {}).get("atomic_positions")
    issues = []
    for finding in verdict.get("findings") or []:
        close_contact = finding.get("code") == "initial_close_contact"
        issues.append({
            "level": "error" if close_contact else "warning",
            "message": f"Input geometry: {finding['message']}",
            "line": line,
            "suggested_fix": (
                "Check the coordinate units and separate overlapping atoms."
                if close_contact
                else "Confirm the intended bonding and starting coordinates."
            ),
        })
    return issues


def _not_assessed(reason: str) -> dict[str, Any]:
    return {
        "schema": "qe-input-structural-analysis/1",
        "scope": "not_assessed",
        "verdict": {
            "status": "not_assessed",
            "reasons": [reason],
        },
    }


__all__ = [
    "analyze_pw_input_geometry",
    "input_geometry_issues",
    "normalize_pw_input_geometry",
]

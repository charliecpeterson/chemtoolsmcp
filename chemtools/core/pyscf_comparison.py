"""Compare one bounded PySCF result with caller-declared reference evidence.

The report preserves settings and field evidence without treating an energy
difference as a correctness verdict.
"""

from __future__ import annotations

import math
from typing import Any

from chemtools.core.common import ELEMENT_TO_Z
from chemtools.core.cube import compare_cube_densities, compare_cube_orbitals
from chemtools.core.units import HARTREE_TO_KCAL_PER_MOL
PYSCF_SINGLE_POINT_RESULT_SCHEMA = "chemtools.pyscf-single-point-result/1"


PYSCF_REFERENCE_COMPARISON_SCHEMA = "chemtools.pyscf-reference-comparison/1"
_GEOMETRY_TOLERANCE_ANGSTROM = 1e-6


def compare_pyscf_reference_calculation(
    pyscf_result: dict[str, Any],
    reference: dict[str, Any],
    *,
    pyscf_orbital_cube: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Report explicit PySCF/reference evidence without selecting a winner."""
    pyscf = _pyscf_record(pyscf_result)
    reference_record = _reference_record(reference)
    orbital_cube = _orbital_cube(
        pyscf_orbital_cube,
        "pyscf_orbital_cube",
    )

    matching = {
        "geometry": _compare_geometry(
            pyscf["geometry"],
            reference_record["geometry"],
        ),
        "calculation": {
            field_name: _compare_value(
                pyscf["calculation"][field_name],
                reference_record["calculation"][field_name],
            )
            for field_name in (
                "method",
                "basis",
                "xc",
                "density_fit",
                "charge",
                "multiplicity",
            )
        },
        "electrons": _compare_value(
            pyscf["electrons"]["total"],
            reference_record["electrons"]["total"],
        ),
        "scf_converged": _compare_value(
            pyscf["scf"]["converged"],
            reference_record["scf"]["converged"],
        ),
    }
    delta_hartree = (
        pyscf["energy"]["total_hartree"]
        - reference_record["energy"]["total_hartree"]
    )
    field_comparisons = {
        "density": _compare_density(
            pyscf.get("density_cube"),
            reference_record.get("density_cube"),
        ),
        "orbital": _compare_orbital(
            orbital_cube,
            reference_record.get("orbital_cube"),
        ),
    }
    mismatched_settings = [
        field_name
        for field_name, comparison in matching["calculation"].items()
        if comparison["status"] == "different"
    ]
    if matching["geometry"]["status"] == "different":
        mismatched_settings.insert(0, "geometry")
    if matching["electrons"]["status"] == "different":
        mismatched_settings.append("electrons.total")
    if matching["scf_converged"]["status"] == "different":
        mismatched_settings.append("scf_converged")
    return {
        "schema_version": PYSCF_REFERENCE_COMPARISON_SCHEMA,
        "status": "compared",
        "conclusion": "evidence_only_no_correctness_verdict",
        "reference": {"label": reference_record["label"]},
        "pyscf": {
            "provenance": pyscf["provenance"],
            "energy_hartree": pyscf["energy"]["total_hartree"],
        },
        "matching": matching,
        "mismatched_settings": mismatched_settings,
        "energy": {
            "reference_total_hartree": reference_record["energy"]["total_hartree"],
            "pyscf_total_hartree": pyscf["energy"]["total_hartree"],
            "pyscf_minus_reference_hartree": delta_hartree,
            "pyscf_minus_reference_kcal_per_mol": (
                delta_hartree * HARTREE_TO_KCAL_PER_MOL
            ),
        },
        "field_comparisons": field_comparisons,
    }


def _pyscf_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("pyscf_result must be an object")
    if value.get("schema_version") != PYSCF_SINGLE_POINT_RESULT_SCHEMA:
        raise ValueError("pyscf_result must be a PySCF single-point result")
    if value.get("status") != "completed":
        raise ValueError("pyscf_result must have completed status")
    required = {
        "calculation",
        "geometry",
        "provenance",
        "scf",
        "energy",
        "electrons",
    }
    if not required <= set(value):
        raise ValueError("pyscf_result is missing comparison evidence")
    calculation = _calculation(value["calculation"], "pyscf_result.calculation")
    geometry = _geometry(value["geometry"], "pyscf_result.geometry")
    provenance = value["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("pyscf_result.provenance must be an object")
    scf = _scf(value["scf"], "pyscf_result.scf")
    energy = _energy(value["energy"], "pyscf_result.energy")
    electrons = _electrons(value["electrons"], "pyscf_result.electrons")
    density_cube = _density_cube(value.get("density_cube"), "pyscf_result.density_cube")
    return {
        "calculation": calculation,
        "geometry": geometry,
        "provenance": dict(provenance),
        "scf": scf,
        "energy": energy,
        "electrons": electrons,
        "density_cube": density_cube,
    }


def _reference_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("reference must be an object")
    required = {"label", "geometry", "calculation", "scf", "energy", "electrons"}
    allowed = required | {"density_cube", "orbital_cube"}
    if not required <= set(value) or not set(value) <= allowed:
        raise ValueError("reference contains unsupported or missing fields")
    label = value["label"]
    if not isinstance(label, str) or not label.strip():
        raise ValueError("reference.label must be a non-empty string")
    return {
        "label": label.strip(),
        "geometry": _geometry(value["geometry"], "reference.geometry"),
        "calculation": _calculation(value["calculation"], "reference.calculation"),
        "scf": _scf(value["scf"], "reference.scf"),
        "energy": _energy(value["energy"], "reference.energy"),
        "electrons": _electrons(value["electrons"], "reference.electrons"),
        "density_cube": _density_cube(value.get("density_cube"), "reference.density_cube"),
        "orbital_cube": _orbital_cube(value.get("orbital_cube"), "reference.orbital_cube"),
    }


def _calculation(value: Any, field_name: str) -> dict[str, Any]:
    required = {"method", "basis", "xc", "density_fit", "charge", "multiplicity"}
    allowed = required | {"atom_count"}
    if (
        not isinstance(value, dict)
        or not required <= set(value) <= allowed
    ):
        raise ValueError(f"{field_name} must contain exactly: {', '.join(sorted(required))}")
    if not isinstance(value["method"], str) or not value["method"].strip():
        raise ValueError(f"{field_name}.method must be a non-empty string")
    if not isinstance(value["basis"], str) or not value["basis"].strip():
        raise ValueError(f"{field_name}.basis must be a non-empty string")
    if value["xc"] is not None and (
        not isinstance(value["xc"], str) or not value["xc"].strip()
    ):
        raise ValueError(f"{field_name}.xc must be a non-empty string or null")
    if not isinstance(value["density_fit"], bool):
        raise ValueError(f"{field_name}.density_fit must be a boolean")
    if isinstance(value["charge"], bool) or not isinstance(value["charge"], int):
        raise ValueError(f"{field_name}.charge must be an integer")
    if (
        isinstance(value["multiplicity"], bool)
        or not isinstance(value["multiplicity"], int)
        or value["multiplicity"] < 1
    ):
        raise ValueError(f"{field_name}.multiplicity must be a positive integer")
    if "atom_count" in value and (
        isinstance(value["atom_count"], bool)
        or not isinstance(value["atom_count"], int)
        or value["atom_count"] < 1
    ):
        raise ValueError(f"{field_name}.atom_count must be a positive integer")
    return {
        "method": value["method"].strip(),
        "basis": value["basis"].strip(),
        "xc": value["xc"].strip() if isinstance(value["xc"], str) else None,
        "density_fit": value["density_fit"],
        "charge": value["charge"],
        "multiplicity": value["multiplicity"],
    }


def _geometry(value: Any, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty atom list")
    atoms = []
    for index, atom in enumerate(value):
        if not isinstance(atom, dict) or set(atom) != {"element", "x", "y", "z"}:
            raise ValueError(f"{field_name}[{index}] must contain element, x, y, and z")
        element = atom["element"]
        canonical = element[:1].upper() + element[1:].lower() if isinstance(element, str) else ""
        if canonical not in ELEMENT_TO_Z:
            raise ValueError(f"{field_name}[{index}].element is not recognized")
        coordinates = []
        for axis in ("x", "y", "z"):
            coordinate = atom[axis]
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise ValueError(f"{field_name}[{index}].{axis} must be finite")
            coordinate = float(coordinate)
            if not math.isfinite(coordinate):
                raise ValueError(f"{field_name}[{index}].{axis} must be finite")
            coordinates.append(coordinate)
        atoms.append({
            "element": canonical,
            "x": coordinates[0],
            "y": coordinates[1],
            "z": coordinates[2],
        })
    return atoms


def _scf(value: Any, field_name: str) -> dict[str, bool]:
    if not isinstance(value, dict) or not isinstance(value.get("converged"), bool):
        raise ValueError(f"{field_name}.converged must be a boolean")
    return {"converged": value["converged"]}


def _energy(value: Any, field_name: str) -> dict[str, float]:
    energy = value.get("total_hartree") if isinstance(value, dict) else None
    if isinstance(energy, bool) or not isinstance(energy, (int, float)):
        raise ValueError(f"{field_name}.total_hartree must be finite")
    energy = float(energy)
    if not math.isfinite(energy):
        raise ValueError(f"{field_name}.total_hartree must be finite")
    return {"total_hartree": energy}


def _electrons(value: Any, field_name: str) -> dict[str, int]:
    total = value.get("total") if isinstance(value, dict) else None
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError(f"{field_name}.total must be a non-negative integer")
    return {"total": total}


def _density_cube(value: Any, field_name: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object or null")
    if "status" in value:
        if value.get("status") != "written":
            return None
        path = value.get("path")
        unit = value.get("density_value_unit")
    else:
        if set(value) != {"path", "density_value_unit"}:
            raise ValueError(f"{field_name} must contain path and density_value_unit")
        path = value["path"]
        unit = value["density_value_unit"]
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{field_name}.path must be a non-empty string")
    if unit not in {"electron_per_bohr3", "electron_per_angstrom3"}:
        raise ValueError(f"{field_name}.density_value_unit is not supported")
    return {"path": path, "density_value_unit": unit}


def _orbital_cube(value: Any, field_name: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must contain path and orbital_label")
    if "status" in value and value.get("status") != "written":
        return None
    if not {"path", "orbital_label"} <= set(value):
        raise ValueError(f"{field_name} must contain path and orbital_label")
    path = value["path"]
    label = value["orbital_label"]
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{field_name}.path must be a non-empty string")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{field_name}.orbital_label must be a non-empty string")
    return {"path": path, "orbital_label": label}


def _compare_geometry(
    pyscf_atoms: list[dict[str, Any]],
    reference_atoms: list[dict[str, Any]],
) -> dict[str, Any]:
    unmatched = list(reference_atoms)
    for atom in pyscf_atoms:
        for index, candidate in enumerate(unmatched):
            if atom["element"] == candidate["element"] and all(
                math.isclose(
                    atom[axis],
                    candidate[axis],
                    rel_tol=0.0,
                    abs_tol=_GEOMETRY_TOLERANCE_ANGSTROM,
                )
                for axis in ("x", "y", "z")
            ):
                unmatched.pop(index)
                break
        else:
            return {
                "status": "different",
                "pyscf_atom_count": len(pyscf_atoms),
                "reference_atom_count": len(reference_atoms),
                "coordinate_tolerance_angstrom": _GEOMETRY_TOLERANCE_ANGSTROM,
            }
    return {
        "status": "matched" if not unmatched else "different",
        "pyscf_atom_count": len(pyscf_atoms),
        "reference_atom_count": len(reference_atoms),
        "coordinate_tolerance_angstrom": _GEOMETRY_TOLERANCE_ANGSTROM,
    }


def _compare_value(pyscf_value: Any, reference_value: Any) -> dict[str, Any]:
    return {
        "status": "matched" if pyscf_value == reference_value else "different",
        "pyscf": pyscf_value,
        "reference": reference_value,
    }


def _compare_density(
    pyscf_cube: dict[str, str] | None,
    reference_cube: dict[str, str] | None,
) -> dict[str, Any]:
    if pyscf_cube is None or reference_cube is None:
        return {
            "status": "not_compared",
            "reason": "both_pyscf_and_reference_density_cubes_are_required",
        }
    return compare_cube_densities(
        reference_cube["path"],
        pyscf_cube["path"],
        reference_density_unit=reference_cube["density_value_unit"],
        candidate_density_unit=pyscf_cube["density_value_unit"],
    )


def _compare_orbital(
    pyscf_cube: dict[str, str] | None,
    reference_cube: dict[str, str] | None,
) -> dict[str, Any]:
    if pyscf_cube is None or reference_cube is None:
        return {
            "status": "not_compared",
            "reason": "both_pyscf_and_reference_orbital_cubes_are_required",
        }
    return compare_cube_orbitals(
        reference_cube["path"],
        pyscf_cube["path"],
        reference_orbital_label=reference_cube["orbital_label"],
        candidate_orbital_label=pyscf_cube["orbital_label"],
    )


__all__ = [
    "PYSCF_REFERENCE_COMPARISON_SCHEMA",
    "compare_pyscf_reference_calculation",
]

"""Phase- and rotation-invariant comparison of matched orbital CUBE subspaces."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from chemtools.core.cube import (
    _cube_descriptor,
    _field_compatibility,
    _infer_cube_kind,
    _load_comparison_cube,
    _orbital_label,
    _trapezoidal_weights,
)


MAX_SUBSPACE_ORBITALS = 8
_RANK_TOLERANCE = 1e-8
_ORBITAL_KINDS = frozenset({"orbital", "orbital_homo", "orbital_lumo"})


def compare_cube_orbital_subspaces(
    reference_orbitals: list[dict[str, str]],
    candidate_orbitals: list[dict[str, str]],
) -> dict[str, Any]:
    """Compare equal-size, caller-declared orbital subspaces."""
    reference_specs = _orbital_specs(reference_orbitals, "reference_orbitals")
    candidate_specs = _orbital_specs(candidate_orbitals, "candidate_orbitals")
    if len(reference_specs) != len(candidate_specs):
        raise ValueError(
            "reference_orbitals and candidate_orbitals must have the same length"
        )

    reference = [_load_orbital(spec) for spec in reference_specs]
    candidate = [_load_orbital(spec) for spec in candidate_specs]
    compatibility = _subspace_compatibility(reference, candidate)
    response = {
        "schema_version": "chemtools.cube-orbital-subspace-comparison/1",
        "status": "comparable" if compatibility["comparable"] else "not_comparable",
        "comparison_scope": "two_explicitly_declared_equal_dimension_orbital_subspaces",
        "reference": [_descriptor(item) for item in reference],
        "candidate": [_descriptor(item) for item in candidate],
        "compatibility": compatibility,
    }
    if not compatibility["comparable"]:
        return response

    weights = np.asarray(_trapezoidal_weights(reference[0]["parsed"]["grid_shape"]))
    voxel_volume = reference[0]["parsed"]["voxel_volume_angstrom3"]
    reference_values = np.stack([item["values"] for item in reference])
    candidate_values = np.stack([item["values"] for item in candidate])
    weighted_reference = reference_values * weights
    weighted_candidate = candidate_values * weights
    reference_gram = voxel_volume * weighted_reference @ reference_values.T
    candidate_gram = voxel_volume * weighted_candidate @ candidate_values.T
    cross_overlap = voxel_volume * weighted_reference @ candidate_values.T
    reference_whitener, reference_error = _inverse_square_root(reference_gram)
    candidate_whitener, candidate_error = _inverse_square_root(candidate_gram)
    if reference_error is not None or candidate_error is not None:
        response["status"] = "not_comparable"
        response["compatibility"]["comparable"] = False
        if reference_error is not None:
            response["compatibility"]["findings"].append({
                "code": "reference_subspace_rank_deficient",
                **reference_error,
            })
        if candidate_error is not None:
            response["compatibility"]["findings"].append({
                "code": "candidate_subspace_rank_deficient",
                **candidate_error,
            })
        return response

    principal_matrix = reference_whitener @ cross_overlap @ candidate_whitener
    singular_values = np.linalg.svd(principal_matrix, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    response["metrics"] = {
        "quadrature": "uniform_trapezoidal_grid",
        "reference_gram_matrix": reference_gram.tolist(),
        "candidate_gram_matrix": candidate_gram.tolist(),
        "cross_overlap_matrix": {
            "reference_orbital_labels": [item["orbital_label"] for item in reference],
            "candidate_orbital_labels": [item["orbital_label"] for item in candidate],
            "values": cross_overlap.tolist(),
        },
        "principal_overlap_singular_values": singular_values.tolist(),
        "principal_angles_degrees": [
            math.degrees(math.acos(value)) for value in singular_values
        ],
        "least_principal_overlap": float(singular_values[-1]),
        "projection_frobenius_distance": float(math.sqrt(
            math.fsum(1.0 - value * value for value in singular_values)
        )),
    }
    return response


def _orbital_specs(value: Any, field_name: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not 2 <= len(value) <= MAX_SUBSPACE_ORBITALS:
        raise ValueError(
            f"{field_name} must contain between 2 and {MAX_SUBSPACE_ORBITALS} orbitals"
        )
    specs = []
    labels = set()
    for index, orbital in enumerate(value):
        if not isinstance(orbital, dict) or set(orbital) != {"path", "orbital_label"}:
            raise ValueError(f"{field_name}[{index}] must contain path and orbital_label")
        path = orbital["path"]
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{field_name}[{index}].path must be a non-empty string")
        label = _orbital_label(
            orbital["orbital_label"],
            f"{field_name}[{index}].orbital_label",
        )
        if label in labels:
            raise ValueError(f"{field_name} orbital labels must be unique")
        labels.add(label)
        specs.append({"path": path, "orbital_label": label})
    return specs


def _load_orbital(specification: dict[str, str]) -> dict[str, Any]:
    parsed = _load_comparison_cube(specification["path"])
    return {
        "path": specification["path"],
        "orbital_label": specification["orbital_label"],
        "parsed": parsed,
        "kind": _infer_cube_kind(
            parsed["title"],
            parsed["comment"],
            Path(specification["path"]).name,
        ),
        "values": np.asarray(parsed["values"], dtype=float),
    }


def _subspace_compatibility(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    anchor = reference[0]
    findings = []
    warnings = []
    for side, members in (("reference", reference), ("candidate", candidate)):
        for index, member in enumerate(members):
            compatibility = _field_compatibility(
                anchor["parsed"],
                member["parsed"],
                anchor["kind"],
                member["kind"],
                expected_kinds=_ORBITAL_KINDS,
                field_label="orbital",
            )
            findings.extend(
                {"side": side, "orbital_label": member["orbital_label"], **finding}
                for finding in compatibility["findings"]
            )
            warnings.extend(
                {"side": side, "orbital_label": member["orbital_label"], **warning}
                for warning in compatibility["warnings"]
            )
    return {
        "comparable": not findings,
        "requires_exact_same_grid": True,
        "grid_tolerance_angstrom": 1e-8,
        "geometry_tolerance_angstrom": 1e-6,
        "findings": findings,
        "warnings": warnings,
    }


def _descriptor(item: dict[str, Any]) -> dict[str, Any]:
    return {
        **_cube_descriptor(item["parsed"], item["path"], item["kind"]),
        "orbital_label": item["orbital_label"],
    }


def _inverse_square_root(matrix: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    largest = float(eigenvalues[-1])
    threshold = largest * _RANK_TOLERANCE
    if largest <= 0.0 or float(eigenvalues[0]) <= threshold:
        return None, {
            "eigenvalues": eigenvalues.tolist(),
            "relative_rank_tolerance": _RANK_TOLERANCE,
            "absolute_threshold": threshold,
        }
    return (
        (eigenvectors * np.power(eigenvalues, -0.5)) @ eigenvectors.T,
        None,
    )


__all__ = ["MAX_SUBSPACE_ORBITALS", "compare_cube_orbital_subspaces"]

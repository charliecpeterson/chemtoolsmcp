from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
from typing import Any

from chemtools.core.common import normalize_path, parse_scientific_float, read_text, ATOMIC_SYMBOLS


BOHR_TO_ANGSTROM = 0.529177210903
MAX_DENSITY_COMPARISON_POINTS = 2_000_000
_GRID_TOLERANCE_ANGSTROM = 1e-8
_GEOMETRY_TOLERANCE_ANGSTROM = 1e-6
_DENSITY_UNITS = frozenset({
    "electron_per_bohr3",
    "electron_per_angstrom3",
})


def parse_cube_file(
    path: str,
    include_values: bool = False,
    *,
    max_points: int | None = None,
    strict_values: bool = False,
) -> dict[str, Any]:
    contents = read_text(path)
    return _parse_cube_text(
        path,
        contents,
        include_values=include_values,
        max_points=max_points,
        strict_values=strict_values,
    )


def summarize_cube_file(path: str, top_atoms: int = 5) -> dict[str, Any]:
    parsed = parse_cube_file(path, include_values=True)
    values: list[float] = parsed["values"]
    voxel_volume = parsed["voxel_volume_angstrom3"]
    abs_values = [abs(value) for value in values]
    max_abs_value = max(abs_values) if abs_values else 0.0

    signed_integral = sum(values) * voxel_volume
    abs_integral = sum(abs_values) * voxel_volume
    positive_integral = sum(value for value in values if value > 0.0) * voxel_volume
    negative_integral = sum(value for value in values if value < 0.0) * voxel_volume
    l2_norm = math.sqrt(sum(value * value for value in values) * voxel_volume)

    threshold = max_abs_value * 0.20 if max_abs_value > 0 else 0.0
    atom_accumulators: dict[int, dict[str, float]] = {
        atom["atom_index"]: {
            "positive_weight": 0.0,
            "negative_weight": 0.0,
            "absolute_weight": 0.0,
        }
        for atom in parsed["atoms"]
    }
    positive_center = [0.0, 0.0, 0.0]
    negative_center = [0.0, 0.0, 0.0]
    positive_weight_total = 0.0
    negative_weight_total = 0.0

    nx, ny, nz = parsed["grid_shape"]
    origin = parsed["origin_angstrom"]
    vx, vy, vz = parsed["voxel_vectors_angstrom"]

    for flat_index, value in enumerate(values):
        magnitude = abs(value)
        if magnitude < threshold or magnitude == 0.0:
            continue
        ix = flat_index // (ny * nz)
        rem = flat_index % (ny * nz)
        iy = rem // nz
        iz = rem % nz
        position = [
            origin[0] + ix * vx[0] + iy * vy[0] + iz * vz[0],
            origin[1] + ix * vx[1] + iy * vy[1] + iz * vz[1],
            origin[2] + ix * vx[2] + iy * vy[2] + iz * vz[2],
        ]
        nearest_atom = _nearest_atom(position, parsed["atoms"])
        if nearest_atom is not None:
            bucket = atom_accumulators[nearest_atom["atom_index"]]
            bucket["absolute_weight"] += magnitude
            if value > 0.0:
                bucket["positive_weight"] += value
            else:
                bucket["negative_weight"] += abs(value)

        if value > 0.0:
            positive_weight_total += value
            for axis in range(3):
                positive_center[axis] += position[axis] * value
        elif value < 0.0:
            abs_value = abs(value)
            negative_weight_total += abs_value
            for axis in range(3):
                negative_center[axis] += position[axis] * abs_value

    localized_atoms = []
    for atom in parsed["atoms"]:
        weights = atom_accumulators[atom["atom_index"]]
        localized_atoms.append(
            {
                "atom_index": atom["atom_index"],
                "element": atom["element"],
                "position_angstrom": atom["position_angstrom"],
                "positive_weight": weights["positive_weight"],
                "negative_weight": weights["negative_weight"],
                "absolute_weight": weights["absolute_weight"],
            }
        )

    localized_atoms.sort(key=lambda item: item["absolute_weight"], reverse=True)
    summary = {
        "metadata": parsed["metadata"],
        "title": parsed["title"],
        "comment": parsed["comment"],
        "dataset_kind": _infer_cube_kind(parsed["title"], parsed["comment"], Path(path).name),
        "atom_count": parsed["atom_count"],
        "grid_shape": parsed["grid_shape"],
        "voxel_volume_angstrom3": voxel_volume,
        "value_range": parsed["value_range"],
        "signed_integral": signed_integral,
        "absolute_integral": abs_integral,
        "positive_integral": positive_integral,
        "negative_integral": negative_integral,
        "l2_norm": l2_norm,
        "localization_threshold": threshold,
        "top_localized_atoms": localized_atoms[:top_atoms],
        "positive_lobe_center_angstrom": (
            [component / positive_weight_total for component in positive_center]
            if positive_weight_total > 0.0
            else None
        ),
        "negative_lobe_center_angstrom": (
            [component / negative_weight_total for component in negative_center]
            if negative_weight_total > 0.0
            else None
        ),
    }
    return summary


def compare_cube_densities(
    reference_path: str,
    candidate_path: str,
    *,
    reference_density_unit: str,
    candidate_density_unit: str,
) -> dict[str, Any]:
    """Compare two declared electron-density CUBE fields on one exact grid."""
    reference_unit = _density_unit(
        reference_density_unit,
        "reference_density_unit",
    )
    candidate_unit = _density_unit(
        candidate_density_unit,
        "candidate_density_unit",
    )
    reference = _load_comparison_cube(reference_path)
    candidate = _load_comparison_cube(candidate_path)
    reference_kind = _infer_cube_kind(
        reference["title"],
        reference["comment"],
        Path(reference_path).name,
    )
    candidate_kind = _infer_cube_kind(
        candidate["title"],
        candidate["comment"],
        Path(candidate_path).name,
    )
    compatibility = _field_compatibility(
        reference,
        candidate,
        reference_kind,
        candidate_kind,
        expected_kinds=frozenset({"density"}),
        field_label="density",
    )
    response = {
        "schema_version": "chemtools.cube-density-comparison/1",
        "status": (
            "comparable" if compatibility["comparable"] else "not_comparable"
        ),
        "reference": _density_descriptor(
            reference,
            reference_path,
            reference_unit,
            reference_kind,
        ),
        "candidate": _density_descriptor(
            candidate,
            candidate_path,
            candidate_unit,
            candidate_kind,
        ),
        "compatibility": compatibility,
    }
    if not compatibility["comparable"]:
        return response

    reference_values = _density_values_in_angstrom3(
        reference["values"],
        reference_unit,
    )
    candidate_values = _density_values_in_angstrom3(
        candidate["values"],
        candidate_unit,
    )
    voxel_volume = reference["voxel_volume_angstrom3"]
    weights = _trapezoidal_weights(reference["grid_shape"])
    differences = [
        candidate_value - reference_value
        for reference_value, candidate_value in zip(
            reference_values,
            candidate_values,
        )
    ]
    reference_electrons = math.fsum(
        value * weight
        for value, weight in zip(reference_values, weights)
    ) * voxel_volume
    candidate_electrons = math.fsum(
        value * weight
        for value, weight in zip(candidate_values, weights)
    ) * voxel_volume
    absolute_difference = math.fsum(
        abs(value) * weight
        for value, weight in zip(differences, weights)
    ) * voxel_volume
    squared_difference = math.fsum(
        value * value * weight
        for value, weight in zip(differences, weights)
    )
    weight_total = math.fsum(weights)
    average_density = (
        abs(reference_electrons) + abs(candidate_electrons)
    ) / 2.0
    response["metrics"] = {
        "quadrature": "uniform_trapezoidal_grid",
        "reference_integrated_electrons": reference_electrons,
        "candidate_integrated_electrons": candidate_electrons,
        "integrated_electron_difference": (
            candidate_electrons - reference_electrons
        ),
        "l1_difference_electrons": absolute_difference,
        "relative_l1_difference": (
            absolute_difference / average_density
            if average_density > 0.0
            else None
        ),
        "l2_difference_electron_per_angstrom_1p5": math.sqrt(
            squared_difference * voxel_volume
        ),
        "rms_density_difference_electron_per_angstrom3": math.sqrt(
            squared_difference / weight_total
        ),
        "max_abs_density_difference_electron_per_angstrom3": max(
            abs(value) for value in differences
        ),
    }
    return response


def compare_cube_orbitals(
    reference_path: str,
    candidate_path: str,
    *,
    reference_orbital_label: str,
    candidate_orbital_label: str,
) -> dict[str, Any]:
    """Compare one explicitly matched non-degenerate orbital CUBE pair."""
    reference_label = _orbital_label(
        reference_orbital_label,
        "reference_orbital_label",
    )
    candidate_label = _orbital_label(
        candidate_orbital_label,
        "candidate_orbital_label",
    )
    reference = _load_comparison_cube(reference_path)
    candidate = _load_comparison_cube(candidate_path)
    reference_kind = _infer_cube_kind(
        reference["title"],
        reference["comment"],
        Path(reference_path).name,
    )
    candidate_kind = _infer_cube_kind(
        candidate["title"],
        candidate["comment"],
        Path(candidate_path).name,
    )
    compatibility = _field_compatibility(
        reference,
        candidate,
        reference_kind,
        candidate_kind,
        expected_kinds=frozenset({
            "orbital",
            "orbital_homo",
            "orbital_lumo",
        }),
        field_label="orbital",
    )
    response = {
        "schema_version": "chemtools.cube-orbital-comparison/1",
        "status": (
            "comparable" if compatibility["comparable"] else "not_comparable"
        ),
        "comparison_scope": "one_explicitly_matched_nondegenerate_orbital",
        "reference": {
            **_cube_descriptor(reference, reference_path, reference_kind),
            "orbital_label": reference_label,
        },
        "candidate": {
            **_cube_descriptor(candidate, candidate_path, candidate_kind),
            "orbital_label": candidate_label,
        },
        "compatibility": compatibility,
    }
    if not compatibility["comparable"]:
        return response

    weights = _trapezoidal_weights(reference["grid_shape"])
    voxel_volume = reference["voxel_volume_angstrom3"]
    reference_norm = math.sqrt(math.fsum(
        value * value * weight
        for value, weight in zip(reference["values"], weights)
    ) * voxel_volume)
    candidate_norm = math.sqrt(math.fsum(
        value * value * weight
        for value, weight in zip(candidate["values"], weights)
    ) * voxel_volume)
    if reference_norm == 0.0 or candidate_norm == 0.0:
        response["status"] = "not_comparable"
        response["compatibility"]["comparable"] = False
        response["compatibility"]["findings"].append({
            "code": "orbital_grid_norm_zero",
            "reference_norm": reference_norm,
            "candidate_norm": candidate_norm,
        })
        return response
    signed_overlap = math.fsum(
        reference_value * candidate_value * weight
        for reference_value, candidate_value, weight in zip(
            reference["values"],
            candidate["values"],
            weights,
        )
    ) * voxel_volume / (reference_norm * candidate_norm)
    signed_overlap = max(-1.0, min(1.0, signed_overlap))
    phase_sign = 1 if signed_overlap >= 0.0 else -1
    phase_aligned_overlap = abs(signed_overlap)
    response["metrics"] = {
        "signed_normalized_overlap": signed_overlap,
        "phase_alignment": "same_sign" if phase_sign > 0 else "flip_candidate_sign",
        "phase_aligned_normalized_overlap": phase_aligned_overlap,
        "phase_aligned_l2_distance": math.sqrt(
            max(0.0, 2.0 - 2.0 * phase_aligned_overlap)
        ),
    }
    return response


def _parse_cube_text(
    path: str,
    contents: str,
    include_values: bool = False,
    *,
    max_points: int | None = None,
    strict_values: bool = False,
) -> dict[str, Any]:
    lines = contents.splitlines()
    if len(lines) < 6:
        raise ValueError(f"cube file too short: {path}")

    title = lines[0].rstrip()
    comment = lines[1].rstrip()
    line3 = lines[2].split()
    if len(line3) < 4:
        raise ValueError(f"invalid cube header line 3 in {path}")
    atom_count_signed = int(line3[0])
    atom_count = abs(atom_count_signed)
    origin_raw = [float(line3[1]), float(line3[2]), float(line3[3])]

    grid_shape: list[int] = []
    voxel_vectors_raw: list[list[float]] = []
    grid_signs: list[int] = []
    for axis in range(3):
        fields = lines[3 + axis].split()
        if len(fields) < 4:
            raise ValueError(f"invalid cube voxel line {axis + 4} in {path}")
        count_signed = int(fields[0])
        if count_signed == 0:
            raise ValueError(f"cube grid count must be non-zero in {path}")
        grid_signs.append(1 if count_signed > 0 else -1)
        grid_shape.append(abs(count_signed))
        voxel_vectors_raw.append([float(fields[1]), float(fields[2]), float(fields[3])])

    if len(set(grid_signs)) != 1:
        raise ValueError(f"cube grid coordinate units are mixed in {path}")
    units_are_bohr = grid_signs[0] > 0

    scale = BOHR_TO_ANGSTROM if units_are_bohr else 1.0
    origin = [component * scale for component in origin_raw]
    voxel_vectors = [
        [component * scale for component in vector]
        for vector in voxel_vectors_raw
    ]

    atoms = []
    atom_start = 6
    for idx in range(atom_count):
        fields = lines[atom_start + idx].split()
        if len(fields) < 5:
            raise ValueError(f"invalid cube atom line {atom_start + idx + 1} in {path}")
        atomic_number = int(fields[0])
        atoms.append(
            {
                "atom_index": idx + 1,
                "atomic_number": atomic_number,
                "element": ATOMIC_SYMBOLS.get(atomic_number, str(atomic_number)),
                "nuclear_charge": float(fields[1]),
                "position_angstrom": [
                    float(fields[2]) * scale,
                    float(fields[3]) * scale,
                    float(fields[4]) * scale,
                ],
            }
        )

    total_points = grid_shape[0] * grid_shape[1] * grid_shape[2]
    if max_points is not None and total_points > max_points:
        raise ValueError(
            f"cube grid exceeds the {max_points} point limit in {path}: "
            f"{total_points}"
        )
    values: list[float] = []
    value_count = 0
    for line in lines[atom_start + atom_count :]:
        for token in line.split():
            value = parse_scientific_float(token)
            if value is None:
                raise ValueError(f"invalid cube grid value {token!r} in {path}")
            if not math.isfinite(value):
                raise ValueError(f"non-finite cube grid value {token!r} in {path}")
            value_count += 1
            if value_count <= total_points:
                values.append(value)
    if len(values) < total_points:
        raise ValueError(f"cube grid truncated in {path}: expected {total_points}, found {len(values)}")
    if strict_values and value_count != total_points:
        raise ValueError(
            f"cube grid has {value_count} values but expected {total_points} in {path}"
        )

    voxel_volume = abs(_triple_product(voxel_vectors[0], voxel_vectors[1], voxel_vectors[2]))
    parsed = {
        "metadata": {
            "file": normalize_path(path),
            "program": "cube",
        },
        "title": title,
        "comment": comment,
        "source_coordinate_unit": "bohr" if units_are_bohr else "angstrom",
        "atom_count": atom_count,
        "atoms": atoms,
        "origin_angstrom": origin,
        "grid_shape": grid_shape,
        "voxel_vectors_angstrom": voxel_vectors,
        "voxel_volume_angstrom3": voxel_volume,
        "value_range": [min(values), max(values)] if values else [None, None],
        "values": values if include_values else None,
    }
    if include_values:
        return parsed
    parsed.pop("values", None)
    return parsed


def _density_unit(value: str, field_name: str) -> str:
    if value not in _DENSITY_UNITS:
        allowed = ", ".join(sorted(_DENSITY_UNITS))
        raise ValueError(f"{field_name} must be one of: {allowed}")
    return value


def _density_descriptor(
    parsed: dict[str, Any],
    path: str,
    density_unit: str,
    inferred_kind: str,
) -> dict[str, Any]:
    return {
        **_cube_descriptor(parsed, path, inferred_kind),
        "declared_density_unit": density_unit,
    }


def _cube_descriptor(
    parsed: dict[str, Any],
    path: str,
    inferred_kind: str,
) -> dict[str, Any]:
    return {
        "path": normalize_path(path),
        "sha256": _file_sha256(path),
        "inferred_kind": inferred_kind,
        "source_coordinate_unit": parsed["source_coordinate_unit"],
        "atom_count": parsed["atom_count"],
        "grid_shape": parsed["grid_shape"],
        "origin_angstrom": parsed["origin_angstrom"],
        "voxel_vectors_angstrom": parsed["voxel_vectors_angstrom"],
        "voxel_volume_angstrom3": parsed["voxel_volume_angstrom3"],
    }


def _load_comparison_cube(path: str) -> dict[str, Any]:
    return parse_cube_file(
        path,
        include_values=True,
        max_points=MAX_DENSITY_COMPARISON_POINTS,
        strict_values=True,
    )


def _orbital_label(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    label = value.strip()
    if len(label) > 200:
        raise ValueError(f"{field_name} must be 200 characters or fewer")
    return label


def _field_compatibility(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    reference_kind: str,
    candidate_kind: str,
    *,
    expected_kinds: frozenset[str],
    field_label: str,
) -> dict[str, Any]:
    findings = []
    warnings = []
    _add_field_kind_evidence(
        warnings,
        findings,
        "reference",
        reference_kind,
        expected_kinds,
        field_label,
    )
    _add_field_kind_evidence(
        warnings,
        findings,
        "candidate",
        candidate_kind,
        expected_kinds,
        field_label,
    )
    if reference["grid_shape"] != candidate["grid_shape"]:
        findings.append({
            "code": "grid_shape_mismatch",
            "reference": reference["grid_shape"],
            "candidate": candidate["grid_shape"],
        })
    if any(size < 2 for size in reference["grid_shape"]):
        findings.append({
            "code": "grid_axis_has_fewer_than_two_points",
            "observed": reference["grid_shape"],
        })
    if not _vectors_close(
        reference["origin_angstrom"],
        candidate["origin_angstrom"],
    ):
        findings.append({"code": "grid_origin_mismatch"})
    for axis, (reference_vector, candidate_vector) in enumerate(zip(
        reference["voxel_vectors_angstrom"],
        candidate["voxel_vectors_angstrom"],
    )):
        if not _vectors_close(reference_vector, candidate_vector):
            findings.append({"code": "grid_vector_mismatch", "axis": axis})
    geometry_pairs = _matching_geometry_atoms(reference["atoms"], candidate["atoms"])
    if geometry_pairs is None:
        findings.append({"code": "nuclear_geometry_mismatch"})
    elif any(
        not math.isclose(
            reference_atom["nuclear_charge"],
            candidate_atom["nuclear_charge"],
            rel_tol=0.0,
            abs_tol=1e-8,
        )
        for reference_atom, candidate_atom in geometry_pairs
    ):
        warnings.append({
            "code": "cube_nuclear_charge_header_difference",
            "message": (
                "CUBE nuclear-charge header values differ, but atomic numbers "
                "and positions match."
            ),
        })
    return {
        "comparable": not findings,
        "requires_exact_same_grid": True,
        "grid_tolerance_angstrom": _GRID_TOLERANCE_ANGSTROM,
        "geometry_tolerance_angstrom": _GEOMETRY_TOLERANCE_ANGSTROM,
        "findings": findings,
        "warnings": warnings,
    }


def _add_field_kind_evidence(
    warnings: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    side: str,
    observed_kind: str,
    expected_kinds: frozenset[str],
    field_label: str,
) -> None:
    side_title = side.capitalize()
    if observed_kind == "unknown":
        warnings.append({
            "code": f"{side}_{field_label}_kind_not_identified",
            "message": (
                f"{side_title} CUBE metadata does not identify the field as "
                f"{field_label}; the caller's declaration is required."
            ),
        })
    elif observed_kind not in expected_kinds:
        article = "an" if field_label[:1].lower() in "aeiou" else "a"
        findings.append({
            "code": f"{side}_not_identified_as_{field_label}",
            "message": (
                f"{side_title} CUBE header and filename do not identify "
                f"{article} {field_label} field."
            ),
            "observed": observed_kind,
        })


def _density_values_in_angstrom3(
    values: list[float],
    density_unit: str,
) -> list[float]:
    if density_unit == "electron_per_angstrom3":
        return values
    scale = BOHR_TO_ANGSTROM ** -3
    return [value * scale for value in values]


def _trapezoidal_weights(grid_shape: list[int]) -> list[float]:
    nx, ny, nz = grid_shape
    return [
        (0.5 if ix in {0, nx - 1} else 1.0)
        * (0.5 if iy in {0, ny - 1} else 1.0)
        * (0.5 if iz in {0, nz - 1} else 1.0)
        for ix in range(nx)
        for iy in range(ny)
        for iz in range(nz)
    ]


def _vectors_close(left: list[float], right: list[float]) -> bool:
    return all(
        math.isclose(
            left_value,
            right_value,
            rel_tol=0.0,
            abs_tol=_GRID_TOLERANCE_ANGSTROM,
        )
        for left_value, right_value in zip(left, right)
    )


def _matching_geometry_atoms(
    reference_atoms: list[dict[str, Any]],
    candidate_atoms: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]] | None:
    if len(reference_atoms) != len(candidate_atoms):
        return None
    unmatched = list(candidate_atoms)
    pairs = []
    for reference_atom in reference_atoms:
        for index, candidate_atom in enumerate(unmatched):
            if (
                reference_atom["atomic_number"] == candidate_atom["atomic_number"]
                and all(
                    math.isclose(
                        reference_coordinate,
                        candidate_coordinate,
                        rel_tol=0.0,
                        abs_tol=_GEOMETRY_TOLERANCE_ANGSTROM,
                    )
                    for reference_coordinate, candidate_coordinate in zip(
                        reference_atom["position_angstrom"],
                        candidate_atom["position_angstrom"],
                    )
                )
            ):
                pairs.append((reference_atom, unmatched.pop(index)))
                break
        else:
            return None
    return pairs


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_cube_kind(title: str, comment: str, filename: str) -> str:
    joined = re.sub(r"[_-]", " ", " ".join([title, comment, filename])).lower()
    if re.search(r"\bspin\b", joined) and re.search(r"\bdensity\b", joined):
        return "spin_density"
    if re.search(r"\bdensity\b", joined):
        return "density"
    if re.search(r"\bhomo\b", joined):
        return "orbital_homo"
    if re.search(r"\blumo\b", joined):
        return "orbital_lumo"
    if re.search(r"\b(?:mo|orbital)\b", joined):
        return "orbital"
    if re.search(r"\bpotential\b", joined):
        return "potential"
    return "unknown"


def _triple_product(a: list[float], b: list[float], c: list[float]) -> float:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _nearest_atom(position: list[float], atoms: list[dict[str, Any]]) -> dict[str, Any] | None:
    best_atom = None
    best_distance = None
    for atom in atoms:
        dx = position[0] - atom["position_angstrom"][0]
        dy = position[1] - atom["position_angstrom"][1]
        dz = position[2] - atom["position_angstrom"][2]
        distance2 = dx * dx + dy * dy + dz * dz
        if best_distance is None or distance2 < best_distance:
            best_distance = distance2
            best_atom = atom
    return best_atom

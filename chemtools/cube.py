from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .common import normalize_path, parse_scientific_float, read_text, ATOMIC_SYMBOLS


BOHR_TO_ANGSTROM = 0.529177210903


def parse_cube_file(path: str, include_values: bool = False) -> dict[str, Any]:
    contents = read_text(path)
    return _parse_cube_text(path, contents, include_values=include_values)


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


def _parse_cube_text(path: str, contents: str, include_values: bool = False) -> dict[str, Any]:
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
    units_are_bohr = True
    for axis in range(3):
        fields = lines[3 + axis].split()
        if len(fields) < 4:
            raise ValueError(f"invalid cube voxel line {axis + 4} in {path}")
        count_signed = int(fields[0])
        if axis == 0:
            units_are_bohr = count_signed > 0
        grid_shape.append(abs(count_signed))
        voxel_vectors_raw.append([float(fields[1]), float(fields[2]), float(fields[3])])

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
    values: list[float] = []
    for line in lines[atom_start + atom_count :]:
        for token in line.split():
            value = parse_scientific_float(token)
            if value is None:
                raise ValueError(f"invalid cube grid value {token!r} in {path}")
            if len(values) < total_points:
                values.append(value)
    if len(values) < total_points:
        raise ValueError(f"cube grid truncated in {path}: expected {total_points}, found {len(values)}")
    values = values[:total_points]

    voxel_volume = abs(_triple_product(voxel_vectors[0], voxel_vectors[1], voxel_vectors[2]))
    parsed = {
        "metadata": {
            "file": normalize_path(path),
            "program": "cube",
        },
        "title": title,
        "comment": comment,
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


def _infer_cube_kind(title: str, comment: str, filename: str) -> str:
    joined = " ".join([title, comment, filename]).lower()
    if "spin" in joined and "density" in joined:
        return "spin_density"
    if "density" in joined:
        return "density"
    if "homo" in joined:
        return "orbital_homo"
    if "lumo" in joined:
        return "orbital_lumo"
    if "mo" in joined or "orbital" in joined:
        return "orbital"
    if "potential" in joined:
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

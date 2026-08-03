"""Parse PWSCF coordinate records and normalize them to angstroms.

Geometry selection and trajectory assembly live in neighboring modules. This
module owns the shared card syntax, unit conversion, and runtime-site parsing.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

from chemtools.core.common import parse_scientific_float
from chemtools.core.types import GeometryAtom
from chemtools.core.units import ANGSTROM_PER_BOHR
from chemtools.programs.qe._elements import element_from_label


_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_ALAT_RE = re.compile(
    rf"lattice parameter \(alat\)\s*=\s*({_FLOAT})\s+a\.u\.",
    re.I,
)
_AXIS_RE = re.compile(
    rf"^\s*a\(\s*([123])\s*\)\s*=\s*\(\s*({_FLOAT})\s+"
    rf"({_FLOAT})\s+({_FLOAT})\s*\)",
    re.I,
)
_SITE_RE = re.compile(
    rf"^\s*\d+\s+(\S+)\s+tau\(\s*\d+\s*\)\s*=\s*\(\s*"
    rf"({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*\)",
    re.I,
)
_ATOM_COUNT_RE = re.compile(r"number of atoms/cell\s*=\s*(\d+)", re.I)


def parse_final_coordinates(
    lines: list[str], start_index: int
) -> dict[str, Any]:
    end_index = next(
        (
            index
            for index in range(start_index + 1, len(lines))
            if lines[index].strip() == "End final coordinates"
        ),
        len(lines),
    )
    block = lines[start_index + 1 : end_index]
    coordinates: dict[str, Any] = {
        "start_line": start_index + 1,
        "end_line": end_index + 1 if end_index < len(lines) else None,
        "cell_parameters": None,
        "atomic_positions": None,
    }
    for offset, line in enumerate(block):
        stripped = line.strip()
        if stripped.upper().startswith("CELL_PARAMETERS"):
            rows = [_numeric_row(row) for row in block[offset + 1 : offset + 4]]
            if all(row is not None and len(row) >= 3 for row in rows):
                coordinates["cell_parameters"] = {
                    **_card_metadata(stripped),
                    "vectors": [row[:3] for row in rows if row is not None],
                    "line": start_index + offset + 2,
                }
        elif stripped.upper().startswith("ATOMIC_POSITIONS"):
            atoms: list[dict[str, Any]] = []
            for atom_line in block[offset + 1 :]:
                fields = atom_line.split()
                if len(fields) < 4:
                    break
                values = [_float_or_none(value) for value in fields[1:4]]
                if any(value is None for value in values):
                    break
                atom: dict[str, Any] = {
                    "label": fields[0],
                    "coordinates": values,
                }
                if (
                    len(fields) >= 7
                    and all(field in {"0", "1"} for field in fields[4:7])
                ):
                    atom["constraints"] = [int(field) for field in fields[4:7]]
                atoms.append(atom)
            coordinates["atomic_positions"] = {
                **_card_metadata(stripped),
                "atoms": atoms,
                "line": start_index + offset + 2,
            }
    return coordinates


def parse_cell_card(lines: list[str], header_index: int) -> dict[str, Any]:
    rows = [
        _numeric_row(line)
        for line in lines[header_index + 1 : header_index + 4]
    ]
    return {
        **_card_metadata(lines[header_index].strip()),
        "vectors": [row[:3] for row in rows if row is not None],
        "line": header_index + 1,
    }


def parse_positions_card(
    lines: list[str],
    header_index: int,
    atom_count: int,
) -> dict[str, Any]:
    atoms = []
    for line in lines[header_index + 1 : header_index + 1 + atom_count]:
        fields = line.split()
        if len(fields) < 4:
            break
        coordinates = [_float_or_none(value) for value in fields[1:4]]
        if any(value is None for value in coordinates):
            break
        atoms.append({
            "label": fields[0],
            "coordinates": coordinates,
        })
    return {
        **_card_metadata(lines[header_index].strip()),
        "atoms": atoms,
        "line": header_index + 1,
    }


def last_final_coordinates(lines: list[str]) -> dict[str, Any] | None:
    starts = [
        index
        for index, line in enumerate(lines)
        if line.strip() == "Begin final coordinates"
    ]
    return parse_final_coordinates(lines, starts[-1]) if starts else None


def normalize_final_geometry(
    lines: list[str], final: dict[str, Any]
) -> dict[str, Any]:
    initial = initial_runtime_geometry(lines)
    initial_cell = (
        initial.get("cell", {}).get("vectors_angstrom")
        if initial.get("status") == "available"
        else None
    )
    initial_alat = initial_alat_bohr(lines)
    cell = normalize_cell(
        final.get("cell_parameters"),
        fallback=initial_cell,
        fallback_alat_bohr=initial_alat,
    )
    positions = final.get("atomic_positions") or {}
    atoms = normalize_atoms(
        positions,
        cell=cell,
        fallback_alat_bohr=initial_alat,
    )
    if cell is None or not atoms:
        return {
            "status": "unavailable",
            "reason": (
                "The converged final-coordinate block could not be normalized "
                "to an angstrom cell and Cartesian positions."
            ),
        }
    return geometry_record(
        atoms,
        cell,
        role="converged_relaxed_structure",
        position_line=positions.get("line"),
        cell_line=(final.get("cell_parameters") or {}).get("line"),
    )


def initial_runtime_geometry(lines: list[str]) -> dict[str, Any]:
    alat_bohr = initial_alat_bohr(lines)
    atom_count = _first_integer(_ATOM_COUNT_RE, lines)
    axes = _crystal_axes(lines)
    sites, position_line = _alat_sites(lines, atom_count)
    if alat_bohr is None or axes is None or not sites:
        return {
            "status": "unavailable",
            "reason": (
                "The output does not contain a complete PWSCF runtime cell "
                "and Cartesian atomic-site table."
            ),
        }
    scale = alat_bohr * ANGSTROM_PER_BOHR
    cell = [[component * scale for component in vector] for vector in axes]
    atoms = [
        {
            "element": element_from_label(site["label"]),
            "x": site["coordinates"][0] * scale,
            "y": site["coordinates"][1] * scale,
            "z": site["coordinates"][2] * scale,
        }
        for site in sites
    ]
    if any(atom["element"] is None for atom in atoms):
        return {
            "status": "unavailable",
            "reason": "One or more PWSCF atomic labels could not be normalized.",
        }
    return geometry_record(
        atoms,
        cell,
        role="calculation_structure",
        position_line=position_line,
        cell_line=_first_axis_line(lines),
    )


def geometry_record(
    atoms: list[GeometryAtom],
    cell: list[list[float]],
    *,
    role: str,
    position_line: int | None,
    cell_line: int | None,
) -> dict[str, Any]:
    return {
        "status": "available",
        "role": role,
        "units": "angstrom",
        "atoms": atoms,
        "atom_count": len(atoms),
        "elements": dict(Counter(atom["element"] for atom in atoms)),
        "cell": {
            "vectors_angstrom": cell,
            "periodic": [True, True, True],
        },
        "source": {
            "position_line": position_line,
            "cell_line": cell_line,
        },
    }


def normalize_cell(
    cell_record: dict[str, Any] | None,
    *,
    fallback: list[list[float]] | None,
    fallback_alat_bohr: float | None,
) -> list[list[float]] | None:
    if cell_record is None:
        return fallback
    vectors = cell_record.get("vectors") or []
    if len(vectors) != 3:
        return None
    units = cell_record.get("units")
    if units == "angstrom":
        scale = 1.0
    elif units == "bohr":
        scale = ANGSTROM_PER_BOHR
    elif units == "alat":
        alat_bohr = cell_record.get("alat_bohr") or fallback_alat_bohr
        if alat_bohr is None:
            return None
        scale = alat_bohr * ANGSTROM_PER_BOHR
    else:
        return None
    return [[float(value) * scale for value in vector] for vector in vectors]


def normalize_atoms(
    positions: dict[str, Any],
    *,
    cell: list[list[float]] | None,
    fallback_alat_bohr: float | None,
) -> list[GeometryAtom]:
    units = positions.get("units")
    native_atoms = positions.get("atoms") or []
    atoms: list[GeometryAtom] = []
    for atom in native_atoms:
        element = element_from_label(str(atom.get("label") or ""))
        coordinates = atom.get("coordinates") or []
        if element is None or len(coordinates) != 3:
            return []
        if units == "angstrom":
            cartesian = [float(value) for value in coordinates]
        elif units == "bohr":
            cartesian = [
                float(value) * ANGSTROM_PER_BOHR for value in coordinates
            ]
        elif units == "alat" and fallback_alat_bohr is not None:
            scale = fallback_alat_bohr * ANGSTROM_PER_BOHR
            cartesian = [float(value) * scale for value in coordinates]
        elif units == "crystal" and cell is not None:
            cartesian = [
                sum(
                    float(coordinates[axis]) * cell[axis][component]
                    for axis in range(3)
                )
                for component in range(3)
            ]
        else:
            return []
        atoms.append({
            "element": element,
            "x": cartesian[0],
            "y": cartesian[1],
            "z": cartesian[2],
        })
    return atoms


def initial_alat_bohr(lines: list[str]) -> float | None:
    for line in lines:
        if match := _ALAT_RE.search(line):
            return _float_or_none(match.group(1))
    return None


def bfgs_converged(lines: list[str]) -> bool:
    return any("bfgs converged in" in line.lower() for line in lines)


def job_done(lines: list[str]) -> bool:
    return any(line.strip() == "JOB DONE." for line in lines)


def is_relaxation(lines: list[str]) -> bool:
    return any(
        marker in line.lower()
        for line in lines
        for marker in (
            "force convergence threshold",
            "press convergence thresh.",
            "bfgs geometry optimization",
        )
    )


def _crystal_axes(lines: list[str]) -> list[list[float]] | None:
    axes: dict[int, list[float]] = {}
    for line in lines:
        if match := _AXIS_RE.match(line):
            axes[int(match.group(1))] = [
                _float(match.group(2)),
                _float(match.group(3)),
                _float(match.group(4)),
            ]
            if len(axes) == 3:
                return [axes[index] for index in (1, 2, 3)]
    return None


def _alat_sites(
    lines: list[str], atom_count: int | None
) -> tuple[list[dict[str, Any]], int | None]:
    for index, line in enumerate(lines):
        if "positions (alat units)" not in line.lower():
            continue
        sites: list[dict[str, Any]] = []
        for candidate in lines[index + 1 :]:
            match = _SITE_RE.match(candidate)
            if match is None:
                if sites:
                    break
                continue
            sites.append({
                "label": match.group(1),
                "coordinates": [
                    _float(match.group(2)),
                    _float(match.group(3)),
                    _float(match.group(4)),
                ],
            })
            if atom_count is not None and len(sites) == atom_count:
                break
        if sites and (atom_count is None or len(sites) == atom_count):
            return sites, index + 1
    return [], None


def _card_metadata(header: str) -> dict[str, Any]:
    option_match = re.search(r"[({]\s*([^)}]+)\s*[)}]", header)
    option = option_match.group(1).strip() if option_match else ""
    unit_match = re.match(r"([A-Za-z_-]+)", option)
    units = unit_match.group(1).lower() if unit_match else None
    alat_match = re.search(rf"\balat\s*=\s*({_FLOAT})", option, re.I)
    metadata: dict[str, Any] = {"units": units}
    if alat_match:
        metadata["alat_bohr"] = _float(alat_match.group(1))
    return metadata


def _numeric_row(line: str) -> list[float] | None:
    fields = line.split()
    if not fields:
        return None
    values = [_float_or_none(field) for field in fields]
    if any(value is None for value in values):
        return None
    return [value for value in values if value is not None]


def _first_integer(pattern: re.Pattern[str], lines: list[str]) -> int | None:
    for line in lines:
        if match := pattern.search(line):
            return int(match.group(1))
    return None


def _first_axis_line(lines: list[str]) -> int | None:
    return next(
        (index for index, line in enumerate(lines, start=1) if _AXIS_RE.match(line)),
        None,
    )


def _float(value: str) -> float:
    parsed = _float_or_none(value)
    if parsed is None:
        raise ValueError(f"invalid Quantum ESPRESSO numeric value: {value!r}")
    return parsed


def _float_or_none(value: str) -> float | None:
    return parse_scientific_float(value)

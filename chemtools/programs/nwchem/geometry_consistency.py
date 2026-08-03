"""Compare NWChem input geometries with coordinate evidence from output.

Single-run and per-task checks share unit conversion and distance metrics here.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Mapping

from chemtools.core.units import ANGSTROM_PER_BOHR
from chemtools.programs.nwchem.parse.geometry import (
    GEOMETRY_DISTANCE_TOLERANCE_ANGSTROM,
)
from chemtools.programs.nwchem.parse.input import (
    extract_nwchem_geometry_block,
)


_CENTER_CHARGE_RE = re.compile(
    r"\bcharge\s+[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?\b",
    re.IGNORECASE,
)
_INPUT_GEOMETRY_UNITS_RE = re.compile(
    r"\bunits\s+(angstroms?|au|a\.u\.?|bohrs?)\b",
    re.IGNORECASE,
)


def compare_single_geometry(
    input_path: Path,
    parsed_input: Mapping[str, Any],
    output_geometry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if parsed_input.get("geometry_block_count") != 1:
        return _not_checked(
            "The input does not contain exactly one geometry block."
        )
    if output_geometry is None:
        return _not_checked(
            "The output contains no complete coordinate table."
        )
    try:
        input_geometry = _read_input_geometry(input_path, 0, None)
    except (OSError, ValueError) as exc:
        return _not_checked(
            f"The input geometry is not a Cartesian block: {exc}"
        )
    if input_geometry["source_units"] is None:
        return _not_checked(
            "The input geometry block does not declare its coordinate units."
        )

    input_atoms = input_geometry["atoms"]
    output_atoms = list(output_geometry.get("atoms") or [])
    input_elements = [atom["element"] for atom in input_atoms]
    output_elements = [atom.get("element") for atom in output_atoms]
    evidence = {
        "input": {
            "atom_count": len(input_atoms),
            "elements": input_elements,
            "source_units": input_geometry["source_units"],
        },
        "output": {
            "atom_count": len(output_atoms),
            "elements": output_elements,
            "source_units": output_geometry.get("source_units"),
        },
        "basis": (
            "Element order and all pair distances in the first complete "
            "output coordinate table."
        ),
    }
    if len(input_atoms) != len(output_atoms):
        return {
            "field": "geometry",
            "status": "mismatch",
            **evidence,
        }
    if any(element is None for element in output_elements):
        return _not_checked(
            "At least one output atom label could not be normalized.",
            input_elements,
            output_elements,
        )
    if input_elements != output_elements:
        return {
            "field": "geometry",
            "status": "mismatch",
            **evidence,
        }

    maximum_delta = _maximum_pair_distance_delta(input_atoms, output_atoms)
    return {
        "field": "geometry",
        "status": (
            "match"
            if maximum_delta <= GEOMETRY_DISTANCE_TOLERANCE_ANGSTROM
            else "mismatch"
        ),
        **evidence,
        "metrics": {
            "pair_count": len(input_atoms) * (len(input_atoms) - 1) // 2,
            "max_pair_distance_delta_angstrom": round(maximum_delta, 10),
            "tolerance_angstrom": GEOMETRY_DISTANCE_TOLERANCE_ANGSTROM,
        },
    }


def load_task_geometry(
    input_path: Path,
    geometry_spec: Mapping[str, Any],
) -> dict[str, Any] | None:
    block_index = geometry_spec.get("block_index")
    if not isinstance(block_index, int):
        return None
    try:
        return _read_input_geometry(
            input_path,
            block_index,
            geometry_spec.get("name"),
        )
    except (OSError, ValueError):
        return None


def compare_task_geometry(
    geometry_spec: Mapping[str, Any],
    input_geometry: Mapping[str, Any] | None,
    output_state: Mapping[str, Any],
    output_states: list[Mapping[str, Any]],
) -> dict[str, Any]:
    geometry_name = geometry_spec.get("name")
    output_geometry = _select_task_geometry(
        output_state,
        geometry_name,
        which="first",
    )
    source = geometry_spec.get("source")
    if source == "input":
        if input_geometry is None:
            return {
                "status": "not_checked",
                "reason": "The selected input geometry is unresolved.",
            }
        if input_geometry.get("atoms") is None:
            return {
                "status": "not_checked",
                "reason": (
                    "The selected input geometry does not declare its "
                    "coordinate units."
                ),
            }
        expected_geometry = input_geometry
        expected_source = "input_geometry"
    elif source == "task_result":
        source_index = geometry_spec.get("source_task_index")
        if not isinstance(source_index, int) or not (
            0 <= source_index < len(output_states)
        ):
            return {
                "status": "not_checked",
                "reason": "The prior geometry-producing task is unresolved.",
            }
        expected_geometry = _select_task_geometry(
            output_states[source_index],
            geometry_name,
            which="last",
        )
        expected_source = f"task_{source_index}_last_geometry"
    else:
        return {
            "status": "not_checked",
            "reason": "The task's active input geometry is unresolved.",
        }

    if expected_geometry is None or output_geometry is None:
        return {
            "status": "not_checked",
            "reason": (
                "The expected and observed task geometries were not both "
                "printed."
            ),
        }
    return _task_geometry_pair_comparison(
        expected_geometry,
        output_geometry,
        expected_source,
    )


def _read_input_geometry(
    input_path: Path,
    block_index: int,
    name: Any,
) -> dict[str, Any]:
    geometry = extract_nwchem_geometry_block(
        str(input_path),
        block_index=block_index,
    )
    units = _input_geometry_units(geometry["header_line"])
    source_lines = input_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()
    geometry_lines = source_lines[
        geometry["start_line"]:geometry["end_line"]
    ]
    return {
        "name": name,
        "atom_count": geometry["atom_count"],
        "elements": [atom["element"] for atom in geometry["atoms"]],
        "source_units": units,
        "has_explicit_center_charges": any(
            _CENTER_CHARGE_RE.search(line.split("#", 1)[0])
            for line in geometry_lines
        ),
        "atoms": (
            [
                _input_atom_in_angstrom(atom, units)
                for atom in geometry["atoms"]
            ]
            if units is not None
            else None
        ),
    }


def _select_task_geometry(
    output_state: Mapping[str, Any],
    geometry_name: Any,
    *,
    which: str,
) -> Mapping[str, Any] | None:
    by_name = output_state.get(f"{which}_geometry_by_name") or {}
    if isinstance(geometry_name, str) and geometry_name in by_name:
        return by_name[geometry_name]
    if by_name:
        return None
    return output_state.get(f"{which}_geometry")


def _task_geometry_pair_comparison(
    expected_geometry: Mapping[str, Any],
    observed_geometry: Mapping[str, Any],
    expected_source: str,
) -> dict[str, Any]:
    expected_atoms = list(expected_geometry.get("atoms") or [])
    observed_atoms = list(observed_geometry.get("atoms") or [])
    expected_elements = [atom.get("element") for atom in expected_atoms]
    observed_elements = [atom.get("element") for atom in observed_atoms]
    comparison = {
        "input": {
            "source": expected_source,
            "name": expected_geometry.get("name"),
            "atom_count": len(expected_atoms),
            "elements": expected_elements,
        },
        "output": {
            "name": observed_geometry.get("name"),
            "atom_count": len(observed_atoms),
            "elements": observed_elements,
        },
    }
    if (
        not expected_atoms
        or not observed_atoms
        or None in expected_elements
        or None in observed_elements
    ):
        return {
            "status": "not_checked",
            "reason": "At least one task geometry is incomplete.",
            **comparison,
        }
    if (
        len(expected_atoms) != len(observed_atoms)
        or expected_elements != observed_elements
    ):
        return {
            "status": "mismatch",
            **comparison,
        }
    maximum_delta = _maximum_pair_distance_delta(
        expected_atoms,
        observed_atoms,
    )
    return {
        "status": (
            "match"
            if maximum_delta <= GEOMETRY_DISTANCE_TOLERANCE_ANGSTROM
            else "mismatch"
        ),
        **comparison,
        "metrics": {
            "pair_count": (
                len(expected_atoms) * (len(expected_atoms) - 1) // 2
            ),
            "max_pair_distance_delta_angstrom": round(maximum_delta, 10),
            "tolerance_angstrom": GEOMETRY_DISTANCE_TOLERANCE_ANGSTROM,
        },
    }


def _input_geometry_units(header_line: str) -> str | None:
    match = _INPUT_GEOMETRY_UNITS_RE.search(header_line)
    if match is None:
        return None
    unit = match.group(1).lower()
    return "angstrom" if unit.startswith("angstrom") else "bohr"


def _input_atom_in_angstrom(
    atom: Mapping[str, Any],
    source_units: str,
) -> dict[str, Any]:
    scale = ANGSTROM_PER_BOHR if source_units == "bohr" else 1.0
    return {
        "element": atom["element"],
        "x": float(atom["x"]) * scale,
        "y": float(atom["y"]) * scale,
        "z": float(atom["z"]) * scale,
    }


def _maximum_pair_distance_delta(
    expected_atoms: list[dict[str, Any]],
    observed_atoms: list[dict[str, Any]],
) -> float:
    maximum = 0.0
    for first in range(len(expected_atoms)):
        for second in range(first + 1, len(expected_atoms)):
            expected_distance = math.dist(
                _coordinates(expected_atoms[first]),
                _coordinates(expected_atoms[second]),
            )
            observed_distance = math.dist(
                _coordinates(observed_atoms[first]),
                _coordinates(observed_atoms[second]),
            )
            maximum = max(
                maximum,
                abs(expected_distance - observed_distance),
            )
    return maximum


def _coordinates(atom: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(atom["x"]),
        float(atom["y"]),
        float(atom["z"]),
    )


def _not_checked(
    reason: str,
    input_value: Any = None,
    output_value: Any = None,
) -> dict[str, Any]:
    check = {
        "field": "geometry",
        "status": "not_checked",
        "reason": reason,
    }
    if input_value is not None:
        check["input"] = input_value
    if output_value is not None:
        check["output"] = output_value
    return check


__all__ = [
    "compare_single_geometry",
    "compare_task_geometry",
    "load_task_geometry",
]

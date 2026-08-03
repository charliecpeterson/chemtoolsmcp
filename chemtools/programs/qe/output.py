"""Parse compact scientific evidence from Quantum ESPRESSO pw.x output.

The parser keeps converged SCF energies separate from ordinary iteration
energies so an interrupted run cannot promote an unconverged value.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from chemtools.core.common import parse_scientific_float
from chemtools.programs.qe.geometry import (
    parse_final_coordinates,
    parse_pw_geometry_text,
)


_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_BANNER_RE = re.compile(r"^\s*Program PWSCF v\.([^\s]+) starts on\b", re.I)
_INPUT_RE = re.compile(r"^\s*Reading input from\s+(.+?)\s*$", re.I)
_BANG_ENERGY_RE = re.compile(
    rf"^\s*!\s+total energy\s*=\s*({_FLOAT})\s+Ry\b", re.I
)
_ITER_ENERGY_RE = re.compile(
    rf"^\s+total energy\s*=\s*({_FLOAT})\s+Ry\b", re.I
)
_ACCURACY_RE = re.compile(
    rf"^\s*estimated scf accuracy\s*[<=>]+\s*({_FLOAT})\s+Ry\b", re.I
)
_ITERATION_RE = re.compile(r"^\s*iteration\s+#\s*(\d+)\b", re.I)
_SCF_CONVERGED_RE = re.compile(
    r"^\s*convergence has been achieved in\s+(\d+)\s+iterations?\b", re.I
)
_SCF_FAILED_RE = re.compile(
    r"^\s*convergence NOT achieved after\s+(\d+)\s+iterations?", re.I
)
_BFGS_RE = re.compile(
    r"^\s*bfgs converged in\s+(\d+)\s+scf cycles and\s+(\d+)\s+bfgs steps",
    re.I,
)
_FINAL_ENTHALPY_RE = re.compile(
    rf"^\s*Final enthalpy\s*=\s*({_FLOAT})\s+Ry\b", re.I
)
_TOTAL_FORCE_RE = re.compile(
    rf"^\s*Total force\s*=\s*({_FLOAT})\s+Total SCF correction\s*=\s*({_FLOAT})",
    re.I,
)
_STRESS_RE = re.compile(
    rf"^\s*total\s+stress\s+\(Ry/bohr\*\*3\)\s+\(kbar\)\s+P=\s*({_FLOAT})",
    re.I,
)
_ERROR_RE = re.compile(r"^\s*Error in routine\s+(\S+)\s+\(([^)]*)\):", re.I)


def parse_pw_output(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return parse_pw_output_text(
        source.read_text(encoding="utf-8", errors="replace")
    )


def parse_pw_output_text(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    parsed: dict[str, Any] = {
        "format": "qe-pw-output/1",
        "program_version": None,
        "input_file": None,
        "calculation_mode": "scf",
        "system": {},
        "scf_cycles": [],
        "last_converged_energy": None,
        "last_iterative_energy": None,
        "last_scf_accuracy": None,
        "scf_converged": False,
        "scf_nonconvergence": None,
        "relaxation_algorithm": None,
        "bfgs": None,
        "final_enthalpy": None,
        "last_total_force": None,
        "last_stress": None,
        "final_coordinates": None,
        "band_calculation": {"start_line": None, "end_line": None},
        "errors": [],
        "job_done": False,
        "job_done_line": None,
        "line_count": len(lines),
    }

    last_iteration: int | None = None
    last_iteration_line: int | None = None
    cycle_start_line: int | None = None
    error_records: dict[tuple[str, str, str], dict[str, Any]] = {}

    for index, line in enumerate(lines):
        line_number = index + 1
        stripped = line.strip()

        if match := _BANNER_RE.match(line):
            parsed["program_version"] = match.group(1)
        elif match := _INPUT_RE.match(line):
            parsed["input_file"] = match.group(1).strip()

        _parse_system_record(line, line_number, parsed["system"])

        if stripped == "Self-consistent Calculation":
            cycle_start_line = line_number
            last_iteration = None
            last_iteration_line = None
        if stripped == "Band Structure Calculation":
            parsed["band_calculation"]["start_line"] = line_number
        elif stripped == "End of band structure calculation":
            parsed["band_calculation"]["end_line"] = line_number
        if stripped == "BFGS Geometry Optimization":
            parsed["relaxation_algorithm"] = "bfgs"

        if match := _ITERATION_RE.match(line):
            last_iteration = int(match.group(1))
            last_iteration_line = line_number
        if match := _BANG_ENERGY_RE.match(line):
            parsed["last_converged_energy"] = _energy_record(
                match.group(1), line_number
            )
        elif match := _ITER_ENERGY_RE.match(line):
            parsed["last_iterative_energy"] = _energy_record(
                match.group(1), line_number
            )
        if match := _ACCURACY_RE.match(line):
            parsed["last_scf_accuracy"] = {
                "value_ry": _float(match.group(1)),
                "line": line_number,
            }

        if match := _SCF_CONVERGED_RE.match(line):
            cycle = {
                "start_line": cycle_start_line,
                "end_line": line_number,
                "outcome": "converged",
                "iterations": int(match.group(1)),
                "energy_ry": _value(parsed["last_converged_energy"], "value_ry"),
                "energy_hartree": _value(
                    parsed["last_converged_energy"], "value_hartree"
                ),
            }
            parsed["scf_cycles"].append(cycle)
            parsed["scf_converged"] = True
            parsed["scf_nonconvergence"] = None
        elif match := _SCF_FAILED_RE.match(line):
            failure = {
                "line": line_number,
                "iterations": int(match.group(1)),
                "last_iteration": last_iteration,
                "last_iteration_line": last_iteration_line,
                "last_iterative_energy_ry": _value(
                    parsed["last_iterative_energy"], "value_ry"
                ),
                "last_iterative_energy_hartree": _value(
                    parsed["last_iterative_energy"], "value_hartree"
                ),
            }
            parsed["scf_cycles"].append({
                "start_line": cycle_start_line,
                "end_line": line_number,
                "outcome": "not_converged",
                "iterations": int(match.group(1)),
                "energy_ry": failure["last_iterative_energy_ry"],
                "energy_hartree": failure["last_iterative_energy_hartree"],
            })
            parsed["scf_converged"] = False
            parsed["scf_nonconvergence"] = failure

        if match := _BFGS_RE.match(line):
            parsed["relaxation_algorithm"] = "bfgs"
            parsed["bfgs"] = {
                "converged": True,
                "scf_cycles": int(match.group(1)),
                "steps": int(match.group(2)),
                "line": line_number,
            }
        if match := _FINAL_ENTHALPY_RE.match(line):
            parsed["final_enthalpy"] = _energy_record(
                match.group(1), line_number
            )
        if match := _TOTAL_FORCE_RE.match(line):
            parsed["last_total_force"] = {
                "value_ry_per_bohr": _float(match.group(1)),
                "scf_correction_ry_per_bohr": _float(match.group(2)),
                "line": line_number,
            }
        if match := _STRESS_RE.match(line):
            parsed["last_stress"] = _parse_stress(
                lines, index, _float(match.group(1))
            )

        if stripped == "Begin final coordinates":
            parsed["final_coordinates"] = parse_final_coordinates(
                lines, index
            )

        if match := _ERROR_RE.match(line):
            message, message_line = _next_nonempty_line(lines, index + 1)
            key = (match.group(1), match.group(2).strip(), message)
            if key in error_records:
                error_records[key]["last_line"] = line_number
                error_records[key]["occurrences"] += 1
            else:
                error_records[key] = {
                    "routine": match.group(1),
                    "code": match.group(2).strip(),
                    "message": message,
                    "first_line": line_number,
                    "last_line": line_number,
                    "message_line": message_line,
                    "occurrences": 1,
                }

        if stripped == "JOB DONE.":
            parsed["job_done"] = True
            parsed["job_done_line"] = line_number

    parsed["errors"] = list(error_records.values())
    parsed["calculation_mode"] = _infer_calculation_mode(lines, parsed)
    parsed["geometry"] = parse_pw_geometry_text(text)
    return parsed


def _parse_system_record(
    line: str, line_number: int, system: dict[str, Any]
) -> None:
    patterns = (
        ("n_atoms", r"^\s*number of atoms/cell\s*=\s*(\d+)", int),
        ("n_atom_types", r"^\s*number of atomic types\s*=\s*(\d+)", int),
        ("n_electrons", rf"^\s*number of electrons\s*=\s*({_FLOAT})", _float),
        ("n_kohn_sham_states", r"^\s*number of Kohn-Sham states\s*=\s*(\d+)", int),
        ("ecutwfc_ry", rf"^\s*kinetic-energy cutoff\s*=\s*({_FLOAT})\s+Ry", _float),
        ("ecutrho_ry", rf"^\s*charge density cutoff\s*=\s*({_FLOAT})\s+Ry", _float),
        ("n_k_points", r"^\s*number of k points\s*=\s*(\d+)", int),
    )
    for key, pattern, conversion in patterns:
        if match := re.match(pattern, line, re.I):
            system[key] = {"value": conversion(match.group(1)), "line": line_number}
            return


def _infer_calculation_mode(lines: list[str], parsed: dict[str, Any]) -> str:
    if parsed["band_calculation"]["start_line"] is not None:
        return "bands_or_nscf"
    lower = "\n".join(lines).lower()
    if "press convergence thresh." in lower or (
        parsed["final_enthalpy"] is not None
        and parsed["final_coordinates"] is not None
        and parsed["final_coordinates"].get("cell_parameters") is not None
    ):
        return "vc-relax"
    if (
        "force convergence threshold" in lower
        or parsed["relaxation_algorithm"] is not None
        or parsed["bfgs"] is not None
    ):
        return "relax"
    return "scf"


def _parse_stress(
    lines: list[str], header_index: int, pressure_kbar: float
) -> dict[str, Any]:
    matrix_ry: list[list[float]] = []
    matrix_kbar: list[list[float]] = []
    for line in lines[header_index + 1 : header_index + 4]:
        values = [_float(token) for token in re.findall(_FLOAT, line)]
        if len(values) < 6:
            break
        matrix_ry.append(values[:3])
        matrix_kbar.append(values[3:6])
    return {
        "pressure_kbar": pressure_kbar,
        "matrix_ry_per_bohr3": matrix_ry if len(matrix_ry) == 3 else None,
        "matrix_kbar": matrix_kbar if len(matrix_kbar) == 3 else None,
        "line": header_index + 1,
    }


def _next_nonempty_line(lines: list[str], start_index: int) -> tuple[str, int | None]:
    for index in range(start_index, min(start_index + 4, len(lines))):
        message = lines[index].strip()
        if message and not set(message) <= {"-", "="}:
            return message, index + 1
    return "Unspecified pw.x runtime error.", None


def _energy_record(value: str, line: int) -> dict[str, Any]:
    value_ry = _float(value)
    return {
        "value_ry": value_ry,
        "value_hartree": value_ry / 2.0,
        "line": line,
    }


def _float(value: str) -> float:
    parsed = parse_scientific_float(value)
    if parsed is None:
        raise ValueError(f"invalid Quantum ESPRESSO numeric value: {value!r}")
    return parsed


def _value(record: dict[str, Any] | None, key: str) -> Any:
    return record.get(key) if record is not None else None


__all__ = ["parse_pw_output", "parse_pw_output_text"]

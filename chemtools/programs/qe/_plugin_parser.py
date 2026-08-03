"""Adapter exposing pw.x input and output parsing through the backend contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.core.types import GeometryAtom, ParsedRun, TaskSummary
from chemtools.programs.qe.charge_spin import inspect_charge_spin
from chemtools.programs.qe.input import parse_pw_input
from chemtools.programs.qe.input_geometry import analyze_pw_input_geometry
from chemtools.programs.qe.kpoints import inspect_k_points
from chemtools.programs.qe.phonon import is_ph_x_input, parse_ph_x_input
from chemtools.programs.qe.pw2qmcpack import (
    is_pw2qmcpack_input,
    parse_pw2qmcpack_input,
)
from chemtools.programs.qe.pw2qmcpack_output import (
    is_pw2qmcpack_output,
    parse_pw2qmcpack_output_text,
)
from chemtools.programs.qe.geometry import parse_pw_geometry
from chemtools.programs.qe.output import parse_pw_output_text
from chemtools.programs.qe.pseudopotentials import (
    inspect_input_pseudopotentials,
)
from chemtools.programs.qe.trajectory import (
    parse_pw_trajectory,
    parse_pw_trajectory_text,
)
from chemtools.programs.qe.trajectory_analysis import analyze_pw_trajectory


_AUTOMATIC_TRAJECTORY_ANALYSIS_LIMIT_BYTES = 16 * 1024 * 1024


class _QeParser:
    def parse_input(self, path: str) -> dict[str, Any]:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
            if is_ph_x_input(text):
                return parse_ph_x_input(path)
            if is_pw2qmcpack_input(text):
                return parse_pw2qmcpack_input(path)
        parsed = parse_pw_input(path)
        pseudo_review = inspect_input_pseudopotentials(
            path,
            parsed,
        )
        parsed["pseudopotential_review"] = pseudo_review
        parsed["charge_spin_review"] = inspect_charge_spin(
            parsed,
            pseudo_review,
        )
        parsed["k_point_review"] = inspect_k_points(parsed)
        parsed["geometry_analysis"] = analyze_pw_input_geometry(parsed)
        return parsed

    def parse_output(self, path: str) -> ParsedRun:
        source = Path(path).expanduser().resolve()
        text = source.read_text(encoding="utf-8", errors="replace")
        if is_pw2qmcpack_output(text):
            return _pw2qmcpack_run(source, text)
        native = parse_pw_output_text(text)
        tasks = [_task_summary(native)]
        derived = _derived(native)
        if native["calculation_mode"] in {"relax", "vc-relax"}:
            derived["qe:trajectory"] = _trajectory_evidence(source, text)
        return {
            "program": "qe",
            "program_version": native["program_version"],
            "file": str(source),
            "file_size_bytes": source.stat().st_size,
            "tasks": tasks,
            "primary_task_index": 0,
            "derived": derived,
            "diagnostics": _diagnostics(native, source),
            "diagnosis": {},
        }

    def task_index(self, path: str) -> list[TaskSummary]:
        return self.parse_output(path)["tasks"]

    def get_geometry(
        self, path: str, task_index: int | None = None
    ) -> list[GeometryAtom]:
        if is_pw2qmcpack_output(Path(path).read_text(encoding="utf-8", errors="replace")):
            raise ValueError("pw2qmcpack.x output does not contain a geometry")
        if task_index not in (None, 0):
            raise IndexError("QE pw.x output contains one summarized task")
        geometry = parse_pw_geometry(path)
        if geometry.get("status") != "available":
            raise ValueError(
                geometry.get("reason") or "No normalized pw.x geometry found"
            )
        return geometry["atoms"]

    def get_trajectory(
        self, path: str, task_index: int | None = None
    ) -> dict[str, Any]:
        if is_pw2qmcpack_output(Path(path).read_text(encoding="utf-8", errors="replace")):
            raise ValueError("pw2qmcpack.x output does not contain a trajectory")
        if task_index not in (None, 0):
            raise IndexError("QE pw.x output contains one summarized task")
        trajectory = parse_pw_trajectory(path)
        if trajectory.get("status") != "available":
            raise ValueError(
                trajectory.get("reason") or "No pw.x relaxation trajectory found"
            )
        trajectory["structural_analysis"] = analyze_pw_trajectory(trajectory)
        return trajectory


def _pw2qmcpack_run(source: Path, text: str) -> ParsedRun:
    native = parse_pw2qmcpack_output_text(text)
    products = native["hdf5_artifacts"]
    errors = native["errors"]
    diagnostics = [{
        "kind": "error",
        "message": error["message"],
        "line": error["line"],
        "file": str(source),
    } for error in errors] + [{
        "kind": "info",
        "message": f"pw2qmcpack reported creating {artifact['path']}.",
        "line": artifact["line"],
        "file": str(source),
    } for artifact in products]
    return {
        "program": "qe",
        "program_version": native["program_version"],
        "file": str(source),
        "file_size_bytes": source.stat().st_size,
        "tasks": [{
            "index": 0,
            "kind": "unknown",
            "name": "pw2qmcpack Conversion",
            "method": "pw2qmcpack",
            "basis": None,
            "energy_hartree": None,
            "line_range": (1, native["line_count"]),
            "outcome": (
                "failed"
                if errors
                else "success" if native["job_done"] and products else "unknown"
            ),
            "has_usable_data": bool(products),
            "selection_priority": 1,
        }],
        "primary_task_index": 0,
        "derived": {
            "qe:program": "pw2qmcpack",
            "qe:pw2qmcpack_hdf5_artifacts": products,
            "qe:pw2qmcpack_compute_seconds": native["compute_qmcpack"],
            "qe:job_done": native["job_done"],
            "qe:job_done_line": native["job_done_line"],
            **({"qe:pw2qmcpack_errors": errors} if errors else {}),
        },
        "diagnostics": diagnostics,
        "diagnosis": {},
    }


def _trajectory_evidence(source: Path, text: str) -> dict[str, Any]:
    size_bytes = source.stat().st_size
    if size_bytes > _AUTOMATIC_TRAJECTORY_ANALYSIS_LIMIT_BYTES:
        return {
            "status": "not_assessed",
            "reason": (
                f"The {size_bytes}-byte output exceeds the automatic "
                "trajectory-analysis limit of "
                f"{_AUTOMATIC_TRAJECTORY_ANALYSIS_LIMIT_BYTES} bytes."
            ),
        }
    trajectory = parse_pw_trajectory_text(text)
    if trajectory.get("status") != "available":
        return trajectory
    return {
        "status": "available",
        "optimization_status": trajectory["optimization_status"],
        "frame_count": trajectory["frame_count"],
        "geometry_role": trajectory["geometry_role"],
        "geometry_source": trajectory["geometry_source"],
        "warning_count": len(trajectory["warnings"]),
        "structural_analysis": analyze_pw_trajectory(trajectory),
    }


def _task_summary(native: dict[str, Any]) -> TaskSummary:
    mode = native["calculation_mode"]
    energy = native["last_converged_energy"]
    if mode == "vc-relax":
        kind, name, priority = "optimize", "PWSCF Variable-Cell Relaxation", 3
    elif mode == "relax":
        kind, name, priority = "optimize", "PWSCF Relaxation", 3
    elif mode == "bands_or_nscf":
        kind, name, priority = "property", "PWSCF Bands/NSCF", 1
    else:
        kind, name, priority = "energy", "PWSCF SCF", 1

    return {
        "index": 0,
        "kind": kind,
        "name": name,
        "method": "DFT/PWSCF",
        "basis": "plane waves",
        "energy_hartree": (
            energy["value_hartree"] if energy is not None else None
        ),
        "line_range": (
            (1, native["line_count"])
            if native["line_count"]
            else (0, 0)
        ),
        "outcome": _task_outcome(native),
        "has_usable_data": bool(
            energy is not None
            or native["final_enthalpy"] is not None
            or native["final_coordinates"] is not None
        ),
        "selection_priority": priority,
    }


def _task_outcome(native: dict[str, Any]) -> str:
    if native["errors"] or native["scf_nonconvergence"] is not None:
        return "failed"
    mode = native["calculation_mode"]
    if mode == "bands_or_nscf":
        complete = native["band_calculation"]["end_line"] is not None
        return "success" if complete and native["job_done"] else "incomplete"
    if mode in {"relax", "vc-relax"}:
        if native["relaxation_algorithm"] == "bfgs":
            if native["bfgs"] is not None and native["job_done"]:
                return "success"
            return "failed" if native["job_done"] else "incomplete"
        return "unknown" if native["job_done"] else "incomplete"
    if native["scf_converged"] and native["job_done"]:
        return "success"
    return "incomplete"


def _derived(native: dict[str, Any]) -> dict[str, Any]:
    derived: dict[str, Any] = {
        "n_tasks": 1,
        "qe:calculation_mode": native["calculation_mode"],
        "qe:job_done": native["job_done"],
        "qe:job_done_line": native["job_done_line"],
        "qe:input_file": native["input_file"],
        "qe:relaxation_algorithm": native["relaxation_algorithm"],
        "qe:system": native["system"],
        "qe:scf_cycles": native["scf_cycles"],
        "scf_converged": native["scf_converged"],
    }
    _copy_record(
        derived,
        "qe:last_iterative_energy",
        native["last_iterative_energy"],
    )
    if native["last_converged_energy"] is not None:
        energy = native["last_converged_energy"]
        derived["final_energy_hartree"] = energy["value_hartree"]
        derived["primary_energy_hartree"] = energy["value_hartree"]
        derived["qe:final_energy_ry"] = energy["value_ry"]
        derived["qe:final_energy_line"] = energy["line"]
    if native["last_scf_accuracy"] is not None:
        derived["qe:last_scf_accuracy_ry"] = native["last_scf_accuracy"][
            "value_ry"
        ]
        derived["qe:last_scf_accuracy_line"] = native["last_scf_accuracy"][
            "line"
        ]
    if native["scf_nonconvergence"] is not None:
        derived["qe:scf_nonconvergence"] = native["scf_nonconvergence"]
    if native["bfgs"] is not None:
        derived["qe:bfgs"] = native["bfgs"]
    if native["final_enthalpy"] is not None:
        enthalpy = native["final_enthalpy"]
        derived["qe:final_enthalpy_ry"] = enthalpy["value_ry"]
        derived["qe:final_enthalpy_hartree"] = enthalpy["value_hartree"]
        derived["qe:final_enthalpy_line"] = enthalpy["line"]
    if native["last_total_force"] is not None:
        derived["qe:last_total_force"] = native["last_total_force"]
    if native["last_stress"] is not None:
        derived["qe:last_stress"] = native["last_stress"]
    if native["final_coordinates"] is not None:
        derived["qe:final_coordinates_native"] = native["final_coordinates"]
    geometry = native["geometry"]
    if geometry.get("status") == "available":
        derived["qe:geometry"] = {
            "status": "available",
            "role": geometry["role"],
            "units": geometry["units"],
            "atom_count": geometry["atom_count"],
            "elements": geometry["elements"],
            "cell": geometry["cell"],
            "source": geometry["source"],
        }
    else:
        derived["qe:geometry"] = geometry
    if native["band_calculation"]["start_line"] is not None:
        derived["qe:band_calculation"] = native["band_calculation"]
    if native["errors"]:
        derived["qe:runtime_errors"] = native["errors"]
    return derived


def _copy_record(
    derived: dict[str, Any], prefix: str, record: dict[str, Any] | None
) -> None:
    if record is None:
        return
    derived[f"{prefix}_ry"] = record["value_ry"]
    derived[f"{prefix}_hartree"] = record["value_hartree"]
    derived[f"{prefix}_line"] = record["line"]


def _diagnostics(
    native: dict[str, Any], source: Path
) -> list[dict[str, Any]]:
    diagnostics = [
        {
            "kind": "error",
            "message": (
                f"pw.x {error['routine']} error: {error['message']} "
                f"({error['occurrences']} occurrence(s))"
            ),
            "line": error["first_line"],
            "file": str(source),
        }
        for error in native["errors"]
    ]
    if native["scf_nonconvergence"] is not None:
        failure = native["scf_nonconvergence"]
        diagnostics.append({
            "kind": "error",
            "message": (
                "SCF convergence was not achieved after "
                f"{failure['iterations']} iterations."
            ),
            "line": failure["line"],
            "file": str(source),
        })
    if not native["job_done"] and not native["errors"]:
        diagnostics.append({
            "kind": "warning",
            "message": "The output has no JOB DONE marker.",
            "line": None,
            "file": str(source),
        })
    if native["calculation_mode"] == "bands_or_nscf":
        diagnostics.append({
            "kind": "info",
            "message": (
                "The pw.x output markers do not distinguish a bands run "
                "from an NSCF run without the input."
            ),
            "line": native["band_calculation"]["start_line"],
            "file": str(source),
        })
    return diagnostics


QE_PARSER = _QeParser()
QE_INPUT_PARSER = QE_PARSER


__all__ = ["QE_INPUT_PARSER", "QE_PARSER"]

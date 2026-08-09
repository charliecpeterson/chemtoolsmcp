"""Expose the validated ORCA parser slice through the backend contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.core.types import GeometryAtom, ParsedRun, TaskSummary
from chemtools.programs._adapter_helpers import pick_primary
from chemtools.programs.orca.input import parse_orca_input
from chemtools.programs.orca.output import parse_orca_output_text


class _OrcaParser:
    def parse_input(self, path: str) -> dict[str, Any]:
        return parse_orca_input(path)

    def parse_output(self, path: str) -> ParsedRun:
        source = Path(path).expanduser().resolve()
        native = parse_orca_output_text(
            source.read_text(encoding="utf-8", errors="replace")
        )
        tasks = _task_summaries(native)
        final_energy = (
            native["multiscale"]["qmmm_energy_hartree"]
            if native["multiscale"] is not None
            and native["multiscale"]["qmmm_energy_hartree"] is not None
            else native["energies"][-1]["value_hartree"]
            if native["energies"]
            else None
        )
        derived: dict[str, Any] = {
            "n_tasks": len(tasks),
            "scf_converged": bool(native["scf_cycles"]),
            "orca:normal_termination": native["normal_termination"],
            "orca:normal_termination_line": native["normal_termination_line"],
            "orca:input_file": native["input_file"],
            "orca:simple_keywords": native["simple_keywords"],
            "orca:basis_sets": native["basis_sets"],
            "orca:auxiliary_basis_sets": native["auxiliary_basis_sets"],
            "orca:number_of_basis_functions": native[
                "number_of_basis_functions"
            ],
            "orca:number_of_electrons": native["number_of_electrons"],
            "orca:relativistic_method": native["relativistic_method"],
            "orca:initial_guess": native["initial_guess"],
            "orca:scf_cycles": native["scf_cycles"],
            "orca:scf_failures": native["scf_failures"],
            "orca:error_termination": native["error_termination"],
            "orca:warning_count": len(native["warnings"]),
            "orca:runtime_seconds": native["runtime_seconds"],
            "orca:wavefunction_type": native["wavefunction_type"],
            "orca:charge": native["charge"],
            "orca:multiplicity": native["multiplicity"],
        }
        if final_energy is not None:
            derived["final_energy_hartree"] = final_energy
            derived["primary_energy_hartree"] = final_energy
            derived["orca:final_energy_line"] = (
                native["multiscale"]["qmmm_energy_line"]
                if native["multiscale"] is not None
                and native["multiscale"]["qmmm_energy_line"] is not None
                else native["energies"][-1]["line"]
            )
        if native["optimization"]["started_line"] is not None:
            derived["orca:optimization"] = native["optimization"]
        if native["frequency"]["started_line"] is not None:
            derived["orca:frequencies_cm1"] = native["frequency"][
                "frequencies_cm1"
            ]
            derived["n_imaginary_modes"] = len(
                native["frequency"]["imaginary_frequencies_cm1"]
            )
            derived["orca:imaginary_frequencies_cm1"] = native[
                "frequency"
            ]["imaginary_frequencies_cm1"]
        if native["thermochemistry"] is not None:
            derived["orca:thermochemistry"] = native["thermochemistry"]
        if native["spin"]:
            derived["orca:spin"] = native["spin"]
        if native["ri_approximation"] is not None:
            derived["orca:ri_approximation"] = native["ri_approximation"]
        if native["coupled_cluster"]["converged_line"] is not None:
            derived["orca:coupled_cluster"] = native["coupled_cluster"]
        if native["multiscale"] is not None:
            derived["orca:multiscale"] = native["multiscale"]
        if native["casscf"] is not None:
            derived["orca:casscf"] = native["casscf"]
        if native["multireference_pt2"] is not None:
            derived["orca:multireference_pt2"] = native[
                "multireference_pt2"
            ]
        if native["mrci"] is not None:
            derived["orca:mrci"] = native["mrci"]
        if native["tddft"] is not None:
            derived["orca:tddft"] = native["tddft"]
        if native["eom_ccsd"] is not None:
            derived["orca:eom_ccsd"] = native["eom_ccsd"]
        if native["esd"] is not None:
            derived["orca:esd"] = native["esd"]

        return {
            "program": "orca",
            "program_version": native["program_version"],
            "file": str(source),
            "file_size_bytes": source.stat().st_size,
            "tasks": tasks,
            "primary_task_index": pick_primary(tasks),
            "derived": derived,
            "diagnostics": [
                {
                    "kind": "warning",
                    "message": warning["message"],
                    "line": warning["line"],
                    "file": str(source),
                }
                for warning in native["warnings"]
            ],
            "diagnosis": _completion_diagnosis(native, tasks),
        }

    def task_index(self, path: str) -> list[TaskSummary]:
        return self.parse_output(path)["tasks"]

    def get_geometry(
        self, path: str, task_index: int | None = None
    ) -> list[GeometryAtom]:
        native = _parse_file(path)
        geometry = native["geometry"]
        if geometry is None:
            raise ValueError("ORCA output does not contain Cartesian coordinates")
        return [
            {
                "element": atom["element"],
                "x": atom["x"],
                "y": atom["y"],
                "z": atom["z"],
            }
            for atom in geometry["atoms"]
        ]

    def get_frequency(
        self, path: str, task_index: int | None = None
    ) -> dict[str, Any]:
        native = _parse_file(path)
        frequency = native["frequency"]
        if frequency["started_line"] is None:
            raise ValueError("ORCA output does not contain vibrational frequencies")
        return {
            **frequency,
            "thermochemistry": native["thermochemistry"],
        }


def _parse_file(path: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    return parse_orca_output_text(
        source.read_text(encoding="utf-8", errors="replace")
    )


def _task_summaries(native: dict[str, Any]) -> list[TaskSummary]:
    normal = native["normal_termination"]
    failed = native["error_termination"] is not None
    last_line = native["normal_termination_line"] or native["line_count"]
    final_energy = (
        native["multiscale"]["qmmm_energy_hartree"]
        if native["multiscale"] is not None
        and native["multiscale"]["qmmm_energy_hartree"] is not None
        else native["energies"][-1]["value_hartree"]
        if native["energies"]
        else None
    )
    method = _method(native)
    basis = native["basis"]
    tasks: list[TaskSummary] = []
    optimization = native["optimization"]
    frequency = native["frequency"]

    if native["esd"] is not None:
        esd = native["esd"]
        finished = esd["finished_line"] is not None
        tasks.append({
            "index": 0,
            "kind": "property",
            "name": f"ORCA_ESD {esd['process']}",
            "method": method,
            "basis": basis,
            "energy_hartree": final_energy,
            "line_range": (esd["started_line"], last_line),
            "outcome": (
                "success" if normal and finished
                else "failed" if normal or failed
                else "incomplete"
            ),
            "has_usable_data": finished,
            "selection_priority": 2,
        })

    if optimization["started_line"] is not None:
        converged = optimization["converged_line"] is not None
        tasks.append({
            "index": len(tasks),
            "kind": "optimize",
            "name": "ORCA Geometry Optimization",
            "method": method,
            "basis": basis,
            "energy_hartree": final_energy,
            "line_range": (
                optimization["started_line"],
                (frequency["started_line"] - 1)
                if frequency["started_line"] is not None
                else last_line,
            ),
            "outcome": (
                "success" if normal and converged
                else "failed" if normal
                else "failed" if failed
                else "incomplete"
            ),
            "has_usable_data": final_energy is not None,
            "selection_priority": 3,
        })

    if frequency["started_line"] is not None:
        has_frequencies = bool(frequency["all_frequencies_cm1"])
        tasks.append({
            "index": len(tasks),
            "kind": "frequency",
            "name": "ORCA Vibrational Frequencies",
            "method": method,
            "basis": basis,
            "energy_hartree": final_energy,
            "line_range": (frequency["started_line"], last_line),
            "outcome": (
                "success" if normal and has_frequencies
                else "failed" if normal
                else "failed" if failed
                else "incomplete"
            ),
            "has_usable_data": has_frequencies,
            "selection_priority": 2,
        })

    if not tasks:
        calculation_converged = _single_point_converged(native)
        tasks.append({
            "index": 0,
            "kind": "energy",
            "name": "ORCA Single Point Energy",
            "method": method,
            "basis": basis,
            "energy_hartree": final_energy,
            "line_range": (1, last_line),
            "outcome": (
                "success"
                if normal and calculation_converged and final_energy is not None
                else "failed" if normal
                else "failed" if failed
                else "incomplete"
            ),
            "has_usable_data": final_energy is not None,
            "selection_priority": 1,
        })
    return tasks


def _method(native: dict[str, Any]) -> str | None:
    if native["multireference_pt2"] is not None:
        return native["multireference_pt2"]["method"]
    if native["mrci"] is not None:
        return "MRCI"
    if native["eom_ccsd"] is not None:
        return "EOM-CCSD"
    if native["tddft"] is not None:
        reference = (
            native["simple_keywords"][0]
            if native["simple_keywords"]
            else None
        )
        return f"TD-DFT/{reference}" if reference else "TD-DFT"
    if native["casscf"] is not None:
        electrons = native["casscf"]["active_electrons"]
        orbitals = native["casscf"]["active_orbitals"]
        if electrons is not None and orbitals is not None:
            return f"CASSCF({electrons},{orbitals})"
        return "CASSCF"
    return native["simple_keywords"][0] if native["simple_keywords"] else None


def _single_point_converged(native: dict[str, Any]) -> bool:
    return bool(native["scf_cycles"]) or any((
        native["casscf"] is not None
        and native["casscf"]["state_average_energy_hartree"] is not None,
        native["multireference_pt2"] is not None,
        native["mrci"] is not None,
        native["tddft"] is not None,
        native["eom_ccsd"] is not None
        and native["eom_ccsd"]["converged_line"] is not None,
    ))


def _completion_diagnosis(
    native: dict[str, Any], tasks: list[TaskSummary]
) -> dict[str, Any]:
    outcomes = [task["outcome"] for task in tasks]
    if native["error_termination"] is not None:
        module = native["error_termination"]["module"]
        return {
            "verdict": {
                "label": "failed",
                "confidence": 0.99,
                "reasons": [f"ORCA reported error termination in {module}."],
            },
            "next_actions": [],
        }
    if native["normal_termination"] and all(
        outcome == "success" for outcome in outcomes
    ):
        return {
            "verdict": {
                "label": "completed",
                "confidence": 0.99,
                "reasons": [
                    "ORCA printed normal termination and every recognized "
                    "operation completed."
                ],
            },
            "next_actions": [],
        }
    if native["normal_termination"]:
        return {
            "verdict": {
                "label": "failed",
                "confidence": 0.9,
                "reasons": [
                    "ORCA terminated normally, but a recognized operation lacks "
                    "its completion evidence."
                ],
            },
            "next_actions": [],
        }
    return {
        "verdict": {
            "label": "incomplete",
            "confidence": 0.9,
            "reasons": ["The output has no ORCA normal-termination marker."],
        },
        "next_actions": [],
    }


ORCA_PARSER = _OrcaParser()


__all__ = ["ORCA_PARSER"]

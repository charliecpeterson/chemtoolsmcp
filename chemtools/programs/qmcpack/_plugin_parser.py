"""Expose QMCPACK XML input parsing through the backend contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.core.types import ParsedRun, TaskSummary
from chemtools.programs.qmcpack.includes import inspect_xml_includes
from chemtools.programs.qmcpack.input import parse_qmcpack_input
from chemtools.programs.qmcpack.output import parse_qmcpack_output_text
from chemtools.programs.qmcpack.sidecars import inspect_hdf5_sidecars


class _QmcpackParser:
    def parse_input(self, path: str) -> dict[str, Any]:
        parsed = parse_qmcpack_input(path)
        parsed["include_review"] = inspect_xml_includes(path, parsed)
        parsed["hdf5_sidecar_review"] = inspect_hdf5_sidecars(
            path,
            parsed,
            parsed["include_review"],
        )
        return parsed

    def parse_output(self, path: str) -> ParsedRun:
        source = Path(path).expanduser().resolve()
        native = parse_qmcpack_output_text(
            source.read_text(encoding="utf-8", errors="replace")
        )
        completed = native["completion"]["success_marker"]
        tasks = _task_summaries(native)
        derived = {
            "n_tasks": len(tasks),
            "qmcpack:success_marker": completed,
            "qmcpack:success_marker_line": native["completion"]["line"],
            "qmcpack:completion_evidence": (
                "explicit_success_marker"
                if completed
                else (
                    "total_execution_time_only"
                    if native["total_execution_time_seconds"] is not None
                    else "none"
                )
            ),
            "qmcpack:total_execution_time_seconds": native[
                "total_execution_time_seconds"
            ],
            "qmcpack:last_total_execution_time_line": native[
                "last_total_execution_time_line"
            ],
            "qmcpack:warning_count": sum(
                warning["occurrences"] for warning in native["warnings"]
            ),
        }
        if native["project"] is not None:
            derived["qmcpack:project_id"] = native["project"]["id"]
            derived["qmcpack:project_line"] = native["project"]["line"]
        if native.get("project_labels"):
            derived["qmcpack:project_labels"] = native["project_labels"]
        if native.get("last_run"):
            derived["qmcpack:last_run_start_line"] = native["last_run"]["start_line"]
        if native.get("runtime_particle_sets"):
            derived["qmcpack:runtime_particle_sets"] = native[
                "runtime_particle_sets"
            ]
        minwalkers_warnings = native["minwalkers_threshold_warnings"]
        if minwalkers_warnings:
            derived["qmcpack:minwalkers_warning_count"] = sum(
                warning["occurrences"] for warning in minwalkers_warnings
            )
            derived["qmcpack:minwalkers_thresholds"] = [
                warning["threshold"] for warning in minwalkers_warnings
            ]
            preceding_weights = [
                warning["minimum_immediately_preceding_effective_weight"]
                for warning in minwalkers_warnings
                if warning["minimum_immediately_preceding_effective_weight"]
                is not None
            ]
            if preceding_weights:
                derived[
                    "qmcpack:minwalkers_minimum_preceding_effective_weight"
                ] = min(preceding_weights)
        input_parameter_corrections = native.get("input_parameter_corrections", [])
        if input_parameter_corrections:
            derived["qmcpack:input_parameter_correction_count"] = sum(
                correction["occurrences"]
                for correction in input_parameter_corrections
            )
            derived["qmcpack:input_parameter_corrections"] = (
                input_parameter_corrections
            )
        optimization_messages = native["optimization_messages"]
        if optimization_messages:
            derived["qmcpack:optimization_messages"] = optimization_messages
        for message in optimization_messages:
            if message["code"] == "cost_function_invalid":
                derived["qmcpack:cost_function_invalid_count"] = message[
                    "occurrences"
                ]
            if message["code"] == "reverting_to_old_parameters":
                derived["qmcpack:reverted_to_old_parameters"] = True
            if message["code"] == "effective_walkers_too_small":
                derived["qmcpack:effective_walkers_too_small_count"] = message[
                    "occurrences"
                ]
                derived[
                    "qmcpack:minimum_reported_effective_walkers"
                ] = message["minimum_reported_effective_walkers"]
            if message["code"] == "linear_optimization_failed_step":
                derived["qmcpack:linear_optimization_failed_step_count"] = message[
                    "occurrences"
                ]
                derived[
                    "qmcpack:largest_failed_linear_optimization_parameter_change"
                ] = message["largest_reported_parameter_change"]
        if (good_step := native["linear_optimization_steps"].get("good")) is not None:
            derived["qmcpack:linear_optimization_good_step_count"] = good_step[
                "occurrences"
            ]
            derived[
                "qmcpack:largest_good_linear_optimization_parameter_change"
            ] = good_step["largest_reported_parameter_change"]
        return {
            "program": "qmcpack",
            "program_version": native["program_version"],
            "file": str(source),
            "file_size_bytes": source.stat().st_size,
            "tasks": tasks,
            "primary_task_index": len(tasks) - 1 if tasks else None,
            "derived": derived,
            "diagnostics": [
                {
                    "kind": "warning",
                    "message": warning["message"],
                    "line": warning["line"],
                    "file": str(source),
                }
                for warning in native["warnings"][:20]
            ] + [
                {
                    "kind": "warning",
                    "message": message["message"],
                    "line": message["first_line"],
                    "file": str(source),
                }
                for message in optimization_messages
            ],
            "diagnosis": _qmcpack_diagnosis(
                input_parameter_corrections,
                optimization_messages,
            ),
        }

    def task_index(self, path: str) -> list[TaskSummary]:
        return self.parse_output(path)["tasks"]


def _qmcpack_diagnosis(
    input_parameter_corrections: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    by_code = {message["code"]: message for message in messages}
    reasons = []
    label = None
    confidence = None
    if input_parameter_corrections:
        corrections = "; ".join(
            (
                f"{correction['parameter']} "
                f"{correction['requested_value']:g} -> "
                f"{correction['corrected_value']:g} "
                f"in {correction.get('section', 'an unclassified section')} "
                f"({correction['occurrences']} occurrence(s))"
            )
            for correction in input_parameter_corrections
        )
        label = "input_parameter_auto_corrected"
        confidence = 0.98
        reasons.append("QMCPACK replaced invalid input values: " + corrections + ".")
    if (message := by_code.get("effective_walkers_too_small")) is not None:
        minimum = message["minimum_reported_effective_walkers"]
        minimum_text = (
            f"; the lowest reported value was {minimum:g}"
            if minimum is not None
            else ""
        )
        if label is None:
            label = "optimization_effective_walkers_too_small"
            confidence = 0.98
        reasons.append(
            "QMCPACK reported too few effective walkers "
            f"{message['occurrences']} time(s){minimum_text}."
        )
    if (message := by_code.get("cost_function_invalid")) is not None:
        if label is None:
            label = "optimization_cost_function_invalid"
            confidence = 0.95
        reasons.append(
            "QMCPACK reported an invalid optimization cost function "
            f"{message['occurrences']} time(s)."
        )
    if (message := by_code.get("reverting_to_old_parameters")) is not None:
        if label is None:
            label = "optimization_reverted_to_old_parameters"
            confidence = 0.9
        reasons.append(
            "QMCPACK reverted to older optimization parameters "
            f"{message['occurrences']} time(s)."
        )
    if label is None:
        return {}
    return {
        "verdict": {
            "label": label,
            "confidence": confidence,
            "reasons": reasons,
        },
        "next_actions": [],
    }


def _task_summaries(native: dict[str, Any]) -> list[TaskSummary]:
    completed = native["completion"]["success_marker"]
    sections = native["sections"]
    if not sections:
        return [{
            "index": 0,
            "kind": "unknown",
            "name": "QMCPACK run",
            "method": "QMCPACK",
            "basis": None,
            "energy_hartree": None,
            "line_range": (1, native["line_count"]),
            "outcome": "success" if completed else "incomplete",
            "has_usable_data": completed,
            "selection_priority": 1,
        }]
    return [
        {
            "index": index,
            "kind": "energy" if section["name"] in {"VMC", "DMC"} else "unknown",
            "name": section["name"],
            "method": section["name"],
            "basis": None,
            "energy_hartree": None,
            "line_range": (
                section["start_line"],
                section["end_line"] or native["line_count"],
            ),
            "outcome": (
                "success"
                if completed and section["execution_time_seconds"] is not None
                else "incomplete"
            ),
            "has_usable_data": section["execution_time_seconds"] is not None,
            "selection_priority": 2 if section["name"] == "DMC" else 1,
        }
        for index, section in enumerate(sections)
    ]


QMCPACK_PARSER = _QmcpackParser()


__all__ = ["QMCPACK_PARSER"]

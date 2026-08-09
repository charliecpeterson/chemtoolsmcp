"""Compare two parsed chemistry runs without overstating energy ordering.

The service reports arithmetic energy differences separately from evidence
that the calculations are scientifically comparable.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from chemtools.application.run_inspection import inspect_run
from chemtools.core.program import ProgramBackend
from chemtools.core.units import HARTREE_TO_KCAL_PER_MOL


RUN_COMPARISON_SCHEMA = "chemtools.compare-runs/1"
ENERGY_EQUALITY_TOLERANCE_HARTREE = 1.0e-8


def compare_runs(
    backend: ProgramBackend,
    reference_output_file: str | Path,
    candidate_output_file: str | Path,
    *,
    reference_input_file: str | Path | None = None,
    candidate_input_file: str | Path | None = None,
) -> dict[str, Any]:
    reference = inspect_run(
        backend,
        reference_output_file,
        resolved_by="explicit",
        artifact_files=(
            [reference_input_file] if reference_input_file is not None else []
        ),
    )
    candidate = inspect_run(
        backend,
        candidate_output_file,
        resolved_by="explicit",
        artifact_files=(
            [candidate_input_file] if candidate_input_file is not None else []
        ),
    )
    return compare_run_inspections(reference, candidate)


def compare_run_inspections(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    reference_program = _nested(reference, "program", "name")
    candidate_program = _nested(candidate, "program", "name")
    if reference_program != candidate_program:
        raise ValueError(
            "run comparison requires both inspections to use the same program"
        )

    reference_task = _primary_energy_task(reference)
    candidate_task = _primary_energy_task(candidate)
    checks = [
        _task_completion_check(reference_task, candidate_task),
        _value_check(
            "task_kind",
            _mapping_value(reference_task, "kind"),
            _mapping_value(candidate_task, "kind"),
        ),
        _value_check(
            "method",
            _mapping_value(reference_task, "method"),
            _mapping_value(candidate_task, "method"),
        ),
        _consistency_value_check(
            reference,
            candidate,
            "charge",
            reference_task,
            candidate_task,
        ),
        _composition_check(
            reference,
            candidate,
            reference_task,
            candidate_task,
        ),
        _consistency_value_check(
            reference,
            candidate,
            "xc_functional",
            reference_task,
            candidate_task,
        ),
        _basis_check(
            reference,
            candidate,
            reference_task,
            candidate_task,
        ),
        {
            "field": "geometry",
            "status": "not_checked",
            "reason": (
                "Each input geometry may be consistent with its own output, "
                "but the normalized inspection does not yet compare the two "
                "geometries with each other."
            ),
        },
    ]
    multiplicity = _consistency_value_check(
        reference,
        candidate,
        "multiplicity",
        reference_task,
        candidate_task,
        comparison_axis=True,
    )
    checks.append(multiplicity)

    blocking_fields = [
        check["field"]
        for check in checks
        if check["field"] != "multiplicity" and check["status"] == "different"
    ]
    unchecked_fields = [
        check["field"]
        for check in checks
        if check["field"] != "multiplicity" and check["status"] == "not_checked"
    ]
    if blocking_fields:
        comparability_status = "not_comparable"
    elif unchecked_fields:
        comparability_status = "partially_checked"
    else:
        comparability_status = "comparable"

    reference_energy = _task_energy(reference_task)
    candidate_energy = _task_energy(candidate_task)
    energy = _energy_comparison(reference_energy, candidate_energy)
    verdict = _verdict(
        energy,
        comparability_status,
        blocking_fields,
        unchecked_fields,
    )
    uncertainty = _uncertainty(
        reference,
        candidate,
        comparability_status,
        blocking_fields,
        unchecked_fields,
    )

    return {
        "schema_version": RUN_COMPARISON_SCHEMA,
        "program": reference_program,
        "sources": {
            "reference_output": _nested(reference, "source", "path"),
            "candidate_output": _nested(candidate, "source", "path"),
        },
        "assessment": {
            "verdict": verdict,
            "comparability": {
                "status": comparability_status,
                "blocking_fields": blocking_fields,
                "unchecked_fields": unchecked_fields,
            },
        },
        "evidence": {
            "reference": _run_summary(reference, reference_task),
            "candidate": _run_summary(candidate, candidate_task),
            "comparability_checks": checks,
            "energy": energy,
        },
        "uncertainty": uncertainty,
        "next_actions": _next_actions(
            energy,
            multiplicity,
            comparability_status,
        ),
    }


def _primary_energy_task(inspection: Mapping[str, Any]) -> Mapping[str, Any] | None:
    tasks = _nested(inspection, "evidence", "tasks")
    if not isinstance(tasks, list):
        return None
    for task in reversed(tasks):
        if isinstance(task, Mapping) and _task_energy(task) is not None:
            return task
    return None


def _task_energy(task: Mapping[str, Any] | None) -> float | None:
    value = _mapping_value(task, "energy_hartree")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    energy = float(value)
    return energy if math.isfinite(energy) else None


def _task_completion_check(
    reference: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    reference_outcome = _mapping_value(reference, "outcome")
    candidate_outcome = _mapping_value(candidate, "outcome")
    if reference_outcome == candidate_outcome == "success":
        return {
            "field": "task_completion",
            "status": "match",
            "reference": reference_outcome,
            "candidate": candidate_outcome,
        }
    return {
        "field": "task_completion",
        "status": "different",
        "reference": reference_outcome,
        "candidate": candidate_outcome,
        "reason": "Both energy-bearing tasks must have a successful outcome.",
    }


def _value_check(field: str, reference: Any, candidate: Any) -> dict[str, Any]:
    if reference is None or candidate is None:
        return {
            "field": field,
            "status": "not_checked",
            "reference": reference,
            "candidate": candidate,
            "reason": f"Both runs did not provide {field} evidence.",
        }
    return {
        "field": field,
        "status": "match" if reference == candidate else "different",
        "reference": reference,
        "candidate": candidate,
    }


def _consistency_value_check(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    field: str,
    reference_task: Mapping[str, Any] | None,
    candidate_task: Mapping[str, Any] | None,
    *,
    comparison_axis: bool = False,
) -> dict[str, Any]:
    reference_value = _consistent_input_value(reference, field, reference_task)
    candidate_value = _consistent_input_value(candidate, field, candidate_task)
    checked = _value_check(field, reference_value, candidate_value)
    if comparison_axis:
        checked["comparison_axis"] = True
    return checked


def _composition_check(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference_task: Mapping[str, Any] | None,
    candidate_task: Mapping[str, Any] | None,
) -> dict[str, Any]:
    reference_geometry = _consistent_input_value(
        reference,
        "geometry",
        reference_task,
    )
    candidate_geometry = _consistent_input_value(
        candidate,
        "geometry",
        candidate_task,
    )
    reference_value = _composition(reference_geometry)
    candidate_value = _composition(candidate_geometry)
    checked = _value_check("composition", reference_value, candidate_value)
    if checked["status"] == "not_checked":
        checked["reason"] = (
            "Both runs did not provide checked atom-count and element-order evidence."
        )
    return checked


def _basis_check(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference_task: Mapping[str, Any] | None,
    candidate_task: Mapping[str, Any] | None,
) -> dict[str, Any]:
    reference_basis = _basis_signature(reference, reference_task)
    candidate_basis = _basis_signature(candidate, candidate_task)
    checked = _value_check("basis", reference_basis, candidate_basis)
    if checked["status"] == "not_checked":
        checked["reason"] = (
            "Both runs did not provide checked runtime basis summaries."
        )
    return checked


def _consistent_input_value(
    inspection: Mapping[str, Any],
    field: str,
    task: Mapping[str, Any] | None,
) -> Any:
    consistency = _nested(
        inspection,
        "evidence",
        "input_output_consistency",
    )
    if not isinstance(consistency, Mapping):
        return None
    checks = consistency.get("checks")
    if not isinstance(checks, list):
        return None
    task_comparison = _task_state_field_comparison(checks, task, field)
    if task_comparison is not None:
        if task_comparison.get("status") != "match":
            return None
        return task_comparison.get("input")
    for check in checks:
        if not isinstance(check, Mapping) or check.get("field") != field:
            continue
        if check.get("status") != "match":
            return None
        return check.get("input")
    return None


def _task_state_field_comparison(
    checks: list[Any],
    task: Mapping[str, Any] | None,
    field: str,
) -> Mapping[str, Any] | None:
    task_index = _mapping_value(task, "index")
    if isinstance(task_index, bool) or not isinstance(task_index, int):
        return None
    for check in checks:
        if not isinstance(check, Mapping) or check.get("field") != "task_states":
            continue
        task_states = check.get("tasks")
        if not isinstance(task_states, list):
            return None
        for task_state in task_states:
            if (
                not isinstance(task_state, Mapping)
                or task_state.get("task_index") != task_index
            ):
                continue
            comparisons = task_state.get("comparisons")
            if not isinstance(comparisons, Mapping):
                return None
            comparison = comparisons.get(field)
            return comparison if isinstance(comparison, Mapping) else None
    return None


def _composition(geometry: Any) -> dict[str, Any] | None:
    if not isinstance(geometry, Mapping):
        return None
    atom_count = geometry.get("atom_count")
    elements = geometry.get("elements")
    if not isinstance(atom_count, int) or not isinstance(elements, list):
        return None
    return {
        "atom_count": atom_count,
        "elements": list(elements),
    }


def _basis_signature(
    inspection: Mapping[str, Any],
    task: Mapping[str, Any] | None,
) -> Any:
    consistency = _nested(
        inspection,
        "evidence",
        "input_output_consistency",
    )
    if not isinstance(consistency, Mapping):
        return None
    checks = consistency.get("checks")
    if not isinstance(checks, list):
        return None
    task_comparison = _task_state_field_comparison(
        checks,
        task,
        "basis_coverage",
    )
    if task_comparison is not None:
        return _matched_basis_signature(task_comparison)
    for check in checks:
        if not isinstance(check, Mapping) or check.get("field") != "basis_coverage":
            continue
        if check.get("status") != "match":
            return None
        return _matched_basis_signature(check)
    return None


def _matched_basis_signature(comparison: Mapping[str, Any]) -> Any:
    if comparison.get("status") != "match":
        return None
    runtime = comparison.get("output")
    if not isinstance(runtime, Mapping):
        return None
    return {
        "elements": runtime.get("elements"),
        "summaries": runtime.get("summaries"),
    }


def _energy_comparison(
    reference: float | None,
    candidate: float | None,
) -> dict[str, Any]:
    if reference is None or candidate is None:
        return {
            "status": "not_checked",
            "unit": "hartree",
            "reference": reference,
            "candidate": candidate,
            "candidate_minus_reference": None,
            "candidate_minus_reference_kcal_per_mol": None,
            "lower_energy_run": None,
        }
    delta = candidate - reference
    if abs(delta) <= ENERGY_EQUALITY_TOLERANCE_HARTREE:
        lower = "equal_within_tolerance"
    elif delta < 0:
        lower = "candidate"
    else:
        lower = "reference"
    return {
        "status": "checked",
        "unit": "hartree",
        "reference": reference,
        "candidate": candidate,
        "candidate_minus_reference": delta,
        "candidate_minus_reference_kcal_per_mol": (
            delta * HARTREE_TO_KCAL_PER_MOL
        ),
        "equality_tolerance_hartree": ENERGY_EQUALITY_TOLERANCE_HARTREE,
        "lower_energy_run": lower,
    }


def _verdict(
    energy: Mapping[str, Any],
    comparability_status: str,
    blocking_fields: list[str],
    unchecked_fields: list[str],
) -> dict[str, Any]:
    lower = energy.get("lower_energy_run")
    if energy.get("status") != "checked":
        return {
            "label": "comparison_incomplete",
            "confidence": 0.2,
            "reasons": ["Both runs did not provide a finite primary energy."],
        }
    if comparability_status == "not_comparable":
        return {
            "label": "comparison_not_supported",
            "confidence": 0.2,
            "reasons": [
                "Energy arithmetic is available, but required settings differ: "
                + ", ".join(blocking_fields)
                + "."
            ],
        }
    label = {
        "candidate": "candidate_lower_energy",
        "reference": "reference_lower_energy",
        "equal_within_tolerance": "energies_equal_within_tolerance",
    }[lower]
    confidence = 0.85 if comparability_status == "comparable" else 0.6
    reasons = [
        f"The {lower.replace('_', ' ')} run has the lower parsed energy."
        if lower in {"candidate", "reference"}
        else "The parsed energies agree within the stated tolerance."
    ]
    if unchecked_fields:
        reasons.append(
            "The ordering is conditional because these fields were not checked: "
            + ", ".join(unchecked_fields)
            + "."
        )
    return {
        "label": label,
        "confidence": confidence,
        "reasons": reasons,
    }


def _uncertainty(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    comparability_status: str,
    blocking_fields: list[str],
    unchecked_fields: list[str],
) -> list[dict[str, Any]]:
    uncertainty = []
    for role, inspection in (
        ("reference", reference),
        ("candidate", candidate),
    ):
        run_uncertainty = inspection.get("uncertainty")
        if isinstance(run_uncertainty, list) and run_uncertainty:
            uncertainty.append({
                "code": f"{role}_run_has_uncertainty",
                "message": (
                    f"The {role} inspection reported "
                    f"{len(run_uncertainty)} uncertainty item(s)."
                ),
                "impact": "Review the nested run inspection before accepting the comparison.",
            })
    if comparability_status == "partially_checked":
        uncertainty.append({
            "code": "run_comparability_partially_checked",
            "message": "Some required comparison fields were unavailable.",
            "fields": unchecked_fields,
            "impact": "The energy ordering is conditional, not a ground-state assignment.",
        })
    elif comparability_status == "not_comparable":
        uncertainty.append({
            "code": "run_settings_differ",
            "message": "Required calculation settings differ between the runs.",
            "fields": blocking_fields,
            "impact": "Do not interpret the energy difference as a state ordering.",
        })
    return uncertainty


def _next_actions(
    energy: Mapping[str, Any],
    multiplicity: Mapping[str, Any],
    comparability_status: str,
) -> list[dict[str, Any]]:
    if comparability_status == "not_comparable":
        return [{
            "action": "align_calculation_settings",
            "reason": "Re-run with matched required settings before interpreting energies.",
            "priority": 1,
        }]
    actions = []
    lower = energy.get("lower_energy_run")
    if lower in {"reference", "candidate"}:
        actions.append({
            "action": "review_state_character",
            "run": lower,
            "reason": (
                "Confirm orbital occupations and state character before "
                "accepting the lower-energy solution."
            ),
            "priority": 1,
        })
    if multiplicity.get("status") == "different":
        actions.append({
            "action": "extend_multiplicity_comparison",
            "reason": (
                "The runs compare different multiplicities; include other "
                "chemically plausible states when needed."
            ),
            "priority": 2,
        })
    return actions


def _run_summary(
    inspection: Mapping[str, Any],
    task: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "output_file": _nested(inspection, "source", "path"),
        "verdict": _nested(inspection, "assessment", "verdict"),
        "task": dict(task) if task is not None else None,
        "multiplicity": _consistent_input_value(
            inspection,
            "multiplicity",
            task,
        ),
        "charge": _consistent_input_value(inspection, "charge", task),
    }


def _nested(mapping: Any, *keys: str) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _mapping_value(mapping: Mapping[str, Any] | None, key: str) -> Any:
    return mapping.get(key) if isinstance(mapping, Mapping) else None


__all__ = [
    "ENERGY_EQUALITY_TOLERANCE_HARTREE",
    "RUN_COMPARISON_SCHEMA",
    "compare_run_inspections",
    "compare_runs",
]

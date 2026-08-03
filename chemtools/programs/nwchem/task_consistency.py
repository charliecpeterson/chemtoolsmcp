"""Pair NWChem input state with evidence inside top-level output tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from chemtools.programs.nwchem.basis_consistency import (
    compare_basis_coverage,
    compare_basis_mode,
    compare_ecp_replacements,
)
from chemtools.programs.nwchem.electron_consistency import (
    compare_electron_count,
    compare_electron_spin_parity,
    compare_spin_occupations,
    compare_wavefunction_class,
)
from chemtools.programs.nwchem.geometry_consistency import (
    compare_task_geometry,
    load_task_geometry,
)
from chemtools.programs.nwchem.xc_consistency import compare_xc_functional


_OPERATION_ALIASES = {
    "freq": "frequency",
    "frequencies": "frequency",
    "opt": "optimize",
    "optimization": "optimize",
    "raman": "frequency",
    "single_point": "energy",
}


def compare_task_states(
    input_path: Path,
    parsed_input: Mapping[str, Any],
    parsed_output: Mapping[str, Any],
    output_states: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    input_states = list(parsed_input.get("task_states") or [])
    output_tasks = list(parsed_output.get("tasks") or [])
    if len(input_states) <= 1 and len(output_tasks) <= 1:
        return None
    if (
        not input_states
        or len(input_states) != len(output_tasks)
        or len(output_states) != len(output_tasks)
    ):
        return _not_checked(
            "Input and output did not expose the same number of top-level tasks.",
            {"task_count": len(input_states)},
            {"task_count": len(output_tasks)},
        )

    input_methods = [
        str(state.get("method") or state.get("module") or "").upper()
        for state in input_states
    ]
    output_methods = [
        str(task.get("method") or "").upper()
        for task in output_tasks
    ]
    input_operations = [
        normalize_operation(state.get("operation"))
        for state in input_states
    ]
    output_operations = [
        normalize_operation(task.get("kind"))
        for task in output_tasks
    ]
    operation_mismatch = any(
        output_operation not in {None, "other"}
        and input_operation != output_operation
        for input_operation, output_operation in zip(
            input_operations,
            output_operations,
        )
    )
    if input_methods != output_methods or operation_mismatch:
        return _not_checked(
            (
                "Task methods or operations could not be paired by "
                "top-level task index."
            ),
            {
                "methods": input_methods,
                "operations": input_operations,
            },
            {
                "methods": output_methods,
                "operations": output_operations,
            },
        )

    task_comparisons = []
    checked_statuses = []
    for input_state, output_state in zip(input_states, output_states):
        geometry_spec = input_state.get("geometry") or {}
        input_geometry = load_task_geometry(input_path, geometry_spec)
        comparisons = {
            "charge": _task_scalar_comparison(
                input_state.get("charge"),
                output_state.get("charges") or [],
                "charge",
            ),
            "multiplicity": _task_scalar_comparison(
                input_state.get("multiplicity"),
                output_state.get("multiplicities") or [],
                "multiplicity",
            ),
            "atom_count": _task_scalar_comparison(
                (
                    input_geometry.get("atom_count")
                    if input_geometry is not None
                    else None
                ),
                output_state.get("atom_counts") or [],
                "atom count",
            ),
            "electron_count": compare_electron_count(
                input_state,
                input_geometry,
                output_state.get("electron_counts") or [],
            ),
            "electron_spin_parity": compare_electron_spin_parity(
                input_state,
                input_geometry,
                output_state.get("electron_counts") or [],
                output_state.get("multiplicities") or [],
            ),
            "spin_occupations": compare_spin_occupations(
                input_state,
                input_geometry,
                output_state.get("alpha_electrons") or [],
                output_state.get("beta_electrons") or [],
            ),
            "wavefunction_class": compare_wavefunction_class(
                input_state,
                output_state.get("wavefunction_classes") or [],
                output_state.get("wavefunction_labels") or [],
            ),
            "xc_functional": compare_xc_functional(
                input_state,
                output_state,
            ),
            "basis_mode": compare_basis_mode(input_state, output_state),
            "basis_coverage": compare_basis_coverage(
                input_geometry,
                output_state,
            ),
            "ecp_replacements": compare_ecp_replacements(
                input_state,
                output_state,
                input_geometry,
            ),
            "geometry": compare_task_geometry(
                geometry_spec,
                input_geometry,
                output_state,
                output_states,
            ),
        }
        statuses = [
            comparison["status"]
            for comparison in comparisons.values()
        ]
        if "mismatch" in statuses:
            task_status = "mismatch"
        elif "match" in statuses:
            task_status = "match"
        else:
            task_status = "not_checked"
        checked_statuses.extend(
            status for status in statuses if status != "not_checked"
        )
        input_state_evidence = {
            "charge": input_state.get("charge"),
            "charge_source": input_state.get("charge_source"),
            "multiplicity": input_state.get("multiplicity"),
            "multiplicity_source": input_state.get("multiplicity_source"),
            "reference": input_state.get("reference"),
            "basis": input_state.get("basis"),
            "ecp": input_state.get("ecp"),
            "geometry": geometry_spec,
        }
        if input_state.get("xc") is not None:
            input_state_evidence["xc"] = input_state["xc"]
        if input_state.get("method") is not None:
            input_state_evidence["method"] = input_state["method"]
            input_state_evidence["method_source"] = input_state.get(
                "method_source"
            )
        task_comparisons.append({
            "task_index": input_state["task_index"],
            "module": input_state["module"],
            "operation": input_state["operation"],
            "status": task_status,
            "input_state": input_state_evidence,
            "output_evidence": {
                "charges": output_state.get("charges") or [],
                "multiplicities": (
                    output_state.get("multiplicities") or []
                ),
                "atom_counts": output_state.get("atom_counts") or [],
                "electron_counts": (
                    output_state.get("electron_counts") or []
                ),
                "alpha_electrons": (
                    output_state.get("alpha_electrons") or []
                ),
                "beta_electrons": (
                    output_state.get("beta_electrons") or []
                ),
                "wavefunction_labels": (
                    output_state.get("wavefunction_labels") or []
                ),
                "wavefunction_classes": (
                    output_state.get("wavefunction_classes") or []
                ),
                "xc_functional_labels": (
                    output_state.get("xc_functional_labels") or []
                ),
                "xc_functional_names": (
                    output_state.get("xc_functional_names") or []
                ),
                "basis_modes": output_state.get("basis_modes") or [],
                "basis_function_counts": (
                    output_state.get("basis_function_counts") or []
                ),
                "basis_shell_counts": (
                    output_state.get("basis_shell_counts") or []
                ),
                "basis_summaries": (
                    output_state.get("basis_summaries") or []
                ),
                "ecp_replacements": (
                    output_state.get("ecp_replacements") or {}
                ),
                "geometry_names": sorted(
                    (output_state.get("first_geometry_by_name") or {}).keys()
                ),
            },
            "comparisons": comparisons,
        })

    if "mismatch" in checked_statuses:
        status = "mismatch"
    elif checked_statuses:
        status = "match"
    else:
        status = "not_checked"
    check = {
        "field": "task_states",
        "status": status,
        "basis": (
            "Charge, multiplicity, atom count, electron count, spin parity, "
            "alpha/beta occupations, wavefunction class, AO basis mode, "
            "XC functional, AO basis element coverage, ECP replacement, "
            "and available geometry evidence "
            "paired by top-level task index."
        ),
        "tasks": task_comparisons,
    }
    if status == "not_checked":
        check["reason"] = (
            "No task exposed both an expected state value and output evidence."
        )
    return check


def compare_single_task_electronic_state_checks(
    input_path: Path,
    parsed_input: Mapping[str, Any],
    output_states: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    input_states = list(parsed_input.get("task_states") or [])
    if len(input_states) != 1 or len(output_states) != 1:
        return []
    input_state = input_states[0]
    output_state = output_states[0]
    input_geometry = load_task_geometry(
        input_path,
        input_state.get("geometry") or {},
    )
    electron_counts = output_state.get("electron_counts") or []
    multiplicities = output_state.get("multiplicities") or []
    checks = []
    if electron_counts:
        checks.append({
            "field": "electron_count",
            **compare_electron_count(
                input_state,
                input_geometry,
                electron_counts,
            ),
            "basis": (
                "Sum of effective nuclear charges after explicit ECP core "
                "replacement, minus molecular charge."
            ),
        })
    if electron_counts and multiplicities:
        checks.append({
            "field": "electron_spin_parity",
            **compare_electron_spin_parity(
                input_state,
                input_geometry,
                electron_counts,
                multiplicities,
            ),
        })
    if output_state.get("alpha_electrons") and output_state.get(
        "beta_electrons"
    ):
        checks.append({
            "field": "spin_occupations",
            **compare_spin_occupations(
                input_state,
                input_geometry,
                output_state["alpha_electrons"],
                output_state["beta_electrons"],
            ),
        })
    if output_state.get("wavefunction_labels"):
        checks.append({
            "field": "wavefunction_class",
            **compare_wavefunction_class(
                input_state,
                output_state.get("wavefunction_classes") or [],
                output_state["wavefunction_labels"],
            ),
        })
    if output_state.get("xc_functional_labels"):
        checks.append({
            "field": "xc_functional",
            **compare_xc_functional(input_state, output_state),
            "basis": (
                "Explicit named XC alias compared with NWChem's runtime "
                "XC Method label."
            ),
        })
    if (
        output_state.get("basis_modes")
        or output_state.get("basis_function_counts")
        or output_state.get("basis_shell_counts")
    ):
        checks.append({
            "field": "basis_mode",
            **compare_basis_mode(input_state, output_state),
            "basis": (
                "Input AO basis spherical/Cartesian selection compared with "
                "the runtime basis summary; shell and function counts are "
                "reported as output evidence."
            ),
        })
    if output_state.get("basis_summaries"):
        checks.append({
            "field": "basis_coverage",
            **compare_basis_coverage(input_geometry, output_state),
            "basis": (
                "Elements in the selected input geometry compared with "
                "the runtime AO basis tag rows; per-tag shell and function "
                "counts are reported as output evidence."
            ),
        })
    if output_state.get("ecp_replacements"):
        checks.append({
            "field": "ecp_replacements",
            **compare_ecp_replacements(
                input_state,
                output_state,
                input_geometry,
            ),
            "basis": (
                "Explicit or bundled-library nelec values compared with "
                "NWChem's printed ECP electron replacements."
            ),
        })
    return checks


def normalize_operation(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).lower()
    return _OPERATION_ALIASES.get(normalized, normalized)


def _not_checked(
    reason: str,
    input_value: Any = None,
    output_value: Any = None,
) -> dict[str, Any]:
    check = {
        "field": "task_states",
        "status": "not_checked",
        "reason": reason,
    }
    if input_value is not None:
        check["input"] = input_value
    if output_value is not None:
        check["output"] = output_value
    return check


def _task_scalar_comparison(
    expected: Any,
    observed: list[Any],
    label: str,
) -> dict[str, Any]:
    unique_observed = list(dict.fromkeys(observed))
    if expected is None:
        return {
            "status": "not_checked",
            "reason": f"The input {label} is unresolved.",
        }
    if len(unique_observed) != 1:
        return {
            "status": "not_checked",
            "reason": (
                f"The task does not expose one unambiguous output {label}."
            ),
            "input": expected,
            "output": unique_observed,
        }
    return {
        "status": (
            "match"
            if expected == unique_observed[0]
            else "mismatch"
        ),
        "input": expected,
        "output": unique_observed[0],
    }


__all__ = [
    "compare_single_task_electronic_state_checks",
    "compare_task_states",
    "normalize_operation",
]

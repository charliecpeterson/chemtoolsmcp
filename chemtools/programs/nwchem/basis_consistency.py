"""Compare NWChem basis and ECP input intent with runtime evidence."""

from __future__ import annotations

from typing import Any, Mapping

from chemtools.programs.nwchem.electron_consistency import (
    resolve_ecp_core_electrons,
)


def compare_basis_mode(
    input_state: Mapping[str, Any],
    output_state: Mapping[str, Any],
) -> dict[str, Any]:
    basis = input_state.get("basis") or {}
    expected_mode = basis.get("mode")
    observed_modes = list(dict.fromkeys(output_state.get("basis_modes") or []))
    output_evidence = {
        "modes": observed_modes,
        "function_counts": list(
            output_state.get("basis_function_counts") or []
        ),
        "shell_counts": list(output_state.get("basis_shell_counts") or []),
    }
    if expected_mode is None:
        return {
            "status": "not_checked",
            "reason": "The active input AO basis representation is unresolved.",
            "input": dict(basis),
            "output": output_evidence,
        }
    if len(observed_modes) != 1:
        return {
            "status": "not_checked",
            "reason": (
                "The task does not expose one unambiguous AO basis "
                "representation."
            ),
            "input": dict(basis),
            "output": output_evidence,
        }
    return {
        "status": (
            "match" if expected_mode == observed_modes[0] else "mismatch"
        ),
        "input": dict(basis),
        "output": {
            "mode": observed_modes[0],
            "function_counts": output_evidence["function_counts"],
            "shell_counts": output_evidence["shell_counts"],
        },
    }


def compare_basis_coverage(
    input_geometry: Mapping[str, Any] | None,
    output_state: Mapping[str, Any],
) -> dict[str, Any]:
    summaries = list(output_state.get("basis_summaries") or [])
    if input_geometry is None:
        return {
            "status": "not_checked",
            "reason": "The selected input geometry is unresolved.",
            "output": {"summaries": summaries},
        }
    expected = list(dict.fromkeys(input_geometry.get("elements") or []))
    if not expected:
        return {
            "status": "not_checked",
            "reason": "The selected input geometry has no resolved elements.",
            "input": {"elements": expected},
            "output": {"summaries": summaries},
        }

    rows = [
        row
        for summary in summaries
        for row in summary.get("rows") or []
    ]
    covered = list(dict.fromkeys(
        row["element"]
        for row in rows
        if row.get("element") is not None
    ))
    unparsed_tags = list(dict.fromkeys(
        row["tag"]
        for row in rows
        if row.get("element") is None
    ))
    output_evidence = {
        "elements": covered,
        "unparsed_tags": unparsed_tags,
        "summaries": summaries,
    }
    if not rows:
        return {
            "status": "not_checked",
            "reason": "The task has no runtime AO basis tag rows.",
            "input": {"elements": expected},
            "output": output_evidence,
        }

    missing = [element for element in expected if element not in covered]
    if missing and unparsed_tags:
        return {
            "status": "not_checked",
            "reason": (
                "At least one runtime AO basis tag could not be mapped "
                "to an element."
            ),
            "input": {"elements": expected},
            "output": output_evidence,
            "missing_elements": missing,
        }
    return {
        "status": "mismatch" if missing else "match",
        "input": {"elements": expected},
        "output": output_evidence,
        "missing_elements": missing,
    }


def compare_ecp_replacements(
    input_state: Mapping[str, Any],
    output_state: Mapping[str, Any],
    input_geometry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    ecp = input_state.get("ecp") or {}
    if ecp.get("source") not in {"none", "explicit"}:
        return {
            "status": "not_checked",
            "reason": "The active input ECP state is unresolved.",
            "input": dict(ecp),
            "output": dict(output_state.get("ecp_replacements") or {}),
        }
    expected, unresolved, sources = resolve_ecp_core_electrons(
        input_state,
        input_geometry,
    )
    observed_lists = output_state.get("ecp_replacements") or {}
    observed = {
        element: values[0]
        for element, values in observed_lists.items()
        if len(values) == 1
    }
    conflicting = {
        element: list(values)
        for element, values in observed_lists.items()
        if len(values) != 1
    }
    if unresolved:
        return {
            "status": "not_checked",
            "reason": (
                "The active ECP library core-electron replacements are "
                "unresolved."
            ),
            "input": {
                "core_electrons": expected,
                "unresolved_elements": unresolved,
                "resolved_library_sources": sources,
            },
            "output": dict(observed_lists),
        }
    if not expected:
        return {
            "status": "not_checked",
            "reason": "The input has no explicit ECP core-electron counts.",
            "input": expected,
            "output": dict(observed_lists),
        }
    if conflicting:
        return {
            "status": "not_checked",
            "reason": "The output reports conflicting ECP replacement counts.",
            "input": expected,
            "output": dict(observed_lists),
        }
    if not set(expected).issubset(observed):
        return {
            "status": "not_checked",
            "reason": (
                "The output does not print every explicit input ECP."
            ),
            "input": expected,
            "output": observed,
        }
    mismatches = {
        element: {
            "input": count,
            "output": observed[element],
        }
        for element, count in expected.items()
        if observed[element] != count
    }
    comparison = {
        "status": "mismatch" if mismatches else "match",
        "input": expected,
        "output": {
            element: observed[element]
            for element in expected
        },
    }
    if sources:
        comparison["input_sources"] = sources
    return comparison


__all__ = [
    "compare_basis_coverage",
    "compare_basis_mode",
    "compare_ecp_replacements",
]

"""Derive NWChem electron counts from task geometry, charge, and ECP state.

The calculation abstains when the effective nuclear charge is not explicit.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

from chemtools.core.common import ELEMENT_TO_Z
from chemtools.programs.nwchem.input.basis_library import (
    bundled_basis_library_path,
    resolve_ecp_set,
)


def compare_electron_count(
    input_state: Mapping[str, Any],
    input_geometry: Mapping[str, Any] | None,
    observed: list[int],
) -> dict[str, Any]:
    expected, reason, calculation = _expected_electron_count(
        input_state,
        input_geometry,
    )
    unique_observed = list(dict.fromkeys(observed))
    if expected is None:
        comparison = {
            "status": "not_checked",
            "reason": reason,
        }
        if calculation:
            comparison["input"] = calculation
        if unique_observed:
            comparison["output"] = unique_observed
        return comparison
    if len(unique_observed) != 1:
        return {
            "status": "not_checked",
            "reason": (
                "The task does not expose one unambiguous output electron "
                "count."
            ),
            "input": calculation,
            "output": unique_observed,
        }
    return {
        "status": (
            "match"
            if expected == unique_observed[0]
            else "mismatch"
        ),
        "input": calculation,
        "output": unique_observed[0],
    }


def compare_electron_spin_parity(
    input_state: Mapping[str, Any],
    input_geometry: Mapping[str, Any] | None,
    observed_electron_counts: list[int],
    observed_multiplicities: list[int],
) -> dict[str, Any]:
    expected, electron_reason, _ = _expected_electron_count(
        input_state,
        input_geometry,
    )
    input_evidence = _parity_evidence(
        expected,
        input_state.get("multiplicity"),
        unresolved_reason=electron_reason,
    )
    unique_electron_counts = list(dict.fromkeys(observed_electron_counts))
    unique_multiplicities = list(dict.fromkeys(observed_multiplicities))
    if (
        len(unique_electron_counts) == 1
        and len(unique_multiplicities) == 1
    ):
        output_evidence = _parity_evidence(
            unique_electron_counts[0],
            unique_multiplicities[0],
            unresolved_reason="",
        )
    else:
        output_evidence = {
            "status": "unresolved",
            "electron_counts": unique_electron_counts,
            "multiplicities": unique_multiplicities,
            "reason": (
                "The task does not expose one unambiguous electron count and "
                "spin multiplicity."
            ),
        }

    checked = [
        evidence
        for evidence in (input_evidence, output_evidence)
        if evidence["status"] == "checked"
    ]
    comparison = {
        "input": input_evidence,
        "output": output_evidence,
        "basis": (
            "Electron count minus (multiplicity minus one) must be even."
        ),
    }
    if any(not evidence["compatible"] for evidence in checked):
        return {"status": "mismatch", **comparison}
    if len(checked) == 2:
        return {"status": "match", **comparison}
    return {
        "status": "not_checked",
        "reason": (
            "Both input-derived and output-reported electron/multiplicity "
            "pairs were not available."
        ),
        **comparison,
    }


def compare_spin_occupations(
    input_state: Mapping[str, Any],
    input_geometry: Mapping[str, Any] | None,
    observed_alpha_electrons: list[int],
    observed_beta_electrons: list[int],
) -> dict[str, Any]:
    expected_electrons, electron_reason, _ = _expected_electron_count(
        input_state,
        input_geometry,
    )
    multiplicity = input_state.get("multiplicity")
    unique_alpha = list(dict.fromkeys(observed_alpha_electrons))
    unique_beta = list(dict.fromkeys(observed_beta_electrons))
    if expected_electrons is None:
        return {
            "status": "not_checked",
            "reason": electron_reason,
            "output": {
                "alpha_electrons": unique_alpha,
                "beta_electrons": unique_beta,
            },
        }
    if not isinstance(multiplicity, int) or multiplicity == 0:
        return {
            "status": "not_checked",
            "reason": "The input spin multiplicity is unresolved.",
            "input": {"electron_count": expected_electrons},
            "output": {
                "alpha_electrons": unique_alpha,
                "beta_electrons": unique_beta,
            },
        }
    if len(unique_alpha) != 1 or len(unique_beta) != 1:
        return {
            "status": "not_checked",
            "reason": (
                "The task does not expose one unambiguous alpha/beta "
                "electron pair."
            ),
            "input": {
                "electron_count": expected_electrons,
                "multiplicity": abs(multiplicity),
            },
            "output": {
                "alpha_electrons": unique_alpha,
                "beta_electrons": unique_beta,
            },
        }

    alpha_electrons = unique_alpha[0]
    beta_electrons = unique_beta[0]
    expected_spin_difference = abs(multiplicity) - 1
    observed_total = alpha_electrons + beta_electrons
    observed_spin_difference = abs(alpha_electrons - beta_electrons)
    return {
        "status": (
            "match"
            if (
                observed_total == expected_electrons
                and observed_spin_difference == expected_spin_difference
            )
            else "mismatch"
        ),
        "input": {
            "electron_count": expected_electrons,
            "multiplicity": abs(multiplicity),
            "spin_difference": expected_spin_difference,
        },
        "output": {
            "alpha_electrons": alpha_electrons,
            "beta_electrons": beta_electrons,
            "electron_count": observed_total,
            "spin_difference": observed_spin_difference,
        },
        "basis": (
            "Alpha plus beta electrons must equal the total electron count; "
            "their absolute difference must equal multiplicity minus one."
        ),
    }


def compare_wavefunction_class(
    input_state: Mapping[str, Any],
    observed_classes: list[str],
    observed_labels: list[str],
) -> dict[str, Any]:
    reference = input_state.get("reference") or {}
    expected_class = reference.get("class")
    unique_classes = list(dict.fromkeys(observed_classes))
    unique_labels = list(dict.fromkeys(observed_labels))
    if expected_class not in {"closed_shell", "open_shell"}:
        return {
            "status": "not_checked",
            "reason": "The input wavefunction class is unresolved.",
            "input": reference,
            "output": {
                "classes": unique_classes,
                "labels": unique_labels,
            },
        }
    if len(unique_classes) != 1:
        return {
            "status": "not_checked",
            "reason": (
                "The task does not expose one unambiguous wavefunction class."
            ),
            "input": reference,
            "output": {
                "classes": unique_classes,
                "labels": unique_labels,
            },
        }
    return {
        "status": (
            "match"
            if expected_class == unique_classes[0]
            else "mismatch"
        ),
        "input": reference,
        "output": {
            "class": unique_classes[0],
            "labels": unique_labels,
        },
        "basis": "Closed-shell versus open-shell wavefunction class.",
    }


def normalize_wavefunction_class(value: str) -> str | None:
    normalized = value.strip().rstrip(".").lower()
    if normalized in {"closed shell", "rhf"}:
        return "closed_shell"
    if normalized in {
        "open shell",
        "spin polarized",
        "odft",
        "rodft",
        "rohf",
        "uhf",
    }:
        return "open_shell"
    return None


def _parity_evidence(
    electron_count: Any,
    multiplicity: Any,
    *,
    unresolved_reason: str,
) -> dict[str, Any]:
    if not isinstance(electron_count, int):
        return {
            "status": "unresolved",
            "reason": unresolved_reason or "The electron count is unresolved.",
        }
    if not isinstance(multiplicity, int) or multiplicity == 0:
        return {
            "status": "unresolved",
            "electron_count": electron_count,
            "reason": "The spin multiplicity is unresolved.",
        }
    physical_multiplicity = abs(multiplicity)
    return {
        "status": "checked",
        "electron_count": electron_count,
        "multiplicity": physical_multiplicity,
        "compatible": (
            electron_count - (physical_multiplicity - 1)
        ) % 2 == 0,
    }


def _expected_electron_count(
    input_state: Mapping[str, Any],
    input_geometry: Mapping[str, Any] | None,
) -> tuple[int | None, str, dict[str, Any]]:
    charge = input_state.get("charge")
    if not isinstance(charge, int):
        return None, "The input molecular charge is unresolved.", {}
    if input_geometry is None or not input_geometry.get("elements"):
        return None, "The selected input geometry is unresolved.", {}
    if input_geometry.get("has_explicit_center_charges"):
        return (
            None,
            (
                "The selected geometry uses explicit center charges, which "
                "this check does not yet resolve."
            ),
            {"molecular_charge": charge},
        )

    elements = list(input_geometry["elements"])
    if any(element not in ELEMENT_TO_Z for element in elements):
        return (
            None,
            "At least one geometry center has no known atomic number.",
            {
                "molecular_charge": charge,
                "elements": elements,
            },
        )

    ecp = input_state.get("ecp") or {
        "source": "none",
        "core_electrons": {},
        "library_elements": [],
        "default_library": False,
    }
    ecp_source = ecp.get("source")
    if ecp_source not in {"none", "explicit"}:
        return (
            None,
            f"The active ECP state is {ecp_source or 'unresolved'}.",
            {"molecular_charge": charge, "ecp": ecp},
        )
    core_electrons, unresolved_library_elements, library_sources = (
        resolve_ecp_core_electrons(input_state, input_geometry)
    )
    if unresolved_library_elements:
        return (
            None,
            (
                "The active ECP library core-electron replacements are "
                "unresolved."
            ),
            {
                "molecular_charge": charge,
                "ecp": ecp,
                "unresolved_elements": unresolved_library_elements,
                "resolved_library_sources": library_sources,
            },
        )

    effective_nuclear_charge = 0
    for element in elements:
        replacement = core_electrons.get(element, 0)
        atomic_number = ELEMENT_TO_Z[element]
        if (
            not isinstance(replacement, int)
            or replacement < 0
            or replacement > atomic_number
        ):
            return (
                None,
                f"The ECP core replacement for {element} is invalid.",
                {"molecular_charge": charge, "ecp": ecp},
            )
        effective_nuclear_charge += atomic_number - replacement
    expected = effective_nuclear_charge - charge
    calculation = {
        "expected": expected,
        "effective_nuclear_charge": effective_nuclear_charge,
        "molecular_charge": charge,
        "ecp_core_electrons": core_electrons,
    }
    if library_sources:
        calculation["ecp_core_electron_sources"] = library_sources
    if expected < 0:
        return None, "The derived electron count is negative.", calculation
    return expected, "", calculation


def resolve_ecp_core_electrons(
    input_state: Mapping[str, Any],
    input_geometry: Mapping[str, Any] | None,
) -> tuple[dict[str, int], list[str], dict[str, dict[str, str]]]:
    ecp = input_state.get("ecp") or {}
    core_electrons = dict(ecp.get("core_electrons") or {})
    elements = list(dict.fromkeys(
        (input_geometry or {}).get("elements") or []
    ))
    assignments = dict(ecp.get("library_assignments") or {})
    default_family = ecp.get("default_library_name")
    unresolved: list[str] = []
    sources: dict[str, dict[str, str]] = {}

    if ecp.get("uses_external_library_file"):
        unresolved.extend(
            element
            for element in elements
            if element in assignments or default_family is not None
        )
        return core_electrons, unresolved, sources

    legacy_library_elements = set(ecp.get("library_elements") or [])
    for element in elements:
        if element in core_electrons:
            continue
        family = assignments.get(element) or default_family
        if family is None:
            if element in legacy_library_elements or ecp.get(
                "default_library"
            ):
                unresolved.append(element)
            continue
        resolved = _resolve_bundled_ecp(family, element)
        if resolved is None:
            unresolved.append(element)
            continue
        nelec, source_file = resolved
        core_electrons[element] = nelec
        sources[element] = {
            "kind": "bundled_nwchem_library",
            "family": family,
            "file": source_file,
        }
    return core_electrons, sorted(set(unresolved)), sources


@lru_cache(maxsize=256)
def _resolve_bundled_ecp(
    family: str,
    element: str,
) -> tuple[int, str] | None:
    try:
        resolved = resolve_ecp_set(
            family,
            [element],
            bundled_basis_library_path(),
        )
    except (OSError, ValueError):
        return None
    nelec = (resolved.get("nelec_by_element") or {}).get(element)
    if not isinstance(nelec, int):
        return None
    return nelec, str(resolved["file"])


__all__ = [
    "compare_electron_count",
    "compare_electron_spin_parity",
    "compare_spin_occupations",
    "compare_wavefunction_class",
    "normalize_wavefunction_class",
    "resolve_ecp_core_electrons",
]

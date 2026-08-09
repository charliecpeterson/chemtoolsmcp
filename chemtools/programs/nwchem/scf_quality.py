"""Shared quality checks for completed NWChem SCF iteration histories."""

from __future__ import annotations

from typing import Any, Mapping


SCF_TRANSIENT_EXCURSION_THRESHOLD_HARTREE = 5.0


def find_converged_scf_excursion(
    scf: Mapping[str, Any],
) -> dict[str, Any] | None:
    if scf.get("status") != "converged":
        return None
    completed_runs = [
        run
        for run in scf.get("runs", [])
        if isinstance(run, Mapping) and run.get("completed")
    ]
    if not completed_runs:
        return None
    primary = completed_runs[-1]
    excursions = [
        iteration
        for iteration in primary.get("iterations", [])
        if isinstance(iteration, Mapping)
        and isinstance(iteration.get("delta_e_hartree"), (int, float))
        and not isinstance(iteration.get("delta_e_hartree"), bool)
        and iteration["delta_e_hartree"]
        > SCF_TRANSIENT_EXCURSION_THRESHOLD_HARTREE
    ]
    if not excursions:
        return None
    largest = max(excursions, key=lambda item: item["delta_e_hartree"])
    return {
        "iteration": largest.get("iteration"),
        "delta_e_hartree": largest["delta_e_hartree"],
        "energy_hartree": largest.get("energy_hartree"),
        "diis_error": largest.get("diis_error"),
        "run_iteration_count": primary.get("iteration_count"),
        "threshold_hartree": SCF_TRANSIENT_EXCURSION_THRESHOLD_HARTREE,
    }


__all__ = [
    "SCF_TRANSIENT_EXCURSION_THRESHOLD_HARTREE",
    "find_converged_scf_excursion",
]

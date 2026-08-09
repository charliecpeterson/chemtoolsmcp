"""Exact policy checks for completed-run SCF stability hardening."""

from __future__ import annotations

from chemtools.programs.nwchem.input._utils import (
    _select_scf_stabilization_strategy,
)
from chemtools.programs.nwchem.scf_quality import (
    find_converged_scf_excursion,
)


def _diagnosis_with_excursion(delta_e_hartree: float) -> dict:
    return {
        "failure_class": "no_clear_failure_detected",
        "scf": {
            "status": "converged",
            "runs": [{
                "completed": True,
                "iteration_count": 21,
                "trend": {"pattern": "well_converged"},
                "iterations": [
                    {
                        "iteration": 9,
                        "energy_hartree": -1649.55,
                        "delta_e_hartree": 0.08,
                        "diis_error": 24.8,
                    },
                    {
                        "iteration": 10,
                        "energy_hartree": -1632.84,
                        "delta_e_hartree": delta_e_hartree,
                        "diis_error": 133.0,
                    },
                ],
            }],
        },
    }


def test_converged_excursion_uses_largest_positive_jump():
    instability = find_converged_scf_excursion(
        _diagnosis_with_excursion(16.7)["scf"]
    )

    assert instability == {
        "iteration": 10,
        "delta_e_hartree": 16.7,
        "energy_hartree": -1632.84,
        "diis_error": 133.0,
        "run_iteration_count": 21,
        "threshold_hartree": 5.0,
    }


def test_small_converged_fluctuation_does_not_trigger_hardening():
    assert find_converged_scf_excursion(
        _diagnosis_with_excursion(5.0)["scf"]
    ) is None


def test_converged_excursion_selects_damping_without_smearing():
    selected = _select_scf_stabilization_strategy(
        reference_diagnosis=_diagnosis_with_excursion(16.7),
        iterations=None,
        smear=None,
        convergence_damp=None,
        convergence_ncydp=None,
        population_print=None,
    )

    assert selected == {
        "strategy": "converged_instability_hardening",
        "notes": [
            "reference_scf_recovered_from_large_transient_excursion"
        ],
        "iterations": 200,
        "smear": None,
        "convergence_damp": 70,
        "convergence_ncydp": 25,
        "population_print": None,
    }

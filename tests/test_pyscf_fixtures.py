"""Contracts for the curated bounded PySCF single-point fixture corpus."""

import json
from pathlib import Path

from chemtools import science_runner


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "pyscf" / "single_point_cases.json"
)


def _fixture_cases() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_pyscf_fixture_corpus_covers_supported_methods_and_outcomes():
    corpus = _fixture_cases()

    assert corpus["schema_version"] == "chemtools.pyscf-single-point-fixtures/1"
    assert corpus["recorded_with"] == {
        "pyscf_version": "2.13.1",
        "python_version": "3.12.13",
        "runner_result_schema": science_runner.PYSCF_SINGLE_POINT_RESULT_SCHEMA,
    }
    cases = corpus["cases"]
    assert [case["id"] for case in cases] == [
        "h2_rhf_sto3g",
        "o2_uhf_triplet_sto3g",
        "h2_rks_pbe_sto3g",
        "h_uks_pbe_sto3g",
        "h2_stretched_rhf_cycle_limit",
        "h2_uhf_doublet_electron_spin_inconsistent",
    ]
    assert {case["request"]["method"] for case in cases} == {
        "rhf", "uhf", "rks", "uks"
    }
    assert [case["expected"]["status"] for case in cases] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
        "runtime_error",
    ]


def test_pyscf_fixture_requests_remain_at_the_runner_boundary():
    cases = _fixture_cases()["cases"]

    normalized = {
        case["id"]: science_runner._pyscf_request(case["request"])
        for case in cases
    }

    assert normalized["h2_rhf_sto3g"]["multiplicity"] == 1
    assert normalized["o2_uhf_triplet_sto3g"]["multiplicity"] == 3
    assert normalized["h2_rks_pbe_sto3g"]["xc"] == "pbe"
    assert normalized["h_uks_pbe_sto3g"]["xc"] == "pbe"
    assert normalized["h2_stretched_rhf_cycle_limit"]["max_cycles"] == 1
    assert normalized[
        "h2_uhf_doublet_electron_spin_inconsistent"
    ]["multiplicity"] == 2


def test_pyscf_fixture_corpus_distinguishes_execution_from_convergence():
    cases = {case["id"]: case for case in _fixture_cases()["cases"]}

    cycle_limited = cases["h2_stretched_rhf_cycle_limit"]["expected"]
    assert cycle_limited["status"] == "completed"
    assert cycle_limited["scf_converged"] is False
    assert cycle_limited["warning_codes"] == ["scf_not_converged"]
    electron_spin_error = cases[
        "h2_uhf_doublet_electron_spin_inconsistent"
    ]["expected"]
    assert electron_spin_error == {
        "status": "runtime_error",
        "message_contains": "Electron number 2 and spin 1 are not consistent",
    }

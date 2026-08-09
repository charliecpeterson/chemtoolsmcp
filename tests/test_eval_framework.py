"""The legacy case evaluator uses direct program owners without API facades."""

from __future__ import annotations

import json
from pathlib import Path

from chemtools.application.evaluation import evaluate_case, evaluate_cases
from chemtools.core import eval as legacy_evaluation


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "nwchem_pyscf"
    / "h2o_rhf_sto3g.out"
)


def test_core_evaluation_imports_are_exact_compatibility_aliases():
    assert legacy_evaluation.evaluate_case is evaluate_case
    assert legacy_evaluation.evaluate_cases is evaluate_cases


def _case(tmp_path: Path) -> Path:
    case_path = tmp_path / "water.case.json"
    case_path.write_text(
        json.dumps({
            "case_id": "nwchem.water_rhf",
            "program": "nwchem",
            "summary": "Completed water RHF single point.",
            "files": {
                "primary_input": None,
                "primary_output": str(FIXTURE.resolve()),
            },
            "eval_expectations": {
                "diagnosis_failure_class": "no_clear_failure_detected",
                "diagnosis_stage": "single_point",
                "recommended_next_action": (
                    "verify_state_quality_before_accepting_result"
                ),
                "workflow": "verification_only",
                "can_auto_prepare": False,
            },
        }),
        encoding="utf-8",
    )
    return case_path


def test_nwchem_case_evaluation_preserves_diagnosis_and_workflow(tmp_path):
    evaluated = evaluate_case(str(_case(tmp_path)))

    assert evaluated["passed"] is True
    assert evaluated["check_count"] == 5
    assert evaluated["pass_count"] == 5
    assert evaluated["fail_count"] == 0
    assert evaluated["diagnosis"] == {
        "failure_class": "no_clear_failure_detected",
        "stage": "single_point",
        "recommended_next_action": (
            "verify_state_quality_before_accepting_result"
        ),
        "task_outcome": "success",
    }
    assert evaluated["workflow"] == {
        "selected_workflow": "verification_only",
        "can_auto_prepare": False,
        "notes": ["no_automatic_repair_needed"],
    }


def test_case_directory_rollup_is_exact(tmp_path):
    _case(tmp_path)

    evaluated = evaluate_cases(str(tmp_path))

    assert evaluated["case_count"] == 1
    assert evaluated["passed_case_count"] == 1
    assert evaluated["failed_case_count"] == 0
    assert [item["case_id"] for item in evaluated["results"]] == [
        "nwchem.water_rhf"
    ]

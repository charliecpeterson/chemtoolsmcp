"""Shared result-envelope contracts for the guided analysis services."""

from pathlib import Path

from chemtools.application.input_review import review_input
from chemtools.application.recovery_planning import plan_recovery
from chemtools.application.run_comparison import compare_runs
from chemtools.application.run_inspection import inspect_run
from chemtools.mcp.catalog import BUILTIN_BACKENDS, load_backend


FIXTURES = Path(__file__).parent / "golden" / "mcp" / "fixtures"
SHARED_FIELDS = {
    "assessment",
    "evidence",
    "uncertainty",
    "next_actions",
}


def test_guided_analysis_services_share_one_result_envelope():
    backend = load_backend(BUILTIN_BACKENDS[0])
    input_file = FIXTURES / "nwchem_h2.nw"
    output_file = FIXTURES / "nwchem_scf.out"
    results = {
        "review_input": review_input(
            backend,
            input_file,
            resolved_by="explicit",
        ),
        "inspect_run": inspect_run(
            backend,
            output_file,
            resolved_by="explicit",
            artifact_files=[input_file],
        ),
        "compare_runs": compare_runs(
            backend,
            output_file,
            output_file,
            reference_input_file=input_file,
            candidate_input_file=input_file,
        ),
        "plan_recovery": plan_recovery(
            backend,
            output_file,
            input_file=input_file,
        ),
    }

    assert {
        name: result["schema_version"]
        for name, result in results.items()
    } == {
        "review_input": "chemtools.review-input/1",
        "inspect_run": "chemtools.inspect-run/1",
        "compare_runs": "chemtools.compare-runs/1",
        "plan_recovery": "chemtools.plan-recovery/1",
    }
    for result in results.values():
        _assert_analysis_envelope(result)


def _assert_analysis_envelope(result):
    assert SHARED_FIELDS <= result.keys()
    verdict = result["assessment"]["verdict"]
    assert isinstance(verdict["label"], str) and verdict["label"]
    assert not isinstance(verdict["confidence"], bool)
    assert 0.0 <= verdict["confidence"] <= 1.0
    assert isinstance(verdict["reasons"], list)
    assert all(
        isinstance(reason, str) and reason
        for reason in verdict["reasons"]
    )
    assert isinstance(result["evidence"], dict)
    assert isinstance(result["uncertainty"], list)
    for item in result["uncertainty"]:
        assert isinstance(item["code"], str) and item["code"]
        assert isinstance(item["message"], str) and item["message"]
        assert isinstance(item["impact"], str) and item["impact"]
    assert isinstance(result["next_actions"], list)
    for action in result["next_actions"]:
        assert isinstance(action["action"], str) and action["action"]
        assert isinstance(action["reason"], str) and action["reason"]
        assert isinstance(action["priority"], int)

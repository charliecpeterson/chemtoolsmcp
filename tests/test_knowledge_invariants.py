"""Exact contracts for scoped sign and monotonicity checks."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from chemtools.knowledge.invariants import (
    assess_failure_sentinel,
    assess_expected_sign,
    assess_monotonicity,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "knowledge"
    / "numeric_invariants"
    / "synthetic_cases.json"
)
SENTINEL_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "knowledge"
    / "optimizer_sentinels"
    / "synthetic_cases.json"
)


def _cases() -> list[dict[str, object]]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "chemtools.synthetic-numeric-invariants/1"
    assert payload["provenance"] == "synthetic_contract_cases"
    assert payload["units"] == "arbitrary"
    return payload["cases"]


def _sentinel_cases() -> list[dict[str, object]]:
    payload = json.loads(SENTINEL_FIXTURE.read_text(encoding="utf-8"))
    assert payload["schema_version"] == (
        "chemtools.synthetic-optimizer-sentinels/1"
    )
    assert payload["provenance"] == "synthetic_contract_cases"
    assert payload["units"] == "arbitrary"
    return payload["cases"]


def test_synthetic_sign_cases_pin_pass_and_failure():
    cases = {
        case["id"]: case
        for case in _cases()
        if case["check"] == "sign"
    }
    assessments = {
        case_id: assess_expected_sign(
            case["value"],
            case["expected_sign"],
            tolerance=case["tolerance"],
        )
        for case_id, case in cases.items()
    }

    assert {
        case_id: assessment.verdict
        for case_id, assessment in assessments.items()
    } == {
        case_id: case["expected_verdict"]
        for case_id, case in cases.items()
    }
    assert assessments["expected_positive_pass"].to_dict() == {
        "value": 0.18,
        "expected_sign": "positive",
        "tolerance": 0.0,
        "verdict": "satisfied",
        "reasons": [
            "Observed value has the expected positive sign outside the zero "
            "tolerance of 0.0."
        ],
    }


def test_synthetic_monotonicity_cases_pin_pass_and_failure():
    cases = {
        case["id"]: case
        for case in _cases()
        if case["check"] == "monotonicity"
    }
    assessments = {
        case_id: assess_monotonicity(
            case["values"],
            case["expected_direction"],
            tolerance=case["tolerance"],
        )
        for case_id, case in cases.items()
    }

    assert {
        case_id: (
            assessment.verdict,
            [list(pair) for pair in assessment.violating_pairs],
        )
        for case_id, assessment in assessments.items()
    } == {
        case_id: (
            case["expected_verdict"],
            case["expected_violating_pairs"],
        )
        for case_id, case in cases.items()
    }
    assert assessments["nonincreasing_basin_jump_failure"].to_dict() == {
        "values": [-10.0, -10.2, -10.1, -10.3],
        "expected_direction": "nonincreasing",
        "tolerance": 0.0,
        "verdict": "violated",
        "violating_pairs": [[2, 3]],
        "reasons": [
            "Expected nonincreasing ordering is violated between one-based "
            "point pairs: 2->3."
        ],
    }


def test_sign_tolerance_is_a_strict_zero_band():
    assert assess_expected_sign(
        0.01,
        "positive",
        tolerance=0.01,
    ).verdict == "violated"
    assert assess_expected_sign(
        -0.0101,
        "negative",
        tolerance=0.01,
    ).verdict == "satisfied"


def test_monotonicity_tolerance_applies_to_each_adjacent_pair():
    assessment = assess_monotonicity(
        (1.0, 1.0005, 0.9),
        "nonincreasing",
        tolerance=0.001,
    )

    assert assessment.verdict == "satisfied"
    assert assessment.violating_pairs == ()


def test_nondecreasing_check_reports_each_violating_pair():
    assessment = assess_monotonicity(
        (1.0, 0.9, 1.1, 1.0),
        "nondecreasing",
    )

    assert assessment.verdict == "violated"
    assert assessment.violating_pairs == ((1, 2), (3, 4))


def test_failure_sentinels_lose_in_both_objective_directions():
    cases = {case["id"]: case for case in _sentinel_cases()}
    assessments = {
        case_id: assess_failure_sentinel(
            case["failure_value"],
            case["objective_direction"],
            valid_lower_bound=case["valid_lower_bound"],
            valid_upper_bound=case["valid_upper_bound"],
        )
        for case_id, case in cases.items()
    }

    assert {
        case_id: assessment.verdict
        for case_id, assessment in assessments.items()
    } == {
        case_id: case["expected_verdict"]
        for case_id, case in cases.items()
    }
    assert assessments["minimize_attractive_failure"].to_dict() == {
        "failure_value": 1.0,
        "objective_direction": "minimize",
        "valid_objective_bounds": [10.0, 40.0],
        "verdict": "violated",
        "reasons": [
            "Failure value can equal or outrank a valid objective in the "
            "declared interval [10.0, 40.0] for minimize."
        ],
    }


def test_failure_sentinel_must_strictly_lose():
    assert assess_failure_sentinel(
        40.0,
        "minimize",
        valid_lower_bound=10.0,
        valid_upper_bound=40.0,
    ).verdict == "violated"
    assert assess_failure_sentinel(
        10.0,
        "maximize",
        valid_lower_bound=10.0,
        valid_upper_bound=40.0,
    ).verdict == "violated"


@pytest.mark.parametrize(
    ("call", "error", "message"),
    (
        (
            lambda: assess_expected_sign(True, "positive"),
            TypeError,
            "value must be a real number",
        ),
        (
            lambda: assess_expected_sign(math.nan, "positive"),
            ValueError,
            "value must be finite",
        ),
        (
            lambda: assess_expected_sign(1.0, "zero"),
            ValueError,
            "expected_sign",
        ),
        (
            lambda: assess_monotonicity((1.0,), "nonincreasing"),
            ValueError,
            "at least two values",
        ),
        (
            lambda: assess_monotonicity((1.0, 2.0), "increasing"),
            ValueError,
            "expected_direction",
        ),
        (
            lambda: assess_monotonicity(
                (1.0, 2.0),
                "nondecreasing",
                tolerance=-1,
            ),
            ValueError,
            "tolerance must be nonnegative",
        ),
        (
            lambda: assess_failure_sentinel(
                math.nan,
                "minimize",
                valid_lower_bound=0.0,
                valid_upper_bound=1.0,
            ),
            ValueError,
            "failure_value must be finite",
        ),
        (
            lambda: assess_failure_sentinel(
                math.inf,
                "minimize",
                valid_lower_bound=0.0,
                valid_upper_bound=1.0,
            ),
            ValueError,
            "failure_value must be finite",
        ),
        (
            lambda: assess_failure_sentinel(
                2.0,
                "minimize",
                valid_lower_bound=1.0,
                valid_upper_bound=math.inf,
            ),
            ValueError,
            "valid_upper_bound must be finite",
        ),
        (
            lambda: assess_failure_sentinel(
                2.0,
                "minimize",
                valid_lower_bound=2.0,
                valid_upper_bound=1.0,
            ),
            ValueError,
            "valid_lower_bound must not exceed valid_upper_bound",
        ),
        (
            lambda: assess_failure_sentinel(
                2.0,
                "smallest",
                valid_lower_bound=0.0,
                valid_upper_bound=1.0,
            ),
            ValueError,
            "objective_direction",
        ),
    ),
)
def test_numeric_invariant_boundary_rejects_invalid_values(call, error, message):
    with pytest.raises(error, match=message):
        call()

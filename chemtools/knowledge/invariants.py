"""Deterministic numeric checks used by curated knowledge cards.

Callers supply the scientifically scoped expectation. These checks only
compare recorded values against that expectation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal, Sequence


InvariantVerdict = Literal["satisfied", "violated"]
ExpectedSign = Literal["positive", "negative"]
MonotonicDirection = Literal["nondecreasing", "nonincreasing"]
ObjectiveDirection = Literal["minimize", "maximize"]


@dataclass(frozen=True)
class SignAssessment:
    value: float
    expected_sign: ExpectedSign
    tolerance: float
    verdict: InvariantVerdict
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "expected_sign": self.expected_sign,
            "tolerance": self.tolerance,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class MonotonicityAssessment:
    values: tuple[float, ...]
    expected_direction: MonotonicDirection
    tolerance: float
    verdict: InvariantVerdict
    violating_pairs: tuple[tuple[int, int], ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "values": list(self.values),
            "expected_direction": self.expected_direction,
            "tolerance": self.tolerance,
            "verdict": self.verdict,
            "violating_pairs": [list(pair) for pair in self.violating_pairs],
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class FailureSentinelAssessment:
    failure_value: float
    objective_direction: ObjectiveDirection
    valid_objective_bounds: tuple[float, float]
    verdict: InvariantVerdict
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "failure_value": self.failure_value,
            "objective_direction": self.objective_direction,
            "valid_objective_bounds": list(self.valid_objective_bounds),
            "verdict": self.verdict,
            "reasons": list(self.reasons),
        }


def assess_expected_sign(
    value: Real,
    expected_sign: ExpectedSign,
    *,
    tolerance: Real = 0.0,
) -> SignAssessment:
    observed = _finite_number(value, "value")
    threshold = _tolerance(tolerance)
    if expected_sign not in ("positive", "negative"):
        raise ValueError("expected_sign must be 'positive' or 'negative'")

    satisfied = (
        observed > threshold
        if expected_sign == "positive"
        else observed < -threshold
    )
    if satisfied:
        reasons = (
            f"Observed value has the expected {expected_sign} sign outside "
            f"the zero tolerance of {threshold}.",
        )
    else:
        reasons = (
            f"Observed value does not have the expected {expected_sign} sign "
            f"outside the zero tolerance of {threshold}.",
        )

    return SignAssessment(
        value=observed,
        expected_sign=expected_sign,
        tolerance=threshold,
        verdict="satisfied" if satisfied else "violated",
        reasons=reasons,
    )


def assess_monotonicity(
    values: Sequence[Real],
    expected_direction: MonotonicDirection,
    *,
    tolerance: Real = 0.0,
) -> MonotonicityAssessment:
    observed = tuple(
        _finite_number(value, f"values[{index}]")
        for index, value in enumerate(values)
    )
    if len(observed) < 2:
        raise ValueError("at least two values are required")
    if expected_direction not in ("nondecreasing", "nonincreasing"):
        raise ValueError(
            "expected_direction must be 'nondecreasing' or 'nonincreasing'"
        )
    threshold = _tolerance(tolerance)

    violating_pairs = tuple(
        (index + 1, index + 2)
        for index, (left, right) in enumerate(zip(observed, observed[1:]))
        if (
            right < left - threshold
            if expected_direction == "nondecreasing"
            else right > left + threshold
        )
    )
    if violating_pairs:
        locations = ", ".join(
            f"{left}->{right}" for left, right in violating_pairs
        )
        reasons = (
            f"Expected {expected_direction} ordering is violated between "
            f"one-based point pairs: {locations}.",
        )
    else:
        reasons = (
            f"All adjacent values satisfy the expected {expected_direction} "
            f"ordering within tolerance {threshold}.",
        )

    return MonotonicityAssessment(
        values=observed,
        expected_direction=expected_direction,
        tolerance=threshold,
        verdict="violated" if violating_pairs else "satisfied",
        violating_pairs=violating_pairs,
        reasons=reasons,
    )


def assess_failure_sentinel(
    failure_value: Real,
    objective_direction: ObjectiveDirection,
    *,
    valid_lower_bound: Real,
    valid_upper_bound: Real,
) -> FailureSentinelAssessment:
    if objective_direction not in ("minimize", "maximize"):
        raise ValueError(
            "objective_direction must be 'minimize' or 'maximize'"
        )
    sentinel = _finite_number(failure_value, "failure_value")
    lower = _finite_number(valid_lower_bound, "valid_lower_bound")
    upper = _finite_number(valid_upper_bound, "valid_upper_bound")
    if lower > upper:
        raise ValueError(
            "valid_lower_bound must not exceed valid_upper_bound"
        )

    limiting_bound = upper if objective_direction == "minimize" else lower
    dominates = (
        sentinel > limiting_bound
        if objective_direction == "minimize"
        else sentinel < limiting_bound
    )
    if dominates:
        reasons = (
            f"Failure value ranks worse than every objective in the declared "
            f"valid interval [{lower}, {upper}] for {objective_direction}.",
        )
    else:
        reasons = (
            f"Failure value can equal or outrank a valid objective in the "
            f"declared interval [{lower}, {upper}] for {objective_direction}.",
        )

    return FailureSentinelAssessment(
        failure_value=sentinel,
        objective_direction=objective_direction,
        valid_objective_bounds=(lower, upper),
        verdict="satisfied" if dominates else "violated",
        reasons=reasons,
    )


def _finite_number(value: Real, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _tolerance(value: Real) -> float:
    tolerance = _finite_number(value, "tolerance")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    return tolerance


__all__ = [
    "ExpectedSign",
    "FailureSentinelAssessment",
    "InvariantVerdict",
    "MonotonicDirection",
    "MonotonicityAssessment",
    "ObjectiveDirection",
    "SignAssessment",
    "assess_failure_sentinel",
    "assess_expected_sign",
    "assess_monotonicity",
]

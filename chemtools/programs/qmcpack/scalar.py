"""Parse QMCPACK scalar block data into bounded estimator summaries."""

from __future__ import annotations

import re
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Iterable


_MAX_ROWS = 250_000
_SECOND_MOMENT_RELATIVE_TOLERANCE = 1e-12
_ACCEPTANCE_RATIO_BOUND_TOLERANCE = 1e-12
_ENERGY_COMPONENT_RELATIVE_TOLERANCE = 1e-9
_SCALAR_FILENAME_RE = re.compile(
    r"^(?P<project_id>.+)\.s(?P<series_index>\d+)\.scalar\.dat$"
)


def parse_scalar_file(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    return _scalar_summary(_parse_file_rows(source), path=source)


def parse_scalar_text(
    text: str,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    return _scalar_summary(_parse_rows(text), path=path)


def _scalar_summary(
    parsed_rows: dict[str, Any],
    *,
    path: Path | None,
) -> dict[str, Any]:
    columns = parsed_rows["columns"]
    values = parsed_rows["values"]
    row_count = parsed_rows["row_count"]
    estimators = {
        column: _summary(values[column])
        for column in columns
        if column != "index"
    }
    local_energy = estimators.get("LocalEnergy")
    if local_energy is not None:
        local_energy["weighted_mean"] = _weighted_mean(
            values["LocalEnergy"],
            values.get("BlockWeight"),
        )
        estimators["LocalEnergy"] = local_energy
    return {
        "schema_version": "chemtools.qmcpack-scalar/1",
        "path": str(path) if path is not None else None,
        "filename_identity": scalar_filename_identity(path),
        "columns": columns,
        "row_count": row_count,
        "first_block": int(values["index"][0]),
        "last_block": int(values["index"][-1]),
        "block_index_sequence": block_index_sequence(values["index"]),
        "invalid_row_count": parsed_rows["invalid_row_count"],
        "invalid_row_reasons": parsed_rows["invalid_row_reasons"],
        "truncated": parsed_rows["truncated"],
        "estimators": estimators,
        "local_energy_second_moment": _local_energy_second_moment(values),
        "acceptance_ratio_bounds": _acceptance_ratio_bounds(values),
        "block_weight_quality": _block_weight_quality(values),
        "energy_component_balance": _energy_component_balance(values),
    }


def scalar_filename_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    match = _SCALAR_FILENAME_RE.match(path.name)
    if match is None:
        return {
            "status": "unrecognized",
            "filename": path.name,
            "scope_limit": (
                "The scalar filename does not match the supported project.sNNN.scalar.dat "
                "convention."
            ),
        }
    return {
        "status": "recognized",
        "project_id": match.group("project_id"),
        "series_index": int(match.group("series_index")),
        "scope_limit": (
            "Filename identity does not establish the source QMC input block or "
            "its controls."
        ),
    }


def read_scalar_rows(path: str | Path) -> dict[str, Any]:
    """Read bounded numeric scalar rows for analysis that needs block order."""
    source = Path(path).expanduser().resolve()
    parsed_rows = _parse_file_rows(source)
    return {
        "path": str(source),
        **parsed_rows,
        "block_index_sequence": block_index_sequence(parsed_rows["values"]["index"]),
        "local_energy_second_moment": _local_energy_second_moment(
            parsed_rows["values"]
        ),
        "acceptance_ratio_bounds": _acceptance_ratio_bounds(parsed_rows["values"]),
        "block_weight_quality": _block_weight_quality(parsed_rows["values"]),
        "energy_component_balance": _energy_component_balance(parsed_rows["values"]),
    }


def _parse_rows(text: str) -> dict[str, Any]:
    return _parse_lines(text.splitlines())


def _parse_file_rows(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", errors="replace") as stream:
        return _parse_lines(stream)


def _parse_lines(lines: Iterable[str]) -> dict[str, Any]:
    columns: list[str] | None = None
    values: dict[str, list[float]] = {}
    invalid_rows = 0
    malformed_rows = 0
    nonfinite_rows = 0
    noninteger_index_rows = 0
    truncated = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if columns is None:
                candidate = stripped.removeprefix("#").split()
                if candidate and candidate[0] == "index":
                    columns = candidate
                    values = {column: [] for column in columns}
            continue
        if columns is None:
            continue
        fields = stripped.split()
        if len(fields) != len(columns):
            invalid_rows += 1
            malformed_rows += 1
            continue
        try:
            row = [float(value) for value in fields]
        except ValueError:
            invalid_rows += 1
            malformed_rows += 1
            continue
        if not all(isfinite(value) for value in row):
            invalid_rows += 1
            nonfinite_rows += 1
            continue
        if not row[0].is_integer():
            invalid_rows += 1
            noninteger_index_rows += 1
            continue
        if len(values["index"]) >= _MAX_ROWS:
            truncated = True
            break
        for column, value in zip(columns, row):
            values[column].append(value)

    if columns is None:
        raise ValueError("QMCPACK scalar data has no '#' column header.")
    row_count = len(values["index"])
    if row_count == 0:
        rejected = []
        for count, reason in (
            (malformed_rows, "malformed"),
            (nonfinite_rows, "non-finite"),
            (noninteger_index_rows, "non-integral-index"),
        ):
            if count:
                rejected.append(f"{count} {reason} row(s)")
        message = "QMCPACK scalar data contains no valid block rows."
        if rejected:
            message += " Rejected: " + ", ".join(rejected) + "."
        raise ValueError(message)
    return {
        "columns": columns,
        "values": values,
        "row_count": row_count,
        "invalid_row_count": invalid_rows,
        "invalid_row_reasons": {
            "malformed": malformed_rows,
            "non_finite": nonfinite_rows,
            "non_integral_index": noninteger_index_rows,
        },
        "truncated": truncated,
    }


def block_index_sequence(indices: list[float]) -> dict[str, int | str]:
    gap_transitions = 0
    nonincreasing_transitions = 0
    for previous, current in zip(indices, indices[1:]):
        if current > previous + 1:
            gap_transitions += 1
        elif current <= previous:
            nonincreasing_transitions += 1
    return {
        "status": (
            "contiguous"
            if not gap_transitions and not nonincreasing_transitions
            else "noncontiguous"
        ),
        "gap_transition_count": gap_transitions,
        "nonincreasing_transition_count": nonincreasing_transitions,
    }


def _summary(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    variance = (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1 else 0.0
    )
    return {
        "mean": mean,
        "sample_stdev": sqrt(variance),
        "minimum": min(values),
        "maximum": max(values),
        "last": values[-1],
    }


def _weighted_mean(
    values: list[float],
    weights: list[float] | None,
) -> float | None:
    if weights is None or len(weights) != len(values) or any(
        weight <= 0 for weight in weights
    ):
        return None
    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def _local_energy_second_moment(values: dict[str, list[float]]) -> dict[str, Any]:
    local_energy = values.get("LocalEnergy")
    second_moment = values.get("LocalEnergy_sq")
    if local_energy is None or second_moment is None:
        return {
            "status": "not_available",
            "reason": "LocalEnergy and LocalEnergy_sq columns are required.",
        }
    differences = [
        moment - energy**2
        for energy, moment in zip(local_energy, second_moment)
    ]
    violations = [
        difference
        for energy, moment, difference in zip(local_energy, second_moment, differences)
        if difference < -_SECOND_MOMENT_RELATIVE_TOLERANCE
        * max(1.0, abs(moment), energy**2)
    ]
    return {
        "status": "consistent" if not violations else "inconsistent",
        "block_count": len(differences),
        "minimum_second_moment_minus_mean_squared": min(differences),
        "negative_bound_violation_count": len(violations),
        "relative_tolerance": _SECOND_MOMENT_RELATIVE_TOLERANCE,
        "scope_limit": (
            "This checks the recorded per-block second-moment bound; it does not "
            "estimate statistical uncertainty or convergence."
        ),
    }


def _acceptance_ratio_bounds(values: dict[str, list[float]]) -> dict[str, Any]:
    acceptance_ratio = values.get("AcceptRatio")
    if acceptance_ratio is None:
        return {
            "status": "not_available",
            "reason": "An AcceptRatio column is required.",
        }
    lower_violations = sum(
        ratio < -_ACCEPTANCE_RATIO_BOUND_TOLERANCE
        for ratio in acceptance_ratio
    )
    upper_violations = sum(
        ratio > 1 + _ACCEPTANCE_RATIO_BOUND_TOLERANCE
        for ratio in acceptance_ratio
    )
    return {
        "status": (
            "within_bounds"
            if not lower_violations and not upper_violations
            else "out_of_bounds"
        ),
        "block_count": len(acceptance_ratio),
        "minimum": min(acceptance_ratio),
        "maximum": max(acceptance_ratio),
        "below_zero_count": lower_violations,
        "above_one_count": upper_violations,
        "bound_tolerance": _ACCEPTANCE_RATIO_BOUND_TOLERANCE,
        "scope_limit": (
            "This checks the recorded acceptance-ratio bounds; it does not "
            "recommend an acceptance-rate target."
        ),
    }


def _block_weight_quality(values: dict[str, list[float]]) -> dict[str, Any]:
    block_weight = values.get("BlockWeight")
    if block_weight is None:
        return {
            "status": "not_available",
            "reason": "A BlockWeight column is required.",
        }
    zero_count = sum(weight == 0 for weight in block_weight)
    negative_count = sum(weight < 0 for weight in block_weight)
    return {
        "status": "positive" if not zero_count and not negative_count else "nonpositive",
        "block_count": len(block_weight),
        "minimum": min(block_weight),
        "maximum": max(block_weight),
        "zero_count": zero_count,
        "negative_count": negative_count,
        "scope_limit": (
            "This reports recorded block-weight sign only; it does not change "
            "unweighted scalar analyses."
        ),
    }


def _energy_component_balance(values: dict[str, list[float]]) -> dict[str, Any]:
    local_energy = values.get("LocalEnergy")
    kinetic = values.get("Kinetic")
    local_potential = values.get("LocalPotential")
    if local_energy is None or kinetic is None or local_potential is None:
        return {
            "status": "not_available",
            "reason": "LocalEnergy, Kinetic, and LocalPotential columns are required.",
        }
    residuals = [
        energy - kinetic_energy - potential_energy
        for energy, kinetic_energy, potential_energy in zip(
            local_energy,
            kinetic,
            local_potential,
        )
    ]
    violations = sum(
        abs(residual) > _ENERGY_COMPONENT_RELATIVE_TOLERANCE
        * max(1.0, abs(energy), abs(kinetic_energy), abs(potential_energy))
        for energy, kinetic_energy, potential_energy, residual in zip(
            local_energy,
            kinetic,
            local_potential,
            residuals,
        )
    )
    return {
        "status": "balanced" if not violations else "unbalanced",
        "block_count": len(residuals),
        "minimum_residual": min(residuals),
        "maximum_residual": max(residuals),
        "maximum_absolute_residual": max(abs(residual) for residual in residuals),
        "violation_count": violations,
        "relative_tolerance": _ENERGY_COMPONENT_RELATIVE_TOLERANCE,
        "scope_limit": (
            "This checks the reported LocalEnergy, Kinetic, and LocalPotential "
            "balance only; it does not establish Hamiltonian completeness."
        ),
    }


__all__ = [
    "block_index_sequence",
    "parse_scalar_file",
    "parse_scalar_text",
    "read_scalar_rows",
    "scalar_filename_identity",
]

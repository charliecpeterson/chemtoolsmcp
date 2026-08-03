"""Evaluate small, source-backed QMCPACK workflow gates."""

from __future__ import annotations

from math import isfinite
from typing import Any

from chemtools.programs.qmcpack.scalar import read_scalar_rows


def inspect_determinant_only_vmc_offsets(
    runs: list[dict[str, Any]],
    *,
    discard_fraction: float = 0.0,
) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("runs must contain at least two determinant-only VMC results.")
    if not 0 <= discard_fraction < 1:
        raise ValueError("discard_fraction must be at least 0 and less than 1.")

    points = [
        _determinant_only_vmc_point(run, discard_fraction)
        for run in runs
    ]
    offsets = [point["vmc_minus_trial_hartree"] for point in points]
    return {
        "schema_version": "chemtools.qmcpack-determinant-vmc-offsets/1",
        "discard_fraction": discard_fraction,
        "points": points,
        "all_vmc_minus_trial_offsets_positive": all(offset > 0 for offset in offsets),
        "offset_trend_in_supplied_state_order": _offset_trend(offsets),
        "warnings": _scalar_quality_warnings(points),
        "scope_limit": (
            "The caller identifies these as determinant-only VMC runs and supplies "
            "the expected state order. This reports offsets without setting a "
            "small-offset threshold or proving Hamiltonian consistency."
        ),
    }


def check_vmc_energy_gate(
    scalar_file: str,
    *,
    trial_scf_energy_hartree: float,
    discard_fraction: float = 0.0,
) -> dict[str, Any]:
    if not _finite_number(trial_scf_energy_hartree):
        raise ValueError("trial_scf_energy_hartree must be finite.")
    if not 0 <= discard_fraction < 1:
        raise ValueError("discard_fraction must be at least 0 and less than 1.")

    parsed, retained_energy, discarded_block_count = _retained_local_energy(
        scalar_file,
        discard_fraction,
    )

    vmc_energy = sum(retained_energy) / len(retained_energy)
    energy_difference = vmc_energy - trial_scf_energy_hartree
    return {
        "schema_version": "chemtools.qmcpack-vmc-energy-gate/1",
        "scalar_file": parsed["path"],
        "source_block_count": parsed["row_count"],
        "discard_fraction": discard_fraction,
        "discarded_block_count": discarded_block_count,
        "retained_block_count": len(retained_energy),
        "vmc_energy_hartree": vmc_energy,
        "trial_scf_energy_hartree": trial_scf_energy_hartree,
        "vmc_minus_trial_hartree": energy_difference,
        "gate": {
            "status": "passed" if energy_difference <= 0 else "failed",
            "criterion": "VMC LocalEnergy <= trial SCF energy",
        },
        "invalid_row_count": parsed["invalid_row_count"],
        "invalid_row_reasons": parsed["invalid_row_reasons"],
        "local_energy_second_moment": parsed["local_energy_second_moment"],
        "acceptance_ratio_bounds": parsed["acceptance_ratio_bounds"],
        "block_weight_quality": parsed["block_weight_quality"],
        "energy_component_balance": parsed["energy_component_balance"],
        "truncated": parsed["truncated"],
        "block_index_sequence": parsed["block_index_sequence"],
        "warnings": _scalar_quality_warnings([parsed]),
        "statistical_limit": (
            "This comparison uses the retained scalar-block mean without "
            "autocorrelation or reblocking analysis."
        ),
    }


def _determinant_only_vmc_point(
    run: dict[str, Any],
    discard_fraction: float,
) -> dict[str, Any]:
    state_label = run.get("state_label")
    scalar_file = run.get("scalar_file")
    trial_energy = run.get("trial_scf_energy_hartree")
    if not isinstance(state_label, str) or not state_label:
        raise ValueError("Every run needs a non-empty state_label.")
    if not isinstance(scalar_file, str) or not scalar_file:
        raise ValueError("Every run needs a non-empty scalar_file.")
    if not _finite_number(trial_energy):
        raise ValueError("Every run needs a finite trial_scf_energy_hartree.")
    parsed, retained_energy, discarded_block_count = _retained_local_energy(
        scalar_file,
        discard_fraction,
    )
    vmc_energy = sum(retained_energy) / len(retained_energy)
    return {
        "state_label": state_label,
        "scalar_file": parsed["path"],
        "source_block_count": parsed["row_count"],
        "discarded_block_count": discarded_block_count,
        "retained_block_count": len(retained_energy),
        "vmc_energy_hartree": vmc_energy,
        "trial_scf_energy_hartree": trial_energy,
        "vmc_minus_trial_hartree": vmc_energy - trial_energy,
        "invalid_row_count": parsed["invalid_row_count"],
        "invalid_row_reasons": parsed["invalid_row_reasons"],
        "local_energy_second_moment": parsed["local_energy_second_moment"],
        "acceptance_ratio_bounds": parsed["acceptance_ratio_bounds"],
        "block_weight_quality": parsed["block_weight_quality"],
        "energy_component_balance": parsed["energy_component_balance"],
        "truncated": parsed["truncated"],
        "block_index_sequence": parsed["block_index_sequence"],
    }


def _offset_trend(offsets: list[float]) -> str:
    if all(earlier > later for earlier, later in zip(offsets, offsets[1:])):
        return "strictly_decreasing"
    if all(earlier < later for earlier, later in zip(offsets, offsets[1:])):
        return "strictly_increasing"
    return "not_strictly_monotonic"


def _retained_local_energy(
    scalar_file: str,
    discard_fraction: float,
) -> tuple[dict[str, Any], list[float], int]:
    parsed = read_scalar_rows(scalar_file)
    if "LocalEnergy" not in parsed["columns"]:
        raise ValueError(f"{parsed['path']} has no LocalEnergy column.")
    energy = parsed["values"]["LocalEnergy"]
    discarded_block_count = int(len(energy) * discard_fraction)
    retained_energy = energy[discarded_block_count:]
    if not retained_energy:
        raise ValueError(f"{parsed['path']} has no retained LocalEnergy blocks.")
    return parsed, retained_energy, discarded_block_count


def _scalar_quality_warnings(records: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if any(record["invalid_row_reasons"]["malformed"] for record in records):
        warnings.append(
            "At least one scalar file contains malformed rows that were excluded."
        )
    if any(record["invalid_row_reasons"]["non_finite"] for record in records):
        warnings.append(
            "At least one scalar file contains non-finite rows that were excluded."
        )
    if any(
        record["invalid_row_reasons"]["non_integral_index"]
        for record in records
    ):
        warnings.append(
            "At least one scalar file contains non-integral-index rows that were excluded."
        )
    if any(
        record["local_energy_second_moment"]["status"] == "inconsistent"
        for record in records
    ):
        warnings.append(
            "At least one scalar file has LocalEnergy_sq below the LocalEnergy "
            "second-moment bound."
        )
    if any(
        record["acceptance_ratio_bounds"]["status"] == "out_of_bounds"
        for record in records
    ):
        warnings.append(
            "At least one scalar file has AcceptRatio values outside [0, 1]."
        )
    if any(
        record["block_weight_quality"]["status"] == "nonpositive"
        for record in records
    ):
        warnings.append(
            "At least one scalar file has non-positive BlockWeight values; its "
            "weighted LocalEnergy mean is unavailable."
        )
    if any(
        record["energy_component_balance"]["status"] == "unbalanced"
        for record in records
    ):
        warnings.append(
            "At least one scalar file has an unbalanced LocalEnergy, Kinetic, "
            "and LocalPotential record."
        )
    if any(
        record["block_index_sequence"]["status"] == "noncontiguous"
        for record in records
    ):
        warnings.append(
            "At least one scalar file has gaps or nonincreasing block indices."
        )
    if any(record["truncated"] for record in records):
        warnings.append(
            "At least one scalar file hit the row limit; its analysis point is incomplete."
        )
    return warnings


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


__all__ = ["check_vmc_energy_gate", "inspect_determinant_only_vmc_offsets"]

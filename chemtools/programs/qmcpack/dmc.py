"""Analyze explicitly labelled QMCPACK DMC scalar series."""

from __future__ import annotations

from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Iterable

from chemtools.programs.qmcpack.input import parse_qmcpack_input
from chemtools.programs.qmcpack.scalar import (
    block_index_sequence,
    read_scalar_rows,
    scalar_filename_identity,
)


_MAX_DMC_ROWS = 250_000


def analyze_dmc_series(
    runs: list[dict[str, Any]],
    *,
    discard_fraction: float = 0.25,
    reblock_count: int = 32,
) -> dict[str, Any]:
    if not 0 <= discard_fraction < 1:
        raise ValueError("discard_fraction must be at least 0 and less than 1.")
    if reblock_count < 2:
        raise ValueError("reblock_count must be at least 2.")
    if not runs:
        raise ValueError("runs must contain at least one DMC scalar file.")

    points = [
        _analyze_run(
            run,
            discard_fraction=discard_fraction,
            reblock_count=reblock_count,
        )
        for run in runs
    ]
    _require_distinct_scalar_files(points, "DMC series")
    groups: dict[bool, list[dict[str, Any]]] = {True: [], False: []}
    for point in points:
        groups[point["nonlocalmoves"]].append(point)

    potential_identity = _potential_identity(runs)
    fits = {
        _tmove_label(nonlocalmoves): (
            _mixed_potential_fit(group)
            if potential_identity["status"] == "mixed"
            else _fit_group(group)
        )
        for nonlocalmoves, group in groups.items()
        if group
    }
    warnings = _series_warnings(points, fits, potential_identity)
    return {
        "schema_version": "chemtools.qmcpack-dmc-series/1",
        "discard_fraction": discard_fraction,
        "requested_reblock_count": reblock_count,
        "points": points,
        "fits": fits,
        "potential_identity": potential_identity,
        "warnings": warnings,
    }


def analyze_dmc_input_series(
    qmcpack_input: str | Path,
    runs: list[dict[str, Any]],
    *,
    discard_fraction: float = 0.25,
    reblock_count: int = 32,
) -> dict[str, Any]:
    """Analyze scalar files bound by the caller to DMC blocks in one input."""
    input_path = Path(qmcpack_input).expanduser().resolve()
    dmc_blocks, input_project_id = _dmc_input_blocks(input_path)
    analysis_runs = []
    bindings = []
    for run in runs:
        analysis_run, binding = _input_bound_dmc_run(
            run,
            dmc_blocks,
            input_path,
            input_project_id,
        )
        analysis_runs.append(analysis_run)
        bindings.append(binding)
    return {
        "schema_version": "chemtools.qmcpack-dmc-input-series/1",
        "qmcpack_input": str(input_path),
        "bindings": bindings,
        "binding_warnings": _binding_warnings(bindings),
        "analysis": analyze_dmc_series(
            analysis_runs,
            discard_fraction=discard_fraction,
            reblock_count=reblock_count,
        ),
        "scope_limit": (
            "The caller supplies each scalar-file-to-QMC-block association because "
            "a scalar file does not record the source QMC block. This does not "
            "establish that the selected block produced the file. Selection is "
            "limited to direct QMC blocks in the supplied primary XML; included "
            "XML is not merged."
        ),
    }


def compare_tmove_locality_shift_from_input(
    qmcpack_input: str | Path,
    tmove: dict[str, Any],
    no_tmove: dict[str, Any],
    *,
    discard_fraction: float = 0.25,
    reblock_count: int = 32,
) -> dict[str, Any]:
    """Compare a caller-bound T-move pair using controls from one input."""
    input_path = Path(qmcpack_input).expanduser().resolve()
    dmc_blocks, input_project_id = _dmc_input_blocks(input_path)
    tmove_run, tmove_binding = _input_bound_dmc_run(
        tmove,
        dmc_blocks,
        input_path,
        input_project_id,
    )
    no_tmove_run, no_tmove_binding = _input_bound_dmc_run(
        no_tmove,
        dmc_blocks,
        input_path,
        input_project_id,
    )
    if tmove_run["nonlocalmoves"] is not True:
        raise ValueError("tmove must select a DMC block with nonlocalmoves enabled.")
    if no_tmove_run["nonlocalmoves"] is not False:
        raise ValueError("no_tmove must select a DMC block with nonlocalmoves disabled.")
    return {
        "schema_version": "chemtools.qmcpack-tmove-locality-shift-input/1",
        "qmcpack_input": str(input_path),
        "tmove_binding": tmove_binding,
        "no_tmove_binding": no_tmove_binding,
        "binding_warnings": _binding_warnings(
            [tmove_binding, no_tmove_binding]
        ),
        "comparison": compare_tmove_locality_shift(
            tmove_run,
            no_tmove_run,
            discard_fraction=discard_fraction,
            reblock_count=reblock_count,
        ),
        "scope_limit": (
            "The caller supplies each scalar-file-to-QMC-block association because "
            "a scalar file does not record the source QMC block. This verifies the "
            "selected input controls, not that those blocks produced the files. "
            "Selection is limited to direct QMC blocks in the supplied primary XML; "
            "included XML is not merged."
        ),
    }


def inspect_dmc_population(
    path: str | Path,
    *,
    target_walkers: float | None = None,
    discard_fraction: float = 0.25,
) -> dict[str, Any]:
    if target_walkers is not None and not _positive_number(target_walkers):
        raise ValueError("target_walkers must be a positive finite number when set.")
    if not 0 <= discard_fraction < 1:
        raise ValueError("discard_fraction must be at least 0 and less than 1.")

    source = Path(path).expanduser().resolve()
    parsed = _read_dmc_rows(source)
    discarded_blocks = int(parsed["row_count"] * discard_fraction)
    values = {
        column: column_values[discarded_blocks:]
        for column, column_values in parsed["values"].items()
    }
    if "NumOfWalkers" not in values:
        raise ValueError(f"{source} has no NumOfWalkers column.")

    walkers = _summary(values["NumOfWalkers"])
    walkers["target_walkers"] = target_walkers
    if target_walkers is not None:
        walkers["mean_relative_deviation"] = (
            walkers["mean"] - target_walkers
        ) / target_walkers
        walkers["last_relative_deviation"] = (
            walkers["last"] - target_walkers
        ) / target_walkers
    return {
        "schema_version": "chemtools.qmcpack-dmc-population/1",
        "path": str(source),
        "columns": parsed["columns"],
        "row_count": parsed["row_count"],
        "invalid_row_count": parsed["invalid_row_count"],
        "invalid_row_reasons": parsed["invalid_row_reasons"],
        "truncated": parsed["truncated"],
        "block_index_sequence": parsed["block_index_sequence"],
        "warnings": _population_quality_warnings(parsed),
        "discard_fraction": discard_fraction,
        "discarded_block_count": discarded_blocks,
        "retained_block_count": len(values["Index"]),
        "walkers": walkers,
        "living_fraction": _summary_or_none(values.get("LivingFraction")),
        "diffusion_efficiency": _summary_or_none(values.get("DiffEff")),
    }


def inspect_dmc_population_from_input(
    qmcpack_input: str | Path,
    dmc_file: str | Path,
    qmc_block_index: int,
    *,
    discard_fraction: float = 0.25,
) -> dict[str, Any]:
    """Inspect one DMC population file against a caller-selected input block."""
    input_path = Path(qmcpack_input).expanduser().resolve()
    dmc_blocks, _ = _dmc_input_blocks(input_path)
    dmc_input = _selected_dmc_input(
        qmc_block_index,
        dmc_blocks,
        input_path,
    )
    population_path = Path(dmc_file).expanduser().resolve()
    return {
        "schema_version": "chemtools.qmcpack-dmc-population-input/1",
        "qmcpack_input": str(input_path),
        "dmc_file": str(population_path),
        "qmc_block_index": qmc_block_index,
        "dmc_input": dmc_input,
        "population": inspect_dmc_population(
            population_path,
            target_walkers=dmc_input["target_walkers"],
            discard_fraction=discard_fraction,
        ),
        "scope_limit": (
            "The caller supplies the DMC-file-to-QMC-block association because a "
            "population file does not record the source QMC block. This uses the "
            "selected walker target without establishing file provenance. Selection "
            "is limited to direct QMC blocks in the supplied primary XML; included "
            "XML is not merged."
        ),
    }
def compare_tmove_locality_shift(
    tmove: dict[str, Any],
    no_tmove: dict[str, Any],
    *,
    discard_fraction: float = 0.25,
    reblock_count: int = 32,
) -> dict[str, Any]:
    tmove_point = _analyze_run(
        {**tmove, "nonlocalmoves": True},
        discard_fraction=discard_fraction,
        reblock_count=reblock_count,
    )
    no_tmove_point = _analyze_run(
        {**no_tmove, "nonlocalmoves": False},
        discard_fraction=discard_fraction,
        reblock_count=reblock_count,
    )
    _require_distinct_scalar_files([tmove_point, no_tmove_point], "T-move locality shift")
    if tmove_point["timestep"] != no_tmove_point["timestep"]:
        raise ValueError("T-move locality-shift comparison requires matching timesteps.")
    potential_identity = _paired_potential_identity(tmove, no_tmove)
    locality_shift = (
        no_tmove_point["local_energy_mean"]
        - tmove_point["local_energy_mean"]
    )
    warnings = _point_quality_warnings([tmove_point, no_tmove_point])
    return {
        "schema_version": "chemtools.qmcpack-tmove-locality-shift/1",
        "timestep": tmove_point["timestep"],
        "discard_fraction": discard_fraction,
        "requested_reblock_count": reblock_count,
        "tmove": tmove_point,
        "no_tmove": no_tmove_point,
        "no_tmove_minus_tmove_hartree": locality_shift,
        "standard_error_hartree": sqrt(
            tmove_point["local_energy_standard_error"] ** 2
            + no_tmove_point["local_energy_standard_error"] ** 2
        ),
        "walker_count_comparability": _walker_count_comparability(
            tmove_point["target_walkers"],
            no_tmove_point["target_walkers"],
        ),
        "potential_identity": potential_identity,
        "warnings": warnings,
        "statistical_limit": (
            "The uncertainty propagates the two reblocked standard errors; "
            "it does not establish autocorrelation convergence."
        ),
    }


def _analyze_run(
    run: dict[str, Any],
    *,
    discard_fraction: float,
    reblock_count: int,
) -> dict[str, Any]:
    scalar_file = run.get("scalar_file")
    timestep = run.get("timestep")
    nonlocalmoves = run.get("nonlocalmoves")
    if not isinstance(scalar_file, str) or not scalar_file:
        raise ValueError("Every run needs a non-empty scalar_file.")
    if not _positive_number(timestep):
        raise ValueError("Every run needs a positive finite timestep.")
    if not isinstance(nonlocalmoves, bool):
        raise ValueError("Every run needs a boolean nonlocalmoves value.")
    target_walkers = run.get("target_walkers")
    if target_walkers is not None and not _positive_number(target_walkers):
        raise ValueError("target_walkers must be a positive finite number when set.")

    parsed = read_scalar_rows(scalar_file)
    if "LocalEnergy" not in parsed["columns"]:
        raise ValueError(f"{parsed['path']} has no LocalEnergy column.")
    energy = parsed["values"]["LocalEnergy"]
    discarded_blocks = int(len(energy) * discard_fraction)
    retained = energy[discarded_blocks:]
    if len(retained) < 2:
        raise ValueError(
            f"{parsed['path']} has fewer than two retained LocalEnergy blocks."
        )
    blocks = _reblock(retained, reblock_count)
    standard_error = _standard_error(blocks)
    block_weights = parsed["values"].get("BlockWeight")
    retained_block_weights = (
        block_weights[discarded_blocks:] if block_weights is not None else None
    )
    return {
        "scalar_file": parsed["path"],
        "timestep": float(timestep),
        "nonlocalmoves": nonlocalmoves,
        "target_walkers": target_walkers,
        "source_block_count": parsed["row_count"],
        "discarded_block_count": discarded_blocks,
        "retained_block_count": len(retained),
        "reblock_count": len(blocks),
        "local_energy_mean": _mean(blocks),
        "local_energy_standard_error": standard_error,
        "invalid_row_count": parsed["invalid_row_count"],
        "invalid_row_reasons": parsed["invalid_row_reasons"],
        "local_energy_second_moment": parsed["local_energy_second_moment"],
        "acceptance_ratio_bounds": parsed["acceptance_ratio_bounds"],
        "block_weight_quality": parsed["block_weight_quality"],
        "energy_component_balance": parsed["energy_component_balance"],
        "truncated": parsed["truncated"],
        "block_index_sequence": parsed["block_index_sequence"],
        "block_weights_constant": _constant(retained_block_weights),
    }


def _dmc_input_blocks(
    input_path: Path,
) -> tuple[dict[int, dict[str, Any]], str | None]:
    parsed_input = parse_qmcpack_input(input_path)
    campaign = parsed_input["dmc_campaign"]
    project = parsed_input["project"]
    project_id = project["id"] if project is not None else None
    if campaign is None:
        return {}, project_id
    return (
        {
            block["qmc_block_index"]: block
            for block in campaign["dmc_blocks"]
        },
        project_id,
    )


def _input_bound_dmc_run(
    run: dict[str, Any],
    dmc_blocks: dict[int, dict[str, Any]],
    input_path: Path,
    input_project_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    block_index = run.get("qmc_block_index")
    dmc_input = _selected_dmc_input(block_index, dmc_blocks, input_path)
    if dmc_input["timestep"] is None or dmc_input["nonlocalmoves"] is None:
        raise ValueError(
            f"DMC block {block_index} needs explicit positive timestep and nonlocalmoves controls."
        )
    scalar_file = Path(run["scalar_file"]).expanduser().resolve()
    analysis_run = {
        "scalar_file": str(scalar_file),
        "timestep": dmc_input["timestep"],
        "nonlocalmoves": dmc_input["nonlocalmoves"],
    }
    if dmc_input["target_walkers"] is not None:
        analysis_run["target_walkers"] = dmc_input["target_walkers"]
    if isinstance(run.get("potential_label"), str) and run["potential_label"]:
        analysis_run["potential_label"] = run["potential_label"]
    return analysis_run, {
        "scalar_file": str(scalar_file),
        "qmc_block_index": block_index,
        "dmc_input": dmc_input,
        "scalar_filename_project": _scalar_filename_project_review(
            input_project_id,
            scalar_filename_identity(scalar_file),
        ),
    }


def _scalar_filename_project_review(
    input_project_id: str | None,
    identity: dict[str, Any] | None,
) -> dict[str, Any]:
    if input_project_id is None:
        return {
            "status": "not_checked",
            "reason": "The QMCPACK input has no project id.",
        }
    if identity is None or identity["status"] != "recognized":
        return {
            "status": "not_checked",
            "input_project_id": input_project_id,
            "reason": "The scalar filename does not have a recognized project label.",
        }
    scalar_project_id = identity["project_id"]
    return {
        "status": "match" if scalar_project_id == input_project_id else "mismatch",
        "input_project_id": input_project_id,
        "scalar_project_id": scalar_project_id,
        "scope_limit": (
            "This compares the filename project label only; it does not establish "
            "the source QMC block or its controls."
        ),
    }


def _binding_warnings(bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings = []
    for binding in bindings:
        review = binding["scalar_filename_project"]
        if review["status"] != "mismatch":
            continue
        warnings.append({
            "code": "scalar_filename_project_mismatch",
            "scalar_file": binding["scalar_file"],
            "input_project_id": review["input_project_id"],
            "scalar_project_id": review["scalar_project_id"],
            "message": (
                "The scalar filename project label does not match the QMCPACK "
                "input project ID."
            ),
        })
    return warnings


def _selected_dmc_input(
    block_index: object,
    dmc_blocks: dict[int, dict[str, Any]],
    input_path: Path,
) -> dict[str, Any]:
    if not isinstance(block_index, int) or isinstance(block_index, bool):
        raise ValueError("Each run needs an integer qmc_block_index.")
    dmc_input = dmc_blocks.get(block_index)
    if dmc_input is None:
        raise ValueError(
            f"qmc_block_index {block_index} does not select a DMC block in the "
            f"primary QMCPACK XML {input_path}; input-bound analysis does not merge "
            "included XML."
        )
    return dmc_input


def _require_distinct_scalar_files(
    points: list[dict[str, Any]],
    analysis_name: str,
) -> None:
    scalar_files = [point["scalar_file"] for point in points]
    if len(set(scalar_files)) != len(scalar_files):
        raise ValueError(f"{analysis_name} requires distinct scalar_file values.")


def _reblock(values: list[float], requested_count: int) -> list[float]:
    count = min(requested_count, len(values))
    return [
        _mean(values[start * len(values) // count:(start + 1) * len(values) // count])
        for start in range(count)
    ]


def _fit_group(points: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(points, key=lambda point: point["timestep"])
    if len({point["timestep"] for point in ordered}) < 2:
        return {
            "status": "not_fit",
            "reason": "at least two distinct positive timesteps are required",
            "point_count": len(ordered),
        }
    if any(point["local_energy_standard_error"] == 0 for point in ordered):
        return {
            "status": "not_fit",
            "reason": "every point needs a non-zero reblocked standard error",
            "point_count": len(ordered),
        }

    weights = [1 / point["local_energy_standard_error"] ** 2 for point in ordered]
    timesteps = [point["timestep"] for point in ordered]
    energies = [point["local_energy_mean"] for point in ordered]
    weight_sum = sum(weights)
    weight_timestep = sum(weight * timestep for weight, timestep in zip(weights, timesteps))
    weight_energy = sum(weight * energy for weight, energy in zip(weights, energies))
    weight_timestep_squared = sum(
        weight * timestep ** 2 for weight, timestep in zip(weights, timesteps)
    )
    weight_timestep_energy = sum(
        weight * timestep * energy
        for weight, timestep, energy in zip(weights, timesteps, energies)
    )
    denominator = weight_sum * weight_timestep_squared - weight_timestep ** 2
    if denominator <= 0:
        return {
            "status": "not_fit",
            "reason": "weighted time-step fit is singular",
            "point_count": len(ordered),
        }
    intercept = (
        weight_timestep_squared * weight_energy
        - weight_timestep * weight_timestep_energy
    ) / denominator
    slope = (
        weight_sum * weight_timestep_energy - weight_timestep * weight_energy
    ) / denominator
    chi_squared = sum(
        weight * (energy - (intercept + slope * timestep)) ** 2
        for weight, timestep, energy in zip(weights, timesteps, energies)
    )
    degrees_of_freedom = len(ordered) - 2
    return {
        "status": "fit",
        "point_count": len(ordered),
        "timesteps": timesteps,
        "zero_timestep_energy": intercept,
        "zero_timestep_standard_error": sqrt(weight_timestep_squared / denominator),
        "slope": slope,
        "slope_standard_error": sqrt(weight_sum / denominator),
        "chi_squared": chi_squared,
        "degrees_of_freedom": degrees_of_freedom,
        "reduced_chi_squared": (
            chi_squared / degrees_of_freedom if degrees_of_freedom else None
        ),
    }


def _mixed_potential_fit(points: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "not_fit",
        "reason": "runs declare multiple potential_label values",
        "point_count": len(points),
    }


def _series_warnings(
    points: list[dict[str, Any]],
    fits: dict[str, dict[str, Any]],
    potential_identity: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    walkers = {point["target_walkers"] for point in points}
    if len(walkers) > 1:
        warnings.append("Runs use different target_walkers values; inspect population-control comparability.")
    if len(fits) > 1:
        warnings.append("T-move and no-T-move points were fit separately and must not be combined.")
    if potential_identity["status"] == "not_assessed":
        warnings.append(
            "Potential identity was not supplied for every run; do not treat the "
            "time-step fit as same-potential evidence."
        )
    if potential_identity["status"] == "mixed":
        warnings.append(
            "Runs declare multiple potential_label values; do not combine different "
            "potentials in one time-step fit."
        )
    if any(point["block_weights_constant"] is False for point in points):
        warnings.append("At least one run has varying BlockWeight; LocalEnergy is reblocked as recorded block values.")
    warnings.extend(_point_quality_warnings(points))
    return warnings


def _point_quality_warnings(points: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if any(point["invalid_row_reasons"]["malformed"] for point in points):
        warnings.append(
            "At least one scalar file contains malformed rows that were excluded."
        )
    if any(point["invalid_row_reasons"]["non_finite"] for point in points):
        warnings.append(
            "At least one scalar file contains non-finite rows that were excluded."
        )
    if any(point["invalid_row_reasons"]["non_integral_index"] for point in points):
        warnings.append(
            "At least one scalar file contains non-integral-index rows that were excluded."
        )
    if any(
        point["local_energy_second_moment"]["status"] == "inconsistent"
        for point in points
    ):
        warnings.append(
            "At least one scalar file has LocalEnergy_sq below the LocalEnergy "
            "second-moment bound."
        )
    if any(
        point["acceptance_ratio_bounds"]["status"] == "out_of_bounds"
        for point in points
    ):
        warnings.append(
            "At least one scalar file has AcceptRatio values outside [0, 1]."
        )
    if any(
        point["block_weight_quality"]["status"] == "nonpositive"
        for point in points
    ):
        warnings.append(
            "At least one scalar file has non-positive BlockWeight values; its "
            "weighted LocalEnergy mean is unavailable."
        )
    if any(
        point["energy_component_balance"]["status"] == "unbalanced"
        for point in points
    ):
        warnings.append(
            "At least one scalar file has an unbalanced LocalEnergy, Kinetic, "
            "and LocalPotential record."
        )
    if any(point["block_index_sequence"]["status"] == "noncontiguous" for point in points):
        warnings.append(
            "At least one scalar file has gaps or nonincreasing block indices."
        )
    if any(point["truncated"] for point in points):
        warnings.append("At least one scalar file hit the row limit; its analysis point is incomplete.")
    return warnings


def _potential_identity(runs: list[dict[str, Any]]) -> dict[str, Any]:
    labels = []
    missing_scalar_files = []
    for run in runs:
        label = run.get("potential_label")
        if not isinstance(label, str) or not label.strip():
            missing_scalar_files.append(run.get("scalar_file"))
            continue
        labels.append(label.strip())
    if missing_scalar_files:
        return {
            "status": "not_assessed",
            "reason": "potential_label was not supplied for every run",
            "missing_scalar_files": missing_scalar_files,
        }
    unique_labels = sorted(set(labels))
    if len(unique_labels) == 1:
        return {
            "status": "uniform",
            "potential_label": unique_labels[0],
        }
    return {
        "status": "mixed",
        "potential_labels": unique_labels,
    }


def _paired_potential_identity(
    tmove: dict[str, Any],
    no_tmove: dict[str, Any],
) -> dict[str, Any]:
    tmove_label = _potential_label(tmove)
    no_tmove_label = _potential_label(no_tmove)
    if tmove_label is None or no_tmove_label is None:
        return {
            "status": "not_assessed",
            "reason": "potential_label was not supplied for both matched runs",
        }
    if tmove_label != no_tmove_label:
        raise ValueError(
            "T-move locality-shift comparison requires matching potential_label values "
            "when supplied."
        )
    return {"status": "matched", "potential_label": tmove_label}


def _potential_label(run: dict[str, Any]) -> str | None:
    label = run.get("potential_label")
    if not isinstance(label, str):
        return None
    return label.strip() or None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _standard_error(values: list[float]) -> float:
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return sqrt(variance / len(values))


def _constant(values: list[float] | None) -> bool | None:
    if values is None:
        return None
    return all(value == values[0] for value in values[1:])


def _positive_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value) and value > 0


def _tmove_label(nonlocalmoves: bool) -> str:
    return "tmove" if nonlocalmoves else "no_tmove"


def _walker_count_comparability(
    tmove_target: float | None,
    no_tmove_target: float | None,
) -> dict[str, Any]:
    if tmove_target is None or no_tmove_target is None:
        return {
            "status": "not_assessed",
            "reason": "target_walkers was not supplied for both matched runs",
        }
    if tmove_target == no_tmove_target:
        return {
            "status": "matched",
            "target_walkers": tmove_target,
        }
    return {
        "status": "different",
        "tmove_target_walkers": tmove_target,
        "no_tmove_target_walkers": no_tmove_target,
    }


def _read_dmc_rows(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", errors="replace") as stream:
        return _parse_dmc_lines(stream)


def _parse_dmc_lines(lines: Iterable[str]) -> dict[str, Any]:
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
            if columns is not None:
                continue
            candidate = stripped.removeprefix("#").split()
            if candidate and candidate[0] == "Index":
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
        if len(values["Index"]) >= _MAX_DMC_ROWS:
            truncated = True
            break
        for column, value in zip(columns, row):
            values[column].append(value)
    if columns is None:
        raise ValueError("QMCPACK DMC data has no '#' column header.")
    if not values["Index"]:
        rejected = []
        for count, reason in (
            (malformed_rows, "malformed"),
            (nonfinite_rows, "non-finite"),
            (noninteger_index_rows, "non-integral-index"),
        ):
            if count:
                rejected.append(f"{count} {reason} row(s)")
        message = "QMCPACK DMC data contains no valid block rows."
        if rejected:
            message += " Rejected: " + ", ".join(rejected) + "."
        raise ValueError(message)
    return {
        "columns": columns,
        "values": values,
        "row_count": len(values["Index"]),
        "invalid_row_count": invalid_rows,
        "invalid_row_reasons": {
            "malformed": malformed_rows,
            "non_finite": nonfinite_rows,
            "non_integral_index": noninteger_index_rows,
        },
        "truncated": truncated,
        "block_index_sequence": block_index_sequence(values["Index"]),
    }


def _population_quality_warnings(parsed: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    reasons = parsed["invalid_row_reasons"]
    if reasons["malformed"]:
        warnings.append(
            "The DMC population file excluded "
            f"{reasons['malformed']} malformed row(s)."
        )
    if reasons["non_finite"]:
        warnings.append(
            "The DMC population file excluded "
            f"{reasons['non_finite']} non-finite row(s)."
        )
    if reasons["non_integral_index"]:
        warnings.append(
            "The DMC population file excluded "
            f"{reasons['non_integral_index']} non-integral-index row(s)."
        )
    if parsed["block_index_sequence"]["status"] == "noncontiguous":
        warnings.append(
            "The DMC population file has gaps or nonincreasing block indices."
        )
    if parsed["truncated"]:
        warnings.append(
            "The DMC population file hit the row limit; retained measurements are incomplete."
        )
    return warnings


def _summary_or_none(values: list[float] | None) -> dict[str, float] | None:
    return _summary(values) if values is not None else None


def _summary(values: list[float]) -> dict[str, float]:
    mean = _mean(values)
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


__all__ = [
    "analyze_dmc_input_series",
    "analyze_dmc_series",
    "compare_tmove_locality_shift",
    "compare_tmove_locality_shift_from_input",
    "inspect_dmc_population",
    "inspect_dmc_population_from_input",
]

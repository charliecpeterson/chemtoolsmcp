"""Compare supported QE inputs with their runtime output summaries.

PWSCF checks abstain when output reports transformed evidence, such as a
symmetry-reduced k-point count. The bounded converter check compares its
reported HDF5 handoff path only.
"""

from __future__ import annotations

import posixpath
from pathlib import Path
from typing import Any, Mapping


_CUTOFF_TOLERANCE_RY = 1.0e-4
_ELECTRON_TOLERANCE = 5.0e-3


class QeRunConsistency:
    def compare_input_output(
        self,
        input_path: str,
        output_path: str,
        parsed_input: Mapping[str, Any],
        parsed_output: Mapping[str, Any],
        _artifact_paths: tuple[str, ...],
    ) -> Mapping[str, Any]:
        derived = _mapping(parsed_output.get("derived"))
        if (
            parsed_input.get("format") == "qe-pw2qmcpack-input/1"
            or derived.get("qe:program") == "pw2qmcpack"
        ):
            return _pw2qmcpack_consistency(
                input_path,
                output_path,
                parsed_input,
                derived,
                _artifact_paths,
            )
        input_system = _mapping(parsed_input.get("system"))
        output_system = _mapping(derived.get("qe:system"))
        checks = [
            _calculation_mode_check(parsed_input, derived),
            _integer_check(
                "atom_count",
                input_system.get("nat"),
                _output_record(output_system, "n_atoms"),
            ),
            _integer_check(
                "atomic_type_count",
                input_system.get("ntyp"),
                _output_record(output_system, "n_atom_types"),
            ),
            _electron_count_check(parsed_input, output_system),
            _numeric_check(
                "ecutwfc",
                input_system.get("ecutwfc_ry"),
                _output_record(output_system, "ecutwfc_ry"),
                units="Ry",
                tolerance=_CUTOFF_TOLERANCE_RY,
                basis=(
                    "The configured wavefunction cutoff compared with the "
                    "PWSCF runtime summary."
                ),
            ),
            _numeric_check(
                "ecutrho",
                _effective_ecutrho(parsed_input),
                _output_record(output_system, "ecutrho_ry"),
                units="Ry",
                tolerance=_CUTOFF_TOLERANCE_RY,
                basis=(
                    "The explicit density cutoff, or QE's documented 4x "
                    "default, compared with the PWSCF runtime summary."
                ),
            ),
            _k_point_count_check(parsed_input, output_system),
        ]
        summary = {
            status: sum(check["status"] == status for check in checks)
            for status in ("match", "mismatch", "not_checked")
        }
        if summary["mismatch"]:
            status = "mismatch"
        elif summary["match"]:
            status = "checked"
        else:
            status = "not_checked"
        return {
            "status": status,
            "input_path": str(Path(input_path).resolve()),
            "summary": summary,
            "checks": checks,
        }


def _pw2qmcpack_consistency(
    input_path: str,
    output_path: str,
    parsed_input: Mapping[str, Any],
    derived: Mapping[str, Any],
    artifact_paths: tuple[str, ...],
) -> Mapping[str, Any]:
    source = str(Path(input_path).resolve())
    if (
        parsed_input.get("format") != "qe-pw2qmcpack-input/1"
        or derived.get("qe:program") != "pw2qmcpack"
    ):
        return {
            "status": "not_checked",
            "input_path": source,
            "summary": {"match": 0, "mismatch": 0, "not_checked": 1},
            "checks": [_not_checked(
                "pw2qmcpack_hdf5_path",
                "Both the converter input and converter output are required.",
                parsed_input.get("format"),
                derived.get("qe:program"),
            )],
        }
    values = _mapping(parsed_input.get("namelist"))
    prefix = values.get("prefix")
    outdir = values.get("outdir")
    if not _non_empty_string(prefix) or not _non_empty_string(outdir):
        return {
            "status": "not_checked",
            "input_path": source,
            "summary": {"match": 0, "mismatch": 0, "not_checked": 1},
            "checks": [_not_checked(
                "pw2qmcpack_hdf5_path",
                "The converter input requires explicit prefix and outdir values.",
                {"prefix": prefix, "outdir": outdir},
                None,
            )],
        }
    expected_path = posixpath.join(outdir, f"{prefix}.pwscf.h5")
    expected_normalized = posixpath.normpath(expected_path)
    artifacts = [
        artifact
        for artifact in derived.get("qe:pw2qmcpack_hdf5_artifacts") or []
        if isinstance(artifact, Mapping) and _non_empty_string(artifact.get("path"))
    ]
    if not artifacts:
        return {
            "status": "not_checked",
            "input_path": source,
            "summary": {"match": 0, "mismatch": 0, "not_checked": 1},
            "checks": [_not_checked(
                "pw2qmcpack_hdf5_path",
                "The converter output did not report an HDF5 artifact path.",
                {"prefix": prefix, "outdir": outdir, "expected_path": expected_path},
                None,
            )],
        }
    reported_paths = [str(artifact["path"]) for artifact in artifacts]
    matches = [
        artifact
        for artifact in artifacts
        if posixpath.normpath(str(artifact["path"])) == expected_normalized
    ]
    check = {
        "field": "pw2qmcpack_hdf5_path",
        "status": "match" if matches else "mismatch",
        "input": {
            "prefix": prefix,
            "outdir": outdir,
            "expected_path": expected_path,
        },
        "output": {
            "reported_paths": reported_paths,
            "matching_paths": [str(artifact["path"]) for artifact in matches],
            "basis": "pw2qmcpack esh5 create output",
        },
    }
    checks = [check]
    sidecar_checks = _pw2qmcpack_sidecar_checks(
        input_path,
        output_path,
        reported_paths,
        artifact_paths,
    )
    checks.extend(sidecar_checks)
    summary = {
        status: sum(item["status"] == status for item in checks)
        for status in ("match", "mismatch", "not_checked")
    }
    return {
        "status": "mismatch" if summary["mismatch"] else "checked",
        "input_path": source,
        "summary": summary,
        "checks": checks,
    }


def _pw2qmcpack_sidecar_checks(
    input_path: str,
    output_path: str,
    reported_paths: list[str],
    artifact_paths: tuple[str, ...],
) -> list[dict[str, Any]]:
    sidecars = [
        Path(path).expanduser().resolve()
        for path in artifact_paths
        if path.casefold().endswith(".pwscf.h5")
    ]
    if not sidecars:
        return []
    output_directory = Path(output_path).expanduser().resolve().parent
    resolved_reported = [
        Path(reported_path).expanduser().resolve()
        if Path(reported_path).is_absolute()
        else (output_directory / reported_path).resolve()
        for reported_path in reported_paths
    ]
    matching = [
        str(sidecar)
        for sidecar in sidecars
        if sidecar in resolved_reported
    ]
    identity_check = {
        "field": "pw2qmcpack_hdf5_artifact",
        "status": "match" if matching else "mismatch",
        "input": {
            "reported_paths": reported_paths,
            "resolved_paths": [str(path) for path in resolved_reported],
            "basis": (
                "Relative pw2qmcpack output paths are resolved against the "
                "converter output directory."
            ),
        },
        "output": {
            "supplied_paths": [str(sidecar) for sidecar in sidecars],
            "matching_paths": matching,
        },
    }
    if not matching:
        return [identity_check]
    input_source = Path(input_path).expanduser().resolve()
    input_modified_ns = input_source.stat().st_mtime_ns
    matching_sidecars = [Path(path) for path in matching]
    stale_paths = [
        str(sidecar)
        for sidecar in matching_sidecars
        if sidecar.stat().st_mtime_ns < input_modified_ns
    ]
    freshness_check = {
        "field": "pw2qmcpack_hdf5_freshness",
        "status": "match" if not stale_paths else "mismatch",
        "input": {
            "input_path": str(input_source),
            "input_modified_ns": input_modified_ns,
        },
        "output": {
            "matching_paths": matching,
            "stale_paths": stale_paths,
            "basis": (
                "The explicit sidecar must be at least as new as the "
                "converter input."
            ),
        },
    }
    return [identity_check, freshness_check]


def _calculation_mode_check(
    parsed_input: Mapping[str, Any], derived: Mapping[str, Any]
) -> dict[str, Any]:
    input_mode = str(parsed_input.get("calculation") or "scf").lower()
    input_group = (
        "bands_or_nscf" if input_mode in {"bands", "nscf"} else input_mode
    )
    output_mode = derived.get("qe:calculation_mode")
    has_mode_evidence = output_mode in {
        "bands_or_nscf",
        "relax",
        "vc-relax",
    } or (
        output_mode == "scf"
        and bool(
            derived.get("qe:scf_cycles")
            or derived.get("final_energy_hartree") is not None
        )
    )
    input_evidence = {
        "calculation": input_mode,
        "comparison_group": input_group,
    }
    if not has_mode_evidence:
        return _not_checked(
            "calculation_mode",
            (
                "The output ended before printing a calculation-specific "
                "PWSCF marker."
            ),
            input_evidence,
            {"calculation_mode": output_mode, "basis": "default inference"},
        )
    output_evidence = {
        "calculation_mode": output_mode,
        "basis": "PWSCF calculation markers",
    }
    return {
        "field": "calculation_mode",
        "status": "match" if input_group == output_mode else "mismatch",
        "input": input_evidence,
        "output": output_evidence,
    }


def _integer_check(
    field: str, input_value: Any, output_record: Mapping[str, Any]
) -> dict[str, Any]:
    input_integer = _integer(input_value)
    output_integer = _integer(output_record.get("value"))
    if input_integer is None or output_integer is None:
        return _not_checked(
            field,
            "Both input and output integer values are required.",
            input_integer,
            _record_evidence(output_record),
        )
    return {
        "field": field,
        "status": "match" if input_integer == output_integer else "mismatch",
        "input": input_integer,
        "output": _record_evidence(output_record),
    }


def _electron_count_check(
    parsed_input: Mapping[str, Any], output_system: Mapping[str, Any]
) -> dict[str, Any]:
    charge_spin = _mapping(parsed_input.get("charge_spin_review"))
    accounting = _mapping(charge_spin.get("electron_accounting"))
    input_value = (
        accounting.get("electron_count")
        if accounting.get("status") == "complete"
        else None
    )
    check = _numeric_check(
        "electron_count",
        input_value,
        _output_record(output_system, "n_electrons"),
        units="electrons",
        tolerance=_ELECTRON_TOLERANCE,
        basis=(
            "UPF valence charges and tot_charge compared with PWSCF's "
            "printed electron count."
        ),
    )
    if input_value is None:
        check["input_accounting_status"] = accounting.get("status") or "unavailable"
    return check


def _numeric_check(
    field: str,
    input_value: Any,
    output_record: Mapping[str, Any],
    *,
    units: str,
    tolerance: float,
    basis: str,
) -> dict[str, Any]:
    input_number = _number(input_value)
    output_number = _number(output_record.get("value"))
    input_evidence = (
        {"value": input_number, "units": units}
        if input_number is not None
        else None
    )
    output_evidence = _record_evidence(output_record, units=units)
    if input_number is None or output_number is None:
        return _not_checked(
            field,
            "Both input and output numeric values are required.",
            input_evidence,
            output_evidence,
        )
    return {
        "field": field,
        "status": (
            "match"
            if abs(input_number - output_number) <= tolerance
            else "mismatch"
        ),
        "input": input_evidence,
        "output": output_evidence,
        "absolute_tolerance": tolerance,
        "basis": basis,
    }


def _effective_ecutrho(parsed_input: Mapping[str, Any]) -> Any:
    pseudo_review = _mapping(parsed_input.get("pseudopotential_review"))
    cutoff_review = _mapping(pseudo_review.get("cutoff_review"))
    return cutoff_review.get("effective_ecutrho_ry")


def _k_point_count_check(
    parsed_input: Mapping[str, Any], output_system: Mapping[str, Any]
) -> dict[str, Any]:
    review = _mapping(parsed_input.get("k_point_review"))
    sampling = _mapping(review.get("sampling"))
    output_record = _output_record(output_system, "n_k_points")
    output_count = _integer(output_record.get("value"))
    input_evidence = dict(sampling)
    output_evidence = _record_evidence(output_record)
    if output_count is None:
        return _not_checked(
            "k_point_count",
            "The output contains no PWSCF k-point count.",
            input_evidence,
        )
    if sampling.get("mode") == "gamma":
        return {
            "field": "k_point_count",
            "status": "match" if output_count == 1 else "mismatch",
            "input": input_evidence,
            "output": output_evidence,
            "basis": "Gamma-only sampling contains one k-point.",
        }
    return _not_checked(
        "k_point_count",
        (
            "The requested input count and PWSCF runtime count are not "
            "directly comparable after symmetry and time-reversal reduction."
        ),
        input_evidence,
        output_evidence,
    )


def _output_record(
    output_system: Mapping[str, Any], key: str
) -> Mapping[str, Any]:
    return _mapping(output_system.get(key))


def _record_evidence(
    record: Mapping[str, Any], *, units: str | None = None
) -> dict[str, Any] | None:
    if not record:
        return None
    evidence = {"value": record.get("value"), "line": record.get("line")}
    if units is not None:
        evidence["units"] = units
    return evidence


def _not_checked(
    field: str,
    reason: str,
    input_value: Any = None,
    output_value: Any = None,
) -> dict[str, Any]:
    check = {"field": field, "status": "not_checked", "reason": reason}
    if input_value is not None:
        check["input"] = input_value
    if output_value is not None:
        check["output"] = output_value
    return check


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


QE_RUN_CONSISTENCY = QeRunConsistency()


__all__ = ["QE_RUN_CONSISTENCY", "QeRunConsistency"]

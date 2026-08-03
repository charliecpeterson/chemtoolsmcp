"""Differential checks between Orbitron JSON and pinned raw chemistry outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chemtools.programs.molcas.parse.freq import parse_last_freq_block
from chemtools.programs.qe.geometry import parse_pw_geometry_text
from chemtools.programs.qe.output import parse_pw_output_text

from .orbitron import OrbitronClient, OrbitronError, OrbitronResponse

MANIFEST_SCHEMA = "chemtools.orbitron-contract-corpus/2"
REFERENCE_CORPUS_ENV = "CHEMTOOLS_REFERENCE_CORPUS"
BOHR_ANGSTROM = 0.529177210903

_TOTAL_ENERGY_RE = re.compile(
    r"^\s*!\s+total energy\s*=\s*([-+0-9.EeDd]+)\s+Ry",
    re.MULTILINE,
)
_SCF_START_RE = re.compile(
    r"^[ \t]*Self-consistent Calculation[ \t]*$",
    re.MULTILINE,
)
_ATOM_COUNT_RE = re.compile(r"number of atoms/cell\s*=\s*(\d+)")
_ALAT_RE = re.compile(
    r"lattice parameter \(alat\)\s*=\s*([-+0-9.EeDd]+)\s+a\.u\."
)
_AXIS_RE = re.compile(
    r"a\(\d\)\s*=\s*\(\s*([-+0-9.EeDd]+)\s+"
    r"([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s*\)"
)
_ATOM_SITE_RE = re.compile(r"^\s*\d+\s+([A-Z][a-z]?)\s+tau\(", re.MULTILINE)
_MODULE_RE = re.compile(r"^--- Start Module:\s+(\w+)", re.MULTILINE | re.IGNORECASE)


class ReferenceParseError(ValueError):
    """A pinned source no longer contains facts required by its contract."""


def load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported contract manifest schema {payload.get('schema')!r}"
        )
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("contract manifest must contain at least one case")
    required = {
        "id",
        "operation",
        "source",
        "size_bytes",
        "sha256",
        "contract",
    }
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"manifest case {index} must be an object")
        missing = required - case.keys()
        if missing:
            raise ValueError(
                f"manifest case {case.get('id', index)!r} is missing {sorted(missing)}"
            )
        if case["operation"] not in {
            "analyze_geometry",
            "info",
            "inspect",
            "analyze_vibrations",
        }:
            raise ValueError(
                f"manifest case {case['id']!r} has unsupported operation "
                f"{case['operation']!r}"
            )
        if (
            not isinstance(case["size_bytes"], int)
            or isinstance(case["size_bytes"], bool)
            or case["size_bytes"] < 0
        ):
            raise ValueError(
                f"manifest case {case['id']!r} has invalid size_bytes"
            )
    return payload


def run_contract(
    manifest_path: str | os.PathLike[str],
    corpus_root: str | os.PathLike[str] | None,
    *,
    executable: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    root = Path(corpus_root).expanduser().resolve() if corpus_root else None
    records = []
    client: OrbitronClient | None = None
    client_error: OrbitronError | None = None

    try:
        client = OrbitronClient(executable)
    except OrbitronError as error:
        client_error = error

    for case in manifest["cases"]:
        source = (root / case["source"]).resolve() if root else None
        if source is not None:
            try:
                source.relative_to(root)
            except ValueError:
                records.append(
                    _record(
                        case,
                        source,
                        "no_reference",
                        reason="reference path escapes the configured corpus root",
                    )
                )
                continue
        reference_record = _verify_reference(case, source)
        if reference_record is not None:
            records.append(reference_record)
            continue

        try:
            reference_text = source.read_text(encoding="utf-8", errors="replace")
            reference = _parse_reference(case["contract"], reference_text)
        except (OSError, ReferenceParseError) as error:
            records.append(
                _record(
                    case,
                    source,
                    "no_reference",
                    reason=f"reference parse failed: {error}",
                )
            )
            continue

        if client_error is not None:
            records.append(
                _record(
                    case,
                    source,
                    "tool_refused",
                    reason=str(client_error),
                )
            )
            continue

        try:
            response = getattr(client, case["operation"])(source)
        except OrbitronError as error:
            records.append(
                _record(
                    case,
                    source,
                    "tool_refused",
                    reason=str(error),
                    tool_error={
                        "type": type(error).__name__,
                        "returncode": getattr(error, "returncode", None),
                        "stderr": getattr(error, "stderr", ""),
                    },
                )
            )
            continue

        checks = _compare(case["contract"], reference, response.payload)
        outcome = "agree" if all(check["agrees"] for check in checks) else "disagree"
        records.append(
            _record(
                case,
                source,
                outcome,
                checks=checks,
                orbitron=_orbitron_provenance(response),
            )
        )

    counts = Counter(record["outcome"] for record in records)
    return {
        "schema": "chemtools.orbitron-contract-report/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(Path(manifest_path).resolve()),
        "corpus_root": str(root) if root else None,
        "case_count": len(records),
        "checked_count": counts["agree"] + counts["disagree"],
        "agree_count": counts["agree"],
        "disagree_count": counts["disagree"],
        "tool_refused_count": counts["tool_refused"],
        "no_reference_count": counts["no_reference"],
        "records": records,
    }


def report_exit_code(report: dict[str, Any]) -> int:
    if report["tool_refused_count"]:
        return 2
    if report["disagree_count"]:
        return 1
    if report["no_reference_count"]:
        return 3
    return 0


def _verify_reference(
    case: dict[str, Any],
    source: Path | None,
) -> dict[str, Any] | None:
    if source is None:
        return _record(
            case,
            None,
            "no_reference",
            reason=f"set {REFERENCE_CORPUS_ENV} or pass --corpus",
        )
    if not source.is_file():
        return _record(
            case,
            source,
            "no_reference",
            reason="reference file is missing",
        )
    actual_size = source.stat().st_size
    if actual_size != case["size_bytes"]:
        return _record(
            case,
            source,
            "no_reference",
            reason="reference size changed; review the case before comparison",
            expected_size_bytes=case["size_bytes"],
            actual_size_bytes=actual_size,
        )
    actual_hash = _sha256(source)
    if actual_hash != case["sha256"]:
        return _record(
            case,
            source,
            "no_reference",
            reason="reference hash changed; review the case before comparison",
            expected_sha256=case["sha256"],
            actual_sha256=actual_hash,
        )
    return None


def _record(
    case: dict[str, Any],
    source: Path | None,
    outcome: str,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "contract": case["contract"],
        "operation": case["operation"],
        "reference_status": case.get("status"),
        "source": str(source) if source else None,
        "size_bytes": case["size_bytes"],
        "sha256": case["sha256"],
        "outcome": outcome,
        **fields,
    }


def _orbitron_provenance(response: OrbitronResponse) -> dict[str, Any]:
    return {
        "executable_version": response.version.version,
        "executable_commit": response.version.commit,
        "schema": response.schema,
        "producer": response.producer,
        "warnings": list(response.warnings),
        "stderr": response.stderr,
    }


def _parse_reference(contract: str, text: str) -> dict[str, Any]:
    if contract == "qe_failed_geometry":
        return _parse_qe_failed_geometry_reference(text)
    if contract == "qe_geometry":
        return _parse_qe_geometry_reference(text)
    if contract == "qe_scf":
        return _parse_qe_scf_reference(text)
    if contract == "qe_relax":
        return _parse_qe_relax_reference(text)
    if contract == "molcas_failure":
        return _parse_molcas_failure_reference(text)
    if contract == "molcas_vibrations":
        return _parse_molcas_vibration_reference(text)
    raise ReferenceParseError(f"unknown reference contract: {contract}")


def _compare(
    contract: str,
    reference: dict[str, Any],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    if contract == "qe_failed_geometry":
        return _compare_qe_failed_geometry(reference, payload)
    if contract == "qe_geometry":
        return _compare_qe_geometry(reference, payload)
    if contract == "qe_scf":
        return _compare_qe_scf(reference, payload)
    if contract == "qe_relax":
        return _compare_qe_relax(reference, payload)
    if contract == "molcas_failure":
        return _compare_molcas_failure(reference, payload)
    if contract == "molcas_vibrations":
        return _compare_molcas_vibrations(reference, payload)
    raise ReferenceParseError(f"unknown comparison contract: {contract}")


def _parse_qe_geometry_reference(text: str) -> dict[str, Any]:
    parsed = parse_pw_geometry_text(text)
    if parsed.get("status") != "available":
        raise ReferenceParseError(
            f"Chemtools QE geometry is unavailable: {parsed.get('reason')}"
        )
    atoms = parsed["atoms"]
    coordinates = [
        [atom[component] for component in ("x", "y", "z")]
        for atom in atoms
    ]
    cell = parsed["cell"]
    geometry_role, geometry_source = _qe_geometry_provenance(text)
    return {
        "atoms": parsed["atom_count"],
        "elements": parsed["elements"],
        "distance_unit": parsed["units"],
        "bounding_box": {
            "min": [min(point[axis] for point in coordinates) for axis in range(3)],
            "max": [max(point[axis] for point in coordinates) for axis in range(3)],
        },
        "cell": cell["vectors_angstrom"],
        "periodic": cell["periodic"],
        "geometry_role": geometry_role,
        "geometry_source": geometry_source,
    }


def _parse_qe_failed_geometry_reference(text: str) -> dict[str, Any]:
    atom_count = _required_int(_ATOM_COUNT_RE, text, "QE atom count")
    atom_sites = _ATOM_SITE_RE.findall(text)
    if len(atom_sites) < atom_count:
        raise ReferenceParseError("QE atomic site table is incomplete")
    geometry_role, geometry_source = _qe_geometry_provenance(text)
    if geometry_role != "last_attempted":
        raise ReferenceParseError(
            "QE failed-geometry contract did not resolve a last attempted geometry"
        )
    return {
        "atoms": atom_count,
        "elements": dict(Counter(atom_sites[:atom_count])),
        "geometry_role": geometry_role,
        "geometry_source": geometry_source,
    }


def _qe_geometry_provenance(text: str) -> tuple[str, str]:
    output = parse_pw_output_text(text)
    geometry = parse_pw_geometry_text(text)
    if geometry.get("role") == "converged_relaxed_structure":
        bfgs = output.get("bfgs") or {}
        step_count = bfgs.get("scf_cycles")
        if not isinstance(step_count, int):
            raise ReferenceParseError("QE converged relaxation step count is missing")
        return (
            "converged_final",
            f"step {step_count} of {step_count}, the converged geometry",
        )
    if geometry.get("role") == "calculation_structure":
        return "single_point", "the only geometry the run reports"
    if output.get("scf_nonconvergence") is not None:
        step_count = len(output.get("scf_cycles") or [])
        if step_count:
            return (
                "last_attempted",
                f"step {step_count} of {step_count}; "
                "the run stopped without converging",
            )
    raise ReferenceParseError("QE geometry provenance could not be established")


def _parse_qe_scf_reference(text: str) -> dict[str, Any]:
    atom_count = _required_int(_ATOM_COUNT_RE, text, "QE atom count")
    alat = _required_float(_ALAT_RE, text, "QE lattice parameter")
    axes = _crystal_axes(text)
    cell = [
        [component * alat * BOHR_ANGSTROM for component in vector]
        for vector in axes
    ]
    atom_sites = _ATOM_SITE_RE.findall(text)
    if len(atom_sites) < atom_count:
        raise ReferenceParseError("QE atomic site table is incomplete")
    energies = _energy_values(text)
    return {
        "atoms": atom_count,
        "elements": dict(Counter(atom_sites[:atom_count])),
        "cell_angstrom": cell,
        "energy_ry": energies[-1],
        "task_count": len(_SCF_START_RE.findall(text)),
    }


def _parse_qe_relax_reference(text: str) -> dict[str, Any]:
    energies = _energy_values(text)
    starts = [
        text.count("\n", 0, match.start()) + 1
        for match in _SCF_START_RE.finditer(text)
    ]
    cell_count = len(re.findall(r"^CELL_PARAMETERS", text, re.MULTILINE))
    atom_count = _required_int(_ATOM_COUNT_RE, text, "QE atom count")
    if len(energies) != len(starts):
        raise ReferenceParseError(
            f"QE relaxation has {len(starts)} SCF blocks but {len(energies)} energies"
        )
    return {
        "atoms": atom_count,
        "energies_ry": energies,
        "task_line_starts": starts,
        "cell_update_count": cell_count,
    }


def _parse_molcas_failure_reference(text: str) -> dict[str, Any]:
    modules = [match.lower() for match in _MODULE_RE.findall(text)]
    if not modules:
        raise ReferenceParseError("OpenMolcas module starts are missing")
    error_lines = [
        (index, line.strip())
        for index, line in enumerate(text.splitlines(), start=1)
        if "ERROR: No such wave function." in line
    ]
    if not error_lines:
        raise ReferenceParseError("OpenMolcas parity diagnostic is missing")
    return {
        "program": "molcas" if "OpenMolcas" in text else None,
        "outcome": "failed" if "_RC_INPUT_ERROR_" in text else None,
        "modules": modules,
        "diagnostic_line": error_lines[0][0],
        "diagnostic_message": error_lines[0][1],
    }


def _parse_molcas_vibration_reference(text: str) -> dict[str, Any]:
    parsed = parse_last_freq_block(text)
    if parsed is None:
        raise ReferenceParseError("Molcas harmonic-frequency block is missing")
    frequencies = parsed["frequencies_cm1"]
    if not frequencies:
        raise ReferenceParseError("Molcas harmonic-frequency block has no modes")
    return {
        "mode_count": len(frequencies),
        "imaginary_count": sum(frequency < 0 for frequency in frequencies),
        "lowest_frequency": min(frequencies),
        "highest_frequency": max(frequencies),
        "mean_frequency": sum(frequencies) / len(frequencies),
        "frequency_sample": sorted(frequencies)[:10],
    }


def _compare_qe_scf(
    reference: dict[str, Any],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    tasks = payload.get("program_tasks") or []
    task = tasks[0] if tasks else {}
    unit_cell = payload.get("unit_cell") or {}
    checks = [
        _check("atoms", reference["atoms"], payload.get("atoms")),
        _check("elements", reference["elements"], payload.get("elements")),
        _check("task_count", reference["task_count"], len(tasks)),
        _check("task.program", "qe", task.get("program")),
        _check("task.kind", "scf", task.get("kind")),
        _check(
            "task.energy_ry",
            reference["energy_ry"],
            _nested(task, "extra", "scf_total_energy_ry"),
            tolerance=1e-8,
        ),
        _check(
            "task.energy_hartree",
            reference["energy_ry"] / 2.0,
            task.get("energy_hartree"),
            tolerance=1e-8,
        ),
    ]
    for index, name in enumerate(("a", "b", "c")):
        checks.append(
            _check(
                f"unit_cell.{name}_angstrom",
                reference["cell_angstrom"][index],
                unit_cell.get(name),
                tolerance=1e-6,
            )
        )
    checks.append(
        _check("unit_cell.periodic", [True, True, True], unit_cell.get("periodic"))
    )
    return checks


def _compare_qe_geometry(
    reference: dict[str, Any],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    bounding_box = payload.get("bounding_box") or {}
    unit_cell = payload.get("unit_cell") or {}
    checks = [
        _check("atoms", reference["atoms"], payload.get("atoms")),
        _check("elements", reference["elements"], payload.get("elements")),
        _check(
            "distance_unit",
            reference["distance_unit"],
            payload.get("distance_unit"),
        ),
        _check(
            "geometry_role",
            reference["geometry_role"],
            payload.get("geometry_role"),
        ),
        _check(
            "geometry_source",
            reference["geometry_source"],
            payload.get("geometry_source"),
        ),
        _check(
            "bounding_box.min",
            reference["bounding_box"]["min"],
            bounding_box.get("min"),
            tolerance=2e-6,
        ),
        _check(
            "bounding_box.max",
            reference["bounding_box"]["max"],
            bounding_box.get("max"),
            tolerance=2e-6,
        ),
    ]
    for index, name in enumerate(("a", "b", "c")):
        checks.append(
            _check(
                f"unit_cell.{name}",
                reference["cell"][index],
                unit_cell.get(name),
                tolerance=2e-6,
            )
        )
    checks.append(
        _check(
            "unit_cell.periodic",
            reference["periodic"],
            unit_cell.get("periodic"),
        )
    )
    return checks


def _compare_qe_failed_geometry(
    reference: dict[str, Any],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _check("atoms", reference["atoms"], payload.get("atoms")),
        _check("elements", reference["elements"], payload.get("elements")),
        _check(
            "geometry_role",
            reference["geometry_role"],
            payload.get("geometry_role"),
        ),
        _check(
            "geometry_source",
            reference["geometry_source"],
            payload.get("geometry_source"),
        ),
    ]


def _compare_qe_relax(
    reference: dict[str, Any],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    tasks = payload.get("program_tasks") or []
    frames = payload.get("frames") or []
    energy_hartree = [task.get("energy_hartree") for task in tasks]
    extra_energy_ry = [
        _nested(task, "extra", "scf_total_energy_ry") for task in tasks
    ]
    task_starts = [task.get("line_start") for task in tasks]
    frame_atoms = [frame.get("atoms") for frame in frames]
    profile = _nested(tasks[0], "extra", "relax_profile") if tasks else None
    profile = profile if isinstance(profile, dict) else {}
    return [
        _check("program", "qe", payload.get("program")),
        _check("detected", "trajectory", payload.get("detected")),
        _check("task_count", len(reference["energies_ry"]), len(tasks)),
        _check(
            "task.energy_hartree_sequence",
            [energy / 2.0 for energy in reference["energies_ry"]],
            energy_hartree,
            tolerance=1e-8,
        ),
        _check(
            "task.extra.scf_total_energy_ry_sequence",
            reference["energies_ry"],
            extra_energy_ry,
            tolerance=1e-8,
        ),
        _check(
            "task.line_start_sequence",
            reference["task_line_starts"],
            task_starts,
        ),
        _check(
            "frame_count",
            reference["cell_update_count"],
            len(frames),
        ),
        _check(
            "frame.atom_counts",
            [reference["atoms"]] * reference["cell_update_count"],
            frame_atoms,
        ),
        _check(
            "relax_profile.initial_energy_ry",
            reference["energies_ry"][0],
            profile.get("initial_energy_ry"),
            tolerance=1e-8,
        ),
        _check(
            "relax_profile.final_energy_ry",
            reference["energies_ry"][-1],
            profile.get("final_energy_ry"),
            tolerance=1e-8,
        ),
        _check(
            "relax_profile.step_count",
            len(reference["energies_ry"]),
            profile.get("step_count"),
        ),
    ]


def _compare_molcas_failure(
    reference: dict[str, Any],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    run = payload.get("run") or {}
    diagnostics = run.get("diagnostics") or []
    diagnostic = diagnostics[0] if diagnostics else {}
    tasks = payload.get("program_tasks") or []
    modules = [
        str(task.get("label", "")).split()[0].lower()
        for task in tasks
        if task.get("label")
    ]
    return [
        _check("program", reference["program"], payload.get("program")),
        _check("run.program", reference["program"], run.get("program")),
        _check("run.outcome", reference["outcome"], run.get("outcome")),
        _check("modules", reference["modules"], modules),
        _check(
            "diagnostic.line",
            reference["diagnostic_line"],
            diagnostic.get("line"),
        ),
        _check(
            "diagnostic.message",
            reference["diagnostic_message"],
            diagnostic.get("message"),
        ),
    ]


def _compare_molcas_vibrations(
    reference: dict[str, Any],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    sampled_frequencies = [
        mode.get("frequency")
        for mode in payload.get("modes", [])
        if isinstance(mode, dict)
    ]
    return [
        _check("mode_count", reference["mode_count"], payload.get("mode_count")),
        _check(
            "imaginary_count",
            reference["imaginary_count"],
            payload.get("imaginary_count"),
        ),
        _check(
            "lowest_frequency",
            reference["lowest_frequency"],
            payload.get("lowest_frequency"),
            tolerance=1e-8,
        ),
        _check(
            "highest_frequency",
            reference["highest_frequency"],
            payload.get("highest_frequency"),
            tolerance=1e-8,
        ),
        _check(
            "mean_frequency",
            reference["mean_frequency"],
            payload.get("mean_frequency"),
            tolerance=1e-8,
        ),
        _check(
            "modes.frequency_sample",
            reference["frequency_sample"],
            sampled_frequencies,
            tolerance=1e-8,
        ),
    ]


def _check(
    name: str,
    reference: Any,
    orbitron: Any,
    *,
    tolerance: float | None = None,
) -> dict[str, Any]:
    agrees = (
        _close(reference, orbitron, tolerance)
        if tolerance is not None
        else reference == orbitron
    )
    check = {
        "field": name,
        "reference": reference,
        "orbitron": orbitron,
        "agrees": agrees,
    }
    if tolerance is not None:
        check["absolute_tolerance"] = tolerance
    return check


def _close(reference: Any, orbitron: Any, tolerance: float) -> bool:
    if isinstance(reference, list):
        return (
            isinstance(orbitron, list)
            and len(reference) == len(orbitron)
            and all(
                _close(reference_item, orbitron_item, tolerance)
                for reference_item, orbitron_item in zip(reference, orbitron)
            )
        )
    if not isinstance(reference, (int, float)) or isinstance(reference, bool):
        return reference == orbitron
    if not isinstance(orbitron, (int, float)) or isinstance(orbitron, bool):
        return False
    return math.isclose(reference, orbitron, rel_tol=0.0, abs_tol=tolerance)


def _energy_values(text: str) -> list[float]:
    energies = [_float(match) for match in _TOTAL_ENERGY_RE.findall(text)]
    if not energies:
        raise ReferenceParseError("QE total energies are missing")
    return energies


def _crystal_axes(text: str) -> list[list[float]]:
    marker = text.find("crystal axes:")
    if marker < 0:
        raise ReferenceParseError("QE crystal axes are missing")
    axes = [
        [_float(component) for component in match.groups()]
        for match in _AXIS_RE.finditer(text[marker:])
    ][:3]
    if len(axes) != 3:
        raise ReferenceParseError("QE crystal axis table is incomplete")
    return axes


def _required_float(pattern: re.Pattern[str], text: str, field: str) -> float:
    match = pattern.search(text)
    if match is None:
        raise ReferenceParseError(f"{field} is missing")
    return _float(match.group(1))


def _required_int(pattern: re.Pattern[str], text: str, field: str) -> int:
    match = pattern.search(text)
    if match is None:
        raise ReferenceParseError(f"{field} is missing")
    return int(match.group(1))


def _float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _nested(mapping: Any, *keys: str) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare pinned raw outputs with Orbitron's versioned JSON."
    )
    default_manifest = (
        Path(__file__).resolve().parents[2]
        / "references"
        / "orbitron_contract_cases.json"
    )
    parser.add_argument("--manifest", default=str(default_manifest))
    parser.add_argument("--corpus", default=os.environ.get(REFERENCE_CORPUS_ENV))
    parser.add_argument("--orbitron")
    parser.add_argument("--output")
    arguments = parser.parse_args(argv)

    report = run_contract(
        arguments.manifest,
        arguments.corpus,
        executable=arguments.orbitron,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return report_exit_code(report)

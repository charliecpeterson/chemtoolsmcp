"""Inspect a bounded GRASP MCDHF ladder and its paired RCI calculations.

The inspector reads explicit stage directories rather than recursively scanning
an arbitrary tree. It keeps energies tied to ``.sum`` and ``.csum`` artifacts
and checks that fixed-orbital Dirac-Coulomb/Breit pairs use the same inputs.
"""

from __future__ import annotations

import hashlib
import math
import re
from itertools import pairwise
from pathlib import Path
from typing import Any

from chemtools.programs.grasp.parse.rci_log import parse_rci_log
from chemtools.programs.grasp.parse.rmcdhf_log import parse_rmcdhf_log
from chemtools.programs.grasp.parse.sum_file import parse_sum, rci_hamiltonian

_MAX_STAGES = 32
_MAX_RCI_VARIANTS = 8
_SUBSHELL_RE = re.compile(r"^[1-9][0-9]*([spdfghi])([-+]?)$")
_ANGULAR_MOMENTUM = {label: index for index, label in enumerate("spdfghi")}
_MODEL_FIELDS = (
    "atomic_number",
    "n_electrons",
    "n_csfs",
    "n_subshells",
    "speed_of_light_au",
    "nucleus",
    "radial_grid",
    "subshells",
    "state_sectors",
)
_SHARED_FIELDS = (
    "atomic_number",
    "n_electrons",
    "speed_of_light_au",
    "nucleus",
    "radial_grid",
    "state_sectors",
)


def inspect_grasp_case_directory(
    mcdhf_root: str,
    rci_root: str | None = None,
    stages: list[str] | None = None,
) -> dict[str, Any]:
    """Inspect an ordered MCDHF model ladder and optional RCI pair tree."""
    mcdhf_path = _require_directory(mcdhf_root, "mcdhf_root")
    rci_path = _require_directory(rci_root, "rci_root") if rci_root else None
    stage_names = _resolve_stages(mcdhf_path, stages)

    stage_rows = [_inspect_mcdhf_stage(mcdhf_path / name, name) for name in stage_names]
    if rci_path is not None:
        for stage in stage_rows:
            stage["rci"] = _inspect_rci_stage(
                rci_path / stage["stage"],
                stage,
            )

    shared_checks = _shared_model_checks(stage_rows)
    issues = [
        {"stage": stage["stage"], **issue}
        for stage in stage_rows
        for issue in stage["issues"]
    ]
    if rci_path is not None:
        issues.extend(
            {"stage": stage["stage"], **issue}
            for stage in stage_rows
            for issue in stage["rci"]["issues"]
        )
    issues.extend(
        {
            "stage": check["stage"],
            "severity": "error",
            "code": "shared_model_mismatch",
            "message": (
                f"{check['field']} differs from reference stage {stage_names[0]}"
            ),
        }
        for check in shared_checks
        if not check["matches"]
    )

    table = [_table_row(stage) for stage in stage_rows]
    increments = {
        "mcdhf": _energy_increments(table, "mcdhf_energy_hartree"),
    }
    if rci_path is not None:
        increments.update(
            {
                "rci_dirac_coulomb": _energy_increments(
                    table,
                    "rci_dirac_coulomb_energy_hartree",
                ),
                "rci_dirac_coulomb_plus_zero_frequency_breit": (
                    _energy_increments(table, "rci_breit0_energy_hartree")
                ),
            }
        )

    verdict = _verdict(issues)
    return {
        "schema": "chemtools.grasp-case-inspection/1",
        "roots": {
            "mcdhf": str(mcdhf_path),
            "rci": str(rci_path) if rci_path is not None else None,
        },
        "assessment": {
            "verdict": verdict,
            "ready_for_numerical_comparison": verdict == "complete",
            "headline": _headline(verdict, len(stage_rows), rci_path is not None),
        },
        "stage_order": stage_names,
        "comparison_table": table,
        "increments": increments,
        "shared_model_checks": shared_checks,
        "stages": stage_rows,
        "issues": issues,
        "uncertainty": [
            (
                "raw_determinant_dimension is the binomial capacity of the "
                "reported peel spinors, not the symmetry-adapted CSF count"
            ),
            (
                "an RCI stdout QED interpolation warning does not prove that "
                "QED changed the mixing; the .csum correction block is "
                "authoritative for included Hamiltonian terms"
            ),
        ],
    }


def _inspect_mcdhf_stage(stage_dir: Path, stage_name: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage": stage_name,
        "path": str(stage_dir),
        "issues": [],
    }
    if not stage_dir.is_dir():
        row["issues"].append(_issue("missing_stage", f"missing directory {stage_dir}"))
        return row

    try:
        summary_path = _select_file(stage_dir, "*.sum", f"{stage_name}.sum")
    except ValueError as exc:
        row["issues"].append(_issue("summary_selection", str(exc)))
        return row
    if summary_path is None:
        row["issues"].append(_issue("missing_summary", "no MCDHF .sum file found"))
        return row

    summary = parse_sum(str(summary_path))
    row["energy_hartree"] = summary.get("ground_energy_au")
    row["model"] = _model_signature(summary)
    row["optimization"] = {
        "mode": summary.get("optimization_mode"),
        "eol_n_levels_optimized": summary.get("eol_n_levels_optimized"),
        "ol_level_optimized": summary.get("ol_level_optimized"),
    }

    csf_path = _select_file(stage_dir, "*.c", f"{stage_name}.c")
    if csf_path is not None:
        row["csf_model"] = _parse_csf_model(
            csf_path,
            summary.get("n_electrons"),
        )
    else:
        row["issues"].append(_issue("missing_csf", "no saved CSF list found"))

    generation_path = stage_dir / "rcsfgenerate.in"
    if generation_path.is_file():
        row["generation"] = _parse_rcsfgenerate_input(generation_path)
    else:
        row["issues"].append(
            _issue(
                "missing_generation_input",
                "rcsfgenerate.in is unavailable",
                "warning",
            )
        )

    stdout_path = stage_dir / "rmcdhf_mem.stdout"
    if not stdout_path.is_file():
        stdout_path = stage_dir / "rmcdhf.stdout"
    stderr_path = stage_dir / "rmcdhf_mem.stderr"
    if not stderr_path.is_file():
        stderr_path = stage_dir / "rmcdhf.stderr"
    if stdout_path.is_file():
        execution_text = stdout_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
        if stderr_path.is_file():
            execution_text += "\n" + stderr_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        execution = parse_rmcdhf_log(execution_text)
        row["execution"] = {
            key: execution.get(key)
            for key in (
                "n_iterations",
                "converged",
                "explicitly_not_converged",
                "orbital_solver_failed",
                "failed_orbitals",
                "struggled_orbitals",
                "error_stop",
                "final_energy_hartree",
                "final_energy_source",
                "convergence_evidence",
                "max_final_orbital_self_consistency",
                "final_orbitals",
                "alternating_norm_orbitals",
                "growing_alternating_orbitals",
                "node_count_changes",
            )
        }
        if not execution.get("converged"):
            row["issues"].append(
                _issue(
                    "rmcdhf_not_converged", "MCDHF stdout does not prove convergence"
                )
            )
        elif execution.get("growing_alternating_orbitals"):
            row["issues"].append(
                _issue(
                    "rmcdhf_unstable_orbital_trace",
                    "MCDHF completed with growing sign-alternating updates "
                    "for "
                    + ", ".join(execution["growing_alternating_orbitals"]),
                    "warning",
                )
            )
        _check_energy_agreement(row, execution.get("final_energy_hartree"))
    else:
        row["execution"] = {"converged": None, "n_iterations": None}
        row["issues"].append(
            _issue(
                "missing_rmcdhf_stdout",
                "MCDHF convergence cannot be checked without stdout",
                "warning",
            )
        )

    row["artifacts"] = _artifact_map(
        {
            "summary": summary_path,
            "csf": csf_path,
            "radial_wavefunction": _select_file(stage_dir, "*.w", f"{stage_name}.w"),
            "nucleus": stage_dir / "isodata",
            "generation_input": generation_path,
            "rmcdhf_input": stage_dir / "rmcdhf.in",
            "rmcdhf_stdout": stdout_path,
            "rmcdhf_stderr": stderr_path,
        }
    )
    for role in ("radial_wavefunction", "nucleus"):
        if role not in row["artifacts"]:
            row["issues"].append(_issue(f"missing_{role}", f"missing {role} artifact"))
    return row


def _inspect_rci_stage(
    stage_dir: Path,
    mcdhf_stage: dict[str, Any],
) -> dict[str, Any]:
    inspection: dict[str, Any] = {
        "path": str(stage_dir),
        "variants": [],
        "issues": [],
    }
    if not stage_dir.is_dir():
        inspection["issues"].append(
            _issue("missing_rci_stage", f"missing RCI directory {stage_dir}")
        )
        return inspection

    variant_dirs = sorted(
        child
        for child in stage_dir.iterdir()
        if child.is_dir() and any(child.glob("*.csum"))
    )
    if len(variant_dirs) > _MAX_RCI_VARIANTS:
        inspection["issues"].append(
            _issue(
                "too_many_rci_variants",
                f"found {len(variant_dirs)} variants; limit is {_MAX_RCI_VARIANTS}",
            )
        )
        return inspection

    for variant_dir in variant_dirs:
        inspection["variants"].append(_inspect_rci_variant(variant_dir))
    if not inspection["variants"]:
        inspection["issues"].append(
            _issue("missing_rci_variants", "no direct child directory contains a .csum")
        )
        return inspection

    for variant in inspection["variants"]:
        if variant["issues"]:
            inspection["issues"].extend(
                {
                    **issue,
                    "message": f"{variant['variant']}: {issue['message']}",
                }
                for issue in variant["issues"]
            )

    pair = _build_rci_pair(inspection["variants"], mcdhf_stage)
    inspection["pair"] = pair
    inspection["issues"].extend(pair.pop("issues"))
    return inspection


def _inspect_rci_variant(variant_dir: Path) -> dict[str, Any]:
    variant: dict[str, Any] = {
        "variant": variant_dir.name,
        "path": str(variant_dir),
        "issues": [],
    }
    try:
        summary_path = _select_file(variant_dir, "*.csum", "state.csum")
    except ValueError as exc:
        variant["issues"].append(_issue("summary_selection", str(exc)))
        return variant
    if summary_path is None:
        variant["issues"].append(_issue("missing_csum", "no RCI .csum found"))
        return variant

    summary = parse_sum(str(summary_path))
    corrections = summary.get("rci_corrections")
    if corrections is None:
        variant["issues"].append(
            _issue("missing_rci_corrections", ".csum has no RCI correction block")
        )
    else:
        variant["hamiltonian"] = rci_hamiltonian(corrections)
        variant["corrections"] = corrections
    variant["energy_hartree"] = summary.get("ground_energy_au")
    variant["model"] = _model_signature(summary)

    stdout_path = variant_dir / "rci.stdout"
    if stdout_path.is_file():
        execution = parse_rci_log(str(stdout_path))
        variant["execution"] = execution
        if not execution["completed"]:
            variant["issues"].append(
                _issue("rci_not_completed", "RCI stdout does not prove completion")
            )
    else:
        variant["execution"] = {"completed": None}
        variant["issues"].append(
            _issue(
                "missing_rci_stdout",
                "RCI completion cannot be checked without stdout",
                "warning",
            )
        )

    variant["artifacts"] = _artifact_map(
        {
            "summary": summary_path,
            "csf": _select_file(variant_dir, "*.c", "state.c"),
            "radial_wavefunction": _select_file(variant_dir, "*.w", "state.w"),
            "nucleus": variant_dir / "isodata",
            "rci_input": variant_dir / "rci.in",
            "rci_stdout": stdout_path,
        }
    )
    for role in ("csf", "radial_wavefunction", "nucleus"):
        if role not in variant["artifacts"]:
            variant["issues"].append(
                _issue(f"missing_{role}", f"missing {role} artifact")
            )
    return variant


def _build_rci_pair(
    variants: list[dict[str, Any]],
    mcdhf_stage: dict[str, Any],
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    by_hamiltonian: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        hamiltonian = variant.get("hamiltonian")
        if hamiltonian:
            by_hamiltonian.setdefault(hamiltonian, []).append(variant)

    dc = by_hamiltonian.get("dirac_coulomb", [])
    breit = by_hamiltonian.get(
        "dirac_coulomb_plus_zero_frequency_breit",
        [],
    )
    if len(dc) != 1:
        issues.append(
            _issue("rci_dc_cardinality", f"expected one DC variant, found {len(dc)}")
        )
    if len(breit) != 1:
        issues.append(
            _issue(
                "rci_breit0_cardinality",
                f"expected one zero-frequency Breit variant, found {len(breit)}",
            )
        )
    pair: dict[str, Any] = {"issues": issues}
    if len(dc) != 1 or len(breit) != 1:
        return pair

    dc_variant = dc[0]
    breit_variant = breit[0]
    pair.update(
        {
            "dirac_coulomb_variant": dc_variant["variant"],
            "breit0_variant": breit_variant["variant"],
            "dirac_coulomb_energy_hartree": dc_variant.get("energy_hartree"),
            "breit0_energy_hartree": breit_variant.get("energy_hartree"),
        }
    )
    if (
        dc_variant.get("energy_hartree") is not None
        and breit_variant.get("energy_hartree") is not None
    ):
        correction = breit_variant["energy_hartree"] - dc_variant["energy_hartree"]
        pair["breit0_correction_hartree"] = correction
        pair["breit0_correction_microhartree"] = correction * 1.0e6

    pair["variant_model_checks"] = [
        {
            "field": field,
            "matches": dc_variant.get("model", {}).get(field)
            == breit_variant.get("model", {}).get(field),
        }
        for field in _MODEL_FIELDS
    ]
    pair["source_model_checks"] = [
        {
            "variant": variant["variant"],
            "field": field,
            "matches": variant.get("model", {}).get(field)
            == mcdhf_stage.get("model", {}).get(field),
        }
        for variant in (dc_variant, breit_variant)
        for field in _MODEL_FIELDS
    ]
    pair["input_hash_checks"] = _rci_input_hash_checks(
        (dc_variant, breit_variant),
        mcdhf_stage,
    )
    failed_checks = [
        check
        for group in (
            pair["variant_model_checks"],
            pair["source_model_checks"],
            pair["input_hash_checks"],
        )
        for check in group
        if not check["matches"]
    ]
    if failed_checks:
        issues.append(
            _issue(
                "rci_pair_model_mismatch",
                f"{len(failed_checks)} model or input identity checks failed",
            )
        )

    mcdhf_energy = mcdhf_stage.get("energy_hartree")
    dc_energy = dc_variant.get("energy_hartree")
    if mcdhf_energy is not None and dc_energy is not None:
        delta = dc_energy - mcdhf_energy
        pair["dc_minus_mcdhf_hartree"] = delta
        pair["dc_minus_mcdhf_microhartree"] = delta * 1.0e6
    return pair


def _rci_input_hash_checks(
    variants: tuple[dict[str, Any], dict[str, Any]],
    mcdhf_stage: dict[str, Any],
) -> list[dict[str, Any]]:
    checks = []
    for role in ("csf", "radial_wavefunction", "nucleus"):
        hashes = [
            variant.get("artifacts", {}).get(role, {}).get("sha256")
            for variant in variants
        ]
        checks.append(
            {
                "scope": "between_rci_variants",
                "role": role,
                "matches": None not in hashes and len(set(hashes)) == 1,
            }
        )
        source_hash = mcdhf_stage.get("artifacts", {}).get(role, {}).get("sha256")
        for variant, variant_hash in zip(variants, hashes, strict=True):
            checks.append(
                {
                    "scope": "rci_vs_mcdhf_source",
                    "variant": variant["variant"],
                    "role": role,
                    "matches": source_hash is not None and source_hash == variant_hash,
                }
            )
    return checks


def _parse_csf_model(path: Path, n_electrons: int | None) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    core_match = re.search(r"Core subshells:\s*\n([^\n]*)", text)
    peel_match = re.search(r"Peel subshells:\s*\n([^\n]*)", text)
    core = core_match.group(1).split() if core_match else []
    peel = peel_match.group(1).split() if peel_match else []
    inactive_electrons = sum(_subshell_capacity(label) for label in core)
    active_spinors = sum(_subshell_capacity(label) for label in peel)
    active_electrons = (
        n_electrons - inactive_electrons if n_electrons is not None else None
    )
    raw_dimension = None
    if active_electrons is not None and 0 <= active_electrons <= active_spinors:
        raw_dimension = math.comb(active_spinors, active_electrons)
    return {
        "core_subshells": core,
        "peel_subshells": peel,
        "inactive_electrons": inactive_electrons,
        "active_electrons": active_electrons,
        "active_spinors": active_spinors,
        "raw_determinant_dimension": raw_dimension,
    }


def _parse_rcsfgenerate_input(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        return {"parse_error": "input is shorter than the first CSF generation list"}
    try:
        separator = lines.index("", 2)
    except ValueError:
        return {"parse_error": "configuration list has no blank terminator"}
    tail = lines[separator + 1 :]
    if len(tail) < 4:
        return {"parse_error": "generation list is incomplete"}
    rank = int(tail[2]) if tail[2].strip().isdigit() else None
    two_j = [part.strip() for part in tail[1].split(",")]
    return {
        "ordering": lines[0],
        "core_code": lines[1],
        "reference_configurations": lines[2:separator],
        "active_orbitals": [
            orbital.strip() for orbital in tail[0].split(",") if orbital.strip()
        ],
        "two_j_min": int(two_j[0]) if len(two_j) == 2 and two_j[0].isdigit() else None,
        "two_j_max": int(two_j[1]) if len(two_j) == 2 and two_j[1].isdigit() else None,
        "substitution_rank": rank,
        "substitution_model": {
            0: "reference_only",
            1: "single",
            2: "single_double",
        }.get(rank, f"rank_{rank}" if rank is not None else None),
        "additional_generation_list": tail[3].strip().lower() == "y",
    }


def _model_signature(summary: dict[str, Any]) -> dict[str, Any]:
    signature = {
        field: summary.get(field)
        for field in _MODEL_FIELDS
        if field not in {"subshells", "state_sectors"}
    }
    signature["subshells"] = [
        subshell["label"] for subshell in summary.get("subshells", [])
    ]
    signature["state_sectors"] = sorted(
        {
            (level["j_str"], level["parity"])
            for level in summary.get("eigenenergies", [])
        }
    )
    return signature


def _shared_model_checks(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not stages or not stages[0].get("model"):
        return []
    reference = stages[0]["model"]
    return [
        {
            "stage": stage["stage"],
            "field": field,
            "matches": stage.get("model", {}).get(field) == reference.get(field),
        }
        for stage in stages[1:]
        for field in _SHARED_FIELDS
    ]


def _table_row(stage: dict[str, Any]) -> dict[str, Any]:
    csf_model = stage.get("csf_model", {})
    execution = stage.get("execution", {})
    pair = stage.get("rci", {}).get("pair", {})
    return {
        "stage": stage["stage"],
        "n_csfs": stage.get("model", {}).get("n_csfs"),
        "n_subshells": stage.get("model", {}).get("n_subshells"),
        "core_subshells": csf_model.get("core_subshells"),
        "active_electrons": csf_model.get("active_electrons"),
        "active_spinors": csf_model.get("active_spinors"),
        "raw_determinant_dimension": csf_model.get("raw_determinant_dimension"),
        "substitution_model": stage.get("generation", {}).get("substitution_model"),
        "mcdhf_energy_hartree": stage.get("energy_hartree"),
        "mcdhf_iterations": execution.get("n_iterations"),
        "mcdhf_converged": execution.get("converged"),
        "mcdhf_convergence_evidence": execution.get("convergence_evidence"),
        "mcdhf_max_final_orbital_self_consistency": execution.get(
            "max_final_orbital_self_consistency"
        ),
        "mcdhf_unstable_orbitals": execution.get(
            "growing_alternating_orbitals"
        ),
        "rci_dirac_coulomb_energy_hartree": pair.get("dirac_coulomb_energy_hartree"),
        "rci_breit0_energy_hartree": pair.get("breit0_energy_hartree"),
        "breit0_correction_microhartree": pair.get("breit0_correction_microhartree"),
        "rci_dc_minus_mcdhf_microhartree": pair.get("dc_minus_mcdhf_microhartree"),
    }


def _energy_increments(
    rows: list[dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    increments = []
    for previous, current in pairwise(rows):
        before = previous.get(field)
        after = current.get(field)
        if before is None or after is None:
            continue
        delta = after - before
        increments.append(
            {
                "from": previous["stage"],
                "to": current["stage"],
                "delta_hartree": delta,
                "delta_microhartree": delta * 1.0e6,
            }
        )
    return increments


def _check_energy_agreement(row: dict[str, Any], stdout_energy: float | None) -> None:
    summary_energy = row.get("energy_hartree")
    if summary_energy is None or stdout_energy is None:
        return
    delta = stdout_energy - summary_energy
    row["execution"]["stdout_minus_summary_hartree"] = delta
    if abs(delta) > 5.0e-9:
        row["issues"].append(
            _issue(
                "rmcdhf_energy_mismatch",
                f"stdout and .sum energies differ by {delta:.3e} hartree",
            )
        )


def _subshell_capacity(label: str) -> int:
    match = _SUBSHELL_RE.fullmatch(label)
    if match is None:
        raise ValueError(f"unsupported GRASP subshell label: {label!r}")
    angular_momentum = _ANGULAR_MOMENTUM[match.group(1)]
    return 2 * angular_momentum if match.group(2) == "-" else 2 * (angular_momentum + 1)


def _artifact_map(paths: dict[str, Path | None]) -> dict[str, dict[str, Any]]:
    return {
        role: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for role, path in paths.items()
        if path is not None and path.is_file()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_file(directory: Path, pattern: str, preferred: str) -> Path | None:
    preferred_path = directory / preferred
    if preferred_path.is_file():
        return preferred_path
    matches = sorted(directory.glob(pattern))
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous {pattern} selection in {directory}: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0] if matches else None


def _resolve_stages(root: Path, stages: list[str] | None) -> list[str]:
    if stages is None:
        names = sorted(
            child.name
            for child in root.iterdir()
            if child.is_dir() and any(child.glob("*.sum"))
        )
    else:
        names = list(stages)
    if not names:
        raise ValueError("no MCDHF stage directories were selected")
    if len(names) > _MAX_STAGES:
        raise ValueError(f"at most {_MAX_STAGES} stages may be inspected")
    if len(set(names)) != len(names):
        raise ValueError("stages must be unique")
    invalid = [
        name
        for name in names
        if not name or name in {".", ".."} or Path(name).name != name
    ]
    if invalid:
        raise ValueError(f"invalid stage names: {invalid}")
    return names


def _require_directory(path: str | None, field: str) -> Path:
    if path is None:
        raise ValueError(f"{field} is required")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"{field} is not a directory: {resolved}")
    return resolved


def _issue(code: str, message: str, severity: str = "error") -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


def _verdict(issues: list[dict[str, Any]]) -> str:
    if any(issue["severity"] == "error" for issue in issues):
        return "invalid"
    if issues:
        return "partial"
    return "complete"


def _headline(verdict: str, stage_count: int, has_rci: bool) -> str:
    scope = "MCDHF and paired RCI" if has_rci else "MCDHF"
    if verdict == "complete":
        return f"{stage_count} {scope} stages passed artifact and model checks"
    if verdict == "partial":
        return f"{stage_count} {scope} stages parsed with incomplete evidence"
    return f"{stage_count} {scope} stages contain comparison-blocking mismatches"


__all__ = ["inspect_grasp_case_directory"]

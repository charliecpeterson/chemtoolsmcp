"""Single-run summaries and bulk triage for DIRAC outputs.

The bulk path returns one compact row per file. The single-run path combines
the text result with optional checkpoint evidence.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from chemtools.core.common import resolve_output_paths
from chemtools.programs.dirac.binary import (
    H5PY_AVAILABLE,
    read_metadata,
    read_orbital_summary,
)
from chemtools.programs.dirac.parse.output import parse_output


def summarize_dirac_run(
    output_file: str,
    h5_file: str | None = None,
) -> dict[str, Any]:
    """Combine one DIRAC text output with optional checkpoint evidence."""
    parsed = parse_output(output_file)
    summary: dict[str, Any] = {
        "program": "dirac",
        "output_file": output_file,
        "program_version": parsed.get("program_version"),
        "tasks_detected": parsed.get("tasks_detected"),
        "total_energy_hartree": parsed.get("total_energy_hartree"),
        "scf_converged": parsed.get("scf_converged"),
        "scf_n_iterations": parsed.get("scf_n_iterations"),
        "symmetry": parsed.get("symmetry"),
        "open_shell_setup": parsed.get("open_shell_setup"),
        "homo_lumo_blocks_count": len(
            parsed.get("homo_lumo_per_symmetry") or []
        ),
    }

    excitations = parsed.get("excitations") or {}
    if excitations.get("available"):
        summary["excited_states"] = {
            "n_excitations": excitations["n_excitations"],
            "lowest_excitation_ev": excitations.get("lowest_excitation_ev"),
            "sum_oscillator_strength": excitations.get(
                "sum_oscillator_strength"
            ),
            "excitations": excitations["excitations"],
        }

    correlation = parsed.get("relccsd") or {}
    if correlation.get("available"):
        summary["correlation"] = {
            "mp2_total_hartree": correlation.get("mp2_total_hartree"),
            "ccsd_total_hartree": correlation.get("ccsd_total_hartree"),
            "ccsd_t_total_hartree": correlation.get("ccsd_t_total_hartree"),
            "mp2_correlation_hartree": correlation.get(
                "mp2_correlation_hartree"
            ),
            "ccsd_correlation_hartree": correlation.get(
                "ccsd_correlation_hartree"
            ),
        }

    cosci = parsed.get("cosci") or {}
    if cosci.get("n_states"):
        states = cosci["states"]
        summary["open_shell_states"] = {
            "n_states": cosci["n_states"],
            "highest_excitation_cm1": max(
                (state["energy_cm1"] for state in states),
                default=None,
            ),
            "states": states,
        }

    if h5_file:
        _add_checkpoint_evidence(summary, h5_file)

    if summary.get("scf_converged"):
        summary["verdict"] = "scf_converged"
    elif summary.get("scf_n_iterations"):
        summary["verdict"] = "scf_did_not_converge"
    else:
        summary["verdict"] = "no_scf_detected"
    return summary


def _add_checkpoint_evidence(
    summary: dict[str, Any],
    h5_file: str,
) -> None:
    if not H5PY_AVAILABLE:
        summary["h5_status"] = "h5py_missing"
        return
    try:
        metadata = read_metadata(h5_file)
        summary["h5_status"] = "loaded"
        summary["h5_version"] = metadata.get("version")
        summary["h5_scf_energy_hartree"] = metadata.get(
            "scf_energy_hartree"
        )
        summary["n_fermion_symmetries"] = metadata.get(
            "n_fermion_symmetries"
        )
        summary["n_mo_per_fsym"] = metadata.get("n_mo_per_fsym")
        summary["n_pos_energy_per_fsym"] = metadata.get(
            "n_pos_energy_per_fsym"
        )
        shell_counts: dict[str, int] = {}
        for orbital in read_orbital_summary(h5_file):
            shell_class = orbital["shell_class"]
            shell_counts[shell_class] = shell_counts.get(shell_class, 0) + 1
        summary["shell_class_counts"] = shell_counts
        text_energy = summary["total_energy_hartree"]
        checkpoint_energy = summary.get("h5_scf_energy_hartree")
        if text_energy is not None and checkpoint_energy is not None:
            summary["text_vs_h5_energy_consistent"] = (
                abs(text_energy - checkpoint_energy) < 1e-6
            )
    except Exception as error:
        summary["h5_status"] = f"error: {error}"


def summarize_dirac_outputs(
    paths: str | list[str],
    pattern: str = "*.out",
    recursive: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    files = resolve_output_paths(paths, pattern, recursive)
    truncated = limit is not None and len(files) > limit
    if truncated:
        files = files[:limit]

    rows: list[dict[str, Any]] = []
    verdicts: Counter = Counter()
    for path in files:
        try:
            parsed = parse_output(path)
        except Exception as exc:  # one unreadable file must not sink the batch
            rows.append({"file": Path(path).name, "path": path, "verdict": "error",
                         "error": f"{type(exc).__name__}: {exc}"})
            verdicts["error"] += 1
            continue
        converged = parsed.get("scf_converged")
        n_iter = parsed.get("scf_n_iterations")
        if converged:
            verdict, headline = "healthy", "SCF converged"
        elif n_iter:
            verdict, headline = "problem", "SCF did not converge"
        else:
            verdict, headline = "caution", "no SCF detected"
        verdicts[verdict] += 1
        row = {
            "file": Path(path).name,
            "path": path,
            "tasks": parsed.get("tasks_detected"),
            "total_energy_hartree": parsed.get("total_energy_hartree"),
            "scf_converged": converged,
            "scf_n_iterations": n_iter,
            "symmetry": parsed.get("symmetry"),
            "verdict": verdict,
            "headline": headline,
        }
        exc = parsed.get("excitations") or {}
        if exc.get("available"):
            row["excited_states"] = {
                "n_excitations": exc["n_excitations"],
                "lowest_excitation_ev": exc.get("lowest_excitation_ev"),
            }
        cc = parsed.get("relccsd") or {}
        if cc.get("available"):
            row["correlation"] = {
                "ccsd_t_total_hartree": cc.get("ccsd_t_total_hartree"),
                "ccsd_total_hartree": cc.get("ccsd_total_hartree"),
            }
        cosci = parsed.get("cosci") or {}
        if cosci.get("n_states"):
            row["open_shell_states"] = {"n_states": cosci["n_states"]}
        rows.append(row)
    return {
        "file_count": len(files),
        "truncated": truncated,
        "verdict_counts": dict(verdicts),
        "rows": rows,
    }

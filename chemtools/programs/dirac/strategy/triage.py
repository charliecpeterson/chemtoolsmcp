"""Bulk triage over many DIRAC outputs — one compact row per file.

The Dirac counterpart of summarize_nwchem_outputs / summarize_molcas_outputs:
assess a directory / glob / list of runs in a single call instead of one
summarize_dirac_run per file.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from chemtools.core.common import resolve_output_paths
from chemtools.programs.dirac.parse.output import parse_output


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

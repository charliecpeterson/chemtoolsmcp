"""Bulk triage over many GRASP working directories — one compact row per run.

The GRASP counterpart of summarize_{nwchem,molcas,dirac}_outputs. GRASP's unit
of analysis is a working *directory* (one atom/term per dir: its .sum, .lsj.lbl,
run log), not a single output file, so this walks subdirectories rather than
globbing files. Built for the actinide-screening workflow: run many atoms, then
assess them all in one call.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from chemtools.core.common import ELEMENT_TO_Z
from chemtools.programs.grasp.parse.sum_file import parse_sum
from chemtools.programs.grasp.parse.lsjlbl import parse_lsjlbl
from chemtools.programs.grasp.parse.rlevels import parse_rlevels

_Z_TO_ELEMENT = {z: sym for sym, z in ELEMENT_TO_Z.items()}


def _is_grasp_dir(d: Path) -> bool:
    return any(d.glob("*.sum")) or any(d.glob("*.lsj.lbl")) or (d / "isodata").exists()


def _find_grasp_dirs(path: str, recursive: bool) -> list[Path]:
    root = Path(path)
    if not root.exists():
        return []
    if root.is_dir() and _is_grasp_dir(root):
        return [root]
    walker = root.rglob("*") if recursive else root.iterdir()
    return sorted(d for d in walker if d.is_dir() and _is_grasp_dir(d))


def _first(d: Path, pattern: str) -> Path | None:
    hits = sorted(d.glob(pattern))
    return hits[0] if hits else None


def _summarize_dir(d: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"run": d.name, "path": str(d)}
    sum_path = _first(d, "*.sum")
    lbl_path = _first(d, "*.lsj.lbl")

    summary = None
    if sum_path:
        try:
            summary = parse_sum(sum_path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            return {**row, "verdict": "error", "error": f"{type(exc).__name__}: {exc}"}

    if summary:
        z = summary.get("atomic_number")
        row["atomic_number"] = z
        row["element"] = _Z_TO_ELEMENT.get(int(z)) if z else None
        row["n_csfs"] = summary.get("n_csfs")
        row["n_subshells"] = summary.get("n_subshells")
        row["is_nonrel_limit"] = summary.get("is_nonrel_limit")

    levels = []
    if lbl_path:
        try:
            levels = (parse_lsjlbl(lbl_path.read_text(encoding="utf-8", errors="replace")) or {}).get("levels", [])
        except Exception:
            levels = []
    if levels:
        ground = min(levels, key=lambda lv: lv.get("energy_au", float("inf")))
        row["n_levels"] = len(levels)
        row["ground_energy_au"] = ground.get("energy_au")
        row["ground_term"] = ground.get("dominant_label")

    # Optional: level splittings from the rlevels stdout, if captured.
    run_out = _first(d, "run_*.out")
    if run_out:
        try:
            rl = parse_rlevels(run_out.read_text(encoding="utf-8", errors="replace"))
            if rl.get("max_splitting_cm1") is not None:
                row["max_splitting_cm1"] = rl["max_splitting_cm1"]
        except Exception:
            pass

    if levels and row.get("ground_energy_au") is not None:
        row["verdict"], row["headline"] = "healthy", f"{row.get('n_levels')} levels, ground term {row.get('ground_term')}"
    elif summary:
        row["verdict"], row["headline"] = "caution", "SCF summary present but no LSJ levels (jj2lsj not run?)"
    else:
        row["verdict"], row["headline"] = "caution", "no .sum / .lsj.lbl found"
    return row


def summarize_grasp_runs(
    paths: str,
    recursive: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Assess every GRASP working directory under ``paths`` — one row per run."""
    dirs = _find_grasp_dirs(paths, recursive)
    truncated = limit is not None and len(dirs) > limit
    if truncated:
        dirs = dirs[:limit]

    rows = [_summarize_dir(d) for d in dirs]
    verdicts: Counter = Counter(r["verdict"] for r in rows)
    elements: Counter = Counter(r["element"] for r in rows if r.get("element"))
    return {
        "n_runs": len(rows),
        "runs": rows,
        "verdict_counts": dict(verdicts),
        "element_counts": dict(elements),
        "truncated": truncated,
    }

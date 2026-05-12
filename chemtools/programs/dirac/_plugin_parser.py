"""DIRAC Parser sub-protocol implementation."""

from __future__ import annotations

from typing import Any

from chemtools.core.types import ParsedRun, TaskSummary, GeometryAtom
from chemtools.programs.dirac.parse import (
    parse_output,
    parse_inp,
    parse_mol,
    looks_like_dirac,
    parse_version,
)


class _DiracParser:
    """Adapts the DIRAC text/inp/mol parsers to the chemtools Parser protocol."""

    def parse_output(self, path: str) -> ParsedRun:
        text = _read(path)
        deep = parse_output(path, contents=text)
        return self._to_parsed_run(path, text, deep)

    def task_index(self, path: str) -> list[TaskSummary]:
        text = _read(path)
        deep = parse_output(path, contents=text)
        return self._build_task_summaries(deep)

    def parse_input(self, path: str) -> dict[str, Any]:
        return parse_inp(path)

    # --- Drill-down sub-protocols ----------------------------------------

    def get_orbitals(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        # Text-output orbital data is sparse; the rich orbital data lives in
        # the .h5 checkpoint. Direct the caller there.
        raise NotImplementedError(
            "DIRAC orbital coefficients are in the .h5 checkpoint — "
            "use the binary reader (BinaryReader.parse(h5_path, 'mocoef'))."
        )

    def get_frequency(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        raise NotImplementedError(
            "DIRAC frequency parser not implemented in Phase DA — coming in DD."
        )

    def get_trajectory(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        raise NotImplementedError(
            "DIRAC geometry optimization trajectory parser not implemented in Phase DA."
        )

    def get_thermochem(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        raise NotImplementedError(
            "DIRAC thermochem parser not implemented in Phase DA."
        )

    def get_geometry(self, path: str, task_index: int | None = None) -> list[GeometryAtom]:
        # If we got a .mol path, parse directly. If we got an .out, look
        # for a sibling .mol. DIRAC's pam wrapper names the output
        # ``<inp_stem>_<mol_stem>.out`` so we can recover the mol stem.
        from pathlib import Path
        p = Path(path)
        if p.suffix.lower() == ".mol":
            mol = parse_mol(path)
            return _atoms_to_geometry(mol)

        # Try, in order:
        #   1) <stem>.mol               (output named after mol)
        #   2) <stem.split("_")[-1]>.mol   (pam naming: inp_mol.out)
        #   3) <stem.split("_")[0]>.mol    (alternative split)
        #   4) any .mol next to it      (fallback for unconventional naming)
        stem = p.stem
        candidates = [p.with_suffix(".mol")]
        if "_" in stem:
            candidates.append(p.parent / (stem.split("_")[-1] + ".mol"))
            candidates.append(p.parent / (stem.split("_")[0] + ".mol"))
        for cand in candidates:
            if cand.exists():
                return _atoms_to_geometry(parse_mol(str(cand)))
        # Fallback: any sibling .mol in the same directory
        for cand in p.parent.glob("*.mol"):
            return _atoms_to_geometry(parse_mol(str(cand)))
        return []

    # --- Internal helpers -----------------------------------------------

    def _to_parsed_run(self, path: str, text: str, deep: dict[str, Any]) -> ParsedRun:
        tasks = self._build_task_summaries(deep)
        primary = (
            tasks[0]["energy_hartree"]
            if tasks and tasks[0].get("energy_hartree") is not None
            else deep.get("total_energy_hartree")
        )

        diagnostics: list[str] = []
        if not deep.get("scf_converged"):
            diagnostics.append("scf_not_converged")
        if not tasks:
            diagnostics.append("no_tasks_detected")

        outcome = "success" if deep.get("scf_converged") else (
            "incomplete" if deep.get("scf_iterations") else "failed"
        )

        return {
            "program": "dirac",
            "program_version": deep.get("program_version"),
            "file": path,
            "file_size_bytes": len(text),
            "tasks": tasks,
            "primary_task_index": 0 if tasks else None,
            "derived": {
                "n_tasks": len(tasks),
                "primary_energy_hartree": primary,
                "final_energy_hartree": primary,
                "scf_converged": deep.get("scf_converged"),
                "scf_n_iterations": deep.get("scf_n_iterations"),
                "symmetry": deep.get("symmetry"),
                "open_shell_setup": deep.get("open_shell_setup"),
            },
            "diagnostics": diagnostics,
            "diagnosis": {
                "verdict": outcome,
                "summary": _diagnosis_summary(deep, tasks),
            },
        }

    def _build_task_summaries(self, deep: dict[str, Any]) -> list[TaskSummary]:
        tasks: list[TaskSummary] = []
        e = deep.get("total_energy_hartree")
        for kind in deep.get("tasks_detected") or ["scf"]:
            tasks.append({
                "program": "dirac",
                "kind": kind,
                "label": kind.upper(),
                "energy_hartree": e if kind in ("scf", "dft") else None,
                "line_start": 0,
                "line_end": 0,
                "extra": {
                    "scf_converged": deep.get("scf_converged"),
                    "scf_n_iterations": deep.get("scf_n_iterations"),
                },
            })
        return tasks


def _read(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _atoms_to_geometry(mol: dict[str, Any]) -> list[GeometryAtom]:
    """Convert parse_mol() output to the unified GeometryAtom shape.

    Coordinates from .mol may be in bohr (default) or angstrom (when the
    coord header carries the 'A' flag). The generic core.geometry
    convention is angstrom — convert here when needed.
    """
    from chemtools.core.units import ANGSTROM_PER_BOHR

    is_bohr = mol.get("units") == "bohr"
    scale = ANGSTROM_PER_BOHR if is_bohr else 1.0

    out: list[GeometryAtom] = []
    for a in mol.get("atoms", []):
        element = _z_to_element(int(a.get("nuclear_charge", 0)))
        out.append({
            "element": element or a.get("label"),
            "x": a["x"] * scale,
            "y": a["y"] * scale,
            "z": a["z"] * scale,
        })
    return out


def _z_to_element(z: int) -> str | None:
    """Look up atomic symbol by Z. Falls back to None for Z<=0 or unknown."""
    from chemtools.core.common import ATOMIC_SYMBOLS
    return ATOMIC_SYMBOLS.get(z)


def _diagnosis_summary(deep: dict[str, Any], tasks: list[TaskSummary]) -> str:
    e = deep.get("total_energy_hartree")
    n = deep.get("scf_n_iterations") or 0
    converged = deep.get("scf_converged")
    open_shell = deep.get("open_shell_setup")
    parts = [
        f"{', '.join(t['kind'] for t in tasks) or 'no tasks'}",
        f"total_energy={e:.6f} Ha" if e is not None else "no energy parsed",
        f"SCF {n} iter, {'converged' if converged else 'not converged'}",
    ]
    if open_shell and open_shell.get("aoc"):
        parts.append("AOC open-shell run")
    return "; ".join(parts)


DIRAC_PARSER = _DiracParser()

__all__ = ["DIRAC_PARSER"]

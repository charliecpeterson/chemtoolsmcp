"""NWChem Parser sub-protocol implementation.

Adapter layer that translates the existing parsers in
`chemtools.programs.nwchem.parse.*` into the program-neutral shapes defined
in `chemtools.core.types` (`ParsedRun`, `TaskSummary`).

Once an agent or MCP tool resolves the NWChem backend, it can call
`backend.parser.parse_output(path)` and get a uniform structured result
regardless of which program produced the output file.

The methods here are thin — they read the file, call the existing parsers,
and reshape the result. They do not introduce new parsing logic.
"""

from __future__ import annotations
import os
from typing import Any

from chemtools.core.common import read_text
from chemtools.core.types import ParsedRun, TaskSummary, GeometryAtom
from chemtools.programs._adapter_helpers import (
    to_task_summary as _to_task_summary,
    pick_primary as _pick_primary,
    compute_derived as _compute_derived,
)
from chemtools.programs.nwchem.parse.tasks import parse_tasks as _parse_tasks
from chemtools.programs.nwchem.parse.mos import parse_mos as _parse_mos
from chemtools.programs.nwchem.parse.freq import (
    NEAR_ZERO_FREQUENCY_THRESHOLD_CM1,
    SIGNIFICANT_IMAGINARY_FREQUENCY_THRESHOLD_CM1,
    parse_freq as _parse_freq,
    parse_trajectory as _parse_trajectory,
)
from chemtools.programs.nwchem.parse.input import inspect_nwchem_input


class _NwchemParser:
    """Implements chemtools.core.program.Parser for NWChem."""

    def parse_output(self, path: str) -> ParsedRun:
        contents = read_text(path)
        tasks_result = _parse_tasks(path, contents)
        program_summary = tasks_result.get("program_summary") or {}
        raw_tasks = (program_summary.get("raw") or {}).get("tasks") or []
        generic_tasks = tasks_result.get("generic_tasks") or []

        # Pair up generic_tasks with raw tasks (they should be parallel lists).
        n = max(len(generic_tasks), len(raw_tasks))
        summaries: list[TaskSummary] = []
        for i in range(n):
            g = generic_tasks[i] if i < len(generic_tasks) else {}
            r = raw_tasks[i] if i < len(raw_tasks) else {}
            summaries.append(_to_task_summary(i, g, r))

        primary_idx = _pick_primary(summaries)
        derived = _compute_derived(summaries, raw_tasks)
        frequency_values = [
            mode["frequency_cm1"]
            for task in raw_tasks
            for mode in task.get("frequency_modes") or []
            if isinstance(mode.get("frequency_cm1"), (int, float))
        ]
        if frequency_values:
            significant_imaginary = [
                value
                for value in frequency_values
                if value <= SIGNIFICANT_IMAGINARY_FREQUENCY_THRESHOLD_CM1
            ]
            derived["n_imaginary_modes"] = len(significant_imaginary)
            derived["significant_imaginary_frequencies_cm1"] = (
                significant_imaginary
            )
            derived["n_near_zero_modes"] = sum(
                abs(value) < NEAR_ZERO_FREQUENCY_THRESHOLD_CM1
                for value in frequency_values
            )
        diagnostics = [
            {
                "kind": d.get("kind", "info"),
                "message": d.get("message", ""),
                "line": d.get("line"),
                "file": path,
            }
            for d in (program_summary.get("diagnostics") or [])
        ]

        try:
            file_size = os.path.getsize(path)
        except OSError:
            file_size = 0

        return {
            "program": "nwchem",
            "program_version": None,
            "file": path,
            "file_size_bytes": file_size,
            "tasks": summaries,
            "primary_task_index": primary_idx,
            "derived": derived,
            "diagnostics": diagnostics,
            "diagnosis": {},
        }

    def task_index(self, path: str) -> list[TaskSummary]:
        contents = read_text(path)
        tasks_result = _parse_tasks(path, contents)
        raw_tasks = (tasks_result.get("program_summary") or {}).get("raw", {}).get("tasks") or []
        generic_tasks = tasks_result.get("generic_tasks") or []
        n = max(len(generic_tasks), len(raw_tasks))
        return [
            _to_task_summary(
                i,
                generic_tasks[i] if i < len(generic_tasks) else {},
                raw_tasks[i] if i < len(raw_tasks) else {},
            )
            for i in range(n)
        ]

    def parse_input(self, path: str) -> dict[str, Any]:
        return inspect_nwchem_input(path)

    def get_orbitals(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        # MO parsing currently treats the file as one aggregate section.
        contents = read_text(path)
        return _parse_mos(path, contents, top_n=10, include_coefficients=False)

    def get_frequency(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        contents = read_text(path)
        return _parse_freq(path, contents)

    def get_trajectory(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        contents = read_text(path)
        return _parse_trajectory(path, contents, include_positions=True)

    def get_thermochem(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        # parse_nwchem_thermochem lives in programs/nwchem/output.py (MCP wrapper).
        from chemtools.programs.nwchem.output import parse_nwchem_thermochem
        return parse_nwchem_thermochem(path)

    def get_geometry(self, path: str, task_index: int | None = None) -> list[GeometryAtom]:
        from chemtools.programs.nwchem.input.geometry import (
            extract_nwchem_geometry,
        )
        result = extract_nwchem_geometry(path)
        atoms = result.get("atoms") or []
        out: list[GeometryAtom] = []
        for a in atoms:
            out.append({
                "element": a.get("element") or a.get("symbol"),
                "x": float(a.get("x", 0.0)),
                "y": float(a.get("y", 0.0)),
                "z": float(a.get("z", 0.0)),
            })
        return out


NWCHEM_PARSER = _NwchemParser()


__all__ = ["NWCHEM_PARSER"]

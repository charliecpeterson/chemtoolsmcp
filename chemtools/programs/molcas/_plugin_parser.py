"""Molcas Parser sub-protocol implementation (parse_tasks only stub)."""

from __future__ import annotations
import os
from typing import Any

from chemtools.core.common import read_text
from chemtools.core.types import ParsedRun, TaskSummary, GeometryAtom
from chemtools.programs._adapter_helpers import (
    to_task_summary,
    pick_primary,
    compute_derived,
)
from chemtools.programs.molcas.parse.output import parse_tasks as _parse_tasks


class _MolcasParser:
    """Implements chemtools.core.program.Parser for Molcas.

    Currently a stub: only parse_output / task_index work. Every other Parser
    method raises NotImplementedError until corresponding Molcas parsers are
    written.
    """

    def parse_output(self, path: str) -> ParsedRun:
        contents = read_text(path)
        tasks_result = _parse_tasks(path, contents)
        generic_tasks = tasks_result.get("generic_tasks") or []

        summaries: list[TaskSummary] = [
            to_task_summary(i, g, None) for i, g in enumerate(generic_tasks)
        ]
        primary_idx = pick_primary(summaries)
        derived = compute_derived(summaries, [])

        try:
            file_size = os.path.getsize(path)
        except OSError:
            file_size = 0

        return {
            "program": "molcas",
            "program_version": None,
            "file": path,
            "file_size_bytes": file_size,
            "tasks": summaries,
            "primary_task_index": primary_idx,
            "derived": derived,
            "diagnostics": [],
            "diagnosis": {},
        }

    def task_index(self, path: str) -> list[TaskSummary]:
        contents = read_text(path)
        tasks_result = _parse_tasks(path, contents)
        generic_tasks = tasks_result.get("generic_tasks") or []
        return [to_task_summary(i, g, None) for i, g in enumerate(generic_tasks)]

    def parse_input(self, path: str) -> dict[str, Any]:
        raise NotImplementedError("Molcas input parser not yet implemented")

    def get_orbitals(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        raise NotImplementedError("Molcas orbital parser not yet implemented")

    def get_frequency(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        raise NotImplementedError("Molcas frequency parser not yet implemented")

    def get_trajectory(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        raise NotImplementedError("Molcas trajectory parser not yet implemented")

    def get_thermochem(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        raise NotImplementedError("Molcas thermochemistry parser not yet implemented")

    def get_geometry(self, path: str, task_index: int | None = None) -> list[GeometryAtom]:
        raise NotImplementedError("Molcas geometry extraction not yet implemented")


MOLCAS_PARSER = _MolcasParser()


__all__ = ["MOLCAS_PARSER"]

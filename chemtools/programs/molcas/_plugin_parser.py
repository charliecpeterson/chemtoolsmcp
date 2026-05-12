"""Molcas Parser sub-protocol implementation."""

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
from chemtools.programs.molcas.parse.output import parse_output_full as _parse_output_full
from chemtools.programs.molcas.parse.mos import parse_last_mo_block as _parse_last_mo_block
from chemtools.programs.molcas.parse.freq import parse_last_freq_block as _parse_last_freq_block
from chemtools.programs.molcas.parse.thermochem import parse_thermochem_block as _parse_thermochem_block
from chemtools.programs.molcas.parse.geometry import (
    parse_final_geometry as _parse_final_geometry,
    parse_trajectory as _parse_trajectory,
    parse_cartesian_blocks as _parse_cartesian_blocks,
)


class _MolcasParser:
    """Implements chemtools.core.program.Parser for Molcas.

    `parse_output` runs the full orchestrator and stuffs the rich payload onto
    each task's `extra` field so the agent can access SCF / RASSCF / CASPT2
    energies plus active-space diagnostics in one call. Per-task drill-downs
    (`get_orbitals`) return the LAST '++ Molecular orbitals:' block from that
    task — RASSCF NOs override SCF MOs, which is what callers want.
    """

    def parse_output(self, path: str) -> ParsedRun:
        contents = read_text(path)
        full = _parse_output_full(path, contents, parse_mo_coefficients=False)
        generic_tasks = full["tasks_overview"]
        payloads = full["task_payloads"]
        # Stuff per-task module details into the generic task `extra` so the
        # standard ParsedRun envelope carries the rich payload.
        merged_tasks: list[TaskSummary] = []
        for i, gtask in enumerate(generic_tasks):
            payload = payloads[i] if i < len(payloads) else {}
            extra = dict(gtask.get("extra") or {})
            extra["details"] = payload.get("details") or {}
            extra["return_code"] = payload.get("return_code")
            merged = {**gtask, "extra": extra}
            # Hoist primary energy onto the task itself when available
            details = extra["details"]
            if details.get("module") == "RASSCF":
                roots = details.get("root_energies") or []
                if roots:
                    merged["energy_hartree"] = roots[0]["energy_hartree"]
            elif details.get("module") == "CASPT2":
                ms = details.get("ms_root_energies") or []
                ss = details.get("ss_root_energies") or []
                if ms:
                    merged["energy_hartree"] = ms[0]["energy_hartree"]
                elif ss:
                    merged["energy_hartree"] = ss[0]["energy_hartree"]
            elif details.get("module") == "SCF":
                e = details.get("final_energy", {}).get("total")
                if e is not None:
                    merged["energy_hartree"] = e
            merged_tasks.append(to_task_summary(i, merged, None))

        primary_idx = pick_primary(merged_tasks)
        derived = compute_derived(merged_tasks, [])
        # Hoist the cross-task energy summary so downstream consumers get the
        # canonical "primary_energy" pick (MS-CASPT2 > CASPT2 > RASSCF > SCF).
        # Also expose under `final_energy_hartree` (Phase 5 standardization)
        # so generic tools that consume ParsedRun see the same key name as
        # NWChem's derived dict.
        if (es := full.get("energy_summary")) and es.get("primary_energy_hartree") is not None:
            derived["primary_energy_hartree"] = es["primary_energy_hartree"]
            derived["primary_energy_label"] = es.get("primary_label")
            derived.setdefault("final_energy_hartree", es["primary_energy_hartree"])
        if full.get("active_space_summary"):
            derived["active_space"] = full["active_space_summary"]

        try:
            file_size = os.path.getsize(path)
        except OSError:
            file_size = 0

        return {
            "program": "molcas",
            "program_version": None,
            "file": path,
            "file_size_bytes": file_size,
            "tasks": merged_tasks,
            "primary_task_index": primary_idx,
            "derived": derived,
            "diagnostics": full.get("warnings") or [],
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
        """Return the LAST `++ Molecular orbitals:` block from the chosen task.

        For RASSCF tasks this returns the natural-orbital block (with
        occupations and dominant AO contributions per orbital); for SCF tasks
        it returns the canonical SCF MOs.
        """
        contents = read_text(path)
        tasks_result = _parse_tasks(path, contents)
        generic_tasks = tasks_result.get("generic_tasks") or []
        if not generic_tasks:
            raise ValueError(f"No tasks found in {path}")
        if task_index is None:
            # Prefer last RASSCF, else last SCF
            preferred = None
            for i, t in enumerate(generic_tasks):
                if t["extra"]["module"] in {"RASSCF", "SCF"}:
                    preferred = i
            if preferred is None:
                raise ValueError(f"No SCF/RASSCF tasks found in {path}")
            task_index = preferred
        if task_index < 0 or task_index >= len(generic_tasks):
            raise IndexError(f"task_index={task_index} out of range; have {len(generic_tasks)} tasks")
        task = generic_tasks[task_index]
        lines = contents.splitlines()
        block_text = "\n".join(lines[task["line_start"] - 1 : task["line_end"]])
        mo = _parse_last_mo_block(block_text, parse_coefficients=True)
        if mo is None:
            raise ValueError(
                f"No '++ Molecular orbitals:' block in task {task_index} ({task['extra']['module']})"
            )
        return {
            "task_index": task_index,
            "module": task["extra"]["module"],
            "mo_block": mo,
        }

    def get_frequency(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        """Return the LAST `Harmonic frequencies` block (typically MCLR-emitted)."""
        contents = read_text(path)
        block = _parse_last_freq_block(contents)
        if block is None:
            raise ValueError(f"No 'Harmonic frequencies' block found in {path}")
        return block

    def get_trajectory(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        """Walk the SLAPAF Energy Statistics + per-iteration geometries."""
        contents = read_text(path)
        return _parse_trajectory(contents)

    def get_thermochem(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        """Parse the per-temperature thermochemistry block (post-frequency)."""
        contents = read_text(path)
        block = _parse_thermochem_block(contents)
        if block is None:
            raise ValueError(f"No thermochemistry block found in {path}")
        return block

    def get_geometry(self, path: str, task_index: int | None = None) -> list[GeometryAtom]:
        """Return the converged geometry, ALWAYS in Angstrom.

        SLAPAF's "Nuclear coordinates for the next iteration" section is in
        bohr; this method normalizes to Å so generic callers (the Phase 4
        ``inspect_geometry`` MCP tool, etc.) can rely on a single unit
        regardless of where the geometry came from in the .out.
        """
        from chemtools.core.units import ANGSTROM_PER_BOHR

        contents = read_text(path)
        if task_index is not None:
            blocks = _parse_cartesian_blocks(contents)
            if not blocks:
                raise ValueError(f"No Cartesian coordinates blocks found in {path}")
            if task_index < 0 or task_index >= len(blocks):
                raise IndexError(
                    f"task_index={task_index} out of range; have {len(blocks)} geometry blocks"
                )
            block = blocks[task_index]
        else:
            block = _parse_final_geometry(contents)
            if block is None:
                raise ValueError(f"No geometry found in {path}")

        atoms = block["atoms"]
        units = (block.get("units") or "angstrom").lower()
        if units == "bohr":
            atoms = [
                {**a,
                 "x": a["x"] * ANGSTROM_PER_BOHR,
                 "y": a["y"] * ANGSTROM_PER_BOHR,
                 "z": a["z"] * ANGSTROM_PER_BOHR}
                for a in atoms
            ]
        return atoms  # type: ignore[return-value]


MOLCAS_PARSER = _MolcasParser()


__all__ = ["MOLCAS_PARSER"]

"""GRASP Parser sub-protocol implementation.

Adapts the file-specific GRASP parsers (rmcdhf .sum, .lsj.lbl, rmcdhf SCF
log, rlevels output) into the chemtools Parser protocol so generic tools
like ``parse_output`` / ``summarize_output`` can route to them.

Because GRASP doesn't have a single canonical "output" file, this parser
dispatches by file extension:

  *.sum           → rmcdhf summary (orbital list + final energy)
  *.lsj.lbl       → LSJ-coupled compositions
  *.log           → if it contains "Iteration number" sections, treat as
                    rmcdhf SCF iteration trace; otherwise just the rmcdhf
                    input-log copy (no useful structured data)
  rlevels stdout  → energy-level table (caller passes the captured stdout
                    via the ``contents`` arg of parse_output)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.core.types import ParsedRun, TaskSummary
from chemtools.programs.grasp.parse.sum_file import parse_sum
from chemtools.programs.grasp.parse.lsjlbl import parse_lsjlbl
from chemtools.programs.grasp.parse.rlevels import parse_rlevels
from chemtools.programs.grasp.parse.rmcdhf_log import parse_rmcdhf_log


class _GraspParser:
    """File-type-dispatching parser for GRASP2018 artifacts."""

    def parse_output(self, path: str) -> ParsedRun:
        text = _read(path)
        kind, parsed = _route(path, text)
        return self._to_parsed_run(path, text, kind, parsed)

    def task_index(self, path: str) -> list[TaskSummary]:
        text = _read(path)
        kind, parsed = _route(path, text)
        return self._build_task_summaries(kind, parsed)

    def parse_input(self, path: str) -> dict[str, Any]:
        # GRASP "inputs" are stdin heredocs that get embedded in a shell
        # script. There's no standalone input grammar to parse.
        raise NotImplementedError(
            "GRASP doesn't have a single input-file format. Use the "
            "input-builders in chemtools.programs.grasp.input.heredoc "
            "to construct stdin for individual exes."
        )

    # --- Drill-down sub-protocols ------------------------------------------

    def get_orbitals(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        """Orbital eigenvalues live in the rmcdhf .sum file."""
        text = _read(path)
        parsed = parse_sum(text)
        return {
            "source": path,
            "subshells": parsed.get("subshells", []),
            "n_subshells": len(parsed.get("subshells", [])),
        }

    def get_frequency(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        raise NotImplementedError("GRASP is an atomic structure code — no vibrational frequencies.")

    def get_trajectory(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        raise NotImplementedError("GRASP is an atomic structure code — no geometry trajectory.")

    def get_thermochem(self, path: str, task_index: int | None = None) -> dict[str, Any]:
        raise NotImplementedError("GRASP is an atomic structure code — no thermochem.")

    # --- Internal helpers --------------------------------------------------

    def _to_parsed_run(self, path: str, text: str, kind: str,
                       parsed: dict[str, Any]) -> ParsedRun:
        file_size = len(text)
        tasks = self._build_task_summaries(kind, parsed)
        derived: dict[str, Any] = {"grasp:file_kind": kind}

        if kind == "rmcdhf_summary":
            if parsed.get("ground_energy_au") is not None:
                derived["final_energy_hartree"] = parsed["ground_energy_au"]
            if parsed.get("speed_of_light_au") is not None:
                derived["grasp:speed_of_light_au"] = parsed["speed_of_light_au"]
                derived["grasp:is_nonrel_limit"] = bool(parsed.get("is_nonrel_limit"))
            if parsed.get("atomic_number") is not None:
                derived["grasp:atomic_number"] = parsed["atomic_number"]
            if parsed.get("n_subshells") is not None:
                derived["grasp:n_subshells"] = parsed["n_subshells"]
            if parsed.get("subshells"):
                derived["grasp:subshells"] = [s["label"] for s in parsed["subshells"]]
        elif kind == "rlevels":
            if parsed.get("ground_state_au") is not None:
                derived["final_energy_hartree"] = parsed["ground_state_au"]
            if parsed.get("n_levels") is not None:
                derived["grasp:n_levels"] = parsed["n_levels"]
            if parsed.get("max_splitting_cm1") is not None:
                derived["grasp:max_splitting_cm1"] = parsed["max_splitting_cm1"]
        elif kind == "lsj_label":
            if parsed.get("n_levels") is not None:
                derived["grasp:n_lsj_levels"] = parsed["n_levels"]
        elif kind == "rmcdhf_log":
            if parsed.get("final_energy_hartree") is not None:
                derived["final_energy_hartree"] = parsed["final_energy_hartree"]
            derived["grasp:n_scf_iterations"] = parsed.get("n_iterations", 0)
            derived["grasp:scf_converged"] = parsed.get("converged", False)

        return ParsedRun(
            program="grasp",
            program_version="2018",
            file=str(Path(path).resolve()),
            file_size_bytes=file_size,
            tasks=tasks,
            primary_task_index=0 if tasks else None,
            derived=derived,
            diagnostics=[],
            diagnosis={"overall": "ok", "next_actions": []},
        )

    def _build_task_summaries(self, kind: str, parsed: dict[str, Any]) -> list[TaskSummary]:
        """GRASP doesn't have multi-task outputs the way NWChem does. We
        emit a single synthetic task summary so downstream code can still
        treat GRASP outputs uniformly."""
        if kind == "rmcdhf_summary":
            return [TaskSummary(
                index=0,
                kind="scf",
                name="rmcdhf summary",
                method="MCDHF",
                basis=None,
                energy_hartree=parsed.get("ground_energy_au"),
                line_range=(1, 0),  # unknown
                outcome="completed",
                has_usable_data=parsed.get("ground_energy_au") is not None,
                selection_priority=1,
            )]
        if kind == "rlevels":
            return [TaskSummary(
                index=0,
                kind="other",
                name="rlevels energy table",
                method="MCDHF",
                basis=None,
                energy_hartree=parsed.get("ground_state_au"),
                line_range=(1, 0),
                outcome="completed",
                has_usable_data=bool(parsed.get("levels")),
                selection_priority=1,
            )]
        if kind == "lsj_label":
            return [TaskSummary(
                index=0,
                kind="other",
                name="jj2lsj LSJ-coupled compositions",
                method="MCDHF",
                basis=None,
                energy_hartree=None,
                line_range=(1, 0),
                outcome="completed",
                has_usable_data=bool(parsed.get("levels")),
                selection_priority=0,
            )]
        if kind == "rmcdhf_log":
            return [TaskSummary(
                index=0,
                kind="scf",
                name="rmcdhf SCF trace",
                method="MCDHF",
                basis=None,
                energy_hartree=parsed.get("final_energy_hartree"),
                line_range=(1, 0),
                outcome="completed" if parsed.get("converged") else "failed",
                has_usable_data=bool(parsed.get("iterations")),
                selection_priority=1,
            )]
        return []


GRASP_PARSER = _GraspParser()


# --- Routing logic --------------------------------------------------------

def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _route(path: str, text: str) -> tuple[str, dict[str, Any]]:
    """Pick the right per-file-type parser based on extension + content.

    Returns ``(kind, parsed_dict)`` where ``kind`` is one of:
      ``rmcdhf_summary``, ``lsj_label``, ``rlevels``, ``rmcdhf_log``.
    """
    p = Path(path)
    name = p.name
    suffix = p.suffix

    # Multi-suffix files (.lsj.lbl) — check by full name
    if name.endswith(".lsj.lbl"):
        return "lsj_label", parse_lsjlbl(text)

    if suffix == ".sum":
        return "rmcdhf_summary", parse_sum(text)

    # rlevels stdout typically captured as .out or piped through tee.
    # Detect by the table header.
    if "No Pos  J Parity" in text or "Energy levels for ..." in text:
        return "rlevels", parse_rlevels(text)

    # rmcdhf .log files have "Iteration number" sections if it's the SCF
    # trace; otherwise it's just the input heredoc copy.
    if "Iteration number" in text:
        return "rmcdhf_log", parse_rmcdhf_log(text)

    # Fallback: treat as rmcdhf summary (best effort).
    return "rmcdhf_summary", parse_sum(text)

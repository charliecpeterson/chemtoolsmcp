"""GRASP Parser sub-protocol implementation.

Adapts the file-specific GRASP parsers (RMCDHF/RCI summaries, properties,
level tables, labels, and SCF logs) into the chemtools Parser protocol so generic tools
like ``parse_output`` / ``summarize_output`` can route to them.

Because GRASP doesn't have a single canonical "output" file, this parser
dispatches by file extension:

  *.sum / *.csum  → RMCDHF or RCI summary (orbitals + final energy)
  *.lsj.lbl       → LSJ-coupled compositions
  *.(c)h(lsj)     → hyperfine constants
  *.(c)i          → isotope-shift factors
  *.(c)t.lsj      → radiative-transition properties
  *.log           → if it contains "Iteration number" sections, treat as
                    rmcdhf SCF iteration trace; otherwise just the rmcdhf
                    input-log copy (no useful structured data)
  rlevels stdout  → energy-level table (caller passes the captured stdout
                    via the ``contents`` arg of parse_output)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from chemtools.core.types import ParsedRun, TaskSummary
from chemtools.programs.grasp.parse.hfs import parse_hfs
from chemtools.programs.grasp.parse.lsjlbl import parse_lsjlbl
from chemtools.programs.grasp.parse.rci_log import parse_rci_log
from chemtools.programs.grasp.parse.ris import parse_ris
from chemtools.programs.grasp.parse.rlevels import parse_rlevels
from chemtools.programs.grasp.parse.rmcdhf_log import parse_rmcdhf_log
from chemtools.programs.grasp.parse.sum_file import parse_sum, rci_hamiltonian
from chemtools.programs.grasp.parse.transition import parse_transition


class _GraspParser:
    """File-type-dispatching parser for GRASP2018 artifacts."""

    def parse_output(self, path: str) -> ParsedRun:
        text = _read(path)
        kind, parsed = _route(path, text)
        return self._to_parsed_run(path, text, kind, parsed)

    def task_index(self, path: str) -> list[TaskSummary]:
        text = _read(path)
        kind, parsed = _route(path, text)
        return self._build_task_summaries(
            kind,
            parsed,
            line_count=max(len(text.splitlines()), 1),
        )

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
        line_count = max(len(text.splitlines()), 1)
        tasks = self._build_task_summaries(
            kind,
            parsed,
            line_count=line_count,
        )
        derived: dict[str, Any] = {"grasp:file_kind": kind}

        if kind in {"rmcdhf_summary", "rci_summary"}:
            if parsed.get("ground_energy_au") is not None:
                derived["final_energy_hartree"] = parsed["ground_energy_au"]
            for key in ("n_electrons", "n_csfs", "n_subshells"):
                if parsed.get(key) is not None:
                    derived[f"grasp:{key}"] = parsed[key]
            if parsed.get("speed_of_light_au") is not None:
                derived["grasp:speed_of_light_au"] = parsed["speed_of_light_au"]
                derived["grasp:is_nonrel_limit"] = bool(parsed.get("is_nonrel_limit"))
            if parsed.get("atomic_number") is not None:
                derived["grasp:atomic_number"] = parsed["atomic_number"]
            if parsed.get("n_subshells") is not None:
                derived["grasp:n_subshells"] = parsed["n_subshells"]
            if parsed.get("subshells"):
                derived["grasp:subshells"] = [s["label"] for s in parsed["subshells"]]
            if parsed.get("eigenenergies"):
                derived["grasp:levels"] = [
                    {
                        "level": level["level"],
                        "j": level["j_str"],
                        "parity": level["parity"],
                        "energy_hartree": level["energy_hartree"],
                    }
                    for level in parsed["eigenenergies"]
                ]
            for key in (
                "nucleus",
                "radial_grid",
                "optimization_mode",
                "eol_n_levels_optimized",
                "ol_level_optimized",
                "max_orbital_self_consistency",
            ):
                if parsed.get(key) is not None:
                    derived[f"grasp:{key}"] = parsed[key]
            if kind == "rci_summary" and parsed.get("rci_corrections"):
                derived["grasp:rci_corrections"] = parsed["rci_corrections"]
                derived["grasp:hamiltonian"] = rci_hamiltonian(
                    parsed["rci_corrections"]
                )
            for key in (
                "posthoc_self_energy_table_printed",
                "posthoc_self_energy_applied_to_mixing",
            ):
                if parsed.get(key) is not None:
                    derived[f"grasp:{key}"] = parsed[key]
            comparison_signature = _comparison_signature(parsed)
            if comparison_signature:
                derived["comparison_signature"] = comparison_signature
            if kind == "rci_summary" and parsed.get("rci_corrections"):
                axes = {
                    "hamiltonian": derived["grasp:hamiltonian"],
                    **{
                        field: value
                        for field, value in parsed["rci_corrections"].items()
                        if field != "is_rci" and value is not None
                    },
                }
                if parsed.get("posthoc_self_energy_applied_to_mixing") is not None:
                    axes["self_energy_applied_to_mixing"] = parsed[
                        "posthoc_self_energy_applied_to_mixing"
                    ]
                derived["comparison_axes"] = axes
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
                derived["grasp:energy_source"] = parsed["final_energy_source"]
            derived["grasp:n_scf_iterations"] = parsed.get("n_iterations", 0)
            derived["grasp:scf_converged"] = parsed.get("converged", False)
            for key in ("n_csfs", "n_subshells"):
                if parsed.get(key) is not None:
                    derived[f"grasp:{key}"] = parsed[key]
            if parsed.get("energy_change") is not None:
                derived["grasp:final_scf_energy_change_hartree"] = parsed[
                    "energy_change"
                ]
            if parsed.get("final_weighted_energy_hartree") is not None:
                derived["grasp:final_weighted_energy_hartree"] = parsed[
                    "final_weighted_energy_hartree"
                ]
            for key in (
                "convergence_evidence",
                "max_final_orbital_self_consistency",
                "final_orbitals",
                "alternating_norm_orbitals",
                "growing_alternating_orbitals",
                "node_count_changes",
            ):
                if parsed.get(key) is not None:
                    derived[f"grasp:{key}"] = parsed[key]
        elif kind == "rci_log":
            for key in ("n_csfs", "n_subshells"):
                if parsed.get(key) is not None:
                    derived[f"grasp:{key}"] = parsed[key]
            derived["grasp:rci_completed"] = parsed.get(
                "completed",
                False,
            )
            derived["grasp:transverse_integrals_computed"] = parsed.get(
                "transverse_integrals_computed",
                False,
            )
            derived["grasp:breit_integrals"] = parsed.get(
                "breit_integrals",
                {},
            )
            derived["grasp:qed_interpolation_warning_count"] = parsed.get(
                "qed_interpolation_warning_count",
                0,
            )
            derived["grasp:energy_source_required"] = parsed["energy_source_required"]
        elif kind == "hfs":
            derived["grasp:n_hfs_levels"] = parsed.get("n_levels", 0)
            for key in (
                "nuclear_spin",
                "dipole_moment_nm",
                "quadrupole_moment_barn",
            ):
                if parsed.get(key) is not None:
                    derived[f"grasp:{key}"] = parsed[key]
        elif kind == "isotope_shift":
            derived["grasp:n_isotope_shift_levels"] = parsed.get("n_levels", 0)
        elif kind == "transition":
            transitions = parsed.get("transitions", [])
            derived["grasp:n_transitions"] = parsed.get("n_transitions", 0)
            disagreements = [
                item.get("length_gauge", {}).get("dt")
                for item in transitions
                if item.get("length_gauge", {}).get("dt") is not None
            ]
            if disagreements:
                derived["grasp:max_gauge_disagreement"] = max(disagreements)

        return ParsedRun(
            program="grasp",
            program_version="2018",
            file=str(Path(path).resolve()),
            file_size_bytes=file_size,
            tasks=tasks,
            primary_task_index=0 if tasks else None,
            derived=derived,
            diagnostics=[],
            diagnosis=_build_diagnosis(kind, parsed),
        )

    def _build_task_summaries(
        self,
        kind: str,
        parsed: dict[str, Any],
        *,
        line_count: int,
    ) -> list[TaskSummary]:
        """GRASP doesn't have multi-task outputs the way NWChem does. We
        emit a single synthetic task summary so downstream code can still
        treat GRASP outputs uniformly."""
        if kind == "rmcdhf_summary":
            energy = parsed.get("ground_energy_au")
            return [TaskSummary(
                index=0,
                kind="energy",
                name="rmcdhf summary",
                method="MCDHF",
                basis=None,
                energy_hartree=energy,
                line_range=(1, line_count),
                outcome="success" if energy is not None else "incomplete",
                has_usable_data=energy is not None,
                selection_priority=1,
            )]
        if kind == "rci_summary":
            energy = parsed.get("ground_energy_au")
            return [TaskSummary(
                index=0,
                kind="energy",
                name="rci summary",
                method="RCI",
                basis=None,
                energy_hartree=energy,
                line_range=(1, line_count),
                outcome="success" if energy is not None else "incomplete",
                has_usable_data=energy is not None,
                selection_priority=1,
            )]
        if kind == "rlevels":
            return [TaskSummary(
                index=0,
                kind="energy",
                name="rlevels energy table",
                method="MCDHF",
                basis=None,
                energy_hartree=parsed.get("ground_state_au"),
                line_range=(1, line_count),
                outcome="success",
                has_usable_data=bool(parsed.get("levels")),
                selection_priority=1,
            )]
        if kind == "lsj_label":
            return [TaskSummary(
                index=0,
                kind="property",
                name="jj2lsj LSJ-coupled compositions",
                method="MCDHF",
                basis=None,
                energy_hartree=None,
                line_range=(1, line_count),
                outcome="success",
                has_usable_data=bool(parsed.get("levels")),
                selection_priority=0,
            )]
        if kind == "rmcdhf_log":
            return [TaskSummary(
                index=0,
                kind="energy",
                name="rmcdhf SCF trace",
                method="MCDHF",
                basis=None,
                energy_hartree=parsed.get("final_energy_hartree"),
                line_range=(1, line_count),
                outcome="success" if parsed.get("converged") else "failed",
                has_usable_data=bool(parsed.get("iterations")),
                selection_priority=1,
            )]
        if kind == "rci_log":
            return [TaskSummary(
                index=0,
                kind="energy",
                name="rci execution log",
                method="RCI",
                basis=None,
                energy_hartree=None,
                line_range=(1, line_count),
                outcome=("success" if parsed.get("completed") else "failed"),
                has_usable_data=bool(
                    parsed.get("blocks") or parsed.get("n_subshells")
                ),
                selection_priority=1,
            )]
        property_tasks = {
            "hfs": (
                "hyperfine structure",
                "MCDHF/HFS",
                bool(parsed.get("levels")),
            ),
            "isotope_shift": (
                "isotope-shift factors",
                "MCDHF/RIS",
                bool(parsed.get("levels")),
            ),
            "transition": (
                "radiative transitions",
                "MCDHF/RTRANSITION",
                bool(parsed.get("transitions")),
            ),
        }
        if kind in property_tasks:
            name, method, has_usable_data = property_tasks[kind]
            return [TaskSummary(
                index=0,
                kind="property",
                name=name,
                method=method,
                basis=None,
                energy_hartree=None,
                line_range=(1, line_count),
                outcome="success",
                has_usable_data=has_usable_data,
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
      ``rmcdhf_summary``, ``rci_summary``, ``hfs``, ``isotope_shift``,
      ``transition``, ``lsj_label``, ``rlevels``, ``rmcdhf_log``,
      ``rci_log``, or ``unknown``.
    """
    p = Path(path)
    name = p.name
    suffix = p.suffix

    # Multi-suffix files (.lsj.lbl) — check by full name
    if name.endswith(".lsj.lbl"):
        return "lsj_label", parse_lsjlbl(text)

    if suffix == ".csum":
        return "rci_summary", parse_sum(text)

    if suffix == ".sum":
        return "rmcdhf_summary", parse_sum(text)

    if "Nuclear spin" in text and (
        "A(MHz)" in text or "A (MHz)" in text
    ):
        return "hfs", parse_hfs(text)

    if (
        "Normal mass shift parameter" in text
        and "Specific mass shift parameter" in text
    ):
        return "isotope_shift", parse_ris(text)

    if "ANGS(VAC)" in text and "AKI =" in text:
        return "transition", parse_transition(text)

    # rlevels stdout typically captured as .out or piped through tee.
    # Detect by the table header.
    if "No Pos  J Parity" in text or "Energy levels for ..." in text:
        return "rlevels", parse_rlevels(text)

    # rmcdhf .log files have "Iteration number" sections if it's the SCF
    # trace; otherwise it's just the input heredoc copy.
    if "Iteration number" in text:
        return "rmcdhf_log", parse_rmcdhf_log(text)

    if re.search(r"^\s*RCI\s*$", text, re.M):
        return "rci_log", parse_rci_log(text)

    return "unknown", {}


def _comparison_signature(parsed: dict[str, Any]) -> dict[str, Any]:
    signature = {}
    for key in (
        "atomic_number",
        "n_electrons",
        "n_csfs",
        "n_subshells",
        "speed_of_light_au",
        "nucleus",
        "radial_grid",
    ):
        if parsed.get(key) is not None:
            signature[key] = parsed[key]
    if parsed.get("subshells"):
        signature["subshells"] = [subshell["label"] for subshell in parsed["subshells"]]
    if parsed.get("eigenenergies"):
        signature["state_sectors"] = [
            {
                "level": level["level"],
                "j": level["j_str"],
                "parity": level["parity"],
            }
            for level in parsed["eigenenergies"]
        ]
    return signature


def _build_diagnosis(
    kind: str,
    parsed: dict[str, Any],
) -> dict[str, Any]:
    if kind in {"rmcdhf_summary", "rci_summary"}:
        energy = parsed.get("ground_energy_au")
        return {
            "verdict": {
                "label": "completed" if energy is not None else "incomplete",
                "confidence": 0.95 if energy is not None else 0.8,
                "reasons": [
                    (
                        "The GRASP summary contains a final eigenenergy."
                        if energy is not None
                        else "The GRASP summary contains no eigenenergy table."
                    )
                ],
            },
            "next_actions": [],
            "anchors": [],
        }
    if kind == "rmcdhf_log":
        converged = bool(parsed.get("converged"))
        unstable_orbitals = parsed.get("growing_alternating_orbitals") or []
        max_residual = parsed.get("max_final_orbital_self_consistency")
        reasons = []
        if not converged:
            reasons.append("RMCDHF lacks positive convergence evidence.")
        elif unstable_orbitals:
            reasons.append(
                "RMCDHF completed, but the final orbital trace has growing "
                "sign-alternating updates for: "
                + ", ".join(unstable_orbitals)
                + "."
            )
        else:
            reasons.append(
                "RMCDHF completed without a terminal failure marker."
            )
        if max_residual is not None:
            reasons.append(
                "The largest final printed orbital self-consistency value is "
                f"{max_residual:.3e}."
            )
        if parsed.get("convergence_evidence") == (
            "execution_complete_without_reported_stopping_test"
        ):
            reasons.append(
                "The stdout completion line does not identify which RMCDHF "
                "stopping test passed."
            )
        next_actions = [
            {
                "action": "inspect_grasp_summary",
                "artifact_kind": "grasp.rmcdhf_summary",
                "reason": (
                    "Use the saved .sum file for the authoritative energy, "
                    "CSF count, nuclear model, and radial-grid settings."
                ),
                "priority": 1,
            }
        ]
        if unstable_orbitals:
            next_actions.insert(0, {
                "action": "stage_grasp_orbital_recovery",
                "orbitals": unstable_orbitals,
                "reason": (
                    "Restart from the last accepted wavefunction, localize "
                    "the triggering CSF or orbital layer, optimize the new "
                    "orbital separately, then release the intended set."
                ),
                "priority": 1,
            })
        return {
            "verdict": {
                "label": (
                    "completed_with_unstable_orbital_trace"
                    if converged and unstable_orbitals
                    else ("converged" if converged else "failed")
                ),
                "confidence": 0.9,
                "reasons": reasons,
            },
            "next_actions": next_actions,
            "anchors": [],
        }
    if kind == "rci_log":
        completed = bool(parsed.get("completed"))
        return {
            "verdict": {
                "label": "completed" if completed else "failed",
                "confidence": 0.9,
                "reasons": [
                    (
                        "RCI completed, but stdout does not contain the "
                        "final eigenenergy."
                        if completed
                        else "RCI lacks a clean completion marker."
                    )
                ],
            },
            "next_actions": [
                {
                    "action": "inspect_grasp_summary",
                    "artifact_kind": "grasp.rci_summary",
                    "reason": (
                        "Use the saved .csum file for the RCI energy and the "
                        "Dirac-Coulomb, Breit, and QED Hamiltonian flags."
                    ),
                    "priority": 1,
                }
            ],
            "anchors": [],
        }
    return {
        "verdict": {
            "label": "incomplete",
            "confidence": 0.8,
            "reasons": ["The artifact is not a recognized GRASP result."],
        },
        "next_actions": [],
        "anchors": [],
    }

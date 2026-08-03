"""Collect electronic-state and geometry evidence inside NWChem tasks."""

from __future__ import annotations

import re
from typing import Any, Mapping

from chemtools.programs.nwchem.electron_consistency import (
    normalize_wavefunction_class,
)
from chemtools.programs.nwchem.input.basis_library import (
    normalize_element_symbol,
)
from chemtools.programs.nwchem.parse.geometry import OutputGeometryScanner
from chemtools.programs.nwchem.xc_consistency import canonical_xc_alias


_CHARGE_RE = re.compile(
    r"^\s*Charge\s*:\s*([+-]?\d+)\s*$",
    re.IGNORECASE,
)
_MULTIPLICITY_RE = re.compile(
    r"^\s*Spin multiplicity\s*:\s*(\d+)\s*$",
    re.IGNORECASE,
)
_ATOM_COUNT_RE = re.compile(
    r"^\s*No\.\s+of atoms\s*:\s*(\d+)\s*$",
    re.IGNORECASE,
)
_ELECTRON_COUNT_RE = re.compile(
    r"^\s*No\.\s+of electrons\s*:\s*(\d+)\s*$",
    re.IGNORECASE,
)
_ALPHA_ELECTRONS_RE = re.compile(
    r"^\s*Alpha electrons\s*:\s*(\d+)\s*$",
    re.IGNORECASE,
)
_BETA_ELECTRONS_RE = re.compile(
    r"^\s*Beta electrons\s*:\s*(\d+)\s*$",
    re.IGNORECASE,
)
_WAVEFUNCTION_RE = re.compile(
    r"^\s*Wavefunction(?:\s+type)?\s*[:=]\s*(.+?)\s*$",
    re.IGNORECASE,
)
_BASIS_MODE_RE = re.compile(
    r'^\s*Summary of\s+"ao basis"\s*->\s*"[^"]*"\s*'
    r"\((spherical|cartesian)\)\s*$",
    re.IGNORECASE,
)
_BASIS_SUMMARY_ROW_RE = re.compile(
    r"^\s*(\S+)\s+(.+?)\s+(\d+)\s+(\d+)\s+(\S+)\s*$"
)
_BASIS_FUNCTION_COUNT_RE = re.compile(
    r"^\s*AO basis - number of functions\s*:\s*(\d+)\s*$",
    re.IGNORECASE,
)
_BASIS_SHELL_COUNT_RE = re.compile(
    r"^\s*number of shells\s*:\s*(\d+)\s*$",
    re.IGNORECASE,
)
_ECP_REPLACEMENT_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s+\([^)]+\)\s+Replaces\s+"
    r"(\d+)\s+electrons\s*$",
    re.IGNORECASE,
)
_XC_FUNCTIONAL_RE = re.compile(
    r"^\s*(.+?)\s+Method\s+XC\s+(?:Functional|Potential)\s*$",
    re.IGNORECASE,
)


class OutputTaskStateScanner:
    def __init__(self, parsed_tasks: list[Mapping[str, Any]]) -> None:
        self._states = []
        self._current_index = 0
        self._task_finished = False
        self._active_basis_summary: dict[str, Any] | None = None
        self._basis_summary_body_seen = False
        for index, task in enumerate(parsed_tasks):
            self._states.append({
                "task_index": index,
                "method": task.get("method"),
                "operation": task.get("kind"),
                "charges": [],
                "multiplicities": [],
                "atom_counts": [],
                "electron_counts": [],
                "alpha_electrons": [],
                "beta_electrons": [],
                "wavefunction_labels": [],
                "wavefunction_classes": [],
                "xc_functional_labels": [],
                "xc_functional_names": [],
                "basis_modes": [],
                "basis_function_counts": [],
                "basis_shell_counts": [],
                "basis_summaries": [],
                "ecp_replacements": {},
                "geometry_scanner": OutputGeometryScanner(),
            })

    def feed(self, line: str) -> None:
        if not self._states:
            return
        if "NWChem Input Module" in line:
            self._active_basis_summary = None
            self._basis_summary_body_seen = False
            if (
                self._task_finished
                and self._current_index + 1 < len(self._states)
            ):
                self._current_index += 1
            self._task_finished = False
            return
        lowered = line.lower()
        if (
            "task" in lowered
            and "times" in lowered
            and ("cpu:" in lowered or "wall:" in lowered)
        ):
            self._task_finished = True
            self._active_basis_summary = None
            self._basis_summary_body_seen = False
            return
        if self._task_finished:
            return
        state = self._states[self._current_index]
        state["geometry_scanner"].feed(line)
        if match := _CHARGE_RE.match(line):
            state["charges"].append(int(match.group(1)))
        if match := _MULTIPLICITY_RE.match(line):
            state["multiplicities"].append(int(match.group(1)))
        if match := _ATOM_COUNT_RE.match(line):
            state["atom_counts"].append(int(match.group(1)))
        if match := _ELECTRON_COUNT_RE.match(line):
            state["electron_counts"].append(int(match.group(1)))
        if match := _ALPHA_ELECTRONS_RE.match(line):
            state["alpha_electrons"].append(int(match.group(1)))
        if match := _BETA_ELECTRONS_RE.match(line):
            state["beta_electrons"].append(int(match.group(1)))
        if match := _WAVEFUNCTION_RE.match(line):
            label = match.group(1).strip().rstrip(".")
            state["wavefunction_labels"].append(label)
            if reference_class := normalize_wavefunction_class(label):
                state["wavefunction_classes"].append(reference_class)
        if match := _XC_FUNCTIONAL_RE.match(line):
            label = match.group(1).strip()
            state["xc_functional_labels"].append(label)
            if name := canonical_xc_alias(label):
                state["xc_functional_names"].append(name)
        if match := _BASIS_MODE_RE.match(line):
            mode = match.group(1).lower()
            state["basis_modes"].append(mode)
            self._active_basis_summary = {"mode": mode, "rows": []}
            self._basis_summary_body_seen = False
        elif self._active_basis_summary is not None:
            if match := _BASIS_SUMMARY_ROW_RE.match(line):
                tag = match.group(1)
                try:
                    element = normalize_element_symbol(tag)
                except ValueError:
                    element = None
                if not self._active_basis_summary["rows"]:
                    state["basis_summaries"].append(
                        self._active_basis_summary
                    )
                self._active_basis_summary["rows"].append({
                    "tag": tag,
                    "element": element,
                    "description": match.group(2).strip(),
                    "shells": int(match.group(3)),
                    "functions": int(match.group(4)),
                    "types": match.group(5),
                })
            if line.strip():
                self._basis_summary_body_seen = True
            elif self._basis_summary_body_seen:
                self._active_basis_summary = None
                self._basis_summary_body_seen = False
        if match := _BASIS_FUNCTION_COUNT_RE.match(line):
            state["basis_function_counts"].append(int(match.group(1)))
        if match := _BASIS_SHELL_COUNT_RE.match(line):
            state["basis_shell_counts"].append(int(match.group(1)))
        if match := _ECP_REPLACEMENT_RE.match(line):
            try:
                element = normalize_element_symbol(match.group(1))
            except ValueError:
                return
            replacements = state["ecp_replacements"].setdefault(element, [])
            replacements.append(int(match.group(2)))

    def finish(self) -> list[dict[str, Any]]:
        finished = []
        for state in self._states:
            scanner = state["geometry_scanner"]
            finished.append({
                "task_index": state["task_index"],
                "method": state["method"],
                "operation": state["operation"],
                "charges": _unique(state["charges"]),
                "multiplicities": _unique(state["multiplicities"]),
                "atom_counts": _unique(state["atom_counts"]),
                "electron_counts": _unique(state["electron_counts"]),
                "alpha_electrons": _unique(state["alpha_electrons"]),
                "beta_electrons": _unique(state["beta_electrons"]),
                "wavefunction_labels": _unique(
                    state["wavefunction_labels"]
                ),
                "wavefunction_classes": _unique(
                    state["wavefunction_classes"]
                ),
                "xc_functional_labels": _unique(
                    state["xc_functional_labels"]
                ),
                "xc_functional_names": _unique(
                    state["xc_functional_names"]
                ),
                "basis_modes": _unique(state["basis_modes"]),
                "basis_function_counts": _unique(
                    state["basis_function_counts"]
                ),
                "basis_shell_counts": _unique(
                    state["basis_shell_counts"]
                ),
                "basis_summaries": _unique_basis_summaries(
                    state["basis_summaries"]
                ),
                "ecp_replacements": {
                    element: _unique(replacements)
                    for element, replacements in state[
                        "ecp_replacements"
                    ].items()
                },
                "first_geometry": scanner.first_geometry,
                "last_geometry": scanner.last_geometry,
                "first_geometry_by_name": scanner.first_by_name,
                "last_geometry_by_name": scanner.last_by_name,
            })
        return finished


def _unique(values: list[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def _unique_basis_summaries(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique = []
    signatures = set()
    for summary in summaries:
        signature = (
            summary["mode"],
            tuple(
                (
                    row["tag"],
                    row["description"],
                    row["shells"],
                    row["functions"],
                    row["types"],
                )
                for row in summary["rows"]
            ),
        )
        if signature not in signatures:
            signatures.add(signature)
            unique.append(summary)
    return unique


__all__ = ["OutputTaskStateScanner"]

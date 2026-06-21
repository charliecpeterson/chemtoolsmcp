"""Parse GRASP hyperfine-structure output from ``rhfs`` / ``rhfs_lsj``.

Two related formats, both handled here:

* ``rhfs``     -> ``name.(c)h``    : nuclear-moment header + a table of diagonal
  interaction constants (Level / J / Parity / A(MHz) / B(MHz) / g_J / dg_J / total g_J).
* ``rhfs_lsj`` -> ``name.(c)hlsj`` : the same A/B/g_J with an LSJ label + level
  energy per row, energy-sortable.

A(MHz) is the magnetic-dipole hyperfine constant, B(MHz) the electric-quadrupole
one; both vanish when the nuclear spin is 0.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_FLOAT = r"-?\d+\.\d+(?:[EeDd][+-]?\d+)?"

_SPIN_RE = re.compile(r"Nuclear spin\s+(" + _FLOAT + r")")
_DIPOLE_RE = re.compile(r"Nuclear magnetic dipole moment\s+(" + _FLOAT + r")")
_QUAD_RE = re.compile(r"Nuclear electric quadrupole moment\s+(" + _FLOAT + r")")

# rhfs .h row: Level J Parity  A  B  g_J  delta_g_J  total_g_J
_H_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\S+)\s+([+-])\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+"
    r"(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s*$",
    re.M,
)
# rhfs_lsj .chlsj row: Energy State_label J P  A  B  gJ
_LSJ_ROW_RE = re.compile(
    r"^\s*(" + _FLOAT + r")\s+(\S+)\s+(\S+)\s+([+-])\s+"
    r"(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s*$",
    re.M,
)


def parse_hfs(path_or_text: str) -> dict[str, Any]:
    text = _as_text(path_or_text)
    out: dict[str, Any] = {}

    if m := _SPIN_RE.search(text):
        out["nuclear_spin"] = _todouble(m.group(1))
    if m := _DIPOLE_RE.search(text):
        out["dipole_moment_nm"] = _todouble(m.group(1))
    if m := _QUAD_RE.search(text):
        out["quadrupole_moment_barn"] = _todouble(m.group(1))

    levels: list[dict[str, Any]] = []
    for m in _H_ROW_RE.finditer(text):
        levels.append({
            "level": int(m.group(1)),
            "j_str": m.group(2),
            "parity": m.group(3),
            "a_mhz": _todouble(m.group(4)),
            "b_mhz": _todouble(m.group(5)),
            "g_j": _todouble(m.group(6)),
            "delta_g_j": _todouble(m.group(7)),
            "total_g_j": _todouble(m.group(8)),
        })
    if not levels:  # fall back to the LSJ-labelled rhfs_lsj table
        for m in _LSJ_ROW_RE.finditer(text):
            levels.append({
                "energy_au": _todouble(m.group(1)),
                "label": m.group(2),
                "j_str": m.group(3),
                "parity": m.group(4),
                "a_mhz": _todouble(m.group(5)),
                "b_mhz": _todouble(m.group(6)),
                "g_j": _todouble(m.group(7)),
            })

    out["levels"] = levels
    out["n_levels"] = len(levels)
    if out.get("nuclear_spin") == 0.0 and levels:
        out["note"] = "nuclear spin is 0 — A and B vanish by construction"
    return out


def _as_text(path_or_text: str) -> str:
    if "\n" in path_or_text or not Path(path_or_text).exists():
        return path_or_text
    return Path(path_or_text).read_text(encoding="utf-8", errors="replace")


def _todouble(s: str) -> float:
    return float(s.replace("D", "E").replace("d", "e"))

"""Parse a GRASP ``<name>.sum`` / ``<name>.csum`` summary file.

Written by ``rmcdhf`` (``.sum``, after ``rsave``) or ``rci`` (``.csum``). The two
share a format; for ``.csum`` an extra ``rci_corrections`` block reports which
Hamiltonian terms (Breit, vacuum polarisation, self-energy, mass shifts) were
added on top of Dirac-Coulomb.

Contents:
  * Number of electrons / CSFs / subshells
  * Atomic number + nucleus parameters
  * Speed of light (137.036 default; 2000 for non-rel limit)
  * Radial grid parameters
  * EOL calculation level indices + weights
  * Radial wavefunction summary (per-subshell eigenvalue + radial moments)
  * Eigenenergies per (block, level)
  * Major CSF contributors per ASF (weights)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_FLOAT_RE = r"-?\d+\.\d+(?:[EeDd][+-]?\d+)?"

_HEADER_RE = re.compile(
    r"There are\s+(\d+)\s+electrons in the cloud\s+"
    r"in\s+(\d+)\s+relativistic CSFs\s+"
    r"based on\s+(\d+)\s+relativistic subshells", re.DOTALL,
)
_Z_RE = re.compile(r"The atomic number is\s+(" + _FLOAT_RE + r")")
_MASS_RE = re.compile(r"the mass of the nucleus is\s+(" + _FLOAT_RE + r")")
_SPEED_OF_LIGHT_RE = re.compile(r"Speed of light\s*=\s*(" + _FLOAT_RE + r")")
_RNT_RE = re.compile(r"RNT\s*=\s*(" + _FLOAT_RE + r")")
_H_RE = re.compile(r"\bH\s*=\s*(" + _FLOAT_RE + r")")
_N_GRID_RE = re.compile(r"^\s*N\s*=\s*(\d+)", re.M)
_EOL_RE = re.compile(r"EOL calculation\.?\s+(\d+)\s+levels will be optimised", re.M)

# Radial wfn summary line: e.g. "  1s   6.9000143068D+01  1.054D+02  1.00  ..."
_SUBSHELL_RE = re.compile(
    r"^\s+([1-9][0-9]?[spdfghi][-+]?)\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+"
    r"(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+"
    r"(" + _FLOAT_RE + r")\s+(\d+)\s*$",
    re.M,
)

# Eigenenergies block:
#  Level J Parity      Hartrees           Kaysers           eV
#    1  1/2  -      -7.40457700D+00   -1.625116808D+06   -2.01D+02
_EIGEN_RE = re.compile(
    r"^\s*(\d+)\s+(\S+)\s+([+-])\s+(" + _FLOAT_RE + r")\s+"
    r"(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s*$",
    re.M,
)


def parse_sum(path_or_text: str) -> dict[str, Any]:
    """Parse a .sum file. Accepts either a path or raw text."""
    text = _as_text(path_or_text)

    out: dict[str, Any] = {}

    if m := _HEADER_RE.search(text):
        out["n_electrons"] = int(m.group(1))
        out["n_csfs"] = int(m.group(2))
        out["n_subshells"] = int(m.group(3))

    if m := _Z_RE.search(text):
        out["atomic_number"] = float(m.group(1))
    if m := _MASS_RE.search(text):
        out["nuclear_mass_electron_units"] = _todouble(m.group(1))
    if m := _SPEED_OF_LIGHT_RE.search(text):
        out["speed_of_light_au"] = _todouble(m.group(1))
        out["is_nonrel_limit"] = out["speed_of_light_au"] > 500.0

    grid: dict[str, Any] = {}
    if m := _RNT_RE.search(text):
        grid["RNT"] = _todouble(m.group(1))
    if m := _H_RE.search(text):
        grid["H"] = _todouble(m.group(1))
    if m := _N_GRID_RE.search(text):
        grid["N"] = int(m.group(1))
    if grid:
        out["radial_grid"] = grid

    if m := _EOL_RE.search(text):
        out["eol_n_levels_optimized"] = int(m.group(1))

    # Per-subshell radial wfn summary
    subshells: list[dict[str, Any]] = []
    for m in _SUBSHELL_RE.finditer(text):
        subshells.append({
            "label": m.group(1),
            "eigenvalue_au": _todouble(m.group(2)),
            "p0": _todouble(m.group(3)),
            "gamma": _todouble(m.group(4)),
            "p2": _todouble(m.group(5)),
            "q2": _todouble(m.group(6)),
            "self_consistency": _todouble(m.group(7)),
            "mtp": int(m.group(8)),
        })
    if subshells:
        out["subshells"] = subshells

    # An RCI .csum writes one Eigenenergies/contributors pair per J/parity
    # block, followed by optional correction-only energy tables. Match only
    # sections terminated by the contributor header so every ASF block is
    # retained and the diagnostic correction tables stay excluded.
    eigen_sections: list[str] = []
    section_start = 0
    for contributors in re.finditer(
        r"Weights of major contributors to ASF:",
        text,
    ):
        header = text.rfind("Eigenenergies", section_start, contributors.start())
        if header >= 0:
            eigen_sections.append(text[header:contributors.start()])
        section_start = contributors.end()
    levels: list[dict[str, Any]] = []
    for eigen_text in eigen_sections:
        for m in _EIGEN_RE.finditer(eigen_text):
            levels.append({
                "level": int(m.group(1)),
                "j_str": m.group(2),
                "parity": m.group(3),
                "energy_hartree": _todouble(m.group(4)),
                "energy_cm1": _todouble(m.group(5)),
                "energy_ev": _todouble(m.group(6)),
            })
    if levels:
        out["eigenenergies"] = levels
        out["ground_energy_au"] = min(
            level["energy_hartree"] for level in levels
        )

    # rci writes a .csum in the same format but adds a Hamiltonian-decomposition
    # block listing which corrections sit on top of Dirac-Coulomb. Report it so
    # the caller knows whether Breit/QED were included.
    if "To H (Dirac Coulomb) is added" in text:
        transverse = "H (Transverse)" in text
        photon_factor = None
        if transverse:
            if fm := re.search(r"factor multiplying the photon frequency:\s*(" + _FLOAT_RE + r")", text):
                photon_factor = _todouble(fm.group(1))
        out["rci_corrections"] = {
            "is_rci": True,
            "transverse_breit": transverse,
            "photon_frequency_factor": photon_factor,
            "vacuum_polarisation": "H (Vacuum Polarisation)" in text,
            "self_energy": "H (Self Energy)" in text,
            "normal_mass_shift": "H (Normal Mass Shift)" in text,
            "specific_mass_shift": "H (Specific Mass Shift)" in text,
        }

    return out


def _as_text(path_or_text: str) -> str:
    if "\n" in path_or_text or not Path(path_or_text).exists():
        return path_or_text
    return Path(path_or_text).read_text(encoding="utf-8", errors="replace")


def _todouble(s: str) -> float:
    """Convert Fortran D-notation float to Python float."""
    return float(s.replace("D", "E").replace("d", "e"))

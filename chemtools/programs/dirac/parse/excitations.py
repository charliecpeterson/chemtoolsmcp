"""Parser for DIRAC TDDFT / response excitation energies.

Without this a TDDFT run reads as a plain SCF/DFT job — the excitation spectrum
(the point of the calculation) is never surfaced.
"""
from __future__ import annotations

import re
from typing import Any

# " Excitation no.   1 excitation energy     0.21796845 a.u.    2.89E-06 (converged)"
_EXC_RE = re.compile(
    r"Excitation no\.\s+(\d+)\s+excitation energy\s+([-\d.]+)\s+a\.u\.\s+\S+\s+\((\w+)\)",
    re.IGNORECASE,
)
# " Nonrel. sym.:   S0A1                        11.2888 eV"
_SYM_EV_RE = re.compile(r"Nonrel\. sym\.:\s+(\S+)\s+([-\d.]+)\s+eV", re.IGNORECASE)
_SUMF_RE = re.compile(r"Sum of oscillator strengths[^:]*:\s+([-\d.E+]+)", re.IGNORECASE)

_HARTREE_TO_EV = 27.211386245988


def parse_excitations(contents: str) -> dict[str, Any]:
    if "excitation energy" not in contents.lower():
        return {"available": False, "n_excitations": 0, "excitations": []}

    lines = contents.splitlines()
    excitations: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        match = _EXC_RE.search(line)
        if not match:
            continue
        exc: dict[str, Any] = {
            "number": int(match.group(1)),
            "excitation_energy_au": float(match.group(2)),
            "converged": match.group(3).lower() == "converged",
            "excitation_energy_ev": None,
            "symmetry": None,
        }
        # The symmetry label + eV sit on the following "Nonrel. sym." line — but
        # that line is not always printed (e.g. core-restricted runs). eV is
        # derived from a.u. directly so it is always available; the printed value
        # only supplies the symmetry label.
        if i + 1 < len(lines):
            sym = _SYM_EV_RE.search(lines[i + 1])
            if sym:
                exc["symmetry"] = sym.group(1)
        exc["excitation_energy_ev"] = round(exc["excitation_energy_au"] * _HARTREE_TO_EV, 4)
        excitations.append(exc)

    if not excitations:
        return {"available": False, "n_excitations": 0, "excitations": []}

    # De-duplicate by excitation number (DIRAC reprints per symmetry block).
    by_number = {e["number"]: e for e in excitations}
    final = [by_number[n] for n in sorted(by_number)]
    sum_f = _SUMF_RE.search(contents)
    lowest = min((e["excitation_energy_ev"] for e in final if e["excitation_energy_ev"] is not None),
                 default=None)
    return {
        "available": True,
        "n_excitations": len(final),
        "excitations": final,
        "lowest_excitation_ev": lowest,
        "sum_oscillator_strength": float(sum_f.group(1)) if sum_f else None,
    }

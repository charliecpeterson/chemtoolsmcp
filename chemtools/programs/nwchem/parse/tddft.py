"""Parser for the NWChem TDDFT module.

Extracts excited-state roots (spin, symmetry, excitation energy in a.u. and eV)
and their oscillator strengths so an excited-state run is recognized and its
spectrum surfaced — otherwise a TDDFT job reads as a plain single-point.
"""
from __future__ import annotations

import re
from typing import Any

from chemtools.core.common import make_metadata, read_text

TDDFT_MODULE_RE = re.compile(r"NWChem TDDFT Module", re.IGNORECASE)
# "  Root   1 singlet a              0.312652235 a.u.                8.5077 eV"
ROOT_RE = re.compile(
    r"^\s*Root\s+(\d+)\s+(singlet|triplet|doublet)?\s*([a-zA-Z]\S*)?\s+"
    r"([-\d.]+)\s*a\.u\.\s+([-\d.]+)\s*eV",
    re.IGNORECASE,
)
OSC_RE = re.compile(r"Total Oscillator Strength\s+([-\d.]+)", re.IGNORECASE)

# Oscillator strength above this counts a state as optically "bright".
_BRIGHT_THRESHOLD = 0.01


def parse_tddft(path: str, contents: str | None = None) -> dict[str, Any]:
    contents = contents if contents is not None else read_text(path)
    if not TDDFT_MODULE_RE.search(contents):
        return {"available": False, "root_count": 0, "roots": []}

    roots: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in contents.splitlines():
        match = ROOT_RE.match(line)
        if match:
            current = {
                "root": int(match.group(1)),
                "spin": (match.group(2) or "").lower() or None,
                "symmetry": match.group(3),
                "excitation_energy_au": float(match.group(4)),
                "excitation_energy_ev": float(match.group(5)),
                "oscillator_strength": None,
            }
            roots.append(current)
            continue
        if current is not None:
            osc = OSC_RE.search(line)
            if osc:
                current["oscillator_strength"] = float(osc.group(1))
                current = None

    # A restart re-prints earlier roots; keep the last occurrence of each.
    by_root = {r["root"]: r for r in roots}
    final = [by_root[k] for k in sorted(by_root)]
    bright = [r for r in final if (r.get("oscillator_strength") or 0.0) > _BRIGHT_THRESHOLD]
    brightest = max(final, key=lambda r: r.get("oscillator_strength") or 0.0) if final else None
    return {
        "metadata": make_metadata(path, contents, "nwchem"),
        "available": True,
        "root_count": len(final),
        "roots": final,
        "lowest_excitation_ev": final[0]["excitation_energy_ev"] if final else None,
        "bright_state_count": len(bright),
        "brightest_state": brightest,
    }

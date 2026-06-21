"""Parse GRASP isotope-shift output (``name.i``) from ``ris4``.

ris4 computes the *electronic* isotope-shift factors (isotope-independent; the
actual shift between two isotopes combines these with nuclear masses and the
change in <r^2>):

* Normal mass shift (NMS) parameters  <K^1>, <K^2+K^3>, <K^1+K^2+K^3>
* Specific mass shift (SMS) parameters (same three operators)
* Electron density at the nucleus (DENS) — the first-order field-shift factor

Each parameter is reported per level in atomic units and (for the mass shifts)
in GHz u.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_FLOAT = r"-?\d+\.\d+(?:[EeDd][+-]?\d+)?"

# Energy row:  "   1        0 +        -0.2651014327D+05  (a.u.)"
_ENERGY_RE = re.compile(
    r"^\s*(\d+)\s+(\S+)\s+([+-])\s+(" + _FLOAT + r")\s*\(a\.u\.\)", re.M)
# Mass-shift a.u. row: "  1   0 +   <K1>   <K2+K3>   <K1+K2+K3>  (a.u.)"
_MS_RE = re.compile(
    r"^\s*(\d+)\s+(\S+)\s+([+-])\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+("
    + _FLOAT + r")\s*\(a\.u\.\)", re.M)
# Electron-density row: "   1   0 +   0.5194244460D+07"
_DENS_RE = re.compile(
    r"^\s*(\d+)\s+(\S+)\s+([+-])\s+(" + _FLOAT + r")\s*$", re.M)


def parse_ris(path_or_text: str) -> dict[str, Any]:
    text = _as_text(path_or_text)
    sections = _split_sections(text)

    levels = [
        {"level": int(m.group(1)), "j_str": m.group(2), "parity": m.group(3),
         "energy_au": _d(m.group(4))}
        for m in _ENERGY_RE.finditer(sections.get("energy", ""))
    ]

    def _mass_rows(key: str) -> list[dict[str, Any]]:
        return [
            {"level": int(m.group(1)), "j_str": m.group(2), "parity": m.group(3),
             "k1": _d(m.group(4)), "k2_k3": _d(m.group(5)), "k1_k2_k3": _d(m.group(6))}
            for m in _MS_RE.finditer(sections.get(key, ""))
        ]

    dens = [
        {"level": int(m.group(1)), "j_str": m.group(2), "parity": m.group(3),
         "density_au": _d(m.group(4))}
        for m in _DENS_RE.finditer(sections.get("density", ""))
    ]

    return {
        "n_levels": len(levels),
        "levels": levels,
        "normal_mass_shift": _mass_rows("nms"),
        "specific_mass_shift": _mass_rows("sms"),
        "electron_density": dens,
    }


def _split_sections(text: str) -> dict[str, str]:
    """Carve the .i file into its four labelled blocks."""
    markers = [
        ("energy", r"Level\s+J\s+Parity\s+Energy"),
        ("nms", r"Normal mass shift parameter"),
        ("sms", r"Specific mass shift parameter"),
        ("density", r"Electron density"),
    ]
    bounds = []
    for key, pat in markers:
        if m := re.search(pat, text):
            bounds.append((m.start(), key))
    bounds.sort()
    out: dict[str, str] = {}
    for i, (start, key) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else len(text)
        out[key] = text[start:end]
    return out


def _as_text(path_or_text: str) -> str:
    if "\n" in path_or_text or not Path(path_or_text).exists():
        return path_or_text
    return Path(path_or_text).read_text(encoding="utf-8", errors="replace")


def _d(s: str) -> float:
    return float(s.replace("D", "E").replace("d", "e"))

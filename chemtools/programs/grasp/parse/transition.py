"""Parse GRASP radiative-transition output (``name1.name2.(c)t.lsj``) from rtransition.

Each transition block:

    1   -7.43353309  1s(2).2s_2S                         <- lower state
    1   -7.36586156  1s(2).2p_2P                         <- upper state
    14852.18 CM-1   6733.02 ANGS(VAC)   6732.32 ANGS(AIR)
   E1  S = 1.13D+01  GF = 5.10D-01  AKI = 3.76D+07  dT = 0.034   <- length gauge
        1.17D+01        5.29D-01        3.89D+07                  <- velocity gauge

S = line strength, GF = weighted oscillator strength, AKI = transition rate
(s^-1), dT = length/velocity gauge disagreement (small is good).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_F = r"-?\d+\.\d+(?:[EeDd][+-]?\d+)?"

_BLOCK_RE = re.compile(
    r"^\s*(\d+)\s+(" + _F + r")\s+(\S+)\s*\n"          # lower: idx energy label
    r"\s*(\d+)\s+(" + _F + r")\s+(\S+)\s*\n"           # upper: idx energy label
    r"\s*(" + _F + r")\s+CM-1\s+(" + _F + r")\s+ANGS\(VAC\)\s+(" + _F + r")\s+ANGS\(AIR\)\s*\n"
    r"\s*(E1|E2|E3|M1|M2|M3)\s+S\s*=\s*(" + _F + r")\s+GF\s*=\s*(" + _F + r")\s+"
    r"AKI\s*=\s*(" + _F + r")\s+dT\s*=\s*(" + _F + r")\s*\n"
    r"\s*(" + _F + r")\s+(" + _F + r")\s+(" + _F + r")",  # velocity gauge: S GF AKI
    re.M,
)


def parse_transition(path_or_text: str) -> dict[str, Any]:
    text = _as_text(path_or_text)
    transitions: list[dict[str, Any]] = []
    for m in _BLOCK_RE.finditer(text):
        transitions.append({
            "lower": {"index": int(m.group(1)), "energy_au": _d(m.group(2)), "label": m.group(3)},
            "upper": {"index": int(m.group(4)), "energy_au": _d(m.group(5)), "label": m.group(6)},
            "energy_cm1": _d(m.group(7)),
            "wavelength_vac_ang": _d(m.group(8)),
            "wavelength_air_ang": _d(m.group(9)),
            "type": m.group(10),
            "length_gauge": {
                "line_strength": _d(m.group(11)),
                "gf": _d(m.group(12)),
                "a_ki_per_s": _d(m.group(13)),
                "dt": _d(m.group(14)),
            },
            "velocity_gauge": {
                "line_strength": _d(m.group(15)),
                "gf": _d(m.group(16)),
                "a_ki_per_s": _d(m.group(17)),
            },
        })
    return {"n_transitions": len(transitions), "transitions": transitions}


def _as_text(path_or_text: str) -> str:
    if "\n" in path_or_text or not Path(path_or_text).exists():
        return path_or_text
    return Path(path_or_text).read_text(encoding="utf-8", errors="replace")


def _d(s: str) -> float:
    return float(s.replace("D", "E").replace("d", "e"))

"""Parse a GRASP ``<name>.lsj.lbl`` file produced by ``jj2lsj``.

Each level block:

   1    0     +        -33058.834159311     100.000%
        -0.75697793    0.57301559   5f(10)3P2.7s(2)_3P
         0.43503420    0.18925475   5f(10)5D1.7s(2)_5D
         ...

Header: Pos / J / Parity / Energy_au / Total_percent.
Body: per LS-component { coefficient, weight_percent_fraction, label }.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_FLOAT_RE = r"-?\d+\.\d+"

_HEADER_RE = re.compile(
    r"^\s*(\d+)\s+(\S+)\s+([+-])\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")%\s*$",
    re.M,
)
_COMP_RE = re.compile(
    r"^\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(\S+)\s*$",
    re.M,
)


def parse_lsjlbl(path_or_text: str) -> dict[str, Any]:
    """Parse an lsj.lbl file. Returns levels with LSJ-coupled composition.

    Returns
    -------
    dict with::

        {
          "levels": [
            {
              "pos": int,
              "j_str": str,
              "j": float,
              "parity": "+"|"-",
              "energy_au": float,
              "total_weight_percent": float,
              "components": [{coefficient, weight_fraction, label}, ...]
            },
            ...
          ],
          "n_levels": int,
        }
    """
    text = _as_text(path_or_text)
    headers = list(_HEADER_RE.finditer(text))

    levels: list[dict[str, Any]] = []
    for i, h in enumerate(headers):
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[h.end():block_end]

        components: list[dict[str, Any]] = []
        for m in _COMP_RE.finditer(block):
            components.append({
                "coefficient": float(m.group(1)),
                "weight_fraction": float(m.group(2)),
                "label": m.group(3).strip(),
            })

        levels.append({
            "pos": int(h.group(1)),
            "j_str": h.group(2),
            "j": _normalize_j(h.group(2)),
            "parity": h.group(3),
            "energy_au": float(h.group(4)),
            "total_weight_percent": float(h.group(5)),
            "components": components,
            "dominant_label": components[0]["label"] if components else None,
            "dominant_weight": components[0]["weight_fraction"] if components else None,
        })

    return {"levels": levels, "n_levels": len(levels)}


def _as_text(path_or_text: str) -> str:
    if "\n" in path_or_text or not Path(path_or_text).exists():
        return path_or_text
    return Path(path_or_text).read_text(encoding="utf-8", errors="replace")


def _normalize_j(j_str: str) -> float:
    if "/" in j_str:
        num, den = j_str.split("/")
        return float(num) / float(den)
    return float(j_str)

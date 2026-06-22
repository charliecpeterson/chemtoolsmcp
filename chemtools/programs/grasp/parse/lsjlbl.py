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
# Label is optional: in heavy-element MR-SD expansions jj2lsj sometimes prints a
# component's coefficient + weight with no LSJ label (and the level's total weight
# drifts off 100%, signalling the jj->LSJ map is unreliable for that level). Keep
# the numeric data rather than dropping the whole component.
_COMP_RE = re.compile(
    r"^\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")(?:\s+(\S.*?))?\s*$",
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
              "components": [{coefficient, weight_fraction, label}, ...],
              "dominant_label": str, "dominant_weight": float,   # leading CSF
              "term_composition": {term: summed_weight, ...},    # by total LS term
              "dominant_term": str, "dominant_term_weight": float,  # LS-coupling purity
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
            label = m.group(3)
            components.append({
                "coefficient": float(m.group(1)),
                "weight_fraction": float(m.group(2)),
                "label": label.strip() if label else None,
            })

        dominant = max(components, key=lambda c: c["weight_fraction"]) if components else None

        # Aggregate the components by their *total* LS term (the suffix after the
        # last '_', e.g. ..._3H). With a correlation expansion the leading CSF can
        # be a small fraction even when the level is a near-pure LS term, because
        # the weight is spread over many configurations that all share that term.
        # Summing by term recovers the LS-coupling purity (the intermediate-
        # coupling measure): dominant_term ~ 1 means clean LS, << 1 means mixed.
        term_comp: dict[str, float] = {}
        for c in components:
            term = _total_term(c["label"])
            if term:
                term_comp[term] = term_comp.get(term, 0.0) + c["weight_fraction"]
        dom_term = max(term_comp.items(), key=lambda kv: kv[1]) if term_comp else None

        levels.append({
            "pos": int(h.group(1)),
            "j_str": h.group(2),
            "j": _normalize_j(h.group(2)),
            "parity": h.group(3),
            "energy_au": float(h.group(4)),
            "total_weight_percent": float(h.group(5)),
            "components": components,
            "dominant_label": dominant["label"] if dominant else None,
            "dominant_weight": dominant["weight_fraction"] if dominant else None,
            "term_composition": term_comp,
            "dominant_term": dom_term[0] if dom_term else None,
            "dominant_term_weight": dom_term[1] if dom_term else None,
        })

    return {"levels": levels, "n_levels": len(levels)}


def _total_term(label: str | None) -> str | None:
    """Extract the total LS term (e.g. '3H') from a CSF label's trailing '_<term>'."""
    if not label or "_" not in label:
        return None
    tail = label.rsplit("_", 1)[-1]
    m = re.match(r"\d[A-Za-z]", tail)
    return m.group(0) if m else None


def _as_text(path_or_text: str) -> str:
    if "\n" in path_or_text or not Path(path_or_text).exists():
        return path_or_text
    return Path(path_or_text).read_text(encoding="utf-8", errors="replace")


def _normalize_j(j_str: str) -> float:
    if "/" in j_str:
        num, den = j_str.split("/")
        return float(num) / float(den)
    return float(j_str)

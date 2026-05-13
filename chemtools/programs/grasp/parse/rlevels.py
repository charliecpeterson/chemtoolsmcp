"""Parse the ``rlevels`` output — atomic energy levels with splittings.

Format::

      No Pos  J Parity Energy Total    Levels     Splitting     Configuration
                        (a.u.)      (cm^-1)     (cm^-1)
    ------------------------------------------------------------------------------------------
       1  1   8  +  -33059.0063932        0.00        0.00  5f(10)5I1.7s(2)_5I
       2  1   7  +  -33058.9557785    11108.64    11108.64  5f(10)5I1.7s(2)_5I
       ...

J is either an integer (e.g. ``8``) or a half-integer string like ``1/2``,
``3/2``, ``5/2`` for systems with odd electron count. Parity is ``+`` or ``-``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_HEADER_RE = re.compile(
    r"^\s*No\s+Pos\s+J\s+Parity\s+Energy\s+Total", re.M
)
# NOTE: use [ \t]+ for inter-column whitespace, not \s+ — \s matches newline,
# which lets configurations bleed across lines when the cfg column is empty
# (e.g., when jj2lsj wasn't run with mixing_coefficients).
_LEVEL_RE = re.compile(
    r"^[ \t]*(\d+)[ \t]+(\d+)[ \t]+(\S+)[ \t]+([+-])[ \t]+"
    r"(-?\d+\.\d+)[ \t]+(-?\d+\.\d+)[ \t]+(-?\d+\.\d+)"
    r"(?:[ \t]+(.*?))?[ \t]*$",
    re.M,
)
_RYDBERG_RE = re.compile(r"Rydberg constant is\s+(-?[\d.]+)")


def parse_rlevels(text_or_path: str) -> dict[str, Any]:
    """Parse rlevels stdout (or a file containing it).

    Accepts either the raw output text or a path to a file/log containing it.
    Returns a dict with: levels (list of per-level dicts), n_levels,
    rydberg_constant, ground_state_au, max_splitting_cm1.
    """
    text = _as_text(text_or_path)
    levels: list[dict[str, Any]] = []
    header_m = _HEADER_RE.search(text)
    body = text[header_m.start():] if header_m else text

    for m in _LEVEL_RE.finditer(body):
        cfg = (m.group(8) or "").strip()
        levels.append({
            "no": int(m.group(1)),
            "pos": int(m.group(2)),
            "j": _normalize_j(m.group(3)),
            "j_str": m.group(3),
            "parity": m.group(4),
            "energy_hartree": float(m.group(5)),
            "energy_cm1": float(m.group(6)),
            "splitting_cm1": float(m.group(7)),
            "configuration": cfg,
        })

    rydberg_m = _RYDBERG_RE.search(text)
    rydberg = float(rydberg_m.group(1)) if rydberg_m else None
    ground = levels[0]["energy_hartree"] if levels else None
    max_split = max((lv["splitting_cm1"] for lv in levels), default=None)

    return {
        "levels": levels,
        "n_levels": len(levels),
        "rydberg_constant": rydberg,
        "ground_state_au": ground,
        "max_splitting_cm1": max_split,
    }


def summarize_terms(rlevels_result: dict[str, Any]) -> dict[str, Any]:
    """Group levels by their LSJ term label (the suffix after the underscore
    in the configuration column) and report term-level structure.

    Each term entry: ``{label, levels, n_levels, j_values, energy_min_cm1,
    energy_max_cm1, spread_cm1}`` (spread = splitting within the multiplet).
    """
    by_term: dict[str, list[dict[str, Any]]] = {}
    for lv in rlevels_result["levels"]:
        cfg = lv["configuration"]
        # Term label is the trailing _<TERM> token, e.g. ..._5I -> 5I
        m = re.search(r"_([A-Za-z0-9]+)\s*$", cfg)
        term = m.group(1) if m else cfg
        by_term.setdefault(term, []).append(lv)

    terms: list[dict[str, Any]] = []
    for term, group in by_term.items():
        e_min = min(lv["energy_cm1"] for lv in group)
        e_max = max(lv["energy_cm1"] for lv in group)
        terms.append({
            "term": term,
            "n_levels": len(group),
            "j_values": [lv["j_str"] for lv in group],
            "parities": sorted({lv["parity"] for lv in group}),
            "energy_min_cm1": e_min,
            "energy_max_cm1": e_max,
            "spread_cm1": e_max - e_min,
            "ground_level_no": min(group, key=lambda lv: lv["energy_cm1"])["no"],
        })
    terms.sort(key=lambda t: t["energy_min_cm1"])
    return {"n_terms": len(terms), "terms": terms}


def compare_rlevels(
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> dict[str, Any]:
    """Pairwise compare two rlevels parses by level index (no/pos).

    Useful for relativistic-vs-non-rel comparisons. Levels are matched by
    their position in the list (level 1 to level 1, etc.). Reports the
    cm-1 shift for each level and the total ground-state shift.
    """
    a = result_a["levels"]
    b = result_b["levels"]
    pairs: list[dict[str, Any]] = []
    for la, lb in zip(a, b):
        pairs.append({
            "no": la["no"],
            "j": la["j_str"],
            "parity": la["parity"],
            "configuration": la["configuration"],
            f"energy_cm1_{label_a}": la["energy_cm1"],
            f"energy_cm1_{label_b}": lb["energy_cm1"],
            "shift_cm1": lb["energy_cm1"] - la["energy_cm1"],
        })
    g_shift_au = (b[0]["energy_hartree"] - a[0]["energy_hartree"]) if a and b else None
    return {
        "n_matched": len(pairs),
        "ground_shift_au": g_shift_au,
        "ground_shift_cm1": g_shift_au * 219474.6313705 if g_shift_au is not None else None,
        "pairs": pairs,
        "label_a": label_a,
        "label_b": label_b,
    }


def _as_text(text_or_path: str) -> str:
    if "\n" in text_or_path or not Path(text_or_path).exists():
        return text_or_path
    return Path(text_or_path).read_text(encoding="utf-8", errors="replace")


def _normalize_j(j_str: str) -> float:
    """Convert J string to float. ``8`` -> 8.0; ``1/2`` -> 0.5; ``3/2`` -> 1.5."""
    if "/" in j_str:
        num, den = j_str.split("/")
        return float(num) / float(den)
    return float(j_str)

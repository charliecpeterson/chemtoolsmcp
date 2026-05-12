"""DIRAC VECPOP / Mulliken-per-MO parser.

When ``.MULPOP`` + ``.VECPOP`` are active in **ANALYZE, DIRAC prints a
Mulliken population analysis grouped by fermion irrep, with one block
per electronic MO containing:

  - Eigenvalue index (1-based within the irrep)
  - Eigenvalue energy (Hartree)
  - Occupation f
  - j-quantum-number label and m_j (e.g. ``s 1/2;  1/2``,  ``d 3/2; -3/2``,
    ``f 5/2; 5/2``, ``f 7/2; 7/2``)
  - Gross alpha+beta populations broken down by AO label
    (``L Ag Nb s``, ``L Ag Nb dxx``, etc.)

This module extracts that data so downstream strategy code can:
- Classify MO chemistry character (s / p / d / f / 4f vs 5f / etc.)
- Verify which spinors carry the open-shell electrons
- Suggest reorderings when the open shell has wrong character

The j-label is the most reliable character indicator for atomic /
linear-molecule systems; for polyatomic systems the AO-level gross
populations carry more information.
"""

from __future__ import annotations

import re
from typing import Any


# A VECPOP block begins with `Fermion ircop <label>` and runs until the
# next ircop header or section change.
_IRCOP_HEADER_RE = re.compile(
    r"Fermion ircop\s+(\S+)\s*\n\s*-+", re.MULTILINE
)

# Per-MO entry inside an ircop block.
#   "* Electronic eigenvalue no.   4: -8.2903648521434   (Occupation : f = 1.0000)  d 3/2;  1/2"
_MO_ENTRY_RE = re.compile(
    r"^\*\s*Electronic eigenvalue no\.\s*(\d+)\s*:\s*"
    r"(-?\d+\.\d+(?:[ED][+-]?\d+)?)\s+"
    r"\(Occupation\s*:\s*f\s*=\s*(\d+\.\d+)\s*\)"
    r"(?:\s+([sSpPdDfFgG])\s+(\d+/\d+);\s*(-?\d+/\d+))?",
    re.MULTILINE,
)

# AO-population row beneath an MO entry.
#   "Gross     Total   |    L Ag Nb dxx    L Ag Nb dyy    L Ag Nb dzz    L B2gNb dxz    L B3gNb dyz "
#   " alpha    0.4000  |      0.0667         0.0667         0.2667         0.0000         0.0000"
_AO_HEADER_RE = re.compile(r"^Gross\s+Total\s+\|\s+(.+)$", re.MULTILINE)


def parse_vecpop(text: str) -> dict[str, Any]:
    """Extract VECPOP / Mulliken-per-MO blocks from a DIRAC output.

    Returns ``{ircops: {<irrep>: [mo, mo, ...]}}`` where each MO dict has
    eigenvalue_index, energy_hartree, occupation, j_label (``s|p|d|f|g``),
    j_value (``"3/2"`` etc.), m_j, and gross_populations dict keyed by
    AO label.
    """
    out: dict[str, Any] = {"ircops": {}}
    if "Mulliken population analysis" not in text:
        return out

    # Slice into per-ircop blocks; each block runs from its header to the
    # next header or to the end of the analysis section.
    ircop_matches = list(_IRCOP_HEADER_RE.finditer(text))
    if not ircop_matches:
        return out
    for i, m in enumerate(ircop_matches):
        irrep = m.group(1)
        start = m.end()
        end = ircop_matches[i + 1].start() if i + 1 < len(ircop_matches) else len(text)
        block = text[start:end]
        # Stop at the next major section if it appears within this block.
        for sentinel in ("End of Mulliken", "Properties", "*****"):
            j = block.find(sentinel)
            if j > 0:
                block = block[:j]
                break
        out["ircops"][irrep] = _parse_ircop_block(block)
    return out


def _parse_ircop_block(block: str) -> list[dict[str, Any]]:
    """Parse the MO entries within a single fermion-ircop block."""
    mos: list[dict[str, Any]] = []
    # Split on the MO-entry sentinel ``* Electronic eigenvalue no.``
    entries = re.split(r"^\*\s*Electronic eigenvalue no\.", block, flags=re.MULTILINE)
    if len(entries) < 2:
        return mos
    for raw in entries[1:]:
        # Restore the sentinel so the regex still matches; easier than
        # rewriting the regex.
        full = "* Electronic eigenvalue no." + raw
        m = _MO_ENTRY_RE.search(full)
        if not m:
            continue
        eigval_no = int(m.group(1))
        e = _to_float(m.group(2))
        occ = _to_float(m.group(3))
        j_label = (m.group(4) or "").lower() or None
        j_value = m.group(5) or None
        m_j = m.group(6) or None

        # Extract gross populations from the table beneath this MO entry.
        gross = _parse_gross_populations(full)

        mos.append({
            "eigenvalue_index": eigval_no,
            "energy_hartree": e,
            "occupation": occ,
            "j_label": j_label,
            "j_value": j_value,
            "m_j": m_j,
            "character_string": (
                f"{j_label} {j_value}" if j_label and j_value else None
            ),
            "gross_populations": gross,
        })
    return mos


def _parse_gross_populations(mo_block: str) -> dict[str, dict[str, float]]:
    """Pull the alpha/beta gross-population row beneath an MO entry."""
    hdr_match = _AO_HEADER_RE.search(mo_block)
    if not hdr_match:
        return {}
    ao_labels_raw = hdr_match.group(1).rstrip()
    # AO labels are 4-char fixed-width-ish but with variable spacing;
    # collapse runs of 2+ spaces into a separator.
    ao_labels = [s.strip() for s in re.split(r"\s{2,}", ao_labels_raw) if s.strip()]

    # The alpha and beta rows follow the header.
    body = mo_block[hdr_match.end():]
    alpha = _parse_pop_row(body, "alpha")
    beta = _parse_pop_row(body, "beta")

    n = min(len(ao_labels), len(alpha), len(beta))
    if n == 0:
        return {}

    grouped: dict[str, dict[str, float]] = {}
    for i in range(n):
        label = ao_labels[i]
        grouped[label] = {
            "alpha": alpha[i],
            "beta": beta[i],
            "total": alpha[i] + beta[i],
        }
    return grouped


def _parse_pop_row(body: str, spin: str) -> list[float]:
    rx = re.compile(
        rf"^\s*{spin}\s+(-?\d+\.\d+)\s+\|\s+(.+)$", re.MULTILINE
    )
    m = rx.search(body)
    if not m:
        return []
    nums = [_to_float(t) for t in m.group(2).split()]
    return [x for x in nums if x is not None]


def classify_mo_character(mo: dict[str, Any]) -> dict[str, Any]:
    """Classify an MO's chemistry character.

    Uses (in priority order):
      1. j_label + j_value (atomic-shell j character — most reliable for atoms
         and linear molecules)
      2. Dominant gross-population AO label (parsed: which atom / which
         angular type contributes most)
      3. Fallback "unknown"

    Returns a dict with ``character`` (e.g. "f 5/2", "d 3/2", "s 1/2"),
    ``dominant_ao`` (e.g. "L Ag Nb dxx"), ``dominant_population``, and
    ``dominant_atom`` (the metal/element symbol parsed from the AO label).
    """
    label = mo.get("j_label")
    jv = mo.get("j_value")
    character = f"{label} {jv}" if label and jv else "unknown"

    gross = mo.get("gross_populations") or {}
    if gross:
        ranked = sorted(
            gross.items(), key=lambda kv: -(kv[1].get("total") or 0.0)
        )
        top_label, top_vals = ranked[0]
        dominant_pop = top_vals.get("total", 0.0)
        # AO label format: "L|S <irrep> <center> <ang_type>" — e.g. "L Ag Nb dxx"
        toks = top_label.split()
        dominant_atom = None
        for tok in reversed(toks):
            if tok.isalpha() and len(tok) <= 3:
                dominant_atom = tok
                break
    else:
        top_label = None
        dominant_pop = None
        dominant_atom = None

    return {
        "character": character,
        "j_label": label,
        "j_value": jv,
        "dominant_ao": top_label if gross else None,
        "dominant_population": dominant_pop,
        "dominant_atom": dominant_atom,
    }


def _to_float(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    s = s.replace("D", "E").replace("d", "e")
    try:
        return float(s)
    except ValueError:
        return None

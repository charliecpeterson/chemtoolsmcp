"""Parser for the Molcas '++ Molecular orbitals:' MO blocks.

Molcas modules emit MO tables in a fixed format:

    ++    Molecular orbitals:
          -------------------

          [optional title line: "SCF orbitals", "Pseudonatural active...", etc.]

          Molecular orbitals for symmetry species 1: a1

          Orbital            1         2         ...        10
          Energy           ...
          Occ. No.         ...

            1 N1   1s       <coeff>   <coeff>   ...
            2 N1   2s       ...
            ...

          [next batch of 10 orbitals — repeats]
          [next "Molecular orbitals for symmetry species ..." block]

The agent should parse the LAST '++ Molecular orbitals:' block in a task
because RASSCF natural-orbital MOs override SCF MOs that appeared earlier.
"""

from __future__ import annotations

import re
from typing import Any


_MO_HEADER_RE = re.compile(r"^[ \t]*\+\+[ \t]*Molecular orbitals:[ \t]*$", re.M)
_MO_SYM_HEADER_RE = re.compile(
    r"^[ \t]*Molecular orbitals for symmetry species\s+(\d+):\s+(\S+)\s*$", re.M
)
_ORB_HEADER_RE = re.compile(
    r"^[ \t]*Orbital\s+((?:\d+[ \t]+)+\d+)[ \t]*$", re.M
)
_ENERGY_RE = re.compile(
    r"^[ \t]*Energy\s+((?:-?\d+\.\d+(?:[Ee][+-]?\d+)?[ \t]*)+)$", re.M
)
_OCC_RE = re.compile(
    r"^[ \t]*Occ\.[ \t]*No\.\s+((?:-?\d+\.\d+(?:[Ee][+-]?\d+)?[ \t]*)+)$", re.M
)
# AO row: "    1 N1     1s         0.9996    0.0000 ..."
_AO_ROW_RE = re.compile(
    r"^\s*(\d+)\s+([A-Z][a-zA-Z0-9_]*)\s+(\S+)\s+((?:-?\d+\.\d+\s*)+)\s*$"
)


def find_mo_blocks(text: str) -> list[tuple[int, int]]:
    """Return (start, end) ranges of every '++ Molecular orbitals:' block.

    A block ends at the next '++' header or '--- Stop Module' marker, whichever
    comes first.
    """
    starts = [m.start() for m in _MO_HEADER_RE.finditer(text)]
    if not starts:
        return []
    next_marker = re.compile(r"^(\+\+|---\s+Stop Module)", re.M)
    ranges: list[tuple[int, int]] = []
    for i, start in enumerate(starts):
        # Find the next ++/Stop Module marker after this header (skipping the
        # '++  Molecular orbitals:' line itself)
        after = text.find("\n", start) + 1
        nxt = next_marker.search(text, after)
        end = nxt.start() if nxt else (starts[i + 1] if i + 1 < len(starts) else len(text))
        ranges.append((start, end))
    return ranges


def parse_last_mo_block(text: str, *, parse_coefficients: bool = True) -> dict[str, Any] | None:
    """Parse only the LAST MO block in `text`. RASSCF NOs win over earlier SCF MOs.

    `parse_coefficients=False` skips the AO-coefficient table (saves memory
    when the caller only needs orbital energies / occupations).
    """
    blocks = find_mo_blocks(text)
    if not blocks:
        return None
    start, end = blocks[-1]
    return parse_mo_block(text[start:end], parse_coefficients=parse_coefficients)


def parse_all_mo_blocks(text: str, *, parse_coefficients: bool = False) -> list[dict[str, Any]]:
    return [
        parse_mo_block(text[start:end], parse_coefficients=parse_coefficients)
        for start, end in find_mo_blocks(text)
    ]


def parse_mo_block(block: str, *, parse_coefficients: bool = True) -> dict[str, Any]:
    """Parse one '++ Molecular orbitals:' block."""
    title = _extract_title(block)
    sym_blocks: list[dict[str, Any]] = []
    sym_starts = [(m.start(), int(m.group(1)), m.group(2)) for m in _MO_SYM_HEADER_RE.finditer(block)]
    for i, (start, sym_idx, sym_label) in enumerate(sym_starts):
        sub_end = sym_starts[i + 1][0] if i + 1 < len(sym_starts) else len(block)
        sub = block[start:sub_end]
        sym_blocks.append(_parse_symmetry_section(sub, sym_idx, sym_label, parse_coefficients))
    return {"title": title, "symmetry_blocks": sym_blocks}


def _extract_title(block: str) -> str | None:
    """The line right after the '-------------------' separator is the title."""
    lines = block.splitlines()
    for i, line in enumerate(lines):
        if "-------------------" in line:
            for j in range(i + 1, min(i + 5, len(lines))):
                stripped = lines[j].strip()
                if not stripped:
                    continue
                if stripped.startswith("Molecular orbitals for symmetry"):
                    return None
                return stripped
            break
    return None


def _parse_symmetry_section(
    text: str, sym_index: int, irrep_label: str, parse_coefficients: bool
) -> dict[str, Any]:
    """Parse all batches inside one symmetry-species block.

    A symmetry block is a sequence of "panels" of up to 10 orbitals each.
    Each panel has Orbital + Energy + Occ. No. headers followed by AO rows.
    """
    orbitals: list[dict[str, Any]] = []
    panel_starts = [m.start() for m in _ORB_HEADER_RE.finditer(text)]
    for i, panel_start in enumerate(panel_starts):
        panel_end = panel_starts[i + 1] if i + 1 < len(panel_starts) else len(text)
        panel_text = text[panel_start:panel_end]
        orb_match = _ORB_HEADER_RE.search(panel_text)
        if not orb_match:
            continue
        orb_indices = [int(x) for x in orb_match.group(1).split()]
        n_orbs = len(orb_indices)
        en_match = _ENERGY_RE.search(panel_text)
        occ_match = _OCC_RE.search(panel_text)
        if not (en_match and occ_match):
            continue
        energies = [float(x) for x in en_match.group(1).split()]
        occupations = [float(x) for x in occ_match.group(1).split()]
        if len(energies) != n_orbs or len(occupations) != n_orbs:
            continue
        ao_rows: list[dict[str, Any]] = []
        if parse_coefficients:
            for line in panel_text.splitlines():
                row_match = _AO_ROW_RE.match(line)
                if not row_match:
                    continue
                ao_idx = int(row_match.group(1))
                atom_label = row_match.group(2)
                ao_label = row_match.group(3)
                values = [float(x) for x in row_match.group(4).split()]
                if len(values) != n_orbs:
                    continue
                ao_rows.append(
                    {
                        "ao_index": ao_idx,
                        "atom": atom_label,
                        "ao_label": ao_label,
                        "coefficients": values,
                    }
                )
        for j, orb_idx in enumerate(orb_indices):
            orb_record: dict[str, Any] = {
                "orbital_index": orb_idx,
                "energy_hartree": energies[j],
                "occupation": occupations[j],
            }
            if parse_coefficients:
                orb_record["ao_contributions"] = [
                    {
                        "ao_index": row["ao_index"],
                        "atom": row["atom"],
                        "ao_label": row["ao_label"],
                        "coefficient": row["coefficients"][j],
                    }
                    for row in ao_rows
                ]
                # Pre-compute the dominant AOs for human inspection
                orb_record["dominant_aos"] = _dominant_aos(orb_record["ao_contributions"], top_n=5)
            orbitals.append(orb_record)
    return {
        "symmetry_index": sym_index,
        "irrep_label": irrep_label,
        "orbitals": orbitals,
    }


def _dominant_aos(ao_contribs: list[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
    """Return the top-N AOs by |coefficient|, useful for orbital labelling."""
    ranked = sorted(ao_contribs, key=lambda c: -abs(c["coefficient"]))[:top_n]
    return [
        {
            "atom": r["atom"],
            "ao_label": r["ao_label"],
            "coefficient": round(r["coefficient"], 4),
        }
        for r in ranked
    ]

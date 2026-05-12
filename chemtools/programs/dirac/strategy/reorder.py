"""DIRAC `.REORDER MO` block — draft + apply to input.

The ``.REORDER MO`` keyword under ``*SCF`` reorders the starting orbitals
read from DFCOEF / .h5 before SCF iterations begin. Format::

    .REORDER MO
     1..8,10,9
     1..oo

One line per fermion ircop. Each line is a comma-separated list of
1-based MO indices using ``a..b`` range syntax (where ``oo`` = infinity
= "remainder unchanged").

Use cases:
- Fix wrong starting orbitals before SCF: a virtual with correct
  character is swapped into the open shell, the wrong-character orbital
  goes out.
- Required when an atomic-start checkpoint has the right orbitals but in
  the wrong order for the requested .OPEN SHELL spec.

This module provides:
- ``draft_reorder_block(per_ircop_orders)`` — render the .REORDER block text
- ``swaps_to_reorder_spec(n_orbitals_per_ircop, swaps_per_ircop)`` — convert
  per-ircop {old: new} swap maps into compact range specs
- ``apply_reorder_to_input(input_text, per_ircop_orders, replace=False)`` —
  insert the .REORDER block into an existing .inp's *SCF subsection
- ``parse_reorder_block(input_text)`` — extract any existing .REORDER spec
"""

from __future__ import annotations

import re
from typing import Any


def draft_reorder_block(per_ircop_orders: list[str]) -> str:
    """Render a ``.REORDER MO`` block from per-ircop order strings.

    ``per_ircop_orders`` is a list of strings, one per fermion ircop,
    e.g. ``["1..8,10,9", "1..oo"]``. Returns the block text suitable for
    insertion under ``*SCF``.
    """
    lines = [".REORDER MO"]
    for spec in per_ircop_orders:
        lines.append(" " + spec.strip())
    return "\n".join(lines) + "\n"


def swaps_to_reorder_spec(
    n_orbitals_total: int,
    swaps: list[tuple[int, int]],
) -> str:
    """Convert a list of swap pairs into a compact DIRAC reorder spec.

    Parameters
    ----------
    n_orbitals_total
        How many MOs are present in this ircop (informs the trailing
        ``..oo`` boundary). Pass any large number > all touched indices.
    swaps
        List of ``(orbital_a, orbital_b)`` 1-based index pairs to swap.
        Both must lie within ``[1, n_orbitals_total]``.

    Returns
    -------
    str
        Reorder spec ready to feed into ``draft_reorder_block`` — e.g.
        ``"1..22,29,24..28,23,30..oo"`` for swap (23, 29).
    """
    if not swaps:
        return "1..oo"

    # Build the new-order list by starting with identity then applying swaps.
    perm = list(range(1, n_orbitals_total + 1))
    for a, b in swaps:
        if 1 <= a <= n_orbitals_total and 1 <= b <= n_orbitals_total:
            ai = perm.index(a)
            bi = perm.index(b)
            perm[ai], perm[bi] = perm[bi], perm[ai]

    # Compress consecutive runs into a..b ranges
    pieces: list[str] = []
    i = 0
    while i < len(perm):
        j = i
        while j + 1 < len(perm) and perm[j + 1] == perm[j] + 1:
            j += 1
        if j == i:
            pieces.append(str(perm[i]))
        elif j - i == 1:
            pieces.append(f"{perm[i]},{perm[j]}")
        else:
            pieces.append(f"{perm[i]}..{perm[j]}")
        i = j + 1

    # Trailing identity range → use ..oo shorthand
    spec = ",".join(pieces)
    spec = re.sub(rf"\b{n_orbitals_total - 1}\.\.{n_orbitals_total}\b$", f"{n_orbitals_total - 1}..oo", spec)
    return spec


def apply_reorder_to_input(
    input_text: str,
    per_ircop_orders: list[str],
    *,
    replace: bool = False,
) -> dict[str, Any]:
    """Insert a ``.REORDER MO`` block into a DIRAC ``.inp`` text.

    DIRAC accepts ``.REORDER MO`` either under ``*SCF`` (preferred) or
    directly under ``**WAVE FUNCTION``. This function:

    1. Looks for an existing ``.REORDER`` anywhere in the **WAVE FUNCTION
       section. If found and ``replace=True``, replaces it in place.
    2. Otherwise inserts the block — under ``*SCF`` when that subsection
       exists, else immediately after the last keyword under
       ``**WAVE FUNCTION``.

    Returns ``{patched_text, action: inserted|replaced|already_present,
    location_line}``.
    """
    lines = input_text.splitlines()

    # Locate the **WAVE FUNCTION section bounds.
    wf_start = None
    wf_end = len(lines)
    for i, raw in enumerate(lines):
        s = raw.strip().upper()
        if s.startswith("**WAVE F"):
            wf_start = i
        elif wf_start is not None and (s.startswith("**") or s == "*END OF INPUT"):
            wf_end = i
            break

    if wf_start is None:
        raise ValueError(
            "Input does not contain a **WAVE FUNCTION section — cannot "
            "place .REORDER MO."
        )

    # Inside the section, optionally find a *SCF subsection.
    scf_start = None
    for i in range(wf_start + 1, wf_end):
        if lines[i].strip().upper() == "*SCF":
            scf_start = i
            break

    # Scan the whole **WAVE FUNCTION section for an existing .REORDER.
    existing_reorder_idx = None
    for k in range(wf_start + 1, wf_end):
        if lines[k].strip().upper().startswith(".REORDER"):
            existing_reorder_idx = k
            break

    block_lines = draft_reorder_block(per_ircop_orders).rstrip("\n").splitlines()

    if existing_reorder_idx is not None:
        if not replace:
            return {
                "patched_text": input_text,
                "action": "already_present",
                "location_line": existing_reorder_idx + 1,
                "message": (
                    "`.REORDER` already exists in **WAVE FUNCTION; pass "
                    "replace=True to overwrite."
                ),
            }
        # Remove the existing .REORDER + its argument lines. Argument lines
        # are spec strings made up of digits, commas, dots, and 'oO' (..oo)
        # — they may or may not have leading whitespace.
        end_k = existing_reorder_idx + 1
        _ARG_LINE = re.compile(r"^\s*[\doO][\doO,.\s]*$")
        while end_k < wf_end:
            t = lines[end_k]
            stripped = t.strip()
            if not stripped:
                end_k += 1
                continue
            if stripped.startswith(".") or stripped.startswith("*"):
                break
            if not _ARG_LINE.match(t):
                break
            end_k += 1
        new_lines = lines[:existing_reorder_idx] + block_lines + lines[end_k:]
        return {
            "patched_text": "\n".join(new_lines) + "\n",
            "action": "replaced",
            "location_line": existing_reorder_idx + 1,
        }

    # Insert: prefer right after *SCF; fall back to right after **WAVE FUNCTION
    # header (well, after the existing keywords there — the last line in the
    # section before the next section starts).
    if scf_start is not None:
        insert_at = scf_start + 1
        placement = "after *SCF"
    else:
        # Insert before wf_end (i.e. at the end of the **WAVE FUNCTION section)
        insert_at = wf_end
        placement = "at end of **WAVE FUNCTION"

    new_lines = lines[:insert_at] + block_lines + lines[insert_at:]
    return {
        "patched_text": "\n".join(new_lines) + "\n",
        "action": "inserted",
        "location_line": insert_at + 1,
        "placement": placement,
    }


def parse_reorder_block(input_text: str) -> dict[str, Any] | None:
    """Extract any existing ``.REORDER MO`` block from an ``.inp``.

    Returns ``{ircop_orders: [str, str, ...], line: int}`` or None.
    """
    lines = input_text.splitlines()
    for i, raw in enumerate(lines):
        s = raw.strip()
        if s.upper().startswith(".REORDER"):
            # Consume continuation lines (each one is one ircop's spec)
            orders: list[str] = []
            for j in range(i + 1, len(lines)):
                t = lines[j].strip()
                if not t:
                    break
                if t.startswith(".") or t.startswith("*"):
                    break
                orders.append(t)
            return {"ircop_orders": orders, "line": i + 1}
    return None

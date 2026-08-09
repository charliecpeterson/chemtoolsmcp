"""Geometry + opt-trajectory parsers.

Three things live here:

  parse_cartesian_blocks(text)
    Walk every ``Cartesian coordinates in angstrom:`` block in the file. These
    are emitted by every module that reports a geometry — typically once per
    SCF/RASSCF/MCLR/etc. invocation.

  parse_final_geometry(text)
    Find the LAST ``Nuclear coordinates of the final structure / bohr`` block
    emitted by SLAPAF; this is the converged geometry of an opt loop. Falls
    back to the last ``Cartesian coordinates in angstrom:`` block if no SLAPAF
    final block is present (e.g. for a single-point run).

  parse_energy_statistics(text)
    Extract the SLAPAF ``Energy Statistics for Geometry Optimization`` table.
    The table is cumulative — each opt iteration extends it with new rows.
    We keep only the LAST emitted block so duplicates aren't appended.

  parse_trajectory(text)
    Combine: walk SLAPAF iterations, pair each with its preceding geometry
    snapshot from the last ``Cartesian coordinates`` block before that
    iteration's Energy-Statistics row.
"""

from __future__ import annotations

import re
from typing import Any


_FLOAT_RE = r"-?\d+\.\d+(?:[Ee][+-]?\d+)?"

# "++    Cartesian coordinates in angstrom:" or just plain "Cartesian coordinates ..."
_CART_HEADER_ANG_RE = re.compile(r"^\s*Cartesian coordinates in angstrom:\s*$", re.M)
_CART_HEADER_BOHR_RE = re.compile(r"^\s*Cartesian coordinates in bohr:\s*$", re.M)
_ATOM_ROW_RE = re.compile(
    r"^\s*\d+\s+([A-Z][A-Za-z0-9]*)\s+("
    + _FLOAT_RE + r")\s+("
    + _FLOAT_RE + r")\s+("
    + _FLOAT_RE + r")\s*(?:" + _FLOAT_RE + r")?\s*$"
)

# SLAPAF final structure block
_SLAPAF_FINAL_BOHR_RE = re.compile(
    r"\*\s*Nuclear coordinates of the final structure / bohr\s*\*", re.M
)
_SLAPAF_FINAL_ANG_RE = re.compile(
    r"\*\s*Nuclear coordinates of the final structure / angstrom\s*\*", re.M
)
_FINAL_ATOM_ROW_RE = re.compile(
    r"^\s*([A-Z][A-Za-z0-9]*)\s+("
    + _FLOAT_RE + r")\s+("
    + _FLOAT_RE + r")\s+("
    + _FLOAT_RE + r")\s*$"
)

# Energy Statistics table
_ENERGY_STATS_HEADER_RE = re.compile(
    r"^\*+\s*\n\*\s*Energy Statistics for Geometry Optimization\s*\*", re.M
)
_ENERGY_STATS_ROW_RE = re.compile(
    r"^[ \t]*(\d+)[ \t]+("
    + _FLOAT_RE + r")[ \t]+("
    + _FLOAT_RE + r")[ \t]+("
    + _FLOAT_RE + r")[ \t]+("
    + _FLOAT_RE + r")[ \t]+(\S+)[ \t]+("
    + _FLOAT_RE + r")[ \t]+(\S+)[ \t]+("
    + _FLOAT_RE + r")[ \t]+(\S+)",
    re.M,
)
_CONVERGED_RE = re.compile(
    r"Geometry is converged in\s+(\d+)\s+iterations to a (\S+)\s+Structure", re.M
)


class GeometryBlockIndexError(IndexError):
    def __init__(self, block_index: int, block_count: int) -> None:
        self.block_index = block_index
        self.block_count = block_count
        super().__init__(
            f"geometry block index {block_index} is outside {block_count} blocks"
        )


def parse_cartesian_blocks(text: str) -> list[dict[str, Any]]:
    """Find every `Cartesian coordinates in angstrom:` block. Returns list of
    {atoms: [...], line_start: int, units: 'angstrom' or 'bohr'}.
    """
    blocks: list[dict[str, Any]] = []
    for header_re, units in (
        (_CART_HEADER_ANG_RE, "angstrom"),
        (_CART_HEADER_BOHR_RE, "bohr"),
    ):
        for m in header_re.finditer(text):
            line_start = text.count("\n", 0, m.start()) + 1
            atoms = _parse_atom_block_after(text, m.end(), _ATOM_ROW_RE)
            if atoms:
                blocks.append({"atoms": atoms, "line_start": line_start, "units": units})
    blocks.sort(key=lambda b: b["line_start"])
    return blocks


def parse_final_geometry(text: str) -> dict[str, Any] | None:
    """Find SLAPAF's `Nuclear coordinates of the final structure / bohr` block.

    Falls back to the LAST Cartesian-coordinates block if SLAPAF didn't run.
    """
    final_bohr = list(_SLAPAF_FINAL_BOHR_RE.finditer(text))
    if final_bohr:
        m = final_bohr[-1]
        atoms = _parse_atom_block_after(text, m.end(), _FINAL_ATOM_ROW_RE)
        if atoms:
            return {
                "atoms": atoms,
                "units": "bohr",
                "source": "slapaf_final",
                "line_start": text.count("\n", 0, m.start()) + 1,
            }
    final_ang = list(_SLAPAF_FINAL_ANG_RE.finditer(text))
    if final_ang:
        m = final_ang[-1]
        atoms = _parse_atom_block_after(text, m.end(), _FINAL_ATOM_ROW_RE)
        if atoms:
            return {
                "atoms": atoms,
                "units": "angstrom",
                "source": "slapaf_final",
                "line_start": text.count("\n", 0, m.start()) + 1,
            }
    # Fallback: last Cartesian-coordinates block
    blocks = parse_cartesian_blocks(text)
    if not blocks:
        return None
    last = blocks[-1]
    return {**last, "source": "last_cartesian_block"}


def select_geometry(
    text: str,
    block_index: int | None = None,
) -> dict[str, Any] | None:
    """Select one explicit Cartesian block or the final usable geometry."""
    if block_index is None:
        return parse_final_geometry(text)
    blocks = parse_cartesian_blocks(text)
    if not blocks:
        return None
    if block_index < 0 or block_index >= len(blocks):
        raise GeometryBlockIndexError(block_index, len(blocks))
    return blocks[block_index]


def parse_energy_statistics(text: str) -> dict[str, Any] | None:
    """Extract the SLAPAF Energy Statistics table.

    The table is cumulative (each opt iteration extends it) — we keep only the
    LAST emitted block so duplicates aren't appended.
    """
    starts = list(_ENERGY_STATS_HEADER_RE.finditer(text))
    if not starts:
        return None
    last = starts[-1]
    end = len(text)
    body = text[last.end():end]
    rows: list[dict[str, Any]] = []
    for m in _ENERGY_STATS_ROW_RE.finditer(body):
        rows.append(
            {
                "iteration": int(m.group(1)),
                "energy_au": float(m.group(2)),
                "energy_change_au": float(m.group(3)),
                "gradient_norm": float(m.group(4)),
                "gradient_max": float(m.group(5)),
                "gradient_max_element": m.group(6),
                "step_max": float(m.group(7)),
                "step_max_element": m.group(8),
                "estimated_final_energy_au": float(m.group(9)),
                "geom_update_method": m.group(10),
            }
        )
    converged = _CONVERGED_RE.search(body)
    return {
        "rows": rows,
        "n_iterations": len(rows),
        "converged": bool(converged),
        "converged_message": converged.group(0) if converged else None,
        "structure_type": converged.group(2) if converged else None,
    }


def parse_trajectory(text: str) -> dict[str, Any]:
    """Combine geometry snapshots with the Energy Statistics table.

    Each iteration row is paired with the LAST geometry snapshot before its
    Energy-Statistics emission. Rows that have no preceding geometry snapshot
    are still returned (geometry=None) so the agent can see the energy
    progression.
    """
    geometry_blocks = parse_cartesian_blocks(text)
    stats = parse_energy_statistics(text)
    final_geometry = parse_final_geometry(text)
    if not stats:
        return {
            "iterations": [],
            "n_iterations": 0,
            "converged": False,
            "final_geometry": final_geometry,
        }
    # Locate the line where Energy Statistics ends (the table is at the END of
    # the SLAPAF block). For a multi-iter run, there are multiple SLAPAF blocks
    # and each has its own Energy-Statistics emission; we kept the last one
    # which is the cumulative table. Pair each row with the geometry block
    # that precedes the corresponding SLAPAF call.
    # Each opt iteration runs SEWARD/SCF/RASSCF/etc., and each invocation
    # emits its own Cartesian-coordinates block. Empirically, the geometry
    # block(s) for iteration N appear in the slice between iter (N-1)'s and
    # iter N's Energy-Statistics emissions — but for short runs, all geometry
    # blocks may precede the cumulative table. Use a simple per-iteration
    # bucketing: take the LAST geometry block whose line_start is at or before
    # the iteration's "effective offset".
    iterations: list[dict[str, Any]] = []
    for idx, row in enumerate(stats["rows"]):
        # Geometry index for iteration N (1-indexed) is the (N-1)-th block,
        # if there are at least n_iter blocks; otherwise use the closest.
        if geometry_blocks:
            target_idx = min(row["iteration"] - 1, len(geometry_blocks) - 1)
            if target_idx < 0:
                target_idx = 0
            geom = geometry_blocks[target_idx]
        else:
            geom = None
        iterations.append({**row, "geometry": geom})
    return {
        "iterations": iterations,
        "n_iterations": stats["n_iterations"],
        "converged": stats["converged"],
        "structure_type": stats.get("structure_type"),
        "final_geometry": final_geometry,
    }


def _parse_atom_block_after(text: str, start_offset: int, row_re: re.Pattern) -> list[dict[str, Any]]:
    """Walk lines after start_offset until non-atom, collecting atom rows."""
    out: list[dict[str, Any]] = []
    seen_first_atom = False
    for line in text[start_offset:].splitlines():
        m = row_re.match(line)
        if m:
            seen_first_atom = True
            label = m.group(1)
            x = float(m.group(2))
            y = float(m.group(3))
            z = float(m.group(4))
            out.append(
                {
                    "label": label,
                    "symbol": _element_from_label(label),
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
            continue
        # Stop only after we've started collecting atoms and hit a non-match
        if seen_first_atom and line.strip():
            # Is it just a separator line (---)? Then we're done.
            if line.strip().startswith("-") or line.strip().startswith("*"):
                break
            # Stop on any other non-atom non-blank line
            break
    return out


def _element_from_label(label: str) -> str:
    """Strip trailing digits: 'C1' → 'C', 'Pb1' → 'Pb'."""
    base = "".join(c for c in label if c.isalpha())
    return base[0].upper() + base[1:].lower() if len(base) > 1 else base.upper()


def _approx_iter_line(text: str, iteration: int) -> int:
    """Best-guess line number where iteration N's geometry was emitted.

    For multi-iter SLAPAF, each Energy-Statistics emission is at the end of the
    SLAPAF block, but the geometry for iteration N appears earlier (before the
    SCF/RASSCF for that iter). Without parsing the full Do-While loop boundary,
    we approximate by scanning for the (N+1)th `Cartesian coordinates` block
    if present, falling back to the last block.
    """
    blocks = list(_CART_HEADER_ANG_RE.finditer(text)) + list(_CART_HEADER_BOHR_RE.finditer(text))
    blocks.sort(key=lambda m: m.start())
    if iteration <= len(blocks):
        return text.count("\n", 0, blocks[iteration - 1].start()) + 1
    return len(text.splitlines())

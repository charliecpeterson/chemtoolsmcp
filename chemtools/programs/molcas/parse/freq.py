"""Parser for the Molcas harmonic-frequency block.

MCLR (analytical) and NUMERICAL_GRADIENT both emit a `Harmonic frequencies in
cm-1` table. The block layout is:

    ************************************
    * Harmonic frequencies in cm-1     *
    * Intensities in km/mole           *
    *                                  *
    * No correction due to curvilinear *
    * representations has been done    *
    *                                  *
    ************************************

      Symmetry a1
      ==============

                    1         2         3         4         5         6
        Frequency:  0.05    847.85    966.06   1044.68   1187.60   1492.41

        Intensity:  6.521E-09 1.258E-03 5.325E+00 4.160E-01 6.388E-02 3.928E+00
        Red. mass:  6.00427   5.14053   2.15320   2.92806   1.22721   1.83626

        C1     z    0.30151   0.35189  -0.21164   0.11598  -0.06875  -0.03291
        ...

      Symmetry a2
      ...

Imaginary frequencies are written `i123.4` (Molcas convention) — we expose
them as negative floats. The 5/6 zero-frequency translation/rotation modes
are kept (the agent can filter them out via |freq| < 1).
"""

from __future__ import annotations

import re
from typing import Any


_BLOCK_HEADER_RE = re.compile(r"\*\s*Harmonic frequencies in cm-1\s*\*")
_SYM_HEADER_RE = re.compile(r"^\s*Symmetry\s+(\S+)\s*$", re.M)
_PANEL_INDEX_RE = re.compile(r"^\s*(\d+(?:\s+\d+)*)\s*$", re.M)
_FREQUENCY_RE = re.compile(r"^\s*Frequency:\s+(.+?)\s*$", re.M)
_INTENSITY_RE = re.compile(r"^\s*Intensity:\s+(.+?)\s*$", re.M)
_RED_MASS_RE = re.compile(r"^\s*Red\.\s*mass:\s+(.+?)\s*$", re.M)
# Mode-displacement row: "C1     z    0.30151   0.35189 ..."
_DISP_ROW_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9]*)\s+([xyz])\s+((?:-?\d+\.\d+\s*)+)\s*$")


def find_freq_blocks(text: str) -> list[tuple[int, int]]:
    """Return (start, end) ranges of every `Harmonic frequencies` table in the text.

    A block ends at the next `++` header (Thermochemistry / next module section)
    or `--- Stop Module` marker, whichever comes first.
    """
    starts = [m.start() for m in _BLOCK_HEADER_RE.finditer(text)]
    if not starts:
        return []
    next_marker_re = re.compile(r"^(\+\+|---\s+Stop Module|\*+\s*$)", re.M)
    ranges: list[tuple[int, int]] = []
    for i, start in enumerate(starts):
        # Skip past the box header itself
        after = text.find("\n", start) + 1
        # Find a delimiter that's clearly outside the freq block
        nxt = next((m for m in next_marker_re.finditer(text, after) if m.start() > after + 200), None)
        end = nxt.start() if nxt else (starts[i + 1] if i + 1 < len(starts) else len(text))
        ranges.append((start, end))
    return ranges


def parse_freq_block(block: str) -> dict[str, Any]:
    """Parse one freq block. Returns symmetry-grouped + flat normal-mode lists.

    Output:
        {
          "symmetry_blocks": [
            {"irrep": "a1", "modes": [{"index_in_sym": 1, "frequency_cm1": 0.05,
                                       "ir_intensity_km_per_mol": 6.521e-9,
                                       "reduced_mass": 6.00427,
                                       "displacements": {"C1_z": 0.30151, ...}},
                                      ...]},
            ...
          ],
          "n_modes": 27,
          "n_imaginary": 3,
          "n_atoms": 9,
          "modes": [...flat list, sorted by symmetry then index_in_sym...],
          "frequencies_cm1": [...],
          "ir_intensities_km_per_mol": [...],
        }
    """
    sym_starts = list(_SYM_HEADER_RE.finditer(block))
    sym_blocks: list[dict[str, Any]] = []
    for i, m in enumerate(sym_starts):
        irrep = m.group(1).strip()
        sub_end = sym_starts[i + 1].start() if i + 1 < len(sym_starts) else len(block)
        sub = block[m.start():sub_end]
        modes = _parse_sym_block(sub)
        sym_blocks.append({"irrep": irrep, "modes": modes})

    flat_modes: list[dict[str, Any]] = []
    for sb in sym_blocks:
        for mode in sb["modes"]:
            flat_modes.append({**mode, "irrep": sb["irrep"]})

    n_imag = sum(1 for m in flat_modes if m["frequency_cm1"] < -0.5)
    labels_seen: set[str] = set()
    for mode in flat_modes:
        for key in (mode.get("displacements") or {}).keys():
            labels_seen.add(key.rsplit("_", 1)[0])
    # Nonlinear molecule: 3N modes. Linear: 3N-1 (or close). Use the cleaner of
    # the two estimates given that this is a hint, not a contract.
    n_atoms_from_modes = len(flat_modes) // 3 if len(flat_modes) % 3 == 0 else None

    return {
        "symmetry_blocks": sym_blocks,
        "modes": flat_modes,
        "n_modes": len(flat_modes),
        "n_imaginary": n_imag,
        "n_unique_atom_labels": len(labels_seen),
        "n_atoms_estimated": n_atoms_from_modes,
        "frequencies_cm1": [m["frequency_cm1"] for m in flat_modes],
        "ir_intensities_km_per_mol": [m.get("ir_intensity_km_per_mol") for m in flat_modes],
    }


def parse_last_freq_block(text: str) -> dict[str, Any] | None:
    """Convenience: find the LAST freq block in `text` and parse it."""
    blocks = find_freq_blocks(text)
    if not blocks:
        return None
    start, end = blocks[-1]
    return parse_freq_block(text[start:end])


def _parse_sym_block(sub: str) -> list[dict[str, Any]]:
    """Parse one symmetry sub-block — may contain multiple panels of up to 6 modes each."""
    out: list[dict[str, Any]] = []
    panels = list(_PANEL_INDEX_RE.finditer(sub))
    for i, m in enumerate(panels):
        try:
            mode_indices = [int(x) for x in m.group(1).split()]
        except ValueError:
            continue
        n = len(mode_indices)
        panel_end = panels[i + 1].start() if i + 1 < len(panels) else len(sub)
        panel = sub[m.start():panel_end]

        freqs = _extract_floats(panel, _FREQUENCY_RE, n, allow_imaginary=True)
        intensities = _extract_floats(panel, _INTENSITY_RE, n)
        red_mass = _extract_floats(panel, _RED_MASS_RE, n)
        if not freqs:
            continue

        # Per-atom-direction displacement rows
        displacement_rows: list[tuple[str, str, list[float]]] = []
        for line in panel.splitlines():
            row_m = _DISP_ROW_RE.match(line)
            if not row_m:
                continue
            atom = row_m.group(1)
            direction = row_m.group(2)
            values = [float(x) for x in row_m.group(3).split()]
            if len(values) != n:
                continue
            displacement_rows.append((atom, direction, values))

        for j, mode_idx in enumerate(mode_indices):
            disp = {}
            for atom, direction, values in displacement_rows:
                disp[f"{atom}_{direction}"] = values[j]
            out.append(
                {
                    "index_in_sym": mode_idx,
                    "frequency_cm1": freqs[j] if j < len(freqs) else None,
                    "ir_intensity_km_per_mol": intensities[j] if j < len(intensities or []) else None,
                    "reduced_mass": red_mass[j] if j < len(red_mass or []) else None,
                    "displacements": disp,
                }
            )
    return out


def _extract_floats(
    panel: str,
    pattern: re.Pattern,
    expected_n: int,
    *,
    allow_imaginary: bool = False,
) -> list[float] | None:
    m = pattern.search(panel)
    if not m:
        return None
    tokens = m.group(1).split()
    out: list[float] = []
    for tok in tokens:
        if allow_imaginary and tok.startswith("i"):
            try:
                out.append(-float(tok[1:]))
            except ValueError:
                pass
        else:
            try:
                out.append(float(tok))
            except ValueError:
                pass
    if len(out) != expected_n:
        return None
    return out


def parse_cartesian_reaction_vector(text: str) -> list[list[float]] | None:
    """Pull the `Cartesian Reaction vector` block from a SLAPAF/MCLR output.

    Returns a list of [x, y, z] rows (one per atom, input order) or None
    if no such block is present.

    Pattern (from prior TS-search SLAPAF output):
        *********************************************************
        * The Cartesian Reaction vector                         *
        *********************************************************
         ATOM              X               Y               Z
         N1              -0.016339        0.000000       -0.003290
         C1               0.030853        0.000000       -0.015361
         H1              -0.140347        0.000000        0.228620
    """
    import re
    m = re.search(
        r"\*\s*The Cartesian Reaction vector\s*\*[\s\S]+?ATOM\s+X\s+Y\s+Z\s*\n([\s\S]+?)\n\s*\n",
        text, re.IGNORECASE,
    )
    if not m:
        return None
    rows: list[list[float]] = []
    for line in m.group(1).splitlines():
        toks = line.split()
        # Expected: <label> x y z — last 3 must be floats
        if len(toks) < 4:
            continue
        try:
            x, y, z = float(toks[-3]), float(toks[-2]), float(toks[-1])
        except ValueError:
            continue
        rows.append([x, y, z])
    return rows or None

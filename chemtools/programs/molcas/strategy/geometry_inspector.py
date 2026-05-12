"""Molcas geometry-inspection adapter.

Thin wrapper that resolves a Molcas-specific geometry source (output_file
parsed via parse_final_geometry, input_file parsed via the SEWARD basis
block extractor, or an explicit atoms list) and then delegates the pure
math to ``chemtools.core.geometry.inspect_geometry``.

Molcas-specific bits:
  - Source resolution (parse_final_geometry / SEWARD-block atoms extractor)
  - Unit normalization (bohr → Å, since parse_final_geometry returns
    coordinates in bohr from the "Nuclear coordinates for the next
    iteration" section)

Everything else (distance/angle/dihedral, formula, bond detection,
fragment detection, measurements, COM) lives in ``chemtools/core/geometry.py``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from chemtools.core.common import read_text
from chemtools.core.units import ANGSTROM_PER_BOHR
from chemtools.core.geometry import (
    inspect_geometry as _core_inspect_geometry,
    norm_element,
)
from chemtools.programs.molcas.parse.geometry import parse_final_geometry


def _extract_atoms_from_input(text: str) -> list[dict] | None:
    """Best-effort: pull (symbol, x, y, z) atom rows from a Molcas input
    file's SEWARD/GATEWAY block. Looks for lines like::

        H1   0.0000000000   0.0000000000   0.0000000000

    inside a ``Basis set`` … ``End of basis`` block.
    """
    atoms: list[dict] = []
    in_basis = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("basis set"):
            in_basis = True
            continue
        if line.lower().startswith("end of basis"):
            in_basis = False
            continue
        if not in_basis:
            continue
        m = re.match(
            r"^([A-Z][a-z]?\d*)\s+([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)\s+"
            r"([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)\s+([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)\s*$",
            line,
        )
        if m:
            label = m.group(1)
            atoms.append({
                "symbol": norm_element(label),
                "label": label,
                "x": float(m.group(2)),
                "y": float(m.group(3)),
                "z": float(m.group(4)),
            })
    return atoms or None


def inspect_geometry(
    *,
    output_file: str | None = None,
    input_file: str | None = None,
    atoms: list[dict] | None = None,
    max_bond_length: float = 2.5,
    min_safe_distance: float = 0.6,
    covalent_tolerance: float = 1.20,
    measurements: dict[str, list[list[int]]] | None = None,
) -> dict[str, Any]:
    """Inspect a Molcas geometry. See ``chemtools.core.geometry.inspect_geometry``
    for the full return-shape documentation.

    Exactly one source must be provided:
      - output_file: parses the final converged geometry from a Molcas .out
      - input_file: extracts atoms from the &SEWARD/&GATEWAY basis blocks
      - atoms: explicit list of {symbol, x, y, z} (assumed in Å)
    """
    n_sources = sum(x is not None for x in (output_file, input_file, atoms))
    if n_sources != 1:
        raise ValueError("Pass exactly one of output_file / input_file / atoms.")

    source_units = "angstrom"
    if output_file is not None:
        if not Path(output_file).is_file():
            raise FileNotFoundError(f"Output file not found: {output_file}")
        text = read_text(output_file)
        geom = parse_final_geometry(text)
        if not geom or not geom.get("atoms"):
            return {
                "verdict": "no_geometry",
                "error": "no_geometry",
                "message": f"Could not find a final geometry in {output_file}.",
            }
        atoms = list(geom["atoms"])
        source_units = geom.get("units", "angstrom")
    elif input_file is not None:
        if not Path(input_file).is_file():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        text = read_text(input_file)
        extracted = _extract_atoms_from_input(text)
        if not extracted:
            return {
                "verdict": "no_geometry",
                "error": "no_geometry",
                "message": f"Could not extract atoms from {input_file}.",
            }
        atoms = extracted
        source_units = "bohr"  # Molcas input convention after our drafter

    # Normalize all coordinates to Å internally so bond detection against
    # COVALENT_RADII (Å) works regardless of source units.
    if source_units.lower() == "bohr":
        atoms = [
            {**a, "x": a["x"] * ANGSTROM_PER_BOHR,
                  "y": a["y"] * ANGSTROM_PER_BOHR,
                  "z": a["z"] * ANGSTROM_PER_BOHR}
            for a in atoms
        ]

    return _core_inspect_geometry(
        atoms,
        max_bond_length=max_bond_length,
        min_safe_distance=min_safe_distance,
        covalent_tolerance=covalent_tolerance,
        measurements=measurements,
        units="angstrom",
    )

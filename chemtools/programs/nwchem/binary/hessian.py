"""NWChem `.hess` (ASCII Hessian) reader.

Format (from NWChem docs `14-PES.pdf.txt` and verified against the bundled
`examples/nwchem/*.hess` fixtures):

  * Plain ASCII text.
  * One Fortran double per line, written in `D` exponent notation
    (e.g. `-6.6285284506D-09`). Python `float()` doesn't accept `D`; we
    translate to `E` before parsing.
  * The values pack the LOWER TRIANGLE of the Cartesian Hessian
    H[i,j] = d²E / dx_i dx_j, row-major:

        line 1:  H[0,0]
        line 2:  H[1,0]
        line 3:  H[1,1]
        line 4:  H[2,0]
        line 5:  H[2,1]
        line 6:  H[2,2]
        ...

    Total entries = nat3 * (nat3 + 1) / 2, where nat3 = 3 * n_atoms.

  * Units are atomic units (Eh / bohr²).

What this is useful for downstream:

  * Seeding a TS optimization (`driver; inhess 2; reuse <file.hess>`).
  * Re-doing frequency analysis with different isotope masses without
    re-running the SCF + analytic 2nd derivative.
  * Sanity checking that an existing `.hess` is well-formed before using
    it in a restart workflow.

The `.drv.hess` variant is a different binary format used by the driver
module during opt — handled by a separate `drv_hessian.py` reader (TBD).
"""

from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Any


def _entries_to_nat3(n_entries: int) -> int:
    """Recover nat3 (3 * n_atoms) from the count of lower-triangle entries.

    n_entries = nat3 * (nat3 + 1) / 2  =>  nat3 = (sqrt(8n + 1) - 1) / 2

    Returns the integer nat3 or raises ValueError if the count isn't a
    valid triangular number.
    """
    disc = 8 * n_entries + 1
    root = int(math.isqrt(disc))
    if root * root != disc:
        raise ValueError(
            f"Hessian file has {n_entries} entries, which is not a triangular "
            f"number (expected nat3 * (nat3 + 1) / 2 for some integer nat3)."
        )
    nat3 = (root - 1) // 2
    if nat3 * (nat3 + 1) // 2 != n_entries:
        raise ValueError(
            f"Hessian file has {n_entries} entries; recovered nat3={nat3} but "
            f"{nat3} * ({nat3} + 1) / 2 = {nat3 * (nat3 + 1) // 2} != {n_entries}."
        )
    return nat3


def parse_nwchem_hessian(
    hessian_path: str,
    *,
    return_matrix: bool = True,
) -> dict[str, Any]:
    """Parse an NWChem `.hess` (ASCII lower-triangle Cartesian Hessian).

    Args:
        hessian_path:    Path to the `.hess` file.
        return_matrix:   If True, return the full symmetric Hessian as a
                         nested list of floats. If False, return only the
                         flat triangle entries (saves memory for big systems).

    Returns dict with:
        hessian_file:       Resolved absolute path.
        file_size_bytes:    Size on disk.
        n_atoms:            Recovered from nat3.
        n_dof:              3 * n_atoms = nat3.
        n_triangle_entries: Count of lower-triangle entries read.
        triangle:           Flat list of nat3*(nat3+1)/2 floats (row-major,
                            lower triangle). Always returned.
        hessian:            Full symmetric n_dof x n_dof matrix as a nested
                            list when return_matrix=True; absent otherwise.
        units:              "hartree/bohr^2" (NWChem .hess is always in
                            atomic units).
        stats:              {max_abs, min_abs_nonzero, frobenius_norm,
                             diagonal_min, diagonal_max} — quick sanity
                             checks for "is this a real Hessian".

    Raises:
        FileNotFoundError if hessian_path does not exist.
        ValueError if the entry count is not a valid triangular number, or
            if a line cannot be parsed as a float after D->E translation.
    """
    path = Path(hessian_path)
    if not path.exists():
        raise FileNotFoundError(f"NWChem .hess file not found: {hessian_path}")

    raw_text = path.read_text(encoding="utf-8", errors="replace")
    triangle: list[float] = []
    for line_idx, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        # NWChem writes Fortran D-exponents — Python wants E.
        token = line.replace("D", "E").replace("d", "e")
        try:
            triangle.append(float(token))
        except ValueError as exc:
            raise ValueError(
                f"{hessian_path}: line {line_idx}: could not parse {line!r} "
                f"as a float (D-exponent translation failed)"
            ) from exc

    nat3 = _entries_to_nat3(len(triangle))
    n_atoms, remainder = divmod(nat3, 3)
    if remainder != 0:
        raise ValueError(
            f"Recovered nat3 = {nat3} is not divisible by 3 (expected nat3 = 3 * n_atoms)."
        )

    result: dict[str, Any] = {
        "hessian_file": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        "n_atoms": n_atoms,
        "n_dof": nat3,
        "n_triangle_entries": len(triangle),
        "triangle": triangle,
        "units": "hartree/bohr^2",
    }

    # Sanity stats — computable directly from the triangle without building the
    # full matrix, so they stay cheap for large systems.
    diag_values: list[float] = []
    cursor = 0
    for i in range(nat3):
        # Row i of the lower triangle has (i+1) entries; the last is the diagonal.
        cursor += i + 1
        diag_values.append(triangle[cursor - 1])

    abs_values = [abs(v) for v in triangle]
    nonzero_abs = [v for v in abs_values if v > 0.0]
    frobenius_sq = 0.0
    for i in range(nat3):
        # Diagonal terms count once, off-diagonal terms count twice (symmetry).
        row_start = i * (i + 1) // 2
        for j in range(i + 1):
            v = triangle[row_start + j]
            frobenius_sq += v * v if i == j else 2.0 * v * v

    result["stats"] = {
        "max_abs": max(abs_values) if abs_values else 0.0,
        "min_abs_nonzero": min(nonzero_abs) if nonzero_abs else 0.0,
        "frobenius_norm": math.sqrt(frobenius_sq),
        "diagonal_min": min(diag_values) if diag_values else 0.0,
        "diagonal_max": max(diag_values) if diag_values else 0.0,
    }

    if return_matrix:
        matrix = [[0.0] * nat3 for _ in range(nat3)]
        cursor = 0
        for i in range(nat3):
            for j in range(i + 1):
                v = triangle[cursor]
                cursor += 1
                matrix[i][j] = v
                matrix[j][i] = v
        result["hessian"] = matrix

    return result


__all__ = ["parse_nwchem_hessian"]

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


# ---------------------------------------------------------------------------
# Harmonic frequency analysis from a parsed Hessian
# ---------------------------------------------------------------------------

# NWChem-printed atomic masses for the most common elements. Falls back to a
# wider table if needed. These specific values match NWChem's default "Atomic
# Mass" output (most-abundant isotope), not IUPAC average atomic weights.
_NWCHEM_DEFAULT_MASSES_AMU: dict[str, float] = {
    "H":  1.007825,  "D":  2.014102,  "T":  3.016049,
    "He": 4.002603,
    "Li": 7.016003,  "Be": 9.012182,
    "B": 11.009305,  "C": 12.000000,  "N": 14.003074,
    "O": 15.994910,  "F": 18.998403,
    "Ne":19.992440,
    "Na":22.989770,  "Mg":23.985042,  "Al":26.981538,
    "Si":27.976927,  "P": 30.973762,  "S": 31.972071,
    "Cl":34.968853,  "Ar":39.962383,
    "K": 38.963707,  "Ca":39.962591,
    "Sc":44.955910,  "Ti":47.947947,  "V": 50.943964,
    "Cr":51.940512,  "Mn":54.938050,  "Fe":55.934942,
    "Co":58.933200,  "Ni":57.935348,  "Cu":62.929601,
    "Zn":63.929147,  "Ga":68.925581,  "Ge":73.921178,
    "As":74.921596,  "Se":79.916522,  "Br":78.918338,
    "Kr":83.911507,  "Rb":84.911789,  "Sr":87.905614,
    "Y": 88.905848,  "Zr":89.904704,  "Nb":92.906378,
    "Mo":97.905408,  "Tc":97.907216,  "Ru":101.904350,
    "Rh":102.905504, "Pd":105.903483, "Ag":106.905093,
    "Cd":113.903358, "In":114.903878, "Sn":119.902197,
    "Sb":120.903818, "Te":129.906223, "I": 126.904468,
    "Xe":131.904155, "Cs":132.905447, "Ba":137.905242,
    "La":138.906349,
    "Hf":179.946549, "Ta":180.947996, "W": 183.950933,
    "Re":186.955751, "Os":191.961479, "Ir":192.962924,
    "Pt":194.964774, "Au":196.966552, "Hg":201.970626,
    "U": 238.050783, "Np":237.048167, "Pu":244.064198,
    "Am":243.061373,
}


# CODATA-based conversion: sqrt(eigenvalue in Eh/(bohr²·amu)) → cm⁻¹.
# Derived from 1 Eh = 4.3597447222071e-18 J, 1 bohr = 5.29177210903e-11 m,
# 1 amu = 1.66053906660e-27 kg, c = 2.99792458e10 cm/s.
# omega² (rad/s)² = (Eh / bohr² / amu) * 9.37582976e+29
# nu_tilde (cm⁻¹) = omega / (2 pi c) = sqrt(eigenval) * 5140.48477...
_AU_TO_CM_INV = 5140.48477


def _masses_for_elements(elements: list[str]) -> list[float]:
    """Look up atomic masses (amu) for a list of element symbols."""
    out: list[float] = []
    for el in elements:
        m = _NWCHEM_DEFAULT_MASSES_AMU.get(el)
        if m is None:
            raise KeyError(
                f"No atomic mass available for element {el!r}; "
                f"supported: {sorted(_NWCHEM_DEFAULT_MASSES_AMU)}"
            )
        out.append(m)
    return out


def compute_nwchem_harmonic_frequencies(
    hessian_path: str,
    elements: list[str],
    masses_amu: list[float] | None = None,
) -> dict[str, Any]:
    """Diagonalize a mass-weighted Hessian and return vibrational frequencies.

    Useful for:
      * Verifying a `.hess` file produces the same frequencies an agent
        saw in the .out file (sanity check before reuse).
      * Re-deriving frequencies under different isotope labels without
        re-running NWChem.
      * Detecting imaginary modes when the .out file is unavailable.

    Args:
        hessian_path:  Path to the `.hess` file (ASCII lower-triangle).
        elements:      Element symbol per atom, in the SAME ORDER as the
                       geometry rows that produced the Hessian. Length
                       must equal n_atoms.
        masses_amu:    Optional explicit masses in amu, overriding the
                       built-in NWChem-default-isotope table. Length
                       must equal n_atoms.

    Returns dict with:
        hessian_file:      Resolved path.
        n_atoms / n_dof:   Matches parse_nwchem_hessian.
        elements:          Echoed back.
        masses_amu:        Used masses (resolved from table if not provided).
        eigenvalues_au:    nat3 eigenvalues in Eh/(bohr²·amu), sorted ascending.
        frequencies_cm1:   nat3 frequencies in cm⁻¹. Negative values mean
                           imaginary (sign(λ) * sqrt(|λ|) * conversion).
        n_imaginary:       Count of frequencies < -1.0 cm⁻¹ (modes
                           whose magnitude exceeds typical numerical
                           noise from the six trans/rot near-zero modes).
        n_near_zero:       Count of |frequency| <= 50 cm⁻¹ (translation,
                           rotation, and tiny numerical artifacts).
    """
    import numpy as np

    parsed = parse_nwchem_hessian(hessian_path, return_matrix=True)
    nat3 = parsed["n_dof"]
    n_atoms = parsed["n_atoms"]

    if len(elements) != n_atoms:
        raise ValueError(
            f"elements has length {len(elements)} but the Hessian dimension "
            f"implies n_atoms = {n_atoms}."
        )

    if masses_amu is None:
        masses_amu = _masses_for_elements(elements)
    elif len(masses_amu) != n_atoms:
        raise ValueError(
            f"masses_amu has length {len(masses_amu)} but n_atoms = {n_atoms}."
        )

    # Mass-weight: F_ij = H_ij / sqrt(m_i * m_j) where m_{i,j} are the
    # masses of the atoms that index i and j belong to.
    H = np.asarray(parsed["hessian"], dtype=np.float64)
    sqrt_m_per_dof = np.array(
        [masses_amu[a] ** 0.5 for a in range(n_atoms) for _ in range(3)],
        dtype=np.float64,
    )  # length nat3
    F = H / np.outer(sqrt_m_per_dof, sqrt_m_per_dof)

    # Symmetric eigenvalue decomposition (eigenvalues only — modes
    # would require the eigenvectors, defer until needed).
    eigvals = np.linalg.eigvalsh(F)

    freqs = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * _AU_TO_CM_INV
    # Conventions:
    #   |freq| <= 50 cm⁻¹       -> translation, rotation, or numerical noise
    #   freq   < -50 cm⁻¹       -> real imaginary mode (TS, broken geometry)
    #   freq   >  50 cm⁻¹       -> real vibration
    near_zero_threshold = 50.0
    imaginary_modes = [
        {"index": i, "frequency_cm1": float(f)}
        for i, f in enumerate(freqs) if f < -near_zero_threshold
    ]
    n_near_zero = int(np.sum(np.abs(freqs) <= near_zero_threshold))

    return {
        "hessian_file": parsed["hessian_file"],
        "n_atoms": n_atoms,
        "n_dof": nat3,
        "elements": list(elements),
        "masses_amu": list(masses_amu),
        "eigenvalues_au": eigvals.tolist(),
        "frequencies_cm1": freqs.tolist(),
        "imaginary_modes": imaginary_modes,
        "n_imaginary": len(imaginary_modes),
        "n_near_zero": n_near_zero,
        "near_zero_threshold_cm1": near_zero_threshold,
    }


__all__ = [
    "parse_nwchem_hessian",
    "compute_nwchem_harmonic_frequencies",
]

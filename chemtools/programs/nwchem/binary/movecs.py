"""NWChem `.movecs` (binary molecular orbital file) reader + swap.

Fortran-unformatted record I/O with automatic endian detection. The
movecs file format is documented in the NWChem source (`movecs_*.F`):

  * A sequence of Fortran-unformatted records, each framed by leading
    and trailing int32 size markers.
  * The longest run of equal-sized records (>= 3) is the MO matrix:
    occupations, eigenvalues, then nmo coefficient vectors.
  * Coefficients are stored as `nbf` doubles each in column-major order.

`parse_nwchem_movecs` reads eigenvalues + occupations (cheap inspection).
`swap_nwchem_movecs` rewrites the file with two MO records interchanged
(used to fix orbital ordering before TCE).

Moved out of `parse/tce.py` to keep the binary-format readers grouped.
The `parse/tce.py` module re-exports these names for back-compat.
"""

from __future__ import annotations
import struct
from pathlib import Path
from typing import Any

import numpy as np


def _read_fortran_records(path: str) -> tuple[list[bytearray], str]:
    """Read all Fortran unformatted records from a binary file."""
    with open(path, "rb") as f:
        raw = f.read()

    records: list[bytearray] = []
    pos = 0
    endian = "<"

    while pos < len(raw):
        if pos + 4 > len(raw):
            break
        size = struct.unpack_from(f"{endian}i", raw, pos)[0]
        if size < 0:
            size_be = struct.unpack_from(">i", raw, pos)[0]
            if 0 <= size_be <= len(raw) - pos - 8:
                endian = ">"
                size = size_be
        start = pos + 4
        end = start + size
        if end + 4 > len(raw):
            break
        end_marker = struct.unpack_from(f"{endian}i", raw, end)[0]
        if size != end_marker:
            break
        records.append(bytearray(raw[start:end]))
        pos = end + 4

    return records, endian


def _write_fortran_records(records: list[bytearray], endian: str, path: str) -> None:
    with open(path, "wb") as f:
        for rec in records:
            size = len(rec)
            marker = struct.pack(f"{endian}i", size)
            f.write(marker)
            f.write(bytes(rec))
            f.write(marker)


def _locate_mo_records(
    records: list[bytearray],
) -> tuple[int, int, int, int, int, np.ndarray]:
    """Locate occupation, eigenvalue, and MO coefficient records.

    Returns (occ_idx, eigval_idx, mo_start_idx, nmo, nbf, eigvals).
    """
    from collections import Counter

    sizes = [len(r) for r in records]
    size_counts = Counter(sizes)

    # Find the longest run of equal-size records (>= 3: occ + eigval + >=1 MO)
    best = None
    for start_idx in range(len(sizes)):
        sz = sizes[start_idx]
        if sz % 8 != 0 or sz == 0:
            continue
        count = 0
        for k in range(start_idx, len(sizes)):
            if sizes[k] == sz:
                count += 1
            else:
                break
        if count >= 3:
            if best is None or count > best[1]:
                best = (start_idx, count, sz)

    if best is None:
        raise ValueError("Could not identify MO records in movecs file.")

    run_start, run_count, rec_size = best
    occ_idx = run_start
    eigval_idx = run_start + 1
    mo_start_idx = run_start + 2
    nmo = rec_size // 8

    eigvals = np.frombuffer(records[eigval_idx], dtype="<f8")
    occs = np.frombuffer(records[occ_idx], dtype="<f8")

    # Sanity-check: occupation numbers should be 0.0 or 2.0 (or 1.0 for UHF)
    unique_occs = set(round(float(v), 5) for v in occs)
    if not unique_occs <= {0.0, 1.0, 2.0}:
        # Try swapped
        occs_try = np.frombuffer(records[eigval_idx], dtype="<f8")
        eigvals_try = np.frombuffer(records[occ_idx], dtype="<f8")
        unique_occs_try = set(round(float(v), 5) for v in occs_try)
        if unique_occs_try <= {0.0, 1.0, 2.0}:
            occ_idx, eigval_idx = eigval_idx, occ_idx
            eigvals = eigvals_try

    mo_records = records[mo_start_idx : mo_start_idx + nmo]
    nbf = len(mo_records[0]) // 8 if mo_records else nmo

    return occ_idx, eigval_idx, mo_start_idx, nmo, nbf, eigvals


def parse_nwchem_movecs(movecs_path: str) -> dict[str, Any]:
    """Read eigenvalues and occupations from a binary NWChem movecs file.

    Useful for inspecting orbital ordering without running NWChem.

    Returns a list of orbital dicts: vector_number (1-based), energy_hartree, occupancy.
    """
    records, endian = _read_fortran_records(movecs_path)
    if len(records) < 4:
        raise ValueError(f"movecs file {movecs_path} is too short or unreadable.")

    occ_idx, eigval_idx, mo_start_idx, nmo, nbf, eigvals = _locate_mo_records(records)
    occs = np.frombuffer(records[occ_idx], dtype="<f8")

    orbitals: list[dict[str, Any]] = []
    for k in range(nmo):
        orbitals.append(
            {
                "vector_number": k + 1,
                "energy_hartree": float(eigvals[k]),
                "occupancy": float(occs[k]),
                "occupied": float(occs[k]) > 0.5,
            }
        )

    occupied = [o for o in orbitals if o["occupied"]]
    virtual = [o for o in orbitals if not o["occupied"]]

    return {
        "movecs_file": str(Path(movecs_path).resolve()),
        "n_mo": nmo,
        "n_bf": nbf,
        "n_occupied": len(occupied),
        "n_virtual": len(virtual),
        "orbitals": orbitals,
    }


def swap_nwchem_movecs(
    movecs_path: str,
    i: int,
    j: int,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Swap two MOs in a binary NWChem movecs file.

    This is the key tool for fixing orbital ordering before a TCE restart.
    The RTDB is NOT modified — if the SCF was already converged, NWChem will
    skip re-running SCF and use the swapped orbitals directly.

    Parameters
    ----------
    movecs_path : str
        Path to the input movecs file.
    i, j : int
        1-based MO indices to swap.
    output_path : str or None
        Output path.  If None, overwrites the input file in-place.

    Returns
    -------
    dict with before/after eigenvalues and the path written.
    """
    out_path = output_path or movecs_path
    i0, j0 = i - 1, j - 1

    records, endian = _read_fortran_records(movecs_path)
    occ_idx, eigval_idx, mo_start_idx, nmo, nbf, eigvals = _locate_mo_records(records)

    if i0 < 0 or i0 >= nmo or j0 < 0 or j0 >= nmo:
        raise ValueError(f"MO indices {i},{j} out of range [1,{nmo}]")

    before = {"mo_i": {"index": i, "energy_hartree": float(eigvals[i0])},
              "mo_j": {"index": j, "energy_hartree": float(eigvals[j0])}}

    # Swap eigenvalues
    eigval_arr = np.frombuffer(records[eigval_idx], dtype="<f8").copy()
    eigval_arr[i0], eigval_arr[j0] = eigval_arr[j0].copy(), eigval_arr[i0].copy()
    records[eigval_idx] = bytearray(eigval_arr.tobytes())

    # Swap occupation numbers
    occ_arr = np.frombuffer(records[occ_idx], dtype="<f8").copy()
    occ_arr[i0], occ_arr[j0] = occ_arr[j0].copy(), occ_arr[i0].copy()
    records[occ_idx] = bytearray(occ_arr.tobytes())

    # Swap MO coefficient records
    mo_i_idx = mo_start_idx + i0
    mo_j_idx = mo_start_idx + j0
    records[mo_i_idx], records[mo_j_idx] = records[mo_j_idx], records[mo_i_idx]

    _write_fortran_records(records, endian, out_path)

    after_eigvals = np.frombuffer(records[eigval_idx], dtype="<f8")
    after = {"mo_i": {"index": i, "energy_hartree": float(after_eigvals[i0])},
             "mo_j": {"index": j, "energy_hartree": float(after_eigvals[j0])}}

    return {
        "written_to": str(Path(out_path).resolve()),
        "n_mo": nmo,
        "swap": {"i": i, "j": j},
        "before": before,
        "after": after,
        "note": (
            "RTDB unchanged. If the SCF was already converged and geometry/basis are "
            "unchanged, NWChem will use these swapped vectors directly for the next task."
        ),
    }

__all__ = [
    "parse_nwchem_movecs",
    "swap_nwchem_movecs",
]

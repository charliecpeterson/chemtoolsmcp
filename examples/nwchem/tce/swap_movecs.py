#!/usr/bin/env python3
"""
Swap two orbitals in an NWChem movecs file (RHF closed-shell).

NWChem movecs binary format (Fortran unformatted, little-endian):
  Records 0-8:  header info (title, basis name, nsets, nbf, nmo, ...)
  Record  9:    occupation numbers (nmo doubles)
  Record 10:    eigenvalues (nmo doubles)
  Records 11..: MO coefficient columns, one record per MO (nbf doubles each)
  Last record:  total SCF energy (2 doubles: energy + ?)

Usage:
    python3 swap_movecs.py input.movecs i j output.movecs
    python3 swap_movecs.py --inplace input.movecs i j
    python3 swap_movecs.py --parse input.movecs
"""

import sys
import struct
import numpy as np


def read_records(path):
    """Read all Fortran unformatted records from file. Returns list of bytearrays."""
    with open(path, 'rb') as f:
        raw = f.read()

    records = []
    pos = 0
    endian = '<'
    while pos < len(raw):
        if pos + 4 > len(raw):
            break
        size = struct.unpack_from(f'{endian}i', raw, pos)[0]
        if size < 0:
            size_be = struct.unpack_from('>i', raw, pos)[0]
            if 0 <= size_be <= len(raw) - pos - 8:
                endian = '>'
                size = size_be
        start = pos + 4
        end = start + size
        if end + 4 > len(raw):
            break
        end_marker = struct.unpack_from(f'{endian}i', raw, end)[0]
        if size != end_marker:
            break
        records.append(bytearray(raw[start:end]))
        pos = end + 4

    return records, endian


def write_records(records, endian, path):
    """Write list of bytearrays as Fortran unformatted records to file."""
    with open(path, 'wb') as f:
        for rec in records:
            size = len(rec)
            marker = struct.pack(f'{endian}i', size)
            f.write(marker)
            f.write(bytes(rec))
            f.write(marker)


def find_eigval_and_mo_records(records):
    """
    Identify which records contain eigenvalues and MO coefficients.

    Returns:
      eigval_idx: index in records list of the eigenvalue record
      mo_start_idx: index of the first MO coefficient record
      nmo: number of MOs
      nbf: number of basis functions
    """
    sizes = [len(r) for r in records]

    # The eigenvalue record and occupation record both have nmo*8 bytes.
    # We identify the eigenvalue record as the one whose values look like orbital energies
    # (negative, reasonable magnitude).
    # The MO coefficient records follow immediately after.

    # Find all records with the same size (likely nmo*8 = nbf*8 since nmo=nbf typically)
    # and look for a sequence of 52+ such records (one per MO).
    from collections import Counter
    size_counts = Counter(sizes)

    # The most common record size (after header) is nbf*8 (one per MO + eigenvalues + occupations)
    # Find runs of records with the same size
    candidates = []
    for start_idx in range(len(sizes)):
        sz = sizes[start_idx]
        if sz % 8 != 0 or sz == 0:
            continue
        n = sz // 8
        # Count consecutive records of this size starting here
        count = 0
        for k in range(start_idx, len(sizes)):
            if sizes[k] == sz:
                count += 1
            else:
                break
        if count >= 3:  # need at least occ + eigval + some MOs
            candidates.append((start_idx, count, sz, n))

    if not candidates:
        raise ValueError("Could not identify MO records. Check file format.")

    # Pick the candidate with the largest run
    candidates.sort(key=lambda x: -x[1])
    best = candidates[0]
    run_start, run_count, rec_size, nmo_or_nbf = best

    # Within the run: first record = occupations, second = eigenvalues, rest = MO coefficients
    # Verify: the eigenvalue record should have negative values typical of orbital energies
    occ_idx = run_start
    eigval_idx = run_start + 1
    mo_start_idx = run_start + 2

    eigvals = np.frombuffer(records[eigval_idx], dtype='<f8')
    occs = np.frombuffer(records[occ_idx], dtype='<f8')

    # Sanity check: occupations should be 0.0 or 2.0
    unique_occs = set(round(v, 6) for v in occs)
    if unique_occs <= {0.0, 2.0, 1.0}:
        pass  # looks good
    else:
        # Maybe occ and eigval are swapped
        eigvals_try = np.frombuffer(records[occ_idx], dtype='<f8')
        occs_try = np.frombuffer(records[eigval_idx], dtype='<f8')
        if set(round(v, 6) for v in occs_try) <= {0.0, 2.0, 1.0}:
            occ_idx, eigval_idx = eigval_idx, occ_idx
            eigvals = eigvals_try

    nmo = nmo_or_nbf
    # The MO records should each have nbf doubles; they might equal nmo
    mo_records = records[mo_start_idx:mo_start_idx + nmo]
    nbf = len(mo_records[0]) // 8 if mo_records else nmo

    return eigval_idx, occ_idx, mo_start_idx, nmo, nbf, eigvals


def swap_orbitals(input_path, output_path, i, j):
    """
    Swap orbitals i and j (1-indexed) in an NWChem movecs file.
    Writes result to output_path.
    """
    i0, j0 = i - 1, j - 1  # convert to 0-indexed

    records, endian = read_records(input_path)
    eigval_idx, occ_idx, mo_start_idx, nmo, nbf, eigvals = find_eigval_and_mo_records(records)

    print(f"Detected: nmo={nmo}, nbf={nbf}")
    print(f"Occupation record: index {occ_idx}")
    print(f"Eigenvalue record: index {eigval_idx}")
    print(f"MO records start:  index {mo_start_idx}")
    print(f"Total records:     {len(records)}")
    print()

    if i0 < 0 or i0 >= nmo or j0 < 0 or j0 >= nmo:
        raise ValueError(f"Orbital indices {i},{j} out of range [1,{nmo}]")

    # Show eigenvalues before swap
    print(f"Eigenvalues before swap:")
    print(f"  MO {i}: {eigvals[i0]:.6f} hartree")
    print(f"  MO {j}: {eigvals[j0]:.6f} hartree")

    # --- Swap eigenvalues ---
    eigval_arr = np.frombuffer(records[eigval_idx], dtype='<f8').copy()
    eigval_arr[i0], eigval_arr[j0] = eigval_arr[j0].copy(), eigval_arr[i0].copy()
    records[eigval_idx] = bytearray(eigval_arr.tobytes())

    # --- Swap occupation numbers ---
    occ_arr = np.frombuffer(records[occ_idx], dtype='<f8').copy()
    occ_arr[i0], occ_arr[j0] = occ_arr[j0].copy(), occ_arr[i0].copy()
    records[occ_idx] = bytearray(occ_arr.tobytes())

    # --- Swap MO coefficient records ---
    mo_i_idx = mo_start_idx + i0
    mo_j_idx = mo_start_idx + j0
    records[mo_i_idx], records[mo_j_idx] = records[mo_j_idx], records[mo_i_idx]

    # Verify
    eigval_new = np.frombuffer(records[eigval_idx], dtype='<f8')
    print(f"Eigenvalues after swap:")
    print(f"  MO {i}: {eigval_new[i0]:.6f} hartree")
    print(f"  MO {j}: {eigval_new[j0]:.6f} hartree")

    write_records(records, endian, output_path)
    print(f"\nWrote swapped movecs to: {output_path}")


def parse_movecs(path):
    """Print a summary of the movecs file structure."""
    records, endian = read_records(path)
    print(f"Endian: {endian}")
    print(f"Total records: {len(records)}")

    try:
        eigval_idx, occ_idx, mo_start_idx, nmo, nbf, eigvals = find_eigval_and_mo_records(records)
        print(f"Occupation record: {occ_idx}")
        print(f"Eigenvalue record: {eigval_idx}")
        print(f"MO records start:  {mo_start_idx}")
        print(f"nmo={nmo}, nbf={nbf}")
        print(f"\nEigenvalues (hartree):")
        for k, e in enumerate(eigvals):
            occ = np.frombuffer(records[occ_idx], dtype='<f8')[k]
            marker = " *" if occ > 0.5 else "  "
            print(f"  MO {k+1:3d}: {e:12.6f}  occ={occ:.1f}{marker}")
    except Exception as ex:
        print(f"Could not parse MO records: {ex}")

    print("\nAll records:")
    for idx, rec in enumerate(records):
        n = len(rec) // 8
        try:
            first = struct.unpack_from('<d', rec)[0]
        except Exception:
            first = None
        print(f"  [{idx:2d}] size={len(rec):6d} bytes ({n} doubles), first={first}")


def main():
    args = sys.argv[1:]

    if '--parse' in args:
        args.remove('--parse')
        parse_movecs(args[0])
        return

    inplace = '--inplace' in args
    if inplace:
        args.remove('--inplace')

    if len(args) < 3:
        print(__doc__)
        sys.exit(1)

    input_path = args[0]
    i = int(args[1])
    j = int(args[2])
    output_path = args[3] if not inplace else input_path

    swap_orbitals(input_path, output_path, i, j)


if __name__ == '__main__':
    main()

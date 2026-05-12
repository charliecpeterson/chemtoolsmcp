"""DIRAC `.mol` (geometry + basis) file parser.

The `.mol` format is positional and Fortran-ish::

    INTGRL                                  ← magic header (or DIRAC, BASIS, MRCONEE)
    Title line 1
    Title line 2
    C   N_TYPES   N_SYMOPS [Y Z X ...]  A   ← coord+symmetry+units header
            Z.   N_ATOMS                    ← atomtype block, Z=nuclear charge
    LABEL    x   y   z                       ← one atom per line
    LARGE BASIS basis_name                   ← per-atomtype basis assignment
    FINISH                                   ← terminator

Variants:
- Header line may be ``C`` (Cartesian) and contain symmetry generators
  (any of X, Y, Z) followed by ``A`` (units = angstrom; absent means au).
- Basis can also be ``LARGE N1 N2 N3 N4`` with explicit contracted segments
  rather than ``LARGE BASIS <name>``.
- ECP is signaled by ``C N_TYPES N_SYMOPS [...]  A`` with a non-zero ECP
  column count; the ECP block follows the basis block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_mol(path: str, contents: str | None = None) -> dict[str, Any]:
    """Parse a DIRAC `.mol` file.

    Returns a flat dict; geometry is in angstrom if the header carries the
    `A` units flag, otherwise bohr.
    """
    if contents is None:
        contents = Path(path).read_text(encoding="utf-8", errors="replace")

    lines = [ln.rstrip() for ln in contents.splitlines()]
    if not lines:
        raise ValueError(f"empty .mol file: {path}")

    header_kind = lines[0].strip().upper().split()[0] if lines[0].strip() else ""
    # Title is the next two lines (often blank or human-readable comments).
    title_a = lines[1].strip() if len(lines) > 1 else ""
    title_b = lines[2].strip() if len(lines) > 2 else ""

    # Skip lines until the C-coord header — the first line starting with C or c.
    coord_header_idx = None
    for i, ln in enumerate(lines[3:], start=3):
        s = ln.strip()
        if s.upper().startswith("C") and not s.upper().startswith("CC"):
            coord_header_idx = i
            break
    if coord_header_idx is None:
        raise ValueError(f"No coord header (line starting with C) in {path}")

    header_tokens = lines[coord_header_idx].split()
    # Format: C N_TYPES N_SYMOPS [sym generators...] [A]
    n_types = int(header_tokens[1])
    n_symops = int(header_tokens[2]) if len(header_tokens) > 2 else 0
    sym_generators = [
        t for t in header_tokens[3:] if t.upper() in ("X", "Y", "Z")
    ]
    units = "angstrom" if any(
        t.upper() == "A" for t in header_tokens[3:]
    ) else "bohr"

    atomtypes: list[dict[str, Any]] = []
    cursor = coord_header_idx + 1
    while cursor < len(lines):
        s = lines[cursor].strip()
        if not s:
            cursor += 1
            continue
        if s.upper() == "FINISH":
            break
        # Atom-type header: "        Z.    N_ATOMS [...]"
        parts = s.split()
        if not parts:
            cursor += 1
            continue
        try:
            z = float(parts[0].rstrip("."))
            n_atoms = int(parts[1])
        except (ValueError, IndexError):
            cursor += 1
            continue

        atomtype: dict[str, Any] = {
            "z": z,
            "n_atoms": n_atoms,
            "atoms": [],
            "large_basis": None,
            "small_basis": None,
            "ecp": None,
        }
        cursor += 1
        # Read n_atoms atom lines
        for _ in range(n_atoms):
            if cursor >= len(lines):
                break
            atom_parts = lines[cursor].split()
            if len(atom_parts) >= 4:
                atomtype["atoms"].append({
                    "label": atom_parts[0],
                    "x": float(atom_parts[1]),
                    "y": float(atom_parts[2]),
                    "z": float(atom_parts[3]),
                })
            cursor += 1
        # Read basis / ECP block(s) until next atomtype or FINISH
        while cursor < len(lines):
            s = lines[cursor].strip()
            if not s or s.upper() == "FINISH":
                break
            su = s.upper()
            # End of basis block: a line that looks like the next atomtype header
            parts2 = s.split()
            if (
                len(parts2) >= 2
                and parts2[0].rstrip(".").replace(".", "").isdigit()
                and not su.startswith(("LARGE", "SMALL", "BASIS", "EXTRA"))
            ):
                # next atomtype
                break
            if su.startswith("LARGE BASIS"):
                atomtype["large_basis"] = s.split(None, 2)[-1].strip()
                cursor += 1
                continue
            if su.startswith("SMALL BASIS"):
                atomtype["small_basis"] = s.split(None, 2)[-1].strip()
                cursor += 1
                continue
            if su.startswith("LARGE ") and atomtype["large_basis"] is None:
                # LARGE N1 N2 ... (explicit contraction segments)
                atomtype["large_basis"] = s
                cursor += 1
                # skip the per-exponent lines until we hit something non-numeric
                while cursor < len(lines):
                    nxt = lines[cursor].strip()
                    if not nxt:
                        cursor += 1
                        continue
                    if nxt.upper().startswith(("LARGE", "SMALL", "FINISH")):
                        break
                    # Numeric line — exponent / coefficients
                    try:
                        float(nxt.split()[0])
                        cursor += 1
                    except (ValueError, IndexError):
                        break
                continue
            cursor += 1

        atomtypes.append(atomtype)

    # Flatten the atom list for convenience. Use "nuclear_charge" rather
    # than "z" to avoid colliding with the z-coordinate key on each atom.
    flat_atoms: list[dict[str, Any]] = []
    for at in atomtypes:
        for a in at["atoms"]:
            flat_atoms.append({**a, "nuclear_charge": at["z"]})

    return {
        "path": str(path),
        "header_kind": header_kind,
        "title": (title_a + "\n" + title_b).strip(),
        "n_atomtypes": n_types,
        "n_symmetry_operators": n_symops,
        "symmetry_generators": sym_generators,
        "units": units,
        "atomtypes": atomtypes,
        "atoms": flat_atoms,
        "n_atoms": len(flat_atoms),
        "basis_assignments": {
            int(at["z"]): {
                "large": at["large_basis"],
                "small": at["small_basis"],
                "ecp": at["ecp"],
            }
            for at in atomtypes
        },
    }

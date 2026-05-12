"""Common helpers shared across Molcas block drafters."""

from __future__ import annotations

from typing import Iterable, Mapping


def format_per_symmetry(values: Iterable[int]) -> str:
    """Render a per-symmetry vector like (12, 1, 9, 1) as ' 12  1  9  1'."""
    return "".join(f"{int(v):>4d}" for v in values)


def normalize_atoms(atoms: list[dict]) -> list[dict]:
    """Ensure each atom has 'symbol', 'x', 'y', 'z' (numeric). Pass-through for keys."""
    out: list[dict] = []
    for a in atoms:
        sym = a.get("symbol") or a.get("element") or a.get("atom")
        if sym is None:
            raise ValueError(f"Atom missing element symbol: {a}")
        x = float(a.get("x", a.get("xyz", [0, 0, 0])[0]))
        y = float(a.get("y", a.get("xyz", [0, 0, 0])[1]))
        z = float(a.get("z", a.get("xyz", [0, 0, 0])[2]))
        out.append({"symbol": str(sym), "x": x, "y": y, "z": z, "label": a.get("label")})
    return out


def group_atoms_by_element(atoms: list[dict]) -> dict[str, list[dict]]:
    """Group atom dicts by their element symbol, preserving input order."""
    groups: dict[str, list[dict]] = {}
    for atom in atoms:
        sym = atom["symbol"]
        groups.setdefault(sym, []).append(atom)
    return groups


def auto_label(atoms: list[dict]) -> list[dict]:
    """Assign labels (C1, C2, H1, ...) per element if `label` is missing."""
    counters: dict[str, int] = {}
    out = []
    for a in atoms:
        sym = a["symbol"]
        if a.get("label"):
            out.append(a)
            continue
        counters[sym] = counters.get(sym, 0) + 1
        out.append({**a, "label": f"{sym}{counters[sym]}"})
    return out


def total_electrons(atoms: list[dict], charge: int) -> int:
    """Sum atomic numbers minus the molecular charge."""
    return sum(_atomic_number(a["symbol"]) for a in atoms) - int(charge)


_ATOMIC_NUMBERS: Mapping[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Hf": 72,
    "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
    # Lanthanides + actinides on demand
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65,
    "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Th": 90, "U": 92,
}


def _atomic_number(symbol: str) -> int:
    s = symbol.strip()
    # Normalize "H1" → "H"
    base = "".join(c for c in s if c.isalpha())
    base = base[0].upper() + base[1:].lower() if len(base) > 1 else base.upper()
    if base not in _ATOMIC_NUMBERS:
        raise ValueError(f"Unknown element symbol {symbol!r}")
    return _ATOMIC_NUMBERS[base]


def element_symbol(label: str) -> str:
    """Extract pure element symbol from a label like 'C1' or 'Pb1'."""
    base = "".join(c for c in label if c.isalpha())
    if not base:
        raise ValueError(f"Cannot extract element from label {label!r}")
    return base[0].upper() + base[1:].lower() if len(base) > 1 else base.upper()


def multiplicity_to_spin(mult: int) -> int:
    """Molcas uses Spin = 2S+1 directly (multiplicity)."""
    return int(mult)


def determine_alpha_beta(n_electrons: int, multiplicity: int) -> tuple[int, int]:
    """For ROHF/UHF: alpha = (n+S2)/2; beta = (n-S2)/2 with S2 = (mult-1)."""
    n_unpaired = multiplicity - 1
    if (n_electrons - n_unpaired) % 2 != 0:
        raise ValueError(
            f"electron count {n_electrons} and multiplicity {multiplicity} are inconsistent "
            f"({n_unpaired} unpaired electrons)"
        )
    n_beta = (n_electrons - n_unpaired) // 2
    n_alpha = n_beta + n_unpaired
    return n_alpha, n_beta

"""Program-agnostic geometry math + inspection.

All functions take a plain ``atoms`` list of dicts ``{symbol, x, y, z,
[label]}`` in Angstrom and return geometric measurements + bond/fragment
detection. No I/O, no parser dependencies — program-specific geometry
parsers (NWChem, Molcas, ...) call ``inspect_geometry()`` after
extracting their atoms list.

The Molcas-specific wrapper (``programs/molcas/strategy/geometry_inspector.py``)
adds the source-resolution step (output_file / input_file / atoms) and
unit normalization (bohr → Å), then delegates the math here. NWChem's
analogous wrapper is yet to be built (Phase 4); it will use this same
core module.
"""

from __future__ import annotations

import math
import re
from typing import Any

from chemtools.core.common import COVALENT_RADII
from chemtools.core.thermochem import ATOMIC_MASSES_AMU


def norm_element(sym: str) -> str:
    """Canonicalize an element label (strip trailing digits, capitalize)."""
    return re.sub(r"\d+$", "", sym).capitalize()


def atom_pair_radius(a_sym: str, b_sym: str) -> float | None:
    """Sum of covalent radii in Å. Returns None if either element is unknown."""
    ra = COVALENT_RADII.get(norm_element(a_sym))
    rb = COVALENT_RADII.get(norm_element(b_sym))
    if ra is None or rb is None:
        return None
    return ra + rb


def distance(a: dict, b: dict) -> float:
    """Cartesian distance between two atom dicts (assumes same units)."""
    dx = a["x"] - b["x"]; dy = a["y"] - b["y"]; dz = a["z"] - b["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def angle_deg(a: dict, b: dict, c: dict) -> float:
    """Bond angle a-b-c in degrees (b is the vertex)."""
    v1 = (a["x"] - b["x"], a["y"] - b["y"], a["z"] - b["z"])
    v2 = (c["x"] - b["x"], c["y"] - b["y"], c["z"] - b["z"])
    n1 = math.sqrt(sum(c * c for c in v1))
    n2 = math.sqrt(sum(c * c for c in v2))
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos = sum(v1[i] * v2[i] for i in range(3)) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def dihedral_deg(a: dict, b: dict, c: dict, d: dict) -> float:
    """Dihedral a-b-c-d in degrees (signed, IUPAC convention)."""
    b1 = (b["x"] - a["x"], b["y"] - a["y"], b["z"] - a["z"])
    b2 = (c["x"] - b["x"], c["y"] - b["y"], c["z"] - b["z"])
    b3 = (d["x"] - c["x"], d["y"] - c["y"], d["z"] - c["z"])

    def _cross(u, v):
        return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0])

    def _dot(u, v):
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

    def _norm(u):
        return math.sqrt(_dot(u, u))

    n1 = _cross(b1, b2)
    n2 = _cross(b2, b3)
    b2_len = _norm(b2)
    if b2_len == 0 or _norm(n1) == 0 or _norm(n2) == 0:
        return float("nan")
    m1 = _cross(n1, (b2[0] / b2_len, b2[1] / b2_len, b2[2] / b2_len))
    x = _dot(n1, n2)
    y = _dot(m1, n2)
    return math.degrees(math.atan2(y, x))


def formula(elements: list[str]) -> str:
    """Hill-system molecular formula from a list of element symbols."""
    counts: dict[str, int] = {}
    for e in elements:
        e = norm_element(e)
        counts[e] = counts.get(e, 0) + 1
    keys = sorted(counts.keys())
    ordered: list[str] = []
    if "C" in keys:
        ordered.append("C"); keys.remove("C")
        if "H" in keys:
            ordered.append("H"); keys.remove("H")
    ordered.extend(sorted(keys))
    parts: list[str] = []
    for k in ordered:
        n = counts[k]
        parts.append(k if n == 1 else f"{k}{n}")
    return "".join(parts)


def atom_label(i: int, atom: dict) -> str:
    """Display label for an atom: prefer the parsed label, else ``{symbol}{1-based-index}``."""
    return atom.get("label") or f"{norm_element(atom['symbol'])}{i + 1}"


def center_of_mass(atoms: list[dict]) -> tuple[list[float], float, list[str]]:
    """Compute center-of-mass + total mass + warnings (for unknown elements).

    Returns ``([x, y, z], total_mass_amu, warnings)`` in the same Cartesian
    units the atoms came in.
    """
    total_mass = 0.0
    com = [0.0, 0.0, 0.0]
    warnings: list[str] = []
    for a in atoms:
        el = norm_element(a["symbol"])
        m = ATOMIC_MASSES_AMU.get(el)
        if m is None:
            warnings.append(f"atomic mass missing for {el}; COM treats as 0")
            continue
        total_mass += m
        com[0] += m * a["x"]
        com[1] += m * a["y"]
        com[2] += m * a["z"]
    if total_mass > 0:
        com = [c / total_mass for c in com]
    return com, total_mass, warnings


def _union_find(n: int, edges: set[tuple[int, int]]) -> dict[int, list[int]]:
    """Group atom indices into connected fragments via the supplied edges."""
    parent = list(range(n))

    def _find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def _union(u: int, v: int) -> None:
        pu, pv = _find(u), _find(v)
        if pu != pv:
            parent[pu] = pv

    for i, j in edges:
        _union(i, j)
    fragments: dict[int, list[int]] = {}
    for i in range(n):
        root = _find(i)
        fragments.setdefault(root, []).append(i)
    return fragments


def inspect_geometry(
    atoms: list[dict],
    *,
    max_bond_length: float = 2.5,
    min_safe_distance: float = 0.6,
    covalent_tolerance: float = 1.20,
    measurements: dict[str, list[list[int]]] | None = None,
    units: str = "angstrom",
) -> dict[str, Any]:
    """Inspect an atoms list and report geometric properties.

    ``atoms`` is a list of ``{symbol, x, y, z, [label]}`` dicts assumed to
    be in ANGSTROM (callers must pre-convert from bohr if needed — see
    ``programs/molcas/strategy/geometry_inspector.py`` for an example).

    Output keys:
      - ``formula``, ``n_atoms``, ``elements``, ``units``, ``atoms``
      - ``center_of_mass``, ``total_mass_amu``
      - ``bond_lengths`` (pairs with r ≤ max_bond_length, sorted), each
        annotated ``within_covalent_sum`` (True if r ≤ covalent_tolerance ×
        Σr_cov)
      - ``close_contacts`` (pairs with r < min_safe_distance — overlap)
      - ``bond_angles`` (through atoms bonded by covalent-sum criterion)
      - ``fragments`` + ``n_fragments`` (disconnection detection via
        union-find on covalent bonds)
      - ``measurements`` (optional user-requested distances/angles/dihedrals
        by 1-based indices)
      - ``warnings``
    """
    elements = [norm_element(a["symbol"]) for a in atoms]
    warnings: list[str] = []

    com, total_mass, com_warnings = center_of_mass(atoms)
    warnings.extend(com_warnings)

    n = len(atoms)
    bonds: list[dict[str, Any]] = []
    close: list[dict[str, Any]] = []
    bonded_pairs: set[tuple[int, int]] = set()

    for i in range(n):
        for j in range(i + 1, n):
            r = distance(atoms[i], atoms[j])
            pair_radius = atom_pair_radius(elements[i], elements[j])
            within_covalent = (
                pair_radius is not None and r <= covalent_tolerance * pair_radius
            )
            if r < min_safe_distance:
                close.append({
                    "atoms": [atom_label(i, atoms[i]), atom_label(j, atoms[j])],
                    "atom_indices": [i + 1, j + 1],
                    "length": round(r, 4),
                    "units": units,
                })
                continue
            if r <= max_bond_length:
                bonds.append({
                    "atoms": [atom_label(i, atoms[i]), atom_label(j, atoms[j])],
                    "atom_indices": [i + 1, j + 1],
                    "length": round(r, 4),
                    "units": units,
                    "within_covalent_sum": within_covalent,
                })
                if within_covalent:
                    bonded_pairs.add((i, j))
    bonds.sort(key=lambda b: b["length"])

    # Bond angles through bonded triples
    neighbors: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, j in bonded_pairs:
        neighbors[i].append(j)
        neighbors[j].append(i)
    angles: list[dict[str, Any]] = []
    for j in range(n):
        nbrs = sorted(neighbors[j])
        for x in range(len(nbrs)):
            for y in range(x + 1, len(nbrs)):
                i, k = nbrs[x], nbrs[y]
                angles.append({
                    "atoms": [atom_label(i, atoms[i]), atom_label(j, atoms[j]),
                              atom_label(k, atoms[k])],
                    "atom_indices": [i + 1, j + 1, k + 1],
                    "angle_deg": round(angle_deg(atoms[i], atoms[j], atoms[k]), 2),
                })

    # User-specified measurements (1-based)
    measurement_results: dict[str, list[dict[str, Any]]] = {}
    if measurements:
        if measurements.get("distances"):
            measurement_results["distances"] = []
            for spec in measurements["distances"]:
                if len(spec) != 2:
                    warnings.append(f"distance spec {spec!r}: needs 2 indices")
                    continue
                i, j = int(spec[0]) - 1, int(spec[1]) - 1
                if not (0 <= i < n and 0 <= j < n):
                    warnings.append(f"distance spec {spec!r}: index out of range")
                    continue
                measurement_results["distances"].append({
                    "atoms": [atom_label(i, atoms[i]), atom_label(j, atoms[j])],
                    "atom_indices": [i + 1, j + 1],
                    "length": round(distance(atoms[i], atoms[j]), 4),
                    "units": units,
                })
        if measurements.get("angles"):
            measurement_results["angles"] = []
            for spec in measurements["angles"]:
                if len(spec) != 3:
                    warnings.append(f"angle spec {spec!r}: needs 3 indices")
                    continue
                i, j, k = int(spec[0]) - 1, int(spec[1]) - 1, int(spec[2]) - 1
                if not all(0 <= idx < n for idx in (i, j, k)):
                    warnings.append(f"angle spec {spec!r}: index out of range")
                    continue
                measurement_results["angles"].append({
                    "atoms": [atom_label(i, atoms[i]), atom_label(j, atoms[j]),
                              atom_label(k, atoms[k])],
                    "atom_indices": [i + 1, j + 1, k + 1],
                    "angle_deg": round(angle_deg(atoms[i], atoms[j], atoms[k]), 2),
                })
        if measurements.get("dihedrals"):
            measurement_results["dihedrals"] = []
            for spec in measurements["dihedrals"]:
                if len(spec) != 4:
                    warnings.append(f"dihedral spec {spec!r}: needs 4 indices")
                    continue
                i, j, k, l = (int(s) - 1 for s in spec)
                if not all(0 <= idx < n for idx in (i, j, k, l)):
                    warnings.append(f"dihedral spec {spec!r}: index out of range")
                    continue
                measurement_results["dihedrals"].append({
                    "atoms": [atom_label(idx, atoms[idx]) for idx in (i, j, k, l)],
                    "atom_indices": [i + 1, j + 1, k + 1, l + 1],
                    "dihedral_deg": round(dihedral_deg(atoms[i], atoms[j], atoms[k], atoms[l]), 2),
                })

    fragments = _union_find(n, bonded_pairs)
    fragment_list = sorted(
        ([atom_label(i, atoms[i]) for i in members] for members in fragments.values()),
        key=lambda f: (-len(f), f[0]),
    )

    if close:
        warnings.append(
            f"{len(close)} close contact(s) below {min_safe_distance} {units} — "
            "geometry may be broken (overlapping atoms)."
        )
    if len(fragment_list) > 1:
        warnings.append(
            f"Molecule is disconnected into {len(fragment_list)} fragments "
            "under the covalent-sum bond criterion."
        )

    return {
        "formula": formula(elements),
        "n_atoms": n,
        "elements": sorted(set(elements)),
        "units": units,
        "atoms": [
            {"index": i + 1, "symbol": elements[i], "label": atom_label(i, atoms[i]),
             "x": atoms[i]["x"], "y": atoms[i]["y"], "z": atoms[i]["z"]}
            for i in range(n)
        ],
        "center_of_mass": [round(c, 6) for c in com],
        "total_mass_amu": round(total_mass, 4),
        "bond_lengths": bonds,
        "n_bonds_within_covalent_sum": len(bonded_pairs),
        "close_contacts": close,
        "bond_angles": angles,
        "measurements": measurement_results,
        "fragments": fragment_list,
        "n_fragments": len(fragment_list),
        "warnings": warnings,
    }

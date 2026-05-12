"""Geometry inspection for Molcas outputs / inputs.

`inspect_geometry` accepts a geometry from one of three sources:
  - output_file: pulls the final converged geometry via parse_final_geometry
  - input_file:  reads the &SEWARD/&GATEWAY block (basis-keyed atom lines)
  - atoms:       explicit list of {symbol, x, y, z} dicts

and returns:
  - formula              "CrO", "H2O", etc.
  - n_atoms, elements
  - center_of_mass       [x, y, z] in input units
  - bond_lengths         all pairs with r ≤ max_bond_length (default 2.5 Å)
                          ranked by distance, each annotated with whether
                          it falls within the covalent-radius-sum window
  - close_contacts       pairs with r < min_safe_distance (default 0.6 Å) —
                          suggests overlap / bad starting geometry
  - bond_angles          all valid 3-atom triplets (i-j-k where both i-j
                          and j-k are bonded by covalent-sum criterion)
  - measurements         optional user-specified {distances, angles, dihedrals}
                          measured exactly as asked, by 1-based atom indices
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from chemtools.core.common import read_text, COVALENT_RADII
from chemtools.programs.molcas.parse.geometry import parse_final_geometry
from chemtools.programs.molcas.strategy.reaction_energy import _ATOMIC_MASSES_AMU


_BOHR_PER_ANGSTROM = 1.8897261245650618
_ANGSTROM_PER_BOHR = 1.0 / _BOHR_PER_ANGSTROM


def _norm_element(sym: str) -> str:
    """Canonicalize an element label (strip digits, capitalize)."""
    return re.sub(r"\d+$", "", sym).capitalize()


def _atom_pair_radius(a_sym: str, b_sym: str) -> float | None:
    """Sum of covalent radii (Å). Returns None if either element is unknown."""
    ra = COVALENT_RADII.get(_norm_element(a_sym))
    rb = COVALENT_RADII.get(_norm_element(b_sym))
    if ra is None or rb is None:
        return None
    return ra + rb


def _distance(a: dict, b: dict) -> float:
    dx = a["x"] - b["x"]; dy = a["y"] - b["y"]; dz = a["z"] - b["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _angle_deg(a: dict, b: dict, c: dict) -> float:
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


def _dihedral_deg(a: dict, b: dict, c: dict, d: dict) -> float:
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


def _formula(elements: list[str]) -> str:
    """Hill-system molecular formula from a list of element symbols."""
    counts: dict[str, int] = {}
    for e in elements:
        e = _norm_element(e)
        counts[e] = counts.get(e, 0) + 1
    # Hill order: C first, H second, then alphabetical
    keys = sorted(counts.keys())
    ordered: list[str] = []
    if "C" in keys:
        ordered.append("C")
        keys.remove("C")
        if "H" in keys:
            ordered.append("H")
            keys.remove("H")
    ordered.extend(sorted(keys))
    parts: list[str] = []
    for k in ordered:
        n = counts[k]
        parts.append(k if n == 1 else f"{k}{n}")
    return "".join(parts)


def _extract_atoms_from_input(text: str) -> list[dict] | None:
    """Best-effort: pull (symbol, x, y, z) atom rows from a Molcas input file's
    SEWARD/GATEWAY block. Looks for lines like::

        H1   0.0000000000   0.0000000000   0.0000000000

    inside a `Basis set` ... `End of basis` block.
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
        # An atom row: starts with a label (1-3 alpha + optional digit), then 3 floats
        m = re.match(
            r"^([A-Z][a-z]?\d*)\s+([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)\s+"
            r"([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)\s+([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)\s*$",
            line,
        )
        if m:
            label = m.group(1)
            sym = _norm_element(label)
            atoms.append({
                "symbol": sym,
                "label": label,
                "x": float(m.group(2)),
                "y": float(m.group(3)),
                "z": float(m.group(4)),
            })
    return atoms or None


def _label(i: int, atom: dict) -> str:
    """Display label for an atom: prefer the parsed label, fall back to
    `{symbol}{1-based-index}`."""
    return atom.get("label") or f"{_norm_element(atom['symbol'])}{i + 1}"


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
    """Inspect a Molcas geometry and report bond lengths / angles / contacts.

    Source priority: output_file > input_file > atoms. Exactly one must be
    provided.

    Parameters
    ----------
    max_bond_length
        Upper distance (Å) for the bond_lengths report. Default 2.5 Å —
        catches all chemical bonds + weak interactions.
    min_safe_distance
        Lower threshold (Å) flagged as a close_contact (likely overlap /
        broken geometry). Default 0.6 Å.
    covalent_tolerance
        A pair is treated as "bonded" for angle enumeration if its distance
        ≤ tolerance × (covalent_radius_a + covalent_radius_b). Default 1.20
        (matches standard Mercury / Avogadro bond detection).
    measurements
        Optional explicit measurements to compute, keyed by type:
          {"distances": [[i, j], ...],
           "angles":    [[i, j, k], ...],
           "dihedrals": [[i, j, k, l], ...]}
        All indices are 1-based against the atom list.

    Returns dict with formula, n_atoms, elements, center_of_mass, bond_lengths,
    close_contacts, bond_angles, measurements (if requested), warnings.
    """
    # ---- Resolve source ----
    n_sources = sum(x is not None for x in (output_file, input_file, atoms))
    if n_sources != 1:
        raise ValueError(
            "Pass exactly one of output_file / input_file / atoms."
        )

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
    # else: atoms passed directly — assume angstrom

    # Normalize all coordinates to Angstrom internally so bond detection
    # against COVALENT_RADII (Å) works regardless of input units.
    if source_units.lower() == "bohr":
        atoms = [
            {**a, "x": a["x"] * _ANGSTROM_PER_BOHR,
                  "y": a["y"] * _ANGSTROM_PER_BOHR,
                  "z": a["z"] * _ANGSTROM_PER_BOHR}
            for a in atoms
        ]
    units = "angstrom"

    # ---- Basic properties ----
    elements = [_norm_element(a["symbol"]) for a in atoms]
    formula = _formula(elements)
    warnings: list[str] = []

    # Center of mass
    total_mass = 0.0
    com = [0.0, 0.0, 0.0]
    for el, a in zip(elements, atoms):
        m = _ATOMIC_MASSES_AMU.get(el)
        if m is None:
            warnings.append(f"atomic mass missing for {el}; COM treats as 0")
            continue
        total_mass += m
        com[0] += m * a["x"]
        com[1] += m * a["y"]
        com[2] += m * a["z"]
    if total_mass > 0:
        com = [c / total_mass for c in com]

    # ---- Pair-wise distances ----
    n = len(atoms)
    bonds: list[dict] = []
    close: list[dict] = []
    bonded_pairs: set[tuple[int, int]] = set()

    for i in range(n):
        for j in range(i + 1, n):
            r = _distance(atoms[i], atoms[j])
            pair_radius = _atom_pair_radius(elements[i], elements[j])
            within_covalent = (
                pair_radius is not None and r <= covalent_tolerance * pair_radius
            )
            if r < min_safe_distance:
                close.append({
                    "atoms": [_label(i, atoms[i]), _label(j, atoms[j])],
                    "atom_indices": [i + 1, j + 1],
                    "length": round(r, 4),
                    "units": units,
                })
                continue  # skip from bond_lengths — separate report
            if r <= max_bond_length:
                bonds.append({
                    "atoms": [_label(i, atoms[i]), _label(j, atoms[j])],
                    "atom_indices": [i + 1, j + 1],
                    "length": round(r, 4),
                    "units": units,
                    "within_covalent_sum": within_covalent,
                })
                if within_covalent:
                    bonded_pairs.add((i, j))

    bonds.sort(key=lambda b: b["length"])

    # ---- Bond angles (through bonded triples) ----
    # For each atom j, find all neighbors i, k; angle i-j-k.
    neighbors: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, j in bonded_pairs:
        neighbors[i].append(j)
        neighbors[j].append(i)

    angles: list[dict] = []
    for j in range(n):
        nbrs = sorted(neighbors[j])
        for x in range(len(nbrs)):
            for y in range(x + 1, len(nbrs)):
                i, k = nbrs[x], nbrs[y]
                ang = _angle_deg(atoms[i], atoms[j], atoms[k])
                angles.append({
                    "atoms": [_label(i, atoms[i]), _label(j, atoms[j]), _label(k, atoms[k])],
                    "atom_indices": [i + 1, j + 1, k + 1],
                    "angle_deg": round(ang, 2),
                })

    # ---- Requested measurements ----
    measurement_results: dict[str, list[dict]] = {}
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
                    "atoms": [_label(i, atoms[i]), _label(j, atoms[j])],
                    "atom_indices": [i + 1, j + 1],
                    "length": round(_distance(atoms[i], atoms[j]), 4),
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
                    "atoms": [_label(i, atoms[i]), _label(j, atoms[j]), _label(k, atoms[k])],
                    "atom_indices": [i + 1, j + 1, k + 1],
                    "angle_deg": round(_angle_deg(atoms[i], atoms[j], atoms[k]), 2),
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
                    "atoms": [_label(idx_pair[0], atoms[idx_pair[0]])
                              for idx_pair in [(i,), (j,), (k,), (l,)]],
                    "atom_indices": [i + 1, j + 1, k + 1, l + 1],
                    "dihedral_deg": round(_dihedral_deg(atoms[i], atoms[j], atoms[k], atoms[l]), 2),
                })

    # ---- Connectivity diagnostics ----
    # Find connected fragments via BFS on bonded_pairs
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

    for i, j in bonded_pairs:
        _union(i, j)
    fragments: dict[int, list[int]] = {}
    for i in range(n):
        root = _find(i)
        fragments.setdefault(root, []).append(i)
    fragment_list = sorted(
        ([_label(i, atoms[i]) for i in members] for members in fragments.values()),
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
        "formula": formula,
        "n_atoms": n,
        "elements": sorted(set(elements)),
        "units": units,
        "atoms": [
            {"index": i + 1, "symbol": elements[i], "label": _label(i, atoms[i]),
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

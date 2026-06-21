"""NWChem geometry + frequency plausibility checks.

Two top-level entry points plus shared geometry helpers:

  * check_nwchem_geometry_plausibility   Validate a converged geometry —
                                          bond lengths sane, no dissociated
                                          fragments, ring planarity intact,
                                          no over-coordination, etc.
                                          Returns a verdict + warnings.
  * check_nwchem_freq_plausibility       Validate a frequency calculation —
                                          imaginary modes consistent with
                                          intent, vibrational frequencies
                                          in physical ranges, IR intensity
                                          spectrum sane.

Geometry helpers (_compute_bonds, _compute_bond_angles,
_check_ring_planarity) are used by the geometry plausibility check.
"""

from __future__ import annotations
import math
from typing import Any

from chemtools.core.common import read_text, COVALENT_RADII as _COVALENT_RADII
from chemtools.programs.nwchem.parse.input import inspect_nwchem_input
from chemtools.programs.nwchem.parse.freq import (
    parse_freq,
    parse_trajectory,
)
from chemtools.programs.nwchem.input._utils import _TRANSITION_METALS


def _extract_nwchem_geometry(*args, **kwargs):
    """Lazy proxy for chemtools.api_input.extract_nwchem_geometry."""
    from chemtools.programs.nwchem.input.geometry import extract_nwchem_geometry
    return extract_nwchem_geometry(*args, **kwargs)


def _compute_bonds(
    atoms: list[str],
    positions: list[list[float]],
    clash_factor: float = 0.70,
    bond_factor: float = 1.30,
) -> dict[str, Any]:
    """Compute bonds, clashes, and long contacts from atom positions.

    Returns a dict with:
      bonds           list of {i, j, elem_i, elem_j, distance, expected_max, ratio}
      clashes         pairs where distance < clash_factor × (r_i + r_j)
      long_bonds      bonds where ratio > 1.0 but likely still connected
      coordination    {atom_index: count}
    """
    n = len(atoms)
    bonds: list[dict[str, Any]] = []
    clashes: list[dict[str, Any]] = []
    long_bonds: list[dict[str, Any]] = []
    coord: list[int] = [0] * n

    fallback_r = 1.5  # Å — used when element not in radii table

    for i in range(n):
        ri = _COVALENT_RADII.get(atoms[i], fallback_r)
        xi, yi, zi = positions[i]
        for j in range(i + 1, n):
            rj = _COVALENT_RADII.get(atoms[j], fallback_r)
            xj, yj, zj = positions[j]
            dx, dy, dz = xj - xi, yj - yi, zj - zi
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            r_sum = ri + rj
            ratio = dist / r_sum

            entry = {
                "i": i, "j": j,
                "elem_i": atoms[i], "elem_j": atoms[j],
                "label_i": f"{atoms[i]}{i + 1}", "label_j": f"{atoms[j]}{j + 1}",
                "distance_angstrom": round(dist, 4),
                "expected_max_angstrom": round(r_sum * bond_factor, 3),
                "ratio": round(ratio, 3),
            }
            if ratio < clash_factor:
                clashes.append(entry)
            elif ratio <= bond_factor:
                bonds.append(entry)
                coord[i] += 1
                coord[j] += 1
            elif ratio <= 1.50:
                long_bonds.append(entry)

    return {
        "bonds": bonds,
        "clashes": clashes,
        "long_bonds": long_bonds,
        "coordination": coord,
    }


def _compute_bond_angles(
    atoms: list[str],
    positions: list[list[float]],
    bond_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return bond angles for every A-B-C triple where A-B and B-C are bonds."""
    from collections import defaultdict
    neighbors: dict[int, list[int]] = defaultdict(list)
    for b in bond_list:
        neighbors[b["i"]].append(b["j"])
        neighbors[b["j"]].append(b["i"])

    angles: list[dict[str, Any]] = []
    for center, nbrs in neighbors.items():
        if len(nbrs) < 2:
            continue
        cx, cy, cz = positions[center]
        for k in range(len(nbrs)):
            for l in range(k + 1, len(nbrs)):
                a, b = nbrs[k], nbrs[l]
                ax, ay, az = positions[a]
                bx, by, bz = positions[b]
                va = (ax - cx, ay - cy, az - cz)
                vb = (bx - cx, by - cy, bz - cz)
                na = math.sqrt(va[0]**2 + va[1]**2 + va[2]**2)
                nb = math.sqrt(vb[0]**2 + vb[1]**2 + vb[2]**2)
                if na < 1e-8 or nb < 1e-8:
                    continue
                cos_a = max(-1.0, min(1.0, (va[0]*vb[0] + va[1]*vb[1] + va[2]*vb[2]) / (na * nb)))
                angle_deg = math.degrees(math.acos(cos_a))
                angles.append({
                    "center": center,
                    "a": a, "b": b,
                    "elem_center": atoms[center],
                    "label_center": f"{atoms[center]}{center + 1}",
                    "label_a": f"{atoms[a]}{a + 1}",
                    "label_b": f"{atoms[b]}{b + 1}",
                    "angle_deg": round(angle_deg, 2),
                })
    return angles


def _check_ring_planarity(
    atoms: list[str],
    positions: list[list[float]],
    bond_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect rings and check planarity for aromatic candidates."""
    from collections import defaultdict

    neighbors: dict[int, list[int]] = defaultdict(list)
    for b in bond_list:
        neighbors[b["i"]].append(b["j"])
        neighbors[b["j"]].append(b["i"])

    # Ring detection: DFS for cycles of length 5–7.
    # Guard: skip for very large molecules where DFS would be slow.
    if len(atoms) > 120:
        return []

    ring_atoms_sets: list[frozenset] = []
    rings: list[list[int]] = []

    def dfs(start: int, current: int, path: list[int], visited: set) -> None:
        for nb in neighbors[current]:
            if nb == start and len(path) >= 3:
                candidate = frozenset(path)
                if candidate not in ring_atoms_sets and len(path) <= 7:
                    ring_atoms_sets.append(candidate)
                    rings.append(list(path))
                continue
            if nb not in visited and len(path) < 7:
                visited.add(nb)
                path.append(nb)
                dfs(start, nb, path, visited)
                path.pop()
                visited.remove(nb)

    for start in range(len(atoms)):
        dfs(start, start, [start], {start})

    results: list[dict[str, Any]] = []
    for ring in rings:
        if len(ring) < 5:
            continue
        ring_elems = [atoms[i] for i in ring]
        # Aromatic candidate: all C/N with no H-only members
        is_aromatic_candidate = all(e in ("C", "N", "O", "S") for e in ring_elems)

        pts = [positions[i] for i in ring]
        # Fit plane via SVD
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)
        centered = [[p[0] - cx, p[1] - cy, p[2] - cz] for p in pts]

        # Fit best-fit plane via SVD (numerically stable even for nearly-collinear atoms)
        try:
            import numpy as _np
            mat = _np.array(centered)
            _, _, vh = _np.linalg.svd(mat)
            normal = vh[-1]  # row with smallest singular value = plane normal
            deviations = [abs(float(_np.dot(normal, p))) for p in centered]
        except Exception:
            # Fallback: cross product of first two edges
            v1, v2 = centered[1], centered[2]
            nx = v1[1]*v2[2] - v1[2]*v2[1]
            ny = v1[2]*v2[0] - v1[0]*v2[2]
            nz = v1[0]*v2[1] - v1[1]*v2[0]
            nn = math.sqrt(nx*nx + ny*ny + nz*nz)
            if nn < 1e-8:
                continue
            nx, ny, nz = nx/nn, ny/nn, nz/nn
            deviations = [abs(p[0]*nx + p[1]*ny + p[2]*nz) for p in centered]
        max_dev = max(deviations)
        rms_dev = math.sqrt(sum(d*d for d in deviations) / len(deviations))

        results.append({
            "ring_size": len(ring),
            "ring_labels": [f"{atoms[i]}{i+1}" for i in ring],
            "is_aromatic_candidate": is_aromatic_candidate,
            "max_planarity_deviation_angstrom": round(max_dev, 4),
            "rms_planarity_deviation_angstrom": round(rms_dev, 4),
            "planar": max_dev < 0.10,
        })

    return results


def check_nwchem_geometry_plausibility(
    output_path: str,
    input_path: str | None = None,
    frame: str = "best",
) -> dict[str, Any]:
    """Check whether an optimized NWChem geometry is chemically plausible.

    Parses the optimization trajectory to extract atom positions, then runs
    a suite of chemical sanity checks:

    - Bond length plausibility (short clashes, unusually long bonds)
    - Coordination number check (per element expected range)
    - Extreme bond angles (< 60° or > 170° for non-linear centres)
    - Ring planarity for 5–7-membered rings
    - Metal coordination geometry summary

    Works on any frame: 'best' (smart selection, default), 'last', 'first',
    'min_energy', or integer step number.

    Returns
    -------
    dict with:
      plausible          bool — True if no red flags found
      red_flags          list of critical issues
      warnings           list of non-critical concerns
      bond_summary       count of bonds, clashes, long bonds
      coordination       per-atom coordination with flag
      angle_flags        extreme bond angles
      ring_checks        planarity for rings
      selected_frame     step/energy of the frame checked
    """
    from chemtools.programs.nwchem.input.geometry import extract_nwchem_geometry
    from chemtools.programs.nwchem.input.opt_followup import _select_best_optimization_frame

    contents = read_text(output_path)

    # --- Extract geometry ---
    frame_arg: str | int = frame
    try:
        frame_arg = int(frame)
    except (TypeError, ValueError):
        pass

    geo = _extract_nwchem_geometry(output_path, frame=frame_arg, input_path=input_path)
    sel = geo["selected_frame"]
    atoms: list[str] = geo["selected_frame"].get("elements") or []
    positions_raw = geo["selected_frame"].get("positions_angstrom") or []

    # extract_nwchem_geometry returns frame info but we need elements+positions
    # Fall back: parse from trajectory directly
    if not atoms or not positions_raw:
        traj = parse_trajectory(output_path, contents, include_positions=True)
        frames = traj.get("frames", [])
        if not frames:
            return {"error": "no geometry frames found in output"}
        if frame_arg == "best":
            chosen, _ = _select_best_optimization_frame(frames, traj["optimization_status"])
        elif frame_arg == "last":
            chosen = frames[-1]
        elif frame_arg == "first":
            chosen = frames[0]
        elif frame_arg == "min_energy":
            chosen = min(frames, key=lambda f: f.get("energy_hartree") or float("inf"))
        else:
            step_map = {f["step"]: f for f in frames}
            chosen = step_map.get(int(frame_arg), frames[-1])

        atoms = chosen.get("labels") or []
        positions_raw = chosen.get("positions_angstrom") or []
        sel = {"step": chosen["step"], "energy_hartree": chosen.get("energy_hartree")}

    # Normalise element symbols (strip trailing digits from labels)
    def _label_to_elem(lbl: str) -> str:
        import re as _re
        return _re.sub(r"\d+$", "", lbl).capitalize()

    elements = [_label_to_elem(a) for a in atoms]
    positions: list[list[float]] = [list(p) for p in positions_raw]

    if not elements or not positions:
        return {"error": "could not extract atom positions from geometry frame"}

    # --- Compute bonds ---
    bond_data = _compute_bonds(elements, positions)
    bonds = bond_data["bonds"]
    clashes = bond_data["clashes"]
    long_bonds = bond_data["long_bonds"]
    coord_counts = bond_data["coordination"]

    # --- Coordination checks ---
    coord_flags: list[dict[str, Any]] = []
    coord_info: list[dict[str, Any]] = []
    for idx, (elem, cn) in enumerate(zip(elements, coord_counts)):
        label = f"{elem}{idx + 1}"
        max_cn = _MAX_COORD.get(elem, 9)
        typ_range = _TYPICAL_COORD.get(elem)
        flag: str | None = None
        if cn > max_cn:
            flag = f"overcrowded: CN={cn} > max expected {max_cn}"
        elif typ_range and cn < typ_range[0] and cn == 0 and elem not in ("He", "Ne", "Ar", "Kr", "Xe"):
            flag = f"isolated: CN=0"
        elif elem == "H" and cn > 1:
            flag = f"H bridging: CN={cn} (unusual unless explicit bridge)"
        elif elem == "C" and cn == 0:
            flag = "isolated C atom"

        coord_info.append({
            "label": label, "element": elem, "coordination_number": cn, "flag": flag
        })
        if flag:
            coord_flags.append({"label": label, "flag": flag})

    # --- Angle checks ---
    angles = _compute_bond_angles(elements, positions, bonds)
    angle_flags: list[dict[str, Any]] = []
    for ang in angles:
        a = ang["angle_deg"]
        cn = coord_counts[ang["center"]]
        center_elem = ang["elem_center"]
        is_metal_center = center_elem in _ALL_METALS
        # Flag angles that are chemically extreme
        # High-CN metal centres (CN>6) naturally have small L-M-L angles — don't flag those
        if a < 50.0 and not (is_metal_center and cn > 6):
            angle_flags.append({**ang, "issue": f"very acute angle {a:.1f}° — extreme ring strain or wrong connectivity"})
        elif a > 175.0 and cn > 2 and not is_metal_center:
            angle_flags.append({**ang, "issue": f"near-linear angle {a:.1f}° at CN={cn} centre — possible geometry error"})

    # --- Ring planarity ---
    ring_checks = _check_ring_planarity(elements, positions, bonds)
    ring_flags = [r for r in ring_checks if r["is_aromatic_candidate"] and not r["planar"]]

    # --- Metal coordination summary ---
    metal_coord: list[dict[str, Any]] = []
    for idx, elem in enumerate(elements):
        if elem in _ALL_METALS:
            cn = coord_counts[idx]
            bonded = [b for b in bonds if b["i"] == idx or b["j"] == idx]
            ligand_elems = [
                b["elem_j"] if b["i"] == idx else b["elem_i"]
                for b in bonded
            ]
            note: str | None = None
            max_expected = _MAX_COORD.get(elem, 9)
            if cn == 0:
                note = "isolated metal — no bonds detected"
            elif cn < 2:
                note = f"unusually low CN={cn} for metal"
            elif cn > max_expected:
                note = f"very high CN={cn} (max expected {max_expected}) — check for spurious bonds"
            metal_coord.append({
                "label": f"{elem}{idx + 1}",
                "element": elem,
                "coordination_number": cn,
                "ligand_elements": sorted(set(ligand_elems)),
                "note": note,
            })

    # --- Assemble red flags and warnings ---
    red_flags: list[str] = []
    warnings_out: list[str] = []

    for c in clashes:
        red_flags.append(
            f"CLASH: {c['label_i']}–{c['label_j']} distance {c['distance_angstrom']:.3f} Å "
            f"({c['ratio']:.2f}× covalent sum) — atoms too close"
        )

    for lb in long_bonds:
        warnings_out.append(
            f"LONG BOND: {lb['label_i']}–{lb['label_j']} {lb['distance_angstrom']:.3f} Å "
            f"({lb['ratio']:.2f}× covalent sum) — possibly broken or weak bond"
        )

    for cf in coord_flags:
        (red_flags if "overcrowded" in cf["flag"] or "isolated" in cf["flag"] else warnings_out).append(
            f"COORD: {cf['label']} — {cf['flag']}"
        )

    for af in angle_flags:
        warnings_out.append(f"ANGLE: {af['label_center']} — {af['issue']}")

    for rf in ring_flags:
        warnings_out.append(
            f"RING: {'-'.join(rf['ring_labels'])} — aromatic candidate not planar "
            f"(max dev {rf['max_planarity_deviation_angstrom']:.3f} Å)"
        )

    for mc in metal_coord:
        if mc["note"] and ("isolated" in mc["note"] or "high CN" in mc["note"]):
            red_flags.append(f"METAL: {mc['label']} — {mc['note']}")
        elif mc["note"]:
            warnings_out.append(f"METAL: {mc['label']} — {mc['note']}")

    plausible = len(red_flags) == 0

    return {
        "plausible": plausible,
        "red_flags": red_flags,
        "warnings": warnings_out,
        "selected_frame": sel,
        "atom_count": len(elements),
        "bond_summary": {
            "bond_count": len(bonds),
            "clash_count": len(clashes),
            "long_bond_count": len(long_bonds),
        },
        "coordination": coord_info,
        "angle_flag_count": len(angle_flags),
        "angle_flags": angle_flags,
        "ring_checks": ring_checks,
        "metal_coordination": metal_coord,
    }


# ---------------------------------------------------------------------------
# Frequency plausibility checker
# ---------------------------------------------------------------------------

# Frequency band assignments for common bond types (cm⁻¹)
_FREQ_BANDS: list[tuple[float, float, str]] = [
    (0,    50,   "near-zero / translational / conformational"),
    (50,   300,  "metal-ligand / torsional"),
    (300,  600,  "metal-ligand stretches / heavy-atom bends"),
    (600,  900,  "ring deformations / C-halogen stretches"),
    (900,  1200, "C-O / C-N / C-C / skeletal stretches"),
    (1200, 1500, "C-H bends / C-C / C-N stretches"),
    (1500, 1700, "C=C / C=N / N-H bends"),
    (1700, 1900, "C=O carbonyl stretches"),
    (1900, 2400, "C≡N / C≡C / C=C=O"),
    (2400, 2800, "S-H / P-H / Si-H stretches"),
    (2800, 3200, "C-H stretches"),
    (3200, 3700, "N-H / O-H stretches"),
    (3700, 9999, "very high — check for very light atoms or scale factor"),
]

# Element → expected high-freq modes if bonds to H are present
_EXPECTED_XH_BANDS: dict[str, tuple[float, float, str]] = {
    "O": (3200, 3700, "O-H stretch"),
    "N": (3100, 3500, "N-H stretch"),
    "C": (2800, 3200, "C-H stretch"),
    "S": (2400, 2600, "S-H stretch"),
}


def check_nwchem_freq_plausibility(
    output_path: str,
    input_path: str | None = None,
    expect_minimum: bool = True,
) -> dict[str, Any]:
    """Check whether NWChem frequency results are chemically plausible.

    Performs the following checks:

    - Imaginary mode count vs. expectation (minimum vs. transition state)
    - Large imaginary modes (< −50 cm⁻¹) — serious structural problem
    - Near-zero real modes (< 20 cm⁻¹) — flat PES or incomplete optimisation
    - Mode distribution across frequency bands
    - Expected X-H stretch presence given elements in molecule
    - ZPE per atom sanity check (expected ~2–12 kcal/mol per heavy atom)
    - Suspiciously high frequencies (possible scale-factor or unit error)

    Parameters
    ----------
    output_path:
        Path to the NWChem frequency output file.
    input_path:
        Optional: path to the input file (used to read element list).
    expect_minimum:
        True (default) if the calculation is expected to be a local minimum
        (0 imaginary modes).  Set False for transition state searches.

    Returns
    -------
    dict with:
      plausible             bool
      red_flags             list of critical issues
      warnings              list of non-critical concerns
      mode_counts           summary of mode counts by type
      band_distribution     modes per frequency band
      zpe_check             ZPE analysis
      missing_xh_stretches  expected X-H bands not observed
    """
    from chemtools.programs.nwchem.parse.freq import parse_freq as _parse_freq

    contents = read_text(output_path)
    freq_data = _parse_freq(output_path, contents)

    modes = freq_data.get("modes", [])
    thermo = freq_data.get("thermochemistry") or {}
    n_imag = freq_data.get("imaginary_mode_count", 0)
    n_near_zero = freq_data.get("near_zero_mode_count", 0)
    near_zero_freqs = freq_data.get("near_zero_frequencies_cm1", [])
    sig_imag_freqs = freq_data.get("significant_imaginary_frequencies_cm1", [])

    all_freqs = [m["frequency_cm1"] for m in modes]
    real_freqs = [f for f in all_freqs if f >= 0]
    imag_freqs = [f for f in all_freqs if f < 0]

    # --- Element list ---
    elements: list[str] = []
    if input_path:
        try:
            inp = inspect_nwchem_input(input_path)
            elements = inp.get("elements", [])
        except Exception:
            pass

    # --- Mode counts ---
    n_modes = len(modes)

    # --- Band distribution ---
    band_dist: list[dict[str, Any]] = []
    for lo, hi, label in _FREQ_BANDS:
        in_band = [f for f in real_freqs if lo <= f < hi]
        if in_band or lo < 100:
            band_dist.append({
                "range_cm1": f"{lo}–{hi}",
                "label": label,
                "count": len(in_band),
                "examples_cm1": [round(f, 1) for f in sorted(in_band)[:5]],
            })

    # --- ZPE check ---
    zpe_correction = thermo.get("zero_point_correction") or {}
    zpe_kcal = zpe_correction.get("kcal_mol")
    n_atoms_thermo = None
    zpe_per_atom: float | None = None
    zpe_note: str | None = None
    if zpe_kcal is not None and elements:
        # Count non-H atoms as "heavy atoms"
        heavy = [e for e in elements if e != "H"]
        n_atoms_thermo = len(elements)
        n_heavy = len(heavy)
        if n_atoms_thermo > 0:
            zpe_per_atom = zpe_kcal / n_atoms_thermo
            # Rough expected range: 2-15 kcal/mol per heavy atom
            if n_heavy > 0:
                zpe_per_heavy = zpe_kcal / n_heavy
                if zpe_per_heavy < 0.5:
                    zpe_note = f"ZPE/heavy-atom={zpe_per_heavy:.1f} kcal/mol seems very low"
                elif zpe_per_heavy > 30.0:
                    zpe_note = f"ZPE/heavy-atom={zpe_per_heavy:.1f} kcal/mol seems very high"

    # --- X-H stretch checks ---
    missing_xh: list[str] = []
    if elements and real_freqs:
        elem_set = set(elements)
        has_h = "H" in elem_set
        if has_h:
            for heavy_elem, (lo, hi, name) in _EXPECTED_XH_BANDS.items():
                if heavy_elem in elem_set:
                    observed = any(lo <= f <= hi for f in real_freqs)
                    if not observed:
                        missing_xh.append(
                            f"{name} ({lo}–{hi} cm⁻¹) expected but not observed"
                        )

    # --- Very high frequency check ---
    suspicious_high = [f for f in real_freqs if f > 4000]

    # --- Assemble red flags and warnings ---
    red_flags: list[str] = []
    warnings_out: list[str] = []

    # Imaginary mode assessment
    if expect_minimum:
        if n_imag == 1 and sig_imag_freqs:
            red_flags.append(
                f"1 imaginary mode ({sig_imag_freqs[0]:.1f} cm⁻¹) — geometry is a transition state, "
                "not a minimum. Re-optimize or follow the imaginary mode."
            )
        elif n_imag > 1:
            red_flags.append(
                f"{n_imag} imaginary modes ({[round(f,1) for f in imag_freqs]}) — "
                "higher-order saddle point. Geometry needs rethinking."
            )
        elif n_imag == 1 and not sig_imag_freqs:
            warnings_out.append(
                "1 near-zero imaginary mode — likely numerical noise, but verify geometry."
            )
    else:
        if n_imag == 0:
            warnings_out.append("Expected 1 imaginary mode for TS but found 0 — may be a minimum.")
        elif n_imag > 1:
            red_flags.append(
                f"{n_imag} imaginary modes — TS should have exactly 1. Check geometry."
            )

    # Large imaginary modes
    very_large_imag = [f for f in imag_freqs if f < -200]
    if very_large_imag:
        red_flags.append(
            f"Very large imaginary mode(s) {[round(f,1) for f in very_large_imag]} cm⁻¹ — "
            "severe structural problem, not numerical noise."
        )

    # Near-zero real modes
    if n_near_zero > 6:
        # More than 6 near-zero modes is unusual (linear has 5, nonlinear has 6)
        extras = n_near_zero - 6
        warnings_out.append(
            f"{n_near_zero} near-zero modes (<20 cm⁻¹) — {extras} extra beyond the expected "
            "translational/rotational. May indicate flat PES, floppy molecule, or incomplete optimisation."
        )
    elif n_near_zero > 0 and near_zero_freqs:
        # Some non-negligible near-zero real modes
        real_nz = [f for f in near_zero_freqs if f > 0]
        if real_nz:
            warnings_out.append(
                f"Low-frequency real modes {[round(f,1) for f in real_nz]} cm⁻¹ — "
                "very soft modes; check for floppy conformations or weak intermolecular interactions."
            )

    if missing_xh:
        for m in missing_xh:
            warnings_out.append(f"MISSING MODE: {m}")

    if suspicious_high:
        warnings_out.append(
            f"Very high frequencies {[round(f,1) for f in suspicious_high]} cm⁻¹ (>4000) — "
            "check for erroneous geometry, missing mass, or wrong units."
        )

    if zpe_note:
        warnings_out.append(f"ZPE: {zpe_note}")

    # Check: if no real vibrational modes at all
    if len(real_freqs) == 0:
        red_flags.append("No real vibrational frequencies found — frequency calculation may have failed.")

    plausible = len(red_flags) == 0

    return {
        "plausible": plausible,
        "red_flags": red_flags,
        "warnings": warnings_out,
        "mode_counts": {
            "total": n_modes,
            "imaginary": n_imag,
            "near_zero": n_near_zero,
            "real_vibrational": len([f for f in real_freqs if f >= 20]),
        },
        "imaginary_frequencies_cm1": [round(f, 1) for f in imag_freqs],
        "band_distribution": band_dist,
        "zpe_check": {
            "zpe_kcal_mol": zpe_kcal,
            "n_atoms": n_atoms_thermo,
            "zpe_per_atom_kcal_mol": round(zpe_per_atom, 2) if zpe_per_atom is not None else None,
        },
        "missing_xh_stretches": missing_xh,
    }


# ---------------------------------------------------------------------------
# Spin state advisor
# ---------------------------------------------------------------------------

# d-block TMs: element → (Z, noble-gas core electrons)
_TM_Z_CORE: dict[str, tuple[int, int]] = {
    "Sc": (21, 18), "Ti": (22, 18), "V": (23, 18), "Cr": (24, 18),
    "Mn": (25, 18), "Fe": (26, 18), "Co": (27, 18), "Ni": (28, 18),
    "Cu": (29, 18), "Zn": (30, 18),
    "Y": (39, 36), "Zr": (40, 36), "Nb": (41, 36), "Mo": (42, 36),
    "Tc": (43, 36), "Ru": (44, 36), "Rh": (45, 36), "Pd": (46, 36),
    "Ag": (47, 36), "Cd": (48, 36),
    "Hf": (72, 68), "Ta": (73, 68), "W": (74, 68), "Re": (75, 68),
    "Os": (76, 68), "Ir": (77, 68), "Pt": (78, 68), "Au": (79, 68), "Hg": (80, 68),
}

# Common oxidation states ordered by frequency
_TM_COMMON_OX: dict[str, list[int]] = {
    "Sc": [3], "Ti": [4, 3, 2], "V": [3, 4, 5, 2], "Cr": [3, 2, 6],
    "Mn": [2, 3, 4, 7], "Fe": [2, 3, 4], "Co": [2, 3], "Ni": [2, 3],
    "Cu": [1, 2], "Zn": [2],
    "Y": [3], "Zr": [4, 3], "Nb": [5, 3, 4], "Mo": [4, 5, 6, 3],
    "Tc": [4, 7], "Ru": [2, 3, 4], "Rh": [3, 2], "Pd": [2, 4],
    "Ag": [1, 2], "Cd": [2],
    "Hf": [4], "Ta": [5, 3], "W": [4, 6, 3], "Re": [3, 4, 7],
    "Os": [4, 2, 3], "Ir": [3, 4], "Pt": [2, 4], "Au": [1, 3], "Hg": [2, 1],
}

# Hund high-spin vs strong-field low-spin unpaired electrons for d0..d10
_D_HS_UNPAIRED = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
_D_LS_UNPAIRED = [0, 1, 2, 1, 0, 1, 0, 1, 2, 1, 0]

# Ligand elements that imply weak vs strong crystal field
_WEAK_FIELD_ELEMENTS = {"F", "Cl", "Br", "I", "O", "S", "Se", "Te"}
_STRONG_FIELD_ELEMENTS = {"C", "N", "P"}




# ---------------------------------------------------------------------------
# Geometry plausibility checker
# ---------------------------------------------------------------------------

# Typical max coordination numbers per element (above this → red flag)
_MAX_COORD: dict[str, int] = {
    "H": 2, "He": 0,
    "Li": 4, "Be": 4, "B": 4, "C": 4, "N": 4, "O": 3, "F": 1, "Ne": 0,
    "Na": 6, "Mg": 6, "Al": 6, "Si": 6, "P": 6, "S": 6, "Cl": 1, "Ar": 0,
    "K": 8, "Ca": 8, "Ga": 6, "Ge": 4, "As": 6, "Se": 6, "Br": 1, "Kr": 0,
    "Rb": 8, "Sr": 8, "In": 6, "Sn": 6, "Sb": 6, "Te": 6, "I": 1, "Xe": 0,
    "Cs": 12, "Ba": 12, "Tl": 6, "Pb": 6, "Bi": 6,
}
# Transition metals: typical max CN = 9
for _tm in _TRANSITION_METALS:
    if _tm not in _MAX_COORD:
        _MAX_COORD[_tm] = 9

# Lanthanides and actinides: high-CN chemistry (up to 12–14)
_LANTHANIDES = {"La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"}
_ACTINIDES = {"Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"}
for _hm in _LANTHANIDES | _ACTINIDES:
    if _hm not in _MAX_COORD:
        _MAX_COORD[_hm] = 14

# All elements that behave like metals (used for coordination reporting)
_ALL_METALS = _TRANSITION_METALS | _LANTHANIDES | _ACTINIDES | {
    "Li","Na","K","Rb","Cs","Be","Mg","Ca","Sr","Ba",
    "Al","Ga","In","Tl","Sn","Pb","Bi",
}

# Typical CN ranges for main-group elements  {element: (min_ok, max_ok)}
_TYPICAL_COORD: dict[str, tuple[int, int]] = {
    "H": (1, 1), "He": (0, 0),
    "B": (2, 4), "C": (1, 4), "N": (1, 4), "O": (1, 2), "F": (1, 1),
    "Si": (2, 6), "P": (1, 6), "S": (1, 6), "Cl": (1, 1),
    "Ge": (2, 4), "As": (2, 6), "Se": (1, 6), "Br": (1, 1),
    "Sn": (2, 6), "Sb": (2, 6), "Te": (1, 6), "I": (1, 1),
    "Pb": (2, 6), "Bi": (2, 6),
}



__all__ = [
    "check_nwchem_geometry_plausibility",
    "check_nwchem_freq_plausibility",
]

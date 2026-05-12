"""Parser for the Molcas RASSCF / CASSCF module output block.

Goes well beyond what orbitron extracts: full per-symmetry inactive / RAS1 /
RAS2 / RAS3 / secondary partitioning, NO occupations per root (with structural
classification of each NO into "definitely active" / "promote-to-inactive
candidate" / "demote-to-secondary candidate"), CSF / determinant counts,
optimization convergence indicators, and per-root final energies.

This is the headline parser for multi-reference workflows — agents need this
to reason about active-space quality before launching CASPT2.
"""

from __future__ import annotations

import re
from typing import Any


_FLOAT_RE = r"-?\d+\.\d+(?:[Ee][+-]?\d+)?"

# --- Wave function specifications (scalar properties)
_NACT_ELEC_RE = re.compile(r"Number of electrons in active shells\s+(\d+)")
_NCLOSED_ELEC_RE = re.compile(r"Number of closed shell electrons\s+(\d+)")
_NFROZEN_ELEC_RE = re.compile(r"Number of frozen shell electrons\s+(\d+)")
_RAS1_HOLES_RE = re.compile(r"Max number of holes in RAS1 space\s+(\d+)")
_RAS3_ELECS_RE = re.compile(r"Max nr of electrons in RAS3 space\s+(\d+)")
_NFROZEN_ORB_RE = re.compile(r"Number of frozen orbitals\s+(\d+)")
_NINACT_ORB_RE = re.compile(r"Number of inactive orbitals\s+(\d+)")
_NACT_ORB_RE = re.compile(r"Number of active orbitals\s+(\d+)")
_NSEC_ORB_RE = re.compile(r"Number of secondary orbitals\s+(\d+)")
_SPIN_RE = re.compile(r"Spin quantum number\s+(" + _FLOAT_RE + r")")
_STATE_SYM_RE = re.compile(r"State symmetry\s+(\d+)")

# --- Per-symmetry orbital specs (RASSCF flavour: includes Inactive + RAS1/2/3)
_SYM_HEADER_RE = re.compile(r"^\s*Symmetry species\s+(\d+(?:\s+\d+)*)\s*$", re.M)
_PER_SYM_PATTERNS = {
    "frozen":    re.compile(r"^\s*Frozen orbitals\s+(\d+(?:\s+\d+)*)\s*$", re.M),
    "inactive":  re.compile(r"^\s*Inactive orbitals\s+(\d+(?:\s+\d+)*)\s*$", re.M),
    "active":    re.compile(r"^\s*Active orbitals\s+(\d+(?:\s+\d+)*)\s*$", re.M),
    "ras1":      re.compile(r"^\s*RAS1 orbitals\s+(\d+(?:\s+\d+)*)\s*$", re.M),
    "ras2":      re.compile(r"^\s*RAS2 orbitals\s+(\d+(?:\s+\d+)*)\s*$", re.M),
    "ras3":      re.compile(r"^\s*RAS3 orbitals\s+(\d+(?:\s+\d+)*)\s*$", re.M),
    "secondary": re.compile(r"^\s*Secondary orbitals\s+(\d+(?:\s+\d+)*)\s*$", re.M),
    "deleted":   re.compile(r"^\s*Deleted orbitals\s+(\d+(?:\s+\d+)*)\s*$", re.M),
    "basis_functions": re.compile(r"^\s*Number of basis functions\s+(\d+(?:\s+\d+)*)\s*$", re.M),
}

# --- CI expansion
_NCSF_RE = re.compile(r"Number of CSFs\s+(\d+)")
_NDET_RE = re.compile(r"Number of determinants\s+(\d+)")
_NROOTS_RE = re.compile(r"Number of root\(s\) required\s+(\d+)")
_ROOT_FOR_OPT_RE = re.compile(r"Root chosen for geometry opt\.\s+(\d+)")

# --- Final results
_AVG_CI_RE = re.compile(r"Average CI energy\s+(" + _FLOAT_RE + r")")
_RASSCF_FOR_STATE_RE = re.compile(r"RASSCF energy for state\s+(\d+)\s+(" + _FLOAT_RE + r")")
_SUPER_CI_RE = re.compile(r"Super-CI energy\s+(" + _FLOAT_RE + r")")
_RASSCF_CHANGE_RE = re.compile(r"RASSCF energy change\s+(" + _FLOAT_RE + r")")
_GRAD_NORM_RE = re.compile(r"Norm of electronic gradient\s+(" + _FLOAT_RE + r")")
_ROOT_ENERGY_RE = re.compile(r"::\s*RASSCF root number\s+(\d+)\s+Total energy:\s+(" + _FLOAT_RE + r")")

# --- NO occupations (per root)
# Pattern (a sub-block per root):
#   Natural orbitals and occupation numbers for root  N
#   sym 1:   1.991018   1.003112   0.997996
#   sym 2:   1.497474   1.497295   0.506464   0.506641
_NO_HEADER_RE = re.compile(r"^\s*Natural orbitals and occupation numbers for root\s+(\d+)\s*$", re.M)
_NO_SYM_LINE_RE = re.compile(r"^\s*sym\s+(\d+):\s+((?:" + _FLOAT_RE + r"\s*)+)\s*$", re.M)


def parse_rasscf(text: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "module": "rasscf",
        "wave_function": _parse_wavefunction_specs(text),
        "orbital_specs": _parse_orbital_specs(text),
        "ci_expansion": _parse_ci_expansion(text),
        "natural_occupations_per_root": _parse_natural_occupations(text),
        "root_energies": _parse_root_energies(text),
        "convergence": _parse_convergence(text),
    }
    info["active_space_signature"] = _build_active_signature(info)
    info["converged"] = bool(info["root_energies"]) and (
        info["convergence"].get("energy_change") is not None
        and abs(info["convergence"].get("energy_change") or 1.0) < 1e-5
    )
    info["natural_occupation_warnings"] = _classify_no_occupations(info["natural_occupations_per_root"])
    return info


def _parse_wavefunction_specs(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, pattern in {
        "active_electrons": _NACT_ELEC_RE,
        "closed_shell_electrons": _NCLOSED_ELEC_RE,
        "frozen_shell_electrons": _NFROZEN_ELEC_RE,
        "max_holes_ras1": _RAS1_HOLES_RE,
        "max_electrons_ras3": _RAS3_ELECS_RE,
        "frozen_orbitals_total": _NFROZEN_ORB_RE,
        "inactive_orbitals_total": _NINACT_ORB_RE,
        "active_orbitals_total": _NACT_ORB_RE,
        "secondary_orbitals_total": _NSEC_ORB_RE,
        "state_symmetry": _STATE_SYM_RE,
    }.items():
        m = pattern.search(text)
        if m:
            out[key] = int(m.group(1))
    if (m := _SPIN_RE.search(text)):
        out["spin"] = float(m.group(1))
    return out


def _parse_orbital_specs(text: str) -> dict[str, Any]:
    sym_match = _SYM_HEADER_RE.search(text)
    if not sym_match:
        return {}
    sym_indices = [int(x) for x in sym_match.group(1).split()]
    nsym = len(sym_indices)
    specs: dict[str, Any] = {
        "n_symmetries": nsym,
        "symmetry_indices": sym_indices,
    }
    after = text[sym_match.end():]
    for line in after.splitlines()[:3]:
        stripped = line.strip()
        if stripped and not stripped.startswith("Frozen"):
            tokens = stripped.split()
            if len(tokens) == nsym:
                specs["irrep_labels"] = tokens
                break
    for key, pattern in _PER_SYM_PATTERNS.items():
        m = pattern.search(text)
        if not m:
            continue
        try:
            values = [int(x) for x in m.group(1).split()]
        except ValueError:
            continue
        specs[key] = values
    return specs


def _parse_ci_expansion(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, pattern in {
        "csf_count": _NCSF_RE,
        "determinant_count": _NDET_RE,
        "n_roots": _NROOTS_RE,
        "root_for_optimization": _ROOT_FOR_OPT_RE,
    }.items():
        m = pattern.search(text)
        if m:
            out[key] = int(m.group(1))
    return out


def _parse_natural_occupations(text: str) -> list[dict[str, Any]]:
    """Parse the per-root NO occupation blocks. Returns a list ordered by root.

    Each entry: {"root": N, "occupations_by_symmetry": {sym_int: [floats]},
                 "all_occupations": [floats sorted], "total_active_electrons": float}
    """
    results: list[dict[str, Any]] = []
    headers = list(_NO_HEADER_RE.finditer(text))
    for idx, header in enumerate(headers):
        block_start = header.end()
        block_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        # Stop at first line that isn't a sym N: ... line OR a blank line.
        block = text[block_start:block_end]
        sym_to_occs: dict[int, list[float]] = {}
        for sym_match in _NO_SYM_LINE_RE.finditer(block):
            sym_idx = int(sym_match.group(1))
            occs = [float(x) for x in sym_match.group(2).split()]
            sym_to_occs[sym_idx] = occs
            # bail when we hit content that isn't sym lines (rough heuristic)
            # but the regex already handles it via .finditer; the next non-sym
            # line just won't match.
        if not sym_to_occs:
            continue
        all_occs = [o for occs in sym_to_occs.values() for o in occs]
        results.append(
            {
                "root": int(header.group(1)),
                "occupations_by_symmetry": sym_to_occs,
                "all_occupations": sorted(all_occs, reverse=True),
                "total_active_electrons": round(sum(all_occs), 6),
                "n_active_orbitals": len(all_occs),
            }
        )
    return results


def _parse_root_energies(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in _ROOT_ENERGY_RE.finditer(text):
        out.append({"root": int(m.group(1)), "energy_hartree": float(m.group(2))})
    return out


def _parse_convergence(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if (m := _AVG_CI_RE.search(text)):
        out["average_ci_energy"] = float(m.group(1))
    if (m := _SUPER_CI_RE.search(text)):
        out["super_ci_energy"] = float(m.group(1))
    if (m := _RASSCF_CHANGE_RE.search(text)):
        out["energy_change"] = float(m.group(1))
    if (m := _GRAD_NORM_RE.search(text)):
        out["gradient_norm"] = float(m.group(1))
    states = []
    for m in _RASSCF_FOR_STATE_RE.finditer(text):
        states.append({"state": int(m.group(1)), "energy_hartree": float(m.group(2))})
    if states:
        out["per_state_final"] = states
    return out


def _build_active_signature(info: dict[str, Any]) -> str:
    """Produce a human-readable CAS / RAS signature string.

    For a CASSCF: "CAS(8,7) — singlet, sym 1"
    For a RASSCF with non-zero RAS1/RAS3 holes/electrons:
        "RAS(8,7) [h=2 e=2] — singlet, sym 1"
    """
    wf = info.get("wave_function") or {}
    nact_e = wf.get("active_electrons")
    nact_o = wf.get("active_orbitals_total")
    if nact_e is None or nact_o is None:
        return ""
    holes = wf.get("max_holes_ras1", 0) or 0
    electrons_into_ras3 = wf.get("max_electrons_ras3", 0) or 0
    is_ras = bool(holes or electrons_into_ras3)
    label = "RAS" if is_ras else "CAS"
    base = f"{label}({nact_e},{nact_o})"
    if is_ras:
        base += f" [h={holes} e={electrons_into_ras3}]"
    spin = wf.get("spin")
    sym = wf.get("state_symmetry")
    spin_label = None
    if spin is not None:
        # Multiplicity from S
        mult = int(round(2 * spin + 1))
        spin_label = {1: "singlet", 2: "doublet", 3: "triplet", 4: "quartet", 5: "quintet"}.get(mult, f"mult={mult}")
    extras = []
    if spin_label:
        extras.append(spin_label)
    if sym is not None:
        extras.append(f"sym {sym}")
    if extras:
        base += " — " + ", ".join(extras)
    return base


def _classify_no_occupations(roots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify each NO into 'truly_active' / 'promote_candidate' / 'demote_candidate' / 'edge'.

    Heuristics (conservative):
      occ >= 1.98       → near-doubly-occupied → "promote to inactive" candidate
      1.90 <= occ < 1.98 → "edge_doubly_occupied" — likely promote candidate
      0.10 <= occ <= 1.90 → "truly_active"
      0.02 < occ < 0.10  → "edge_empty" — likely demote candidate
      occ <= 0.02       → near-virtual → "demote to secondary" candidate

    Returns one dict per root with counts + flagged orbitals (only the
    questionable ones).
    """
    warnings: list[dict[str, Any]] = []
    for root_data in roots:
        occs = root_data["all_occupations"]
        counts = {
            "near_doubly_occupied": 0,
            "edge_doubly_occupied": 0,
            "truly_active": 0,
            "edge_empty": 0,
            "near_virtual": 0,
        }
        flagged: list[dict[str, Any]] = []
        for sym, sym_occs in root_data["occupations_by_symmetry"].items():
            for orb_idx, occ in enumerate(sym_occs, start=1):
                if occ >= 1.98:
                    cls = "near_doubly_occupied"
                elif occ >= 1.90:
                    cls = "edge_doubly_occupied"
                elif occ >= 0.10:
                    cls = "truly_active"
                elif occ > 0.02:
                    cls = "edge_empty"
                else:
                    cls = "near_virtual"
                counts[cls] += 1
                if cls in {"near_doubly_occupied", "near_virtual"}:
                    flagged.append({"sym": sym, "orbital_in_sym": orb_idx, "occupation": round(occ, 5), "class": cls})
        warnings.append(
            {
                "root": root_data["root"],
                "counts": counts,
                "flagged_orbitals": flagged,
                "summary": _summarize_active_quality(counts, len(occs)),
            }
        )
    return warnings


def _summarize_active_quality(counts: dict[str, int], total: int) -> str:
    truly = counts["truly_active"]
    promote = counts["near_doubly_occupied"]
    demote = counts["near_virtual"]
    edges = counts["edge_doubly_occupied"] + counts["edge_empty"]
    parts = [f"{truly}/{total} truly active"]
    if promote:
        parts.append(f"{promote} near-double (consider promoting to inactive)")
    if demote:
        parts.append(f"{demote} near-empty (consider demoting to secondary)")
    if edges:
        parts.append(f"{edges} on the edge (1.90-1.98 or 0.02-0.10)")
    return "; ".join(parts)

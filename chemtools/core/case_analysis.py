"""Generic case-analysis helpers used by per-program analyze_*_case tools.

The per-program tools (analyze_molcas_case, analyze_nwchem_case) own the
high-level dispatching — they choose which parser to call, which checks
to run, and the verdict mapping. This module provides the chemistry-
agnostic pieces:

  - classify_imaginary_modes(): split frequency list into physical vs
    projection-artifact based on |f| < threshold
  - check_charge_spin_parity(): verify 2S and electron count have
    matching parity
  - bond_table_for_atoms(): short-list bond lengths for tiny molecules,
    delegating to core.geometry

Each per-program analyze_*_case builds an IssueCollector
(from core.issues), runs the generic checks, runs its own program-
specific checks (CASPT2 reference weight, MCSCF state mixing, etc.), and
emits the unified result shape.
"""

from __future__ import annotations

from typing import Any

from chemtools.core.geometry import inspect_geometry


def classify_imaginary_modes(
    frequencies_cm1: list[float],
    *,
    artifact_threshold_cm1: float = 50.0,
    noise_floor_cm1: float = 5.0,
) -> dict[str, list[float]]:
    """Split a list of frequencies into physical imaginary modes vs
    projection artifacts.

    Convention: an entry is "imaginary" if its value is **negative** (the
    common encoding for ``i123.4 cm⁻¹``). Frequencies between
    ``-noise_floor_cm1`` and ``-artifact_threshold_cm1`` are flagged as
    artifacts (unprojected translation/rotation); everything more negative
    is "physical".

    Returns ``{"physical": [...], "artifacts": [...]}``.
    """
    physical: list[float] = []
    artifacts: list[float] = []
    for f in frequencies_cm1:
        if f >= -noise_floor_cm1:
            continue  # not imaginary
        if f >= -artifact_threshold_cm1:
            artifacts.append(f)
        else:
            physical.append(f)
    return {"physical": physical, "artifacts": artifacts}


def check_charge_spin_parity(
    n_active_electrons: int | None,
    multiplicity: int | None,
) -> dict[str, Any] | None:
    """Verify ``n_active_electrons`` and ``2S = multiplicity - 1`` share
    parity. Returns an issue dict (severity=problematic) if mismatched,
    None if consistent or if inputs are missing.
    """
    if n_active_electrons is None or multiplicity is None:
        return None
    two_s = int(multiplicity) - 1
    if (int(n_active_electrons) % 2) != (two_s % 2):
        return {
            "severity": "problematic",
            "message": (
                f"Charge/spin parity mismatch: {n_active_electrons} active "
                f"electrons cannot give 2S={two_s} unpaired (parity)."
            ),
            "hint": (
                "Recompute the active-space partition with "
                "compute_molcas_active_space_partition (Molcas) or analogous "
                "NWChem tool."
            ),
        }
    return None


def bond_table_for_atoms(
    atoms: list[dict],
    *,
    max_atoms: int = 12,
    max_bonds: int = 6,
) -> list[dict[str, Any]] | None:
    """Compute a short bond-length table for small molecules using
    core.geometry. Returns None for large systems (n_atoms > max_atoms) to
    keep summarize output compact.

    The atoms list is assumed in Angstrom (per the core.geometry
    convention); callers must pre-normalize bohr coordinates.
    """
    if not atoms or len(atoms) > max_atoms:
        return None
    info = inspect_geometry(atoms, max_bond_length=2.5)
    bonds = info.get("bond_lengths") or []
    return bonds[:max_bonds] or None

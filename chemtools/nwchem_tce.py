from __future__ import annotations

"""
nwchem_tce.py — TCE (Tensor Contraction Engine) support for NWChem.

Covers:
  - parse_tce_output: extract energies, convergence, frozen orbital info
  - parse_nwchem_movecs: read eigenvalues/occupations from a binary movecs file
  - swap_nwchem_movecs: swap two MOs in a binary movecs file
  - suggest_tce_freeze_count: chemically-aware core orbital count (never freeze atomic)
  - analyze_tce_orbital_ordering: warn when orbital ordering may require swaps
"""

import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .common import make_metadata, parse_float_after_delimiter


# ---------------------------------------------------------------------------
# Atomic core orbital counts (what NWChem "freeze atomic" would freeze)
# Maps element symbol → number of doubly-occupied core *orbitals* to freeze
# These are the filled inner shells, not the valence/active shells.
# NOTE: for transition metals the 3d/4d/5d are NOT included because they
#       are chemically active valence orbitals, not core.
# ---------------------------------------------------------------------------
_ATOMIC_CORE_ORBS: dict[str, int] = {
    # Period 1 — no core
    "H": 0, "He": 0,
    # Period 2 — 1s frozen (1 orbital)
    "Li": 1, "Be": 1, "B": 1, "C": 1, "N": 1, "O": 1, "F": 1, "Ne": 1,
    # Period 3 — 1s2s2p frozen (5 orbitals)
    "Na": 5, "Mg": 5, "Al": 5, "Si": 5, "P": 5, "S": 5, "Cl": 5, "Ar": 5,
    # Period 4 — K/Ca: 1s2s2p3s3p = 9; TMs same (3d not frozen); Ga-Kr: +3d = 14
    "K": 9, "Ca": 9,
    "Sc": 9, "Ti": 9, "V": 9, "Cr": 9, "Mn": 9, "Fe": 9,
    "Co": 9, "Ni": 9, "Cu": 9, "Zn": 9,
    "Ga": 14, "Ge": 14, "As": 14, "Se": 14, "Br": 14, "Kr": 14,
    # Period 5 — Rb/Sr: 18; 4d TMs: 18 (3d5 included now); In-Xe: +4d = 23
    "Rb": 18, "Sr": 18,
    "Y": 18, "Zr": 18, "Nb": 18, "Mo": 18, "Tc": 18, "Ru": 18,
    "Rh": 18, "Pd": 18, "Ag": 18, "Cd": 18,
    "In": 23, "Sn": 23, "Sb": 23, "Te": 23, "I": 23, "Xe": 23,
    # Period 6 — Cs/Ba: 27; lanthanides 27 (4f not frozen by default); Hf-Rn: +4f+5d varies
    "Cs": 27, "Ba": 27,
    "La": 27, "Ce": 27, "Pr": 27, "Nd": 27, "Pm": 27, "Sm": 27,
    "Eu": 27, "Gd": 27, "Tb": 27, "Dy": 27, "Ho": 27, "Er": 27,
    "Tm": 27, "Yb": 27, "Lu": 27,
    "Hf": 34, "Ta": 34, "W": 34, "Re": 34, "Os": 34, "Ir": 34,
    "Pt": 34, "Au": 34, "Hg": 34,
    "Tl": 39, "Pb": 39, "Bi": 39, "Po": 39, "At": 39, "Rn": 39,
}

# Regex patterns for TCE output parsing
_TCE_HEADER_RE = re.compile(
    r"NWChem Extensible Many-Electron Theory Module", re.IGNORECASE
)
_TCE_WFTYPE_RE = re.compile(r"Wavefunction type\s*:\s*(.+)", re.IGNORECASE)
_TCE_NELEC_RE = re.compile(r"No\. of electrons\s*:\s*(\d+)", re.IGNORECASE)
_TCE_NORB_RE = re.compile(r"No\. of orbitals\s*:\s*(\d+)", re.IGNORECASE)
_TCE_FROZEN_CORE_RE = re.compile(r"Alpha frozen cores\s*:\s*(\d+)", re.IGNORECASE)
_TCE_FROZEN_VIRT_RE = re.compile(r"Alpha frozen virtuals\s*:\s*(\d+)", re.IGNORECASE)
_TCE_CALC_TYPE_RE = re.compile(r"Calculation type\s*:\s*(.+)", re.IGNORECASE)
_TCE_PERT_RE = re.compile(r"Perturbative correction\s*:\s*(.+)", re.IGNORECASE)
_TCE_MAXITER_RE = re.compile(r"Max iterations\s*:\s*(\d+)", re.IGNORECASE)
_TCE_THRESH_RE = re.compile(r"Residual threshold\s*:\s*([0-9.DEde+\-]+)", re.IGNORECASE)
_TCE_MULT_RE = re.compile(r"Spin multiplicity\s*:\s*(\w+)", re.IGNORECASE)

_MBPT2_CORR_RE = re.compile(
    r"MBPT\(2\) correlation energy\s*/\s*hartree\s*=\s*([+-]?[0-9]+\.[0-9]+(?:[DEde][+-]?[0-9]+)?)",
    re.IGNORECASE,
)
_MBPT2_TOTAL_RE = re.compile(
    r"MBPT\(2\) total energy\s*/\s*hartree\s*=\s*([+-]?[0-9]+\.[0-9]+(?:[DEde][+-]?[0-9]+)?)",
    re.IGNORECASE,
)
_CCSD_CORR_RE = re.compile(
    r"^[ \t]*CCSD correlation energy\s*/\s*hartree\s*=\s*([+-]?[0-9]+\.[0-9]+(?:[DEde][+-]?[0-9]+)?)",
    re.IGNORECASE | re.MULTILINE,
)
_CCSD_TOTAL_RE = re.compile(
    r"^[ \t]*CCSD total energy\s*/\s*hartree\s*=\s*([+-]?[0-9]+\.[0-9]+(?:[DEde][+-]?[0-9]+)?)",
    re.IGNORECASE | re.MULTILINE,
)
_CCSDT_CORR_RE = re.compile(
    r"CCSD\(T\) correlation energy\s*/\s*hartree\s*=\s*([+-]?[0-9]+\.[0-9]+(?:[DEde][+-]?[0-9]+)?)",
    re.IGNORECASE,
)
_CCSDT_TOTAL_RE = re.compile(
    r"CCSD\(T\) total energy\s*/\s*hartree\s*=\s*([+-]?[0-9]+\.[0-9]+(?:[DEde][+-]?[0-9]+)?)",
    re.IGNORECASE,
)
_CCSDBT_CORR_RE = re.compile(
    r"CCSD\[T\] correlation energy\s*/\s*hartree\s*=\s*([+-]?[0-9]+\.[0-9]+(?:[DEde][+-]?[0-9]+)?)",
    re.IGNORECASE,
)
_CCSDBT_TOTAL_RE = re.compile(
    r"CCSD\[T\] total energy\s*/\s*hartree\s*=\s*([+-]?[0-9]+\.[0-9]+(?:[DEde][+-]?[0-9]+)?)",
    re.IGNORECASE,
)
_ITER_CONVERGED_RE = re.compile(r"Iterations converged", re.IGNORECASE)
_SCF_CONVERGED_RE = re.compile(r"The SCF is already converged", re.IGNORECASE)
_CCSD_ITER_TABLE_RE = re.compile(
    r"^\s*(\d+)\s+([0-9.DEde+\-]+)\s+([+-][0-9.DEde+\-]+)\s+[0-9.]+\s+[0-9.]+",
    re.MULTILINE,
)
_MBPT2_ITER_TABLE_RE = re.compile(
    r"^\s*(\d+)\s+([0-9.DEde+\-]+)\s+([+-][0-9.DEde+\-]+)\s+[0-9.]+\s+[0-9.]+",
    re.MULTILINE,
)


def _parse_float(s: str) -> float | None:
    try:
        return float(s.replace("D", "e").replace("d", "e"))
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# TCE output parser
# ---------------------------------------------------------------------------

def parse_tce_output(path: str, contents: str) -> dict[str, Any]:
    """Parse NWChem TCE output sections from a .out file.

    Returns a dict with:
      - tce_sections: list of per-section dicts (one per TCE task)
      - energies: aggregated final energies across all sections
    """
    lines = contents.splitlines()
    sections: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        if _TCE_HEADER_RE.search(lines[i]):
            section, i = _parse_tce_section(lines, i)
            sections.append(section)
        else:
            i += 1

    # Aggregate the last (or only) section's energies
    energies: dict[str, float | None] = {
        "scf_total": None,
        "correlation": None,
        "total": None,
        "method": None,
    }
    ccsd_corr_correction: float | None = None
    ccsdt_correction: float | None = None

    # Scan full contents for all energy lines (handles multi-section files)
    m = _MBPT2_TOTAL_RE.search(contents)
    if m:
        energies["total"] = _parse_float(m.group(1))
        energies["method"] = "MP2"
    m = _MBPT2_CORR_RE.search(contents)
    if m:
        energies["correlation"] = _parse_float(m.group(1))

    # CCSD (note: look for the last occurrence)
    ccsd_total_matches = list(_CCSD_TOTAL_RE.finditer(contents))
    if ccsd_total_matches:
        energies["total"] = _parse_float(ccsd_total_matches[-1].group(1))
        energies["method"] = "CCSD"
    ccsd_corr_matches = list(_CCSD_CORR_RE.finditer(contents))
    if ccsd_corr_matches:
        energies["correlation"] = _parse_float(ccsd_corr_matches[-1].group(1))

    # CCSD(T)
    m = _CCSDT_TOTAL_RE.search(contents)
    if m:
        energies["total"] = _parse_float(m.group(1))
        energies["method"] = "CCSD(T)"
    m = _CCSDT_CORR_RE.search(contents)
    if m:
        energies["correlation"] = _parse_float(m.group(1))
    m = _CCSDBT_TOTAL_RE.search(contents)
    if m:
        ccsd_bracket_t = _parse_float(m.group(1))
    else:
        ccsd_bracket_t = None

    # SCF total energy
    for line in lines:
        if "total scf energy" in line.lower():
            val = parse_float_after_delimiter(line, "=")
            if val is not None:
                energies["scf_total"] = val

    return {
        "metadata": make_metadata(path, contents, "nwchem"),
        "tce_sections": sections,
        "scf_total_energy_hartree": energies["scf_total"],
        "method": energies["method"],
        "correlation_energy_hartree": energies["correlation"],
        "total_energy_hartree": energies["total"],
        "ccsd_bracket_t_total_hartree": ccsd_bracket_t,
        "converged": any(s.get("converged") for s in sections),
    }


def _parse_tce_section(lines: list[str], start: int) -> tuple[dict[str, Any], int]:
    """Parse a single TCE section starting at line `start`."""
    section: dict[str, Any] = {
        "wavefunction_type": None,
        "n_electrons": None,
        "n_orbitals": None,
        "frozen_cores": None,
        "frozen_virtuals": None,
        "multiplicity": None,
        "calculation_type": None,
        "perturbative_correction": None,
        "max_iterations": None,
        "residual_threshold": None,
        "method": None,
        "correlation_energy_hartree": None,
        "total_energy_hartree": None,
        "correction_energy_hartree": None,
        "converged": False,
        "iterations_run": None,
    }
    i = start + 1
    n = len(lines)

    # Parse header until we hit "Memory Information" or energies
    while i < n:
        line = lines[i]
        stripped = line.strip()

        m = _TCE_WFTYPE_RE.search(stripped)
        if m:
            section["wavefunction_type"] = m.group(1).strip()
        m = _TCE_NELEC_RE.search(stripped)
        if m:
            section["n_electrons"] = int(m.group(1))
        m = _TCE_NORB_RE.search(stripped)
        if m:
            section["n_orbitals"] = int(m.group(1))
        m = _TCE_FROZEN_CORE_RE.search(stripped)
        if m:
            section["frozen_cores"] = int(m.group(1))
        m = _TCE_FROZEN_VIRT_RE.search(stripped)
        if m:
            section["frozen_virtuals"] = int(m.group(1))
        m = _TCE_MULT_RE.search(stripped)
        if m:
            section["multiplicity"] = m.group(1).strip()
        m = _TCE_CALC_TYPE_RE.search(stripped)
        if m:
            section["calculation_type"] = m.group(1).strip()
        m = _TCE_PERT_RE.search(stripped)
        if m:
            section["perturbative_correction"] = m.group(1).strip()
        m = _TCE_MAXITER_RE.search(stripped)
        if m:
            section["max_iterations"] = int(m.group(1))
        m = _TCE_THRESH_RE.search(stripped)
        if m:
            section["residual_threshold"] = m.group(1).strip()

        # Energy lines
        m = _MBPT2_CORR_RE.search(stripped)
        if m:
            section["correlation_energy_hartree"] = _parse_float(m.group(1))
            section["method"] = "MP2"
        m = _MBPT2_TOTAL_RE.search(stripped)
        if m:
            section["total_energy_hartree"] = _parse_float(m.group(1))
            section["method"] = "MP2"

        m = _CCSD_CORR_RE.search(stripped)
        if m:
            section["correlation_energy_hartree"] = _parse_float(m.group(1))
            section["method"] = "CCSD"
        m = _CCSD_TOTAL_RE.search(stripped)
        if m:
            section["total_energy_hartree"] = _parse_float(m.group(1))
            section["method"] = "CCSD"

        m = _CCSDT_CORR_RE.search(stripped)
        if m:
            section["correlation_energy_hartree"] = _parse_float(m.group(1))
            section["method"] = "CCSD(T)"
        m = _CCSDT_TOTAL_RE.search(stripped)
        if m:
            section["total_energy_hartree"] = _parse_float(m.group(1))
            section["method"] = "CCSD(T)"

        if _ITER_CONVERGED_RE.search(stripped):
            section["converged"] = True

        # Detect start of next TCE section or end of file
        if i > start + 5 and _TCE_HEADER_RE.search(line):
            break

        # Detect end of task
        if "task  times" in line.lower() and ("cpu:" in line.lower() or "wall:" in line.lower()):
            i += 1
            break

        i += 1

    # Infer method from calculation_type / perturbative_correction if not already set
    if section["method"] is None:
        calc = (section["calculation_type"] or "").lower()
        pert = (section["perturbative_correction"] or "").strip().lower()
        if "ccsd" in calc and "(t)" in pert:
            section["method"] = "CCSD(T)"
        elif "ccsd" in calc:
            section["method"] = "CCSD"
        elif "mbpt" in calc or "perturbation" in calc:
            section["method"] = "MP2"

    return section, i


# ---------------------------------------------------------------------------
# Freeze count advisor
# ---------------------------------------------------------------------------

def suggest_tce_freeze_count(
    elements: list[str],
    ecp_core_electrons: dict[str, int] | None = None,
    charge: int = 0,
    multiplicity: int = 1,
) -> dict[str, Any]:
    """Suggest the number of orbitals to freeze in a TCE calculation.

    This implements the same logic as NWChem's ``freeze atomic`` but expressed
    as an explicit count so the agent can verify and adjust.  The agent MUST
    inspect the actual SCF orbital ordering before trusting this number.

    Parameters
    ----------
    elements:
        List of element symbols present in the molecule (can repeat).
    ecp_core_electrons:
        Optional dict mapping element → number of electrons replaced by ECP
        (the ``nelec`` value from the ECP block).  ECPs reduce the core count.
    charge:
        Molecular charge (used for electron count consistency check).
    multiplicity:
        Spin multiplicity (used to verify electron count parity).

    Returns
    -------
    dict with:
      freeze_count          total number of orbitals to freeze
      n_electrons           estimated total electron count (None if elements unknown)
      n_correlated          estimated correlated electrons after freeze
      per_element           per-element breakdown
      warnings              list of issues the agent must check
      notes                 explanation of the logic
    """
    from collections import Counter
    from .common import ELEMENT_TO_Z
    ecp = ecp_core_electrons or {}
    atom_counts = Counter(elements)
    unique_elements = list(dict.fromkeys(elements))  # order-preserving unique

    per_element: list[dict[str, Any]] = []
    total_freeze = 0
    warnings: list[str] = []
    notes: list[str] = []

    # --- Compute expected n_electrons from atomic numbers ---
    n_electrons: int | None = None
    try:
        nuclear_electrons = sum(ELEMENT_TO_Z[e] for e in elements)
        ecp_removed = sum(ecp.get(e, 0) * atom_counts[e] for e in unique_elements)
        n_electrons = nuclear_electrons - ecp_removed - charge
    except KeyError:
        pass  # unknown element, skip

    for elem in unique_elements:
        count = _ATOMIC_CORE_ORBS.get(elem)
        n_atoms = atom_counts[elem]
        if count is None:
            warnings.append(
                f"Element {elem} not in core-orbital table; manual inspection required."
            )
            per_element.append({
                "element": elem,
                "n_atoms": n_atoms,
                "core_orbitals": None,
                "ecp_removes": 0,
                "freeze_orbitals": 0,
            })
            continue

        ecp_electrons = ecp.get(elem, 0)
        # ECP replaces ecp_electrons electrons → ecp_electrons/2 orbitals removed from
        # the all-electron count per atom.  Clamp at 0.
        ecp_orbs_removed = ecp_electrons // 2
        freeze_per_atom = max(0, count - ecp_orbs_removed)
        freeze = freeze_per_atom * n_atoms
        per_element.append({
            "element": elem,
            "n_atoms": n_atoms,
            "all_electron_core_orbitals_per_atom": count,
            "ecp_electrons": ecp_electrons,
            "ecp_orbitals_removed_per_atom": ecp_orbs_removed,
            "freeze_orbitals_per_atom": freeze_per_atom,
            "freeze_orbitals": freeze,
        })
        total_freeze += freeze

    # --- ECP coverage warnings ---
    # If any element has a library ECP but no nelec was provided, freeze may be wrong
    ecp_elements_no_nelec = [e for e in unique_elements if ecp.get(e, 0) == 0
                              and _ATOMIC_CORE_ORBS.get(e, 0) > 0
                              and e in {"La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
                                        "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
                                        "Tl","Pb","Bi","Po","At","Rn",
                                        "U","Th","Pu","Am","Np"}]
    if ecp_elements_no_nelec:
        warnings.append(
            f"Heavy elements {ecp_elements_no_nelec} may use an ECP but no nelec was "
            "provided. If an ECP is in use, the freeze count is overestimated. "
            "Provide ecp_core_electrons or verify the input file has an explicit ECP block."
        )

    notes.append(
        "freeze_count is a chemically-informed starting estimate, never 'freeze atomic'. "
        "Always inspect the actual SCF orbital eigenvalues and character before using "
        "this number in the tce block."
    )
    notes.append(
        "Use parse_nwchem_movecs (on the .movecs file) or parse_mos (on the SCF output) "
        "to verify orbital ordering. If the lowest orbitals are not the expected cores, "
        "use swap_nwchem_movecs to reorder them before running TCE."
    )

    # --- Electron count consistency checks ---
    n_correlated: int | None = None
    if n_electrons is not None:
        n_correlated = n_electrons - 2 * total_freeze

        # Parity check: n_electrons must match multiplicity parity
        # n_unpaired = multiplicity - 1; n_paired = (n_electrons - n_unpaired)
        # n_electrons - (multiplicity - 1) must be even
        if (n_electrons - (multiplicity - 1)) % 2 != 0:
            warnings.append(
                f"INCONSISTENCY: n_electrons={n_electrons} and multiplicity={multiplicity} "
                "have incompatible parity. Check charge and ECP nelec values."
            )

        if n_correlated <= 0:
            warnings.append(
                f"CRITICAL: freeze_count={total_freeze} leaves only {n_correlated} "
                "correlated electrons. The freeze count exceeds or equals n_electrons/2. "
                "Reduce freeze_count before running TCE."
            )
        elif n_correlated < 4:
            warnings.append(
                f"WARNING: only {n_correlated} correlated electrons after freeze. "
                "TCE will run but results may be trivial."
            )

        if n_electrons <= 0:
            warnings.append(
                f"CRITICAL: computed n_electrons={n_electrons} is non-positive. "
                "Check charge ({charge}) and ECP nelec values."
            )

    # Warn for elements with ambiguous d-shell treatment (4d/5d main-group and heavy atoms)
    heavy_elements = [
        e for e in unique_elements
        if e in {"Ga","Ge","As","Se","Br","Kr",
                 "In","Sn","Sb","Te","I","Xe",
                 "Tl","Pb","Bi","Po","At","Rn"}
    ]
    if heavy_elements:
        warnings.append(
            f"Elements {heavy_elements} have d-shells that may or may not be frozen "
            "depending on the ECP definition. The suggested freeze count includes "
            "inner d-shells as core, but you may want fewer frozen orbitals if the "
            "d-shell is partially valence. Always verify against the actual SCF orbital "
            "energies and character."
        )

    return {
        "freeze_count": total_freeze,
        "n_electrons": n_electrons,
        "n_correlated": n_correlated,
        "per_element": per_element,
        "warnings": warnings,
        "notes": notes,
    }


def analyze_tce_orbital_ordering(
    orbitals: list[dict[str, Any]],
    freeze_count: int,
) -> dict[str, Any]:
    """Check if the lowest freeze_count occupied orbitals are all chemically core-like.

    Parameters
    ----------
    orbitals:
        List of orbital dicts as returned by parse_mos, with at least:
        ``vector_number``, ``energy_hartree``, ``occupancy``, ``dominant_character``
    freeze_count:
        Proposed number of orbitals to freeze.

    Returns
    -------
    dict with:
      ordering_ok         bool — True if the freeze window looks clean
      warnings            issues found
      swap_suggestions    list of {from_mo, to_mo} swap pairs needed
      proposed_orbitals   summary of the freeze window
    """
    occupied = sorted(
        [orb for orb in orbitals if orb.get("occupancy", 0) > 0.5],
        key=lambda o: o["vector_number"],
    )
    if not occupied:
        return {
            "ordering_ok": False,
            "warnings": ["No occupied orbitals found in the orbital list."],
            "swap_suggestions": [],
            "proposed_orbitals": [],
        }

    freeze_window = occupied[:freeze_count]
    remaining = occupied[freeze_count:]

    warnings: list[str] = []
    swap_suggestions: list[dict[str, Any]] = []

    # Check for suspiciously high-energy orbitals inside the freeze window
    # and low-energy orbitals outside it.
    freeze_energies = [o["energy_hartree"] for o in freeze_window]
    remain_energies = [o["energy_hartree"] for o in remaining]

    if freeze_energies and remain_energies:
        max_freeze_e = max(freeze_energies)
        min_remain_e = min(remain_energies)
        if max_freeze_e > min_remain_e:
            # An orbital inside the freeze window is higher in energy than one outside.
            warnings.append(
                f"Orbital ordering anomaly: the highest-energy frozen orbital "
                f"(E={max_freeze_e:.4f} h) is above the lowest correlated orbital "
                f"(E={min_remain_e:.4f} h). "
                "This can happen when a ligand core (e.g. O 1s) is lower than a "
                "metal inner-shell (e.g. Zn 3s/3p). Inspect the dominant character "
                "of each orbital and consider swap_nwchem_movecs."
            )
            # Suggest swaps: find the pairs that need to cross
            for f_orb in freeze_window:
                for r_orb in remaining:
                    if f_orb["energy_hartree"] > r_orb["energy_hartree"]:
                        swap_suggestions.append({
                            "from_mo": r_orb["vector_number"],
                            "to_mo": f_orb["vector_number"],
                            "reason": (
                                f"MO {r_orb['vector_number']} (E={r_orb['energy_hartree']:.4f} h, "
                                f"{r_orb.get('dominant_character','?')}) should be in the freeze "
                                f"window; MO {f_orb['vector_number']} "
                                f"(E={f_orb['energy_hartree']:.4f} h, "
                                f"{f_orb.get('dominant_character','?')}) should be correlated."
                            ),
                        })

    proposed: list[dict[str, Any]] = []
    for orb in freeze_window:
        proposed.append({
            "mo": orb["vector_number"],
            "energy_hartree": orb["energy_hartree"],
            "dominant_character": orb.get("dominant_character"),
        })

    ordering_ok = len(warnings) == 0

    return {
        "ordering_ok": ordering_ok,
        "warnings": warnings,
        "swap_suggestions": swap_suggestions,
        "proposed_freeze_orbitals": proposed,
    }


# ---------------------------------------------------------------------------
# TCE amplitude file analysis
# ---------------------------------------------------------------------------

def parse_tce_amplitudes(output_path: str) -> dict[str, Any]:
    """Parse saved TCE amplitude files (*.t1_copy.*, *.t2_copy.*) and compute
    multireference diagnostics.

    NWChem writes these files only when the input contains::

        set tce:save_t T T

    Returns T1, D1 (nosym runs only), T2 statistics, triples fraction, and a
    combined MR assessment verdict.  If no amplitude files are found, returns a
    result with ``available: false`` and a hint.

    Parameters
    ----------
    output_path:
        Path to the NWChem TCE ``.out`` file.  Amplitude files are located by
        glob-matching ``<stem>.t1_copy.*`` / ``<stem>.t2_copy.*`` in the same
        directory.
    """
    from .common import read_text
    from pathlib import Path
    import glob as _glob

    out_path = Path(output_path)
    parent = out_path.parent

    # NWChem amplitude files are named after the `start` directive, not the output
    # file.  Parse the actual base name from the output ("t1 file name = ./foo.t1").
    _T1_FILE_RE = re.compile(r"t1 file name\s*=\s*\.?/?(.*?)\.t1\s*$", re.IGNORECASE | re.MULTILINE)
    contents_for_stem = read_text(output_path)
    stem_match = _T1_FILE_RE.search(contents_for_stem)
    if stem_match:
        amp_stem = Path(stem_match.group(1).strip()).name
    else:
        amp_stem = out_path.stem  # fallback: try output file stem

    # Locate amplitude files (suffix varies, e.g. .40000)
    t1_matches = sorted(_glob.glob(str(parent / f"{amp_stem}.t1_copy.*")))
    t2_matches = sorted(_glob.glob(str(parent / f"{amp_stem}.t2_copy.*")))

    if not t1_matches:
        return {
            "available": False,
            "reason": (
                "No t1_copy file found. Re-run with 'set tce:save_t T T' in the input "
                "to persist amplitude files for diagnostic analysis."
            ),
            "t1_file": None,
            "t2_file": None,
        }

    t1_file = t1_matches[0]
    t2_file = t2_matches[0] if t2_matches else None

    # Parse TCE output to get orbital/electron counts and energies
    # (reuse the contents already read for stem detection)
    contents = contents_for_stem
    tce = parse_tce_output(output_path, contents)
    section = tce.get("tce_sections", [{}])[0]
    n_electrons = section.get("n_electrons")
    n_frozen = section.get("frozen_cores") or 0
    n_corr = (n_electrons - 2 * n_frozen) if n_electrons is not None else None

    # Read T1 amplitudes
    with open(t1_file, "rb") as f:
        t1_raw = f.read()
    t1_vals = np.frombuffer(t1_raw, dtype="<f8")

    t1_norm = float(np.linalg.norm(t1_vals))
    t1_diag = float(t1_norm / np.sqrt(n_corr)) if n_corr else None

    # D1: max column norm of T1 matrix — only valid when the file contains
    # exactly n_occ * n_virt values (no-symmetry run).
    # n_orbitals from TCE output is spin-orbital count (spatial × 2)
    n_orb_so = section.get("n_orbitals")
    n_occ = ((n_electrons // 2) - n_frozen) if n_electrons is not None else None
    n_virt = (n_orb_so // 2 - (n_electrons // 2)) if (n_orb_so is not None and n_electrons is not None) else None
    d1_diag: float | None = None
    nosym = False
    if n_occ is not None and n_virt is not None and len(t1_vals) == n_occ * n_virt:
        nosym = True
        T1_mat = t1_vals.reshape(n_occ, n_virt)
        d1_diag = float(np.max(np.linalg.norm(T1_mat, axis=0)))

    # Read T2 amplitudes
    t2_norm: float | None = None
    t2_max: float | None = None
    t2_top10: list[float] = []
    t2_dominance: float | None = None
    t2_count_05: int | None = None
    t2_count_10: int | None = None
    t1_t2_singles_weight: float | None = None
    t2_norm_diagnostic: float | None = None

    if t2_file:
        with open(t2_file, "rb") as f:
            t2_raw = f.read()
        t2_vals = np.frombuffer(t2_raw, dtype="<f8")

        t2_norm = float(np.linalg.norm(t2_vals))
        t2_abs = np.abs(t2_vals)
        t2_max = float(t2_abs.max())

        # Top 10 magnitudes
        top_idx = np.argpartition(t2_abs, -min(10, len(t2_abs)))[-min(10, len(t2_abs)):]
        t2_top10 = sorted([float(t2_abs[i]) for i in top_idx], reverse=True)

        # Fraction of ||T2||^2 captured by top 10
        t2_norm_sq = t2_norm ** 2
        top10_sq = sum(v ** 2 for v in t2_top10)
        t2_dominance = float(top10_sq / t2_norm_sq) if t2_norm_sq > 0 else None

        # Counts above thresholds
        t2_count_05 = int(np.sum(t2_abs > 0.05))
        t2_count_10 = int(np.sum(t2_abs > 0.10))

        # Singles weight: ||T1||^2 / (||T1||^2 + ||T2||^2)
        denom = t1_norm ** 2 + t2_norm ** 2
        t1_t2_singles_weight = float(t1_norm ** 2 / denom) if denom > 0 else None

        # T2 normalized diagnostic (T2 analog of T1, normalized by electron pairs)
        n_pairs = n_corr * (n_corr - 1) if n_corr is not None and n_corr > 1 else None
        t2_norm_diagnostic = float(t2_norm / np.sqrt(n_pairs)) if n_pairs else None

    # Triples fraction: |E(T)| / |E_CCSD_corr| (from output, no amplitude file needed)
    triples_fraction: float | None = None
    ccsd_corr = tce.get("correlation_energy_hartree")
    total = tce.get("total_energy_hartree")
    ccsd_total_matches = list(_CCSD_TOTAL_RE.finditer(contents))
    ccsd_bracket_t = tce.get("ccsd_bracket_t_total_hartree")
    method = tce.get("method", "")
    if method == "CCSD(T)" and ccsd_corr is not None and ccsd_corr != 0:
        # CCSD(T) total - CCSD total = (T) correction
        if ccsd_total_matches and total is not None:
            ccsd_total_only = _parse_float(ccsd_total_matches[-1].group(1))
            if ccsd_total_only is not None:
                triples_energy = abs(total - ccsd_total_only)
                triples_fraction = float(triples_energy / abs(ccsd_corr))

    # MR assessment verdict
    flags: list[str] = []
    if t1_diag is not None and t1_diag > 0.05:
        flags.append(f"T1={t1_diag:.3f}>0.05_strong_mr")
    elif t1_diag is not None and t1_diag > 0.02:
        flags.append(f"T1={t1_diag:.3f}>0.02_moderate_mr")
    # D1 thresholds are higher than T1 — D1 is a max column norm, not a global norm,
    # and naturally takes larger values. Closed-shell: >0.10 strong, >0.05 moderate.
    if d1_diag is not None and d1_diag > 0.10:
        flags.append(f"D1={d1_diag:.3f}>0.10_strong_mr")
    elif d1_diag is not None and d1_diag > 0.05:
        flags.append(f"D1={d1_diag:.3f}>0.05_moderate_mr")
    if t2_max is not None and t2_max > 0.20:
        flags.append(f"max_t2={t2_max:.3f}>0.20_strong_mr")
    elif t2_max is not None and t2_max > 0.10:
        flags.append(f"max_t2={t2_max:.3f}>0.10_moderate_mr")
    if triples_fraction is not None and triples_fraction > 0.15:
        flags.append(f"triples_fraction={triples_fraction:.3f}>0.15_cc_hierarchy_suspect")
    elif triples_fraction is not None and triples_fraction > 0.05:
        flags.append(f"triples_fraction={triples_fraction:.3f}>0.05_triples_significant")

    strong_flags = [f for f in flags if "strong_mr" in f or "cc_hierarchy_suspect" in f]
    moderate_flags = [f for f in flags if "moderate_mr" in f or "triples_significant" in f]

    if strong_flags and (
        (t1_diag is not None and t1_diag > 0.05)
        or (triples_fraction is not None and triples_fraction > 0.15)
    ):
        mr_assessment = "unreliable_ccsd"
    elif strong_flags:
        mr_assessment = "strong_mr_character"
    elif moderate_flags:
        mr_assessment = "moderate_mr_character"
    else:
        mr_assessment = "single_reference_ok"

    return {
        "available": True,
        "t1_file": t1_file,
        "t2_file": t2_file,
        "nosym": nosym,
        "n_correlated_electrons": n_corr,
        "n_occ": n_occ,
        "n_virt": n_virt,
        # T1 diagnostics
        "t1_norm": round(t1_norm, 6),
        "t1_diagnostic": round(t1_diag, 6) if t1_diag is not None else None,
        "d1_diagnostic": round(d1_diag, 6) if d1_diag is not None else None,
        # T2 diagnostics
        "t2_frobenius_norm": round(t2_norm, 6) if t2_norm is not None else None,
        "t2_max_amplitude": round(t2_max, 6) if t2_max is not None else None,
        "t2_top_amplitudes": [round(v, 6) for v in t2_top10],
        "t2_dominance_fraction": round(t2_dominance, 4) if t2_dominance is not None else None,
        "t2_count_above_005": t2_count_05,
        "t2_count_above_010": t2_count_10,
        "t2_normalized_diagnostic": round(t2_norm_diagnostic, 6) if t2_norm_diagnostic is not None else None,
        # Cross diagnostics
        "t1_t2_singles_weight": round(t1_t2_singles_weight, 4) if t1_t2_singles_weight is not None else None,
        "triples_fraction": round(triples_fraction, 4) if triples_fraction is not None else None,
        # Verdict
        "mr_assessment": mr_assessment,
        "mr_flags": flags,
    }


# ---------------------------------------------------------------------------
# Binary movecs file I/O
# ---------------------------------------------------------------------------

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

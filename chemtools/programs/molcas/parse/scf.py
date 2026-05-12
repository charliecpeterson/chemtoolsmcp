"""Parser for the Molcas SCF (and KS-DFT) module output block.

Extracts:
  * Final SCF energy + decomposition (one-elec, two-elec, nuclear repulsion, kinetic, virial)
  * Spin S and S(S+1)
  * Per-symmetry orbital specifications (frozen / occupied / secondary / deleted)
  * Convergence status (was the iteration cycle finished cleanly?)
  * Reaction-field / solvation flag
  * Source of starting orbitals (GUESSORB / SCFORB / external)
"""

from __future__ import annotations

import re
from typing import Any


_FLOAT_RE = r"-?\d+\.\d+(?:[Ee][+-]?\d+)?"

# Final-energy line — matches both forms emitted by Molcas SCF:
#   "::    Total SCF energy                               -455.5542974319"   (HF/ROHF)
#   "::    Total KS-DFT energy                            -75.0239826189"    (KSDFT)
_TOTAL_SCF_RE = re.compile(
    r"::\s*Total\s+(?:SCF|KS-DFT)\s+energy\s+(" + _FLOAT_RE + r")"
)
_IS_KSDFT_RE = re.compile(r"::\s*Total\s+KS-DFT\s+energy")
_ONE_ELEC_RE = re.compile(r"^\s*One-electron\s+energy\s+(" + _FLOAT_RE + r")", re.M)
_TWO_ELEC_RE = re.compile(r"^\s*Two-electron\s+energy\s+(" + _FLOAT_RE + r")", re.M)
_NUC_REP_RE = re.compile(r"^\s*Nuclear\s+repulsion\s+energy\s+(" + _FLOAT_RE + r")", re.M)
_KINETIC_RE = re.compile(r"^\s*Kinetic\s+energy\s*\(interpolated\)\s+(" + _FLOAT_RE + r")", re.M)
_VIRIAL_RE = re.compile(r"^\s*Virial\s+theorem\s+(" + _FLOAT_RE + r")", re.M)
_SPIN_S_RE = re.compile(r"^\s*Total\s+spin,\s+S\s+(" + _FLOAT_RE + r")", re.M)
_SPIN_SS1_RE = re.compile(r"^\s*Total\s+spin,\s+S\(S\+1\)\s+(" + _FLOAT_RE + r")", re.M)

_SYM_HEADER_RE = re.compile(r"^\s*Symmetry species\s+(\d+(?:\s+\d+)*)\s*$", re.M)
_SYM_LABELS_RE = re.compile(r"^\s+([a-zA-Z0-9 ]+)\s*$")
_FROZEN_RE = re.compile(r"^\s*Frozen orbitals\s+(\d+(?:\s+\d+)*)\s*$", re.M)
_OCC_RE = re.compile(r"^\s*Occupied orbitals\s+(\d+(?:\s+\d+)*)\s*$", re.M)
_SEC_RE = re.compile(r"^\s*Secondary orbitals\s+(\d+(?:\s+\d+)*)\s*$", re.M)
_DEL_RE = re.compile(r"^\s*Deleted orbitals\s+(\d+(?:\s+\d+)*)\s*$", re.M)
_TOTAL_ORB_RE = re.compile(r"^\s*Total number of orbitals\s+(\d+(?:\s+\d+)*)\s*$", re.M)
_NBF_RE = re.compile(r"^\s*Number of basis functions\s+(\d+(?:\s+\d+)*)\s*$", re.M)

_REACTION_FIELD_RE = re.compile(r"Reaction Field calculation", re.I)
_GUESSORB_RE = re.compile(r"Detected guessorb starting orbitals", re.I)
_INPORB_RE = re.compile(r"^\s*The MO-coefficients are taken from the file:\s*\n\s*(\S+)", re.M)


def parse_scf(text: str) -> dict[str, Any]:
    """Parse one Molcas SCF module block (text already sliced to that block).

    Returns an empty `final_energy` dict when no Final Results block was emitted
    (job aborted), so callers can still see the orbital/spec metadata.
    """
    info: dict[str, Any] = {
        "module": "scf",
        "method": "KSDFT" if _IS_KSDFT_RE.search(text) else "SCF",
        "final_energy": {},
        "orbital_specs": {},
        "starting_orbitals": None,
        "reaction_field": False,
        "converged": None,
    }

    energy = info["final_energy"]
    if (m := _TOTAL_SCF_RE.search(text)):
        energy["total"] = float(m.group(1))
    if (m := _ONE_ELEC_RE.search(text)):
        energy["one_electron"] = float(m.group(1))
    if (m := _TWO_ELEC_RE.search(text)):
        energy["two_electron"] = float(m.group(1))
    if (m := _NUC_REP_RE.search(text)):
        energy["nuclear_repulsion"] = float(m.group(1))
    if (m := _KINETIC_RE.search(text)):
        energy["kinetic"] = float(m.group(1))
    if (m := _VIRIAL_RE.search(text)):
        energy["virial_theorem"] = float(m.group(1))
    if (m := _SPIN_S_RE.search(text)):
        energy["spin_S"] = float(m.group(1))
    if (m := _SPIN_SS1_RE.search(text)):
        energy["spin_S_S_plus_1"] = float(m.group(1))

    # Orbital spec block — values are space-separated integers, one per symmetry
    info["orbital_specs"] = _parse_orbital_specs(text)

    # Starting orbitals
    if _INPORB_RE.search(text):
        info["starting_orbitals"] = "INPORB"
    elif _GUESSORB_RE.search(text):
        info["starting_orbitals"] = "GuessOrb"

    info["reaction_field"] = bool(_REACTION_FIELD_RE.search(text))

    # Convergence: the Final Results banner only appears for converged runs.
    info["converged"] = bool(energy.get("total") is not None)

    # Also pull occupations vector from the input section (most informative even
    # for unconverged runs)
    occ_vector = _OCC_RE.search(text)
    if occ_vector:
        info["occupied_per_symmetry"] = [int(x) for x in occ_vector.group(1).split()]

    return info


def _parse_orbital_specs(text: str) -> dict[str, Any]:
    """The first ++Orbital specifications block under SCF prints a 4-row table:
        Symmetry species     1   2   3   4
                            a1  a2  b2  b1
        Frozen orbitals      0   0   0   0
        Occupied orbitals   20   2  12   5
        Secondary orbitals  53  17  41  24
        Deleted orbitals     0   0   0   0
        Total number...     73  19  53  29
        Number of basis...  73  19  53  29
    """
    specs: dict[str, Any] = {}
    sym_match = _SYM_HEADER_RE.search(text)
    if not sym_match:
        return specs
    sym_indices = [int(x) for x in sym_match.group(1).split()]
    specs["n_symmetries"] = len(sym_indices)
    specs["symmetry_indices"] = sym_indices
    # The next non-blank line after the header carries irrep labels.
    after = text[sym_match.end():]
    for line in after.splitlines()[:3]:
        stripped = line.strip()
        if stripped and not stripped.startswith("Frozen"):
            tokens = stripped.split()
            if len(tokens) == len(sym_indices):
                specs["irrep_labels"] = tokens
                break

    def _vec(pattern):
        m = pattern.search(text)
        if not m:
            return None
        try:
            values = [int(x) for x in m.group(1).split()]
        except ValueError:
            return None
        if len(values) != len(sym_indices):
            return values  # surface anyway; caller can flag mismatch
        return values

    if (v := _vec(_FROZEN_RE)) is not None:
        specs["frozen"] = v
    if (v := _vec(_OCC_RE)) is not None:
        specs["occupied"] = v
    if (v := _vec(_SEC_RE)) is not None:
        specs["secondary"] = v
    if (v := _vec(_DEL_RE)) is not None:
        specs["deleted"] = v
    if (v := _vec(_TOTAL_ORB_RE)) is not None:
        specs["total"] = v
    if (v := _vec(_NBF_RE)) is not None:
        specs["basis_functions"] = v
    return specs

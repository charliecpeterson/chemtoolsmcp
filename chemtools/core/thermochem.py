"""Program-agnostic thermochemistry helpers.

Currently exposes:

  - ``ATOMIC_MASSES_AMU``        — atomic mass table (Z=1..30 coverage)
  - ``atomic_ideal_gas_thermochem`` — Sackur-Tetrode translational entropy
                                  + electronic-degeneracy entropy for
                                  monoatomic species (vibrational and
                                  rotational contributions = 0)

Both used by Molcas's compute_reaction_energy when a reaction has atomic
species without parsed thermochem (atoms have no vibrations → Molcas
emits no thermochem block); the helper synthesizes ideal-gas thermochem
analytically. Future generic reaction-energy tools across all programs
will share this same code path.
"""

from __future__ import annotations

import math
from typing import Any

from chemtools.core.units import (
    HARTREE_TO_J_PER_MOL,
    HARTREE_TO_KCAL_PER_MOL,
    GAS_CONSTANT_J_PER_MOL_K,
    BOLTZMANN_J_PER_K,
    PLANCK_J_S,
    ATOMIC_MASS_KG,
    ATM_TO_PASCAL,
    KCAL_TO_J,
)


# Atomic mass table in amu (IUPAC standard atomic weights, rounded).
# Coverage: Z=1..30 — enough for ATOMIC_GROUND_STATES + the typical
# main-group and 1st-row TM atomization use cases. Extend as needed.
ATOMIC_MASSES_AMU: dict[str, float] = {
    "H": 1.00794, "He": 4.002602,
    "Li": 6.941, "Be": 9.012182, "B": 10.811, "C": 12.0107,
    "N": 14.0067, "O": 15.9994, "F": 18.9984032, "Ne": 20.1797,
    "Na": 22.98977, "Mg": 24.305, "Al": 26.9815386, "Si": 28.0855,
    "P": 30.973762, "S": 32.065, "Cl": 35.453, "Ar": 39.948,
    "K": 39.0983, "Ca": 40.078,
    "Sc": 44.955912, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961,
    "Mn": 54.938045, "Fe": 55.845, "Co": 58.933195, "Ni": 58.6934,
    "Cu": 63.546, "Zn": 65.38,
}


def atomic_ideal_gas_thermochem(
    element: str,
    multiplicity: int,
    temperature_k: float = 298.15,
    pressure_atm: float = 1.0,
) -> dict[str, Any]:
    """Translational (Sackur-Tetrode) + electronic ideal-gas thermochem for
    one atom. Returns a dict with the same shape as Molcas's parsed
    thermochem so it slots into compute_reaction_energy without special-
    casing.

    Vibrational + rotational contributions = 0 (atoms have neither). The
    electronic entropy is R × ln(2S+1) from the ground-state degeneracy
    given by the spin multiplicity.

    Sackur-Tetrode:
        S_trans = R × {ln[(2πmkT/h²)^1.5 × kT/p] + 5/2}
    """
    if element.capitalize() not in ATOMIC_MASSES_AMU:
        raise KeyError(
            f"No atomic mass for {element!r}; ATOMIC_MASSES_AMU covers Z=1..30."
        )
    p_pa = pressure_atm * ATM_TO_PASCAL
    T = temperature_k
    m = ATOMIC_MASSES_AMU[element.capitalize()] * ATOMIC_MASS_KG

    arg = (2 * math.pi * m * BOLTZMANN_J_PER_K * T / PLANCK_J_S**2) ** 1.5 \
        * (BOLTZMANN_J_PER_K * T / p_pa)
    S_trans_J = GAS_CONSTANT_J_PER_MOL_K * (math.log(arg) + 2.5)             # J/mol/K
    S_elec_J = GAS_CONSTANT_J_PER_MOL_K * math.log(max(multiplicity, 1))     # J/mol/K
    S_total_J = S_trans_J + S_elec_J

    U_trans_J = 1.5 * GAS_CONSTANT_J_PER_MOL_K * T                           # J/mol
    H_trans_J = 2.5 * GAS_CONSTANT_J_PER_MOL_K * T                           # J/mol

    J_to_au = 1.0 / HARTREE_TO_J_PER_MOL
    return {
        "zpve_au": 0.0,
        "zpve_kcal_per_mol": 0.0,
        "thermal_internal_energy_au": U_trans_J * J_to_au,
        "thermal_internal_energy_kcal_per_mol": U_trans_J * J_to_au * HARTREE_TO_KCAL_PER_MOL,
        "thermal_enthalpy_au": H_trans_J * J_to_au,
        "thermal_enthalpy_kcal_per_mol": H_trans_J * J_to_au * HARTREE_TO_KCAL_PER_MOL,
        "entropy_total_J_per_mol_K": S_total_J,
        "entropy_total_kcal_per_mol_K": S_total_J / KCAL_TO_J,
        "thermal_gibbs_au": (H_trans_J - T * S_total_J) * J_to_au,
        "thermal_gibbs_kcal_per_mol": (H_trans_J - T * S_total_J) * J_to_au * HARTREE_TO_KCAL_PER_MOL,
        "source": "ideal_gas_atomic",
        "components": {
            "S_trans_kcal_per_mol_K": S_trans_J / KCAL_TO_J,
            "S_elec_kcal_per_mol_K": S_elec_J / KCAL_TO_J,
            "S_rot_kcal_per_mol_K": 0.0,
            "S_vib_kcal_per_mol_K": 0.0,
        },
    }

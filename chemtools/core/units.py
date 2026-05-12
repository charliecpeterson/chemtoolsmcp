"""Shared physical-constant + unit-conversion tables.

Single source of truth so NWChem and Molcas (and future programs) all
agree on conversion factors — avoids drift if one module rounds to 5
sig figs and another to 7.

Constants are CODATA 2018 values where possible.
"""

from __future__ import annotations


# Energy conversions from atomic units (Hartree)
HARTREE_TO_KCAL_PER_MOL: float = 627.5094740631
HARTREE_TO_KJ_PER_MOL: float = 2625.4996394798
HARTREE_TO_EV: float = 27.211386245988
HARTREE_TO_J_PER_MOL: float = 2625500.2  # kJ * 1000
HARTREE_TO_CM1: float = 219474.6313632  # wavenumbers
HARTREE_TO_KELVIN: float = 315775.02480407

# Inverse conversions
KCAL_PER_MOL_TO_HARTREE: float = 1.0 / HARTREE_TO_KCAL_PER_MOL
EV_TO_HARTREE: float = 1.0 / HARTREE_TO_EV

# Length conversions
BOHR_PER_ANGSTROM: float = 1.8897261245650618
ANGSTROM_PER_BOHR: float = 1.0 / BOHR_PER_ANGSTROM

# Physical constants (SI)
GAS_CONSTANT_J_PER_MOL_K: float = 8.31446261815324
BOLTZMANN_J_PER_K: float = 1.380649e-23
PLANCK_J_S: float = 6.62607015e-34
AVOGADRO: float = 6.02214076e23
ATOMIC_MASS_KG: float = 1.66053906660e-27   # amu → kg
ATM_TO_PASCAL: float = 101325.0
KCAL_TO_J: float = 4184.0

"""Parser for the Molcas thermochemistry block.

Format (emitted by MCLR after the harmonic-frequency table):

    ZPVE                60.110 kcal/mol      0.095792 au.
    ZPVE corrected energy                 -192.790265 au.

    *****************************************************
    Temperature =     0.00 kelvin, Pressure =   1.00 atm
    -----------------------------------------------------
    Molecular Partition Function and Molar Entropy:
                           q/V (M**-3)    S(kcal/mol*K)
    Electronic            1.000000E+00        0.000
    Translational         1.000000E+00        0.000
    Rotational            1.000000E+00        2.981
    Vibrational           1.000000E+00        0.000
    TOTAL                 1.000000E+00        2.981

    Thermal contributions to INTERNAL ENERGY:
    Electronic           0.000 kcal/mol      0.000000 au.
    Translational        0.000 kcal/mol      0.000000 au.
    Rotational           0.000 kcal/mol      0.000000 au.
    Vibrational         60.110 kcal/mol      0.095792 au.
    TOTAL               60.110 kcal/mol      0.095792 au.

    Thermal contributions to
    ENTHALPY            60.110 kcal/mol      0.095792 au.
    GIBBS FREE ENERGY   60.110 kcal/mol      0.095792 au.

    Sum of energy and thermal contributions
    INTERNAL ENERGY                       -192.790265 au.
    ENTHALPY                              -192.790265 au.
    GIBBS FREE ENERGY                     -192.790265 au.
    -----------------------------------------------------

The block repeats for each temperature in the input deck (default Molcas set
is 0 / 100 / 273.15 / 298.15 / 323.15 / 373.15 / 473.15 K).
"""

from __future__ import annotations

import re
from typing import Any


_FLOAT_RE = r"-?\d+\.\d+(?:[Ee][+-]?\d+)?"

_THERMO_HEADER_RE = re.compile(r"\*+\s*\n\s*\*\s*THERMOCHEMISTRY\s*\*", re.M)
_ZPVE_RE = re.compile(r"^\s*ZPVE\s+(" + _FLOAT_RE + r")\s+kcal/mol\s+(" + _FLOAT_RE + r")\s+au\.", re.M)
_ZPVE_CORRECTED_RE = re.compile(r"^\s*ZPVE corrected energy\s+(" + _FLOAT_RE + r")\s+au\.", re.M)
_MASS_RE = re.compile(r"^\s*Molecular mass:\s+(" + _FLOAT_RE + r")", re.M)
_ROT_CONST_GHZ_RE = re.compile(r"^\s*Rotational Constants \(GHz\)\s*:\s+((?:" + _FLOAT_RE + r"\s*)+)", re.M)
_ROT_CONST_CM_RE = re.compile(r"^\s*Rotational Constants \(cm-1\)\s*:\s+((?:" + _FLOAT_RE + r"\s*)+)", re.M)
_ROT_SYM_RE = re.compile(r"^\s*Rotational Symmetry factor:\s+(\d+)", re.M)

_TEMP_BLOCK_START_RE = re.compile(
    r"^\s*Temperature\s*=\s*(" + _FLOAT_RE + r")\s*kelvin,\s*Pressure\s*=\s*(" + _FLOAT_RE + r")\s*atm",
    re.M,
)


def parse_thermochem_block(text: str) -> dict[str, Any] | None:
    """Parse the entire ++Thermochemistry section. Returns None if absent.

    Output:
        {
          "zpve_kcal_per_mol": 60.110,
          "zpve_au": 0.095792,
          "zpve_corrected_energy_au": -192.790265,
          "molecular_mass_amu": 66.046950,
          "rotational_constants_ghz": [...],
          "rotational_constants_cm1": [...],
          "rotational_symmetry_factor": 1,
          "temperatures_k": [0.0, 100.0, ..., 473.15],
          "per_temperature": [
              {"temperature_k": 0.0, "pressure_atm": 1.0,
               "entropy_kcal_per_mol_K": {"electronic": 0.0, "translational": 0.0, "rotational": 2.981, "vibrational": 0.0, "total": 2.981},
               "thermal_internal_energy_kcal_per_mol": {...}, "_au": {...},
               "thermal_enthalpy_kcal_per_mol": ..., "thermal_gibbs_kcal_per_mol": ...,
               "internal_energy_total_au": -192.790265, "enthalpy_au": ..., "gibbs_au": ...},
              ...
          ],
          "standard_298_15": {... pointer to the 298.15 K row, or closest one ...},
        }
    """
    if not _THERMO_HEADER_RE.search(text) and not _ZPVE_RE.search(text):
        return None

    info: dict[str, Any] = {}
    if (m := _ZPVE_RE.search(text)):
        info["zpve_kcal_per_mol"] = float(m.group(1))
        info["zpve_au"] = float(m.group(2))
    if (m := _ZPVE_CORRECTED_RE.search(text)):
        info["zpve_corrected_energy_au"] = float(m.group(1))
    if (m := _MASS_RE.search(text)):
        info["molecular_mass_amu"] = float(m.group(1))
    if (m := _ROT_CONST_GHZ_RE.search(text)):
        info["rotational_constants_ghz"] = [float(x) for x in m.group(1).split()]
    if (m := _ROT_CONST_CM_RE.search(text)):
        info["rotational_constants_cm1"] = [float(x) for x in m.group(1).split()]
    if (m := _ROT_SYM_RE.search(text)):
        info["rotational_symmetry_factor"] = int(m.group(1))

    # Per-temperature blocks
    temp_starts = list(_TEMP_BLOCK_START_RE.finditer(text))
    per_temp: list[dict[str, Any]] = []
    for i, m in enumerate(temp_starts):
        end = temp_starts[i + 1].start() if i + 1 < len(temp_starts) else len(text)
        block = text[m.start():end]
        per_temp.append(_parse_one_temperature_block(block, float(m.group(1)), float(m.group(2))))
    info["per_temperature"] = per_temp
    info["temperatures_k"] = [t["temperature_k"] for t in per_temp]

    # Closest-to-298.15 pointer
    if per_temp:
        std = min(per_temp, key=lambda t: abs(t["temperature_k"] - 298.15))
        info["standard_298_15"] = std

    return info


def _parse_one_temperature_block(block: str, temperature: float, pressure: float) -> dict[str, Any]:
    """Parse one Temperature= ... block."""
    out: dict[str, Any] = {
        "temperature_k": temperature,
        "pressure_atm": pressure,
    }

    # Entropy table (Molar Entropy column)
    entropy = _parse_entropy_table(block)
    if entropy:
        out["entropy_kcal_per_mol_K"] = entropy

    # Thermal contributions to INTERNAL ENERGY
    internal = _parse_two_column_table(block, "Thermal contributions to INTERNAL ENERGY:")
    if internal:
        out["thermal_internal_energy_kcal_per_mol"] = {k: v[0] for k, v in internal.items()}
        out["thermal_internal_energy_au"] = {k: v[1] for k, v in internal.items()}

    # ENTHALPY / GIBBS FREE ENERGY (single-line each)
    h_match = re.search(
        r"^\s*ENTHALPY\s+(" + _FLOAT_RE + r")\s+kcal/mol\s+(" + _FLOAT_RE + r")\s+au\.",
        block,
        re.M,
    )
    g_match = re.search(
        r"^\s*GIBBS FREE ENERGY\s+(" + _FLOAT_RE + r")\s+kcal/mol\s+(" + _FLOAT_RE + r")\s+au\.",
        block,
        re.M,
    )
    if h_match:
        out["thermal_enthalpy_kcal_per_mol"] = float(h_match.group(1))
        out["thermal_enthalpy_au"] = float(h_match.group(2))
    if g_match:
        out["thermal_gibbs_kcal_per_mol"] = float(g_match.group(1))
        out["thermal_gibbs_au"] = float(g_match.group(2))

    # Sum of energy and thermal contributions
    sum_block = re.search(
        r"Sum of energy and thermal contributions(.*?)(?=\*\*\*|\Z)",
        block,
        flags=re.DOTALL,
    )
    if sum_block:
        sub = sum_block.group(1)
        for label, key in (
            ("INTERNAL ENERGY", "internal_energy_total_au"),
            ("ENTHALPY", "enthalpy_total_au"),
            ("GIBBS FREE ENERGY", "gibbs_total_au"),
        ):
            m = re.search(rf"^\s*{label}\s+(" + _FLOAT_RE + r")\s+au\.", sub, re.M)
            if m:
                out[key] = float(m.group(1))
    return out


def _parse_entropy_table(block: str) -> dict[str, float] | None:
    """Lines in form `Electronic  1.000E+00  0.000` — col 2 is q/V, col 3 is S."""
    section = re.search(
        r"Molar Entropy:\s*\n\s*q/V \(M\*\*-3\)\s+S\(kcal/mol\*K\)\s*\n(.+?)(?=Thermal contributions)",
        block,
        flags=re.DOTALL,
    )
    if not section:
        return None
    out: dict[str, float] = {}
    for line in section.group(1).splitlines():
        m = re.match(
            r"^\s*([A-Z][A-Za-z ]*?)\s+(?:" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s*$",
            line,
        )
        if m:
            out[m.group(1).strip().lower()] = float(m.group(2))
    return out or None


def _parse_two_column_table(block: str, header: str) -> dict[str, tuple[float, float]] | None:
    """Lines in form `Electronic  0.000 kcal/mol  0.000000 au.`"""
    pattern = re.compile(
        re.escape(header) + r"(.+?)(?=Thermal contributions to|\*\*\*|Sum of energy)",
        re.DOTALL,
    )
    m = pattern.search(block)
    if not m:
        return None
    out: dict[str, tuple[float, float]] = {}
    for line in m.group(1).splitlines():
        row = re.match(
            r"^\s*([A-Z][A-Za-z]*?)\s+(" + _FLOAT_RE + r")\s+kcal/mol\s+(" + _FLOAT_RE + r")\s+au\.",
            line,
        )
        if row:
            out[row.group(1).strip().lower()] = (float(row.group(2)), float(row.group(3)))
    return out or None

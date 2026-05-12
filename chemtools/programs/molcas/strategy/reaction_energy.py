"""Reaction-energy computation + active-space consistency diagnostics.

Two related tools for multireference reaction-energy workflows (atomization,
binding, dissociation, isodesmic, etc.):

- ``compute_reaction_energy`` — post-hoc calculator. Takes converged Molcas
  outputs for products and reactants with signed stoichiometric coefficients
  and returns ΔE in au / kcal/mol / eV, plus optional ZPVE/thermal corrections.

- ``check_active_space_consistency`` — diagnostic. Compares a molecule's CAS
  spec to the summed CAS spec of its dissociation fragments and flags cases
  where the molecule's active space is too small to span the fragments — the
  classic CASSCF "negative binding energy" trap on transition-metal molecules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.core.common import read_text
from chemtools.programs.molcas.parse.output import parse_output_full
from chemtools.programs.molcas.parse.mos import parse_last_mo_block
from chemtools.programs.molcas.parse.thermochem import parse_thermochem_block
from chemtools.programs.molcas.parse.geometry import parse_final_geometry


HARTREE_TO_KCAL_PER_MOL = 627.5094740631
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_J_PER_MOL = 2625500.0
KCAL_TO_AU = 1.0 / HARTREE_TO_KCAL_PER_MOL
KCAL_PER_MOL_K_TO_AU_PER_K = KCAL_TO_AU  # entropy unit: kcal/mol/K → au/K


# Atomic mass table (amu), used for Sackur-Tetrode translational entropy on atomic
# species when their Molcas output has no parsed thermochem (atoms have no
# vibrations so Molcas can't emit a thermochem block).
_ATOMIC_MASSES_AMU = {
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


def _atomic_ideal_gas_thermochem(
    element: str,
    multiplicity: int,
    temperature_k: float,
    pressure_atm: float,
) -> dict[str, float]:
    """Translational (Sackur-Tetrode) + electronic ideal-gas thermochem for one
    atom in mol⁻¹ units. Vibrational and rotational contributions are zero.

    Outputs are in au unless suffixed (matches parse_molcas_thermochem shapes
    where convenient).

    S_trans = R × {ln[(2πmkT/h²)^1.5 × kT/p] + 5/2}
    S_elec  = R × ln(2S+1)               (assumes ground-state degeneracy = multiplicity)
    U_trans = (3/2)RT
    H_trans = (5/2)RT = U_trans + RT
    """
    import math
    R_J = 8.31446261815324  # J/mol/K
    k_B = 1.380649e-23      # J/K
    h = 6.62607015e-34      # J·s
    N_A = 6.02214076e23
    AMU = 1.66053906660e-27 # kg
    p_pa = pressure_atm * 101325.0
    T = temperature_k

    m = _ATOMIC_MASSES_AMU[element.capitalize()] * AMU

    # Sackur-Tetrode translational entropy
    arg = (2 * math.pi * m * k_B * T / h**2) ** 1.5 * (k_B * T / p_pa)
    S_trans_J = R_J * (math.log(arg) + 2.5)             # J/mol/K
    S_elec_J = R_J * math.log(multiplicity)             # J/mol/K, from ground-state degeneracy

    S_total_J = S_trans_J + S_elec_J
    U_trans_J = 1.5 * R_J * T                            # J/mol
    H_trans_J = 2.5 * R_J * T                            # J/mol = U + RT
    # G = H - TS (full thermodynamic), but for an atom with no ZPVE the
    # reference is just the electronic energy.

    # Convert to au
    J_to_au = 1.0 / HARTREE_TO_J_PER_MOL
    return {
        "zpve_au": 0.0,
        "zpve_kcal_per_mol": 0.0,
        "thermal_internal_energy_au": U_trans_J * J_to_au,         # contribution above E_elec
        "thermal_internal_energy_kcal_per_mol": U_trans_J * J_to_au * HARTREE_TO_KCAL_PER_MOL,
        "thermal_enthalpy_au": H_trans_J * J_to_au,
        "thermal_enthalpy_kcal_per_mol": H_trans_J * J_to_au * HARTREE_TO_KCAL_PER_MOL,
        "entropy_total_J_per_mol_K": S_total_J,
        "entropy_total_kcal_per_mol_K": S_total_J / 4184.0,
        "thermal_gibbs_au": (H_trans_J - T * S_total_J) * J_to_au,
        "thermal_gibbs_kcal_per_mol": (H_trans_J - T * S_total_J) * J_to_au * HARTREE_TO_KCAL_PER_MOL,
        "source": "ideal_gas_atomic",
        "components": {
            "S_trans_kcal_per_mol_K": S_trans_J / 4184.0,
            "S_elec_kcal_per_mol_K": S_elec_J / 4184.0,
            "S_rot_kcal_per_mol_K": 0.0,
            "S_vib_kcal_per_mol_K": 0.0,
        },
    }


def _extract_thermochem(
    parsed: dict,
    output_file: str,
    temperature_k: float,
    pressure_atm: float,
    species_multiplicity: int | None,
) -> dict[str, Any] | None:
    """Pull thermochem for one species. Tries Molcas's ++Thermochemistry block
    first; falls back to ideal-gas atomic if the species is monoatomic and has
    no parsed thermochem.

    Returns the standardized thermochem dict (zpve, thermal_h, thermal_g,
    entropy) or None if not derivable.
    """
    text = read_text(output_file)
    th = parse_thermochem_block(text)
    if th and th.get("standard_298_15"):
        std = th["standard_298_15"]
        entropy = std.get("entropy_kcal_per_mol_K") or {}
        return {
            "zpve_au": th.get("zpve_au", 0.0),
            "zpve_kcal_per_mol": th.get("zpve_kcal_per_mol", 0.0),
            "thermal_internal_energy_au": (std.get("thermal_internal_energy_au") or {}).get("total", 0.0),
            "thermal_internal_energy_kcal_per_mol": (std.get("thermal_internal_energy_kcal_per_mol") or {}).get("total", 0.0),
            "thermal_enthalpy_au": std.get("thermal_enthalpy_au", 0.0),
            "thermal_enthalpy_kcal_per_mol": std.get("thermal_enthalpy_kcal_per_mol", 0.0),
            "thermal_gibbs_au": std.get("thermal_gibbs_au", 0.0),
            "thermal_gibbs_kcal_per_mol": std.get("thermal_gibbs_kcal_per_mol", 0.0),
            "entropy_total_kcal_per_mol_K": (entropy.get("total") or 0.0) / 1000.0,
            "source": "parsed_freq",
            "components": entropy,
        }
    # No parsed thermochem — check if it's an atomic species
    geom = parse_final_geometry(text)
    atoms = (geom or {}).get("atoms") or []
    if len(atoms) == 1 and species_multiplicity:
        element = atoms[0]["symbol"]
        if element.capitalize() in _ATOMIC_MASSES_AMU:
            return _atomic_ideal_gas_thermochem(
                element, species_multiplicity, temperature_k, pressure_atm
            )
    return None


def _species_multiplicity(parsed: dict) -> int | None:
    """Extract spin multiplicity (2S+1) from a parsed Molcas output's RASSCF
    or SCF task."""
    for tp in parsed.get("task_payloads", []):
        if tp.get("module") in {"RASSCF", "SCF"}:
            wave = tp.get("details", {}).get("wave_function", {}) or {}
            spin = wave.get("spin")
            if spin is not None:
                return int(round(2 * float(spin) + 1))
    return None


# ---------------------------------------------------------------------------
# compute_reaction_energy
# ---------------------------------------------------------------------------


def _resolve_energy(parsed: dict, energy_kind: str) -> tuple[float, str]:
    """Pull the requested energy from a parsed Molcas output dict.

    energy_kind: one of "primary" (= primary_energy_hartree from energy_summary,
    follows CASPT2 > RASSCF > SCF) / "scf" / "rasscf" / "caspt2" /
    "rassi_so" / "rassi_sf".
    """
    es = parsed.get("energy_summary") or {}
    if energy_kind == "primary":
        e = es.get("primary_energy_hartree")
        label = es.get("primary_label") or "primary"
        if e is None:
            raise ValueError("Parsed output has no primary energy")
        return float(e), label
    # RASSCF / CASPT2 / etc. live as lists of {root, energy_hartree} under
    # *_root_energies; "rasscf" returns root 1 by convention.
    list_keymap = {
        "rasscf": ("rasscf_root_energies", "RASSCF root 1"),
        "caspt2": ("caspt2_root_energies", "CASPT2 root 1"),
        "ms_caspt2": ("ms_caspt2_root_energies", "MS-CASPT2 root 1"),
    }
    scalar_keymap = {
        "scf": ("scf_total_hartree", "SCF"),
        "rassi_sf": ("rassi_sf_ground_hartree", "RASSI-SF"),
        "rassi_so": ("rassi_so_ground_hartree", "RASSI-SO"),
    }
    if energy_kind in list_keymap:
        field, label = list_keymap[energy_kind]
        roots = es.get(field) or []
        if not roots:
            raise ValueError(
                f"Parsed output has no {label} energy (energy_summary[{field!r}] empty/missing)"
            )
        return float(roots[0]["energy_hartree"]), label
    if energy_kind in scalar_keymap:
        field, label = scalar_keymap[energy_kind]
        e = es.get(field)
        if e is None:
            raise ValueError(
                f"Parsed output has no {label} energy (energy_summary[{field!r}] missing)"
            )
        return float(e), label
    raise ValueError(
        f"Unknown energy_kind={energy_kind!r}. Use one of: "
        f"primary / scf / rasscf / caspt2 / ms_caspt2 / rassi_sf / rassi_so"
    )


def _parse_species(output_file: str) -> dict:
    text = read_text(output_file)
    return parse_output_full(output_file, text)


def compute_reaction_energy(
    *,
    products: list[dict],
    reactants: list[dict],
    energy_kind: str = "primary",
    label: str | None = None,
    include_thermochem: bool = False,
    temperature_k: float = 298.15,
    pressure_atm: float = 1.0,
) -> dict[str, Any]:
    """Compute a reaction energy from converged Molcas outputs.

    ΔE = Σ_products(coef_i × E_i) − Σ_reactants(coef_j × E_j)

    Each entry in ``products`` and ``reactants`` is a dict with:
      output_file: str            path to the Molcas .out / .log file
      coefficient: int|float      stoichiometric coefficient (positive)
      label:       str (optional) display label; default = basename of file

    For atomization energies the convention is:
      products = atomic references, reactants = molecule, ΔE = positive bound

    Parameters
    ----------
    energy_kind
        Which energy field to pull. "primary" uses the parser's hierarchy
        (CASPT2 > RASSCF > SCF). Pass "rasscf" or "caspt2" to force a specific
        level for consistency across species.
    include_thermochem
        If True, also compute ΔZPVE, D_0, ΔH(T), ΔG(T), ΔS(T). Pulls thermochem
        from each species' ++Thermochemistry block (Molcas MCLR freq calc).
        For monoatomic species lacking thermochem, falls back to ideal-gas
        Sackur-Tetrode (translational) + electronic-degeneracy entropy. If a
        polyatomic species has no thermochem, sets thermal_corrections=missing
        and emits a warning instead of failing.
    temperature_k, pressure_atm
        Used only for atomic ideal-gas fallback. The parsed Molcas thermochem
        is always read from the species' own output (whatever T was set there;
        defaults to 298.15 K, 1 atm in Molcas as well).
    label
        Optional name for the reaction (used in next_actions / summary).

    Returns dict with delta_e_au / delta_e_kcal_per_mol / delta_e_ev,
    a per-species ``breakdown``, and the chosen energy_kind/label. When
    include_thermochem=True, also returns ΔZPVE / D_0 / ΔH / ΔG / ΔS.
    """
    if not products or not reactants:
        return {
            "verdict": "missing_species",
            "error": "missing_species",
            "message": "Both products and reactants must be non-empty lists.",
        }

    breakdown_products: list[dict] = []
    breakdown_reactants: list[dict] = []
    species_labels: dict[str, str] = {}

    # Accumulators for signed thermochem contributions (Σ products − Σ reactants).
    # Each entry is {field: signed_au_sum}; we add coef × value on products and
    # subtract on reactants.
    thermo_accum = {"zpve_au": 0.0, "thermal_h_au": 0.0, "thermal_g_au": 0.0, "entropy_kcal_per_mol_K": 0.0}
    thermo_missing: list[str] = []  # species that should have thermochem but don't

    def _process(side: list[dict], out: list[dict], sign: int) -> float:
        total = 0.0
        for entry in side:
            of = entry["output_file"]
            coef = float(entry.get("coefficient", 1))
            if not Path(of).is_file():
                raise FileNotFoundError(f"Output file not found: {of}")
            parsed = _parse_species(of)
            e, e_label = _resolve_energy(parsed, energy_kind)
            disp = entry.get("label") or Path(of).stem
            species_labels[of] = disp
            contribution = coef * e
            record: dict[str, Any] = {
                "output_file": of,
                "label": disp,
                "coefficient": coef,
                "energy_au": e,
                "energy_label": e_label,
                "contribution_au": contribution,
            }
            if include_thermochem:
                mult = _species_multiplicity(parsed)
                th = _extract_thermochem(parsed, of, temperature_k, pressure_atm, mult)
                if th is None:
                    thermo_missing.append(disp)
                    record["thermochem"] = None
                else:
                    record["thermochem"] = {
                        "zpve_au": th["zpve_au"],
                        "zpve_kcal_per_mol": th["zpve_kcal_per_mol"],
                        "thermal_h_au": th["thermal_enthalpy_au"],
                        "thermal_g_au": th["thermal_gibbs_au"],
                        "entropy_kcal_per_mol_K": th.get("entropy_total_kcal_per_mol_K"),
                        "source": th["source"],
                    }
                    thermo_accum["zpve_au"] += sign * coef * th["zpve_au"]
                    thermo_accum["thermal_h_au"] += sign * coef * th["thermal_enthalpy_au"]
                    thermo_accum["thermal_g_au"] += sign * coef * th["thermal_gibbs_au"]
                    if th.get("entropy_total_kcal_per_mol_K") is not None:
                        thermo_accum["entropy_kcal_per_mol_K"] += sign * coef * th["entropy_total_kcal_per_mol_K"]
            out.append(record)
            total += contribution
        return total

    try:
        sum_products = _process(products, breakdown_products, sign=+1)
        sum_reactants = _process(reactants, breakdown_reactants, sign=-1)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "verdict": "parse_error",
            "error": str(exc),
            "message": str(exc),
        }

    dE_au = sum_products - sum_reactants
    dE_kcal = dE_au * HARTREE_TO_KCAL_PER_MOL
    dE_eV = dE_au * HARTREE_TO_EV

    # Heuristic interpretation: when reactants is a single molecule and
    # products are its atomic fragments, the reaction "molecule → atoms" has
    # ΔE = E(atoms) - E(molecule) = the binding / dissociation energy.
    # ΔE > 0 means dissociation costs energy → molecule is bound.
    # ΔE < 0 (atoms more stable than molecule) → unphysical for a real bond.
    is_atomization_like = (
        len(reactants) == 1 and len(products) > 1
        and abs(reactants[0].get("coefficient", 1) - 1.0) < 1e-6
    )

    result: dict[str, Any] = {
        "verdict": "ok",
        "label": label,
        "energy_kind": energy_kind,
        "delta_e_au": dE_au,
        "delta_e_kcal_per_mol": dE_kcal,
        "delta_e_ev": dE_eV,
        "sum_products_au": sum_products,
        "sum_reactants_au": sum_reactants,
        "breakdown": {
            "products": breakdown_products,
            "reactants": breakdown_reactants,
        },
    }
    if is_atomization_like:
        # For the reaction "molecule → atoms" with products=atoms,
        # reactants=molecule: delta_e IS the binding/dissociation energy
        # (positive for bound, negative for unphysical "unbound" results).
        result["binding_energy_au"] = dE_au
        result["binding_energy_kcal_per_mol"] = dE_kcal
        result["binding_energy_ev"] = dE_eV
        result["is_bound"] = dE_au > 0
        result["atomization_interpretation"] = (
            "Reaction written as molecule → atoms; ΔE > 0 means dissociation "
            "costs energy (molecule is bound). delta_e and binding_energy are "
            "identical here. ΔE < 0 (unbound) usually indicates an active-space "
            "mismatch — run check_active_space_consistency."
        )

    # Thermochem deltas (when requested + all species have thermochem)
    if include_thermochem:
        if thermo_missing:
            result["thermochem_status"] = "missing_for_some_species"
            result["thermochem_missing_species"] = thermo_missing
            result["thermochem_note"] = (
                f"Cannot compute thermal corrections — no thermochem block found for: "
                f"{', '.join(thermo_missing)}. Run a frequency calculation "
                f"(`prepare_molcas_opt_freq_workflow`) on these species, or accept "
                f"the electronic-only ΔE."
            )
        else:
            d_zpve_au = thermo_accum["zpve_au"]
            d_h_au = thermo_accum["thermal_h_au"]
            d_g_au = thermo_accum["thermal_g_au"]
            d_s_kcal_K = thermo_accum["entropy_kcal_per_mol_K"]

            d_h_total_au = dE_au + d_zpve_au + d_h_au   # ΔH(T) = ΔE_elec + ΔZPVE + Δ(H_thermal)
            d_g_total_au = dE_au + d_zpve_au + d_g_au   # ΔG(T) = ΔE_elec + ΔZPVE + Δ(G_thermal)
            d0_au = dE_au + d_zpve_au                   # ZPVE-corrected (no thermal)

            result["thermochem_status"] = "ok"
            result["temperature_k"] = temperature_k
            result["pressure_atm"] = pressure_atm
            result["delta_zpve_au"] = d_zpve_au
            result["delta_zpve_kcal_per_mol"] = d_zpve_au * HARTREE_TO_KCAL_PER_MOL
            result["d0_au"] = d0_au
            result["d0_kcal_per_mol"] = d0_au * HARTREE_TO_KCAL_PER_MOL
            result["d0_ev"] = d0_au * HARTREE_TO_EV
            result["delta_thermal_h_au"] = d_h_au
            result["delta_h_au"] = d_h_total_au
            result["delta_h_kcal_per_mol"] = d_h_total_au * HARTREE_TO_KCAL_PER_MOL
            result["delta_h_ev"] = d_h_total_au * HARTREE_TO_EV
            result["delta_thermal_g_au"] = d_g_au
            result["delta_g_au"] = d_g_total_au
            result["delta_g_kcal_per_mol"] = d_g_total_au * HARTREE_TO_KCAL_PER_MOL
            result["delta_g_ev"] = d_g_total_au * HARTREE_TO_EV
            result["delta_s_kcal_per_mol_K"] = d_s_kcal_K
            result["thermochem_interpretation"] = (
                "ΔE_elec = electronic; D_0 = ΔE_elec + ΔZPVE (0 K); "
                "ΔH(T), ΔG(T), ΔS(T) at the specified temperature. For "
                "atomization (molecule → atoms), positive ΔH means dissociation "
                "is endothermic at T; ΔG > 0 means the molecule is bound at T."
            )
    return result


# ---------------------------------------------------------------------------
# check_active_space_consistency
# ---------------------------------------------------------------------------


def _extract_cas(parsed: dict) -> dict | None:
    """Pull the active-space dimensions from a parsed Molcas output's RASSCF
    task. Returns None if no RASSCF task is found.
    """
    for tp in parsed.get("task_payloads", []):
        if tp.get("module") == "RASSCF":
            details = tp.get("details", {})
            specs = details.get("orbital_specs", {}) or {}
            wave = details.get("wave_function", {}) or {}
            return {
                "n_active_electrons": wave.get("active_electrons"),
                "n_active_orbitals": sum(specs.get("active", []) or [0]),
                "n_inactive_orbitals": sum(specs.get("inactive", []) or [0]),
                "n_frozen_orbitals": sum(specs.get("frozen", []) or [0]),
                "n_basis_functions": sum(specs.get("basis_functions", []) or [0]),
                "spin": wave.get("spin"),
                "state_symmetry": wave.get("state_symmetry"),
                "details": details,
            }
    return None


def check_active_space_consistency(
    *,
    molecule_output: str,
    fragments: list[dict],
    target_character_atom: str | None = None,
    target_character_ao: str | None = None,
) -> dict[str, Any]:
    """Compare a molecule's CAS spec to the sum of its dissociation fragments'
    CAS specs. Flags the "molecule CAS too small to span fragment CASes" trap
    that produces unphysical (negative) binding energies at CASSCF.

    Parameters
    ----------
    molecule_output
        Path to the converged molecule .out / .log.
    fragments
        List of dicts, each with:
          output_file: str        path to fragment .out
          stoichiometry: int      how many of this fragment (default 1)
          label: str (optional)   display name
    target_character_atom, target_character_ao
        If both given, also report how many active orbitals in the molecule
        carry character matching (atom, ao) and compare to the count across
        fragments. Useful for "should there be N Cr 3d orbitals?" checks.

    Returns dict with:
      verdict                "consistent" | "molecule_undersized" | "fragments_oversized" | "char_mismatch"
      molecule_cas           {n_active_electrons, n_active_orbitals, n_inactive}
      fragments_cas          per-fragment dimensions
      fragments_sum          sum across all fragments × stoichiometry
      delta_electrons        molecule - sum_fragments (signed)
      delta_orbitals         molecule - sum_fragments (signed)
      suggested_cas          (M, N) suggestion if undersized
      character_analysis     (optional) target-character orbital counts
      rationale              human-readable summary
      next_actions           agent-actionable list
    """
    mol_parsed = _parse_species(molecule_output)
    mol_cas = _extract_cas(mol_parsed)
    if mol_cas is None:
        return {
            "verdict": "no_molecule_rasscf",
            "error": "no_rasscf_task",
            "message": f"No RASSCF task in {molecule_output}",
        }

    frag_specs: list[dict] = []
    sum_active_e = 0
    sum_active_o = 0
    for f in fragments:
        of = f["output_file"]
        stoich = int(f.get("stoichiometry", 1))
        parsed = _parse_species(of)
        cas = _extract_cas(parsed)
        if cas is None:
            return {
                "verdict": "no_fragment_rasscf",
                "error": "no_rasscf_task",
                "message": f"No RASSCF task in fragment output {of}",
            }
        cas["stoichiometry"] = stoich
        cas["output_file"] = of
        cas["label"] = f.get("label") or Path(of).stem
        frag_specs.append(cas)
        sum_active_e += stoich * (cas["n_active_electrons"] or 0)
        sum_active_o += stoich * (cas["n_active_orbitals"] or 0)

    delta_e = (mol_cas["n_active_electrons"] or 0) - sum_active_e
    delta_o = (mol_cas["n_active_orbitals"] or 0) - sum_active_o

    if delta_o == 0 and delta_e == 0:
        verdict = "consistent"
        rationale = (
            f"Molecule CAS({mol_cas['n_active_electrons']}e, "
            f"{mol_cas['n_active_orbitals']}o) matches the sum of fragment CASes."
        )
        suggested_cas: tuple[int, int] | None = None
    elif delta_o < 0 or delta_e < 0:
        verdict = "molecule_undersized"
        rationale = (
            f"Molecule CAS({mol_cas['n_active_electrons']}e, "
            f"{mol_cas['n_active_orbitals']}o) is SMALLER than the sum of fragment "
            f"CASes ({sum_active_e}e, {sum_active_o}o). The molecule cannot "
            f"dissociate cleanly into the chosen atomic references — CASSCF will "
            f"give an unphysical (often negative) reaction energy. Expand the "
            f"molecule's CAS to at least ({sum_active_e}e, {sum_active_o}o) "
            f"OR shrink the atomic CASes to match."
        )
        suggested_cas = (sum_active_e, sum_active_o)
    else:
        # Molecule larger than fragments — usually means atomic refs are
        # over-truncated. Less harmful (gives non-negative binding) but still
        # inconsistent.
        verdict = "fragments_undersized"
        rationale = (
            f"Molecule CAS({mol_cas['n_active_electrons']}e, "
            f"{mol_cas['n_active_orbitals']}o) is LARGER than the sum of fragment "
            f"CASes ({sum_active_e}e, {sum_active_o}o). Atomic references may be "
            f"missing orbitals that the molecule treats as active (e.g. 4s on "
            f"a TM atom). Binding energy will be overestimated."
        )
        suggested_cas = None  # rebuild fragment CASes, not the molecule

    result: dict[str, Any] = {
        "verdict": verdict,
        "molecule_output": molecule_output,
        "molecule_cas": {
            "n_active_electrons": mol_cas["n_active_electrons"],
            "n_active_orbitals": mol_cas["n_active_orbitals"],
            "n_inactive_orbitals": mol_cas["n_inactive_orbitals"],
            "n_basis_functions": mol_cas["n_basis_functions"],
            "spin": mol_cas["spin"],
        },
        "fragments_cas": [
            {
                "label": fs["label"],
                "output_file": fs["output_file"],
                "stoichiometry": fs["stoichiometry"],
                "n_active_electrons": fs["n_active_electrons"],
                "n_active_orbitals": fs["n_active_orbitals"],
                "n_inactive_orbitals": fs["n_inactive_orbitals"],
                "spin": fs["spin"],
            }
            for fs in frag_specs
        ],
        "fragments_sum": {
            "n_active_electrons": sum_active_e,
            "n_active_orbitals": sum_active_o,
        },
        "delta_electrons": delta_e,
        "delta_orbitals": delta_o,
        "suggested_cas": suggested_cas,
        "rationale": rationale,
        "next_actions": [],
    }

    # Optional character check
    if target_character_atom and target_character_ao:
        mol_count = _count_character_active(
            molecule_output, mol_cas, target_character_atom, target_character_ao
        )
        frag_count = 0
        for fs in frag_specs:
            c = _count_character_active(
                fs["output_file"], fs, target_character_atom, target_character_ao
            )
            frag_count += fs["stoichiometry"] * c
        result["character_analysis"] = {
            "target": f"{target_character_atom} {target_character_ao}",
            "molecule_count": mol_count,
            "fragments_sum_count": frag_count,
            "consistent": mol_count == frag_count,
        }
        if mol_count != frag_count:
            # Don't override more critical verdict; just narrow it
            if verdict == "consistent":
                result["verdict"] = "char_mismatch"
            result["rationale"] += (
                f" Character check: molecule has {mol_count} "
                f"'{target_character_atom} {target_character_ao}' active orbitals "
                f"but fragments sum to {frag_count}."
            )

    if verdict == "molecule_undersized":
        sm, sn = suggested_cas  # type: ignore[misc]
        result["next_actions"].append(
            {
                "tool": "prepare_molcas_casscf_setup",
                "args": {
                    "cas_active_electrons": sm,
                    "cas_active_orbitals": sn,
                },
                "rationale": (
                    f"Re-draft the molecule with CAS({sm}, {sn}) so it can "
                    f"dissociate cleanly into the fragment CASes."
                ),
            }
        )
    elif verdict == "fragments_undersized":
        result["next_actions"].append(
            {
                "tool": "prepare_molcas_casscf_setup",
                "rationale": (
                    "Rebuild the fragment references with larger CASes so the "
                    "sum matches the molecule's active orbitals."
                ),
            }
        )
    elif verdict == "consistent":
        result["next_actions"].append(
            {
                "tool": "compute_reaction_energy",
                "rationale": (
                    "Active-space dimensions are consistent — compute the "
                    "reaction energy from the converged outputs."
                ),
            }
        )
    return result


def _count_character_active(
    output_file: str, cas: dict, atom_pat: str, ao_pat: str
) -> int:
    """Count how many active orbitals in this output have dominant AO
    character matching (atom_pat, ao_pat). Case-insensitive substring match.
    """
    text = read_text(output_file)
    mo = parse_last_mo_block(text, parse_coefficients=True)
    if not mo or not mo.get("symmetry_blocks"):
        return 0

    atom_lower = atom_pat.lower()
    ao_lower = ao_pat.lower()

    # Determine active-orbital index range across all symmetries
    details = cas.get("details", {})
    specs = details.get("orbital_specs", {}) or {}
    frozen = specs.get("frozen") or [0]
    inactive = specs.get("inactive") or [0]
    active = specs.get("active") or [0]

    count = 0
    for sym_idx, sym in enumerate(mo["symmetry_blocks"]):
        f = frozen[sym_idx] if sym_idx < len(frozen) else 0
        i = inactive[sym_idx] if sym_idx < len(inactive) else 0
        a = active[sym_idx] if sym_idx < len(active) else 0
        first_active = f + i + 1  # 1-indexed
        last_active = f + i + a
        for orb in sym.get("orbitals", []):
            idx = orb.get("orbital_index", 0)
            if not (first_active <= idx <= last_active):
                continue
            top_ao = (orb.get("dominant_aos") or [{}])[0]
            atom = (top_ao.get("atom") or "").lower()
            ao = (top_ao.get("ao_label") or "").lower()
            if atom_lower in atom and ao_lower in ao:
                count += 1
    return count

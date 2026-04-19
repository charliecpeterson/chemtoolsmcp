from __future__ import annotations

from typing import Any

from .common import detect_program, make_metadata, read_text
from .cube import parse_cube_file, summarize_cube_file
from .diagnostics import (
    analyze_frontier_orbitals as analyze_nwchem_frontier_orbitals,
    diagnose_nwchem_output,
    parse_scf,
    suggest_vectors_swaps as suggest_nwchem_vectors_swaps,
    summarize_nwchem_output,
)
from .nwchem_input import inspect_nwchem_input
from .nwchem_tce import parse_tce_output as _parse_tce_output
from . import molcas, molpro, nwchem


def _dispatch_parse_mos(
    path: str, contents: str, top_n: int = 5, include_coefficients: bool = False,
    include_all_orbitals: bool = False,
) -> dict[str, Any]:
    """Dispatch parse_mos to the correct program module."""
    program = detect_program(contents)
    if program == "molpro":
        return molpro.parse_mos(path, contents, top_n=top_n, include_coefficients=include_coefficients)
    return nwchem.parse_mos(
        path, contents, top_n=top_n, include_coefficients=include_coefficients,
        include_all_orbitals=include_all_orbitals,
    )


def parse_tasks(path: str) -> dict[str, Any]:
    contents = read_text(path)
    program = detect_program(contents)
    if program == "nwchem":
        return nwchem.parse_tasks(path, contents)
    if program == "molpro":
        return molpro.parse_tasks(path, contents)
    if program == "molcas":
        return molcas.parse_tasks(path, contents)
    return nwchem.parse_tasks(path, contents)


def parse_mos(
    path: str, top_n: int = 5, include_coefficients: bool = False,
    include_all_orbitals: bool = False,
) -> dict[str, Any]:
    contents = read_text(path)
    return _dispatch_parse_mos(
        path, contents, top_n=top_n, include_coefficients=include_coefficients,
        include_all_orbitals=include_all_orbitals,
    )


def parse_population_analysis(path: str) -> dict[str, Any]:
    contents = read_text(path)
    return nwchem.parse_population_analysis(path, contents)


def parse_mcscf_output(path: str) -> dict[str, Any]:
    contents = read_text(path)
    return nwchem.parse_mcscf_output(path, contents)


def parse_freq(path: str, include_displacements: bool = False) -> dict[str, Any]:
    contents = read_text(path)
    return nwchem.parse_freq(path, contents, include_displacements=include_displacements)


def parse_freq_progress(path: str) -> dict[str, Any]:
    contents = read_text(path)
    return nwchem.parse_freq_progress(path, contents)


def parse_trajectory(path: str, include_positions: bool = False) -> dict[str, Any]:
    contents = read_text(path)
    return nwchem.parse_trajectory(path, contents, include_positions=include_positions)


def analyze_frontier_orbitals(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
) -> dict[str, Any]:
    contents = read_text(output_path)
    program = detect_program(contents)
    if program != "nwchem":
        raise ValueError(f"frontier orbital analysis is not implemented for {program or 'unknown'}")

    mos = nwchem.parse_mos(output_path, contents, top_n=8)
    population = nwchem.parse_population_analysis(output_path, contents)
    input_summary = inspect_nwchem_input(input_path) if input_path else None
    metals = expected_metal_elements or (input_summary["transition_metals"] if input_summary else [])
    somo_target = expected_somo_count
    if somo_target is None and input_summary and input_summary["multiplicity"] and input_summary["multiplicity"] > 1:
        somo_target = input_summary["multiplicity"] - 1

    analysis = analyze_nwchem_frontier_orbitals(
        mos,
        population_payload=population,
        expected_metal_elements=metals,
        expected_somo_count=somo_target,
    )
    return {
        "metadata": make_metadata(output_path, contents, "nwchem"),
        "input_summary": input_summary,
        "expected_metal_elements": metals,
        "expected_somo_count": somo_target,
        "analysis": analysis,
    }


def suggest_vectors_swaps(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
) -> dict[str, Any]:
    contents = read_text(output_path)
    program = detect_program(contents)
    if program != "nwchem":
        raise ValueError(f"vectors swap suggestions are not implemented for {program or 'unknown'}")

    mos = nwchem.parse_mos(output_path, contents, top_n=8)
    input_summary = inspect_nwchem_input(input_path) if input_path else None
    metals = expected_metal_elements or (input_summary["transition_metals"] if input_summary else [])
    somo_target = expected_somo_count
    if somo_target is None and input_summary and input_summary["multiplicity"] and input_summary["multiplicity"] > 1:
        somo_target = input_summary["multiplicity"] - 1

    payload = suggest_nwchem_vectors_swaps(
        mos,
        expected_metal_elements=metals,
        expected_somo_count=somo_target,
        vectors_input=vectors_input,
        vectors_output=vectors_output,
    )
    return {
        "metadata": make_metadata(output_path, contents, "nwchem"),
        "input_summary": input_summary,
        "expected_metal_elements": metals,
        "expected_somo_count": somo_target,
        "suggestion": payload,
    }


def parse_cube(path: str, include_values: bool = False) -> dict[str, Any]:
    return parse_cube_file(path, include_values=include_values)


def summarize_cube(path: str, top_atoms: int = 5) -> dict[str, Any]:
    return summarize_cube_file(path, top_atoms=top_atoms)


def parse_output(
    path: str,
    sections: list[str] | None = None,
    top_n: int = 5,
    include_coefficients: bool = False,
    include_displacements: bool = False,
    include_positions: bool = False,
) -> dict[str, Any]:
    contents = read_text(path)
    metadata = make_metadata(path, contents)
    selected = sections or ["tasks"]

    output = {
        "metadata": metadata,
        "tasks": None,
        "mos": None,
        "population": None,
        "mcscf": None,
        "frequency": None,
        "trajectory": None,
        "errors": [],
    }

    for section in selected:
        try:
            if section == "tasks":
                output["tasks"] = parse_tasks(path)
            elif section == "mos":
                output["mos"] = parse_mos(path, top_n=top_n, include_coefficients=include_coefficients)
            elif section == "population":
                output["population"] = parse_population_analysis(path)
            elif section == "mcscf":
                output["mcscf"] = parse_mcscf_output(path)
            elif section == "freq":
                output["frequency"] = parse_freq(path, include_displacements=include_displacements)
            elif section == "trajectory":
                output["trajectory"] = parse_trajectory(path, include_positions=include_positions)
            else:
                output["errors"].append(f"unknown section: {section}")
        except Exception as exc:  # pragma: no cover
            output["errors"].append(f"{section}: {exc}")

    return output


def diagnose_output(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
) -> dict[str, Any]:
    return diagnose_nwchem_output(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
    )


def summarize_output(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    detail_level: str = "summary",
) -> dict[str, Any]:
    return summarize_nwchem_output(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
        detail_level=detail_level,
    )


def parse_scf_output(path: str) -> dict[str, Any]:
    """Alias for parse_scf (diagnostics module)."""
    return parse_scf(path)


def parse_tce_output(path: str) -> dict[str, Any]:
    """Parse NWChem TCE output: energies, frozen orbital counts, convergence."""
    contents = read_text(path)
    return _parse_tce_output(path, contents)


_HARTREE_TO_KCAL = 627.5094740631
_HARTREE_TO_EV = 27.211386245988
_CAL_TO_KCAL = 0.001


def parse_nwchem_thermochem(
    path: str,
    T: float = 298.15,
    P: float = 1.0,
) -> dict[str, Any]:
    """Parse thermochemistry from an NWChem frequency output.

    Combines the electronic energy from the last DFT/SCF task with the
    thermochemical corrections (ZPE, thermal enthalpy, entropy) from the
    frequency analysis to produce H(T) and G(T).

    Parameters
    ----------
    path : str
        Path to the NWChem frequency output file.
    T : float
        Temperature in Kelvin (used only for reporting; NWChem computes
        corrections at the temperature specified in the input).
    P : float
        Pressure in atm (for reporting only).

    Returns
    -------
    dict with E_scf, ZPE, H(T), G(T), S, Cv, and warnings.
    """
    from .nwchem_tasks import parse_tasks as _parse_tasks
    from .nwchem_freq import parse_freq as _parse_freq_raw

    contents = read_text(path)
    warnings: list[str] = []

    # --- Get electronic energy ---
    tasks_result = _parse_tasks(path, contents)
    task_list = (
        tasks_result.get("program_summary", {}).get("raw", {}).get("tasks", []) or []
    )
    # Use the last completed task energy (the freq task's SCF energy)
    e_scf: float | None = None
    method_used: str | None = None
    for t in reversed(task_list):
        e = t.get("total_energy_hartree")
        if e is not None:
            e_scf = e
            method_used = t.get("method", "SCF")
            break

    if e_scf is None:
        warnings.append("No converged electronic energy found in output.")

    # --- Get thermochemistry corrections ---
    freq_data = _parse_freq_raw(path, contents)
    thermo = freq_data.get("thermochemistry")
    if thermo is None:
        return {
            "error": "No thermochemistry section found — is this a frequency output?",
            "warnings": warnings,
        }

    zpe = thermo.get("zero_point_correction") or {}
    thermal_enthalpy = thermo.get("thermal_correction_enthalpy") or {}
    total_entropy = thermo.get("total_entropy_cal_mol_k")
    nwchem_T = thermo.get("temperature_kelvin", T)

    zpe_hartree = zpe.get("hartree")
    zpe_kcal = zpe.get("kcal_mol")
    h_corr_hartree = thermal_enthalpy.get("hartree")
    h_corr_kcal = thermal_enthalpy.get("kcal_mol")

    # Check for imaginary modes
    n_imaginary = freq_data.get("significant_imaginary_mode_count", 0)
    if n_imaginary > 0:
        imag_freqs = freq_data.get("significant_imaginary_frequencies_cm1", [])
        warnings.append(
            f"{n_imaginary} imaginary mode(s) detected ({', '.join(f'{f:.1f}' for f in imag_freqs[:5])} cm⁻¹). "
            "Thermochemistry may be unreliable — this is likely a saddle point, not a minimum."
        )

    # Temperature mismatch check
    if abs(nwchem_T - T) > 0.1:
        warnings.append(
            f"Requested T={T} K but NWChem computed corrections at {nwchem_T} K. "
            "Using NWChem's temperature."
        )

    # --- Compute composite quantities ---
    # E + ZPE
    e_plus_zpe: float | None = None
    if e_scf is not None and zpe_hartree is not None:
        e_plus_zpe = e_scf + zpe_hartree

    # H(T) = E_scf + H_thermal_correction (which already includes ZPE + PV)
    h_T: float | None = None
    if e_scf is not None and h_corr_hartree is not None:
        h_T = e_scf + h_corr_hartree

    # G(T) = H(T) - T*S
    g_T: float | None = None
    ts_hartree: float | None = None
    if h_T is not None and total_entropy is not None:
        ts_kcal = nwchem_T * total_entropy * _CAL_TO_KCAL  # kcal/mol
        ts_hartree = ts_kcal / _HARTREE_TO_KCAL
        g_T = h_T - ts_hartree

    return {
        "T_K": nwchem_T,
        "P_atm": P,
        "method": method_used,
        "E_scf_hartree": e_scf,
        "ZPE_hartree": zpe_hartree,
        "ZPE_kcal_mol": zpe_kcal,
        "E_plus_ZPE_hartree": e_plus_zpe,
        "H_thermal_correction_hartree": h_corr_hartree,
        "H_thermal_correction_kcal_mol": h_corr_kcal,
        "H_T_hartree": h_T,
        "S_total_cal_mol_K": total_entropy,
        "S_translational_cal_mol_K": thermo.get("translational_entropy_cal_mol_k"),
        "S_rotational_cal_mol_K": thermo.get("rotational_entropy_cal_mol_k"),
        "S_vibrational_cal_mol_K": thermo.get("vibrational_entropy_cal_mol_k"),
        "TS_hartree": ts_hartree,
        "G_T_hartree": g_T,
        "Cv_total_cal_mol_K": thermo.get("cv_total_cal_mol_k"),
        "Cv_translational_cal_mol_K": thermo.get("cv_translational_cal_mol_k"),
        "Cv_rotational_cal_mol_K": thermo.get("cv_rotational_cal_mol_k"),
        "Cv_vibrational_cal_mol_K": thermo.get("cv_vibrational_cal_mol_k"),
        "molecular_weight": thermo.get("molecular_weight"),
        "symmetry_number": thermo.get("symmetry_number"),
        "linear_molecule": thermo.get("linear_molecule", False),
        "imaginary_modes_count": n_imaginary,
        "frequency_scaling_parameter": thermo.get("frequency_scaling_parameter"),
        "warnings": warnings,
    }


def compute_reaction_energy(
    species: dict[str, str],
    reactants: dict[str, float],
    products: dict[str, float],
    method: str | None = None,
    include_thermochem: bool = False,
) -> dict[str, Any]:
    """Compute a reaction energy from a set of NWChem output files.

    Collects the best available energy for each species (preferring the
    highest-level method: CCSD(T) > CCSD > MP2 > DFT > SCF) and computes:

        ΔE = Σ coeff_i · E_i(products) − Σ coeff_i · E_i(reactants)

    When ``include_thermochem=True``, also extracts thermochemical corrections
    (ZPE, H(T), G(T)) from frequency outputs and computes ΔH and ΔG.
    Species without frequency data contribute only ΔE; a warning is emitted.

    Parameters
    ----------
    species:
        Dict mapping label → output file path.
        Example: ``{"FeO2-": "feo2.out", "Fe": "fe.out", "O": "o.out"}``
    reactants:
        Dict mapping label → stoichiometric coefficient (positive integers).
        Example: ``{"FeO2-": 1}``
    products:
        Dict mapping label → stoichiometric coefficient (positive integers).
        Example: ``{"Fe": 1, "O": 2}``
    method:
        If provided, only use energies from this method level (e.g. ``"CCSD"``).
        If None (default), uses the highest-level converged energy per species.
    include_thermochem:
        If True, extract ZPE, H(T), G(T) from each species (requires freq output)
        and compute ΔE+ZPE, ΔH(T), ΔG(T) in addition to ΔE.

    Returns
    -------
    dict with:
      ``delta_e_hartree``, ``delta_e_kcal_mol``, ``delta_e_ev``,
      ``species_breakdown``, ``formula``, ``warnings``.
      When ``include_thermochem=True``, also: ``delta_e_plus_zpe_*``,
      ``delta_h_*``, ``delta_g_*``.
    """
    from .nwchem_tasks import parse_tasks as _parse_tasks_nwchem

    def parse_tasks(path: str) -> dict[str, Any]:
        contents = read_text(path)
        return _parse_tasks_nwchem(path, contents)

    from .nwchem_freq import parse_freq as _parse_freq_raw

    _METHOD_PRIORITY = {"CCSD(T)": 5, "CCSD": 4, "MP2": 3, "DFT": 2, "SCF": 1}
    _norm_method = (method or "").strip().upper()

    species_energies: dict[str, float | None] = {}
    method_used: dict[str, str | None] = {}
    # Thermochem per species: H(T) and G(T) in Hartree
    species_h: dict[str, float | None] = {}
    species_g: dict[str, float | None] = {}
    species_e_zpe: dict[str, float | None] = {}
    warnings: list[str] = []

    all_labels = set(reactants) | set(products)
    for label in all_labels:
        if label not in species:
            warnings.append(f"Species '{label}' not found in species dict.")
            species_energies[label] = None
            method_used[label] = None
            continue

        out_file = species[label]
        try:
            tasks_result = parse_tasks(out_file)
        except Exception as exc:
            warnings.append(f"Could not parse '{label}' ({out_file}): {exc}")
            species_energies[label] = None
            method_used[label] = None
            continue

        # Tasks are nested under program_summary.raw.tasks in the parse_tasks output
        task_list = (
            tasks_result.get("program_summary", {}).get("raw", {}).get("tasks", []) or []
        )

        # Collect (method, energy) pairs from all completed tasks
        candidates: list[tuple[str, float]] = []
        for t in task_list:
            e = t.get("total_energy_hartree")
            m = (t.get("method") or "SCF").strip().upper()
            if e is not None:
                candidates.append((m, e))

        # Also try TCE output directly
        try:
            from .nwchem_tce import parse_tce_output as _ptce
            tce = _ptce(out_file, read_text(out_file))
            if tce.get("total_energy_hartree") is not None:
                candidates.append((tce["method"] or "TCE", tce["total_energy_hartree"]))
        except Exception:
            pass

        if not candidates:
            warnings.append(f"No converged energy found for '{label}' in {out_file}.")
            species_energies[label] = None
            method_used[label] = None
            continue

        if _norm_method:
            # Filter to requested method
            filtered = [(m, e) for m, e in candidates if m == _norm_method]
            if not filtered:
                warnings.append(
                    f"No '{_norm_method}' energy for '{label}'; "
                    f"available methods: {sorted({m for m, _ in candidates})}."
                )
                filtered = candidates
            candidates = filtered

        # Pick highest-priority method
        best_m, best_e = max(candidates, key=lambda me: _METHOD_PRIORITY.get(me[0], 0))
        species_energies[label] = best_e
        method_used[label] = best_m

        # --- Thermochem extraction (if requested) ---
        if include_thermochem:
            try:
                contents_tc = read_text(out_file)
                freq_data = _parse_freq_raw(out_file, contents_tc)
                thermo = freq_data.get("thermochemistry")
                if thermo is not None:
                    zpe_h = (thermo.get("zero_point_correction") or {}).get("hartree")
                    h_corr = (thermo.get("thermal_correction_enthalpy") or {}).get("hartree")
                    s_total = thermo.get("total_entropy_cal_mol_k")
                    T_nw = thermo.get("temperature_kelvin", 298.15)

                    if zpe_h is not None and best_e is not None:
                        species_e_zpe[label] = best_e + zpe_h
                    if h_corr is not None and best_e is not None:
                        species_h[label] = best_e + h_corr
                    if h_corr is not None and s_total is not None and best_e is not None:
                        ts = T_nw * s_total * _CAL_TO_KCAL / _HARTREE_TO_KCAL
                        species_g[label] = best_e + h_corr - ts

                    # Warn about imaginary modes
                    n_imag = freq_data.get("significant_imaginary_mode_count", 0)
                    if n_imag > 0:
                        warnings.append(
                            f"Species '{label}' has {n_imag} imaginary mode(s) — "
                            "thermochemistry may be unreliable."
                        )
                else:
                    warnings.append(
                        f"No thermochemistry found for '{label}' — "
                        "ΔH/ΔG will be incomplete."
                    )
            except Exception:
                warnings.append(
                    f"Could not parse thermochemistry for '{label}'."
                )

    # --- Compute ΔE ---
    delta_e: float | None = None
    missing = [lbl for lbl in all_labels if species_energies.get(lbl) is None]
    if not missing:
        delta_e = 0.0
        for lbl, coeff in products.items():
            delta_e += coeff * species_energies[lbl]  # type: ignore[operator]
        for lbl, coeff in reactants.items():
            delta_e -= coeff * species_energies[lbl]  # type: ignore[operator]
    else:
        warnings.append(f"ΔE cannot be computed; missing energies for: {', '.join(missing)}.")

    # --- Format human-readable formula ---
    def _fmt_side(d: dict[str, float]) -> str:
        parts = []
        for lbl, c in d.items():
            parts.append(f"{int(c)}·{lbl}" if c != 1 else lbl)
        return " + ".join(parts)

    formula_str = f"{_fmt_side(reactants)} → {_fmt_side(products)}"

    # --- Compute thermochem deltas ---
    def _delta(values: dict[str, float | None]) -> float | None:
        if any(lbl not in values or values[lbl] is None for lbl in all_labels):
            return None
        d = 0.0
        for lbl, coeff in products.items():
            d += coeff * values[lbl]  # type: ignore[operator]
        for lbl, coeff in reactants.items():
            d -= coeff * values[lbl]  # type: ignore[operator]
        return d

    delta_e_zpe = _delta(species_e_zpe) if include_thermochem else None
    delta_h = _delta(species_h) if include_thermochem else None
    delta_g = _delta(species_g) if include_thermochem else None

    # Build per-species breakdown
    breakdown: list[dict[str, Any]] = []
    for lbl in all_labels:
        coeff_reactant = reactants.get(lbl, 0)
        coeff_product = products.get(lbl, 0)
        net_coeff = coeff_product - coeff_reactant
        e = species_energies.get(lbl)
        entry: dict[str, Any] = {
            "label": lbl,
            "output_file": species.get(lbl),
            "method": method_used.get(lbl),
            "energy_hartree": e,
            "stoich_reactant": coeff_reactant,
            "stoich_product": coeff_product,
            "net_coefficient": net_coeff,
            "contribution_hartree": (net_coeff * e) if e is not None else None,
        }
        if include_thermochem:
            entry["E_plus_ZPE_hartree"] = species_e_zpe.get(lbl)
            entry["H_T_hartree"] = species_h.get(lbl)
            entry["G_T_hartree"] = species_g.get(lbl)
        breakdown.append(entry)
    breakdown.sort(key=lambda x: (x["stoich_reactant"] != 0, x["label"]))

    result: dict[str, Any] = {
        "formula": formula_str,
        "delta_e_hartree": delta_e,
        "delta_e_kcal_mol": (delta_e * _HARTREE_TO_KCAL) if delta_e is not None else None,
        "delta_e_ev": (delta_e * _HARTREE_TO_EV) if delta_e is not None else None,
        "species_breakdown": breakdown,
        "method_requested": method or "auto",
        "warnings": warnings,
    }

    if include_thermochem:
        result["delta_e_plus_zpe_hartree"] = delta_e_zpe
        result["delta_e_plus_zpe_kcal_mol"] = (
            delta_e_zpe * _HARTREE_TO_KCAL if delta_e_zpe is not None else None
        )
        result["delta_h_hartree"] = delta_h
        result["delta_h_kcal_mol"] = (
            delta_h * _HARTREE_TO_KCAL if delta_h is not None else None
        )
        result["delta_g_hartree"] = delta_g
        result["delta_g_kcal_mol"] = (
            delta_g * _HARTREE_TO_KCAL if delta_g is not None else None
        )

    return result

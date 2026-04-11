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


def compute_reaction_energy(
    species: dict[str, str],
    reactants: dict[str, float],
    products: dict[str, float],
    method: str | None = None,
) -> dict[str, Any]:
    """Compute a reaction energy from a set of NWChem output files.

    Collects the best available energy for each species (preferring the
    highest-level method: CCSD(T) > CCSD > MP2 > DFT > SCF) and computes:

        ΔE = Σ coeff_i · E_i(products) − Σ coeff_i · E_i(reactants)

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

    Returns
    -------
    dict with:
      ``delta_e_hartree``, ``delta_e_kcal_mol``, ``delta_e_ev``,
      ``species_energies``, ``method_used_per_species``, ``formula_str``,
      ``warnings``.
    """
    from .nwchem_tasks import parse_tasks as _parse_tasks_nwchem

    def parse_tasks(path: str) -> dict[str, Any]:
        contents = read_text(path)
        return _parse_tasks_nwchem(path, contents)

    _METHOD_PRIORITY = {"CCSD(T)": 5, "CCSD": 4, "MP2": 3, "DFT": 2, "SCF": 1}
    _norm_method = (method or "").strip().upper()

    species_energies: dict[str, float | None] = {}
    method_used: dict[str, str | None] = {}
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

        task_list = tasks_result.get("tasks", []) or []

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

    # Build per-species breakdown
    breakdown: list[dict[str, Any]] = []
    for lbl in all_labels:
        coeff_reactant = reactants.get(lbl, 0)
        coeff_product = products.get(lbl, 0)
        net_coeff = coeff_product - coeff_reactant
        e = species_energies.get(lbl)
        breakdown.append({
            "label": lbl,
            "output_file": species.get(lbl),
            "method": method_used.get(lbl),
            "energy_hartree": e,
            "stoich_reactant": coeff_reactant,
            "stoich_product": coeff_product,
            "net_coefficient": net_coeff,
            "contribution_hartree": (net_coeff * e) if e is not None else None,
        })
    breakdown.sort(key=lambda x: (x["stoich_reactant"] != 0, x["label"]))

    return {
        "formula": formula_str,
        "delta_e_hartree": delta_e,
        "delta_e_kcal_mol": (delta_e * _HARTREE_TO_KCAL) if delta_e is not None else None,
        "delta_e_ev": (delta_e * _HARTREE_TO_EV) if delta_e is not None else None,
        "species_breakdown": breakdown,
        "method_requested": method or "auto",
        "warnings": warnings,
    }

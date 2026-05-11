"""NWChem MCSCF / CASSCF active-space selection advisor.

Builds a recommended CAS(M,N) window from an SCF reference output:

  * Spatial-orbital grouping (alpha+beta -> spatial)
  * Per-orbital character classification (d-like / f-like / metal- vs
    ligand-centered)
  * Frontier-orbital state check (matches expected metal-centered SOMO
    pattern?)
  * Minimal / expanded active-space candidate scoring
  * Swap_in / swap_out candidate flagging for non-active orbitals

Pairs with `chemtools/programs/nwchem/strategy/active_space.py::prepare_nwchem_mcscf_setup`
which wraps this advisor with the Diagnosis envelope and next_actions
routing.
"""

from __future__ import annotations
from typing import Any

from chemtools.core.common import read_text, detect_program, make_metadata
from chemtools.programs.nwchem.parse.input import inspect_nwchem_input
from chemtools.programs.nwchem.parse.mos import (
    parse_mos,
    parse_population_analysis,
)
from chemtools.programs.nwchem.strategy.diagnose import (
    analyze_frontier_orbitals as analyze_nwchem_frontier_orbitals,
)


def suggest_nwchem_mcscf_active_space(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
) -> dict[str, Any]:
    contents = read_text(output_path)
    program = detect_program(contents)
    if program != "nwchem":
        raise ValueError(f"MCSCF active-space suggestions are not implemented for {program or 'unknown'}")

    mos = parse_mos(output_path, contents, top_n=8)
    population = parse_population_analysis(output_path, contents)
    input_summary = inspect_nwchem_input(input_path) if input_path else None
    metal_elements = expected_metal_elements or (input_summary["transition_metals"] if input_summary else [])
    metal_set = {element.lower() for element in metal_elements}
    somo_target = expected_somo_count
    if somo_target is None and input_summary and input_summary["multiplicity"] and input_summary["multiplicity"] > 1:
        somo_target = input_summary["multiplicity"] - 1

    frontier = analyze_nwchem_frontier_orbitals(
        mos,
        population_payload=population,
        expected_metal_elements=metal_elements,
        expected_somo_count=somo_target,
    )
    grouped_orbitals = _build_mcscf_spatial_orbitals(mos, metal_set)
    total_occupied_spatial = sum(1 for item in grouped_orbitals if item["classification"] != "virtual")
    occupied = [item for item in grouped_orbitals if item["classification"] in {"doubly_occupied", "singly_occupied"}]
    occupied.sort(key=lambda item: item["reference_energy_hartree"], reverse=True)
    virtual = [item for item in grouped_orbitals if item["classification"] == "virtual"]
    virtual.sort(key=lambda item: item["reference_energy_hartree"])

    singlies = [item for item in grouped_orbitals if item["classification"] == "singly_occupied"]
    ligand_hole_candidate = frontier.get("assessment") == "metal_state_mismatch_suspected" and (
        frontier.get("ligand_like_somo_count", 0) > frontier.get("metal_like_somo_count", 0)
    )

    occupied_candidates = [
        item
        for item in occupied
        if item["classification"] == "doubly_occupied"
        and (
            item["metal_like"]
            or item["d_like"]
            or item["f_like"]
            or item["character_class"] == "mixed_metal_ligand"
            or (ligand_hole_candidate and item["character_class"].startswith("ligand_centered_pi"))
        )
    ]
    virtual_candidates = [
        item
        for item in virtual
        if (
            item["metal_like"]
            or item["d_like"]
            or item["f_like"]
            or item["character_class"] == "mixed_metal_ligand"
            or (ligand_hole_candidate and item["character_class"].startswith("ligand_centered_pi"))
        )
    ]
    occupied_candidates = occupied_candidates[:12]
    virtual_candidates = virtual_candidates[:12]
    occupied_candidates.sort(key=_mcscf_candidate_score, reverse=True)
    virtual_candidates.sort(key=_mcscf_candidate_score, reverse=True)

    minimal_target = max(6, len(singlies) + 2)
    if metal_set:
        minimal_target = max(minimal_target, 8 if len(singlies) <= 4 else 10)
    expanded_target = minimal_target + (2 if metal_set else 0)

    minimal = _select_mcscf_active_space(
        grouped_orbitals=grouped_orbitals,
        singly_occupied=singlies,
        occupied_candidates=occupied_candidates,
        virtual_candidates=virtual_candidates,
        target_orbitals=minimal_target,
        total_occupied_spatial=total_occupied_spatial,
    )
    expanded = _select_mcscf_active_space(
        grouped_orbitals=grouped_orbitals,
        singly_occupied=singlies,
        occupied_candidates=occupied_candidates,
        virtual_candidates=virtual_candidates,
        target_orbitals=expanded_target,
        total_occupied_spatial=total_occupied_spatial,
    )

    frontier_vectors = {
        orbital["vector_number"]
        for orbital in frontier.get("somos", [])
    }
    frontier_channels = frontier.get("frontier_channels") or {}
    for channel_payload in frontier_channels.values():
        for key in ("homo", "lumo"):
            orbital = channel_payload.get(key)
            if orbital:
                frontier_vectors.add(orbital["vector_number"])

    active_minimal_vectors = set(minimal["vector_numbers"])
    swap_in_candidates = [
        item
        for item in occupied_candidates + virtual_candidates
        if item["vector_number"] not in active_minimal_vectors
    ][:6]
    swap_out_candidates = [
        item
        for item in grouped_orbitals
        if item["vector_number"] in frontier_vectors
        and item["vector_number"] not in set(item2["vector_number"] for item2 in singlies)
        and (
            item["character_class"].startswith("ligand_centered")
            or (not item["metal_like"] and not item["d_like"] and not item["f_like"])
        )
    ][:6]

    notes: list[str] = []
    if ligand_hole_candidate:
        notes.append("ligand_hole_or_covalent_high_spin_signals_detected")
    if frontier.get("assessment") == "somo_count_mismatch":
        notes.append("frontier_somo_count_mismatch_makes_active_space_less_certain")
    if not metal_set:
        notes.append("no_expected_metal_elements_supplied_or_inferred")

    return {
        "metadata": make_metadata(output_path, contents, "nwchem"),
        "input_summary": input_summary,
        "expected_metal_elements": metal_elements,
        "expected_somo_count": somo_target,
        "frontier_assessment": frontier.get("assessment"),
        "ligand_hole_candidate": ligand_hole_candidate,
        "orbital_count": len(grouped_orbitals),
        "singly_occupied_vectors": [item["vector_number"] for item in singlies],
        "minimal_active_space": minimal,
        "expanded_active_space": expanded,
        "swap_in_candidates": swap_in_candidates,
        "swap_out_candidates": swap_out_candidates,
        "candidate_pool": {
            "occupied": occupied_candidates[:8],
            "virtual": virtual_candidates[:8],
        },
        "notes": notes,
    }


def _build_mcscf_spatial_orbitals(
    mos_payload: dict[str, Any],
    metal_set: set[str],
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for orbital in mos_payload.get("orbitals", []):
        vector = orbital["vector_number"]
        entry = grouped.setdefault(
            vector,
            {
                "vector_number": vector,
                "alpha_occupancy": 0.0,
                "beta_occupancy": 0.0,
                "alpha_energy_hartree": None,
                "beta_energy_hartree": None,
                "reference_orbital": None,
            },
        )
        spin = orbital.get("spin")
        if spin == "beta":
            entry["beta_occupancy"] = orbital["occupancy"]
            entry["beta_energy_hartree"] = orbital["energy_hartree"]
        else:
            entry["alpha_occupancy"] = orbital["occupancy"]
            entry["alpha_energy_hartree"] = orbital["energy_hartree"]
        reference = entry["reference_orbital"]
        if reference is None or orbital["occupancy"] > reference["occupancy"]:
            entry["reference_orbital"] = orbital

    orbitals: list[dict[str, Any]] = []
    for vector, entry in grouped.items():
        reference = entry["reference_orbital"]
        if reference is None:
            continue
        summary = _summarize_active_space_orbital(reference, metal_set)
        total_occ = (entry["alpha_occupancy"] or 0.0) + (entry["beta_occupancy"] or 0.0)
        if entry["alpha_occupancy"] > 0.1 and entry["beta_occupancy"] > 0.1:
            classification = "doubly_occupied"
        elif total_occ > 0.1:
            classification = "singly_occupied"
        else:
            classification = "virtual"
        energies = [
            value for value in (entry["alpha_energy_hartree"], entry["beta_energy_hartree"]) if value is not None
        ]
        reference_energy = sum(energies) / len(energies) if energies else reference["energy_hartree"]
        orbitals.append(
            {
                "vector_number": vector,
                "alpha_occupancy": entry["alpha_occupancy"],
                "beta_occupancy": entry["beta_occupancy"],
                "total_occupancy": total_occ,
                "classification": classification,
                "reference_energy_hartree": reference_energy,
                "alpha_energy_hartree": entry["alpha_energy_hartree"],
                "beta_energy_hartree": entry["beta_energy_hartree"],
                **summary,
            }
        )
    orbitals.sort(key=lambda item: item["reference_energy_hartree"], reverse=True)
    return orbitals


def _summarize_active_space_orbital(orbital: dict[str, Any], metal_set: set[str]) -> dict[str, Any]:
    top_atoms = orbital.get("top_atom_contributions") or []
    metal_fraction = sum(
        item.get("fraction_of_visible", 0.0)
        for item in top_atoms
        if item.get("element", "").lower() in metal_set
    )
    ligand_fraction = sum(
        item.get("fraction_of_visible", 0.0)
        for item in top_atoms
        if item.get("element", "").lower() not in metal_set
    )
    shell_contributions = {
        item["ao_shell"]: item.get("fraction_of_visible", 0.0)
        for item in (orbital.get("ao_shell_contributions") or [])
    }
    d_fraction = shell_contributions.get("d", 0.0)
    f_fraction = shell_contributions.get("f", 0.0)
    p_fraction = shell_contributions.get("p", 0.0)
    s_fraction = shell_contributions.get("s", 0.0)
    if metal_fraction >= 0.6 and d_fraction >= 0.35:
        character_class = "metal_centered_d"
    elif metal_fraction >= 0.6 and f_fraction >= 0.25:
        character_class = "metal_centered_f"
    elif metal_fraction >= 0.6:
        character_class = "metal_centered_mixed"
    elif metal_fraction >= 0.3 and ligand_fraction >= 0.3:
        character_class = "mixed_metal_ligand"
    elif p_fraction >= 0.45:
        character_class = "ligand_centered_pi"
    elif s_fraction >= 0.45:
        character_class = "ligand_centered_sigma"
    else:
        character_class = "ligand_centered_mixed"
    return {
        "spin_reference": orbital.get("spin"),
        "symmetry": orbital.get("symmetry"),
        "dominant_character": orbital.get("dominant_character"),
        "top_atom_contributions": top_atoms,
        "ao_shell_contributions": orbital.get("ao_shell_contributions") or [],
        "metal_fraction": metal_fraction,
        "ligand_fraction": ligand_fraction,
        "d_fraction": d_fraction,
        "f_fraction": f_fraction,
        "p_fraction": p_fraction,
        "s_fraction": s_fraction,
        "metal_like": metal_fraction >= 0.5,
        "ligand_like": metal_fraction < 0.35,
        "d_like": d_fraction >= 0.35,
        "f_like": f_fraction >= 0.25,
        "character_class": character_class,
    }


def _mcscf_candidate_score(orbital: dict[str, Any]) -> tuple[float, float, float, float]:
    frontier_bonus = -abs(orbital["reference_energy_hartree"])
    return (
        orbital["metal_fraction"] + orbital["d_fraction"] + orbital["f_fraction"],
        1.0 if orbital["classification"] == "singly_occupied" else 0.0,
        1.0 if orbital["character_class"] == "mixed_metal_ligand" else 0.0,
        frontier_bonus,
    )


def _select_mcscf_active_space(
    *,
    grouped_orbitals: list[dict[str, Any]],
    singly_occupied: list[dict[str, Any]],
    occupied_candidates: list[dict[str, Any]],
    virtual_candidates: list[dict[str, Any]],
    target_orbitals: int,
    total_occupied_spatial: int,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    selected_vectors: set[int] = set()

    def add_orbital(orbital: dict[str, Any]) -> None:
        if orbital["vector_number"] in selected_vectors:
            return
        selected_vectors.add(orbital["vector_number"])
        selected.append(orbital)

    for orbital in singly_occupied:
        add_orbital(orbital)

    toggle = 0
    occupied_index = 0
    virtual_index = 0
    while len(selected) < target_orbitals and (occupied_index < len(occupied_candidates) or virtual_index < len(virtual_candidates)):
        if toggle % 2 == 0 and occupied_index < len(occupied_candidates):
            add_orbital(occupied_candidates[occupied_index])
            occupied_index += 1
        elif virtual_index < len(virtual_candidates):
            add_orbital(virtual_candidates[virtual_index])
            virtual_index += 1
        elif occupied_index < len(occupied_candidates):
            add_orbital(occupied_candidates[occupied_index])
            occupied_index += 1
        toggle += 1

    selected.sort(key=lambda item: item["reference_energy_hartree"], reverse=True)
    electron_count = int(round(sum(item["total_occupancy"] for item in selected)))
    occupied_count = sum(1 for item in selected if item["classification"] != "virtual")
    virtual_count = sum(1 for item in selected if item["classification"] == "virtual")
    closed_shell_count = max(0, total_occupied_spatial - occupied_count)
    return {
        "active_electrons": electron_count,
        "active_orbitals": len(selected),
        "occupied_like_count": occupied_count,
        "virtual_like_count": virtual_count,
        "closed_shell_count": closed_shell_count,
        "vector_numbers": [item["vector_number"] for item in selected],
        "orbitals": selected,
    }



__all__ = ["suggest_nwchem_mcscf_active_space"]

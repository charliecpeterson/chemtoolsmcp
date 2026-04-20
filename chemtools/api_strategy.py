from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .common import detect_program, make_metadata, read_text, ELEMENT_TO_Z
from .diagnostics import (
    analyze_frontier_orbitals as analyze_nwchem_frontier_orbitals,
    diagnose_nwchem_output,
    parse_scf,
    suggest_vectors_swaps as suggest_nwchem_vectors_swaps,
    summarize_nwchem_output,
)
from .nwchem_input import inspect_nwchem_input
from . import nwchem
from ._api_utils import _TRANSITION_METALS, _COVALENT_RADII, _strategy_entry, _coerce_api_int, _coerce_api_float
from .api_output import parse_mcscf_output


def check_spin_charge_state(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
) -> dict[str, Any]:
    diagnosis = diagnose_nwchem_output(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
    )
    input_summary = diagnosis["input_summary"]
    state_check = diagnosis["state_check"]
    multiplicity = input_summary["multiplicity"] if input_summary else None
    charge = input_summary["charge"] if input_summary else None
    inferred_expected_somos = expected_somo_count
    if inferred_expected_somos is None and multiplicity is not None and multiplicity > 1:
        inferred_expected_somos = multiplicity - 1

    reasons: list[str] = []
    assessment = "unavailable"
    confidence = diagnosis["confidence"]

    if not state_check["available"]:
        if state_check["assessment"] != "unavailable":
            assessment = "suspicious"
            reasons.append(state_check["assessment"])
        else:
            reasons.append("frontier_state_analysis_unavailable")
    else:
        if multiplicity == 1 and state_check["somo_count"] > 0:
            assessment = "suspicious"
            reasons.append("singlet_input_but_open_shell_somos_found")
        elif inferred_expected_somos is not None and state_check["somo_count"] != inferred_expected_somos:
            assessment = "suspicious"
            reasons.append("somo_count_does_not_match_multiplicity_or_expected_state")
        elif state_check["assessment"] in {"metal_state_mismatch_suspected", "somo_count_mismatch"}:
            assessment = "suspicious"
            reasons.append(state_check["assessment"])
        else:
            assessment = "plausible"
            reasons.append("frontier_and_spin_signals_are_consistent_with_requested_state")

    dominant_site = (state_check.get("spin_density_summary") or {}).get("dominant_site")
    if dominant_site is not None:
        reasons.append(
            f"dominant_spin_density_on_{dominant_site['element']}{dominant_site['atom_index']}"
        )

    return {
        "output_file": output_path,
        "input_file": input_path,
        "assessment": assessment,
        "confidence": confidence,
        "charge": charge,
        "multiplicity": multiplicity,
        "expected_somo_count": inferred_expected_somos,
        "observed_somo_count": state_check.get("somo_count"),
        "metal_like_somo_count": state_check.get("metal_like_somo_count"),
        "ligand_like_somo_count": state_check.get("ligand_like_somo_count"),
        "state_check_assessment": state_check.get("assessment"),
        "dominant_spin_site": dominant_site,
        "reasons": reasons,
        "recommended_next_action": diagnosis["recommended_next_action"],
        "input_summary": input_summary,
    }


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

    mos = nwchem.parse_mos(output_path, contents, top_n=8)
    population = nwchem.parse_population_analysis(output_path, contents)
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


def suggest_nwchem_scf_fix_strategy(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
) -> dict[str, Any]:
    diagnosis = diagnose_nwchem_output(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
    )
    scf = diagnosis.get("scf") or {}
    last_run = scf.get("last_run") or {}
    trend = ((last_run.get("trend") or {}).get("pattern")) or ((scf.get("trend") or {}).get("pattern")) or "unknown"
    iteration_count = last_run.get("iteration_count") or scf.get("iteration_count") or 0
    hit_max = bool(last_run.get("hit_max_iterations") or scf.get("hit_max_iterations"))
    failure_class = diagnosis.get("failure_class")
    state_check = diagnosis.get("state_check") or {}

    strategies: list[dict[str, Any]] = []
    notes: list[str] = []
    strategy_family = "review_only"

    if failure_class == "no_clear_failure_detected" and scf.get("status") == "success":
        strategy_family = "no_scf_fix_needed"
        strategies.append(
            _strategy_entry(
                name="no_scf_fix_needed",
                priority=1,
                rationale="The SCF portion is converged and no SCF-specific failure was detected.",
                tool="review_nwchem_case",
                docs_topics=["scf_open_shell"],
                when_to_use="Use this when the run completed and the remaining question is about state quality or chemistry, not SCF rescue.",
            )
        )
    elif failure_class == "wrong_state_convergence":
        strategy_family = "state_recovery_over_scf_tuning"
        notes.append("do_not_just_add_iterations_for_wrong_state_convergence")
        strategies.extend(
            [
                _strategy_entry(
                    name="vectors_swap_restart",
                    priority=1,
                    rationale="The SCF converged, but it converged to the wrong state; occupation steering is more appropriate than generic SCF damping.",
                    tool="prepare_nwchem_next_step",
                    docs_topics=["scf_open_shell"],
                    when_to_use="Use when SOMO count is numerically right but SOMO character is wrong.",
                ),
                _strategy_entry(
                    name="fragment_guess_seed",
                    priority=2,
                    rationale="A different initial guess can move the calculation into a different electronic basin when swap restarts do not redirect the state.",
                    tool="suggest_nwchem_state_recovery_strategy",
                    docs_topics=["fragment_guess", "scf_open_shell"],
                    when_to_use="Use when repeated swap restarts keep returning to the same suspicious state.",
                ),
                _strategy_entry(
                    name="mcscf_seed_or_validation",
                    priority=3,
                    rationale="If DFT keeps returning to the same basin, a multiconfigurational reference can test whether a metal-centered state exists nearby.",
                    tool="suggest_nwchem_state_recovery_strategy",
                    docs_topics=["mcscf"],
                    when_to_use="Use for transition-metal wrong-state cases where Fe/Co/etc. d-manifold character is important.",
                ),
            ]
        )
    elif failure_class == "scf_nonconvergence":
        strategy_family = "scf_recovery"
        if trend == "oscillatory":
            strategies.extend(
                [
                    _strategy_entry(
                        name="damp_and_smear_restart",
                        priority=1,
                        rationale=f"SCF is oscillatory after {iteration_count} iterations, so damping/smearing is more appropriate than blindly increasing maxiter.",
                        tool="draft_nwchem_scf_stabilization_input",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use when energies or densities bounce rather than trend steadily.",
                    ),
                    _strategy_entry(
                        name="different_guess_source",
                        priority=2,
                        rationale="Oscillatory open-shell runs often need a different orbital guess, not just stronger DIIS stabilization.",
                        tool="suggest_nwchem_state_recovery_strategy",
                        docs_topics=["fragment_guess", "mcscf"],
                        when_to_use="Use when oscillation persists after one stabilization-style retry.",
                    ),
                ]
            )
        elif trend == "stalled":
            strategies.extend(
                [
                    _strategy_entry(
                        name="stabilization_restart",
                        priority=1,
                        rationale=f"SCF appears stalled after {iteration_count} iterations, so a restart with damping/ncydp/smearing is the first conservative fix.",
                        tool="draft_nwchem_scf_stabilization_input",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use when errors flatten without reaching threshold.",
                    ),
                    _strategy_entry(
                        name="change_guess_or_state_model",
                        priority=2,
                        rationale="A stalled open-shell run may indicate the current guess or state model is poor, especially for transition-metal chemistry.",
                        tool="suggest_nwchem_state_recovery_strategy",
                        docs_topics=["fragment_guess", "mcscf"],
                        when_to_use="Use when one stabilization retry does not materially change the trend.",
                    ),
                ]
            )
        elif trend in {"slow_improving", "nearly_converged"} and hit_max:
            strategies.extend(
                [
                    _strategy_entry(
                        name="gentle_iteration_extension",
                        priority=1,
                        rationale=f"SCF was still improving near the iteration limit ({iteration_count} iterations), so a modest extension is justified.",
                        tool="draft_nwchem_scf_stabilization_input",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use only when the SCF pattern is monotonic or nearly converged, not oscillatory.",
                    ),
                    _strategy_entry(
                        name="light_restart_from_latest_vectors",
                        priority=2,
                        rationale="Restarting from the latest vectors can finish a nearly converged SCF more cleanly than rerunning from scratch.",
                        tool="draft_nwchem_scf_stabilization_input",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use when the max-iteration stop happened late and the density/error trend is already small.",
                    ),
                ]
            )
        else:
            strategies.extend(
                [
                    _strategy_entry(
                        name="stabilization_restart",
                        priority=1,
                        rationale="A conservative SCF stabilization restart is the safest first step when the failure pattern is not yet well classified.",
                        tool="draft_nwchem_scf_stabilization_input",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use as the generic first retry for SCF nonconvergence.",
                    ),
                    _strategy_entry(
                        name="review_open_shell_syntax_and_guess",
                        priority=2,
                        rationale="Early or low-information SCF failures can come from state specification or an unsuitable guess rather than ordinary DIIS instability.",
                        tool="review_nwchem_case",
                        docs_topics=["scf_open_shell", "fragment_guess"],
                        when_to_use="Use when there are too few iterations to trust a trend classification.",
                    ),
                ]
            )
        if state_check.get("assessment") in {"metal_state_mismatch_suspected", "somo_count_mismatch"}:
            notes.append("state_signals_also_suggest_guess_or_state_problem")
    else:
        strategies.append(
            _strategy_entry(
                name="manual_review",
                priority=1,
                rationale="No SCF-specific automatic recovery path matches this case yet.",
                tool="review_nwchem_case",
                docs_topics=["scf_open_shell"],
                when_to_use="Use when the task failed outside the SCF loop or the failure is primarily not electronic.",
            )
        )

    return {
        "output_file": output_path,
        "input_file": input_path,
        "failure_class": failure_class,
        "task_outcome": diagnosis.get("task_outcome"),
        "scf_status": scf.get("status"),
        "scf_pattern": trend,
        "iteration_count": iteration_count,
        "hit_max_iterations": hit_max,
        "state_assessment": state_check.get("assessment"),
        "strategy_family": strategy_family,
        "primary_strategy": strategies[0]["name"] if strategies else None,
        "strategies": strategies,
        "notes": notes,
    }


def suggest_nwchem_state_recovery_strategy(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
) -> dict[str, Any]:
    diagnosis = diagnose_nwchem_output(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
    )
    state = check_spin_charge_state(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
    )
    state_check = diagnosis.get("state_check") or {}
    spin_summary = state_check.get("spin_density_summary") or {}
    dominant_site = spin_summary.get("dominant_site")
    metal_like = state_check.get("metal_like_somo_count") or 0
    ligand_like = state_check.get("ligand_like_somo_count") or 0
    observed_somos = state_check.get("somo_count")
    expected_somos = state.get("expected_somo_count")

    regime = "manual_review"
    notes: list[str] = []
    strategies: list[dict[str, Any]] = []

    if state["assessment"] == "plausible":
        regime = "no_state_recovery_needed"
        strategies.append(
            _strategy_entry(
                name="accept_or_verify_state",
                priority=1,
                rationale="The current spin/frontier signals are internally consistent with the requested state.",
                tool="review_nwchem_case",
                docs_topics=["scf_open_shell"],
                when_to_use="Use when the main remaining question is chemical interpretation, not state rescue.",
            )
        )
    else:
        covalent_candidate = bool(
            dominant_site is not None
            and expected_metal_elements
            and dominant_site.get("element") in set(expected_metal_elements)
            and metal_like == 0
            and ligand_like > 0
        )
        if expected_somos is not None and observed_somos is not None and observed_somos != expected_somos:
            regime = "occupancy_or_multiplicity_mismatch"
            strategies.extend(
                [
                    _strategy_entry(
                        name="review_multiplicity_or_charge_model",
                        priority=1,
                        rationale="The observed SOMO count does not match the requested state, so multiplicity/charge may be wrong before deeper recovery attempts.",
                        tool="review_nwchem_input_request",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use when the electron count implied by the target state is inconsistent with the solution found.",
                    ),
                    _strategy_entry(
                        name="vectors_swap_restart",
                        priority=2,
                        rationale="If the target multiplicity is still chemically correct, a swap restart may recover the intended occupation pattern.",
                        tool="prepare_nwchem_next_step",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use when the mismatch is small and the target state is otherwise well motivated.",
                    ),
                ]
            )
        elif covalent_candidate:
            regime = "covalent_ligand_hole_candidate"
            notes.append("do_not_treat_ligand_dominated_somos_as_automatic_garbage")
            strategies.extend(
                [
                    _strategy_entry(
                        name="fragment_guess_validation",
                        priority=1,
                        rationale="A true fragment guess can test whether the high-spin state is merely a bad guess artifact or a robust covalent basin.",
                        tool="suggest_nwchem_scf_fix_strategy",
                        docs_topics=["fragment_guess", "scf_open_shell"],
                        when_to_use="Use when SOMOs are ligand-dominated but most total spin remains on the metal.",
                    ),
                    _strategy_entry(
                        name="mcscf_validation",
                        priority=2,
                        rationale="MCSCF is a strong next step when DFT high-spin solutions look covalent and the metal d-manifold needs explicit validation.",
                        tool="suggest_nwchem_scf_fix_strategy",
                        docs_topics=["mcscf"],
                        when_to_use="Use when you need to determine whether a metal-centered high-spin state exists near the DFT solution.",
                    ),
                    _strategy_entry(
                        name="method_or_multiplicity_scan",
                        priority=3,
                        rationale="If the state is robustly covalent across guesses, changing method or multiplicity is more informative than repeating swap restarts.",
                        tool="create_nwchem_dft_input_from_request",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use when multiple restarts collapse into the same covalent high-spin state.",
                    ),
                ]
            )
        else:
            regime = "metal_state_mismatch"
            strategies.extend(
                [
                    _strategy_entry(
                        name="vectors_swap_restart",
                        priority=1,
                        rationale="The current state looks electronically wrong, but a nearby occupation pattern may be reachable by swapping buried metal-centered orbitals into the SOMO window.",
                        tool="prepare_nwchem_next_step",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use when metal-centered occupied orbitals exist below ligand-like SOMOs.",
                    ),
                    _strategy_entry(
                        name="fragment_guess_seed",
                        priority=2,
                        rationale="If swap restarts do not redirect the state, a fragment guess gives a stronger initial bias toward the desired basin.",
                        tool="suggest_nwchem_scf_fix_strategy",
                        docs_topics=["fragment_guess"],
                        when_to_use="Use when the state repeatedly reconverges to the same suspicious pattern.",
                    ),
                    _strategy_entry(
                        name="mcscf_seed_or_reference",
                        priority=3,
                        rationale="When DFT is not preserving the desired open-shell character, MCSCF can supply a better state model or at least a diagnostic reference.",
                        tool="suggest_nwchem_scf_fix_strategy",
                        docs_topics=["mcscf"],
                        when_to_use="Use for transition-metal cases where d-orbital character matters more than a single-determinant description.",
                    ),
                    _strategy_entry(
                        name="cube_and_population_validation",
                        priority=4,
                        rationale="Visualizing SOMOs and checking Mulliken/Lowdin spin can confirm whether the suspicious state is actually covalent rather than merely wrong.",
                        tool="draft_nwchem_frontier_cube_input",
                        docs_topics=["fragment_guess"],
                        when_to_use="Use when frontier character and total spin seem contradictory.",
                    ),
                ]
            )

    return {
        "output_file": output_path,
        "input_file": input_path,
        "failure_class": diagnosis.get("failure_class"),
        "task_outcome": diagnosis.get("task_outcome"),
        "state_assessment": state["assessment"],
        "state_check_assessment": state.get("state_check_assessment"),
        "observed_somo_count": observed_somos,
        "expected_somo_count": expected_somos,
        "metal_like_somo_count": metal_like,
        "ligand_like_somo_count": ligand_like,
        "dominant_spin_site": dominant_site,
        "regime": regime,
        "primary_strategy": strategies[0]["name"] if strategies else None,
        "strategies": strategies,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# TCE helpers for summarize_nwchem_case
# ---------------------------------------------------------------------------

def _try_parse_tce(output_path: str, _contents: str | None = None) -> "dict[str, Any] | None":
    """Return parse_tce_output result if the file contains a TCE section, else None."""
    try:
        from .common import read_text
        from .nwchem_tce import parse_tce_output as _parse_tce_output
        contents = _contents if _contents is not None else read_text(output_path)
        result = _parse_tce_output(output_path, contents)
        if result.get("tce_sections"):
            return result
    except Exception:
        pass
    return None


def _tce_summary_bullets(tce: "dict[str, Any]", amp: "dict[str, Any] | None" = None) -> "list[str]":
    """Build summary bullet strings from a parse_tce_output payload.

    Parameters
    ----------
    tce:
        Result of ``parse_tce_output``.
    amp:
        Optional result of ``parse_tce_amplitudes``.  When provided, T1/D1/T2
        diagnostics and the MR verdict are included in the bullets.
    """
    bullets: list[str] = []
    method = tce.get("method") or "TCE"
    converged = tce.get("converged", False)
    total_e = tce.get("total_energy_hartree")
    corr_e = tce.get("correlation_energy_hartree")

    # Pull frozen core count and n_orbitals from the first section
    section = tce["tce_sections"][0] if tce.get("tce_sections") else {}
    frozen = section.get("frozen_cores")
    n_orb = section.get("n_orbitals")
    n_elec = section.get("n_electrons")

    status_str = "converged" if converged else "NOT converged"
    bullets.append(f"TCE method: {method} — {status_str}")

    if total_e is not None:
        bullets.append(f"TCE total energy: {total_e:.12f} Ha")
    if corr_e is not None:
        bullets.append(f"Correlation energy: {corr_e:.12f} Ha")
    if frozen is not None:
        core_note = f"{frozen} frozen core orbital{'s' if frozen != 1 else ''}"
        if n_orb is not None and n_elec is not None:
            correlated_elec = n_elec - 2 * frozen
            bullets.append(f"Frozen core: {core_note}; {correlated_elec} correlated electrons, {n_orb} orbitals")
        else:
            bullets.append(f"Frozen core: {core_note}")

    # Amplitude diagnostics (only when amplitude files were saved and parsed)
    if amp and amp.get("available"):
        t1 = amp.get("t1_diagnostic")
        d1 = amp.get("d1_diagnostic")
        t2_max = amp.get("t2_max_amplitude")
        t2_norm = amp.get("t2_frobenius_norm")
        t2_dom = amp.get("t2_dominance_fraction")
        sw = amp.get("t1_t2_singles_weight")
        tf = amp.get("triples_fraction")
        mr = amp.get("mr_assessment", "unknown")
        flags = amp.get("mr_flags", [])
        n_above_10 = amp.get("t2_count_above_010", 0)

        t1_str = f"T1={t1:.4f}" if t1 is not None else "T1=n/a"
        d1_str = f"D1={d1:.4f}" if d1 is not None else "D1=nosym_required"
        bullets.append(f"MR diagnostics: {t1_str}, {d1_str}")

        if t2_norm is not None:
            top = amp.get("t2_top_amplitudes", [])
            top_str = f"max={t2_max:.4f}" if t2_max is not None else ""
            dom_str = f", top-10 captures {t2_dom*100:.0f}% of ||T2||²" if t2_dom is not None else ""
            large_str = f", {n_above_10} amplitudes >0.10" if n_above_10 else ""
            bullets.append(f"T2 amplitudes: {top_str}{large_str}{dom_str}")
            if top:
                bullets.append(f"T2 top-10 magnitudes: {[round(v,4) for v in top[:10]]}")

        if sw is not None:
            bullets.append(
                f"Singles/doubles balance: {sw*100:.1f}% singles weight "
                f"({'dominant doubles — check for bond-breaking/diradical character' if sw < 0.05 else 'normal'})"
            )
        if tf is not None:
            bullets.append(
                f"Triples fraction: {tf*100:.1f}% of CCSD correlation "
                f"({'large — CC hierarchy may not converge' if tf > 0.15 else 'acceptable' if tf > 0.05 else 'small'})"
            )

        verdict_map = {
            "single_reference_ok": "single-reference OK — CCSD(T) results reliable",
            "moderate_mr_character": "moderate MR character — verify with MCSCF/CASSCF",
            "strong_mr_character": "strong MR character — CCSD likely unreliable",
            "unreliable_ccsd": "CCSD unreliable — use MCSCF/CASSCF instead",
        }
        bullets.append(f"MR verdict: {verdict_map.get(mr, mr)}")
        if flags:
            bullets.append(f"MR flags triggered: {', '.join(flags)}")
    elif amp and not amp.get("available"):
        bullets.append(
            "MR diagnostics: amplitude files not found — re-run with 'set tce:save_t T T' to enable"
        )

    # Next-step suggestion
    if not converged:
        bullets.append("TCE next step: increase maxiter or restart from existing amplitudes")
    elif method == "CCSD":
        bullets.append("TCE next step: consider CCSD(T) for perturbative triples correction")
    elif method == "MP2":
        bullets.append("TCE next step: consider CCSD or CCSD(T) for higher accuracy")
    else:
        bullets.append("TCE next step: calculation complete — verify freeze count and state before accepting")

    return bullets


def _build_state_check(
    diagnosis: "dict[str, Any]",
    output_path: str,
    input_path: "str | None",
    expected_somo_count: "int | None",
) -> "dict[str, Any]":
    """Build the spin/state check result from an already-computed diagnosis dict.

    Replicates check_spin_charge_state logic without re-calling diagnose_nwchem_output.
    """
    input_summary = diagnosis["input_summary"]
    state_check = diagnosis["state_check"]
    multiplicity = input_summary["multiplicity"] if input_summary else None
    charge = input_summary["charge"] if input_summary else None
    inferred_expected_somos = expected_somo_count
    if inferred_expected_somos is None and multiplicity is not None and multiplicity > 1:
        inferred_expected_somos = multiplicity - 1

    reasons: list[str] = []
    assessment = "unavailable"
    confidence = diagnosis["confidence"]

    if not state_check["available"]:
        if state_check["assessment"] != "unavailable":
            assessment = "suspicious"
            reasons.append(state_check["assessment"])
        else:
            reasons.append("frontier_state_analysis_unavailable")
    else:
        if multiplicity == 1 and state_check["somo_count"] > 0:
            assessment = "suspicious"
            reasons.append("singlet_input_but_open_shell_somos_found")
        elif inferred_expected_somos is not None and state_check["somo_count"] != inferred_expected_somos:
            assessment = "suspicious"
            reasons.append("somo_count_does_not_match_multiplicity_or_expected_state")
        elif state_check["assessment"] in {"metal_state_mismatch_suspected", "somo_count_mismatch"}:
            assessment = "suspicious"
            reasons.append(state_check["assessment"])
        else:
            assessment = "plausible"
            reasons.append("frontier_and_spin_signals_are_consistent_with_requested_state")

    dominant_site = (state_check.get("spin_density_summary") or {}).get("dominant_site")
    if dominant_site is not None:
        reasons.append(
            f"dominant_spin_density_on_{dominant_site['element']}{dominant_site['atom_index']}"
        )

    return {
        "output_file": output_path,
        "input_file": input_path,
        "assessment": assessment,
        "confidence": confidence,
        "charge": charge,
        "multiplicity": multiplicity,
        "expected_somo_count": inferred_expected_somos,
        "observed_somo_count": state_check.get("somo_count"),
        "metal_like_somo_count": state_check.get("metal_like_somo_count"),
        "ligand_like_somo_count": state_check.get("ligand_like_somo_count"),
        "state_check_assessment": state_check.get("assessment"),
        "dominant_spin_site": dominant_site,
        "reasons": reasons,
        "recommended_next_action": diagnosis["recommended_next_action"],
        "input_summary": input_summary,
    }


def summarize_nwchem_case(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    library_path: str | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
    err_file: str | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    # Lazy import to break circular dependency with api_input
    from .api_input import prepare_nwchem_next_step, lint_nwchem_input, find_restart_assets
    from .common import read_text

    # Read the output file once — reused by all downstream parsers to avoid redundant I/O
    output_contents = read_text(output_path)

    # Detect TCE early — drives what we skip and what we add below
    tce = _try_parse_tce(output_path, _contents=output_contents)
    is_tce = tce is not None

    # Try amplitude diagnostics (only present if save_t was set)
    tce_amp: dict[str, Any] | None = None
    if is_tce:
        try:
            from .nwchem_tce import parse_tce_amplitudes as _parse_amp
            tce_amp = _parse_amp(output_path)
        except Exception:
            tce_amp = None

    # detail_level="full" embeds the full diagnosis — lets us build the state check
    # inline without a second diagnose_nwchem_output call on the same file.
    summary = summarize_nwchem_output(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
        detail_level="full",
        err_file=err_file,
        _contents=output_contents,
    )
    next_step = prepare_nwchem_next_step(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
        output_dir=output_dir,
        base_name=base_name,
        write_files=False,
        _precomputed_summary=summary,
    )
    # Strip full input_text from prepared_artifacts — reduces response by 2-10 KB.
    # Callers who need the draft input should call the drafter tools directly with write_file=True.
    for artifact in next_step.get("prepared_artifacts", {}).values():
        artifact.pop("input_text", None)
        artifact.pop("plus_input_text", None)
        artifact.pop("minus_input_text", None)
    lint = lint_nwchem_input(input_path, library_path=library_path) if input_path else None
    assets = find_restart_assets(input_path or output_path)
    # Spin/state check is only meaningful for SCF/DFT runs where frontier MOs are printed.
    # Skip it for TCE — the correlated wavefunction doesn't expose frontier MOs the same way.
    # Build from the already-computed diagnosis to avoid a second full file parse.
    state: dict[str, Any] | None
    if is_tce:
        state = None
    else:
        diagnosis = summary["diagnosis"]
        state = _build_state_check(
            diagnosis=diagnosis,
            output_path=output_path,
            input_path=input_path,
            expected_somo_count=expected_somo_count,
        )

    bullets = list(summary["summary_bullets"])

    # TCE-specific bullets inserted right after the base summary
    if is_tce:
        bullets.extend(_tce_summary_bullets(tce, amp=tce_amp))

    if lint is not None:
        bullets.append(
            f"Input lint: {lint['status']} ({lint['counts']['error']} errors, {lint['counts']['warning']} warnings)"
        )
    if state is not None:
        bullets.append(f"Spin/state plausibility: {state['assessment']}")
    elif is_tce:
        bullets.append("Spin/state check: skipped for TCE (state is from SCF reference; verify freeze count separately)")
    preferred = assets["preferred"]
    if preferred.get("vectors_file"):
        bullets.append(f"Preferred restart vectors: {Path(preferred['vectors_file']).name}")
    if preferred.get("database_file"):
        bullets.append(f"Preferred restart database: {Path(preferred['database_file']).name}")
    bullets.append(f"Prepared workflow: {next_step['selected_workflow']}")

    summary_text = "\n".join(f"- {item}" for item in bullets)
    payload = {
        "output_file": output_path,
        "input_file": input_path,
        "summary_bullets": bullets,
        "summary_text": summary_text,
        "diagnosis_summary": summary,
        "tce": tce,
        "tce_amplitudes": tce_amp,
        "lint": lint,
        "restart_assets": assets,
        "spin_charge_state": state,
        "next_step": next_step,
    }
    if compact:
        return _build_compact_case_summary(payload)
    payload["compact_summary"] = _build_compact_case_summary(payload)
    return payload


def review_nwchem_case(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    library_path: str | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
) -> dict[str, Any]:
    return summarize_nwchem_case(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
        library_path=library_path,
        output_dir=output_dir,
        base_name=base_name,
        compact=True,
    )


def review_nwchem_mcscf_case(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
) -> dict[str, Any]:
    parsed = parse_mcscf_output(output_path)
    input_summary = inspect_nwchem_input(input_path) if input_path else None
    metal_elements = expected_metal_elements or (input_summary["transition_metals"] if input_summary else [])
    occupation_review = _review_mcscf_occupations(parsed)
    active_space_density = _review_mcscf_active_space_density(parsed, metal_elements)
    convergence_review = _review_mcscf_convergence(parsed)

    status = parsed["status"]
    failure_mode = parsed["failure_mode"]
    if status == "failed" and failure_mode == "input_parse_error":
        recommended_action = "fix_mcscf_block_syntax_before_retry"
        rationale = "The current MCSCF input failed before the wavefunction optimization started."
    elif status != "converged":
        recommended_action = "adjust_seed_levelshift_or_active_space_before_retry"
        rationale = "The MCSCF run did not reach a stable final state."
    elif occupation_review["assessment"] == "healthy_active_space":
        recommended_action = "use_mcscf_as_reference_or_seed_for_follow_up"
        rationale = "The active-space occupations show meaningful partial occupancy and the MCSCF run converged."
    elif occupation_review["assessment"] == "borderline_active_space":
        recommended_action = "inspect_active_space_edges_before_large_follow_up"
        rationale = "The MCSCF run converged, but at least one active orbital is effectively pinned and worth reviewing."
    else:
        recommended_action = "refine_active_space_then_rerun_mcscf"
        rationale = "The current active window looks too pinned to be an optimal long-term active space."

    bullets = [
        f"MCSCF status: {status}",
        f"Active space: CAS({parsed['settings']['active_electrons']},{parsed['settings']['active_orbitals']}) with multiplicity {parsed['settings']['multiplicity']}",
    ]
    if parsed["final_energy_hartree"] is not None:
        bullets.append(f"Final MCSCF energy: {parsed['final_energy_hartree']:.12f} Ha")
    bullets.append(convergence_review["summary"])
    if occupation_review["summary"]:
        bullets.append(occupation_review["summary"])
    if active_space_density["summary"]:
        bullets.append(active_space_density["summary"])
    bullets.append(f"Next action: {recommended_action}")

    return {
        "output_file": output_path,
        "input_file": input_path,
        "input_summary": input_summary,
        "status": status,
        "failure_mode": failure_mode,
        "summary_bullets": bullets,
        "summary_text": "\n".join(f"- {item}" for item in bullets),
        "settings": parsed["settings"],
        "convergence_review": convergence_review,
        "occupation_review": occupation_review,
        "active_space_density_review": active_space_density,
        "recommended_next_action": recommended_action,
        "rationale": rationale,
        "raw_mcscf": parsed,
    }


def _compact_tce(tce: "dict[str, Any] | None") -> "dict[str, Any] | None":
    """Return a trimmed TCE summary for the compact case payload."""
    if tce is None:
        return None
    section = tce["tce_sections"][0] if tce.get("tce_sections") else {}
    return {
        "method": tce.get("method"),
        "converged": tce.get("converged", False),
        "total_energy_hartree": tce.get("total_energy_hartree"),
        "correlation_energy_hartree": tce.get("correlation_energy_hartree"),
        "frozen_cores": section.get("frozen_cores"),
        "n_electrons": section.get("n_electrons"),
        "n_orbitals": section.get("n_orbitals"),
    }


def _build_compact_case_summary(payload: dict[str, Any]) -> dict[str, Any]:
    lint = payload.get("lint")
    assets = payload["restart_assets"]
    state = payload["spin_charge_state"]
    next_step = payload["next_step"]
    diagnosis = payload["diagnosis_summary"]["diagnosis"]
    preferred = assets["preferred"]

    return {
        "output_file": payload["output_file"],
        "input_file": payload.get("input_file"),
        "summary_text": payload["summary_text"],
        "summary_bullets": payload["summary_bullets"],
        "diagnosis": {
            "stage": diagnosis["stage"],
            "task_outcome": diagnosis["task_outcome"],
            "failure_class": diagnosis["failure_class"],
            "likely_cause": diagnosis["likely_cause"],
            "recommended_next_action": diagnosis["recommended_next_action"],
            "confidence": diagnosis["confidence"],
        },
        "lint": None
        if lint is None
        else {
            "status": lint["status"],
            "counts": lint["counts"],
            "top_issues": lint["issues"][:5],
        },
        "spin_charge_state": None
        if state is None
        else {
            "assessment": state["assessment"],
            "confidence": state["confidence"],
            "charge": state["charge"],
            "multiplicity": state["multiplicity"],
            "expected_somo_count": state["expected_somo_count"],
            "observed_somo_count": state["observed_somo_count"],
            "state_check_assessment": state["state_check_assessment"],
            "reasons": state["reasons"][:5],
            "recommended_next_action": state["recommended_next_action"],
        },
        "tce": _compact_tce(payload.get("tce")),
        "restart_assets": {
            "job_dir": assets["job_dir"],
            "focus_stem": assets["focus_stem"],
            "preferred": {
                key: value for key, value in preferred.items() if value is not None
            },
            "restart_candidates": assets["restart_candidates"][:6],
        },
        "next_step": {
            "selected_workflow": next_step["selected_workflow"],
            "can_auto_prepare": next_step["can_auto_prepare"],
            "artifact_order": next_step["artifact_order"],
            "prepared_artifact_summaries": next_step["prepared_artifact_summaries"],
            "notes": next_step["notes"],
        },
    }


def _review_mcscf_convergence(parsed: dict[str, Any]) -> dict[str, Any]:
    status = parsed["status"]
    iterations = parsed.get("iteration_count") or 0
    warnings = parsed.get("precondition_warning_count") or 0
    initial_level = parsed.get("settings", {}).get("initial_levelshift")
    final_level = parsed.get("final_levelshift")
    negative_curvatures = parsed.get("negative_curvature_count") or 0

    if status == "failed":
        assessment = "input_or_convergence_failure"
    elif status != "converged":
        assessment = "incomplete_mcscf_convergence"
    elif warnings >= 20 or negative_curvatures >= 3:
        assessment = "converged_with_stiff_orbital_optimization"
    elif warnings > 0:
        assessment = "converged_with_minor_preconditioning_warnings"
    else:
        assessment = "clean_mcscf_convergence"

    parts = [f"MCSCF macroiterations: {iterations}"]
    if warnings:
        parts.append(f"{warnings} precondition warnings")
    if initial_level is not None and final_level is not None:
        parts.append(f"level shift {initial_level:.2f} -> {final_level:.2f}")
    if negative_curvatures:
        parts.append(f"{negative_curvatures} negative-curvature events")

    return {
        "assessment": assessment,
        "iteration_count": iterations,
        "precondition_warning_count": warnings,
        "negative_curvature_count": negative_curvatures,
        "initial_levelshift": initial_level,
        "final_levelshift": final_level,
        "summary": "; ".join(parts),
    }


def _review_mcscf_occupations(parsed: dict[str, Any]) -> dict[str, Any]:
    natural = parsed.get("natural_occupations") or []
    settings = parsed.get("settings") or {}
    inactive_shells = settings.get("inactive_shells")
    active_orbitals = settings.get("active_orbitals")

    active_window: list[dict[str, Any]] = []
    if natural and inactive_shells is not None and active_orbitals:
        start = inactive_shells + 1
        end = inactive_shells + active_orbitals
        active_window = [item for item in natural if start <= item["orbital_index"] <= end]
        if len(active_window) < active_orbitals:
            active_window = natural[-active_orbitals:]
    elif natural and active_orbitals:
        active_window = natural[-active_orbitals:]

    if not active_window:
        return {
            "assessment": "occupations_unavailable",
            "active_window": [],
            "strongly_occupied_count": 0,
            "partially_occupied_count": 0,
            "near_empty_count": 0,
            "edge_candidates": {"occupied_side": [], "virtual_side": []},
            "summary": "Natural occupations were not available for active-space review.",
        }

    strongly_occupied = [item for item in active_window if (item["occupation"] or 0.0) >= 1.98]
    near_empty = [item for item in active_window if (item["occupation"] or 0.0) <= 0.02]
    partially_occupied = [item for item in active_window if item not in strongly_occupied and item not in near_empty]

    by_index = {item["orbital_index"]: item["occupation"] for item in natural}
    occupied_edge: list[dict[str, Any]] = []
    virtual_edge: list[dict[str, Any]] = []
    if inactive_shells is not None and active_orbitals:
        start = inactive_shells + 1
        end = inactive_shells + active_orbitals
        for orbital_index in range(max(1, start - 3), start):
            occupation = by_index.get(orbital_index)
            if occupation is not None and occupation < 1.98:
                occupied_edge.append({"orbital_index": orbital_index, "occupation": occupation})
        for orbital_index in range(end + 1, end + 4):
            occupation = by_index.get(orbital_index)
            if occupation is not None and occupation > 0.02:
                virtual_edge.append({"orbital_index": orbital_index, "occupation": occupation})

    if len(partially_occupied) >= max(3, (active_orbitals or len(active_window)) // 2) and not occupied_edge and not virtual_edge:
        assessment = "healthy_active_space"
    elif len(partially_occupied) >= 2:
        assessment = "borderline_active_space"
    else:
        assessment = "overly_pinned_active_space"

    summary = (
        f"Active occupations: {len(strongly_occupied)} near-2, "
        f"{len(partially_occupied)} fractional, {len(near_empty)} near-0"
    )
    if occupied_edge or virtual_edge:
        edge_notes = []
        if occupied_edge:
            edge_notes.append("occupied edge orbitals just below the active space are not fully pinned")
        if virtual_edge:
            edge_notes.append("virtual edge orbitals just above the active space are not fully empty")
        summary += "; " + "; ".join(edge_notes)

    return {
        "assessment": assessment,
        "active_window": active_window,
        "strongly_occupied_count": len(strongly_occupied),
        "partially_occupied_count": len(partially_occupied),
        "near_empty_count": len(near_empty),
        "edge_candidates": {
            "occupied_side": occupied_edge,
            "virtual_side": virtual_edge,
        },
        "swap_out_candidates": [item["orbital_index"] for item in strongly_occupied + near_empty],
        "summary": summary,
    }


def _review_mcscf_active_space_density(parsed: dict[str, Any], metal_elements: list[str]) -> dict[str, Any]:
    active_density = parsed.get("active_space_mulliken") or {}
    atoms = active_density.get("atoms") or []
    if not atoms:
        return {
            "assessment": "active_space_density_unavailable",
            "metal_fraction": None,
            "dominant_atoms": [],
            "summary": None,
        }

    total_population = sum(item.get("charge") or 0.0 for item in atoms)
    dominant_atoms = sorted(atoms, key=lambda item: item.get("charge") or 0.0, reverse=True)
    metal_set = {element.lower() for element in metal_elements}
    metal_population = sum((item.get("charge") or 0.0) for item in atoms if item["element"].lower() in metal_set)
    metal_fraction = (metal_population / total_population) if total_population else None

    if metal_fraction is None:
        assessment = "active_space_density_unavailable"
    elif metal_fraction >= 0.45:
        assessment = "metal_participation_significant"
    elif metal_fraction >= 0.2:
        assessment = "mixed_metal_ligand_active_space"
    else:
        assessment = "ligand_dominated_active_space"

    top_labels = ", ".join(
        f"{item['element']}{item['atom_index']} {item['charge']:.2f}" for item in dominant_atoms[:3]
    )
    summary = (
        f"Active-space Mulliken density: top contributors {top_labels}"
        + (f"; metal fraction {metal_fraction:.2f}" if metal_fraction is not None else "")
    )
    return {
        "assessment": assessment,
        "metal_fraction": metal_fraction,
        "dominant_atoms": dominant_atoms[:6],
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Geometry plausibility checker
# ---------------------------------------------------------------------------

# Typical max coordination numbers per element (above this → red flag)
_MAX_COORD: dict[str, int] = {
    "H": 2, "He": 0,
    "Li": 4, "Be": 4, "B": 4, "C": 4, "N": 4, "O": 3, "F": 1, "Ne": 0,
    "Na": 6, "Mg": 6, "Al": 6, "Si": 6, "P": 6, "S": 6, "Cl": 1, "Ar": 0,
    "K": 8, "Ca": 8, "Ga": 6, "Ge": 4, "As": 6, "Se": 6, "Br": 1, "Kr": 0,
    "Rb": 8, "Sr": 8, "In": 6, "Sn": 6, "Sb": 6, "Te": 6, "I": 1, "Xe": 0,
    "Cs": 12, "Ba": 12, "Tl": 6, "Pb": 6, "Bi": 6,
}
# Transition metals: typical max CN = 9
for _tm in _TRANSITION_METALS:
    if _tm not in _MAX_COORD:
        _MAX_COORD[_tm] = 9

# Lanthanides and actinides: high-CN chemistry (up to 12–14)
_LANTHANIDES = {"La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"}
_ACTINIDES = {"Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"}
for _hm in _LANTHANIDES | _ACTINIDES:
    if _hm not in _MAX_COORD:
        _MAX_COORD[_hm] = 14

# All elements that behave like metals (used for coordination reporting)
_ALL_METALS = _TRANSITION_METALS | _LANTHANIDES | _ACTINIDES | {
    "Li","Na","K","Rb","Cs","Be","Mg","Ca","Sr","Ba",
    "Al","Ga","In","Tl","Sn","Pb","Bi",
}

# Typical CN ranges for main-group elements  {element: (min_ok, max_ok)}
_TYPICAL_COORD: dict[str, tuple[int, int]] = {
    "H": (1, 1), "He": (0, 0),
    "B": (2, 4), "C": (1, 4), "N": (1, 4), "O": (1, 2), "F": (1, 1),
    "Si": (2, 6), "P": (1, 6), "S": (1, 6), "Cl": (1, 1),
    "Ge": (2, 4), "As": (2, 6), "Se": (1, 6), "Br": (1, 1),
    "Sn": (2, 6), "Sb": (2, 6), "Te": (1, 6), "I": (1, 1),
    "Pb": (2, 6), "Bi": (2, 6),
}


def _compute_bonds(
    atoms: list[str],
    positions: list[list[float]],
    clash_factor: float = 0.70,
    bond_factor: float = 1.30,
) -> dict[str, Any]:
    """Compute bonds, clashes, and long contacts from atom positions.

    Returns a dict with:
      bonds           list of {i, j, elem_i, elem_j, distance, expected_max, ratio}
      clashes         pairs where distance < clash_factor × (r_i + r_j)
      long_bonds      bonds where ratio > 1.0 but likely still connected
      coordination    {atom_index: count}
    """
    import math
    n = len(atoms)
    bonds: list[dict[str, Any]] = []
    clashes: list[dict[str, Any]] = []
    long_bonds: list[dict[str, Any]] = []
    coord: list[int] = [0] * n

    fallback_r = 1.5  # Å — used when element not in radii table

    for i in range(n):
        ri = _COVALENT_RADII.get(atoms[i], fallback_r)
        xi, yi, zi = positions[i]
        for j in range(i + 1, n):
            rj = _COVALENT_RADII.get(atoms[j], fallback_r)
            xj, yj, zj = positions[j]
            dx, dy, dz = xj - xi, yj - yi, zj - zi
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            r_sum = ri + rj
            ratio = dist / r_sum

            entry = {
                "i": i, "j": j,
                "elem_i": atoms[i], "elem_j": atoms[j],
                "label_i": f"{atoms[i]}{i + 1}", "label_j": f"{atoms[j]}{j + 1}",
                "distance_angstrom": round(dist, 4),
                "expected_max_angstrom": round(r_sum * bond_factor, 3),
                "ratio": round(ratio, 3),
            }
            if ratio < clash_factor:
                clashes.append(entry)
            elif ratio <= bond_factor:
                bonds.append(entry)
                coord[i] += 1
                coord[j] += 1
            elif ratio <= 1.50:
                long_bonds.append(entry)

    return {
        "bonds": bonds,
        "clashes": clashes,
        "long_bonds": long_bonds,
        "coordination": coord,
    }


def _compute_bond_angles(
    atoms: list[str],
    positions: list[list[float]],
    bond_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return bond angles for every A-B-C triple where A-B and B-C are bonds."""
    import math
    from collections import defaultdict
    neighbors: dict[int, list[int]] = defaultdict(list)
    for b in bond_list:
        neighbors[b["i"]].append(b["j"])
        neighbors[b["j"]].append(b["i"])

    angles: list[dict[str, Any]] = []
    for center, nbrs in neighbors.items():
        if len(nbrs) < 2:
            continue
        cx, cy, cz = positions[center]
        for k in range(len(nbrs)):
            for l in range(k + 1, len(nbrs)):
                a, b = nbrs[k], nbrs[l]
                ax, ay, az = positions[a]
                bx, by, bz = positions[b]
                va = (ax - cx, ay - cy, az - cz)
                vb = (bx - cx, by - cy, bz - cz)
                na = math.sqrt(va[0]**2 + va[1]**2 + va[2]**2)
                nb = math.sqrt(vb[0]**2 + vb[1]**2 + vb[2]**2)
                if na < 1e-8 or nb < 1e-8:
                    continue
                cos_a = max(-1.0, min(1.0, (va[0]*vb[0] + va[1]*vb[1] + va[2]*vb[2]) / (na * nb)))
                angle_deg = math.degrees(math.acos(cos_a))
                angles.append({
                    "center": center,
                    "a": a, "b": b,
                    "elem_center": atoms[center],
                    "label_center": f"{atoms[center]}{center + 1}",
                    "label_a": f"{atoms[a]}{a + 1}",
                    "label_b": f"{atoms[b]}{b + 1}",
                    "angle_deg": round(angle_deg, 2),
                })
    return angles


def _check_ring_planarity(
    atoms: list[str],
    positions: list[list[float]],
    bond_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect rings and check planarity for aromatic candidates."""
    import math
    from collections import defaultdict

    neighbors: dict[int, list[int]] = defaultdict(list)
    for b in bond_list:
        neighbors[b["i"]].append(b["j"])
        neighbors[b["j"]].append(b["i"])

    # Ring detection: DFS for cycles of length 5–7.
    # Guard: skip for very large molecules where DFS would be slow.
    if len(atoms) > 120:
        return []

    ring_atoms_sets: list[frozenset] = []
    rings: list[list[int]] = []

    def dfs(start: int, current: int, path: list[int], visited: set) -> None:
        for nb in neighbors[current]:
            if nb == start and len(path) >= 3:
                candidate = frozenset(path)
                if candidate not in ring_atoms_sets and len(path) <= 7:
                    ring_atoms_sets.append(candidate)
                    rings.append(list(path))
                continue
            if nb not in visited and len(path) < 7:
                visited.add(nb)
                path.append(nb)
                dfs(start, nb, path, visited)
                path.pop()
                visited.remove(nb)

    for start in range(len(atoms)):
        dfs(start, start, [start], {start})

    results: list[dict[str, Any]] = []
    for ring in rings:
        if len(ring) < 5:
            continue
        ring_elems = [atoms[i] for i in ring]
        # Aromatic candidate: all C/N with no H-only members
        is_aromatic_candidate = all(e in ("C", "N", "O", "S") for e in ring_elems)

        pts = [positions[i] for i in ring]
        # Fit plane via SVD
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)
        centered = [[p[0] - cx, p[1] - cy, p[2] - cz] for p in pts]

        # Fit best-fit plane via SVD (numerically stable even for nearly-collinear atoms)
        try:
            import numpy as _np
            mat = _np.array(centered)
            _, _, vh = _np.linalg.svd(mat)
            normal = vh[-1]  # row with smallest singular value = plane normal
            deviations = [abs(float(_np.dot(normal, p))) for p in centered]
        except Exception:
            # Fallback: cross product of first two edges
            v1, v2 = centered[1], centered[2]
            nx = v1[1]*v2[2] - v1[2]*v2[1]
            ny = v1[2]*v2[0] - v1[0]*v2[2]
            nz = v1[0]*v2[1] - v1[1]*v2[0]
            nn = math.sqrt(nx*nx + ny*ny + nz*nz)
            if nn < 1e-8:
                continue
            nx, ny, nz = nx/nn, ny/nn, nz/nn
            deviations = [abs(p[0]*nx + p[1]*ny + p[2]*nz) for p in centered]
        max_dev = max(deviations)
        rms_dev = math.sqrt(sum(d*d for d in deviations) / len(deviations))

        results.append({
            "ring_size": len(ring),
            "ring_labels": [f"{atoms[i]}{i+1}" for i in ring],
            "is_aromatic_candidate": is_aromatic_candidate,
            "max_planarity_deviation_angstrom": round(max_dev, 4),
            "rms_planarity_deviation_angstrom": round(rms_dev, 4),
            "planar": max_dev < 0.10,
        })

    return results


def check_nwchem_geometry_plausibility(
    output_path: str,
    input_path: str | None = None,
    frame: str = "best",
) -> dict[str, Any]:
    """Check whether an optimized NWChem geometry is chemically plausible.

    Parses the optimization trajectory to extract atom positions, then runs
    a suite of chemical sanity checks:

    - Bond length plausibility (short clashes, unusually long bonds)
    - Coordination number check (per element expected range)
    - Extreme bond angles (< 60° or > 170° for non-linear centres)
    - Ring planarity for 5–7-membered rings
    - Metal coordination geometry summary

    Works on any frame: 'best' (smart selection, default), 'last', 'first',
    'min_energy', or integer step number.

    Returns
    -------
    dict with:
      plausible          bool — True if no red flags found
      red_flags          list of critical issues
      warnings           list of non-critical concerns
      bond_summary       count of bonds, clashes, long bonds
      coordination       per-atom coordination with flag
      angle_flags        extreme bond angles
      ring_checks        planarity for rings
      selected_frame     step/energy of the frame checked
    """
    from .api_input import extract_nwchem_geometry, _select_best_optimization_frame
    from . import nwchem

    contents = read_text(output_path)

    # --- Extract geometry ---
    frame_arg: str | int = frame
    try:
        frame_arg = int(frame)
    except (TypeError, ValueError):
        pass

    geo = extract_nwchem_geometry(output_path, frame=frame_arg, input_path=input_path)
    sel = geo["selected_frame"]
    atoms: list[str] = geo["selected_frame"].get("elements") or []
    positions_raw = geo["selected_frame"].get("positions_angstrom") or []

    # extract_nwchem_geometry returns frame info but we need elements+positions
    # Fall back: parse from trajectory directly
    if not atoms or not positions_raw:
        traj = nwchem.parse_trajectory(output_path, contents, include_positions=True)
        frames = traj.get("frames", [])
        if not frames:
            return {"error": "no geometry frames found in output"}
        if frame_arg == "best":
            chosen, _ = _select_best_optimization_frame(frames, traj["optimization_status"])
        elif frame_arg == "last":
            chosen = frames[-1]
        elif frame_arg == "first":
            chosen = frames[0]
        elif frame_arg == "min_energy":
            chosen = min(frames, key=lambda f: f.get("energy_hartree") or float("inf"))
        else:
            step_map = {f["step"]: f for f in frames}
            chosen = step_map.get(int(frame_arg), frames[-1])

        atoms = chosen.get("labels") or []
        positions_raw = chosen.get("positions_angstrom") or []
        sel = {"step": chosen["step"], "energy_hartree": chosen.get("energy_hartree")}

    # Normalise element symbols (strip trailing digits from labels)
    def _label_to_elem(lbl: str) -> str:
        import re as _re
        return _re.sub(r"\d+$", "", lbl).capitalize()

    elements = [_label_to_elem(a) for a in atoms]
    positions: list[list[float]] = [list(p) for p in positions_raw]

    if not elements or not positions:
        return {"error": "could not extract atom positions from geometry frame"}

    # --- Compute bonds ---
    bond_data = _compute_bonds(elements, positions)
    bonds = bond_data["bonds"]
    clashes = bond_data["clashes"]
    long_bonds = bond_data["long_bonds"]
    coord_counts = bond_data["coordination"]

    # --- Coordination checks ---
    coord_flags: list[dict[str, Any]] = []
    coord_info: list[dict[str, Any]] = []
    for idx, (elem, cn) in enumerate(zip(elements, coord_counts)):
        label = f"{elem}{idx + 1}"
        max_cn = _MAX_COORD.get(elem, 9)
        typ_range = _TYPICAL_COORD.get(elem)
        flag: str | None = None
        if cn > max_cn:
            flag = f"overcrowded: CN={cn} > max expected {max_cn}"
        elif typ_range and cn < typ_range[0] and cn == 0 and elem not in ("He", "Ne", "Ar", "Kr", "Xe"):
            flag = f"isolated: CN=0"
        elif elem == "H" and cn > 1:
            flag = f"H bridging: CN={cn} (unusual unless explicit bridge)"
        elif elem == "C" and cn == 0:
            flag = "isolated C atom"

        coord_info.append({
            "label": label, "element": elem, "coordination_number": cn, "flag": flag
        })
        if flag:
            coord_flags.append({"label": label, "flag": flag})

    # --- Angle checks ---
    angles = _compute_bond_angles(elements, positions, bonds)
    angle_flags: list[dict[str, Any]] = []
    for ang in angles:
        a = ang["angle_deg"]
        cn = coord_counts[ang["center"]]
        center_elem = ang["elem_center"]
        is_metal_center = center_elem in _ALL_METALS
        # Flag angles that are chemically extreme
        # High-CN metal centres (CN>6) naturally have small L-M-L angles — don't flag those
        if a < 50.0 and not (is_metal_center and cn > 6):
            angle_flags.append({**ang, "issue": f"very acute angle {a:.1f}° — extreme ring strain or wrong connectivity"})
        elif a > 175.0 and cn > 2 and not is_metal_center:
            angle_flags.append({**ang, "issue": f"near-linear angle {a:.1f}° at CN={cn} centre — possible geometry error"})

    # --- Ring planarity ---
    ring_checks = _check_ring_planarity(elements, positions, bonds)
    ring_flags = [r for r in ring_checks if r["is_aromatic_candidate"] and not r["planar"]]

    # --- Metal coordination summary ---
    metal_coord: list[dict[str, Any]] = []
    for idx, elem in enumerate(elements):
        if elem in _ALL_METALS:
            cn = coord_counts[idx]
            bonded = [b for b in bonds if b["i"] == idx or b["j"] == idx]
            ligand_elems = [
                b["elem_j"] if b["i"] == idx else b["elem_i"]
                for b in bonded
            ]
            note: str | None = None
            max_expected = _MAX_COORD.get(elem, 9)
            if cn == 0:
                note = "isolated metal — no bonds detected"
            elif cn < 2:
                note = f"unusually low CN={cn} for metal"
            elif cn > max_expected:
                note = f"very high CN={cn} (max expected {max_expected}) — check for spurious bonds"
            metal_coord.append({
                "label": f"{elem}{idx + 1}",
                "element": elem,
                "coordination_number": cn,
                "ligand_elements": sorted(set(ligand_elems)),
                "note": note,
            })

    # --- Assemble red flags and warnings ---
    red_flags: list[str] = []
    warnings_out: list[str] = []

    for c in clashes:
        red_flags.append(
            f"CLASH: {c['label_i']}–{c['label_j']} distance {c['distance_angstrom']:.3f} Å "
            f"({c['ratio']:.2f}× covalent sum) — atoms too close"
        )

    for lb in long_bonds:
        warnings_out.append(
            f"LONG BOND: {lb['label_i']}–{lb['label_j']} {lb['distance_angstrom']:.3f} Å "
            f"({lb['ratio']:.2f}× covalent sum) — possibly broken or weak bond"
        )

    for cf in coord_flags:
        (red_flags if "overcrowded" in cf["flag"] or "isolated" in cf["flag"] else warnings_out).append(
            f"COORD: {cf['label']} — {cf['flag']}"
        )

    for af in angle_flags:
        warnings_out.append(f"ANGLE: {af['label_center']} — {af['issue']}")

    for rf in ring_flags:
        warnings_out.append(
            f"RING: {'-'.join(rf['ring_labels'])} — aromatic candidate not planar "
            f"(max dev {rf['max_planarity_deviation_angstrom']:.3f} Å)"
        )

    for mc in metal_coord:
        if mc["note"] and ("isolated" in mc["note"] or "high CN" in mc["note"]):
            red_flags.append(f"METAL: {mc['label']} — {mc['note']}")
        elif mc["note"]:
            warnings_out.append(f"METAL: {mc['label']} — {mc['note']}")

    plausible = len(red_flags) == 0

    return {
        "plausible": plausible,
        "red_flags": red_flags,
        "warnings": warnings_out,
        "selected_frame": sel,
        "atom_count": len(elements),
        "bond_summary": {
            "bond_count": len(bonds),
            "clash_count": len(clashes),
            "long_bond_count": len(long_bonds),
        },
        "coordination": coord_info,
        "angle_flag_count": len(angle_flags),
        "angle_flags": angle_flags,
        "ring_checks": ring_checks,
        "metal_coordination": metal_coord,
    }


# ---------------------------------------------------------------------------
# Frequency plausibility checker
# ---------------------------------------------------------------------------

# Frequency band assignments for common bond types (cm⁻¹)
_FREQ_BANDS: list[tuple[float, float, str]] = [
    (0,    50,   "near-zero / translational / conformational"),
    (50,   300,  "metal-ligand / torsional"),
    (300,  600,  "metal-ligand stretches / heavy-atom bends"),
    (600,  900,  "ring deformations / C-halogen stretches"),
    (900,  1200, "C-O / C-N / C-C / skeletal stretches"),
    (1200, 1500, "C-H bends / C-C / C-N stretches"),
    (1500, 1700, "C=C / C=N / N-H bends"),
    (1700, 1900, "C=O carbonyl stretches"),
    (1900, 2400, "C≡N / C≡C / C=C=O"),
    (2400, 2800, "S-H / P-H / Si-H stretches"),
    (2800, 3200, "C-H stretches"),
    (3200, 3700, "N-H / O-H stretches"),
    (3700, 9999, "very high — check for very light atoms or scale factor"),
]

# Element → expected high-freq modes if bonds to H are present
_EXPECTED_XH_BANDS: dict[str, tuple[float, float, str]] = {
    "O": (3200, 3700, "O-H stretch"),
    "N": (3100, 3500, "N-H stretch"),
    "C": (2800, 3200, "C-H stretch"),
    "S": (2400, 2600, "S-H stretch"),
}


def check_nwchem_freq_plausibility(
    output_path: str,
    input_path: str | None = None,
    expect_minimum: bool = True,
) -> dict[str, Any]:
    """Check whether NWChem frequency results are chemically plausible.

    Performs the following checks:

    - Imaginary mode count vs. expectation (minimum vs. transition state)
    - Large imaginary modes (< −50 cm⁻¹) — serious structural problem
    - Near-zero real modes (< 20 cm⁻¹) — flat PES or incomplete optimisation
    - Mode distribution across frequency bands
    - Expected X-H stretch presence given elements in molecule
    - ZPE per atom sanity check (expected ~2–12 kcal/mol per heavy atom)
    - Suspiciously high frequencies (possible scale-factor or unit error)

    Parameters
    ----------
    output_path:
        Path to the NWChem frequency output file.
    input_path:
        Optional: path to the input file (used to read element list).
    expect_minimum:
        True (default) if the calculation is expected to be a local minimum
        (0 imaginary modes).  Set False for transition state searches.

    Returns
    -------
    dict with:
      plausible             bool
      red_flags             list of critical issues
      warnings              list of non-critical concerns
      mode_counts           summary of mode counts by type
      band_distribution     modes per frequency band
      zpe_check             ZPE analysis
      missing_xh_stretches  expected X-H bands not observed
    """
    from . import nwchem as _nwchem
    from .nwchem_freq import parse_freq as _parse_freq

    contents = read_text(output_path)
    freq_data = _parse_freq(output_path, contents)

    modes = freq_data.get("modes", [])
    thermo = freq_data.get("thermochemistry") or {}
    n_imag = freq_data.get("imaginary_mode_count", 0)
    n_near_zero = freq_data.get("near_zero_mode_count", 0)
    near_zero_freqs = freq_data.get("near_zero_frequencies_cm1", [])
    sig_imag_freqs = freq_data.get("significant_imaginary_frequencies_cm1", [])

    all_freqs = [m["frequency_cm1"] for m in modes]
    real_freqs = [f for f in all_freqs if f >= 0]
    imag_freqs = [f for f in all_freqs if f < 0]

    # --- Element list ---
    elements: list[str] = []
    if input_path:
        try:
            inp = inspect_nwchem_input(input_path)
            elements = inp.get("elements", [])
        except Exception:
            pass

    # --- Mode counts ---
    n_modes = len(modes)

    # --- Band distribution ---
    band_dist: list[dict[str, Any]] = []
    for lo, hi, label in _FREQ_BANDS:
        in_band = [f for f in real_freqs if lo <= f < hi]
        if in_band or lo < 100:
            band_dist.append({
                "range_cm1": f"{lo}–{hi}",
                "label": label,
                "count": len(in_band),
                "examples_cm1": [round(f, 1) for f in sorted(in_band)[:5]],
            })

    # --- ZPE check ---
    zpe_correction = thermo.get("zero_point_correction") or {}
    zpe_kcal = zpe_correction.get("kcal_mol")
    n_atoms_thermo = None
    zpe_per_atom: float | None = None
    zpe_note: str | None = None
    if zpe_kcal is not None and elements:
        # Count non-H atoms as "heavy atoms"
        heavy = [e for e in elements if e != "H"]
        n_atoms_thermo = len(elements)
        n_heavy = len(heavy)
        if n_atoms_thermo > 0:
            zpe_per_atom = zpe_kcal / n_atoms_thermo
            # Rough expected range: 2-15 kcal/mol per heavy atom
            if n_heavy > 0:
                zpe_per_heavy = zpe_kcal / n_heavy
                if zpe_per_heavy < 0.5:
                    zpe_note = f"ZPE/heavy-atom={zpe_per_heavy:.1f} kcal/mol seems very low"
                elif zpe_per_heavy > 30.0:
                    zpe_note = f"ZPE/heavy-atom={zpe_per_heavy:.1f} kcal/mol seems very high"

    # --- X-H stretch checks ---
    missing_xh: list[str] = []
    if elements and real_freqs:
        elem_set = set(elements)
        has_h = "H" in elem_set
        if has_h:
            for heavy_elem, (lo, hi, name) in _EXPECTED_XH_BANDS.items():
                if heavy_elem in elem_set:
                    observed = any(lo <= f <= hi for f in real_freqs)
                    if not observed:
                        missing_xh.append(
                            f"{name} ({lo}–{hi} cm⁻¹) expected but not observed"
                        )

    # --- Very high frequency check ---
    suspicious_high = [f for f in real_freqs if f > 4000]

    # --- Assemble red flags and warnings ---
    red_flags: list[str] = []
    warnings_out: list[str] = []

    # Imaginary mode assessment
    if expect_minimum:
        if n_imag == 1 and sig_imag_freqs:
            red_flags.append(
                f"1 imaginary mode ({sig_imag_freqs[0]:.1f} cm⁻¹) — geometry is a transition state, "
                "not a minimum. Re-optimize or follow the imaginary mode."
            )
        elif n_imag > 1:
            red_flags.append(
                f"{n_imag} imaginary modes ({[round(f,1) for f in imag_freqs]}) — "
                "higher-order saddle point. Geometry needs rethinking."
            )
        elif n_imag == 1 and not sig_imag_freqs:
            warnings_out.append(
                "1 near-zero imaginary mode — likely numerical noise, but verify geometry."
            )
    else:
        if n_imag == 0:
            warnings_out.append("Expected 1 imaginary mode for TS but found 0 — may be a minimum.")
        elif n_imag > 1:
            red_flags.append(
                f"{n_imag} imaginary modes — TS should have exactly 1. Check geometry."
            )

    # Large imaginary modes
    very_large_imag = [f for f in imag_freqs if f < -200]
    if very_large_imag:
        red_flags.append(
            f"Very large imaginary mode(s) {[round(f,1) for f in very_large_imag]} cm⁻¹ — "
            "severe structural problem, not numerical noise."
        )

    # Near-zero real modes
    if n_near_zero > 6:
        # More than 6 near-zero modes is unusual (linear has 5, nonlinear has 6)
        extras = n_near_zero - 6
        warnings_out.append(
            f"{n_near_zero} near-zero modes (<20 cm⁻¹) — {extras} extra beyond the expected "
            "translational/rotational. May indicate flat PES, floppy molecule, or incomplete optimisation."
        )
    elif n_near_zero > 0 and near_zero_freqs:
        # Some non-negligible near-zero real modes
        real_nz = [f for f in near_zero_freqs if f > 0]
        if real_nz:
            warnings_out.append(
                f"Low-frequency real modes {[round(f,1) for f in real_nz]} cm⁻¹ — "
                "very soft modes; check for floppy conformations or weak intermolecular interactions."
            )

    if missing_xh:
        for m in missing_xh:
            warnings_out.append(f"MISSING MODE: {m}")

    if suspicious_high:
        warnings_out.append(
            f"Very high frequencies {[round(f,1) for f in suspicious_high]} cm⁻¹ (>4000) — "
            "check for erroneous geometry, missing mass, or wrong units."
        )

    if zpe_note:
        warnings_out.append(f"ZPE: {zpe_note}")

    # Check: if no real vibrational modes at all
    if len(real_freqs) == 0:
        red_flags.append("No real vibrational frequencies found — frequency calculation may have failed.")

    plausible = len(red_flags) == 0

    return {
        "plausible": plausible,
        "red_flags": red_flags,
        "warnings": warnings_out,
        "mode_counts": {
            "total": n_modes,
            "imaginary": n_imag,
            "near_zero": n_near_zero,
            "real_vibrational": len([f for f in real_freqs if f >= 20]),
        },
        "imaginary_frequencies_cm1": [round(f, 1) for f in imag_freqs],
        "band_distribution": band_dist,
        "zpe_check": {
            "zpe_kcal_mol": zpe_kcal,
            "n_atoms": n_atoms_thermo,
            "zpe_per_atom_kcal_mol": round(zpe_per_atom, 2) if zpe_per_atom is not None else None,
        },
        "missing_xh_stretches": missing_xh,
    }


# ---------------------------------------------------------------------------
# Spin state advisor
# ---------------------------------------------------------------------------

# d-block TMs: element → (Z, noble-gas core electrons)
_TM_Z_CORE: dict[str, tuple[int, int]] = {
    "Sc": (21, 18), "Ti": (22, 18), "V": (23, 18), "Cr": (24, 18),
    "Mn": (25, 18), "Fe": (26, 18), "Co": (27, 18), "Ni": (28, 18),
    "Cu": (29, 18), "Zn": (30, 18),
    "Y": (39, 36), "Zr": (40, 36), "Nb": (41, 36), "Mo": (42, 36),
    "Tc": (43, 36), "Ru": (44, 36), "Rh": (45, 36), "Pd": (46, 36),
    "Ag": (47, 36), "Cd": (48, 36),
    "Hf": (72, 68), "Ta": (73, 68), "W": (74, 68), "Re": (75, 68),
    "Os": (76, 68), "Ir": (77, 68), "Pt": (78, 68), "Au": (79, 68), "Hg": (80, 68),
}

# Common oxidation states ordered by frequency
_TM_COMMON_OX: dict[str, list[int]] = {
    "Sc": [3], "Ti": [4, 3, 2], "V": [3, 4, 5, 2], "Cr": [3, 2, 6],
    "Mn": [2, 3, 4, 7], "Fe": [2, 3, 4], "Co": [2, 3], "Ni": [2, 3],
    "Cu": [1, 2], "Zn": [2],
    "Y": [3], "Zr": [4, 3], "Nb": [5, 3, 4], "Mo": [4, 5, 6, 3],
    "Tc": [4, 7], "Ru": [2, 3, 4], "Rh": [3, 2], "Pd": [2, 4],
    "Ag": [1, 2], "Cd": [2],
    "Hf": [4], "Ta": [5, 3], "W": [4, 6, 3], "Re": [3, 4, 7],
    "Os": [4, 2, 3], "Ir": [3, 4], "Pt": [2, 4], "Au": [1, 3], "Hg": [2, 1],
}

# Hund high-spin vs strong-field low-spin unpaired electrons for d0..d10
_D_HS_UNPAIRED = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
_D_LS_UNPAIRED = [0, 1, 2, 1, 0, 1, 0, 1, 2, 1, 0]

# Ligand elements that imply weak vs strong crystal field
_WEAK_FIELD_ELEMENTS = {"F", "Cl", "Br", "I", "O", "S", "Se", "Te"}
_STRONG_FIELD_ELEMENTS = {"C", "N", "P"}


def _d_count_for_ox(element: str, oxidation_state: int) -> int | None:
    tm = _TM_Z_CORE.get(element)
    if tm is None:
        return None
    z, core = tm
    d = z - core - oxidation_state
    return d if 0 <= d <= 10 else None


def suggest_spin_state(
    elements: list[str],
    charge: int = 0,
    metal_oxidation_states: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Suggest likely spin multiplicities for a molecule given elements and charge.

    For transition-metal systems this computes d-electron counts and returns
    high-spin (Hund) and low-spin (strong-field octahedral) multiplicity
    candidates with plain-language explanations.

    Args:
        elements: All element symbols in the molecule (duplicates OK, e.g. ['Fe', 'Cl', 'Cl']).
        charge: Total molecular charge.
        metal_oxidation_states: Optional dict mapping metal symbol to formal oxidation state,
            e.g. {'Fe': 2}.  When omitted, common oxidation states for each metal are enumerated.

    Returns dict with 'recommended_multiplicity', 'metal_analyses', and 'summary'.
    """
    norm = [e[0].upper() + e[1:].lower() for e in elements]
    unique_elements = list(dict.fromkeys(norm))
    metal_elements = [e for e in unique_elements if e in _TRANSITION_METALS]
    ligand_elements = [e for e in unique_elements if e not in _TRANSITION_METALS]

    has_strong_field = any(e in _STRONG_FIELD_ELEMENTS for e in ligand_elements)
    has_weak_field = any(e in _WEAK_FIELD_ELEMENTS for e in ligand_elements)

    metal_analyses: list[dict[str, Any]] = []

    for metal in metal_elements:
        if metal_oxidation_states and metal in metal_oxidation_states:
            ox_list = [metal_oxidation_states[metal]]
            ox_source = "provided"
        else:
            ox_list = _TM_COMMON_OX.get(metal, [2, 3])
            ox_source = "common_states"

        ox_analyses = []
        for ox in ox_list:
            d = _d_count_for_ox(metal, ox)
            if d is None:
                continue
            hs_u = _D_HS_UNPAIRED[d]
            ls_u = _D_LS_UNPAIRED[d]
            hs_mult = hs_u + 1
            ls_mult = ls_u + 1
            spin_states = [{"spin_state": "high-spin", "multiplicity": hs_mult, "unpaired": hs_u, "d_count": d}]
            if ls_mult != hs_mult:
                spin_states.append({"spin_state": "low-spin", "multiplicity": ls_mult, "unpaired": ls_u, "d_count": d})

            if len(spin_states) == 1:
                rec_idx, rec_reason = 0, "only one possible spin state for d%d" % d
            elif has_strong_field and not has_weak_field:
                rec_idx, rec_reason = 1, "strong-field ligands (C/N/P donors) favor low-spin"
            elif has_weak_field and not has_strong_field:
                rec_idx, rec_reason = 0, "weak-field ligands (halide/chalcogenide) favor high-spin"
            else:
                rec_idx, rec_reason = 0, "defaulting to high-spin (Hund's rule); verify ligand field"

            ox_analyses.append({
                "oxidation_state": ox,
                "oxidation_state_source": ox_source,
                "d_count": d,
                "spin_states": spin_states,
                "recommended_spin_state": spin_states[rec_idx]["spin_state"],
                "recommended_multiplicity": spin_states[rec_idx]["multiplicity"],
                "recommendation_reason": rec_reason,
            })

        metal_analyses.append({"element": metal, "oxidation_state_analyses": ox_analyses})

    # Non-TM case: infer from total electron count
    if not metal_elements:
        total_e = sum(ELEMENT_TO_Z.get(e, 0) for e in norm) - charge
        rec_mult = 2 if total_e % 2 == 1 else 1
        return {
            "elements": unique_elements,
            "charge": charge,
            "has_transition_metals": False,
            "total_electrons": total_e,
            "recommended_multiplicity": rec_mult,
            "metal_analyses": [],
            "summary": (
                f"No transition metals. Total electrons: {total_e}. "
                f"Recommended multiplicity: {rec_mult} "
                f"({'doublet' if rec_mult == 2 else 'singlet'})."
            ),
        }

    # Derive overall recommendation from first metal / first (most common) ox state
    rec_mult: int | None = None
    rec_spin: str | None = None
    if metal_analyses and metal_analyses[0]["oxidation_state_analyses"]:
        first_ox = metal_analyses[0]["oxidation_state_analyses"][0]
        rec_mult = first_ox["recommended_multiplicity"]
        rec_spin = first_ox["recommended_spin_state"]

    summary_lines = []
    for ma in metal_analyses:
        for oa in ma["oxidation_state_analyses"]:
            mults = ", ".join(f"{s['spin_state']}=mult{s['multiplicity']}" for s in oa["spin_states"])
            summary_lines.append(
                f"{ma['element']}({oa['oxidation_state']:+d}) d{oa['d_count']}: {mults}"
                f" → recommended {oa['recommended_spin_state']} (mult={oa['recommended_multiplicity']},"
                f" nopen={oa['recommended_multiplicity'] - 1})"
            )

    return {
        "elements": unique_elements,
        "charge": charge,
        "has_transition_metals": True,
        "metal_elements": metal_elements,
        "ligand_elements": ligand_elements,
        "ligand_field_hints": {"has_strong_field": has_strong_field, "has_weak_field": has_weak_field},
        "metal_analyses": metal_analyses,
        "recommended_multiplicity": rec_mult,
        "recommended_spin_state": rec_spin,
        "recommended_nopen": (rec_mult - 1) if rec_mult is not None else None,
        "summary": "\n".join(summary_lines) if summary_lines else "No analysis available.",
    }


# ---------------------------------------------------------------------------
# Basis set advisor
# ---------------------------------------------------------------------------

def suggest_basis_set(
    elements: list[str],
    purpose: str = "geometry",
    library_path: str | None = None,
) -> dict[str, Any]:
    """Suggest an appropriate basis set (and ECP when needed) for a molecule.

    Args:
        elements: Element symbols present in the molecule.
        purpose: One of "geometry" (fast opt), "single_point" (DFT energy),
                 "correlated" (MP2/CCSD), or "heavy_elements" (post-Kr metals).
        library_path: Optional path to basis library (used only for validation note).

    Returns dict with 'basis_assignments', 'ecp_assignments', 'recommended_basis',
    and 'notes' ready to pass to create_nwchem_input.
    """
    norm = list(dict.fromkeys(e[0].upper() + e[1:].lower() for e in elements))
    heavy = [e for e in norm if ELEMENT_TO_Z.get(e, 0) > 36]
    has_heavy = bool(heavy)
    has_tm = any(e in _TRANSITION_METALS for e in norm)
    has_lanthanides = any(57 <= ELEMENT_TO_Z.get(e, 0) <= 71 for e in norm)

    p = purpose.strip().lower()

    if p in ("geometry", "opt", "optimization"):
        basis = "def2-svp"
        ecp = "def2-ecp" if has_heavy else None
        explanation = (
            "def2-SVP for geometry optimization — balanced speed and accuracy. "
            + ("def2-ECP applied to heavy elements (Z>36). " if has_heavy else "")
        )
        alternatives = ["def2-tzvp", "6-31gs"]
    elif p in ("single_point", "sp", "energy", "dft"):
        basis = "def2-tzvp"
        ecp = "def2-ecp" if has_heavy else None
        explanation = (
            "def2-TZVP for production DFT single-point energies. "
            + ("def2-ECP for heavy elements (Z>36). " if has_heavy else "")
        )
        alternatives = ["def2-svp", "cc-pvtz"]
    elif p in ("correlated", "ccsd", "mp2", "post-hf", "wft"):
        if has_heavy or has_tm:
            basis = "def2-tzvp"
            ecp = "def2-ecp" if has_heavy else None
            explanation = (
                "def2-TZVP for correlated calculations with transition metals. "
                + ("def2-ECP for heavy elements. " if has_heavy else "")
                + "For pure main-group systems, cc-pVTZ is preferred."
            )
            alternatives = ["cc-pvtz", "def2-svp"]
        else:
            basis = "cc-pvtz"
            ecp = None
            explanation = (
                "cc-pVTZ for correlated methods (MP2, CCSD, CCSD(T)) on main-group elements. "
                "Designed for systematic basis-set convergence."
            )
            alternatives = ["cc-pvdz", "aug-cc-pvtz", "def2-tzvp"]
    elif p in ("heavy", "heavy_elements", "lanthanides", "actinides"):
        basis = "def2-tzvp"
        ecp = "def2-ecp"
        explanation = "def2-TZVP + Stuttgart def2-ECP for relativistic treatment of heavy elements."
        if has_lanthanides:
            explanation += " Note: lanthanides may need dedicated f-basis (e.g. ano-rcc or cc-pVTZ-PP)."
        alternatives = ["def2-svp", "crenbl"]
    else:
        basis = "def2-svp"
        ecp = "def2-ecp" if has_heavy else None
        explanation = f"Unknown purpose '{purpose}'; defaulting to def2-SVP."
        alternatives = ["def2-tzvp"]

    basis_assignments = {e: basis for e in norm}
    ecp_assignments: dict[str, str] | None = None
    if ecp:
        ecp_assignments = {e: ecp for e in heavy}
        if not ecp_assignments:
            ecp_assignments = None

    return {
        "elements": norm,
        "purpose": p,
        "has_heavy_elements": has_heavy,
        "has_transition_metals": has_tm,
        "recommended_basis": basis,
        "recommended_ecp": ecp,
        "explanation": explanation.strip(),
        "alternatives": alternatives,
        "basis_assignments": basis_assignments,
        "ecp_assignments": ecp_assignments,
        "usage_note": (
            "Pass basis_assignments (and ecp_assignments if not None) directly to "
            "create_nwchem_input or create_nwchem_dft_workflow_input."
        ),
    }


# ---------------------------------------------------------------------------
# Memory advisor
# ---------------------------------------------------------------------------

_BASIS_SCALE: dict[str, float] = {
    "sto-3g": 0.3, "sto": 0.3,
    "3-21g": 0.5, "3-21": 0.5,
    "6-31g": 1.0, "6-31gs": 1.0, "6-31gss": 1.2, "6-311g": 1.5,
    "svp": 1.0, "def2-svp": 1.0, "def2-svpp": 1.2,
    "tzvp": 2.5, "def2-tzvp": 2.5, "def2-tzvpp": 3.0,
    "qzvp": 6.0, "def2-qzvp": 6.0,
    # Dunning correlation-consistent families
    "pvdz": 1.0, "cc-pvdz": 1.0, "aug-cc-pvdz": 1.4,
    "pvtz": 2.5, "cc-pvtz": 2.5, "aug-cc-pvtz": 3.5,
    "pvqz": 6.0, "cc-pvqz": 6.0, "aug-cc-pvqz": 8.0,
    "pv5z": 12.0, "cc-pv5z": 12.0,
    # Douglas-Kroll Dunning (same size as base set)
    "pvdz-dk": 1.0, "cc-pvdz-dk": 1.0,
    "pvtz-dk": 2.5, "cc-pvtz-dk": 2.5,
    "pvqz-dk": 6.0, "cc-pvqz-dk": 6.0,
    # Segmented DK (Stuttgart)
    "dhf-svp": 1.0, "dhf-tzvp": 2.5, "dhf-tzvpp": 3.0,
    # ANO families
    "ano-rcc": 3.0, "ano-r": 2.5,
    # Pople diffuse
    "6-31+g": 1.2, "6-31++g": 1.4, "6-311+g": 1.8,
}


def _analyze_job_size(input_file: str) -> dict[str, Any]:
    """Shared helper: inspect input and extract job-size metrics.

    Returns dict with: summary, all_elements, n_atoms, n_heavy, tasks,
    main_task, module, operation, is_freq, is_opt, is_tce, basis_name,
    basis_scale, n_bf.
    """
    from .nwchem_input import inspect_nwchem_input, inspect_all_nwchem_basis_blocks

    summary = inspect_nwchem_input(input_file)
    all_elements = summary.get("all_elements") or summary.get("elements", [])
    n_atoms = summary.get("atom_count") or len(all_elements) or 1
    n_heavy = sum(1 for e in all_elements if e != "H") if all_elements else n_atoms
    tasks = summary.get("tasks") or [{}]
    main_task = tasks[-1] if tasks else {}
    module = (main_task.get("module") or "dft").lower()
    operation = (main_task.get("operation") or "energy").lower()

    is_freq = operation in ("freq", "frequencies", "vib")
    is_opt = operation in ("optimize", "saddle")
    is_tce = module == "tce"

    basis_blocks = inspect_all_nwchem_basis_blocks(input_file)
    basis_name = ""
    if basis_blocks:
        basis_name = basis_blocks[0].get("default_library", "") or ""
    scale = _basis_scale(basis_name) if basis_name else 1.5
    n_bf = max(10, int(n_heavy * 15 * scale))

    return {
        "summary": summary, "all_elements": all_elements,
        "n_atoms": n_atoms, "n_heavy": n_heavy,
        "tasks": tasks, "main_task": main_task,
        "module": module, "operation": operation,
        "is_freq": is_freq, "is_opt": is_opt, "is_tce": is_tce,
        "basis_name": basis_name, "basis_scale": scale, "n_bf": n_bf,
    }


def _basis_scale(basis: str) -> float:
    b = basis.strip().lower()
    if b in _BASIS_SCALE:
        return _BASIS_SCALE[b]
    for key, scale in sorted(_BASIS_SCALE.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key in b:
            return scale
    return 1.5


# Empirical target: basis functions per MPI rank for good parallel efficiency.
# Below this, communication overhead dominates.
_BF_PER_RANK_TARGET: dict[str, int] = {
    "spr":     60,   # AVX-512, high memory bandwidth (Stampede3 SPR)
    "skx":     80,   # AVX-512, standard Skylake (Stampede3 SKX)
    "icx":     75,   # AVX-512, Ice Lake — slightly faster than SKX (Stampede3 ICX)
    "avx512":  70,
    "avx2":    90,
    "knl":    120,   # KNL: high core count but weak single-core
    "generic": 80,
}


def suggest_resources(
    input_file: str,
    hw_specs: dict[str, Any],
) -> dict[str, Any]:
    """Recommend mpi_ranks and memory_per_rank_mb for a NWChem job.

    .. deprecated::
        For HPC jobs, use :func:`suggest_hpc_resources` instead — it is
        profile-aware, multi-node capable, and handles task-type-specific
        walltime and memory estimation.  This function only handles
        single-node rank/memory selection.

    Args:
        input_file: Path to the NWChem .nw input file.
        hw_specs: Hardware specs dict.
            Expected keys: cpus_per_node (or available_cores), node_memory_mb
            (or available_mem_mb), cpu_arch.
    """
    job = _analyze_job_size(input_file)
    n_atoms = job["n_atoms"]
    M = job["n_bf"]
    method = job["module"]
    basis_name = job["basis_name"]

    # CPU-arch-aware parallelism target
    arch = hw_specs.get("cpu_arch", "generic")
    bf_per_rank = _BF_PER_RANK_TARGET.get(arch, 80)

    # Ranks from scaling model
    max_cores = hw_specs.get("cpus_per_node") or hw_specs.get("available_cores") or 1
    ranks_by_scaling = max(1, M // bf_per_rank)

    # Ranks from memory budget
    node_mem = hw_specs.get("node_memory_mb") or hw_specs.get("available_mem_mb")
    if node_mem:
        min_mem_per_rank = 400  # MB: floor for NWChem to start
        ranks_by_memory = max(1, int(node_mem * 0.80 / min_mem_per_rank))
    else:
        ranks_by_memory = max_cores

    optimal_ranks = min(ranks_by_scaling, ranks_by_memory, max_cores)
    optimal_ranks = max(1, optimal_ranks)

    rationale = f"BF/rank model: {M} BF / {bf_per_rank} target = {ranks_by_scaling} ranks"

    # Memory per rank
    if node_mem:
        mem_per_rank = int(node_mem * 0.80 / optimal_ranks)
    else:
        mem_suggestion = suggest_memory(
            n_atoms=n_atoms, basis=basis_name or "6-31g*", method=method,
        )
        mem_per_rank = mem_suggestion["recommended_total_mb"]

    return {
        "mpi_ranks": optimal_ranks,
        "memory_per_rank_mb": mem_per_rank,
        "estimated_basis_functions": M,
        "bf_per_rank_actual": round(M / optimal_ranks, 1),
        "cpu_arch": arch,
        "max_cores_available": max_cores,
        "node_memory_mb": node_mem,
        "rationale": rationale,
    }


def suggest_memory(
    n_atoms: int,
    basis: str,
    method: str,
    n_heavy_atoms: int | None = None,
) -> dict[str, Any]:
    """Suggest NWChem memory settings for a calculation.

    Returns a memory string ready for NWChem's ``memory`` directive.

    Args:
        n_atoms: Total number of atoms.
        basis: Basis set name (used to scale memory estimate).
        method: Computational method: "scf", "dft", "mp2", "ccsd", "ccsd(t)".
        n_heavy_atoms: Number of non-hydrogen atoms (optional; uses n_atoms if omitted).

    Returns dict with 'nwchem_directive' and 'memory_string'.
    """
    eff = n_heavy_atoms if n_heavy_atoms is not None else n_atoms
    scale = _basis_scale(basis)
    m = method.strip().lower()

    # Estimated basis functions: ~15 per heavy atom at double-zeta baseline
    n_bf = max(10, int(eff * 15 * scale))

    # SCF/DFT: dominated by Fock matrix + AO integrals
    fock_mb = max(64, int(8 * n_bf ** 2 / 1e6))

    if m in ("scf", "dft", "hf", "rhf", "rohf", "uhf"):
        total_mb = max(500, fock_mb * 4)
    elif m == "mp2":
        # n_occ ~ n_bf/3 is a reliable heuristic for typical neutral molecules
        # (roughly: 1/3 of basis functions are occupied at double-zeta)
        n_occ = max(1, n_bf // 3)
        n_virt = max(1, n_bf - n_occ)
        t2_mb = max(256, int(8 * (n_occ * n_virt) ** 2 / 1e6 / 4))
        total_mb = max(1000, fock_mb * 2 + t2_mb * 3)
    elif m in ("ccsd", "ccsd(t)", "tce"):
        n_occ = max(1, n_bf // 3)
        n_virt = max(1, n_bf - n_occ)
        t2_mb = max(256, int(8 * (n_occ * n_virt) ** 2 / 1e6 / 4))
        total_mb = max(2000, fock_mb * 2 + t2_mb * 6)
    else:
        total_mb = max(1000, fock_mb * 4)

    # Round to nearest 500 mb, cap at 128 GB
    total_mb = min(((total_mb + 499) // 500) * 500, 128 * 1024)

    heap_mb = max(128, total_mb // 4)
    stack_mb = max(128, total_mb // 6)
    global_mb = max(256, total_mb - heap_mb - stack_mb)
    # Ensure total >= sum of sub-components (max() floors can push sum over total)
    total_mb = max(total_mb, heap_mb + stack_mb + global_mb)

    memory_string = f"total {total_mb} mb stack {stack_mb} mb heap {heap_mb} mb global {global_mb} mb"

    return {
        "n_atoms": n_atoms,
        "n_heavy_atoms": eff,
        "basis": basis,
        "method": m,
        "basis_scale_factor": scale,
        "estimated_basis_functions": n_bf,
        "recommended_total_mb": total_mb,
        "memory_string": memory_string,
        "nwchem_directive": f"memory {memory_string}",
        "notes": (
            "Estimates are heuristic. Increase if NWChem aborts with out-of-memory errors. "
            "For CCSD(T) memory is the dominant bottleneck — more is always better."
        ),
    }


# ---------------------------------------------------------------------------
# Memory fitness check (profile-aware)
# ---------------------------------------------------------------------------


def check_memory_fit(
    input_file: str,
    profile_resources: dict[str, Any] | None = None,
    nodes: int = 1,
    mpi_ranks: int = 1,
    node_memory_mb: int | None = None,
) -> dict[str, Any]:
    """Check if an NWChem input's memory directive fits the target node.

    Reads the ``memory total`` line from *input_file* and compares against
    the node capacity.  Returns warnings and a corrected memory string when
    the requested allocation would exceed available RAM.

    *profile_resources* is the ``resources`` dict from a runner profile.
    If provided, ``nodes``, ``mpi_ranks``, and ``node_memory_mb`` are read
    from it (explicit kwargs override).
    """
    pr = profile_resources or {}
    nodes = nodes if nodes != 1 else int(pr.get("nodes", nodes))
    mpi_ranks = mpi_ranks if mpi_ranks != 1 else int(pr.get("mpi_ranks", mpi_ranks))
    node_memory_mb = node_memory_mb or pr.get("node_memory_mb")

    # Read input to find memory directive
    text = Path(input_file).read_text(encoding="utf-8", errors="replace")
    mem_line = ""
    for line in text.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("memory "):
            mem_line = stripped
            break

    # Extract total MB from memory line
    requested_mb_per_rank = 0
    if mem_line:
        import re
        m = re.search(r"total\s+(\d+)\s*(mb|mw|gb)", mem_line)
        if m:
            val = int(m.group(1))
            unit = m.group(2)
            if unit == "gb":
                val *= 1024
            elif unit == "mw":
                val *= 8  # 1 MW = 8 MB (64-bit words)
            requested_mb_per_rank = val

    if not requested_mb_per_rank:
        return {
            "status": "no_memory_directive",
            "message": "No 'memory total' directive found in input. NWChem will use defaults.",
            "warnings": [],
        }

    ranks_per_node = max(1, mpi_ranks // max(1, nodes))
    total_requested_per_node = requested_mb_per_rank * ranks_per_node

    warnings: list[dict[str, Any]] = []
    safe_mb_per_rank: int | None = None

    if node_memory_mb:
        # Reserve 15% for OS + MPI runtime
        usable_mb = int(node_memory_mb * 0.85)
        if total_requested_per_node > usable_mb:
            safe_mb_per_rank = max(400, (usable_mb // ranks_per_node // 100) * 100)
            warnings.append({
                "code": "memory_exceeds_node",
                "severity": "error",
                "message": (
                    f"Requested {requested_mb_per_rank} MB/rank × {ranks_per_node} ranks "
                    f"= {total_requested_per_node} MB, but node has {node_memory_mb} MB "
                    f"(~{usable_mb} MB usable). Job will crash with MA_init error."
                ),
                "fix": f"memory total {safe_mb_per_rank} mb",
            })
        elif total_requested_per_node > usable_mb * 0.9:
            warnings.append({
                "code": "memory_tight",
                "severity": "warning",
                "message": (
                    f"Requested {total_requested_per_node} MB/node is within 10% of "
                    f"usable capacity ({usable_mb} MB). Consider reducing for safety."
                ),
            })

    return {
        "status": "error" if any(w["severity"] == "error" for w in warnings) else (
            "warning" if warnings else "ok"),
        "requested_mb_per_rank": requested_mb_per_rank,
        "ranks_per_node": ranks_per_node,
        "total_mb_per_node": total_requested_per_node,
        "node_memory_mb": node_memory_mb,
        "safe_mb_per_rank": safe_mb_per_rank,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Frequency walltime estimation
# ---------------------------------------------------------------------------


def estimate_freq_walltime(
    n_atoms: int,
    seconds_per_displacement: float | None = None,
    n_displacements: int | None = None,
    mpi_ranks: int = 1,
    nodes: int = 1,
    max_walltime_hours: float = 48.0,
) -> dict[str, Any]:
    """Estimate walltime needed for a numerical frequency calculation.

    NWChem numerical frequencies require 6*N_atoms gradient evaluations
    (central differences: +/- displacement for each Cartesian coordinate).
    Each gradient evaluation is roughly the cost of a single-point SCF.

    If *seconds_per_displacement* is not provided, uses a rough heuristic
    based on atom count and MPI parallelism.

    IMPORTANT: NWChem cannot checkpoint mid-frequency. If the job runs out
    of walltime, ALL progress is lost. The job must complete in one submission.
    """
    if n_displacements is None:
        n_displacements = n_atoms * 6  # central differences: ±Δ for x,y,z per atom

    if seconds_per_displacement is None:
        # Rough heuristic: ~5 min per displacement for a 20-atom DFT/6-31G* on 48 cores
        # Scale quadratically with atom count, inversely with MPI parallelism
        base_seconds = 300.0 * (n_atoms / 20.0) ** 1.5
        total_ranks = mpi_ranks * nodes
        # Diminishing returns past ~64 ranks for intra-displacement parallelism
        effective_speedup = min(total_ranks, 64) + max(0, total_ranks - 64) * 0.3
        seconds_per_displacement = base_seconds * (48.0 / max(1, effective_speedup))

    total_seconds = n_displacements * seconds_per_displacement
    total_hours = total_seconds / 3600.0

    fits_in_walltime = total_hours <= max_walltime_hours
    safety_margin = max_walltime_hours - total_hours if fits_in_walltime else 0

    # Estimate how many nodes needed to fit in max_walltime
    if not fits_in_walltime and nodes == 1:
        # Try scaling up nodes
        for n in [2, 3, 4, 6, 8]:
            total_ranks_n = mpi_ranks * n
            eff = min(total_ranks_n, 64) + max(0, total_ranks_n - 64) * 0.3
            scaled_seconds = (48.0 / max(1, eff)) * 300.0 * (n_atoms / 20.0) ** 1.5
            scaled_hours = (n_displacements * scaled_seconds) / 3600.0
            if scaled_hours <= max_walltime_hours * 0.9:
                suggested_nodes = n
                break
        else:
            suggested_nodes = None
    else:
        suggested_nodes = None

    result: dict[str, Any] = {
        "n_atoms": n_atoms,
        "n_displacements": n_displacements,
        "seconds_per_displacement": round(seconds_per_displacement, 1),
        "estimated_total_hours": round(total_hours, 1),
        "max_walltime_hours": max_walltime_hours,
        "fits_in_walltime": fits_in_walltime,
        "safety_margin_hours": round(safety_margin, 1),
        "mpi_ranks": mpi_ranks,
        "nodes": nodes,
        "cannot_checkpoint": True,
        "warning": (
            "NWChem CANNOT checkpoint numerical frequency calculations. "
            "If the job exceeds walltime, ALL progress is lost. "
            "Ensure sufficient walltime and consider multi-node to speed up."
        ),
    }
    if suggested_nodes:
        result["suggested_nodes"] = suggested_nodes
        result["suggestion"] = (
            f"Single-node estimate is {total_hours:.0f}h which exceeds "
            f"{max_walltime_hours:.0f}h walltime. Use {suggested_nodes} nodes "
            f"({mpi_ranks * suggested_nodes} total MPI ranks) to fit within walltime."
        )
    elif not fits_in_walltime:
        result["suggestion"] = (
            f"Estimated {total_hours:.0f}h exceeds {max_walltime_hours:.0f}h walltime. "
            f"Even with multi-node scaling this may not fit. Consider analytical "
            f"frequencies (if available) or a smaller basis set."
        )

    return result


# ---------------------------------------------------------------------------
# Relativistic correction advisor
# ---------------------------------------------------------------------------

# Elements where relativistic effects are chemically significant
# 3d TMs (Z>=21): notable core-level effects; DK-basis use makes X2C appropriate
# 4d/heavy main-group (Z>=37): scalar relativistic important for energetics
# 5d metals, lanthanides, actinides (Z>=57): mandatory
_REL_SIGNIFICANT_Z = 21   # 3d transition metals — recommend when DK basis detected
_REL_IMPORTANT_Z = 37     # 4d metals and heavier — strongly recommend
_REL_CRITICAL_Z = 57      # 5d metals, lanthanides, actinides — mandatory

# DK-quality basis sets (designed for relativistic calculations)
_DK_BASIS_PATTERNS = {
    "cc-pvdz-dk", "cc-pvtz-dk", "cc-pvqz-dk", "cc-pv5z-dk",
    "aug-cc-pvdz-dk", "aug-cc-pvtz-dk", "aug-cc-pvqz-dk",
    "cc-pwcvdz-dk", "cc-pwcvtz-dk", "cc-pwcvqz-dk",
    "x2c-svpall", "x2c-tzvpall", "x2c-qzvpall",
    "dyall-v2z", "dyall-v3z", "dyall-v4z",
    "sarc-dkh2",
}

# Relativistic methods available in NWChem
_REL_METHODS = {
    "x2c": {
        "nwchem_block": "relativistic\n  x2c\nend",
        "description": "Exact Two-Component (X2C) — recommended for production quality. "
                       "Decouples large and small components exactly at the 1-electron level. "
                       "Use with DK-family basis sets (cc-pVDZ-DK, cc-pVTZ-DK, etc.).",
        "cost": "moderate",
        "suitable_for": ["single_point", "optimization", "frequency", "mp2", "ccsd"],
    },
    "dkh2": {
        "nwchem_block": "relativistic\n  douglas-kroll 2\nend",
        "description": "Douglas-Kroll-Hess 2nd order (DKH2) — widely tested, good accuracy "
                       "for 4d/5d metals. Use with DK-family basis sets.",
        "cost": "moderate",
        "suitable_for": ["single_point", "optimization", "frequency"],
    },
    "dkh3": {
        "nwchem_block": "relativistic\n  douglas-kroll 3\nend",
        "description": "Douglas-Kroll-Hess 3rd order (DKH3) — higher-order correction over DKH2. "
                       "Minimal improvement over DKH2 in most cases.",
        "cost": "moderate",
        "suitable_for": ["single_point"],
    },
    "zora": {
        "nwchem_block": "relativistic\n  zora\nend",
        "description": "ZORA (Zeroth Order Regular Approximation) — lower cost but less rigorous. "
                       "Not recommended for high-accuracy work.",
        "cost": "low",
        "suitable_for": ["single_point", "optimization"],
    },
}


def suggest_relativistic_correction(
    elements: list[str],
    basis_assignments: dict[str, str] | None = None,
    ecp_assignments: dict[str, str] | None = None,
    purpose: str = "dft",
) -> dict[str, Any]:
    """Advise on relativistic corrections for a molecular calculation.

    Returns a recommendation (or "none needed") with the NWChem block to add,
    and compatibility warnings when ECPs are present.

    Parameters
    ----------
    elements:
        All element symbols in the system.
    basis_assignments:
        Dict mapping element → basis name.  Used to detect DK-type bases.
    ecp_assignments:
        Dict mapping element → ECP name.  If present, warns about X2C/DKH incompatibility.
    purpose:
        One of "dft", "scf", "ccsd", "property".  Affects recommendation.

    Returns dict with ``recommended_method``, ``nwchem_block``, ``reason``,
    ``warnings``, and ``per_element_z_scores``.
    """
    norm = [e[0].upper() + e[1:].lower() for e in elements]
    unique = list(dict.fromkeys(norm))

    has_ecp = bool(ecp_assignments)
    ecp_elements = list((ecp_assignments or {}).keys())

    # Z-based analysis
    per_element: list[dict[str, Any]] = []
    max_z = 0
    for el in unique:
        z = ELEMENT_TO_Z.get(el, 0)
        max_z = max(max_z, z)
        if z >= _REL_CRITICAL_Z:
            level = "critical"
        elif z >= _REL_IMPORTANT_Z:
            level = "important"
        elif z >= _REL_SIGNIFICANT_Z:
            level = "significant"
        elif z >= 18:
            level = "minor"
        else:
            level = "negligible"
        per_element.append({"element": el, "Z": z, "relativistic_importance": level})

    has_critical = any(p["relativistic_importance"] == "critical" for p in per_element)
    has_important = any(p["relativistic_importance"] == "important" for p in per_element)
    has_significant = any(p["relativistic_importance"] == "significant" for p in per_element)

    # Detect DK basis sets
    basis_lower = {k.lower(): v.lower() for k, v in (basis_assignments or {}).items()}
    has_dk_basis = any(
        any(b.startswith(dk) or b in _DK_BASIS_PATTERNS for dk in _DK_BASIS_PATTERNS)
        for b in basis_lower.values()
    ) if basis_lower else False

    warnings: list[str] = []
    incompatible_elements: list[str] = []

    if has_ecp:
        # X2C/DKH treat core relativistic effects via all-electron; ECP replaces the core.
        # Using both is either redundant (same element) or inconsistent.
        incompatible_elements = [
            el for el in ecp_elements
            if el in unique
        ]
        if incompatible_elements:
            warnings.append(
                f"INCOMPATIBILITY: Elements {incompatible_elements} use ECPs — "
                "X2C and DKH are all-electron methods that replace ECPs. "
                "You must choose ONE: (a) all-electron + relativistic block, OR "
                "(b) ECP (removes core electrons; no relativistic block needed). "
                "Using both for the same element is incorrect."
            )

    # Detect Pople-style basis sets — they use SP shells, incompatible with X2C/DKH
    _POPLE_PATTERNS = ("sto-", "3-21g", "6-21g", "4-31g", "6-31g", "6-311g", "6-31+g", "6-311+g")
    pople_elements: list[str] = []
    for el, bname in (basis_assignments or {}).items():
        bname_lower = bname.lower().replace(" ", "")
        if any(bname_lower.startswith(p) or p in bname_lower for p in _POPLE_PATTERNS):
            pople_elements.append(el)
    has_pople = bool(pople_elements)

    # Recommendation logic
    if not (has_critical or has_important or has_significant):
        recommended = "none"
        nwchem_block = None
        reason = (
            f"All elements have Z < {_REL_SIGNIFICANT_Z} — relativistic effects are negligible. "
            "No relativistic block needed."
        )
    elif has_ecp and incompatible_elements:
        # ECPs already implicitly encode relativistic effects for heavy atoms
        recommended = "ecp_implicit"
        nwchem_block = None
        reason = (
            "ECP is in use for heavy elements — the ECP implicitly accounts for scalar "
            "relativistic effects on the core. Do not add a relativistic block for ECP-covered elements. "
            "If you want explicit all-electron relativistic treatment, remove the ECP and use "
            "an all-electron basis with X2C or DKH2."
        )
    elif has_critical:
        recommended = "x2c"
        nwchem_block = _REL_METHODS["x2c"]["nwchem_block"]
        reason = (
            f"Heavy element(s) with Z ≥ {_REL_CRITICAL_Z} present (5d metals or heavy p-block). "
            "Relativistic effects are chemically critical — X2C is the recommended method. "
            "Pair with cc-pVTZ-DK, cc-pVDZ-DK, or x2c-TZVPall basis sets."
        )
        if has_pople:
            warnings.append(
                f"INCOMPATIBILITY: Pople-style basis detected for {pople_elements}. "
                "6-31G* / 6-311G** and similar Pople bases use SP-contracted shells, which are "
                "incompatible with X2C/DKH. NWChem will crash with 'dimensions not the same' "
                "during the relativistic uncontraction step. "
                "Replace with cc-pVDZ-DK, cc-pVTZ-DK, or def2-SVP / def2-TZVP."
            )
        elif not has_dk_basis:
            warnings.append(
                "BASIS WARNING: No DK-quality basis detected. "
                "X2C/DKH calculations require bases designed for relativistic calculations "
                "(cc-pVDZ-DK, cc-pVTZ-DK, x2c-SVPall, etc.). "
                "Standard def2 bases are acceptable; avoid Pople bases (SP-shell incompatibility)."
            )
    elif has_important:
        recommended = "x2c"
        nwchem_block = _REL_METHODS["x2c"]["nwchem_block"]
        reason = (
            f"Element(s) with Z ≥ {_REL_IMPORTANT_Z} (4d metals / heavy main-group) present. "
            "Scalar relativistic effects are important for accurate energetics. "
            "X2C with DK basis sets recommended."
        )
        if has_pople:
            warnings.append(
                f"INCOMPATIBILITY: Pople-style basis detected for {pople_elements}. "
                "6-31G* / 6-311G** and similar Pople bases use SP-contracted shells, which are "
                "incompatible with X2C/DKH. NWChem will crash with 'dimensions not the same'. "
                "Replace with cc-pVDZ-DK, cc-pVTZ-DK, or def2-SVP / def2-TZVP."
            )
        elif not has_dk_basis:
            warnings.append(
                "BASIS WARNING: Consider switching to cc-pVDZ-DK or cc-pVTZ-DK basis sets."
            )
    else:
        # Z >= _REL_SIGNIFICANT_Z (3d TMs): recommend X2C when DK basis present, optional otherwise
        if has_dk_basis:
            recommended = "x2c"
            nwchem_block = _REL_METHODS["x2c"]["nwchem_block"]
            reason = (
                f"DK-type basis set detected with element(s) in the 3d/4d transition metal range. "
                "DK-family bases are designed for use with relativistic Hamiltonians (X2C or DKH2). "
                "X2C is strongly recommended — using a DK basis without a relativistic block "
                "gives inconsistent results."
            )
        else:
            recommended = "x2c_optional"
            nwchem_block = _REL_METHODS["x2c"]["nwchem_block"]
            reason = (
                f"Element(s) with Z ≥ {_REL_SIGNIFICANT_Z} present — scalar relativistic effects "
                "are non-negligible but often acceptable without correction at this level. "
                "Add X2C with a DK-type basis if targeting high accuracy."
            )
        if has_pople and recommended in ("x2c", "x2c_optional"):
            warnings.append(
                f"INCOMPATIBILITY: Pople-style basis detected for {pople_elements}. "
                "6-31G* / 6-311G** and similar Pople bases use SP-contracted shells, which are "
                "incompatible with X2C/DKH. NWChem will crash with 'dimensions not the same'. "
                "Replace with cc-pVDZ, cc-pVTZ, def2-SVP, or def2-TZVP."
            )

    # Performance note for X2C + SAD
    sad_note: str | None = None
    if recommended in ("x2c", "x2c_optional") and nwchem_block:
        heavy_tms = [p["element"] for p in per_element if p["Z"] >= _REL_SIGNIFICANT_Z]
        if heavy_tms:
            sad_note = (
                f"PERFORMANCE NOTE: X2C requires solving relativistic atomic SCFs for {heavy_tms} "
                "during the SAD initial guess. This runs with no output for potentially 30–120+ minutes. "
                "This is expected behavior — do not terminate the job during this phase."
            )
            warnings.append(sad_note)

    return {
        "recommended_method": recommended,
        "nwchem_block": nwchem_block,
        "reason": reason,
        "per_element": per_element,
        "max_z": max_z,
        "has_dk_basis": has_dk_basis,
        "has_ecp": has_ecp,
        "ecp_incompatible_elements": incompatible_elements,
        "has_pople_basis": has_pople,
        "pople_basis_elements": pople_elements,
        "available_methods": {k: {
            "nwchem_block": v["nwchem_block"],
            "description": v["description"],
            "cost": v["cost"],
        } for k, v in _REL_METHODS.items()},
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Frequency restart helper
# ---------------------------------------------------------------------------

def prepare_freq_restart(
    input_file: str,
    output_file: str,
    profile: str | None = None,
) -> dict[str, Any]:
    """Validate that a freq restart is ready and return a submit-ready report.

    Checks: restart keyword present, .fdrst exists, reports progress.
    Does NOT submit — caller decides whether to call launch_nwchem_run.
    """
    import re
    from pathlib import Path
    from .common import read_text
    from .nwchem_freq import parse_freq_progress

    nw_text = Path(input_file).read_text(encoding="utf-8")
    issues: list[str] = []

    # Check restart keyword
    has_restart = bool(re.search(r"^\s*restart\b", nw_text, re.MULTILINE | re.IGNORECASE))
    if not has_restart:
        issues.append("Input is missing 'restart' keyword — NWChem will start from scratch")

    # Determine restart prefix name
    restart_match = re.search(r"^\s*restart\s+(\S+)", nw_text, re.MULTILINE | re.IGNORECASE)
    if restart_match:
        restart_prefix = restart_match.group(1)
    else:
        start_match = re.search(r"^\s*start\s+(\S+)", nw_text, re.MULTILINE | re.IGNORECASE)
        restart_prefix = start_match.group(1) if start_match else Path(input_file).stem

    # Check .fdrst and .db exist
    job_dir = Path(input_file).parent
    fdrst_path = job_dir / f"{restart_prefix}.fdrst"
    db_path = job_dir / f"{restart_prefix}.db"

    has_fdrst = fdrst_path.exists()
    has_db = db_path.exists()
    if not has_fdrst:
        issues.append(f"Checkpoint file {fdrst_path.name} not found — freq will start from atom 1")
    if not has_db:
        issues.append(f"Database file {restart_prefix}.db not found — restart may fail")

    fdrst_info: dict[str, Any] = {"path": str(fdrst_path), "exists": has_fdrst}
    if has_fdrst:
        from datetime import datetime, timezone as _tz
        stat = fdrst_path.stat()
        fdrst_info.update({
            "size_kb": round(stat.st_size / 1024, 1),
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=_tz.utc).isoformat(),
        })

    # Parse progress from previous output
    progress: dict[str, Any] = {}
    out_path = Path(output_file)
    if out_path.exists():
        try:
            out_text = read_text(str(out_path))
            progress = parse_freq_progress(str(out_path), out_text)
        except Exception:
            pass

    return {
        "ready_to_restart": len(issues) == 0,
        "issues": issues,
        "input_file": str(Path(input_file).resolve()),
        "restart_prefix": restart_prefix,
        "has_restart_keyword": has_restart,
        "fdrst": fdrst_info,
        "db_exists": has_db,
        "progress": progress,
        "suggested_profile": profile,
    }


# ---------------------------------------------------------------------------
# Preflight check
# ---------------------------------------------------------------------------

def preflight_check(
    input_file: str,
    profile: str,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    """Run all pre-submission checks and return a pass/fail report.

    Combines: lint, movecs file existence, memory vs node RAM ceiling.
    """
    import re
    from pathlib import Path
    from .api_input import lint_nwchem_input
    from .runner import load_runner_profiles, _resolve_profile

    checks: list[dict[str, Any]] = []

    # 1. Lint
    lint = lint_nwchem_input(input_file)
    lint_errors = [i for i in lint.get("issues", []) if i.get("level") == "error"]
    checks.append({
        "check": "lint",
        "passed": len(lint_errors) == 0,
        "issues": lint.get("issues", []),
    })

    # 2. movecs input files exist
    nw_text = Path(input_file).read_text(encoding="utf-8")
    job_dir = Path(input_file).parent
    for m in re.finditer(r"vectors\s+input\s+(\S+)", nw_text, re.IGNORECASE):
        movecs_name = m.group(1)
        movecs_path = job_dir / movecs_name
        exists = movecs_path.exists()
        checks.append({
            "check": f"movecs_exists:{movecs_name}",
            "passed": exists,
            "issues": [] if exists else [
                {"level": "error", "message": f"vectors input file not found: {movecs_path}"}
            ],
        })

    # 3. Memory vs. node RAM
    try:
        profiles = load_runner_profiles(profiles_path)
        profile_payload = _resolve_profile(profiles, profile)
        resources = profile_payload.get("resources", {})
        partition = resources.get("partition")
        launcher = profile_payload.get("launcher", {})
        scheduler_type = (
            profile_payload.get("scheduler", {}).get("system")
            or launcher.get("scheduler_type", "slurm")
        ).lower()

        # Query real partition specs if on a scheduler
        node_mem_mb = None
        cpus_per_node = None
        if launcher.get("kind") == "scheduler" and partition:
            from .runner import query_partition_specs
            hw = query_partition_specs(partition, scheduler_type)
            node_mem_mb = hw.get("node_memory_mb")
            cpus_per_node = hw.get("cpus_per_node")

        # Parse memory directive from input
        mem_match = re.search(r"memory\s+(?:total\s+)?(\d+)\s*(mb|mw|gb)", nw_text, re.IGNORECASE)
        if mem_match and node_mem_mb:
            mem_val = int(mem_match.group(1))
            mem_unit = mem_match.group(2).lower()
            if mem_unit == "gb":
                mem_val *= 1024
            elif mem_unit == "mw":
                mem_val *= 8  # 1 MW = 8 MB

            mpi_ranks = resources.get("mpi_ranks", cpus_per_node or 1)
            total_requested_mb = mem_val * mpi_ranks
            ceiling_mb = int(node_mem_mb * 0.90)
            ok = total_requested_mb <= ceiling_mb
            checks.append({
                "check": "memory_ceiling",
                "passed": ok,
                "details": {
                    "memory_per_rank_mb": mem_val,
                    "mpi_ranks": mpi_ranks,
                    "total_requested_mb": total_requested_mb,
                    "node_memory_mb": node_mem_mb,
                    "ceiling_90pct_mb": ceiling_mb,
                },
                "issues": [] if ok else [
                    {"level": "error",
                     "message": (
                         f"Memory request {total_requested_mb} MB ({mem_val} MB × {mpi_ranks} ranks) "
                         f"exceeds 90% of node RAM ({node_mem_mb} MB). "
                         f"Reduce memory directive or mpi_ranks."
                     )}
                ],
            })
    except Exception as exc:
        checks.append({
            "check": "memory_ceiling",
            "passed": True,
            "issues": [{"level": "info", "message": f"Could not check memory ceiling: {exc}"}],
        })

    all_passed = all(c["passed"] for c in checks)
    return {
        "ready_to_submit": all_passed,
        "checks": checks,
        "summary": (
            f"{'PASS' if all_passed else 'FAIL'}: "
            f"{sum(c['passed'] for c in checks)}/{len(checks)} checks passed"
        ),
    }


# ---------------------------------------------------------------------------
# Workflow state machine
# ---------------------------------------------------------------------------

WORKFLOW_STATES = [
    "pending",                # not yet submitted
    "queued",                 # submitted, waiting in scheduler queue
    "running_scf",            # SCF/DFT in progress
    "running_opt",            # geometry optimization in progress
    "running_freq",           # freq displacements in progress
    "freq_timelimited",       # freq hit walltime, fdrst exists, can restart
    "freq_complete",          # all displacements done
    "opt_converged",          # geometry optimized, ready for next step
    "opt_failed_convergence", # optimization not converging
    "scf_failed",             # SCF not converging
    "imaginary_modes",        # freq done but imaginary modes present
    "oom",                    # out of memory
    "completed",              # all tasks done, results valid
    "cancelled",              # job was cancelled
    "failed",                 # generic failure
    "needs_user_input",       # tool cannot decide; escalate to human
]


def get_nwchem_workflow_state(
    input_file: str | None = None,
    output_file: str = "",
    profile: str = "",
    error_file: str | None = None,
) -> dict[str, Any]:
    """Determine workflow state and return the exact next tool call to advance.

    Encodes domain logic so the LLM does not need to reason about NWChem
    internals — it just calls this tool, executes ``next_action``, and repeats.

    ``input_file`` is optional — when only the ``.out`` file is available the
    function parses the NWChem input echo from the output to determine the
    task type.  Missing companion files (``.nw``, ``.fdrst``, ``.err``, etc.)
    are reported in ``missing_files`` so the model can ask the user.

    If SLURM is available, ``squeue`` is checked for running jobs whose name
    matches the output stem.  Matches are reported in ``related_jobs`` so the
    model can ask the user whether the running job belongs to this output —
    it should **never assume** a match without confirmation.
    """
    import shutil
    import subprocess
    from .nwchem_freq import parse_freq_progress as _parse_freq_progress, analyze_imaginary_modes as _analyze_imag

    out = Path(output_file)
    inp = Path(input_file) if input_file else None

    # Try to find the input file from the output path if not provided
    if inp is None or not inp.exists():
        candidate_nw = out.with_suffix(".nw")
        if candidate_nw.exists():
            inp = candidate_nw
            input_file = str(inp)

    # Derive error file if not given
    if error_file is None:
        candidate = out.with_suffix(".err")
        if candidate.exists():
            error_file = str(candidate)

    # Track missing companion files
    missing_files: list[str] = []
    if inp is None or not inp.exists():
        missing_files.append(f"{out.stem}.nw (input file)")
    if error_file is None or not Path(error_file).exists():
        missing_files.append(f"{out.stem}.err (error file)")

    # --- Check for related running SLURM jobs ---
    related_jobs = _check_related_slurm_jobs(out.stem)

    # --- 0. Output file must exist ---
    if not out.exists() or out.stat().st_size == 0:
        if inp and inp.exists():
            # Check if job is queued via .jobid
            jobid_file = out.with_suffix(".jobid")
            if not jobid_file.exists():
                jobid_file = inp.with_suffix(".jobid")
            if jobid_file.exists():
                r = _wf_result(
                    "queued", 0,
                    "Job submitted (jobid file exists) but no output yet.",
                    {"tool": "get_nwchem_run_status",
                     "params": {"output_file": output_file,
                                "input_file": input_file or "", "profile": profile}},
                    0.85,
                )
                r["related_jobs"] = related_jobs
                r["missing_files"] = missing_files
                return r
            r = _wf_result(
                "pending", 0,
                "Output file not found or empty — job has not started.",
                {"tool": "launch_nwchem_run",
                 "params": {"input_file": input_file or "", "profile": profile}},
                0.90,
            )
            r["related_jobs"] = related_jobs
            r["missing_files"] = missing_files
            return r
        # No input, no output
        r = _wf_result(
            "needs_user_input", 0,
            "Neither input nor output file found. Provide the correct file paths.",
            None, 0.3,
        )
        r["related_jobs"] = related_jobs
        r["missing_files"] = missing_files
        return r

    contents = out.read_text(encoding="utf-8", errors="replace")

    # Read input text — from file if available, otherwise parse from output echo
    input_text = ""
    if inp and inp.exists():
        input_text = inp.read_text(encoding="utf-8", errors="replace")
    else:
        input_text = _extract_input_echo(contents)

    err_text = ""
    if error_file and Path(error_file).exists():
        err_text = Path(error_file).read_text(encoding="utf-8", errors="replace")

    # --- Determine task type from input text ---
    is_freq = bool(re.search(r"task\s+\w+\s+freq", input_text, re.IGNORECASE))
    is_opt = bool(re.search(r"task\s+\w+\s+optim", input_text, re.IGNORECASE))

    # --- Find fdrst: check both input dir and output dir ---
    fdrst_path = _find_fdrst(inp, out, input_text)

    # --- Check for other restart assets ---
    for ext in (".db", ".movecs"):
        asset = out.with_suffix(ext)
        if not asset.exists() and inp:
            asset = inp.with_suffix(ext)
        if not asset.exists():
            missing_files.append(f"{out.stem}{ext}")

    if fdrst_path is None and is_freq:
        missing_files.append(f"{out.stem}.fdrst (freq checkpoint)")

    # Helper to attach context to every result
    def _enrich(result: dict[str, Any]) -> dict[str, Any]:
        result["related_jobs"] = related_jobs
        result["missing_files"] = missing_files
        if not input_file and inp and inp.exists():
            result["resolved_input_file"] = str(inp)
        if related_jobs:
            result["related_jobs_note"] = (
                "SLURM shows job(s) with a similar name. "
                "Confirm with the user whether any of these belong to this output "
                "before assuming the job is still running."
            )
        return result

    # --- 1. OOM? ---
    if re.search(r"MA_ERR|insufficient\s+memory|failed to allocate", contents, re.IGNORECASE) or \
       re.search(r"MA_ERR|MemoryError|Killed", err_text, re.IGNORECASE):
        return _enrich(_wf_result(
            "oom", 0,
            "Out of memory — reduce memory directive or mpi_ranks.",
            {"tool": "create_nwchem_input_variant",
             "params": {"source_input": input_file or "",
                        "changes": {"memory": "800 mb"},
                        "reason": "OOM failure — reducing memory"}} if input_file else None,
            0.85,
        ))

    # --- 2. Timelimit? ---
    if "DUE TO TIME LIMIT" in err_text or ("CANCELLED" in err_text and "TIME" in err_text):
        if is_freq and fdrst_path:
            try:
                progress = _parse_freq_progress(output_file, contents)
                pct = progress.get("pct_complete", 0) or 0
            except Exception:
                pct = 0
            return _enrich(_wf_result(
                "freq_timelimited", pct,
                f"Freq hit walltime at {pct:.0f}% complete. fdrst checkpoint valid — resubmit to continue.",
                {"tool": "launch_nwchem_run",
                 "params": {"input_file": input_file or "", "profile": profile,
                            "resource_overrides": {"walltime": "48:00:00"}}} if input_file else None,
                0.95,
            ))
        return _enrich(_wf_result(
            "cancelled", 0,
            "Job cancelled due to time limit.",
            {"tool": "launch_nwchem_run",
             "params": {"input_file": input_file or "", "profile": profile,
                        "resource_overrides": {"walltime": "48:00:00"}}} if input_file else None,
            0.70,
        ))

    # --- 3. SCF failed? ---
    if re.search(r"(convergence|scf)\s+(has\s+)?not\s+been?\s+(achieved|reached|converged)", contents, re.IGNORECASE):
        return _enrich(_wf_result(
            "scf_failed", 0,
            "SCF did not converge.",
            {"tool": "suggest_nwchem_recovery",
             "params": {"output_file": output_file, "input_file": input_file or "", "mode": "scf"}},
            0.90,
        ))

    # --- 4. Check if still running (no "Total times" line = incomplete) ---
    has_total_times = bool(re.search(r"Total\s+times\s+cpu:", contents, re.IGNORECASE))
    if not has_total_times:
        # Distinguish "incomplete and still running" from "incomplete and stopped"
        job_is_running = _is_job_running_for_output(out, related_jobs)

        if is_freq:
            try:
                progress = _parse_freq_progress(output_file, contents)
                pct = progress.get("pct_complete", 0) or 0
                n_done = progress.get("n_done_cumulative", 0)
                n_total = progress.get("n_total_displacements", 0)
            except Exception:
                pct, n_done, n_total = 0, 0, 0

            if job_is_running:
                return _enrich(_wf_result(
                    "running_freq", pct,
                    f"Frequency calculation in progress — {n_done}/{n_total} displacements ({pct:.0f}% done).",
                    {"tool": "watch_nwchem_run",
                     "params": {"output_file": output_file, "input_file": input_file or "",
                                "profile": profile}},
                    0.90,
                ))
            else:
                # Stopped mid-freq — needs restart
                if fdrst_path:
                    return _enrich(_wf_result(
                        "freq_timelimited", pct,
                        f"Freq stopped at {pct:.0f}% ({n_done}/{n_total} displacements). "
                        f"fdrst checkpoint exists — can restart.",
                        {"tool": "prepare_nwchem_freq_restart",
                         "params": {"input_file": input_file or "", "output_file": output_file,
                                    "profile": profile}} if input_file else None,
                        0.85,
                    ))
                else:
                    return _enrich(_wf_result(
                        "freq_timelimited", pct,
                        f"Freq stopped at {pct:.0f}% ({n_done}/{n_total} displacements). "
                        f"No .fdrst found — restart may repeat completed work. "
                        f"Check if .fdrst exists in the original job directory.",
                        {"tool": "analyze_nwchem_case",
                         "params": {"output_file": output_file, "input_file": input_file or ""}},
                        0.60,
                    ))
        elif is_opt:
            state = "running_opt" if job_is_running else "failed"
            summary = ("Geometry optimization in progress." if job_is_running
                       else "Optimization stopped before converging (walltime or error).")
            return _enrich(_wf_result(
                state, 0, summary,
                {"tool": "watch_nwchem_run" if job_is_running else "suggest_nwchem_recovery",
                 "params": {"output_file": output_file, "input_file": input_file or "",
                            "profile": profile} if job_is_running else
                           {"output_file": output_file, "input_file": input_file or "", "mode": "auto"}},
                0.85 if job_is_running else 0.70,
            ))
        else:
            state = "running_scf" if job_is_running else "failed"
            summary = ("SCF/DFT calculation in progress." if job_is_running
                       else "Calculation stopped before completing (walltime or error).")
            return _enrich(_wf_result(
                state, 0, summary,
                {"tool": "watch_nwchem_run" if job_is_running else "analyze_nwchem_case",
                 "params": {"output_file": output_file, "input_file": input_file or "",
                            "profile": profile} if job_is_running else
                           {"output_file": output_file, "input_file": input_file or ""}},
                0.80 if job_is_running else 0.65,
            ))

    # --- 5. Completed — determine what finished ---

    # 5a. Freq job: check for imaginary modes
    if re.search(r"P\.Frequency|Normal\s+Mode\s+Eigenvalue", contents, re.IGNORECASE):
        try:
            imag = _analyze_imag(output_file, contents)
            sig_count = imag.get("significant_imaginary_mode_count", 0)
        except Exception:
            sig_count = 0

        if sig_count > 0:
            return _enrich(_wf_result(
                "imaginary_modes", 100,
                f"Freq complete but {sig_count} significant imaginary mode(s) found.",
                {"tool": "draft_nwchem_imaginary_mode_inputs",
                 "params": {"output_file": output_file, "input_file": input_file or ""}},
                0.85,
            ))
        return _enrich(_wf_result(
            "freq_complete", 100,
            "Frequency calculation completed — no significant imaginary modes.",
            {"tool": "parse_nwchem_output",
             "params": {"output_file": output_file, "sections": ["freq", "tasks"]}},
            0.95,
        ))

    # 5b. Optimization: check convergence
    if re.search(r"Optimization\s+converged", contents, re.IGNORECASE):
        return _enrich(_wf_result(
            "opt_converged", 100,
            "Geometry optimization converged.",
            {"tool": "extract_nwchem_geometry",
             "params": {"output_file": output_file, "frame": "best"}},
            0.90,
        ))
    if is_opt:
        return _enrich(_wf_result(
            "opt_failed_convergence", 0,
            "Optimization did not converge.",
            {"tool": "suggest_nwchem_recovery",
             "params": {"output_file": output_file, "input_file": input_file or "", "mode": "auto"}},
            0.80,
        ))

    # 5c. General success
    return _enrich(_wf_result(
        "completed", 100,
        "Calculation completed.",
        {"tool": "analyze_nwchem_case",
         "params": {"output_file": output_file, "input_file": input_file or ""}},
        0.85,
    ))


def _extract_input_echo(contents: str) -> str:
    """Extract the NWChem input echo from the output file.

    NWChem echoes the full input between markers like::

        ============================== echo of input deck ==============================
        ...
        ================================================================================
    """
    m = re.search(
        r"={10,}\s*echo of input deck\s*={10,}\n(.*?)={10,}",
        contents, re.DOTALL | re.IGNORECASE,
    )
    return m.group(1) if m else ""


def _find_fdrst(
    inp: "Path | None",
    out: Path,
    input_text: str,
) -> "Path | None":
    """Search for .fdrst file in both input and output directories."""
    # Try stem from start/restart keyword
    stem_match = re.search(r"^\s*(?:start|restart)\s+(\S+)", input_text, re.MULTILINE | re.IGNORECASE)
    stems_to_try = [out.stem]
    if stem_match:
        stems_to_try.insert(0, stem_match.group(1))
    if inp and inp.stem not in stems_to_try:
        stems_to_try.append(inp.stem)

    dirs_to_check = [out.parent]
    if inp and inp.parent != out.parent:
        dirs_to_check.append(inp.parent)

    for d in dirs_to_check:
        for stem in stems_to_try:
            candidate = d / (stem + ".fdrst")
            if candidate.exists():
                return candidate
    return None


def _check_related_slurm_jobs(output_stem: str) -> list[dict[str, str]]:
    """Check squeue for running/pending jobs whose name matches the output stem.

    Returns a list of dicts with job_id, name, partition, state, time.
    Never assumes a match — the model must ask the user to confirm.
    """
    import shutil
    import subprocess

    if not shutil.which("squeue"):
        return []

    try:
        proc = subprocess.run(
            ["squeue", "-u", str(subprocess.check_output(["whoami"]).decode().strip()),
             "-h", "-o", "%i %j %P %T %M"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return []
    except Exception:
        return []

    # Normalize the output stem for fuzzy matching
    # e.g. "am2pba3h2_s_reopt_v2_freq" should match job name "am2pba3h"
    stem_lower = output_stem.lower()
    # Extract a short prefix for matching (first 8 chars or first token)
    stem_prefix = re.split(r"[_\-.]", stem_lower)[0] if stem_lower else ""

    related = []
    for line in proc.stdout.strip().splitlines():
        parts = line.split(None, 4)
        if len(parts) < 4:
            continue
        job_id, name, partition, state = parts[0], parts[1], parts[2], parts[3]
        time_str = parts[4] if len(parts) > 4 else ""

        name_lower = name.lower()
        # Match if: stem starts with job name, job name starts with stem,
        # or they share a significant prefix (>=6 chars)
        is_related = (
            stem_lower.startswith(name_lower)
            or name_lower.startswith(stem_lower)
            or (stem_prefix and len(stem_prefix) >= 6 and name_lower.startswith(stem_prefix))
            or (len(name_lower) >= 6 and stem_lower.startswith(name_lower[:8]))
        )
        if is_related:
            related.append({
                "job_id": job_id,
                "name": name,
                "partition": partition,
                "state": state,
                "time": time_str,
            })

    return related


def _is_job_running_for_output(out: Path, related_jobs: list[dict[str, str]]) -> bool:
    """Determine if a SLURM job is currently running that writes to this output.

    Checks: (1) .jobid file exists and the job is in squeue, or
    (2) the output file was modified very recently (< 5 min ago).
    """
    import time

    # Check .jobid
    jobid_file = out.with_suffix(".jobid")
    if jobid_file.exists():
        try:
            jid = jobid_file.read_text().strip()
            if any(j["job_id"] == jid and j["state"] == "RUNNING" for j in related_jobs):
                return True
        except Exception:
            pass

    # Check if output was modified very recently
    try:
        mtime = out.stat().st_mtime
        if (time.time() - mtime) < 300:  # 5 minutes
            return True
    except Exception:
        pass

    return False


def _wf_result(
    state: str,
    progress_pct: float,
    summary: str,
    next_action: dict[str, Any] | None,
    confidence: float,
) -> dict[str, Any]:
    """Build a workflow state result dict."""
    result: dict[str, Any] = {
        "state": state,
        "progress_pct": round(progress_pct, 1),
        "human_summary": summary,
        "confidence": confidence,
    }
    if next_action is not None:
        result["next_action"] = next_action
    return result


# ---------------------------------------------------------------------------
# HPC resource advisor (profile-aware, multi-node, task-type-aware)
# ---------------------------------------------------------------------------


def _parse_walltime_hours(wt: str | None) -> float | None:
    """Parse HH:MM:SS walltime string to hours."""
    if not wt:
        return None
    parts = wt.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) + int(parts[1]) / 60 + int(parts[2]) / 3600
        if len(parts) == 2:
            return int(parts[0]) + int(parts[1]) / 60
        return float(parts[0])
    except (ValueError, IndexError):
        return None


def _format_walltime(hours: float) -> str:
    """Format hours as HH:MM:SS walltime string."""
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h}:{m:02d}:00"


def detect_hpc_accounts(
    profile: str,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    """Detect available HPC allocation accounts for a runner profile.

    Runs the profile's ``account_command`` (e.g. ``/usr/local/etc/taccinfo``)
    and parses the output to find project names and available SUs.

    Returns a dict with ``accounts`` list (each with name, avail_sus, expires)
    and ``recommended`` (the account with the most SUs remaining).
    """
    import subprocess
    from .runner import load_runner_profiles, _resolve_profile

    loaded = load_runner_profiles(profiles_path)
    profile_payload = _resolve_profile(loaded, profile)
    res = profile_payload.get("resources", {})
    account_cmd = res.get("account_command")

    if not account_cmd:
        static_account = res.get("account")
        if static_account:
            return {
                "accounts": [{"name": static_account, "avail_sus": None, "expires": None}],
                "recommended": static_account,
                "source": "profile_static",
            }
        return {
            "accounts": [],
            "recommended": None,
            "source": "none",
            "message": "No account_command or static account in profile. "
                       "Set resources.account or resources.account_command.",
        }

    try:
        result = subprocess.run(
            account_cmd, shell=True, capture_output=True, text=True, timeout=15,
        )
        output = result.stdout
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {
            "accounts": [],
            "recommended": res.get("account"),
            "source": "error",
            "message": f"Failed to run account_command: {exc}",
        }

    # Parse taccinfo-style output:
    #   | Name           Avail SUs     Expires |
    #   | TG-CHE250093         818  2026-06-08 |
    # Also handle generic formats: "account_name  SUs  date"
    import re
    accounts: list[dict[str, Any]] = []
    for line in output.splitlines():
        # Strip table borders
        stripped = line.strip().strip("|").strip()
        if not stripped or stripped.startswith("-") or "Name" in stripped:
            continue
        # Try to match: project_name  number  date
        m = re.match(
            r"(\S+)\s+(\d+)\s+(\d{4}-\d{2}-\d{2})", stripped
        )
        if m:
            accounts.append({
                "name": m.group(1),
                "avail_sus": int(m.group(2)),
                "expires": m.group(3),
            })

    # Pick the account with the most SUs as recommended
    recommended = None
    if accounts:
        best = max(accounts, key=lambda a: a["avail_sus"] or 0)
        recommended = best["name"]

    return {
        "accounts": accounts,
        "recommended": recommended,
        "source": "account_command",
        "command": account_cmd,
        "raw_output": output.strip(),
    }


def suggest_hpc_resources(
    input_file: str,
    profile: str,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    """Recommend optimal HPC resources for a NWChem job based on profile hardware specs.

    Analyzes the input file (atoms, basis, method, task type) and the profile's
    hardware description (cores_per_node, node_memory_mb, max_nodes, max_walltime,
    cpu_arch) to recommend:
    - nodes, mpi_ranks, walltime
    - NWChem memory directive
    - partition (if profile specifies one)
    - account (auto-detected from account_command if available)

    Returns a dict with recommended ``resource_overrides`` ready to pass to
    ``launch_nwchem_run``, plus rationale explaining the choices.

    Args:
        input_file: Path to the NWChem .nw input file.
        profile: Runner profile name (must have hardware fields populated).
        profiles_path: Optional path to profiles YAML/JSON.
    """
    from .runner import load_runner_profiles, _resolve_profile

    # --- Load profile hardware specs ---
    loaded = load_runner_profiles(profiles_path)
    profile_payload = _resolve_profile(loaded, profile)
    res = profile_payload.get("resources", {})

    cores_per_node = res.get("cores_per_node") or res.get("mpi_ranks") or 48
    node_memory_mb = res.get("node_memory_mb")
    max_nodes = res.get("max_nodes") or 1
    max_wt_str = res.get("max_walltime")
    max_wt_hours = _parse_walltime_hours(max_wt_str) or 48.0
    cpu_arch = res.get("cpu_arch") or "generic"
    partition = res.get("partition")

    # --- Analyze input file ---
    job = _analyze_job_size(input_file)
    n_atoms = job["n_atoms"]
    n_heavy = job["n_heavy"]
    n_bf = job["n_bf"]
    module = job["module"]
    is_freq = job["is_freq"]
    is_opt = job["is_opt"]
    is_tce = job["is_tce"]
    basis_name = job["basis_name"]

    rationale: list[str] = []
    warnings: list[str] = []

    # --- Step 1: Determine optimal MPI ranks per node ---
    bf_per_rank = _BF_PER_RANK_TARGET.get(cpu_arch, 80)
    ranks_by_scaling = max(1, n_bf // bf_per_rank)

    # Memory constraint on ranks per node
    if node_memory_mb:
        # Reserve 15% for OS + MPI
        usable_mb = int(node_memory_mb * 0.85)
        # Need at least 400 MB/rank for NWChem to start
        max_ranks_by_memory = max(1, usable_mb // 400)
    else:
        max_ranks_by_memory = cores_per_node

    ranks_per_node = min(ranks_by_scaling, max_ranks_by_memory, cores_per_node)
    ranks_per_node = max(1, ranks_per_node)

    # For small molecules, don't use more ranks than useful
    if ranks_by_scaling < cores_per_node // 2:
        ranks_per_node = max(1, ranks_by_scaling)
        rationale.append(
            f"Small molecule ({n_atoms} atoms, ~{n_bf} BF): using {ranks_per_node} "
            f"ranks/node instead of full {cores_per_node} cores for efficiency"
        )
    else:
        ranks_per_node = min(cores_per_node, max_ranks_by_memory)
        rationale.append(
            f"{n_atoms} atoms, ~{n_bf} BF, {bf_per_rank} BF/rank target → "
            f"{ranks_per_node} ranks/node (of {cores_per_node} available)"
        )

    # --- Step 2: Determine number of nodes ---
    nodes = 1

    if is_freq:
        # Numerical freq: 6*N_atoms displacements, no checkpoint, need to finish in one go
        n_displacements = n_atoms * 6
        # Rough heuristic: seconds per displacement
        base_seconds = 300.0 * (n_atoms / 20.0) ** 1.5
        total_ranks_1node = ranks_per_node
        eff_1 = min(total_ranks_1node, 64) + max(0, total_ranks_1node - 64) * 0.3
        secs_per_disp_1 = base_seconds * (48.0 / max(1, eff_1))
        hours_1node = (n_displacements * secs_per_disp_1) / 3600.0

        if hours_1node > max_wt_hours * 0.85:
            # Need multi-node
            for n in range(2, max_nodes + 1):
                total_ranks_n = ranks_per_node * n
                eff_n = min(total_ranks_n, 64) + max(0, total_ranks_n - 64) * 0.3
                secs_n = base_seconds * (48.0 / max(1, eff_n))
                hours_n = (n_displacements * secs_n) / 3600.0
                if hours_n <= max_wt_hours * 0.85:
                    nodes = n
                    break
            else:
                nodes = min(max_nodes, 8)
                warnings.append(
                    f"Frequency job estimated at {hours_1node:.0f}h on 1 node. "
                    f"Even with {nodes} nodes it may exceed {max_wt_hours:.0f}h walltime. "
                    f"Consider analytical frequencies or a smaller basis."
                )
            rationale.append(
                f"Numerical freq: {n_displacements} displacements, estimated "
                f"{hours_1node:.0f}h on 1 node → {nodes} nodes to fit in "
                f"{max_wt_hours:.0f}h walltime"
            )
        else:
            rationale.append(
                f"Numerical freq: {n_displacements} displacements, estimated "
                f"{hours_1node:.1f}h — fits on 1 node"
            )
        warnings.append(
            "NWChem CANNOT checkpoint numerical frequencies. If the job "
            "exceeds walltime, ALL progress is lost."
        )

    elif is_tce:
        # TCE is memory-hungry — may need multi-node for memory
        mem_rec = suggest_memory(n_atoms=n_atoms, basis=basis_name or "6-31g*",
                                method="tce", n_heavy_atoms=n_heavy)
        total_mem_needed = mem_rec["recommended_total_mb"] * ranks_per_node
        if node_memory_mb and total_mem_needed > node_memory_mb * 0.80:
            nodes = min(max_nodes, max(2, math.ceil(
                total_mem_needed / (node_memory_mb * 0.80)
            )))
            rationale.append(
                f"TCE: estimated {total_mem_needed} MB total memory needed, "
                f"node has {node_memory_mb} MB → {nodes} nodes for memory"
            )
        else:
            rationale.append("TCE single-point: fits on 1 node")

    total_ranks = ranks_per_node * nodes

    # --- Step 3: Determine walltime ---
    if is_freq:
        n_displacements = n_atoms * 6
        base_seconds = 300.0 * (n_atoms / 20.0) ** 1.5
        eff = min(total_ranks, 64) + max(0, total_ranks - 64) * 0.3
        secs_per_disp = base_seconds * (48.0 / max(1, eff))
        est_hours = (n_displacements * secs_per_disp) / 3600.0
        # Add 20% safety margin
        walltime_hours = min(max_wt_hours, est_hours * 1.2)
        walltime_hours = max(2.0, walltime_hours)  # minimum 2h
        rationale.append(
            f"Freq walltime: {est_hours:.1f}h estimated + 20% margin → "
            f"{walltime_hours:.1f}h"
        )
    elif is_opt:
        # Optimization: moderate walltime, depends on molecule size
        if n_atoms <= 10:
            walltime_hours = min(max_wt_hours, 4.0)
        elif n_atoms <= 30:
            walltime_hours = min(max_wt_hours, 12.0)
        elif n_atoms <= 60:
            walltime_hours = min(max_wt_hours, 24.0)
        else:
            walltime_hours = min(max_wt_hours, 48.0)
        rationale.append(
            f"Optimization: {n_atoms} atoms → {walltime_hours:.0f}h walltime"
        )
    elif is_tce:
        # TCE single-points can be long
        if n_atoms <= 5:
            walltime_hours = min(max_wt_hours, 6.0)
        elif n_atoms <= 15:
            walltime_hours = min(max_wt_hours, 24.0)
        else:
            walltime_hours = min(max_wt_hours, 48.0)
        rationale.append(
            f"TCE single-point: {n_atoms} atoms → {walltime_hours:.0f}h walltime"
        )
    else:
        # Single-point energy: usually fast
        if n_atoms <= 5:
            walltime_hours = min(max_wt_hours, 1.0)
        elif n_atoms <= 20:
            walltime_hours = min(max_wt_hours, 4.0)
        elif n_atoms <= 50:
            walltime_hours = min(max_wt_hours, 8.0)
        else:
            walltime_hours = min(max_wt_hours, 24.0)
        rationale.append(
            f"Single-point energy: {n_atoms} atoms → {walltime_hours:.0f}h walltime"
        )

    walltime_str = _format_walltime(walltime_hours)

    # --- Step 4: Determine NWChem memory directive ---
    if node_memory_mb:
        usable_mb = int(node_memory_mb * 0.85)
        mem_per_rank = max(400, (usable_mb // ranks_per_node // 100) * 100)
    else:
        mem_rec = suggest_memory(
            n_atoms=n_atoms, basis=basis_name or "6-31g*",
            method=module, n_heavy_atoms=n_heavy,
        )
        mem_per_rank = mem_rec["recommended_total_mb"]

    mem_suggestion = suggest_memory(
        n_atoms=n_atoms, basis=basis_name or "6-31g*",
        method=module, n_heavy_atoms=n_heavy,
    )
    # Use the larger of: what the method needs, or what fits the node
    recommended_mem = max(mem_suggestion["recommended_total_mb"], 500)
    if node_memory_mb:
        usable_mb = int(node_memory_mb * 0.85)
        ceiling = max(400, (usable_mb // ranks_per_node // 100) * 100)
        if recommended_mem > ceiling:
            warnings.append(
                f"Recommended memory {recommended_mem} MB/rank exceeds safe "
                f"ceiling {ceiling} MB/rank for {ranks_per_node} ranks on "
                f"{node_memory_mb} MB node. Capping to {ceiling} MB."
            )
            recommended_mem = ceiling
    nwchem_mem = recommended_mem

    # --- Step 5: Detect account ---
    account = res.get("account")
    account_info: dict[str, Any] | None = None
    if not account and res.get("account_command"):
        acct_result = detect_hpc_accounts(profile, profiles_path)
        account_info = acct_result
        if acct_result.get("recommended"):
            account = acct_result["recommended"]
            # Find the recommended account's SU balance
            rec_sus = next(
                (a.get("avail_sus", "?") for a in acct_result["accounts"]
                 if a["name"] == account), "?"
            )
            rationale.append(
                f"Account auto-detected: {account} ({rec_sus} SUs available)"
            )

    # Build resource_overrides dict
    resource_overrides: dict[str, Any] = {
        "nodes": nodes,
        "mpi_ranks": total_ranks,
        "walltime": walltime_str,
    }
    if account:
        resource_overrides["account"] = account

    result: dict[str, Any] = {
        "profile": profile,
        "resource_overrides": resource_overrides,
        "recommended_memory_per_rank_mb": nwchem_mem,
        "nwchem_memory_directive": f"memory total {nwchem_mem} mb",
        "nodes": nodes,
        "ranks_per_node": ranks_per_node,
        "total_mpi_ranks": total_ranks,
        "walltime": walltime_str,
        "estimated_basis_functions": n_bf,
        "n_atoms": n_atoms,
        "n_heavy_atoms": n_heavy,
        "method": module,
        "task_type": operation,
        "partition": partition,
        "hardware": {
            "cores_per_node": cores_per_node,
            "node_memory_mb": node_memory_mb,
            "max_nodes": max_nodes,
            "max_walltime": max_wt_str,
            "cpu_arch": cpu_arch,
        },
        "rationale": rationale,
        "warnings": warnings,
    }
    if account:
        result["account"] = account
    if account_info:
        result["account_info"] = account_info
    return result


# ── Smart partition / queue selection ────────────────────────────────────


def suggest_partition(
    input_file: str,
    profiles_path: str | None = None,
    check_queue: bool = True,
) -> dict[str, Any]:
    """Suggest the best partition/queue for a job across all available profiles.

    Scans all scheduler-type profiles, evaluates job fit (memory, walltime),
    checks if dev queues are suitable for short jobs, and optionally queries
    ``sinfo`` for current queue availability.

    Args:
        input_file: Path to the NWChem .nw input file.
        profiles_path: Optional path to runner profiles YAML/JSON.
        check_queue: If True, run ``sinfo`` to check partition availability.

    Returns:
        Dict with ``recommended_profile``, ``recommended_partition``,
        ``resource_overrides``, comparison table, and rationale.
    """
    import subprocess
    from .runner import load_runner_profiles, _resolve_profile
    loaded = load_runner_profiles(profiles_path)
    all_profile_names = list((loaded.get("profiles") or {}).keys())

    # --- Analyze the input file once ---
    job = _analyze_job_size(input_file)
    n_atoms = job["n_atoms"]
    n_bf = job["n_bf"]
    is_freq = job["is_freq"]
    is_opt = job["is_opt"]
    is_tce = job["is_tce"]
    basis_name = job["basis_name"]

    # --- Estimate walltime needed ---
    if is_freq:
        n_disp = n_atoms * 6
        base_seconds = 300.0 * (n_atoms / 20.0) ** 1.5
        est_hours = (n_disp * base_seconds) / 3600.0 / max(1, n_bf // 80)
    elif is_opt:
        if n_atoms <= 10:
            est_hours = 0.5
        elif n_atoms <= 30:
            est_hours = 4.0
        else:
            est_hours = 12.0
    elif is_tce:
        if n_atoms <= 5:
            est_hours = 2.0
        elif n_atoms <= 15:
            est_hours = 12.0
        else:
            est_hours = 24.0
    else:
        if n_atoms <= 5:
            est_hours = 0.25
        elif n_atoms <= 20:
            est_hours = 1.0
        else:
            est_hours = 4.0

    # Estimate memory per rank
    mem_rec = suggest_memory(
        n_atoms=n_atoms, basis=basis_name or "6-31g*",
        method=module, n_heavy_atoms=n_heavy,
    )
    mem_per_rank_needed = mem_rec["recommended_total_mb"]

    # --- Get queue status if requested ---
    queue_info: dict[str, dict[str, Any]] = {}
    if check_queue:
        try:
            proc = subprocess.run(
                ["sinfo", "-o", "%P %a %F %l", "--noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0:
                for line in proc.stdout.strip().splitlines():
                    parts = line.split()
                    if len(parts) >= 4:
                        pname = parts[0].rstrip("*")
                        avail = parts[1]
                        # Node counts: allocated/idle/other/total
                        node_counts = parts[2]
                        timelimit = parts[3]
                        idle = 0
                        total = 0
                        try:
                            nc = node_counts.split("/")
                            idle = int(nc[1]) if len(nc) > 1 else 0
                            total = int(nc[3]) if len(nc) > 3 else 0
                        except (ValueError, IndexError):
                            pass
                        queue_info[pname] = {
                            "available": avail == "up",
                            "idle_nodes": idle,
                            "total_nodes": total,
                            "timelimit": timelimit,
                        }
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

    # --- Evaluate each profile ---
    candidates: list[dict[str, Any]] = []
    for prof_name in all_profile_names:
        try:
            prof = _resolve_profile(loaded, prof_name)
        except ValueError:
            continue

        launcher = prof.get("launcher", {})
        if launcher.get("kind") != "scheduler":
            continue

        res = prof.get("resources", {})
        cores_per_node = res.get("cores_per_node") or res.get("mpi_ranks") or 48
        node_memory_mb = res.get("node_memory_mb")
        max_wt_str = res.get("max_walltime")
        max_wt_hours = _parse_walltime_hours(max_wt_str) or 48.0
        partition = res.get("partition", "")
        cpu_arch = res.get("cpu_arch") or "generic"
        max_nodes = res.get("max_nodes") or 1

        # Memory per core
        mem_per_core = (node_memory_mb / cores_per_node) if node_memory_mb else 4000

        # Does the job fit in walltime?
        fits_walltime = est_hours <= max_wt_hours * 0.90

        # Does the job fit in memory?
        fits_memory = mem_per_rank_needed <= mem_per_core * 0.85 if node_memory_mb else True

        # Is this a dev queue?
        is_dev = "dev" in prof_name.lower() or "dev" in partition.lower()

        # SU rate (default 1.0 if not specified)
        # Infer from common TACC naming
        su_rate = 1.0
        if "spr" in prof_name.lower():
            su_rate = 2.0
        elif "icx" in prof_name.lower():
            su_rate = 1.5

        # Queue status
        q_status = queue_info.get(partition, {})
        queue_available = q_status.get("available", True)  # assume available if no sinfo
        idle_nodes = q_status.get("idle_nodes", 0)

        # --- Scoring ---
        score = 0.0

        if not fits_walltime:
            score -= 100  # Disqualify
        if not fits_memory:
            score -= 50

        if not queue_available:
            score -= 200

        # Dev queue bonus for short jobs
        if is_dev and est_hours <= max_wt_hours * 0.75:
            score += 30  # Strong preference for dev queue when job fits

        # Cheaper SU rate is better
        score -= su_rate * 5

        # Memory headroom is good
        if node_memory_mb:
            headroom = mem_per_core / max(mem_per_rank_needed, 100)
            score += min(10, headroom * 2)

        # Idle nodes bonus
        if idle_nodes > 0:
            score += min(10, idle_nodes)

        candidates.append({
            "profile": prof_name,
            "partition": partition,
            "cores_per_node": cores_per_node,
            "node_memory_mb": node_memory_mb,
            "mem_per_core_mb": round(mem_per_core),
            "max_walltime": max_wt_str or "unknown",
            "max_walltime_hours": max_wt_hours,
            "cpu_arch": cpu_arch,
            "su_rate": su_rate,
            "is_dev": is_dev,
            "fits_walltime": fits_walltime,
            "fits_memory": fits_memory,
            "queue_available": queue_available,
            "idle_nodes": idle_nodes,
            "score": round(score, 1),
        })

    # Sort by score descending
    candidates.sort(key=lambda c: -c["score"])

    rationale: list[str] = []
    rationale.append(
        f"Job: {n_atoms} atoms, ~{n_bf} BF, {module}/{operation}, "
        f"estimated ~{est_hours:.1f}h, ~{mem_per_rank_needed} MB/rank"
    )

    recommended = candidates[0] if candidates else None
    if recommended:
        rec_name = recommended["profile"]
        rec_part = recommended["partition"]

        if recommended["is_dev"]:
            rationale.append(
                f"Recommended dev queue '{rec_part}' — job estimated at "
                f"{est_hours:.1f}h fits within {recommended['max_walltime']} "
                f"max walltime, faster queue turnaround"
            )
        else:
            rationale.append(
                f"Recommended '{rec_part}' — best fit for memory "
                f"({recommended['mem_per_core_mb']} MB/core) and cost "
                f"({recommended['su_rate']}x SU rate)"
            )

        if queue_info:
            idle = recommended.get("idle_nodes", 0)
            if idle > 0:
                rationale.append(f"Queue status: {idle} idle nodes on {rec_part}")
            else:
                rationale.append(f"Queue status: no idle nodes on {rec_part} (job will queue)")

        # Run full resource suggestion for the recommended profile
        full_suggestion = suggest_hpc_resources(input_file, rec_name, profiles_path)

        return {
            "recommended_profile": rec_name,
            "recommended_partition": rec_part,
            "resource_overrides": full_suggestion.get("resource_overrides", {}),
            "nwchem_memory_directive": full_suggestion.get("nwchem_memory_directive", ""),
            "estimated_walltime_hours": round(est_hours, 2),
            "job_summary": {
                "n_atoms": n_atoms,
                "n_heavy_atoms": n_heavy,
                "estimated_basis_functions": n_bf,
                "method": module,
                "task_type": operation,
                "mem_per_rank_needed_mb": mem_per_rank_needed,
            },
            "partition_comparison": candidates,
            "queue_status_available": bool(queue_info),
            "rationale": rationale + full_suggestion.get("rationale", []),
            "warnings": full_suggestion.get("warnings", []),
        }

    return {
        "recommended_profile": None,
        "recommended_partition": None,
        "error": "No suitable scheduler profiles found",
        "profiles_scanned": all_profile_names,
        "rationale": rationale,
    }

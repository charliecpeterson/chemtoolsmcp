"""TCE / MCSCF setup orchestrators.

These are the "thick tool" implementations described in REFACTOR.md item
12 — they collapse the multi-step setup that an LLM would otherwise have
to drive (parse MOs, decide freeze count, check ordering, suggest swaps,
draft the input) into a single deterministic Python call.

`prepare_nwchem_tce_setup` is the canonical example: given an SCF output
and a target post-HF method, it returns the freeze count, an ordering
verdict, a vectors-swap list (when applicable), and a ready Diagnosis
envelope with next_actions for the agent.

The functions live here rather than inside the Strategist plugin class
because they orchestrate across the parser AND drafter sub-protocols —
they are program-specific composite tools rather than primitive
operations on a single sub-protocol.
"""

from __future__ import annotations
from typing import Any

from chemtools.core.common import read_text, make_metadata
from chemtools.core.types import Diagnosis, NextAction, Verdict
from chemtools.programs.nwchem.parse.mos import parse_mos
from chemtools.programs.nwchem.parse.tce import (
    suggest_tce_freeze_count,
    analyze_tce_orbital_ordering,
)
from chemtools.programs.nwchem.strategy.diagnose import suggest_vectors_swaps


def _elements_from_mos(mos_payload: dict[str, Any]) -> list[str]:
    """Best-effort element list inferred from orbital atom contributions."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for orb in mos_payload.get("orbitals") or []:
        for contrib in orb.get("top_atom_contributions") or []:
            elem = contrib.get("element")
            if elem and elem not in seen_set:
                seen_set.add(elem)
                seen.append(elem)
    return seen


def prepare_nwchem_tce_setup(
    scf_output_path: str,
    target_method: str = "ccsd(t)",
    *,
    elements: list[str] | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    ecp_core_electrons: dict[str, int] | None = None,
) -> dict[str, Any]:
    """One-call TCE prep: freeze count + ordering check + swap recs + Diagnosis.

    Inputs:
      scf_output_path        — path to the converged SCF (or DFT) output file
      target_method          — TCE method tag ("ccsd", "ccsd(t)", "mp2", ...)
      elements               — explicit element list; inferred from MO data if omitted
      charge, multiplicity   — molecular charge and spin multiplicity (M = 2S+1)
      expected_metal_elements — for open-shell ordering checks
      expected_somo_count    — for open-shell ordering checks
      ecp_core_electrons     — per-element ECP core counts (e.g. {"Au": 60})

    Returns a single dict containing:
      mos                  — parsed MO payload (cheap version: top_n=10)
      freeze_count         — recommended freeze N for `freeze N` in the tce block
      freeze_rationale     — text explaining the freeze count
      ordering_analysis    — analyze_tce_orbital_ordering output
      vectors_swap_suggestion — suggest_vectors_swaps output (when SOMOs present)
      diagnosis            — Diagnosis envelope with verdict + next_actions
      target_method        — echoed back
      metadata             — file metadata
    """
    contents = read_text(scf_output_path)
    mos = parse_mos(scf_output_path, contents, top_n=10)

    if elements is None:
        elements = _elements_from_mos(mos)

    freeze = suggest_tce_freeze_count(
        elements,
        ecp_core_electrons=ecp_core_electrons,
        charge=charge,
        multiplicity=multiplicity,
    )
    freeze_count: int = freeze.get("freeze_count", 0)
    freeze_rationale: str = freeze.get("rationale", "")

    ordering = analyze_tce_orbital_ordering(
        orbitals=mos.get("orbitals") or [],
        freeze_count=freeze_count,
    )

    # Vectors-swap recs only meaningful when there are SOMOs or an explicit
    # expectation. analyze_frontier_orbitals + suggest_vectors_swaps drive
    # this together.
    swap_suggestion: dict[str, Any] | None = None
    if expected_somo_count is not None or expected_metal_elements:
        swap_suggestion = suggest_vectors_swaps(
            mos,
            expected_metal_elements=expected_metal_elements or [],
            expected_somo_count=expected_somo_count,
        )

    # ---- Assemble Diagnosis envelope ----
    issues: list[str] = []
    if ordering.get("warnings"):
        issues.extend(ordering["warnings"])
    if swap_suggestion and swap_suggestion.get("swaps_recommended"):
        issues.append(
            f"{len(swap_suggestion.get('swap_pairs') or [])} orbital swap(s) recommended before TCE"
        )

    if not issues:
        verdict_label = "ready_to_run_tce"
        confidence = 0.85
    elif swap_suggestion and swap_suggestion.get("swaps_recommended"):
        verdict_label = "vectors_swap_required"
        confidence = 0.8
    else:
        verdict_label = "ordering_warning_review_before_tce"
        confidence = 0.55

    verdict: Verdict = {
        "label": verdict_label,
        "confidence": confidence,
        "reasons": issues,
    }

    next_actions: list[NextAction] = []
    if swap_suggestion and swap_suggestion.get("swaps_recommended"):
        next_actions.append({
            "tool": "swap_nwchem_movecs",
            "params": {
                "swap_pairs": swap_suggestion.get("swap_pairs") or [],
            },
            "reason": "Apply the recommended SOMO swaps before launching TCE.",
            "confidence": 0.75,
            "priority": 1,
        })
        next_actions.append({
            "tool": "draft_nwchem_tce_input",
            "params": {
                "scf_output_path": scf_output_path,
                "method": target_method,
                "freeze_count": freeze_count,
            },
            "reason": "Draft the TCE input using the post-swap movecs and the computed freeze count.",
            "confidence": 0.8,
            "priority": 2,
        })
    elif issues:
        # Ordering issue without a swap recommendation — needs human/agent review
        # of the actual orbital characters before drafting.
        next_actions.append({
            "tool": "parse_nwchem_mos",
            "params": {
                "output_file": scf_output_path,
                "top_n": 12,
                "include_coefficients": False,
            },
            "reason": (
                "Inspect the dominant character of frozen vs correlated orbitals "
                "to confirm the ordering anomaly before drafting TCE input."
            ),
            "confidence": 0.7,
            "priority": 1,
        })
        next_actions.append({
            "tool": "draft_nwchem_tce_input",
            "params": {
                "scf_output_path": scf_output_path,
                "method": target_method,
                "freeze_count": freeze_count,
            },
            "reason": (
                "After confirming orbital character, draft the TCE input with the "
                "computed freeze count (or override if the anomaly is benign)."
            ),
            "confidence": 0.55,
            "priority": 2,
        })
    else:
        # Clean — proceed straight to drafting.
        next_actions.append({
            "tool": "draft_nwchem_tce_input",
            "params": {
                "scf_output_path": scf_output_path,
                "method": target_method,
                "freeze_count": freeze_count,
            },
            "reason": "Ordering looks correct. Draft the TCE input with the computed freeze count.",
            "confidence": 0.85,
            "priority": 1,
        })

    diagnosis: Diagnosis = {
        "verdict": verdict,
        "next_actions": next_actions,
        "anchors": [],
    }

    return {
        "metadata": make_metadata(scf_output_path, contents, "nwchem"),
        "target_method": target_method,
        "elements": elements,
        "charge": charge,
        "multiplicity": multiplicity,
        "mos": {
            "orbital_count": mos.get("orbital_count"),
            "occupied_count": mos.get("occupied_count"),
            "virtual_count": mos.get("virtual_count"),
            "somo_count": mos.get("somo_count"),
            "homo": mos.get("homo"),
            "lumo": mos.get("lumo"),
            "homo_lumo_gap_hartree": mos.get("homo_lumo_gap_hartree"),
        },
        "freeze_count": freeze_count,
        "freeze_rationale": freeze_rationale,
        "ordering_analysis": ordering,
        "vectors_swap_suggestion": swap_suggestion,
        "diagnosis": diagnosis,
    }


def prepare_nwchem_mcscf_setup(
    scf_output_path: str,
    *,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    prefer_expanded: bool = False,
) -> dict[str, Any]:
    """One-call MCSCF/CASSCF setup: active space recommendation + Diagnosis.

    Companion to prepare_nwchem_tce_setup but for the multireference branch.
    Calls suggest_nwchem_mcscf_active_space, then wraps its richer recommendation
    payload with a Diagnosis envelope so the agent gets a clear verdict and a
    routed next_action.

    Inputs:
      scf_output_path        — path to the converged SCF (or DFT) reference output
      input_path             — optional path to the SCF input file (improves
                                expected_somo_count inference when multiplicity > 1)
      expected_metal_elements — for open-shell metal complexes
      expected_somo_count    — overrides multiplicity-based inference
      prefer_expanded        — when True, route the agent toward the expanded
                                CAS window (typically CAS(2N,2N+lone_pairs))
                                instead of the minimal one. Default minimal.

    Returns:
      mcscf_recommendation  — passthrough of suggest_nwchem_mcscf_active_space
      recommended_active_space — picked window (minimal or expanded)
      frontier_assessment   — state-check verdict from the underlying analyzer
      diagnosis             — Diagnosis envelope routing the agent to draft_input,
                              parse_mos for deeper inspection, or vectors_swap_input
    """
    # Lazy import — api_strategy is still flat; will clean up after Phase 14.
    from chemtools.api_strategy import suggest_nwchem_mcscf_active_space

    rec = suggest_nwchem_mcscf_active_space(
        output_path=scf_output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
    )

    minimal = rec.get("minimal_active_space") or {}
    expanded = rec.get("expanded_active_space") if isinstance(rec.get("expanded_active_space"), dict) else None
    # frontier_assessment may be a string label (current shape from
    # suggest_nwchem_mcscf_active_space) or a dict (future shape if expanded).
    frontier_raw = rec.get("frontier_assessment")
    if isinstance(frontier_raw, dict):
        assessment = frontier_raw.get("assessment") or ""
        frontier_payload: Any = frontier_raw
    else:
        assessment = frontier_raw or ""
        frontier_payload = {"assessment": assessment}
    notes = rec.get("notes") or []
    swap_in = rec.get("swap_in_candidates") or []
    swap_out = rec.get("swap_out_candidates") or []

    chosen = (expanded or minimal) if prefer_expanded else minimal

    # ---- Verdict logic ----
    issues: list[str] = []
    if assessment == "metal_state_mismatch_suspected":
        issues.append(
            "Frontier orbitals do not match the expected metal-centered open-shell pattern; "
            "the SCF guess may need correction (vectors swap or different starting orbitals)."
        )
    if not chosen.get("active_electrons") or not chosen.get("active_orbitals"):
        issues.append("Active space recommendation has empty electron/orbital counts.")
    if swap_in or swap_out:
        issues.append(
            f"{len(swap_in)} swap_in / {len(swap_out)} swap_out candidates flagged for the active window."
        )

    if not issues and chosen.get("active_orbitals"):
        verdict_label = "ready_to_draft_mcscf"
        confidence = 0.8
    elif assessment == "metal_state_mismatch_suspected":
        verdict_label = "state_mismatch_review_frontier_first"
        confidence = 0.55
    else:
        verdict_label = "active_space_review_recommended"
        confidence = 0.6

    verdict: Verdict = {
        "label": verdict_label,
        "confidence": confidence,
        "reasons": issues,
    }

    # ---- Next actions ----
    next_actions: list[NextAction] = []
    if verdict_label == "state_mismatch_review_frontier_first":
        next_actions.append({
            "tool": "analyze_nwchem_frontier_orbitals",
            "params": {
                "output_path": scf_output_path,
                "input_path": input_path,
                "expected_metal_elements": expected_metal_elements or [],
                "expected_somo_count": expected_somo_count,
            },
            "reason": (
                "Verify the frontier orbital characters and confirm the metal-centered "
                "open-shell expectation before committing to an active space."
            ),
            "confidence": 0.75,
            "priority": 1,
        })
        next_actions.append({
            "tool": "draft_nwchem_vectors_swap_input",
            "params": {
                "output_path": scf_output_path,
                "input_path": input_path,
            },
            "reason": (
                "If the state mismatch is confirmed, a vectors swap on the SCF guess is the "
                "usual fix before attempting MCSCF."
            ),
            "confidence": 0.6,
            "priority": 2,
        })
    elif verdict_label == "active_space_review_recommended":
        next_actions.append({
            "tool": "parse_nwchem_mos",
            "params": {
                "output_file": scf_output_path,
                "top_n": 15,
            },
            "reason": (
                "Inspect a wider window of frontier orbitals to disambiguate the active space "
                "before drafting MCSCF."
            ),
            "confidence": 0.6,
            "priority": 1,
        })
        if chosen.get("active_electrons") and chosen.get("active_orbitals"):
            next_actions.append({
                "tool": "draft_nwchem_mcscf_input",
                "params": {
                    "reference_output_path": scf_output_path,
                    "active_electrons": chosen["active_electrons"],
                    "active_orbitals": chosen["active_orbitals"],
                },
                "reason": (
                    f"Tentative CAS({chosen.get('active_electrons')}, {chosen.get('active_orbitals')}) "
                    "based on the orbital window. Review first."
                ),
                "confidence": 0.5,
                "priority": 2,
            })
    else:
        # ready_to_draft_mcscf
        next_actions.append({
            "tool": "draft_nwchem_mcscf_input",
            "params": {
                "reference_output_path": scf_output_path,
                "active_electrons": chosen["active_electrons"],
                "active_orbitals": chosen["active_orbitals"],
            },
            "reason": (
                f"Draft MCSCF with CAS({chosen.get('active_electrons')}, "
                f"{chosen.get('active_orbitals')}) — orbital characters look clean."
            ),
            "confidence": 0.8,
            "priority": 1,
        })

    diagnosis: Diagnosis = {
        "verdict": verdict,
        "next_actions": next_actions,
        "anchors": [],
    }

    return {
        "metadata": rec.get("metadata"),
        "expected_metal_elements": rec.get("expected_metal_elements"),
        "expected_somo_count": rec.get("expected_somo_count"),
        "frontier_assessment": frontier_payload,
        "recommended_active_space": chosen,
        "alternative_active_space": expanded if not prefer_expanded else minimal,
        "swap_in_candidates": swap_in,
        "swap_out_candidates": swap_out,
        "notes": notes,
        "diagnosis": diagnosis,
    }


__all__ = ["prepare_nwchem_tce_setup", "prepare_nwchem_mcscf_setup"]

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


__all__ = ["prepare_nwchem_tce_setup"]

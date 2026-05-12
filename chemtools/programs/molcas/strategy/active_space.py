"""Active-space and CASPT2-setup advisory functions, plus character-aware
orbital-swap suggestions for the active-space-tuning workflow.

These tools take the structured output from the RASSCF / CASPT2 parsers and
INPORB reader and emit deterministic, actionable recommendations that small
LLMs can follow without re-deriving the rules.

Public API:

  analyze_active_space(rasscf_payload | inporb_payload) -> {
      "signature": "CAS(8,7) — quintet, sym 1",
      "per_root_quality": [...],
      "verdict": "healthy" / "marginal" / "poor",
      "recommendations": [{"action": "promote_to_inactive", "orbitals": [...], "rationale": ...}, ...],
      "next_actions": [...],     # MCP-style next-action envelope
  }

  validate_caspt2_setup(caspt2_payload) -> {
      "verdict": "healthy" / "caution" / "unreliable",
      "checks": [...],
      "next_actions": [...],
  }
"""

from __future__ import annotations

from typing import Any


# Occupation-number thresholds (literature-cited / common practice)
PROMOTE_THRESHOLD = 1.98   # occ >= this → promote to inactive (no longer "active")
EDGE_DOUBLY_LOW = 1.90     # 1.90 <= occ < 1.98 → "edge_doubly_occupied" (suspect)
EDGE_EMPTY_HIGH = 0.10     # 0.02 < occ < 0.10 → "edge_empty" (suspect)
DEMOTE_THRESHOLD = 0.02    # occ <= this → demote to secondary

# CASPT2 reference-weight quality bands
REFW_HEALTHY = 0.85
REFW_CAUTION = 0.70


def analyze_active_space(payload: dict[str, Any]) -> dict[str, Any]:
    """Accepts either a RASSCF parser payload or an INPORB parser payload.

    Returns a deterministic verdict + recommendations.
    """
    no_per_root = _extract_no_per_root(payload)
    signature = _extract_signature(payload)

    per_root_quality: list[dict[str, Any]] = []
    for root_data in no_per_root:
        classification = _classify_orbitals(root_data["all_occupations"])
        per_root_quality.append(
            {
                "root": root_data["root"],
                "n_active": len(root_data["all_occupations"]),
                "n_truly_active": classification["counts"]["truly_active"],
                "n_promote_candidates": (
                    classification["counts"]["near_doubly_occupied"]
                    + classification["counts"]["edge_doubly_occupied"]
                ),
                "n_demote_candidates": (
                    classification["counts"]["near_virtual"]
                    + classification["counts"]["edge_empty"]
                ),
                "summary": classification["summary"],
                "promote_orbitals": classification["promote_orbitals"],
                "demote_orbitals": classification["demote_orbitals"],
            }
        )

    verdict = _verdict_from_quality(per_root_quality)
    recommendations = _build_recommendations(per_root_quality, payload)
    next_actions = _next_actions_from_verdict(verdict, recommendations)

    return {
        "signature": signature,
        "per_root_quality": per_root_quality,
        "verdict": verdict,
        "recommendations": recommendations,
        "next_actions": next_actions,
    }


def _extract_no_per_root(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Coerce RASSCF parser output OR INPORB parser output into a uniform shape:
    [{"root": int, "all_occupations": [floats]}, ...]
    """
    # RASSCF parser shape:
    if "natural_occupations_per_root" in payload:
        return [
            {"root": r["root"], "all_occupations": r["all_occupations"]}
            for r in payload["natural_occupations_per_root"]
        ]
    # INPORB parser shape: state-averaged occupations only (one virtual "root")
    if "active_space_partition" in payload:
        active = payload["active_space_partition"].get("active_orbital_occupations") or []
        if not active:
            return []
        return [
            {
                "root": 1,
                "all_occupations": sorted(
                    [a["occupation"] for a in active], reverse=True
                ),
            }
        ]
    return []


def _extract_signature(payload: dict[str, Any]) -> str | None:
    if "active_space_signature" in payload:
        return payload["active_space_signature"]
    if "active_space_partition" in payload:
        return payload["active_space_partition"].get("signature")
    return None


def _classify_orbitals(occupations: list[float]) -> dict[str, Any]:
    counts = {
        "near_doubly_occupied": 0,
        "edge_doubly_occupied": 0,
        "truly_active": 0,
        "edge_empty": 0,
        "near_virtual": 0,
    }
    promote: list[dict[str, Any]] = []
    demote: list[dict[str, Any]] = []
    for orb_idx, occ in enumerate(occupations, start=1):
        if occ >= PROMOTE_THRESHOLD:
            counts["near_doubly_occupied"] += 1
            promote.append({"orbital_in_active_set": orb_idx, "occupation": round(occ, 5)})
        elif occ >= EDGE_DOUBLY_LOW:
            counts["edge_doubly_occupied"] += 1
            promote.append({"orbital_in_active_set": orb_idx, "occupation": round(occ, 5)})
        elif occ >= EDGE_EMPTY_HIGH:
            counts["truly_active"] += 1
        elif occ > DEMOTE_THRESHOLD:
            counts["edge_empty"] += 1
            demote.append({"orbital_in_active_set": orb_idx, "occupation": round(occ, 5)})
        else:
            counts["near_virtual"] += 1
            demote.append({"orbital_in_active_set": orb_idx, "occupation": round(occ, 5)})
    total = len(occupations)
    parts = [f"{counts['truly_active']}/{total} truly active"]
    if promote:
        parts.append(f"{len(promote)} candidate(s) for promotion to inactive")
    if demote:
        parts.append(f"{len(demote)} candidate(s) for demotion to secondary")
    return {
        "counts": counts,
        "summary": "; ".join(parts),
        "promote_orbitals": promote,
        "demote_orbitals": demote,
    }


def _verdict_from_quality(per_root: list[dict[str, Any]]) -> str:
    if not per_root:
        return "unknown"
    truly = sum(r["n_truly_active"] for r in per_root)
    total = sum(r["n_active"] for r in per_root)
    if total == 0:
        return "unknown"
    truly_fraction = truly / total
    promote_frac = sum(r["n_promote_candidates"] for r in per_root) / total
    demote_frac = sum(r["n_demote_candidates"] for r in per_root) / total
    # Hard "near-virtual" (occ < 0.02) is the worst signal — those orbitals
    # never enter the wave function and waste CSF count.
    has_near_virtual = any(
        any(d["occupation"] < DEMOTE_THRESHOLD for d in r["demote_orbitals"])
        for r in per_root
    )
    if truly_fraction >= 0.6 and not has_near_virtual:
        return "healthy"
    if truly_fraction >= 0.3 or (promote_frac + demote_frac) <= 0.5:
        return "marginal"
    return "poor"


def _build_recommendations(
    per_root: list[dict[str, Any]],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    promote_idxs: set[int] = set()
    demote_idxs: set[int] = set()
    for r in per_root:
        for o in r["promote_orbitals"]:
            promote_idxs.add(o["orbital_in_active_set"])
        for o in r["demote_orbitals"]:
            demote_idxs.add(o["orbital_in_active_set"])
    if promote_idxs:
        recs.append(
            {
                "action": "promote_to_inactive",
                "orbital_indices_in_active_set": sorted(promote_idxs),
                "rationale": (
                    "These orbitals have NO occupations >= 1.90 across one or more roots; "
                    "they behave as inactive doubles and inflate the CI expansion without "
                    "carrying meaningful correlation. Promote them to Inactive in the next "
                    "RASSCF input."
                ),
            }
        )
    if demote_idxs:
        recs.append(
            {
                "action": "demote_to_secondary",
                "orbital_indices_in_active_set": sorted(demote_idxs),
                "rationale": (
                    "These orbitals have NO occupations <= 0.10; they contribute negligibly "
                    "to the wave function. Removing them from the active space frees room "
                    "to add chemically meaningful orbitals."
                ),
            }
        )
    if not promote_idxs and not demote_idxs:
        recs.append(
            {
                "action": "keep_active_space",
                "rationale": "All active orbitals have NO occupations in [0.10, 1.90] — the active space is well-balanced.",
            }
        )
    return recs


def _next_actions_from_verdict(verdict: str, recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if verdict == "healthy":
        return [
            {
                "tool": "draft_molcas_caspt2",
                "rationale": "Active space is healthy — proceed to CASPT2 setup.",
            }
        ]
    if verdict == "marginal":
        return [
            {
                "tool": "parse_molcas_inporb",
                "args": {"path": "<RasOrb file>"},
                "rationale": "Inspect RasOrb to identify which orbitals to promote/demote, then redraft RASSCF.",
            },
            {
                "tool": "draft_molcas_input",
                "rationale": "Modify Inactive / RAS2 partitioning per recommendations, then re-run RASSCF before attempting CASPT2.",
            },
        ]
    return [
        {
            "tool": "search_molcas_docs",
            "args": {"query": "RASSCF active space selection guidance"},
            "rationale": "Active space looks unhealthy — consult docs and reconsider chemically motivated orbital choice.",
        }
    ]


# --- CASPT2 setup validator ----------------------------------------------------


def validate_caspt2_setup(payload: dict[str, Any]) -> dict[str, Any]:
    """Inspect a parsed CASPT2 payload (post-run) and emit a structured verdict.

    Checks:
      1. Reference weight per group (high / caution / unreliable bands)
      2. IPEA shift (warn if 0.0 with no SHIFT/IMAGINARY SHIFT/SIG2 set)
      3. Real intruder states (large coeff AND small denominator)
      4. Multistate setup consistency
    """
    specs = payload.get("specifications") or {}
    groups = payload.get("per_group_results") or []
    intruder_report = payload.get("intruder_state_report") or {}
    checks: list[dict[str, Any]] = []

    # 1. Reference weights
    weights = [g["reference_weight"] for g in groups if g.get("reference_weight") is not None]
    if not weights:
        checks.append(
            {"id": "reference_weight", "status": "missing", "message": "No reference weight reported."}
        )
    else:
        min_w = min(weights)
        if min_w >= REFW_HEALTHY:
            status = "ok"
        elif min_w >= REFW_CAUTION:
            status = "caution"
        else:
            status = "fail"
        checks.append(
            {
                "id": "reference_weight",
                "status": status,
                "min_weight": min_w,
                "n_groups": len(weights),
                "message": _weight_message(min_w),
            }
        )

    # 2. IPEA / shift consistency
    ipea = specs.get("ipea_shift")
    real_shift = specs.get("real_shift") or 0.0
    imag_shift = specs.get("imaginary_shift") or 0.0
    sigma_p = specs.get("sigma_p_regularization")
    has_any_shift = (real_shift > 0) or (imag_shift > 0) or bool(sigma_p)
    if ipea is not None and ipea == 0.0 and not has_any_shift:
        checks.append(
            {
                "id": "ipea_shift",
                "status": "warning",
                "message": (
                    "IPEA shift is 0.0 and no other shift / regularization is active. The default "
                    "Molcas IPEA shift is 0.25 — reproducing original CASPT2 behaviour requires an "
                    "explicit choice. Confirm this matches the user's intent."
                ),
                "ipea_shift": ipea,
            }
        )
    elif ipea is not None:
        checks.append(
            {
                "id": "ipea_shift",
                "status": "info",
                "message": f"IPEA shift = {ipea}; real shift = {real_shift}; imaginary shift = {imag_shift}",
                "ipea_shift": ipea,
                "real_shift": real_shift,
                "imaginary_shift": imag_shift,
            }
        )

    # 3. Real intruders
    intruders = intruder_report.get("intruders") or []
    if intruders:
        max_c = max(abs(r["coefficient"]) for r in intruders)
        min_d = min(abs(r["denominator"]) for r in intruders)
        status = "fail" if max_c >= 0.10 or min_d < 0.05 else "caution"
        if has_any_shift or sigma_p:
            shift_advice = "Existing shift may already mitigate this; verify by raising the shift."
        else:
            shift_advice = "Consider IMAGINARY SHIFT 0.1-0.2 or SIG2 regularization."
        checks.append(
            {
                "id": "intruder_states",
                "status": status,
                "n_intruders": len(intruders),
                "max_coefficient": max_c,
                "min_denominator": min_d,
                "message": (
                    f"{len(intruders)} intruder excitation(s) with small denominator + large coeff. "
                    + shift_advice
                ),
            }
        )

    # 4. Multistate hint when there are multiple groups but type is SS
    if (specs.get("calculation_type", "").upper().startswith("SS")) and len(groups) > 1:
        checks.append(
            {
                "id": "multistate_hint",
                "status": "warning",
                "message": (
                    "Multiple SS-CASPT2 groups present but calculation type is SS-CASPT2. If "
                    "states are close in energy or share the same symmetry, consider MS / "
                    "XMS / RMS / XDW CASPT2 to handle state mixing."
                ),
            }
        )

    verdict = _caspt2_verdict_from_checks(checks)
    next_actions = _caspt2_next_actions(verdict, checks, has_any_shift)
    return {
        "verdict": verdict,
        "checks": checks,
        "next_actions": next_actions,
    }


def _weight_message(min_w: float) -> str:
    if min_w >= REFW_HEALTHY:
        return f"Lowest reference weight = {min_w:.3f} — healthy."
    if min_w >= REFW_CAUTION:
        return (
            f"Lowest reference weight = {min_w:.3f} — borderline. CASPT2 is still meaningful but "
            "consider expanding the active space (add π* or Rydberg orbitals)."
        )
    return (
        f"Lowest reference weight = {min_w:.3f} (< 0.70) — CASPT2 is unreliable. Active space "
        "is missing important configurations; do not trust the energies."
    )


def _caspt2_verdict_from_checks(checks: list[dict[str, Any]]) -> str:
    statuses = [c["status"] for c in checks]
    if "fail" in statuses:
        return "unreliable"
    if "caution" in statuses or "warning" in statuses:
        return "caution"
    return "healthy"


def _caspt2_next_actions(
    verdict: str,
    checks: list[dict[str, Any]],
    has_any_shift: bool,
) -> list[dict[str, Any]]:
    if verdict == "unreliable":
        return [
            {
                "tool": "analyze_molcas_active_space",
                "rationale": "Reference weight or intruder problem — re-evaluate the RASSCF active space first.",
            }
        ]
    if verdict == "caution":
        if not has_any_shift:
            return [
                {
                    "tool": "draft_molcas_input",
                    "args": {"hint": "add IMAGINARY SHIFT 0.1 to CASPT2 block"},
                    "rationale": "Add a shift to the CASPT2 input as insurance against weak intruders.",
                }
            ]
        return [
            {
                "tool": "analyze_molcas_active_space",
                "rationale": "CASPT2 has caution-level signals (often a low reference weight). Re-check the active space before trusting energies.",
            }
        ]
    return [
        {
            "tool": "parse_molcas_output",
            "args": {"task_index": "auto"},
            "rationale": "CASPT2 setup looks healthy — proceed to property analysis.",
        }
    ]


# --- Character-aware orbital-swap suggester -----------------------------------

def suggest_orbital_swaps_by_character(
    *,
    mo_block: dict,                  # output of parse_last_mo_block (has dominant_aos)
    rasscf_orbital_specs: dict,      # output of parse_rasscf()["orbital_specs"]
    target_atom_pattern: str,
    target_ao_pattern: str,
    symmetry: int = 1,
    top_dominant_aos: int = 1,
) -> dict:
    """Suggest orbital swaps to move target-character orbitals INTO the active space.

    Walks the LAST `++ Molecular orbitals:` block (mo_block — natural orbitals
    of the RASSCF run), classifies each orbital's space (inactive / active /
    secondary) from the RASSCF orbital_specs vectors, and matches each
    orbital's dominant AO(s) against (target_atom_pattern, target_ao_pattern).

    "Active misses" = orbitals currently in the active space whose dominant
    AO does NOT match the target character.
    "Outside matches" = orbitals OUTSIDE active (inactive or secondary) whose
    dominant AO DOES match.

    The suggested swaps pair each "active miss" with the highest-coefficient
    "outside match" not already proposed. After applying these swaps via
    `swap_molcas_inporb_orbitals` and re-running RASSCF with FILEORB, the
    active space should be enriched in target-character orbitals.

    Parameters
    ----------
    mo_block
        The `mo_block` field returned by parse_last_mo_block / get_molcas_orbitals.
    rasscf_orbital_specs
        The `orbital_specs` dict from parse_rasscf (has frozen/inactive/active/etc.
        per-symmetry vectors).
    target_atom_pattern
        Case-insensitive prefix to match an atom label, e.g. "Cr" matches "CR1".
    target_ao_pattern
        Case-insensitive prefix to match an AO label, e.g. "3d" matches "3d2-".
    symmetry
        1-indexed symmetry irrep to analyze. Default 1 (correct for C1).
    top_dominant_aos
        Number of top dominant AOs to consider for each orbital. Default 1
        (only the largest-coefficient AO must match).

    Returns
    -------
    dict with keys:
        target_pattern         summary of the pattern
        orbital_classification list of per-orbital records {index, space,
                                occupation, dominant_aos, matches_target}
        active_misses          orbital indices currently in active that don't match
        outside_matches        orbital indices outside active that do match
        suggested_swaps        list of (active_orb, outside_orb) pairs
        rationale              short explanation
    """
    sym_idx = symmetry - 1
    syms = mo_block.get("symmetry_blocks") or []
    if sym_idx >= len(syms):
        raise ValueError(f"mo_block has {len(syms)} symmetries; can't access sym {symmetry}")
    sym_block = syms[sym_idx]
    orbitals = sym_block.get("orbitals") or []

    # Space classification from RASSCF specs
    frozen = (rasscf_orbital_specs.get("frozen") or [0])[sym_idx]
    inactive = (rasscf_orbital_specs.get("inactive") or [0])[sym_idx]
    active = (rasscf_orbital_specs.get("active") or [0])[sym_idx]
    deleted = (rasscf_orbital_specs.get("deleted") or [0])[sym_idx]
    # boundaries (1-indexed): orbs 1..frozen are frozen, then inactive, then active, then secondary
    frozen_end = frozen
    inactive_end = frozen + inactive
    active_end = frozen + inactive + active

    def _space(orb_idx_1: int) -> str:
        if orb_idx_1 <= frozen_end:
            return "frozen"
        if orb_idx_1 <= inactive_end:
            return "inactive"
        if orb_idx_1 <= active_end:
            return "active"
        return "secondary"

    atom_pat = target_atom_pattern.upper()
    ao_pat = target_ao_pattern.lower()

    def _matches_target(orb: dict) -> tuple[bool, float]:
        """Return (matches, dominance_score). Dominance = |coeff| of top matching AO."""
        dom_aos = orb.get("dominant_aos") or []
        for ao in dom_aos[:top_dominant_aos]:
            atom = str(ao.get("atom", "")).upper()
            label = str(ao.get("ao_label", "")).lower()
            if atom.startswith(atom_pat) and label.startswith(ao_pat):
                return True, abs(ao.get("coefficient", 0.0))
        return False, 0.0

    classification: list[dict] = []
    for orb in orbitals:
        idx = orb["orbital_index"]
        space = _space(idx)
        matches, score = _matches_target(orb)
        classification.append(
            {
                "orbital_index": idx,
                "space": space,
                "occupation": orb.get("occupation"),
                "energy_hartree": orb.get("energy_hartree"),
                "dominant_aos": (orb.get("dominant_aos") or [])[:top_dominant_aos],
                "matches_target": matches,
                "match_score": score,
            }
        )

    active_misses = [c for c in classification if c["space"] == "active" and not c["matches_target"]]
    outside_matches = [
        c for c in classification
        if c["space"] in {"inactive", "secondary"} and c["matches_target"]
    ]
    # Rank outside-matches by score (descending) — best candidates first
    outside_matches_sorted = sorted(outside_matches, key=lambda c: -c["match_score"])

    # Pair each active miss with the next-best unused outside match
    suggested_swaps: list[dict] = []
    used = set()
    for miss in active_misses:
        for cand in outside_matches_sorted:
            if cand["orbital_index"] in used:
                continue
            suggested_swaps.append(
                {
                    "active_orbital": miss["orbital_index"],
                    "swap_with": cand["orbital_index"],
                    "candidate_space": cand["space"],
                    "candidate_dominant_ao": cand["dominant_aos"][0] if cand["dominant_aos"] else None,
                    "current_active_dominant_ao": miss["dominant_aos"][0] if miss["dominant_aos"] else None,
                }
            )
            used.add(cand["orbital_index"])
            break

    rationale = (
        f"Looking for active orbitals dominated by AO pattern '{target_atom_pattern} "
        f"{target_ao_pattern}'. Found {sum(1 for c in classification if c['matches_target'] and c['space'] == 'active')} "
        f"matches already in active, {len(active_misses)} non-matching active orbitals, "
        f"{len(outside_matches)} matching candidates outside active. "
        f"Suggested {len(suggested_swaps)} swap(s) to bring matches into active."
    )
    return {
        "target_pattern": {"atom": target_atom_pattern, "ao_label": target_ao_pattern},
        "symmetry": symmetry,
        "orbital_classification": classification,
        "active_misses": [c["orbital_index"] for c in active_misses],
        "outside_matches": [c["orbital_index"] for c in outside_matches_sorted],
        "suggested_swaps": suggested_swaps,
        "rationale": rationale,
    }

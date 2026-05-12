"""DIRAC open-shell analysis — character verification + quality verdict.

Builds on the basic ``analyze_dirac_open_shell`` (input AOC vs h5 occupation
parity) by adding:
- MO character classification via VECPOP / Mulliken-per-MO blocks
- Cross-check observed open-shell j-character against an expected character
  (e.g. ``f 5/2`` + ``f 7/2`` for actinide 5f^n) — flags character mismatches
- Energy-clustering check (open shell should sit between closed and virtual)
- Concrete swap suggestions when the open shell holds wrong-character MOs

Used by the MCP tool ``analyze_dirac_open_shell_quality``.
"""

from __future__ import annotations

from typing import Any

from chemtools.core.issues import IssueCollector
from chemtools.programs.dirac.parse.vecpop import (
    parse_vecpop,
    classify_mo_character,
)


# Common open-shell chemistry → expected j-character signatures.
_EXPECTED_CHARACTER: dict[str, list[str]] = {
    "valence_d":  ["d 3/2", "d 5/2"],
    "valence_f":  ["f 5/2", "f 7/2"],
    "actinide_5f": ["f 5/2", "f 7/2"],
    "lanthanide_4f": ["f 5/2", "f 7/2"],
    "transition_metal_d": ["d 3/2", "d 5/2"],
    "valence_p":  ["p 1/2", "p 3/2"],
    "single_unpaired_s": ["s 1/2"],
}


def analyze_open_shell_quality(
    output_path: str,
    *,
    h5_orbitals: list[dict[str, Any]] | None = None,
    expected_character: str | list[str] | None = None,
) -> dict[str, Any]:
    """Run a full open-shell quality analysis.

    Parameters
    ----------
    output_path
        Path to the DIRAC ``.out`` (must have MULPOP/VECPOP blocks for
        character analysis to work).
    h5_orbitals
        Optional pre-parsed orbital summary from
        ``read_orbital_summary(.h5)``. If provided, cross-checks character
        labels against h5 occupations. If None, we use VECPOP alone.
    expected_character
        Either a chemistry hint (``valence_d``, ``actinide_5f``, etc.) or
        an explicit list of j-character strings (``["f 5/2", "f 7/2"]``).
        When provided, mismatches are flagged as caution-level issues.

    Returns
    -------
    dict with verdict (healthy / caution / problematic), issues, summary.
    """
    with open(output_path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    vecpop = parse_vecpop(text)
    collector = IssueCollector()

    # Pick out the open-shell (fractional occupation) MOs across all ircops.
    open_mos: list[dict[str, Any]] = []
    for ircop, mos in vecpop["ircops"].items():
        for mo in mos:
            if 1e-4 < mo["occupation"] < 1.0 - 1e-4:
                entry = dict(mo)
                entry["ircop"] = ircop
                entry["character_info"] = classify_mo_character(mo)
                open_mos.append(entry)

    # Closed-shell + virtual MOs grouped per ircop for energy-clustering checks.
    closed_mos: dict[str, list[dict[str, Any]]] = {}
    virtual_mos: dict[str, list[dict[str, Any]]] = {}
    for ircop, mos in vecpop["ircops"].items():
        closed_mos[ircop] = [m for m in mos if m["occupation"] >= 1.0 - 1e-4]
        virtual_mos[ircop] = [m for m in mos if m["occupation"] <= 1e-4]

    # --- Check 1: VECPOP block actually had data ----------------------
    if not vecpop["ircops"]:
        collector.add(
            "caution",
            "No VECPOP / Mulliken-per-MO output found.",
            hint=(
                "Enable **ANALYZE / .MULPOP plus *MULPOP / .VECPOP in the "
                "input to print per-MO j-character and AO populations."
            ),
        )

    # --- Check 2: open-shell MOs identified --------------------------
    if not open_mos and vecpop["ircops"]:
        collector.add(
            "info",
            "VECPOP found no fractionally-occupied (open-shell) MOs.",
            hint=(
                "If the input requested .OPEN SHELL, the SCF may have "
                "converged into a closed-shell solution — check the AOC "
                "configuration."
            ),
        )

    # --- Check 3: character matches expectation -----------------------
    expected_list: list[str] = []
    if expected_character:
        if isinstance(expected_character, str):
            expected_list = _EXPECTED_CHARACTER.get(
                expected_character.lower(), [expected_character]
            )
        else:
            expected_list = list(expected_character)
    if expected_list and open_mos:
        observed = {mo["character_info"]["character"] for mo in open_mos}
        observed = {c for c in observed if c and c != "unknown"}
        unexpected = observed - set(expected_list)
        missing = set(expected_list) - observed
        if unexpected:
            collector.add(
                "problematic",
                f"Open shell has unexpected character(s): {sorted(unexpected)} "
                f"(expected {expected_list}).",
                hint=(
                    "Likely starting-orbital mistake. Use suggest_dirac_orbital_swaps "
                    "to identify candidate spinors with the right character to swap in."
                ),
            )
        elif missing:
            collector.add(
                "caution",
                f"Some expected open-shell character(s) missing: {sorted(missing)}.",
                hint="May indicate partial convergence or that fewer states than expected are populated.",
            )

    # --- Check 4: energy clustering -----------------------------------
    # The open-shell orbitals should sit between the highest closed orbital
    # and the lowest virtual orbital in their ircop.
    for mo in open_mos:
        ircop = mo["ircop"]
        e = mo["energy_hartree"]
        cls = closed_mos.get(ircop, [])
        virt = virtual_mos.get(ircop, [])
        if cls and e < min(m["energy_hartree"] for m in cls if m.get("energy_hartree") is not None):
            collector.add(
                "problematic",
                f"Open-shell MO {mo['eigenvalue_index']} in {ircop} (E={e:.4f} Ha) "
                f"sits below all closed-shell orbitals — likely a mis-ordering. "
                f"Character: {mo['character_info']['character']}.",
                hint="Run suggest_dirac_orbital_swaps to find a better starting configuration.",
            )
        if virt and e > min(m["energy_hartree"] for m in virt if m.get("energy_hartree") is not None) + 0.1:
            collector.add(
                "caution",
                f"Open-shell MO {mo['eigenvalue_index']} in {ircop} (E={e:.4f} Ha) "
                f"is higher than a virtual orbital — open shell may have wrong identity.",
            )

    # --- Build summary ------------------------------------------------
    by_ircop: dict[str, list[dict[str, Any]]] = {}
    for mo in open_mos:
        by_ircop.setdefault(mo["ircop"], []).append({
            "eigenvalue_index": mo["eigenvalue_index"],
            "energy_hartree": mo["energy_hartree"],
            "occupation": mo["occupation"],
            "character": mo["character_info"]["character"],
            "dominant_ao": mo["character_info"]["dominant_ao"],
            "dominant_atom": mo["character_info"]["dominant_atom"],
        })

    # Energy ordering description: where the open shell sits in each ircop.
    energy_summary: dict[str, Any] = {}
    for ircop in by_ircop:
        cls = closed_mos.get(ircop, [])
        virt = virtual_mos.get(ircop, [])
        opens = [m for m in open_mos if m["ircop"] == ircop]
        if opens:
            e_open = [m["energy_hartree"] for m in opens]
            energy_summary[ircop] = {
                "highest_closed_hartree": max(
                    (m["energy_hartree"] for m in cls if m.get("energy_hartree") is not None),
                    default=None,
                ),
                "open_shell_energy_range": [min(e_open), max(e_open)] if e_open else None,
                "lowest_virtual_hartree": min(
                    (m["energy_hartree"] for m in virt if m.get("energy_hartree") is not None),
                    default=None,
                ),
            }

    return {
        "verdict": collector.verdict,
        "issues": collector.issues,
        "open_shell_mos_by_ircop": by_ircop,
        "energy_clustering": energy_summary,
        "expected_character": expected_list or None,
        "ircops_analyzed": list(vecpop["ircops"].keys()),
        "n_open_shell_mos": len(open_mos),
    }


# j-character parity (gerade/ungerade) follows orbital angular momentum:
#   s, d, g (even L) → gerade   (DIRAC fermion ircop 1 / E1g)
#   p, f, h (odd L)  → ungerade (DIRAC fermion ircop 2 / E1u)
_GERADE_LABELS = frozenset({"s", "d", "g", "i"})
_UNGERADE_LABELS = frozenset({"p", "f", "h", "k"})


def _character_parity(character: str) -> str | None:
    """Return ``'gerade'``, ``'ungerade'``, or None for an unknown j-character.

    Accepts both raw j-labels (``"s"``, ``"d"``, ``"f"``) and the formatted
    character strings the VECPOP classifier emits (``"f 5/2"``).
    """
    if not character:
        return None
    head = character.split()[0].lower()
    if head in _GERADE_LABELS:
        return "gerade"
    if head in _UNGERADE_LABELS:
        return "ungerade"
    return None


def suggest_orbital_swaps(
    output_path: str,
    target_character: list[str],
    *,
    n_candidates: int = 6,
) -> dict[str, Any]:
    """Find candidate MOs with the target character to swap into the open shell.

    Walks VECPOP to identify (a) current open-shell MOs whose character is
    NOT in ``target_character`` (wrong-character open MOs to swap OUT) and
    (b) virtual / closed MOs with the target character in the SAME fermion
    ircop (candidates to swap IN).

    Returns explicit, actionable swap pairings WITHIN each ircop plus a
    pre-rendered ``.REORDER MO`` spec ready for ``apply_dirac_reorder_to_input``.
    Also detects the **parity-incompatible** case (target character lives in
    a different ircop than the current open shell) — in that case .REORDER
    can't help and the agent must redraft the ``.OPEN SHELL`` spec instead.

    Possible verdicts:
      - ``no_action_needed``       — open shell already has target character
      - ``swaps_available``        — intra-ircop swaps are actionable;
                                     ``per_ircop_orders`` is ready to apply
      - ``parity_incompatible``    — wrong-char open and target-char candidates
                                     are in DIFFERENT ircops; need .OPEN SHELL
                                     redraft, not a .REORDER
      - ``no_candidates_found``    — target character not present in this run
                                     at all (suggests the chemistry hint is
                                     wrong, or basis is too small)
    """
    with open(output_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    vecpop = parse_vecpop(text)
    targets = set(target_character)

    target_parities: set[str] = set()
    for c in targets:
        p = _character_parity(c)
        if p:
            target_parities.add(p)

    suggestions: dict[str, dict[str, Any]] = {}
    ircop_order = list(vecpop["ircops"].keys())

    # Stable ircop → fermion-symmetry index mapping (1-based, matches DIRAC).
    # The VECPOP block iteration order follows the output's print order:
    # fsym 1 first (E1g / gerade), then fsym 2 (E1u / ungerade), then any
    # extras. For atomic / linear systems this is the canonical pairing.
    ircop_fsym_idx: dict[str, int] = {
        name: i + 1 for i, name in enumerate(ircop_order)
    }

    # Map ircop name → parity, using the DIRAC convention.
    ircop_parity: dict[str, str] = {}
    for name in ircop_order:
        u = name.upper()
        if "U" in u or "ungerade" in name.lower():
            ircop_parity[name] = "ungerade"
        else:
            ircop_parity[name] = "gerade"

    for ircop, mos in vecpop["ircops"].items():
        n_orbitals_in_ircop = len(mos)
        wrong_open: list[dict[str, Any]] = []
        for mo in mos:
            if not (1e-4 < mo["occupation"] < 1.0 - 1e-4):
                continue
            c = classify_mo_character(mo).get("character")
            if c not in targets:
                wrong_open.append({
                    "eigenvalue_index": mo["eigenvalue_index"],
                    "energy_hartree": mo["energy_hartree"],
                    "character": c,
                })
        # Sort wrong-open by index for stable pairing.
        wrong_open.sort(key=lambda m: m["eigenvalue_index"])

        candidates: list[dict[str, Any]] = []
        for mo in mos:
            occ = mo["occupation"]
            if 1e-4 < occ < 1.0 - 1e-4:
                continue  # already in open shell
            c = classify_mo_character(mo).get("character")
            if c in targets:
                candidates.append({
                    "eigenvalue_index": mo["eigenvalue_index"],
                    "energy_hartree": mo["energy_hartree"],
                    "occupation": occ,
                    "character": c,
                    "currently": "closed" if occ >= 1.0 - 1e-4 else "virtual",
                })
        # Sort candidates by energy ascending — lowest-energy candidates are
        # the closest to the open-shell window, most chemically sensible.
        candidates.sort(key=lambda m: m["energy_hartree"])
        candidates_truncated = candidates[:n_candidates]

        # Build explicit swap pairings (wrong[i] ↔ candidate[i]) within
        # this ircop. Limit to min(len(wrong), len(candidates)).
        swap_pairs: list[tuple[int, int]] = []
        if wrong_open and candidates:
            n_pairs = min(len(wrong_open), len(candidates))
            for i in range(n_pairs):
                swap_pairs.append((
                    wrong_open[i]["eigenvalue_index"],
                    candidates[i]["eigenvalue_index"],
                ))

        # Render the per-ircop REORDER spec when swaps are actionable.
        reorder_spec: str | None = None
        if swap_pairs:
            from chemtools.programs.dirac.strategy.reorder import swaps_to_reorder_spec
            reorder_spec = swaps_to_reorder_spec(n_orbitals_in_ircop, swap_pairs)

        suggestions[ircop] = {
            "fsym_index": ircop_fsym_idx[ircop],
            "parity": ircop_parity[ircop],
            "n_orbitals_in_ircop": n_orbitals_in_ircop,
            "wrong_open_shell": wrong_open,
            "candidates_with_target_character": candidates_truncated,
            "n_candidates_available": len(candidates),
            "suggested_swaps": [
                {"wrong_eigenvalue_index": a, "candidate_eigenvalue_index": b}
                for a, b in swap_pairs
            ],
            "reorder_spec": reorder_spec,
        }

    # Cross-ircop synthesis: figure out the verdict + per-ircop spec list
    n_wrong = sum(len(v["wrong_open_shell"]) for v in suggestions.values())
    n_cand_actionable = sum(
        len(v["suggested_swaps"]) for v in suggestions.values()
    )
    n_cand_total = sum(
        v["n_candidates_available"] for v in suggestions.values()
    )

    # Identify ircop parities of wrong-open vs candidates
    wrong_parities = {
        suggestions[ic]["parity"] for ic in suggestions
        if suggestions[ic]["wrong_open_shell"]
    }
    cand_parities = {
        suggestions[ic]["parity"] for ic in suggestions
        if suggestions[ic]["candidates_with_target_character"]
    }

    verdict: str
    explanation: str
    next_actions: list[dict[str, Any]] = []

    if n_wrong == 0:
        verdict = "no_action_needed"
        explanation = "Open shell already has target character — no reorder needed."
    elif n_cand_total == 0:
        verdict = "no_candidates_found"
        explanation = (
            f"No MOs with character {sorted(targets)} found anywhere in the "
            f"electronic spectrum. The chemistry hint may be wrong, the "
            f"basis may not span the target shell (e.g. you asked for d "
            f"character on a basis with only s + p), or the run hasn't "
            f"yet been converged enough to expose virtuals of that kind."
        )
        next_actions.append({
            "tool": "list_dirac_docs",
            "rationale": "Confirm basis library has functions for the target shell.",
        })
    elif n_cand_actionable == 0:
        # Wrong-open and candidates exist but never share an ircop.
        verdict = "parity_incompatible"
        explanation = (
            f"The current open shell sits in ircop(s) {sorted(wrong_parities)} "
            f"but target character {sorted(targets)} lives in ircop(s) "
            f"{sorted(cand_parities)}. ``.REORDER MO`` only reorders within a "
            f"fermion ircop, so a reorder cannot bridge the parity gap. "
            f"What the agent should do instead: redraft the ``.OPEN SHELL`` "
            f"spec to put the open electrons in the correct-parity manifold "
            f"(e.g. ``2/10,0`` for a gerade d-shell with 2 electrons in 10 "
            f"spinors of fsym 1, vs. ``2/0,14`` for an ungerade f-shell)."
        )
        next_actions.append({
            "tool": "draft_dirac_input",
            "rationale": (
                "Redraft the input with the .OPEN SHELL spec moved into the "
                f"correct-parity ircop ({sorted(cand_parities)})."
            ),
        })
    else:
        verdict = "swaps_available"
        per_ircop_orders = [
            suggestions[ic].get("reorder_spec") or "1..oo"
            for ic in ircop_order
        ]
        explanation = (
            f"Found {n_cand_actionable} actionable intra-ircop swaps. "
            f"Apply via ``apply_dirac_reorder_to_input`` with "
            f"per_ircop_orders={per_ircop_orders}."
        )
        next_actions.append({
            "tool": "apply_dirac_reorder_to_input",
            "rationale": "Insert the .REORDER MO block to swap wrong-character open MOs with target-character candidates.",
            "args": {"per_ircop_orders": per_ircop_orders},
        })

    return {
        "target_character": target_character,
        "verdict": verdict,
        "explanation": explanation,
        "ircops": suggestions,
        "ircop_order": ircop_order,
        "wrong_open_shell_total": n_wrong,
        "candidates_total": n_cand_total,
        "actionable_swaps_total": n_cand_actionable,
        "wrong_open_parities": sorted(wrong_parities),
        "candidate_parities": sorted(cand_parities),
        "next_actions": next_actions,
        "per_ircop_orders": (
            [suggestions[ic].get("reorder_spec") or "1..oo" for ic in ircop_order]
            if verdict == "swaps_available" else None
        ),
    }

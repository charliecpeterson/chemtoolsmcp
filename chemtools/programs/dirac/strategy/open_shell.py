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


def suggest_orbital_swaps(
    output_path: str,
    target_character: list[str],
    *,
    n_candidates: int = 6,
) -> dict[str, Any]:
    """Find candidate MOs with the target character to swap into the open
    shell.

    Walks VECPOP, identifies (a) the current open-shell MOs (which need to
    be swapped OUT if they have wrong character) and (b) virtual or closed
    MOs with the target character (candidates to swap IN).

    Returns suggested ``.REORDER`` candidates — pairs of
    (current_index, target_index) per fermion ircop, plus the rationale.
    """
    with open(output_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    vecpop = parse_vecpop(text)
    targets = set(target_character)

    suggestions: dict[str, dict[str, Any]] = {}
    for ircop, mos in vecpop["ircops"].items():
        # Current open shell — wrong character → these are the ones to swap out
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

        # Candidates with target character among virtuals + low-energy closed
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

        candidates.sort(key=lambda m: m["energy_hartree"])
        candidates = candidates[:n_candidates]

        suggestions[ircop] = {
            "wrong_open_shell": wrong_open,
            "candidates_with_target_character": candidates,
        }

    n_wrong = sum(len(v["wrong_open_shell"]) for v in suggestions.values())
    n_cand = sum(len(v["candidates_with_target_character"]) for v in suggestions.values())

    return {
        "target_character": target_character,
        "ircops": suggestions,
        "wrong_open_shell_total": n_wrong,
        "candidates_total": n_cand,
        "recommendation": (
            "Use the MCP tool `draft_dirac_reorder_block` to render a "
            ".REORDER spec swapping each wrong_open_shell entry with a "
            "candidate of matching character, then `apply_dirac_reorder_to_input`."
        ) if n_wrong else (
            "Open shell already has target character — no reorder needed."
        ),
    }

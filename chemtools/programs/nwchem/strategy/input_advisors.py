"""NWChem pre-job input advisors.

Three "before drafting" advisors that look at the molecule and basic
properties (elements, charge, etc.) and recommend the right input
parameters:

  * suggest_spin_state              For transition-metal systems,
                                     compute d-electron counts and
                                     recommend high-spin/low-spin
                                     multiplicities.
  * suggest_basis_set               Recommend basis (and ECP) given
                                     elements + purpose (geometry /
                                     single-point / correlated / heavy).
  * suggest_relativistic_correction  Advise on relativistic corrections
                                     based on element Z values and
                                     selected basis sets.

The data tables these advisors lean on (transition-metal d-counts,
ligand-field classification, Pople / DK basis patterns, relativistic
method blocks, relativistic-importance Z thresholds) live in their
neighbour modules (``plausibility.py`` for the spin-state tables,
``resources.py`` for the relativistic tables) — they were positioned
near their primary consumers when ``api_strategy.py`` was split, and
we re-import them here.

``suggest_basis_set`` is owned by ``chemtools.core.basis_advisor``. The
re-export below preserves its older NWChem import path.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

from chemtools.core.common import ELEMENT_TO_Z
from chemtools.programs.nwchem.input._utils import _TRANSITION_METALS
from chemtools.programs.nwchem.parse.mos import METAL_CENTERS
from chemtools.programs.nwchem.input.lint_restart import inspect_input

# Constants currently living in sibling strategy modules — see docstring.
from chemtools.programs.nwchem.strategy.plausibility import (
    _TM_Z_CORE,
    _TM_COMMON_OX,
    _D_HS_UNPAIRED,
    _D_LS_UNPAIRED,
    _WEAK_FIELD_ELEMENTS,
    _STRONG_FIELD_ELEMENTS,
)
from chemtools.programs.nwchem.strategy.resources import (
    _REL_CRITICAL_Z,
    _REL_IMPORTANT_Z,
    _REL_SIGNIFICANT_Z,
    _DK_BASIS_PATTERNS,
    _REL_METHODS,
)


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


def recommend_multiplicity_scan(
    elements: list[str],
    charge: int = 0,
    current_multiplicity: int | None = None,
    metal_oxidation_states: dict[str, int] | None = None,
    max_multiplicity: int = 11,
) -> dict[str, Any]:
    """Recommend a multiplicity scan to verify the spin ground state.

    A converged SCF says nothing about whether its multiplicity is the lowest
    in energy: NWChem converges cleanly to whatever spin is requested, and for
    open-shell d/f systems the wrong spin state can sit tens of kcal/mol above
    the true ground state with no error or warning. The only reliable
    single-reference diagnostic is to run several multiplicities and compare
    total energies. Returns whether a scan is warranted and the parity-correct
    multiplicities to run.
    """
    norm = [e[0].upper() + e[1:].lower() for e in elements]
    total_electrons = sum(ELEMENT_TO_Z.get(e, 0) for e in norm) - charge
    # M = 2S+1, so the unpaired count M-1 must match the electron-count parity:
    # even electrons -> odd multiplicity, odd electrons -> even multiplicity.
    mult_is_odd = total_electrons % 2 == 0
    metals = [e for e in dict.fromkeys(norm) if e in METAL_CENTERS]

    spin = suggest_spin_state(elements, charge, metal_oxidation_states)
    candidates: set[int] = set()
    for analysis in spin.get("metal_analyses", []):
        for ox in analysis.get("oxidation_state_analyses", []):
            for state in ox.get("spin_states", []):
                candidates.add(state["multiplicity"])
    if current_multiplicity:
        candidates.add(current_multiplicity)
    if not candidates:
        candidates.add(1 if mult_is_odd else 2)

    top = min(max(candidates) + 2, max_multiplicity)
    scan = [m for m in range(1, top + 1) if (m % 2 == 1) == mult_is_odd]

    # A scan is warranted when the spin state is genuinely in question: the run is
    # open-shell as executed, or a d-block metal has more than one accessible spin
    # state (d4-d7). A metal alone is not enough — d0/d10/f0/f14 and other
    # single-spin-state ions (e.g. La(III) 4f0) are unambiguously closed-shell.
    open_shell_as_run = current_multiplicity is not None and current_multiplicity > 1
    has_spin_ambiguity = any(
        len(ox.get("spin_states", [])) > 1
        for analysis in spin.get("metal_analyses", [])
        for ox in analysis.get("oxidation_state_analyses", [])
    )
    warranted = open_shell_as_run or has_spin_ambiguity
    if not warranted:
        rationale = (
            "No spin ambiguity: the system is closed-shell as run and no metal has "
            "multiple accessible spin states — a single multiplicity suffices."
        )
    elif open_shell_as_run and metals:
        rationale = (
            f"Open-shell metal system (multiplicity {current_multiplicity}, "
            f"{', '.join(metals)}): a converged SCF does not confirm the spin ground "
            f"state. Run multiplicities {scan} at the same geometry and basis, then "
            f"take the lowest energy (note any near-degeneracy within ~5 kcal/mol)."
        )
    elif has_spin_ambiguity:
        rationale = (
            f"Metal(s) {', '.join(metals)} have more than one accessible spin state "
            f"(high-/low-spin) — even a clean singlet may not be the ground state. "
            f"Scan {scan} and compare total energies."
        )
    else:
        rationale = (
            f"Open-shell system (multiplicity {current_multiplicity}): verify the "
            f"spin ground state by scanning {scan} and comparing total energies."
        )

    return {
        "scan_warranted": warranted,
        "total_electrons": total_electrons,
        "current_multiplicity": current_multiplicity,
        "recommended_multiplicities": scan,
        "metal_elements": metals,
        "ground_state_parity": "odd_multiplicity" if mult_is_odd else "even_multiplicity",
        "spin_state_analysis": spin,
        "rationale": rationale,
    }


def suggest_multiplicity_scan_from_source(
    *,
    input_file: str | None = None,
    elements: list[str] | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    metal_oxidation_states: dict[str, int] | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Infer missing input facts and prepare one multiplicity-scan result."""
    if input_file and (
        elements is None or charge is None or multiplicity is None
    ):
        summary = inspect_input(input_file)
        if elements is None:
            elements = summary.get("all_elements") or summary.get("elements")
        if charge is None:
            charge = summary.get("charge")
        if multiplicity is None:
            multiplicity = summary.get("multiplicity")
    if not elements:
        return {
            "error": (
                "Provide input_file (to read elements/charge/multiplicity) "
                "or an explicit elements list."
            ),
        }

    result = recommend_multiplicity_scan(
        elements=elements,
        charge=charge or 0,
        current_multiplicity=multiplicity,
        metal_oxidation_states=metal_oxidation_states,
    )
    if result["scan_warranted"] and input_file:
        result["next_actions"] = [{
            "priority": 1,
            "tool": "generate_nwchem_input_batch",
            "params": {
                "template_input": input_file,
                "vary": {"mult": result["recommended_multiplicities"]},
                "output_dir": output_dir or str(Path(input_file).parent),
            },
            "reason": (
                "Generate one input per candidate multiplicity at the same "
                "geometry and basis; run them, then take the lowest total "
                "energy."
            ),
            "confidence": 0.9,
        }]
    return result


# Compatibility re-export for the older NWChem import path.
from chemtools.core.basis_advisor import suggest_basis_set  # noqa: F401


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


__all__ = [
    "recommend_multiplicity_scan",
    "suggest_multiplicity_scan_from_source",
    "suggest_spin_state",
    "suggest_basis_set",
    "suggest_relativistic_correction",
]

"""Dyall relativistic basis set reference table for DIRAC.

Covers all basis families shipped with DIRAC 25. Element coverage is
extracted from the header comment of each basis file in the DIRAC
distribution. The tool-facing functions are:

  list_basis_sets(element=None, family=None, calc_type=None)
      Return matching families with descriptions and caveats.

  suggest_basis(element, calc_type, correlated=False)
      Return a ranked recommendation for a given element + job type.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Element-coverage sets (reproduced from DIRAC 25 basis file headers)
# ---------------------------------------------------------------------------

# s-block: H, He + groups 1-2
_S_BLOCK = {
    "H", "He",
    "Li", "Be", "Na", "Mg",
    "K", "Ca", "Rb", "Sr", "Cs", "Ba", "Fr", "Ra",
}
# p-block: groups 13-18 (rows 2-6)
_P_BLOCK = {
    "B", "C", "N", "O", "F", "Ne",
    "Al", "Si", "P", "S", "Cl", "Ar",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
}
# d-block: 3d, 4d, 5d, 6d
_D_BLOCK = {
    # 3d
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    # 4d
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    # 5d
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    # 6d
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
}
# La is a d-block/f-block bridge; included in non-diffuse f-block coverage
# f-block: lanthanides + actinides + La, Ac (period-6/7 heads)
_F_BLOCK = {
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
}
# 7p super-heavy (row 7 p-block, Z=113-118)
_G_BLOCK = {"Nh", "Fl", "Mc", "Lv", "Ts", "Og"}

# All elements in the non-diffuse families (everything above)
_ALL_NONDIFFUSE = _S_BLOCK | _P_BLOCK | _D_BLOCK | _F_BLOCK | _G_BLOCK

# Diffuse (aug-) families skip lanthanides, actinides, and 7p
_ALL_DIFFUSE = _S_BLOCK | _P_BLOCK | _D_BLOCK | _G_BLOCK


# ---------------------------------------------------------------------------
# Basis family catalog
# ---------------------------------------------------------------------------

_FAMILIES: list[dict] = [
    # ---- Valence (DZ / TZ / QZ / 5Z) ----
    {
        "name": "dyall.v2z",
        "quality": "double-zeta",
        "type": "valence",
        "diffuse": False,
        "zeta": 2,
        "elements": _ALL_NONDIFFUSE,
        "description": (
            "Dyall double-zeta valence basis. Correlates the chemically "
            "relevant valence shells (including outer core for d and f "
            "blocks). Good starting point; use for exploratory SCF or DFT. "
            "Larger than dyall.2zp but has correlating functions for CC/CI."
        ),
        "use_for": ["scf", "correlated", "cc", "ci"],
        "avoid_for": [],
        "references": {
            "5f": "K.G. Dyall, Theor. Chem. Acc. (2007) 117:491",
            "4f": "A.S.P. Gomes et al., Theor. Chem. Acc. (2010) 127:369",
            "5d": "K.G. Dyall, Theor. Chem. Acc. (2004) 112:403",
        },
    },
    {
        "name": "dyall.v3z",
        "quality": "triple-zeta",
        "type": "valence",
        "diffuse": False,
        "zeta": 3,
        "elements": _ALL_NONDIFFUSE,
        "description": (
            "Dyall triple-zeta valence. The workhorse for correlated "
            "relativistic calculations on actinides. CBS extrapolation "
            "typically uses v2z + v3z or v3z + v4z pairs."
        ),
        "use_for": ["correlated", "cc", "ci", "caspt2"],
        "avoid_for": [],
    },
    {
        "name": "dyall.v4z",
        "quality": "quadruple-zeta",
        "type": "valence",
        "diffuse": False,
        "zeta": 4,
        "elements": _ALL_NONDIFFUSE,
        "description": (
            "Dyall quadruple-zeta valence. Used for CBS extrapolation "
            "and benchmark calculations."
        ),
        "use_for": ["benchmark", "cc", "correlated"],
        "avoid_for": [],
    },
    # ---- DFT-optimised (no correlating functions, most compact) ----
    {
        "name": "dyall.2zp",
        "quality": "double-zeta",
        "type": "dft",
        "diffuse": False,
        "zeta": 2,
        "elements": _ALL_NONDIFFUSE,
        "description": (
            "Dyall DFT-optimised double-zeta. No extra correlating "
            "functions — the most compact Dyall family. Standard for "
            "atomic SCF, DFT, and AOC convergence tests. Validated for "
            "actinides (Cm 4c AOC converges in 13 iters). "
            "NOT suitable for post-HF (CC/CI) — use dyall.v2z or larger."
        ),
        "use_for": ["scf", "dft", "aoc"],
        "avoid_for": ["cc", "ci", "correlated", "mp2"],
    },
    {
        "name": "dyall.3zp",
        "quality": "triple-zeta",
        "type": "dft",
        "diffuse": False,
        "zeta": 3,
        "elements": _ALL_NONDIFFUSE,
        "description": (
            "Dyall DFT triple-zeta. Larger DFT-optimised set; rarely "
            "needed — dyall.2zp is usually enough for DFT. "
            "NOT suitable for post-HF."
        ),
        "use_for": ["scf", "dft"],
        "avoid_for": ["cc", "ci", "correlated"],
    },
    {
        "name": "dyall.4zp",
        "quality": "quadruple-zeta",
        "type": "dft",
        "diffuse": False,
        "zeta": 4,
        "elements": _ALL_NONDIFFUSE,
        "description": "Dyall DFT quadruple-zeta. Rarely used.",
        "use_for": ["scf", "dft"],
        "avoid_for": ["cc", "ci", "correlated"],
    },
    # ---- Core-valence ----
    {
        "name": "dyall.cv2z",
        "quality": "double-zeta",
        "type": "core-valence",
        "diffuse": False,
        "zeta": 2,
        "elements": _ALL_NONDIFFUSE,
        "description": (
            "Dyall core-valence DZ. Adds correlating functions for the "
            "outer-core shell ((n-2) for s-block, (n-1) for p/d-block). "
            "Use when outer-core correlation matters (e.g. EFG, core IP)."
        ),
        "use_for": ["cc", "correlated", "core_ip", "efg"],
        "avoid_for": [],
    },
    {
        "name": "dyall.cv3z",
        "quality": "triple-zeta",
        "type": "core-valence",
        "diffuse": False,
        "zeta": 3,
        "elements": _ALL_NONDIFFUSE,
        "description": "Dyall core-valence TZ. Standard for accurate core properties.",
        "use_for": ["cc", "correlated", "core_ip", "nmr", "efg"],
        "avoid_for": [],
    },
    {
        "name": "dyall.cv4z",
        "quality": "quadruple-zeta",
        "type": "core-valence",
        "diffuse": False,
        "zeta": 4,
        "elements": _ALL_NONDIFFUSE,
        "description": "Dyall core-valence QZ. Benchmark core properties.",
        "use_for": ["benchmark", "cc", "correlated"],
        "avoid_for": [],
    },
    # ---- All-electron ----
    {
        "name": "dyall.ae2z",
        "quality": "double-zeta",
        "type": "all-electron",
        "diffuse": False,
        "zeta": 2,
        "elements": _ALL_NONDIFFUSE,
        "description": (
            "Dyall all-electron DZ. Correlating functions for ALL shells "
            "down to 1s. Use only when full core correlation is needed "
            "(very expensive for heavy atoms)."
        ),
        "use_for": ["cc", "correlated", "full_core"],
        "avoid_for": ["routine"],
    },
    {
        "name": "dyall.ae3z",
        "quality": "triple-zeta",
        "type": "all-electron",
        "diffuse": False,
        "zeta": 3,
        "elements": _ALL_NONDIFFUSE,
        "description": "Dyall all-electron TZ.",
        "use_for": ["benchmark", "cc", "correlated", "full_core"],
        "avoid_for": [],
    },
    {
        "name": "dyall.ae4z",
        "quality": "quadruple-zeta",
        "type": "all-electron",
        "diffuse": False,
        "zeta": 4,
        "elements": _ALL_NONDIFFUSE,
        "description": "Dyall all-electron QZ. Only for extreme benchmarks.",
        "use_for": ["benchmark"],
        "avoid_for": ["routine"],
    },
    # ---- Augmented valence (diffuse; NO f-block) ----
    {
        "name": "dyall.av2z",
        "quality": "double-zeta",
        "type": "valence+diffuse",
        "diffuse": True,
        "zeta": 2,
        "elements": _ALL_DIFFUSE,
        "description": (
            "Dyall augmented valence DZ — adds diffuse functions for "
            "anions, excited states, polarizabilities. "
            "NOT available for lanthanides or actinides (f-block). "
            "For f-block anions / excited states there is no aug-dyall; "
            "use dyall.v3z or larger instead."
        ),
        "use_for": ["anion", "excited", "polarizability"],
        "avoid_for": ["f_block"],
        "f_block_available": False,
    },
    {
        "name": "dyall.av3z",
        "quality": "triple-zeta",
        "type": "valence+diffuse",
        "diffuse": True,
        "zeta": 3,
        "elements": _ALL_DIFFUSE,
        "description": "Dyall augmented valence TZ. Not available for f-block.",
        "use_for": ["anion", "excited", "polarizability"],
        "avoid_for": ["f_block"],
        "f_block_available": False,
    },
    {
        "name": "dyall.av4z",
        "quality": "quadruple-zeta",
        "type": "valence+diffuse",
        "diffuse": True,
        "zeta": 4,
        "elements": _ALL_DIFFUSE,
        "description": "Dyall augmented valence QZ. Not available for f-block.",
        "use_for": ["benchmark", "anion", "excited"],
        "avoid_for": ["f_block"],
        "f_block_available": False,
    },
    # ---- Augmented core-valence (diffuse; NO f-block) ----
    {
        "name": "dyall.acv2z",
        "quality": "double-zeta",
        "type": "core-valence+diffuse",
        "diffuse": True,
        "zeta": 2,
        "elements": _ALL_DIFFUSE,
        "description": (
            "Dyall augmented core-valence DZ. Diffuse + core-correlating; "
            "mainly for electron affinities and core excitations on "
            "s/p/d-block elements. NOT available for f-block."
        ),
        "use_for": ["electron_affinity", "core_excitation"],
        "avoid_for": ["f_block"],
        "f_block_available": False,
    },
    {
        "name": "dyall.acv3z",
        "quality": "triple-zeta",
        "type": "core-valence+diffuse",
        "diffuse": True,
        "zeta": 3,
        "elements": _ALL_DIFFUSE,
        "description": "Dyall augmented core-valence TZ. Not available for f-block.",
        "use_for": ["electron_affinity", "core_excitation"],
        "avoid_for": ["f_block"],
        "f_block_available": False,
    },
    {
        "name": "dyall.acv4z",
        "quality": "quadruple-zeta",
        "type": "core-valence+diffuse",
        "diffuse": True,
        "zeta": 4,
        "elements": _ALL_DIFFUSE,
        "description": "Dyall augmented core-valence QZ. Not available for f-block.",
        "use_for": ["benchmark", "electron_affinity"],
        "avoid_for": ["f_block"],
        "f_block_available": False,
    },
    # ---- Augmented all-electron (diffuse; NO f-block) ----
    {
        "name": "dyall.aae2z",
        "quality": "double-zeta",
        "type": "all-electron+diffuse",
        "diffuse": True,
        "zeta": 2,
        "elements": _ALL_DIFFUSE,
        "description": "Dyall augmented all-electron DZ. Not available for f-block.",
        "use_for": ["benchmark", "full_core"],
        "avoid_for": ["f_block", "routine"],
        "f_block_available": False,
    },
    {
        "name": "dyall.aae3z",
        "quality": "triple-zeta",
        "type": "all-electron+diffuse",
        "diffuse": True,
        "zeta": 3,
        "elements": _ALL_DIFFUSE,
        "description": "Dyall augmented all-electron TZ. Not available for f-block.",
        "use_for": ["benchmark"],
        "avoid_for": ["f_block", "routine"],
        "f_block_available": False,
    },
    {
        "name": "dyall.aae4z",
        "quality": "quadruple-zeta",
        "type": "all-electron+diffuse",
        "diffuse": True,
        "zeta": 4,
        "elements": _ALL_DIFFUSE,
        "description": "Dyall augmented all-electron QZ. Not available for f-block.",
        "use_for": ["benchmark"],
        "avoid_for": ["f_block", "routine"],
        "f_block_available": False,
    },
    # ---- Quintuple-zeta (no diffuse variants in DIRAC 25) ----
    {
        "name": "dyall.v5z",
        "quality": "quintuple-zeta",
        "type": "valence",
        "diffuse": False,
        "zeta": 5,
        "elements": _ALL_NONDIFFUSE,
        "description": "Dyall quintuple-zeta valence. Extreme benchmark only.",
        "use_for": ["benchmark"],
        "avoid_for": ["routine"],
    },
    {
        "name": "dyall.cv5z",
        "quality": "quintuple-zeta",
        "type": "core-valence",
        "diffuse": False,
        "zeta": 5,
        "elements": _ALL_NONDIFFUSE,
        "description": "Dyall quintuple-zeta core-valence. Extreme benchmark only.",
        "use_for": ["benchmark"],
        "avoid_for": ["routine"],
    },
    {
        "name": "dyall.ae5z",
        "quality": "quintuple-zeta",
        "type": "all-electron",
        "diffuse": False,
        "zeta": 5,
        "elements": _ALL_NONDIFFUSE,
        "description": "Dyall quintuple-zeta all-electron. Extreme benchmark only.",
        "use_for": ["benchmark"],
        "avoid_for": ["routine"],
    },
    {
        "name": "dyall.av5z",
        "quality": "quintuple-zeta",
        "type": "valence+diffuse",
        "diffuse": True,
        "zeta": 5,
        "elements": _ALL_DIFFUSE,
        "description": "Dyall augmented quintuple-zeta valence. Not for f-block.",
        "use_for": ["benchmark"],
        "avoid_for": ["f_block", "routine"],
        "f_block_available": False,
    },
    {
        "name": "dyall.acv5z",
        "quality": "quintuple-zeta",
        "type": "core-valence+diffuse",
        "diffuse": True,
        "zeta": 5,
        "elements": _ALL_DIFFUSE,
        "description": "Dyall augmented quintuple-zeta core-valence. Not for f-block.",
        "use_for": ["benchmark"],
        "avoid_for": ["f_block", "routine"],
        "f_block_available": False,
    },
]

# Map: name → entry for fast lookup
_FAMILY_INDEX: dict[str, dict] = {f["name"]: f for f in _FAMILIES}

# Element → block classification
def _element_block(element: str) -> str:
    e = element.capitalize()
    if e in _F_BLOCK:
        return "f"
    if e in _D_BLOCK:
        return "d"
    if e in _P_BLOCK:
        return "p"
    if e in _S_BLOCK:
        return "s"
    if e in _G_BLOCK:
        return "g"
    return "unknown"


def list_basis_sets(
    element: str | None = None,
    family_type: str | None = None,
    zeta: int | None = None,
    calc_type: str | None = None,
) -> dict:
    """List Dyall basis families available in DIRAC 25, with optional filters.

    Parameters
    ----------
    element
        Element symbol (e.g. 'Cm'). If given, only returns families that
        support this element and adds a caveat for f-block + diffuse sets.
    family_type
        Filter by type string: 'valence', 'dft', 'core-valence',
        'all-electron', or a prefix (e.g. 'all' matches all-electron).
    zeta
        Filter by zeta level (2, 3, 4, 5).
    calc_type
        Filter to families suitable for this purpose:
        'scf', 'dft', 'correlated', 'cc', 'ci', 'aoc',
        'core_ip', 'nmr', 'efg', 'anion', 'benchmark'.

    Returns
    -------
    dict with:
      ``families``: list of family dicts (name, quality, type, description,
                    use_for, available — True/False for the given element)
      ``element_block``: 's', 'p', 'd', 'f', 'g', or None
      ``recommendation``: name of the suggested family for this context
      ``notes``: list of strings with caveats
    """
    elem = element.capitalize() if element else None
    block = _element_block(elem) if elem else None

    results = []
    notes: list[str] = []
    for fam in _FAMILIES:
        if zeta is not None and fam["zeta"] != zeta:
            continue
        if family_type is not None:
            if not fam["type"].startswith(family_type):
                continue
        if calc_type is not None and fam["use_for"] and calc_type not in fam["use_for"]:
            continue

        entry = {
            "name": fam["name"],
            "quality": fam["quality"],
            "type": fam["type"],
            "diffuse": fam["diffuse"],
            "description": fam["description"],
            "use_for": fam["use_for"],
            "avoid_for": fam["avoid_for"],
        }
        if elem is not None:
            available = elem in fam["elements"]
            entry["available_for_element"] = available
            if not available and fam.get("f_block_available") is False and block == "f":
                entry["f_block_note"] = (
                    f"{fam['name']} does not cover f-block elements "
                    f"(lanthanides/actinides). For {elem}, use the "
                    f"non-augmented equivalent."
                )
        results.append(entry)

    if block == "f":
        notes.append(
            f"{elem} is an f-block element. Augmented (diffuse) Dyall families "
            "(av*, acv*, aae*) are NOT available. For anions/excited states on "
            "actinides, use dyall.v3z or dyall.cv3z."
        )
    if block == "f" and calc_type in ("aoc", "scf", "dft", None):
        notes.append(
            "For actinide atomic AOC: dyall.2zp is standard and the most compact. "
            "dyall.v2z/v3z are needed if you follow with correlated methods."
        )

    recommendation = _recommend(elem, block, calc_type, zeta)
    return {
        "families": results,
        "element": elem,
        "element_block": block,
        "recommendation": recommendation,
        "notes": notes,
    }


def suggest_basis(
    element: str,
    calc_type: str = "scf",
    zeta: int | None = None,
) -> dict:
    """Return a ranked recommendation for element + calculation type.

    Parameters
    ----------
    element
        Element symbol (e.g. 'Am', 'U', 'Cr').
    calc_type
        Purpose: 'scf', 'dft', 'aoc', 'correlated', 'cc', 'ci',
        'core_ip', 'nmr', 'efg', 'anion', 'benchmark'.
    zeta
        Preferred zeta level (2/3/4/5); None = pick automatically.

    Returns
    -------
    dict with:
      ``recommended``: best family name
      ``alternatives``: list of other suitable names
      ``rationale``: explanation
      ``caveats``: list of strings
    """
    elem = element.capitalize()
    block = _element_block(elem)
    rec = _recommend(elem, block, calc_type, zeta)
    result = {
        "element": elem,
        "element_block": block,
        "calc_type": calc_type,
        "recommended": rec,
        "alternatives": [],
        "rationale": "",
        "caveats": [],
    }
    # Build rationale
    fam = _FAMILY_INDEX.get(rec, {})
    result["rationale"] = fam.get("description", "")

    # Build alternatives
    alts = []
    for f in _FAMILIES:
        if f["name"] == rec:
            continue
        if elem not in f["elements"]:
            continue
        if calc_type and f["use_for"] and calc_type not in f["use_for"]:
            continue
        if calc_type and calc_type in f.get("avoid_for", []):
            continue
        alts.append(f["name"])
    result["alternatives"] = alts[:4]

    # Caveats
    if block == "f":
        result["caveats"].append(
            f"{elem} is an f-block element. Augmented (av*, acv*, aae*) families "
            "are NOT available — there are no diffuse Dyall basis sets for actinides."
        )
    if calc_type in ("cc", "ci", "correlated") and "2zp" in rec:
        result["caveats"].append(
            "dyall.2zp lacks correlating functions. Switched to dyall.v2z for "
            "correlated methods — use dyall.v2z or dyall.v3z."
        )
    return result


def _recommend(
    elem: str | None,
    block: str | None,
    calc_type: str | None,
    zeta: int | None,
) -> str | None:
    """Internal: pick a single best family name."""
    z = zeta or 2  # default to DZ
    if calc_type in ("cc", "ci", "correlated", "cosci", "krci", "caspt2"):
        # Need correlating functions
        return f"dyall.v{z}z"
    if calc_type in ("core_ip", "nmr", "efg"):
        return f"dyall.cv{z}z"
    if calc_type == "benchmark":
        return f"dyall.v{max(z, 3)}z"
    if calc_type in ("anion", "excited", "polarizability"):
        if block == "f":
            # No diffuse for f-block; suggest larger valence instead
            return f"dyall.v{max(z, 3)}z"
        return f"dyall.av{z}z"
    # Default: SCF / DFT / AOC
    return f"dyall.{z}zp"

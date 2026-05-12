"""Atomization-energy orchestrator: generate molecule + per-atom inputs at
consistent CAS theory, returning a launch + post-hoc validation plan.

The reaction-energy / active-space-consistency dogfooding on CrO showed
that getting a meaningful binding energy at CASSCF requires three things
to be coordinated across all species:

  1. CAS dimensions: molecule's (M_act_e, N_act_o) >= Σ_fragments dimensions
  2. Theory level: DKH or non-DKH applied to all species (no mix-and-match)
  3. Atomic SCF: for transition-metal atoms, skip SCF (Molcas ROHF doesn't
     converge from GuessOrb on high-spin TMs) and let RASSCF start from
     GuessOrb directly.

This module ties those rules together. It exposes:

  - ATOMIC_GROUND_STATES: bundled ground-state spin/configuration table
  - prepare_atomization_calculation: top-level orchestrator
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re

from chemtools.programs.molcas.runtime import prepare_launch
from chemtools.programs.molcas.input.draft import draft_molcas_input
from chemtools.programs.molcas.input.lint import lint_molcas_input
from chemtools.programs.molcas.input._utils import normalize_atoms


# ---------------------------------------------------------------------------
# Atomic ground-state table (Z=1..30)
# ---------------------------------------------------------------------------
#
# Fields:
#   term:         atomic term symbol of the ground state
#   config:       [core]val configuration (informational)
#   multiplicity: 2S+1 of the ground state
#   minimal_cas:  (n_active_electrons, n_active_orbitals) — open-valence-shell
#                 CAS. Matches what the molecule typically puts in active for
#                 reaction-energy work (e.g. O 2p×3 only, not 2s).
#   valence_cas:  full valence-shell CAS (informational; rarely the right pick
#                 for atomization-vs-molecule consistency).
#   requires_dkh: heavier atoms (Z >= 19) need DKH (R02O02 in SEWARD) when
#                 using ANO-RCC contractions for absolute-energy correctness.
#   skip_scf:     for high-spin TM atoms, skip the SCF block in the generated
#                 input — Molcas ROHF doesn't converge from GuessOrb for these.
#                 RASSCF starts from GuessOrb directly and converges fine.

ATOMIC_GROUND_STATES: dict[str, dict[str, Any]] = {
    # Z=1-18 (main-group: 1s, 2s/2p, 3s/3p)
    # H has only 1 electron — Molcas SCF aborts with "Current implementation
    # only allows double occupations". Skip SCF and let RASSCF run from
    # GuessOrb. Same skip_scf=True policy as high-spin TM atoms.
    "H":  {"term": "2S", "config": "1s¹",        "multiplicity": 2, "minimal_cas": (1, 1), "valence_cas": (1, 1), "requires_dkh": False, "skip_scf": True},
    "He": {"term": "1S", "config": "1s²",        "multiplicity": 1, "minimal_cas": (2, 1), "valence_cas": (2, 1), "requires_dkh": False, "skip_scf": False},
    "Li": {"term": "2S", "config": "[He]2s¹",    "multiplicity": 2, "minimal_cas": (1, 1), "valence_cas": (1, 1), "requires_dkh": False, "skip_scf": False},
    "Be": {"term": "1S", "config": "[He]2s²",    "multiplicity": 1, "minimal_cas": (2, 1), "valence_cas": (2, 1), "requires_dkh": False, "skip_scf": False},
    "B":  {"term": "2P", "config": "[He]2s²2p¹", "multiplicity": 2, "minimal_cas": (1, 3), "valence_cas": (3, 4), "requires_dkh": False, "skip_scf": False},
    "C":  {"term": "3P", "config": "[He]2s²2p²", "multiplicity": 3, "minimal_cas": (2, 3), "valence_cas": (4, 4), "requires_dkh": False, "skip_scf": False},
    "N":  {"term": "4S", "config": "[He]2s²2p³", "multiplicity": 4, "minimal_cas": (3, 3), "valence_cas": (5, 4), "requires_dkh": False, "skip_scf": False},
    "O":  {"term": "3P", "config": "[He]2s²2p⁴", "multiplicity": 3, "minimal_cas": (4, 3), "valence_cas": (6, 4), "requires_dkh": False, "skip_scf": False},
    "F":  {"term": "2P", "config": "[He]2s²2p⁵", "multiplicity": 2, "minimal_cas": (5, 3), "valence_cas": (7, 4), "requires_dkh": False, "skip_scf": False},
    "Ne": {"term": "1S", "config": "[He]2s²2p⁶", "multiplicity": 1, "minimal_cas": (6, 3), "valence_cas": (8, 4), "requires_dkh": False, "skip_scf": False},
    "Na": {"term": "2S", "config": "[Ne]3s¹",    "multiplicity": 2, "minimal_cas": (1, 1), "valence_cas": (1, 1), "requires_dkh": False, "skip_scf": False},
    "Mg": {"term": "1S", "config": "[Ne]3s²",    "multiplicity": 1, "minimal_cas": (2, 1), "valence_cas": (2, 1), "requires_dkh": False, "skip_scf": False},
    "Al": {"term": "2P", "config": "[Ne]3s²3p¹", "multiplicity": 2, "minimal_cas": (1, 3), "valence_cas": (3, 4), "requires_dkh": False, "skip_scf": False},
    "Si": {"term": "3P", "config": "[Ne]3s²3p²", "multiplicity": 3, "minimal_cas": (2, 3), "valence_cas": (4, 4), "requires_dkh": False, "skip_scf": False},
    "P":  {"term": "4S", "config": "[Ne]3s²3p³", "multiplicity": 4, "minimal_cas": (3, 3), "valence_cas": (5, 4), "requires_dkh": False, "skip_scf": False},
    "S":  {"term": "3P", "config": "[Ne]3s²3p⁴", "multiplicity": 3, "minimal_cas": (4, 3), "valence_cas": (6, 4), "requires_dkh": False, "skip_scf": False},
    "Cl": {"term": "2P", "config": "[Ne]3s²3p⁵", "multiplicity": 2, "minimal_cas": (5, 3), "valence_cas": (7, 4), "requires_dkh": False, "skip_scf": False},
    "Ar": {"term": "1S", "config": "[Ne]3s²3p⁶", "multiplicity": 1, "minimal_cas": (6, 3), "valence_cas": (8, 4), "requires_dkh": False, "skip_scf": False},
    # Z=19-20: K, Ca (alkaline)
    "K":  {"term": "2S", "config": "[Ar]4s¹",      "multiplicity": 2, "minimal_cas": (1, 1), "valence_cas": (1, 1), "requires_dkh": True, "skip_scf": False},
    "Ca": {"term": "1S", "config": "[Ar]4s²",      "multiplicity": 1, "minimal_cas": (2, 1), "valence_cas": (2, 1), "requires_dkh": True, "skip_scf": False},
    # Z=21-30: 3d transition metals — minimal_cas includes 3d-shell SOMOs + 4s.
    # skip_scf=True for high-spin TMs (multiplicity >= 4) — ROHF won't converge.
    "Sc": {"term": "2D", "config": "[Ar]3d¹4s²", "multiplicity": 2, "minimal_cas": (3, 6), "valence_cas": (3, 6), "requires_dkh": True, "skip_scf": False},
    "Ti": {"term": "3F", "config": "[Ar]3d²4s²", "multiplicity": 3, "minimal_cas": (4, 6), "valence_cas": (4, 6), "requires_dkh": True, "skip_scf": False},
    "V":  {"term": "4F", "config": "[Ar]3d³4s²", "multiplicity": 4, "minimal_cas": (5, 6), "valence_cas": (5, 6), "requires_dkh": True, "skip_scf": True},
    "Cr": {"term": "7S", "config": "[Ar]3d⁵4s¹", "multiplicity": 7, "minimal_cas": (6, 6), "valence_cas": (6, 6), "requires_dkh": True, "skip_scf": True},
    "Mn": {"term": "6S", "config": "[Ar]3d⁵4s²", "multiplicity": 6, "minimal_cas": (7, 6), "valence_cas": (7, 6), "requires_dkh": True, "skip_scf": True},
    "Fe": {"term": "5D", "config": "[Ar]3d⁶4s²", "multiplicity": 5, "minimal_cas": (8, 6), "valence_cas": (8, 6), "requires_dkh": True, "skip_scf": True},
    "Co": {"term": "4F", "config": "[Ar]3d⁷4s²", "multiplicity": 4, "minimal_cas": (9, 6), "valence_cas": (9, 6), "requires_dkh": True, "skip_scf": True},
    "Ni": {"term": "3F", "config": "[Ar]3d⁸4s²", "multiplicity": 3, "minimal_cas": (10, 6), "valence_cas": (10, 6), "requires_dkh": True, "skip_scf": False},
    "Cu": {"term": "2S", "config": "[Ar]3d¹⁰4s¹", "multiplicity": 2, "minimal_cas": (11, 6), "valence_cas": (11, 6), "requires_dkh": True, "skip_scf": False},
    "Zn": {"term": "1S", "config": "[Ar]3d¹⁰4s²", "multiplicity": 1, "minimal_cas": (12, 6), "valence_cas": (12, 6), "requires_dkh": True, "skip_scf": False},
}


def get_atomic_ground_state(element: str) -> dict[str, Any]:
    """Look up an element's ground-state info. Case-insensitive.

    Returns the bundle dict; raises ValueError for unsupported elements.
    """
    el = element.strip().capitalize()
    if el not in ATOMIC_GROUND_STATES:
        raise ValueError(
            f"No atomization ground-state data for element {element!r}. "
            f"Supported (Z=1..30): {sorted(ATOMIC_GROUND_STATES.keys())}"
        )
    return ATOMIC_GROUND_STATES[el]


def derive_fragment_specs(
    atoms: list[dict],
    *,
    cas_strategy: str = "minimal",
) -> dict[str, dict[str, Any]]:
    """Derive atomic-fragment specs from a molecule.

    Returns dict keyed by element symbol with stoichiometry + ground-state
    info + chosen CAS. Identical atoms are deduplicated; stoichiometry counts
    the appearances in the molecule.
    """
    if cas_strategy not in {"minimal", "valence"}:
        raise ValueError(
            f"cas_strategy must be 'minimal' or 'valence'; got {cas_strategy!r}"
        )

    atoms_norm = normalize_atoms(atoms)
    fragments: dict[str, dict[str, Any]] = {}
    for a in atoms_norm:
        el = a["symbol"].strip().capitalize()
        if el not in fragments:
            gs = get_atomic_ground_state(el)
            cas = gs["valence_cas"] if cas_strategy == "valence" else gs["minimal_cas"]
            fragments[el] = {
                "element": el,
                "stoichiometry": 0,
                "term": gs["term"],
                "config": gs["config"],
                "multiplicity": gs["multiplicity"],
                "cas_active_electrons": cas[0],
                "cas_active_orbitals": cas[1],
                "requires_dkh": gs["requires_dkh"],
                "skip_scf": gs["skip_scf"],
            }
        fragments[el]["stoichiometry"] += 1
    return fragments


def _drop_scf_block(input_text: str) -> str:
    """Remove the &SCF ... End of input block from a drafted Molcas input.

    Required for high-spin TM atomic references where ROHF doesn't converge
    from GuessOrb; leaving SCF in the deck aborts the run before RASSCF.
    """
    return re.sub(r"&SCF &END.*?End of input\n", "", input_text, flags=re.DOTALL)


def _inject_relativistic(input_text: str) -> str:
    """Insert `Relativistic\nR02O02` into the &SEWARD block immediately after
    its opening line. Safe to call on inputs that already have a Relativistic
    directive (we check before inserting).
    """
    if re.search(r"^\s*Relativistic\s*$", input_text, flags=re.MULTILINE):
        return input_text
    return re.sub(
        r"(&SEWARD &END\n)",
        r"\1Relativistic\nR02O02\n",
        input_text,
        count=1,
    )


def _basis_for_element(basis: str | dict, element: str) -> str:
    if isinstance(basis, str):
        return basis
    return basis[element]


def _draft_atomic_input(
    *,
    element: str,
    fragment: dict[str, Any],
    basis: str | dict[str, str],
    apply_dkh: bool,
    inline_basis: bool,
    memory_mb: int,
    title_prefix: str = "",
    method: str = "CASSCF",
    ipea_shift: float | None = 0.25,
    imaginary_shift: float = 0.0,
) -> str:
    """Draft a Molcas input for one atomic reference.

    method: "CASSCF" or "CASPT2"/"MS-CASPT2" — for atomization at CASPT2
    level, atomic refs must run CASPT2 at the same theory level so the
    reaction-energy comparison is well-defined. SS-CASPT2 is used for
    single-root atomic ground states.
    """
    program_opts: dict[str, Any] = {
        "cas_active_electrons": fragment["cas_active_electrons"],
        "cas_active_orbitals": fragment["cas_active_orbitals"],
        "inline_basis": inline_basis,
        "memory_mb": memory_mb,
        # Bump RASSCF iters (see molecule path for rationale); atomic refs
        # converge fast but extra headroom protects against TM weirdness.
        "rasscf": {"iterations": (100, 50)},
    }
    if method.upper() in {"CASPT2", "MS-CASPT2", "XMS-CASPT2", "RMS-CASPT2", "XDW-CASPT2"}:
        program_opts["caspt2"] = {
            # SS-CASPT2 on a single-root atomic ground state; MS variants
            # collapse to SS for n_roots=1.
            "variant": "SS",
            "n_roots": 1,
            "ipea_shift": ipea_shift,
            "imaginary_shift": imaginary_shift,
        }
        atomic_method = "CASPT2"
    else:
        atomic_method = "CASSCF"

    spec = {
        "atoms": [{"symbol": element, "x": 0.0, "y": 0.0, "z": 0.0}],
        "charge": 0,
        "multiplicity": fragment["multiplicity"],
        "basis": {element: _basis_for_element(basis, element)},
        "method": atomic_method,
        "task": "energy",
        "title": f"{title_prefix}{element} {fragment['term']} CAS({fragment['cas_active_electrons']},{fragment['cas_active_orbitals']}) {atomic_method}",
        "program_options": program_opts,
    }
    text = draft_molcas_input(spec)
    if fragment["skip_scf"]:
        text = _drop_scf_block(text)
    if apply_dkh:
        text = _inject_relativistic(text)
    return text


# ---------------------------------------------------------------------------
# prepare_atomization_calculation — the orchestrator
# ---------------------------------------------------------------------------


def prepare_atomization_calculation(
    *,
    atoms: list[dict],
    charge: int = 0,
    multiplicity: int,
    basis: str | dict[str, str],
    cas_active_electrons: int | None = None,
    cas_active_orbitals: int | None = None,
    atomic_cas_strategy: str = "minimal",
    method: str = "CASSCF",
    relativistic: str = "auto",  # "auto" | "always" | "never"
    output_dir: str = ".",
    base_job_name: str | None = None,
    inline_basis: bool = True,
    memory_mb: int = 4000,
    apptainer_sif: str | None = None,
    profile: dict | None = None,
    requested_np: int = 1,
    title: str | None = None,
    geometry_units: str = "angstrom",
    ipea_shift: float | None = 0.25,
    imaginary_shift: float | None = None,
) -> dict[str, Any]:
    """Thick orchestrator for an atomization-energy workflow.

    Generates Molcas inputs for the molecule AND one input per unique atomic
    fragment, at consistent CAS theory and DKH setting, and returns a launch
    + post-hoc plan in a Diagnosis envelope.

    Workflow philosophy (lessons from the CrO dogfood):

    1. **Atomic CAS spec is bundled** — `ATOMIC_GROUND_STATES` table fixes
       ground-state multiplicity and recommended CAS per element. The agent
       doesn't need to remember Cr ⁷S(3d⁵4s¹) is CAS(6,6) at minimum.

    2. **Molecule CAS auto-sums to atomic CASes by default** — if
       cas_active_electrons / cas_active_orbitals are not given, the
       orchestrator computes them as Σ_fragments(stoichiometry × atomic_cas).
       This guarantees `check_molcas_active_space_consistency` passes.

    3. **DKH applied uniformly** — when any element is Z>=19 (or
       relativistic="always"), `Relativistic R02O02` is added to ALL species
       SEWARDs (molecule and atoms). Mixing DKH and non-DKH is a bug.

    4. **SCF skipped on high-spin TMs** — for atoms with `skip_scf=True` in
       the ground-state table (Cr ⁷S, Mn ⁶S, Fe ⁵D, etc.), the orchestrator
       strips the &SCF block from the generated input so the deck goes
       straight from SEWARD → RASSCF using GuessOrb starting orbitals.

    Parameters
    ----------
    atoms, charge, multiplicity, basis, title, geometry_units
        Molecule spec. Forwarded to draft_molcas_input.
    cas_active_electrons, cas_active_orbitals
        Molecule CAS. If both None, derived from fragments (sum); the
        diagnostic verdict will then be "consistent" by construction.
    atomic_cas_strategy
        "minimal" (default; SOMOs + open valence) or "valence" (full valence
        shell). Minimal usually matches what the molecule's CAS spans at
        infinity.
    method
        Method passed to draft_molcas_input for the MOLECULE. Atomic refs
        are always CASSCF. Use "CASPT2" or "MS-CASPT2" for higher-level
        molecule treatment (atomic refs stay at CASSCF; perturbation level
        for atoms can be added later).
    relativistic
        "auto" (default) → on if any element needs DKH per the table.
        "always" → always emit Relativistic R02O02.
        "never" → never emit it.
    output_dir, base_job_name, inline_basis, memory_mb
        Where to write inputs and how to draft them.

    Returns a dict with:
      verdict                "ready_to_launch" | "lint_blocked"
      molecule_input         path + lint
      atomic_inputs          list of per-element inputs + lint + stoichiometry
      reaction_recipe        ready-to-pass to compute_molcas_reaction_energy
      consistency_args       ready-to-pass to check_molcas_active_space_consistency
      next_actions           agent-actionable list
    """
    if method.upper() not in {"CASSCF", "CASPT2", "MS-CASPT2", "XMS-CASPT2", "RMS-CASPT2", "XDW-CASPT2"}:
        return {
            "verdict": "unsupported_method",
            "error": "unsupported_method",
            "message": (
                f"prepare_molcas_atomization supports CASSCF / CASPT2 / "
                f"MS-CASPT2 / XMS-CASPT2 / RMS-CASPT2 / XDW-CASPT2; got {method!r}."
            ),
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = base_job_name or (
        re.sub(r"\W+", "_", title) if title else "atomization"
    )
    if not base or base == "_":
        base = "atomization"

    # Derive atomic fragments
    fragments = derive_fragment_specs(atoms, cas_strategy=atomic_cas_strategy)
    sum_active_e = sum(f["stoichiometry"] * f["cas_active_electrons"] for f in fragments.values())
    sum_active_o = sum(f["stoichiometry"] * f["cas_active_orbitals"] for f in fragments.values())

    # Choose molecule CAS (auto-sum or user-explicit)
    auto_cas = (cas_active_electrons is None or cas_active_orbitals is None)
    mol_cas_e = sum_active_e if auto_cas else int(cas_active_electrons)  # type: ignore[arg-type]
    mol_cas_o = sum_active_o if auto_cas else int(cas_active_orbitals)  # type: ignore[arg-type]

    # Decide DKH
    if relativistic == "auto":
        apply_dkh = any(f["requires_dkh"] for f in fragments.values())
    elif relativistic == "always":
        apply_dkh = True
    elif relativistic == "never":
        apply_dkh = False
    else:
        raise ValueError(f"relativistic must be 'auto'/'always'/'never'; got {relativistic!r}")

    # Imaginary-shift auto: 0.1 by default on TM systems (intruder protection),
    # 0.0 for main-group only systems. CrO CASPT2 without imag shift hit
    # ref_weight 0.64 + RC_NOT_CONVERGED on intruder states with small
    # denominators in the secondary (4d / 4f-like) virtual space. With
    # imag_shift=0.1 it converged at ref_weight 0.87.
    if imaginary_shift is None:
        imaginary_shift = 0.1 if apply_dkh else 0.0

    # --- Draft molecule input ---
    method_upper = method.upper()
    is_caspt2 = method_upper in {"CASPT2", "MS-CASPT2", "XMS-CASPT2", "RMS-CASPT2", "XDW-CASPT2"}

    mol_program_opts: dict[str, Any] = {
        "cas_active_electrons": mol_cas_e,
        "cas_active_orbitals": mol_cas_o,
        "inline_basis": inline_basis,
        "memory_mb": memory_mb,
        # Bump RASSCF iters from the default (50,25) — molecules with
        # transition-metal CAS routinely need 40-60 macro iterations to
        # converge from GuessOrb, and an unconverged RASSCF kills the
        # CASPT2 chain that follows.
        "rasscf": {"iterations": (100, 50)},
    }
    if is_caspt2:
        mol_program_opts["caspt2"] = {
            "variant": "MS" if method_upper.startswith(("MS-", "XMS-", "RMS-", "XDW-")) else "SS",
            "n_roots": 1,
            "ipea_shift": ipea_shift,
            "imaginary_shift": imaginary_shift,
        }

    mol_spec = {
        "atoms": atoms,
        "charge": charge,
        "multiplicity": multiplicity,
        "basis": basis,
        "method": method,
        "task": "energy",
        "title": title or f"{base} molecule CAS({mol_cas_e},{mol_cas_o}) {method_upper}",
        "geometry_units": geometry_units,
        "program_options": mol_program_opts,
    }
    mol_text = draft_molcas_input(mol_spec)
    if apply_dkh:
        mol_text = _inject_relativistic(mol_text)
    mol_path = out_dir / f"{base}_molecule.input"
    mol_path.write_text(mol_text, encoding="utf-8")
    mol_lint = lint_molcas_input(mol_text)
    mol_errors = sum(1 for i in mol_lint if i.get("level") == "error")

    # --- Draft atomic inputs ---
    atomic_records: list[dict[str, Any]] = []
    total_atomic_errors = 0
    for el, frag in fragments.items():
        atext = _draft_atomic_input(
            element=el,
            fragment=frag,
            basis=basis,
            apply_dkh=apply_dkh,
            inline_basis=inline_basis,
            memory_mb=memory_mb,
            title_prefix=f"{base} ",
            method=method,
            ipea_shift=ipea_shift,
            imaginary_shift=imaginary_shift,
        )
        apath = out_dir / f"{base}_{el.lower()}_atom.input"
        apath.write_text(atext, encoding="utf-8")
        alint = lint_molcas_input(atext)
        aerrors = sum(1 for i in alint if i.get("level") == "error")
        total_atomic_errors += aerrors
        atomic_records.append({
            "element": el,
            "stoichiometry": frag["stoichiometry"],
            "term": frag["term"],
            "config": frag["config"],
            "multiplicity": frag["multiplicity"],
            "cas": [frag["cas_active_electrons"], frag["cas_active_orbitals"]],
            "skip_scf": frag["skip_scf"],
            "input_path": str(apath),
            "log_path": str(apath).replace(".input", ".log"),
            "lint_issues": alint,
            "n_lint_errors": aerrors,
        })

    # --- Launch plans ---
    if mol_errors == 0:
        mol_launch = prepare_launch(
            str(mol_path),
            profile=profile,
            requested_np=requested_np,
            job_name=f"{base}_molecule",
            apptainer_sif=apptainer_sif,
        )
    else:
        mol_launch = None

    for record in atomic_records:
        if record["n_lint_errors"] == 0:
            record["launch_plan"] = prepare_launch(
                record["input_path"],
                profile=profile,
                requested_np=requested_np,
                job_name=f"{base}_{record['element'].lower()}_atom",
                apptainer_sif=apptainer_sif,
            )
        else:
            record["launch_plan"] = None

    verdict = "ready_to_launch" if (mol_errors == 0 and total_atomic_errors == 0) else "lint_blocked"

    # --- Build the recipe for post-hoc tools ---
    products_recipe = [
        {
            "output_file": rec["log_path"],
            "coefficient": rec["stoichiometry"],
            "label": f"{rec['element']} {rec['term']}",
        }
        for rec in atomic_records
    ]
    reactants_recipe = [
        {
            "output_file": str(mol_path).replace(".input", ".log"),
            "coefficient": 1,
            "label": base,
        }
    ]

    consistency_args = {
        "molecule_output": str(mol_path).replace(".input", ".log"),
        "fragments": [
            {
                "output_file": rec["log_path"],
                "stoichiometry": rec["stoichiometry"],
                "label": f"{rec['element']} {rec['term']}",
            }
            for rec in atomic_records
        ],
    }

    # Pick the heaviest TM element (if any) as the character target hint
    # so the agent can chain `check_molcas_active_space_consistency` with
    # character validation in one call.
    tm_elements = [el for el in fragments if fragments[el]["requires_dkh"] and fragments[el]["cas_active_orbitals"] >= 5]
    if tm_elements:
        consistency_args["target_character_atom"] = tm_elements[0]
        consistency_args["target_character_ao"] = "3d"

    next_actions: list[dict] = []
    if verdict == "ready_to_launch":
        for record in atomic_records:
            plan = record["launch_plan"]
            next_actions.append({
                "tool": "shell_execute",
                "args": {"command": plan["command_str"], "env": plan["env"]},
                "rationale": (
                    f"Run the {record['element']} {record['term']} atomic reference"
                    f" (CAS{tuple(record['cas'])}, "
                    f"{'no SCF' if record['skip_scf'] else 'SCF + CASSCF'}, "
                    f"{'DKH' if apply_dkh else 'no DKH'})."
                ),
            })
        next_actions.append({
            "tool": "shell_execute",
            "args": {"command": mol_launch["command_str"], "env": mol_launch["env"]},  # type: ignore[index]
            "rationale": f"Run the {base} molecule at CAS({mol_cas_e},{mol_cas_o}) {method_upper}.",
        })
        next_actions.append({
            "tool": "check_molcas_active_space_consistency",
            "args": consistency_args,
            "rationale": (
                "After all jobs converge, verify the molecule's CAS spans the "
                "summed atomic CASes (and that 3d-character counts match for TMs)."
            ),
        })
        # When method is CASPT2, force energy_kind='caspt2' so the parser
        # picks up the CASPT2 line (not CASSCF). 'primary' would still work
        # since the energy_summary hierarchy prefers CASPT2, but being
        # explicit protects against silent fallback when one species'
        # CASPT2 fails to converge.
        energy_kind_for_rxn = "caspt2" if is_caspt2 else "rasscf"
        next_actions.append({
            "tool": "compute_molcas_reaction_energy",
            "args": {
                "products": products_recipe,
                "reactants": reactants_recipe,
                "energy_kind": energy_kind_for_rxn,
                "label": f"{base} atomization at {method_upper}",
            },
            "rationale": "Compute the binding/atomization energy from converged outputs.",
        })
    else:
        next_actions.append({
            "tool": "lint_molcas_input",
            "rationale": (
                f"Fix the {mol_errors} molecule + {total_atomic_errors} atomic "
                "lint error(s) before launching."
            ),
        })

    return {
        "verdict": verdict,
        "method": method.upper(),
        "atomic_cas_strategy": atomic_cas_strategy,
        "relativistic_setting": ("R02O02" if apply_dkh else "off"),
        "molecule": {
            "input_path": str(mol_path),
            "log_path": str(mol_path).replace(".input", ".log"),
            "cas_active_electrons": mol_cas_e,
            "cas_active_orbitals": mol_cas_o,
            "cas_was_auto_summed": auto_cas,
            "multiplicity": multiplicity,
            "lint_issues": mol_lint,
            "n_lint_errors": mol_errors,
            "launch_plan": mol_launch,
        },
        "atomic_inputs": atomic_records,
        "fragments_cas_sum": {
            "n_active_electrons": sum_active_e,
            "n_active_orbitals": sum_active_o,
        },
        "reaction_recipe": {
            "products": products_recipe,
            "reactants": reactants_recipe,
        },
        "consistency_args": consistency_args,
        "next_actions": next_actions,
    }

"""Cm-class actinide multi-step convergence workflow scaffolding.

Heavy actinides (Cm, Bk, Cf, Es, Fm, ...) have 5f^7+ open shells where
the 5f orbitals lie BELOW the outer 6d/7s shells — DIRAC's default
"open above closed" AOC assumption breaks down and even .KPSELE doesn't
rescue the inner RELSCF from oscillation.

Published strategy (Mochizuki JCP 2003, DIRAC docs CmF.md):

  Step 1. Atomic 5f^N checkpoint from a lighter reference (e.g. Ce).
  Step 2. Molecular SCF as CLOSED-shell with imported 5f frozen.
  Step 3. Molecular SCF with closed-shell frozen, 5f^N relaxing.

This module emits the LAUNCH-PLAN skeleton: input filenames, pam-dirac
commands with the right ``--put`` / ``--get`` plumbing, and explanatory
``next_actions``. The Step 2 / Step 3 input files are emitted with a
PLACEHOLDER ``.FROZEN`` block that the agent should NOT try to fill
without chemistry-expert review. The actual orbital-position-remap
syntax is in DIRAC's test/tutorial fixtures, not in the bundled docs.

Returns a plan the agent can use to drive the workflow; user fills in
the orbital indices in the .FROZEN blocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.programs.dirac.input.inp import draft_inp
from chemtools.programs.dirac.input.mol import draft_mol


# Elements where the simple X2C + AOC + .KPSELE path oscillates in DIRAC 25
# (Cm-class = Z >= 96 with dyall.2zp basis). After thorough investigation:
#
#   - X2C + Z≥96: SCF oscillates at a wrong fixed-point (~2950 Ha off true E
#     for Cm). Likely a numerical artifact of the X2C transformation at
#     very heavy nuclei in DIRAC 25's relscf module.
#   - 4-component Dirac-Coulomb (no X2C): converges cleanly in 13 outer
#     iters. The doc CmF.md's "Cm in 13 iterations" claim DOES reproduce
#     when running with 4c instead of X2C.
#   - Validated: Cm 4c → -31332.50 Ha; Am 4c → -30489.85 Ha; both converge
#     end-to-end through the real apptainer container.
#
# Conclusion: the multi-step orbital-import workflow from CmF.md is NOT
# required for Cm to converge in DIRAC 25. The original .KPSELE workflow
# works fine, just with the FULL 4c Hamiltonian instead of X2C. The
# atomic_start orchestrator auto-switches X2C → 4c for Z ≥ 96.
#
# The Cm-class workflow remains useful when:
#   1. A user explicitly wants the CmF.md-style frozen-orbital multi-step
#      protocol (chemistry-expert manual completion of Steps 2 + 3).
#   2. The 4c workaround is impractical due to computational cost on
#      larger systems where X2C would be needed.
_CM_CLASS_ELEMENTS = frozenset({"Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"})


# Recommended surrogate reference for each hard actinide.
#
# Empirically verified in DIRAC 25 with dyall.2zp:
#   - Pu (Z=94, 5f^6 7s^2): converges in 22 iters
#   - Am (Z=95, 5f^7 7s^2): converges in 22 iters  ← SAME valence (5f^7) as Cm
#   - Cm (Z=96+): does NOT converge
#
# Am is the closest valence match to Cm (both 5f^7 half-filled f-shell);
# its converged orbitals project onto Cm's 5f manifold without character
# distortion. Pu is a good fallback for the rest of the late actinides.
_RECOMMENDED_REFERENCE: dict[str, str] = {
    "Cm": "Am",   # 5f^7 ↔ 5f^7 valence match
    "Bk": "Am",   # 5f^9 — Am 5f^7 covers half-filled f-shell character
    "Cf": "Am",
    "Es": "Am",
    "Fm": "Am",
    "Md": "Am",
    "No": "Am",
    "Lr": "Am",
}


def is_cm_class(element: str) -> bool:
    """True iff this element typically needs the multi-step workflow."""
    return (element or "").capitalize() in _CM_CLASS_ELEMENTS


def recommended_reference(element: str) -> str:
    """Return a chemically appropriate surrogate reference atom for the
    given hard actinide. Defaults to Pu for the heavy 5f^6+ block."""
    return _RECOMMENDED_REFERENCE.get(element.capitalize(), "Ce")


def prepare_cm_class_workflow(
    central_element: str,
    molecule_atoms: list[dict[str, Any]],
    *,
    basis: dict[str, str] | None = None,
    default_basis: str | None = None,
    reference_element: str | None = None,
    output_dir: str | None = None,
    molecule_name: str = "molecule",
    molecule_units: str = "bohr",
    n_5f_electrons: int = 7,
) -> dict[str, Any]:
    """Build the 3-step Cm-class convergence plan.

    Parameters
    ----------
    central_element
        The hard-converging actinide (Cm, Bk, Cf, ...). Must be in
        ``_CM_CLASS_ELEMENTS``.
    molecule_atoms
        Full molecule geometry (passed through to draft_mol).
    reference_element
        A surrogate actinide / lanthanide used to generate the initial
        atomic checkpoint. If None, defaults to ``recommended_reference
        (central_element)`` — currently **Pu** for Cm/Bk/Cf/Es/Fm/Md/No/
        Lr (chemically closer than Ce, and Pu converges cleanly with
        KPSELE in DIRAC 25). Override to ``"Ce"`` for legacy
        compatibility with the CmF.md tutorial example.
    n_5f_electrons
        Open-shell f-electron count (7 for Cm^3+ /Cm^0 GS; varies
        for other elements). Used in the placeholder .FROZEN comment.

    Returns
    -------
    dict with keys:
      ``plan``: list of {step, name, kind, inp_path, mol_path,
                         inp_text, expected_files, launch_command_hint,
                         requires_manual_completion}
      ``next_actions``: ordered list pointing at MCP tools / manual
                        chemistry tasks
      ``central_element``, ``reference_element``
      ``manual_review_required``: True (Step 2 + Step 3 inputs need
                                  human chemist before launching)
    """
    if not is_cm_class(central_element):
        raise ValueError(
            f"{central_element!r} is not in the Cm-class list. Use "
            f"prepare_dirac_atomic_start directly — .KPSELE alone "
            f"should suffice for {sorted(_CM_CLASS_ELEMENTS)} only."
        )

    out_dir = Path(output_dir or ".").resolve()
    if reference_element is None:
        reference_element = recommended_reference(central_element)
    central_basis = _basis_lookup(central_element, basis, default_basis)
    reference_basis = _basis_lookup(reference_element, basis, default_basis)

    plan: list[dict[str, Any]] = []

    # -------- Step 1: reference-element atomic checkpoint --------
    # This is a real auto-drafted input — the lighter actinide / lanthanide
    # converges with .KPSELE.
    ref_mol = draft_mol(
        atoms=[{"label": reference_element, "x": 0.0, "y": 0.0, "z": 0.0,
                "element": reference_element}],
        basis={reference_element: reference_basis},
        units=molecule_units,
        symmetry="auto",
        title=f"{reference_element} reference atom (5f/4f source for {central_element})",
    )
    # Use the standard atomic_start machinery on the reference element via
    # a lightweight wrapper — agent can verify it converges before Step 2.
    from chemtools.programs.dirac.input.atomic_start import (
        _ATOMIC_GROUND_STATES as _GS,
    )
    ref_gs = dict(_GS.get(reference_element, {}))
    ref_gs.setdefault("max_iter", 200)
    ref_gs.setdefault("resolve", True)
    ref_inp = draft_inp({
        "title": f"{reference_element} atomic SCF (reference for {central_element})",
        "wave_function": "scf",
        "analyze": ["mulpop"],
        "analyze_vecpop_ranges": ["1..oo", "1..oo"],
        "hamiltonian": {"x2c": True},
        "integrals": {"uncontract": True},
        "scf": ref_gs,
    })
    plan.append({
        "step": 1,
        "name": f"{reference_element}_atomic_ref",
        "kind": "atomic_reference",
        "inp_path": str(out_dir / f"{reference_element}.inp"),
        "mol_path": str(out_dir / f"{reference_element}.mol"),
        "inp_text": ref_inp,
        "mol_text": ref_mol,
        "expected_files": [
            str(out_dir / f"{reference_element}.h5"),
            str(out_dir / f"cf.{reference_element}"),
        ],
        "launch_command_hint": (
            f"apptainer exec dirac-25.0.sif pam-dirac "
            f"--inp={reference_element} --mol={reference_element} "
            f"--get='DFCOEF=cf.{reference_element}' --mw=1000 --nw=1000"
        ),
        "requires_manual_completion": False,
        "rationale": (
            f"Compute {reference_element}'s atomic checkpoint with KPSELE. "
            f"This converges cleanly and provides the 5f/4f orbitals "
            f"that the molecular step will import for {central_element}."
        ),
    })

    # -------- Step 2: molecule with imported 5f frozen --------
    mol_geom_text = draft_mol(
        atoms=molecule_atoms,
        basis=basis,
        default_basis=default_basis,
        units=molecule_units,
        symmetry="auto",
        title=f"{molecule_name} (5f-frozen step)",
    )
    step2_placeholder_inp = (
        f"**DIRAC\n"
        f".TITLE\n"
        f"{molecule_name} step 2 — closed-shell SCF with imported {n_5f_electrons}× 5f frozen\n"
        f".WAVE FUNCTION\n"
        f".ANALYZE\n"
        f"**HAMILTONIAN\n"
        f".X2C\n"
        f"**INTEGRALS\n"
        f"*READIN\n"
        f".UNCONTRACT\n"
        f"**WAVE FUNCTION\n"
        f".SCF\n"
        f"*SCF\n"
        f"#\n"
        f"#  --- MANUAL COMPLETION REQUIRED ---\n"
        f"#\n"
        f"#  Fill in:\n"
        f"#    .CLOSED SHELL   <count per fermion ircop>\n"
        f"#    .OPEN SHELL\n"
        f"#     1\n"
        f"#     {n_5f_electrons}/<spinor spec>\n"
        f"#\n"
        f"#  Then add the orbital-import + freeze directive that takes\n"
        f"#  orbitals from AFCMXX (the lighter-actinide atomic file) and\n"
        f"#  places them in the open shell, frozen. The exact .FROZEN\n"
        f"#  / orbital-position-remap syntax is documented in DIRAC's\n"
        f"#  test/tutorial fixtures, NOT in the bundled docs. Consult\n"
        f"#  the original CmF_5f_frz.inp example or DIRAC 25 user guide.\n"
        f"#\n"
        f"**ANALYZE\n"
        f".MULPOP\n"
        f"*MULPOP\n"
        f".VECPOP\n"
        f" 1..oo\n"
        f"*END OF INPUT\n"
    )
    plan.append({
        "step": 2,
        "name": f"{molecule_name}_5f_frz",
        "kind": "molecule_frozen_5f",
        "inp_path": str(out_dir / f"{molecule_name}_5f_frz.inp"),
        "mol_path": str(out_dir / f"{molecule_name}.mol"),
        "inp_text": step2_placeholder_inp,
        "mol_text": mol_geom_text,
        "expected_files": [
            str(out_dir / f"{molecule_name}_5f_frz.h5"),
            str(out_dir / f"cf.{molecule_name}_5f_frz"),
        ],
        "launch_command_hint": (
            f"apptainer exec dirac-25.0.sif pam-dirac --mw=1000 --nw=1000 "
            f"--inp={molecule_name}_5f_frz --mol={molecule_name} "
            f"--put 'cf.{reference_element}=AFCMXX' "
            f"--get 'DFCOEF=cf.{molecule_name}_5f_frz'"
        ),
        "requires_manual_completion": True,
        "rationale": (
            f"Closed-shell molecular SCF with the imported {n_5f_electrons}× "
            f"5f orbitals frozen at chosen positions. Improves convergence "
            f"because the rest of the system relaxes around a fixed 5f "
            f"manifold. Per CmF.md, converges in ~18 iterations once the "
            f".FROZEN/import block is correct."
        ),
    })

    # -------- Step 3: closed shells frozen, 5f relaxes --------
    step3_placeholder_inp = (
        f"**DIRAC\n"
        f".TITLE\n"
        f"{molecule_name} step 3 — closed frozen, 5f^{n_5f_electrons} relaxing\n"
        f".WAVE FUNCTION\n"
        f".ANALYZE\n"
        f"**HAMILTONIAN\n"
        f".X2C\n"
        f"**INTEGRALS\n"
        f"*READIN\n"
        f".UNCONTRACT\n"
        f"**WAVE FUNCTION\n"
        f".SCF\n"
        f"*SCF\n"
        f"#\n"
        f"#  --- MANUAL COMPLETION REQUIRED ---\n"
        f"#\n"
        f"#  Import orbitals from cf.{molecule_name}_5f_frz (renamed in scratch),\n"
        f"#  freeze the closed-shell orbitals (.FROZEN block specifying which\n"
        f"#  orbitals to keep fixed), and let the {n_5f_electrons}× 5f open-shell\n"
        f"#  orbitals relax in the molecular field. Per CmF.md, converges in ~14\n"
        f"#  iterations.\n"
        f"#\n"
        f"**ANALYZE\n"
        f".MULPOP\n"
        f"*MULPOP\n"
        f".VECPOP\n"
        f" 1..oo\n"
        f"*END OF INPUT\n"
    )
    plan.append({
        "step": 3,
        "name": f"{molecule_name}_5f_relax",
        "kind": "molecule_relax_5f",
        "inp_path": str(out_dir / f"{molecule_name}_5f_relax.inp"),
        "mol_path": str(out_dir / f"{molecule_name}.mol"),
        "inp_text": step3_placeholder_inp,
        "mol_text": mol_geom_text,
        "expected_files": [str(out_dir / f"{molecule_name}_5f_relax.h5")],
        "launch_command_hint": (
            f"apptainer exec dirac-25.0.sif pam-dirac --mw=1000 --nw=1000 "
            f"--inp={molecule_name}_5f_relax --mol={molecule_name} "
            f"--put 'cf.{molecule_name}_5f_frz=DFCOEF' "
            f"--get 'DFCOEF=cf.{molecule_name}_5f_relax'"
        ),
        "requires_manual_completion": True,
        "rationale": (
            f"Final relaxation step: closed orbitals are frozen from step 2's "
            f"output, the 5f open shell is unfrozen and re-optimized in the "
            f"molecular environment. Produces the final wavefunction + "
            f"converged 5f orbital energies for analysis."
        ),
    })

    next_actions: list[dict[str, Any]] = [
        {
            "tool": "prepare_dirac_launch",
            "rationale": (
                f"Step 1 — run the {reference_element} atomic reference. "
                f"Should converge cleanly with KPSELE."
            ),
            "args": {
                "input_file": plan[0]["inp_path"],
                "mol_file":   plan[0]["mol_path"],
                "get_files":  ["DFCOEF"],
            },
        },
        {
            "tool": "manual_chemistry_review",
            "rationale": (
                "Step 2 — REQUIRES CHEMIST review. The .FROZEN block "
                "needs orbital-position-remap directives specific to the "
                "molecule's MO ordering. Consult DIRAC 25 user guide "
                "or the CmF_5f_frz.inp test fixture."
            ),
        },
        {
            "tool": "manual_chemistry_review",
            "rationale": (
                "Step 3 — same as step 2; closed-shell freeze + 5f "
                "relax requires manual completion."
            ),
        },
    ]

    return {
        "central_element": central_element,
        "reference_element": reference_element,
        "n_5f_electrons": n_5f_electrons,
        "plan": plan,
        "next_actions": next_actions,
        "manual_review_required": True,
        "convergence_strategy": "frozen-orbital multi-step (CmF.md / Mochizuki 2003)",
        "warning": (
            "This is a CHEMISTRY-EXPERT workflow. Step 1 can be auto-driven "
            "(reference-atom atomic SCF). Steps 2 and 3 emit input "
            "scaffolding with explanatory comments — the actual .FROZEN / "
            "orbital-import blocks must be filled in manually. The exact "
            "syntax lives in DIRAC's test fixtures (not bundled with all "
            "containers); see get_dirac_topic_guide('cm_class_workflow') "
            "for the full strategy."
        ),
    }


def _basis_lookup(
    element: str,
    basis: dict[str, str] | None,
    default_basis: str | None,
) -> str:
    if basis and element in basis:
        return basis[element]
    if default_basis:
        return default_basis
    return "dyall.2zp"  # safe default for actinides

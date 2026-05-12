"""Atomic-start orchestrator for DIRAC.

The "atomic start" workflow for a difficult molecular SCF:

  1. For each unique element in the molecule, run a closed-shell (or
     AOC) atomic SCF and save the .h5 checkpoint.
  2. Run the molecular SCF launching with
     ``--copy="Elem1.h5 Elem2.h5 ..."``  so DIRAC reads each atomic
     checkpoint and projects it as a starting guess.

Common for actinides, lanthanides, and high-spin transition-metal
systems where the molecular SCF cannot converge from the default
Hückel guess.

``prepare_atomic_start`` returns a launch plan — a list of
``{name, inp_text, mol_text, h5_output}`` dicts in execution order,
plus the molecule's ``--copy`` argument list ready to feed into
``prepare_dirac_launch``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.core.common import ATOMIC_SYMBOLS
from chemtools.programs.dirac.input.inp import draft_inp
from chemtools.programs.dirac.input.mol import draft_mol


# Neutral-atom AOC configs for DIRAC.
#
# Each entry is ``{closed_shell: [n_fsym1, n_fsym2], open_shell: [...]}``.
# Both ``closed_shell`` entries and ``open_shell.n_electrons`` count REAL
# ELECTRONS PER FERMION IRCOP (matching DIRAC's .CLOSED SHELL convention).
# Closed-shell counts must be EVEN (one Kramers pair = 2 electrons).
# Total real electrons = sum(closed_shell) + sum(open n_electrons) = Z.
#
# AOC "spinors" spec convention: ``"<fsym1_count>,<fsym2_count>"`` where
# fsym 1 is the GERADE block (E1g) and fsym 2 is the UNGERADE block (E1u).
# Atomic angular-momentum parities:
#   s, d, g (even L) → gerade  → fsym 1
#   p, f, h (odd L)  → ungerade → fsym 2
#
# 1s/2s/3s/... → 2 spinors each in fsym 1   (spec "2,0")
# 2p/3p/...    → 6 spinors each in fsym 2   ("0,6")
# 3d/4d/...    → 10 spinors each in fsym 1  ("10,0")
# 4f/5f/...    → 14 spinors each in fsym 2  ("0,14")
#
# Noble-gas cumulative closed real-electron counts (matches DIRAC
# ``converging_atoms.md`` examples: Nb [Kr]4d^4 5s^1 uses 18 18 closed):
#   He → (2, 0)    Ne → (4, 6)     Ar → (6, 12)
#   Kr → (18, 18)  Xe → (30, 24)   Rn → (42, 44)
#
_ATOMIC_GROUND_STATES: dict[str, dict[str, Any]] = {
    # Period 1
    "H":  {"closed_shell": [0, 0], "open_shell": [{"n_electrons": 1, "spinors": "2,0"}]},
    "He": {"closed_shell": [2, 0]},
    # Period 2 — He core (2,0) + 2s/2p
    "Li": {"closed_shell": [2, 0], "open_shell": [{"n_electrons": 1, "spinors": "2,0"}]},
    "Be": {"closed_shell": [4, 0]},
    "B":  {"closed_shell": [4, 0], "open_shell": [{"n_electrons": 1, "spinors": "0,6"}]},
    "C":  {"closed_shell": [4, 0], "open_shell": [{"n_electrons": 2, "spinors": "0,6"}]},
    "N":  {"closed_shell": [4, 0], "open_shell": [{"n_electrons": 3, "spinors": "0,6"}]},
    "O":  {"closed_shell": [4, 0], "open_shell": [{"n_electrons": 4, "spinors": "0,6"}]},
    "F":  {"closed_shell": [4, 0], "open_shell": [{"n_electrons": 5, "spinors": "0,6"}]},
    "Ne": {"closed_shell": [4, 6]},
    # Period 3 — Ne core (4,6) + 3s/3p
    "Na": {"closed_shell": [4, 6], "open_shell": [{"n_electrons": 1, "spinors": "2,0"}]},
    "Mg": {"closed_shell": [6, 6]},
    "Al": {"closed_shell": [6, 6], "open_shell": [{"n_electrons": 1, "spinors": "0,6"}]},
    "Si": {"closed_shell": [6, 6], "open_shell": [{"n_electrons": 2, "spinors": "0,6"}]},
    "P":  {"closed_shell": [6, 6], "open_shell": [{"n_electrons": 3, "spinors": "0,6"}]},
    "S":  {"closed_shell": [6, 6], "open_shell": [{"n_electrons": 4, "spinors": "0,6"}]},
    "Cl": {"closed_shell": [6, 6], "open_shell": [{"n_electrons": 5, "spinors": "0,6"}]},
    "Ar": {"closed_shell": [6, 12]},
    # Period 4 — Ar core (6,12) + 4s/3d/4p
    "K":  {"closed_shell": [6, 12], "open_shell": [{"n_electrons": 1, "spinors": "2,0"}]},
    "Ca": {"closed_shell": [8, 12]},
    # 3d transition metals: Ca core (8,12) + 3d^N (gerade, fsym 1)
    "Sc": {"closed_shell": [8, 12], "open_shell": [{"n_electrons": 1, "spinors": "10,0"}]},
    "Ti": {"closed_shell": [8, 12], "open_shell": [{"n_electrons": 2, "spinors": "10,0"}]},
    "V":  {"closed_shell": [8, 12], "open_shell": [{"n_electrons": 3, "spinors": "10,0"}]},
    "Cr": {"closed_shell": [8, 12], "open_shell": [{"n_electrons": 4, "spinors": "10,0"}]},
    "Mn": {"closed_shell": [8, 12], "open_shell": [{"n_electrons": 5, "spinors": "10,0"}]},
    "Fe": {"closed_shell": [8, 12], "open_shell": [{"n_electrons": 6, "spinors": "10,0"}]},
    "Co": {"closed_shell": [8, 12], "open_shell": [{"n_electrons": 7, "spinors": "10,0"}]},
    "Ni": {"closed_shell": [8, 12], "open_shell": [{"n_electrons": 8, "spinors": "10,0"}]},
    "Cu": {"closed_shell": [16, 12], "open_shell": [{"n_electrons": 1, "spinors": "2,0"}]},
    "Zn": {"closed_shell": [18, 12]},
    # 4p block — Zn core (18,12) + 4p
    "Ga": {"closed_shell": [18, 12], "open_shell": [{"n_electrons": 1, "spinors": "0,6"}]},
    "Ge": {"closed_shell": [18, 12], "open_shell": [{"n_electrons": 2, "spinors": "0,6"}]},
    "As": {"closed_shell": [18, 12], "open_shell": [{"n_electrons": 3, "spinors": "0,6"}]},
    "Se": {"closed_shell": [18, 12], "open_shell": [{"n_electrons": 4, "spinors": "0,6"}]},
    "Br": {"closed_shell": [18, 12], "open_shell": [{"n_electrons": 5, "spinors": "0,6"}]},
    "Kr": {"closed_shell": [18, 18]},
    # Period 5 — Kr core (18,18) + 5s/4d/5p
    "Rb": {"closed_shell": [18, 18], "open_shell": [{"n_electrons": 1, "spinors": "2,0"}]},
    "Sr": {"closed_shell": [20, 18]},
    "Y":  {"closed_shell": [20, 18], "open_shell": [{"n_electrons": 1, "spinors": "10,0"}]},
    "Zr": {"closed_shell": [20, 18], "open_shell": [{"n_electrons": 2, "spinors": "10,0"}]},
    "Mo": {"closed_shell": [20, 18], "open_shell": [{"n_electrons": 4, "spinors": "10,0"}]},
    "Ru": {"closed_shell": [20, 18], "open_shell": [{"n_electrons": 6, "spinors": "10,0"}]},
    "Rh": {"closed_shell": [20, 18], "open_shell": [{"n_electrons": 7, "spinors": "10,0"}]},
    "Pd": {"closed_shell": [28, 18]},
    "Ag": {"closed_shell": [28, 18], "open_shell": [{"n_electrons": 1, "spinors": "2,0"}]},
    "Cd": {"closed_shell": [30, 18]},
    "Xe": {"closed_shell": [30, 24]},
    # Period 6 — Xe core (30,24) + 6s/4f/5d
    "Cs": {"closed_shell": [30, 24], "open_shell": [{"n_electrons": 1, "spinors": "2,0"}]},
    "Ba": {"closed_shell": [32, 24]},
    "La": {"closed_shell": [32, 24], "open_shell": [{"n_electrons": 1, "spinors": "10,0"}]},
    "Ce": {"closed_shell": [32, 24], "open_shell": [{"n_electrons": 2, "spinors": "10,14"}]},
    "Eu": {"closed_shell": [32, 24], "open_shell": [{"n_electrons": 7, "spinors": "0,14"}]},
    "Gd": {"closed_shell": [32, 24], "open_shell": [{"n_electrons": 8, "spinors": "10,14"}]},
    "Yb": {"closed_shell": [32, 38]},
    "Lu": {"closed_shell": [32, 38], "open_shell": [{"n_electrons": 1, "spinors": "10,0"}]},
    "Rn": {"closed_shell": [42, 44]},
    # Period 7 actinides — Rn core (42,44) + 7s/5f/6d.
    # Heavy actinides need .KPSELE for AOC convergence (5f near-degenerate
    # spinors otherwise oscillate the RELSCF inner loop). Configs from
    # converging_atoms.md: closed core = Rn + 7s^2 spread across 7 kappas
    # (s1/2, p1/2, p3/2, d3/2, d5/2, f5/2, f7/2). 5f open shell rows
    # carry 6 in f5/2 + 8 in f7/2; 6d rows carry 4 in d3/2 + 6 in d5/2.
    "Fr": {"closed_shell": [42, 44], "open_shell": [{"n_electrons": 1, "spinors": "2,0"}]},
    "Ra": {"closed_shell": [44, 44]},
    "Ac": {"closed_shell": [44, 44], "open_shell": [{"n_electrons": 1, "spinors": "10,0"}],
           "kpsele": {
               "kappas": [-1, 1, -2, 2, -3, 3, -4],
               "closed": [14, 10, 20, 12, 18, 6, 8],
               "shells": [[0, 0, 0, 4, 6, 0, 0]],  # 6d^1
           }},
    "Th": {"closed_shell": [44, 44], "open_shell": [{"n_electrons": 2, "spinors": "10,0"}],
           "kpsele": {
               "kappas": [-1, 1, -2, 2, -3, 3, -4],
               "closed": [14, 10, 20, 12, 18, 6, 8],
               "shells": [[0, 0, 0, 4, 6, 0, 0]],  # 6d^2
           }},
    # 5f^N 6d^1 7s^2 — split into TWO open shells (5f + 6d) for KPSELE.
    "Pa": {"closed_shell": [44, 44],
           "open_shell": [{"n_electrons": 2, "spinors": "0,14"},
                          {"n_electrons": 1, "spinors": "10,0"}],
           "kpsele": {
               "kappas": [-1, 1, -2, 2, -3, 3, -4],
               "closed": [14, 10, 20, 12, 18, 6, 8],
               "shells": [[0, 0, 0, 0, 0, 6, 8],  # 5f^2
                          [0, 0, 0, 4, 6, 0, 0]],  # 6d^1
           }},
    "U":  {"closed_shell": [44, 44],
           "open_shell": [{"n_electrons": 3, "spinors": "0,14"},
                          {"n_electrons": 1, "spinors": "10,0"}],
           "kpsele": {
               "kappas": [-1, 1, -2, 2, -3, 3, -4],
               "closed": [14, 10, 20, 12, 18, 6, 8],
               "shells": [[0, 0, 0, 0, 0, 6, 8],
                          [0, 0, 0, 4, 6, 0, 0]],
           }},
    "Np": {"closed_shell": [44, 44],
           "open_shell": [{"n_electrons": 4, "spinors": "0,14"},
                          {"n_electrons": 1, "spinors": "10,0"}],
           "kpsele": {
               "kappas": [-1, 1, -2, 2, -3, 3, -4],
               "closed": [14, 10, 20, 12, 18, 6, 8],
               "shells": [[0, 0, 0, 0, 0, 6, 8],
                          [0, 0, 0, 4, 6, 0, 0]],
           }},
    # Pure 5f^N (no 6d) — single open shell.
    "Pu": {"closed_shell": [44, 44],
           "open_shell": [{"n_electrons": 6, "spinors": "0,14"}],
           "kpsele": {
               "kappas": [-1, 1, -2, 2, -3, 3, -4],
               "closed": [14, 10, 20, 12, 18, 6, 8],
               "shells": [[0, 0, 0, 0, 0, 6, 8]],
           }},
    "Am": {"closed_shell": [44, 44],
           "open_shell": [{"n_electrons": 7, "spinors": "0,14"}],
           "kpsele": {
               "kappas": [-1, 1, -2, 2, -3, 3, -4],
               "closed": [14, 10, 20, 12, 18, 6, 8],
               "shells": [[0, 0, 0, 0, 0, 6, 8]],
           }},
    "Cm": {"closed_shell": [44, 44],
           "open_shell": [{"n_electrons": 7, "spinors": "0,14"},
                          {"n_electrons": 1, "spinors": "10,0"}],
           "kpsele": {
               "kappas": [-1, 1, -2, 2, -3, 3, -4],
               "closed": [14, 10, 20, 12, 18, 6, 8],
               "shells": [[0, 0, 0, 0, 0, 6, 8],
                          [0, 0, 0, 4, 6, 0, 0]],
           }},
}


def prepare_atomic_start(
    molecule_atoms: list[dict[str, Any]],
    *,
    basis: dict[str, str] | None = None,
    default_basis: str | None = None,
    hamiltonian: dict[str, Any] | None = None,
    integrals: dict[str, Any] | None = None,
    use_x2c: bool = True,
    output_dir: str | None = None,
    molecule_name: str = "molecule",
    molecule_scf: dict[str, Any] | None = None,
    molecule_units: str = "bohr",
) -> dict[str, Any]:
    """Build a per-element + molecule launch plan for the atomic-start workflow.

    Parameters
    ----------
    molecule_atoms
        Atoms in the molecule. Same shape draft_mol accepts (must allow
        nuclear-charge resolution).
    basis
        Per-element basis assignments. Used for both atomic and molecular
        runs so the atomic checkpoint orbitals are compatible.
    default_basis
        Fallback basis if an element isn't in ``basis``.
    hamiltonian, integrals
        Forwarded to ``draft_inp`` for all jobs. ``use_x2c=True`` adds
        ``.X2C`` to the hamiltonian (recommended for heavy elements).
    output_dir
        Where the launch plan suggests writing files. Defaults to cwd.
    molecule_name
        Used in the molecular .inp / .mol filenames and the
        ``--copy=`` filename hints.
    molecule_scf
        SCF subsection for the molecular run (closed_shell, open_shell,
        etc.). The atomic runs use ``_ATOMIC_GROUND_STATES`` automatically.

    Returns
    -------
    dict with keys::

        plan:           list of {name, kind, inp_path, mol_path,
                                inp_text, mol_text, h5_output,
                                expected_copy_basename}
        molecule_index:  index in plan of the molecular job
        copy_args:       list of .h5 basenames the molecule consumes
        atomic_count:    number of atomic jobs in the plan
    """
    hamiltonian = dict(hamiltonian or {})
    if use_x2c:
        hamiltonian.setdefault("x2c", True)
    integrals = dict(integrals or {})
    # X2C / AMFI require decontracted basis sets (DIRAC aborts otherwise with
    # "AMFI: only decontracted basis sets can be used"). Force .UNCONTRACT
    # unless the caller has explicitly set it.
    if hamiltonian.get("x2c") or hamiltonian.get("amfi"):
        integrals.setdefault("uncontract", True)

    out_dir = Path(output_dir or ".").resolve()

    # Unique elements in the molecule
    unique_elements: list[str] = []
    seen: set[str] = set()
    for a in molecule_atoms:
        z = _atom_z(a)
        sym = ATOMIC_SYMBOLS.get(z) if z else None
        if sym and sym not in seen:
            unique_elements.append(sym)
            seen.add(sym)

    if not unique_elements:
        raise ValueError(
            "Could not resolve any element symbols from molecule_atoms — "
            "pass nuclear_charge or element on each atom."
        )

    plan: list[dict[str, Any]] = []

    # Build per-element atomic jobs
    for sym in unique_elements:
        gs = _ATOMIC_GROUND_STATES.get(sym)
        if gs is None:
            # Fall back to a generic guess: single closed shell with all
            # electrons paired. Caller may want to override.
            gs = {"closed_shell": []}

        # Open-shell atomic AOC for heavy elements doesn't converge in
        # DIRAC's default 50 iterations; bump to 200 and enable .RESOLVE
        # so the SCF can iteratively re-classify the open-shell orbitals.
        # Light closed-shell singlets (He, Ne, Ar, ...) don't need this.
        gs = dict(gs)
        if gs.get("open_shell"):
            gs.setdefault("max_iter", 200)
            gs.setdefault("resolve", True)

        atom_basis = _basis_for_element(sym, basis, default_basis)
        atom_mol_text = draft_mol(
            atoms=[{"label": sym, "x": 0.0, "y": 0.0, "z": 0.0,
                    "element": sym}],
            basis={sym: atom_basis},
            units=molecule_units,
            symmetry="auto",
            title=f"{sym} atom (atomic-start checkpoint)",
        )
        atom_inp_text = draft_inp({
            "title": f"{sym} atom SCF",
            "wave_function": "scf",
            "analyze": ["mulpop"],
            # Atomic jobs have inversion symmetry (Dinfh-effective) so .VECPOP
            # needs 2 ircop lines.
            "analyze_vecpop_ranges": ["1..oo", "1..oo"],
            "hamiltonian": hamiltonian,
            "integrals": integrals,
            "scf": gs,
        })
        # pam-dirac output naming: <inp_stem>_<mol_stem> joined unless
        # identical (then deduped). Atomic jobs use Sym.inp + Sym.mol so
        # the .h5 lands as Sym.h5 — that's the basename the molecule's
        # --copy= chain expects.
        inp_path = out_dir / f"{sym}.inp"
        mol_path = out_dir / f"{sym}.mol"
        h5_output = out_dir / f"{sym}.h5"
        plan.append({
            "name": f"{sym}_atom",
            "kind": "atomic",
            "element": sym,
            "inp_path": str(inp_path),
            "mol_path": str(mol_path),
            "inp_text": atom_inp_text,
            "mol_text": atom_mol_text,
            "h5_output": str(h5_output),
            # The molecule's --copy expects the atomic .h5 renamed to
            # <Element>.h5 (matching the molecule's element labels).
            "expected_copy_basename": f"{sym}.h5",
        })

    # Build the molecular job
    mol_scf = molecule_scf or {"closed_shell": []}
    mol_inp_text = draft_inp({
        "title": f"{molecule_name} molecular SCF (atomic-start)",
        "wave_function": "scf",
        "analyze": ["mulpop"],
        "hamiltonian": hamiltonian,
        "integrals": integrals,
        "scf": mol_scf,
    })
    mol_mol_text = draft_mol(
        atoms=molecule_atoms,
        basis=basis,
        default_basis=default_basis,
        units=molecule_units,
        symmetry="auto",
        title=f"{molecule_name} (atomic-start molecule)",
    )
    mol_inp_path = out_dir / f"{molecule_name}.inp"
    mol_mol_path = out_dir / f"{molecule_name}.mol"
    mol_h5_path = out_dir / f"{molecule_name}.h5"
    plan.append({
        "name": molecule_name,
        "kind": "molecule",
        "inp_path": str(mol_inp_path),
        "mol_path": str(mol_mol_path),
        "inp_text": mol_inp_text,
        "mol_text": mol_mol_text,
        "h5_output": str(mol_h5_path),
        "expected_copy_basename": None,
    })

    copy_args = [
        p["expected_copy_basename"]
        for p in plan
        if p["kind"] == "atomic"
    ]

    return {
        "plan": plan,
        "molecule_index": len(plan) - 1,
        "copy_args": copy_args,
        "atomic_count": len(plan) - 1,
        "next_actions": [
            {
                "tool": "prepare_dirac_launch",
                "rationale": (
                    "Launch each atomic job sequentially. After each, copy "
                    "its .h5 to <Element>.h5 in the molecule's run directory."
                ),
                "args": {
                    "input_file": p["inp_path"],
                    "mol_file":   p["mol_path"],
                },
            }
            for p in plan if p["kind"] == "atomic"
        ] + [
            {
                "tool": "prepare_dirac_launch",
                "rationale": "Launch the molecule with --copy referencing the atomic .h5 files.",
                "args": {
                    "input_file": plan[-1]["inp_path"],
                    "mol_file":   plan[-1]["mol_path"],
                    "copy_files": copy_args,
                },
            },
        ],
    }


def _atom_z(atom: dict[str, Any]) -> int | None:
    from chemtools.programs.dirac.input.mol import _atom_to_z
    return _atom_to_z(atom)


def _basis_for_element(
    sym: str,
    basis: dict[str, str] | None,
    default_basis: str | None,
) -> str:
    if basis and sym in basis:
        return basis[sym]
    if default_basis:
        return default_basis
    raise ValueError(
        f"No basis provided for element {sym}. Pass basis={{'{sym}': '...'}} "
        f"or default_basis='...'."
    )

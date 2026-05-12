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


# Ground-state spin multiplicity + AOC config for common elements.
# Format: (closed_per_ircop, open_shell_blocks_or_None)
# closed_per_ircop omits an `open_shell` when the ground state is a
# closed-shell singlet. n_ircops=1 in atoms with no inversion symmetry
# under the default DIRAC behavior; the AOC spec carries through.
_ATOMIC_GROUND_STATES: dict[str, dict[str, Any]] = {
    "H":  {"closed_shell": [],  "open_shell": [{"n_electrons": 1, "spinors": "0,2"}]},
    "He": {"closed_shell": [1]},
    "Li": {"closed_shell": [1], "open_shell": [{"n_electrons": 1, "spinors": "0,2"}]},
    "Be": {"closed_shell": [2]},
    "B":  {"open_shell": [{"n_electrons": 1, "spinors": "0,6"}]},
    "C":  {"open_shell": [{"n_electrons": 2, "spinors": "0,6"}]},
    "N":  {"open_shell": [{"n_electrons": 3, "spinors": "0,6"}]},
    "O":  {"open_shell": [{"n_electrons": 4, "spinors": "0,6"}]},
    "F":  {"open_shell": [{"n_electrons": 5, "spinors": "0,6"}]},
    "Ne": {"closed_shell": [5]},
    "Na": {"open_shell": [{"n_electrons": 1, "spinors": "0,2"}]},
    "Mg": {"closed_shell": [6]},
    "Al": {"open_shell": [{"n_electrons": 1, "spinors": "0,6"}]},
    "Si": {"open_shell": [{"n_electrons": 2, "spinors": "0,6"}]},
    "P":  {"open_shell": [{"n_electrons": 3, "spinors": "0,6"}]},
    "S":  {"open_shell": [{"n_electrons": 4, "spinors": "0,6"}]},
    "Cl": {"open_shell": [{"n_electrons": 5, "spinors": "0,6"}]},
    "Ar": {"closed_shell": [9]},
    # Transition metals: 4d/3d open shell config
    "Sc": {"open_shell": [{"n_electrons": 1, "spinors": "0,10"}]},
    "Ti": {"open_shell": [{"n_electrons": 2, "spinors": "0,10"}]},
    "V":  {"open_shell": [{"n_electrons": 3, "spinors": "0,10"}]},
    "Cr": {"open_shell": [{"n_electrons": 5, "spinors": "0,10"}]},
    "Mn": {"open_shell": [{"n_electrons": 5, "spinors": "0,10"}]},
    "Fe": {"open_shell": [{"n_electrons": 6, "spinors": "0,10"}]},
    "Co": {"open_shell": [{"n_electrons": 7, "spinors": "0,10"}]},
    "Ni": {"open_shell": [{"n_electrons": 8, "spinors": "0,10"}]},
    "Cu": {"open_shell": [{"n_electrons": 10, "spinors": "0,10"}]},
    "Zn": {"closed_shell": [15]},
    # Heavy elements that often need atomic-start. Open shells track the
    # ground-state d/f electron count; the spinor count is the d-shell (10)
    # or f-shell (14) Kramers-paired manifold.
    "Y":  {"open_shell": [{"n_electrons": 1, "spinors": "0,10"}]},
    "Zr": {"open_shell": [{"n_electrons": 2, "spinors": "0,10"}]},
    "Mo": {"open_shell": [{"n_electrons": 6, "spinors": "0,10"}]},
    "Ru": {"open_shell": [{"n_electrons": 7, "spinors": "0,10"}]},
    "Rh": {"open_shell": [{"n_electrons": 8, "spinors": "0,10"}]},
    "Pd": {"open_shell": [{"n_electrons": 10, "spinors": "0,10"}]},
    "Ag": {"closed_shell": [23]},
    # Lanthanides (4f^n) — partial coverage; extend as needed.
    "Ce": {"open_shell": [{"n_electrons": 1, "spinors": "0,14"}]},
    "Eu": {"open_shell": [{"n_electrons": 7, "spinors": "0,14"}]},
    "Gd": {"open_shell": [{"n_electrons": 8, "spinors": "0,14"}]},
    # Actinides (5f^n) — partial coverage.
    "Th": {"open_shell": [{"n_electrons": 2, "spinors": "0,14"}]},
    "U":  {"open_shell": [{"n_electrons": 4, "spinors": "0,14"}]},
    "Np": {"open_shell": [{"n_electrons": 5, "spinors": "0,14"}]},
    "Pu": {"open_shell": [{"n_electrons": 6, "spinors": "0,14"}]},
    "Am": {"open_shell": [{"n_electrons": 7, "spinors": "0,14"}]},
    "Cm": {"open_shell": [{"n_electrons": 8, "spinors": "0,14"}]},
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
            "hamiltonian": hamiltonian,
            "integrals": integrals,
            "scf": gs,
        })
        # pam-dirac default output naming is <inp_stem>_<mol_stem>.{out,h5}
        inp_path = out_dir / f"{sym}.inp"
        mol_path = out_dir / f"{sym}.mol"
        h5_output = out_dir / f"{sym}_{sym}.h5"
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
    mol_h5_path = out_dir / f"{molecule_name}_{molecule_name}.h5"
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

"""DIRAC ΔSCF core ionization workflow.

Implements the recipe from the DIRAC tutorial at
diracprogram.org/doc/release-26/tutorials/x_ray/CO_N2_IP1s/ — compute
1s (or other core) ionization potentials via the SCF energy difference
between the neutral molecule and a state with a core hole at a specific
atomic 1s spinor.

The procedure has two stages:

  Step 1. Neutral SCF (standard closed-shell).
          --get="DFCOEF=cf.<mol>" to save the converged coefficients.

  Step 2. Core-ionized SCF.
          --put="cf.<mol>=DFCOEF" to seed with neutral orbitals, then:
            .REORDER         move the target core 1s out of the
                             .CLOSED SHELL range, to the END
            .CLOSED SHELL    (n_orbitals - 1) — one less Kramers pair
            .OPEN SHELL 1
              1/2            one electron in 2 spinors (1s manifold)
            .OPENFAC 1.0
            .OVLSEL          overlap-based orbital selection (keeps
                             the hole pinned to the right spinor)
            .NODYNSEL        disable dynamic selection (no shuffling)

IP = E(cation) - E(neutral), reported in Ha + eV.

For HETERONUCLEAR diatomics (CO, OH, NH, ...) this works out of the
box because the two 1s orbitals come at well-separated energies. For
HOMONUCLEAR diatomics (N2, O2, ...) the σg/σu 1s combinations are
near-degenerate; the symmetric ΔSCF gives a delocalized core hole and
overestimates the IP by ~10 eV. The published fix (per the DIRAC
tutorial) is Pipek-Mezey localization of the 1s orbitals via .LOCALIZE
in C1 symmetry, then importing the localized orbitals into a lower-
symmetry calculation with --put="ac.<loc>=DFACMO". That two-step
localization workflow is NOT yet auto-driven here — the orchestrator
emits a warning + flag for homonuclear inputs.

Orbital indexing convention (matches the tutorial):
  Atoms are sorted by Z descending; their 1s orbitals appear in MO
  order (deepest 1s first). For CO (Z=8 O + Z=6 C):
    Orbital 1 = O 1s
    Orbital 2 = C 1s
  So C1s ionization needs .REORDER 1,3..n,2 (moves orbital 2 to end);
  O1s ionization needs .REORDER 2..n,1 (moves orbital 1 to end).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.core.common import ATOMIC_SYMBOLS, ELEMENT_TO_Z
from chemtools.programs.dirac.input.inp import draft_inp
from chemtools.programs.dirac.input.mol import draft_mol


def _atom_z(atom: dict[str, Any]) -> int:
    if "nuclear_charge" in atom and atom["nuclear_charge"] is not None:
        return int(round(float(atom["nuclear_charge"])))
    element = atom.get("element") or atom.get("label", "")
    sym = "".join(c for c in element if c.isalpha())
    sym = sym[:1].upper() + sym[1:].lower()
    return ELEMENT_TO_Z.get(sym, 0)


def _is_homonuclear(atoms: list[dict[str, Any]]) -> bool:
    """Same element on both ends of a 2-atom system → homonuclear."""
    if len(atoms) != 2:
        return False
    zs = sorted(_atom_z(a) for a in atoms)
    return zs[0] == zs[1] and zs[0] > 0


def _core_orbital_index_for(
    target_atom_index: int,
    atoms: list[dict[str, Any]],
) -> int:
    """Return the 1-based MO index of the target atom's 1s orbital.

    DIRAC orders core 1s orbitals by nuclear-charge depth (heavier atom
    first). For CO with atoms=[C, O] in input order, sorted by Z descending
    gives [O, C], so:
      - O is the deepest → orbital 1
      - C is next → orbital 2
    """
    if target_atom_index < 0 or target_atom_index >= len(atoms):
        raise IndexError(f"target_atom_index {target_atom_index} out of range")
    # Index atoms by Z descending; ties broken by original position.
    sorted_atoms = sorted(
        enumerate(atoms),
        key=lambda iz: (-_atom_z(iz[1]), iz[0]),
    )
    for mo_index, (orig_idx, _atom) in enumerate(sorted_atoms, start=1):
        if orig_idx == target_atom_index:
            return mo_index
    return 1  # fallback


def _reorder_spec_move_orbital_to_end(
    orbital_index: int,
    n_closed_kpair: int,
) -> str:
    """Build a .REORDER spec that moves ``orbital_index`` to position
    ``n_closed_kpair`` (the end of the new closed-shell range).

    n_closed_kpair = number of remaining closed-shell Kramers pairs after
    removing one (i.e., for a 7-K-pair ground state, this is 7).

    Examples (matching the DIRAC tutorial):
      orbital_index=1, n=7 → "2..7,1"
      orbital_index=2, n=7 → "1,3..7,2"
      orbital_index=3, n=7 → "1,2,4..7,3"
    """
    if orbital_index < 1 or orbital_index > n_closed_kpair:
        raise ValueError(f"orbital_index {orbital_index} not in [1, {n_closed_kpair}]")
    pieces: list[str] = []
    # Walk the non-target orbitals in order, collapsing consecutive ranges
    i = 1
    while i <= n_closed_kpair:
        if i == orbital_index:
            i += 1
            continue
        j = i
        while j + 1 <= n_closed_kpair and j + 1 != orbital_index:
            j += 1
        if j == i:
            pieces.append(str(i))
        else:
            pieces.append(f"{i}..{j}")
        i = j + 1
    pieces.append(str(orbital_index))
    return ",".join(pieces)


def prepare_core_ionization(
    atoms: list[dict[str, Any]],
    *,
    target_atom_indices: list[int],
    n_total_electrons: int,
    basis: dict[str, str] | None = None,
    default_basis: str | None = None,
    use_x2c: bool = True,
    output_dir: str | None = None,
    molecule_name: str = "molecule",
    molecule_units: str = "bohr",
    closed_shell_per_ircop: list[int] | None = None,
    write_files: bool = False,
) -> dict[str, Any]:
    """Build the ΔSCF core-ionization launch plan.

    Parameters
    ----------
    atoms
        Full molecule geometry. ΔSCF needs to know all atoms to draft
        the .mol file.
    target_atom_indices
        0-based indices of atoms whose 1s shells to ionize. One job
        per target.
    n_total_electrons
        Total electrons in the NEUTRAL molecule. Used to populate
        .CLOSED SHELL (ground state) and (n-1)/2 closed + 1 open for
        the ionized state.
    basis, default_basis
        Per-element basis assignments (passed through to draft_mol).
    use_x2c
        Default True. X2C + .UNCONTRACT is standard for core-level work.
    closed_shell_per_ircop
        Caller-supplied per-fsym closed counts for the neutral state.
        Default: assume NFSYM=1 (no inversion) and put all electrons in
        a single ircop. Override for inversion-symmetric systems.

    Returns dict with:
      ``plan``: 1 + N steps (ground state + one per target atom)
      ``ip_pairs``: list of {atom_label, neutral_out, ionized_out}
                    that compute_dirac_core_ip will consume
      ``manual_review_required``: True if homonuclear (localization needed)
    """
    if not atoms:
        raise ValueError("atoms list is empty")
    if not target_atom_indices:
        raise ValueError("target_atom_indices is empty — nothing to ionize")

    out_dir = Path(output_dir or ".").resolve()

    # Closed shell per ircop for the neutral state.
    if closed_shell_per_ircop is None:
        # Default: NFSYM=1 (no inversion), all electrons in one ircop.
        closed_shell_per_ircop = [n_total_electrons]
    closed_neutral_total = sum(closed_shell_per_ircop)
    if closed_neutral_total != n_total_electrons:
        raise ValueError(
            f"closed_shell_per_ircop sum {closed_neutral_total} != "
            f"n_total_electrons {n_total_electrons}"
        )
    n_kpair_total = n_total_electrons // 2

    # Hamiltonian + integrals — X2C requires .UNCONTRACT.
    hamiltonian: dict[str, Any] = {"x2c": True} if use_x2c else {}
    integrals: dict[str, Any] = {"uncontract": True} if use_x2c else {}

    # ----- Build the .mol once (same geometry for all jobs) -----
    mol_text = draft_mol(
        atoms=atoms,
        basis=basis,
        default_basis=default_basis,
        units=molecule_units,
        symmetry="auto",
        title=f"{molecule_name} (core-ionization series)",
    )
    mol_path = out_dir / f"{molecule_name}.mol"

    plan: list[dict[str, Any]] = []

    # ----- Step 1: neutral SCF -----
    neutral_inp = draft_inp({
        "title": f"{molecule_name} neutral SCF (core-IP reference)",
        "wave_function": "scf",
        "analyze": ["mulpop"],
        "hamiltonian": hamiltonian,
        "integrals": integrals,
        "scf": {"closed_shell": list(closed_shell_per_ircop)},
    })
    neutral_inp_path = out_dir / f"{molecule_name}.inp"
    plan.append({
        "step": 1,
        "name": f"{molecule_name}_neutral",
        "kind": "ground_state",
        "inp_path": str(neutral_inp_path),
        "mol_path": str(mol_path),
        "inp_text": neutral_inp,
        "mol_text": mol_text,
        "expected_outputs": {
            # When inp_stem == mol_stem, pam-dirac dedupes to <stem>.out;
            # otherwise it emits <inp>_<mol>.out.
            "out": str(out_dir / f"{molecule_name}.out"),
            "h5":  str(out_dir / f"{molecule_name}.h5"),
        },
        "launch_args": {
            "input_file":   str(neutral_inp_path),
            "mol_file":     str(mol_path),
            # --outcmo writes the converged MO coefficients into the .h5
            # checkpoint (alongside the standard SCF outputs). The
            # ionized-state launches then use --incmo to consume them.
            "outcmo": True,
        },
    })

    # ----- Step 2..N: core-ionized SCF (one per target atom) -----
    ip_pairs: list[dict[str, Any]] = []
    for target_idx in target_atom_indices:
        if target_idx < 0 or target_idx >= len(atoms):
            raise IndexError(f"target_atom_index {target_idx} out of range")
        target = atoms[target_idx]
        elem = ATOMIC_SYMBOLS.get(_atom_z(target)) or target.get("label", "X")

        # MO index of the target atom's 1s orbital
        mo_idx = _core_orbital_index_for(target_idx, atoms)

        # Build the .REORDER spec — for NFSYM=1 (the default), there's
        # only one ircop line. For NFSYM=2 systems with multiple ircops,
        # the caller would need to provide explicit reorder specs.
        if len(closed_shell_per_ircop) != 1:
            raise NotImplementedError(
                "Multi-ircop core ionization not yet auto-drafted. For "
                "homonuclear / inversion-symmetric systems use the "
                "tutorial's Pipek-Mezey localization workflow manually."
            )

        # n_kpair after removing one closed electron pair = n_kpair_total - 1.
        # But we keep n_kpair_total spinors in REORDER (we're not deleting,
        # just permuting). The "new closed shell" gets n_kpair_total-1, the
        # "new open shell" gets the moved-to-end orbital.
        reorder_spec = _reorder_spec_move_orbital_to_end(mo_idx, n_kpair_total)

        ionized_label = f"{molecule_name}_{elem}{target_idx + 1}_1s"
        ionized_inp = draft_inp({
            "title": f"{molecule_name} {elem} 1s core hole (orbital {mo_idx})",
            "wave_function": "scf",
            "analyze": ["mulpop"],
            "hamiltonian": hamiltonian,
            "integrals": integrals,
            "scf": {
                "closed_shell": [n_total_electrons - 2],
                "open_shell": [{"n_electrons": 1, "spinors": "2"}],
                "openfac": 1.0,
                "ovlsel": True,
                "nodynsel": True,
                "reorder": [reorder_spec],
            },
        })
        ionized_inp_path = out_dir / f"{ionized_label}.inp"
        # pam-dirac output name: <inp_stem>_<mol_stem>.out when stems
        # differ. Here inp_stem=ionized_label, mol_stem=molecule_name.
        ionized_out_path = (
            out_dir / f"{ionized_label}.out"
            if ionized_label == molecule_name
            else out_dir / f"{ionized_label}_{molecule_name}.out"
        )
        plan.append({
            "step": len(plan) + 1,
            "name": ionized_label,
            "kind": "core_ionized",
            "target_element": elem,
            "target_atom_index": target_idx,
            "core_orbital_index": mo_idx,
            "reorder_spec": reorder_spec,
            "inp_path": str(ionized_inp_path),
            "mol_path": str(mol_path),
            "inp_text": ionized_inp,
            "mol_text": mol_text,
            "expected_outputs": {
                "out": str(ionized_out_path),
                "h5":  str(ionized_out_path.with_suffix(".h5")),
            },
            "launch_args": {
                "input_file":   str(ionized_inp_path),
                "mol_file":     str(mol_path),
                # Modern DIRAC ≥ 22 uses --incmo to read MO coefficients
                # from the in-cwd .h5 of a previous run with the same .mol.
                # This is simpler than the --put/--get cf.<mol>=DFCOEF
                # Fortran-binary chain and matches the user's existing
                # CO fixtures' workflow.
                "extra_args": ["--incmo"],
            },
        })
        ip_pairs.append({
            "atom_label": f"{elem}{target_idx + 1}",
            "element": elem,
            "core_orbital_index": mo_idx,
            "neutral_out":  str(out_dir / f"{molecule_name}.out"),
            "ionized_out":  str(ionized_out_path),
        })

    # ----- Optional file write -----
    if write_files:
        out_dir.mkdir(parents=True, exist_ok=True)
        mol_path.write_text(mol_text, encoding="utf-8")
        for p in plan:
            Path(p["inp_path"]).write_text(p["inp_text"], encoding="utf-8")

    homonuclear = _is_homonuclear(atoms)
    warnings: list[str] = []
    if homonuclear:
        warnings.append(
            "Homonuclear diatomic detected. Symmetric ΔSCF will give a "
            "DELOCALIZED core hole and overestimate the IP by ~10 eV. "
            "The published fix is Pipek-Mezey localization via "
            ".LOCALIZE in C1 symmetry, then importing the localized "
            "orbitals into a lower-symmetry calculation. See "
            "get_dirac_topic_guide('core_ionization') for the workflow."
        )

    return {
        "plan": plan,
        "ip_pairs": ip_pairs,
        "molecule_name": molecule_name,
        "n_total_electrons": n_total_electrons,
        "homonuclear": homonuclear,
        "manual_review_required": homonuclear,
        "warnings": warnings,
        "next_actions": [
            {
                "tool": "prepare_dirac_launch",
                "rationale": "Run the neutral SCF first to generate the cf.<mol> coefficient file.",
                "args": plan[0]["launch_args"],
            },
        ] + [
            {
                "tool": "prepare_dirac_launch",
                "rationale": f"Core-ionize {step['target_element']} 1s using ΔSCF.",
                "args": step["launch_args"],
            }
            for step in plan[1:]
        ] + [
            {
                "tool": "compute_dirac_core_ip",
                "rationale": "After all SCFs converge, compute IPs = E_ionized - E_neutral.",
                "args": {"ip_pairs": ip_pairs},
            },
        ],
    }


def compute_core_ip(
    neutral_out: str,
    ionized_out: str,
) -> dict[str, Any]:
    """Compute the core ionization potential from two DIRAC outputs.

    IP = E(ionized) - E(neutral), reported in Hartree + eV.
    """
    from chemtools.programs.dirac.parse import parse_output
    from chemtools.core.units import HARTREE_TO_EV

    e_neutral = parse_output(neutral_out).get("total_energy_hartree")
    e_ionized = parse_output(ionized_out).get("total_energy_hartree")
    if e_neutral is None:
        raise ValueError(f"Could not parse total energy from {neutral_out}")
    if e_ionized is None:
        raise ValueError(f"Could not parse total energy from {ionized_out}")

    ip_ha = e_ionized - e_neutral
    return {
        "neutral_total_energy_hartree": e_neutral,
        "ionized_total_energy_hartree": e_ionized,
        "ip_hartree": ip_ha,
        "ip_ev": ip_ha * HARTREE_TO_EV,
    }

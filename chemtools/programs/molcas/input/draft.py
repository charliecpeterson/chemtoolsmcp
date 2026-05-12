"""Top-level Molcas input drafter — orchestrates SEWARD + SCF + RASSCF + CASPT2.

Translates a program-neutral `InputSpec` (or a Molcas-specific dict) into a
full input deck. Methods supported:

  * ``HF``, ``SCF``                 → SEWARD + SCF
  * ``DFT``, ``KSDFT``              → SEWARD + SCF (KSDFT)
  * ``CASSCF``, ``RASSCF``          → SEWARD + SCF + RASSCF
  * ``CASPT2``, ``RASPT2``          → SEWARD + SCF + RASSCF + CASPT2 (SS)
  * ``MS-CASPT2``, ``XMS-CASPT2``,
    ``RMS-CASPT2``, ``XDW-CASPT2``  → SEWARD + SCF + RASSCF (multi-root) + CASPT2 (MS variant)

For active-space methods the InputSpec must include either:
  * ``program_options.cas_active_electrons`` and ``cas_active_orbitals``, OR
  * an explicit per-symmetry split via ``program_options.rasscf``.
"""

from __future__ import annotations

from typing import Any

from chemtools.programs.molcas.input._utils import (
    auto_label,
    element_symbol,
    normalize_atoms,
    total_electrons,
)
from chemtools.programs.molcas.input.seward import render_seward_block
from chemtools.programs.molcas.input.scf import render_scf_block
from chemtools.programs.molcas.input.rasscf import (
    compute_active_space_partition,
    render_rasscf_block,
)
from chemtools.programs.molcas.input.caspt2 import render_caspt2_block


_ACTIVE_SPACE_METHODS = {
    "casscf", "rasscf", "caspt2", "raspt2",
    "ms-caspt2", "xms-caspt2", "rms-caspt2", "xdw-caspt2",
    "ms-raspt2", "xms-raspt2",
}


def draft_molcas_input(spec: dict[str, Any]) -> str:
    """Build a full Molcas input deck from an InputSpec-shaped dict.

    Required keys: atoms, charge, multiplicity, method, basis, task.
    Common optional: title, geometry_units, functional, program_options.

    program_options recognized keys:
      memory_mb              int (default 2000) — exported via MOLCAS_MEM
      symmetry               str — symmetry generators (e.g. "X XY"); None = C1
      n_symmetries           int — must match symmetry; auto-derived for C1
      occupied_per_symmetry  list[int] — required for SCF when n_symmetries > 1
      n_basis_per_symmetry   list[int] — needed for RASSCF secondary computation
      pkthrs / cholesky / ricd      — passed to SEWARD
      seward_extra_keywords  list[str] — appended into SEWARD block
      scf                    dict — extra knobs passed to render_scf_block
      cas_active_electrons   int — for active-space methods
      cas_active_orbitals    int — for active-space methods
      rasscf                 dict — overrides for render_rasscf_block (per-sym splits, n_roots, ...)
      caspt2                 dict — overrides for render_caspt2_block
    """
    atoms = auto_label(normalize_atoms(spec["atoms"]))
    charge = int(spec.get("charge", 0))
    multiplicity = int(spec.get("multiplicity", 1))
    method = str(spec["method"]).strip().lower()
    basis = spec["basis"]
    title = spec.get("title")
    units = spec.get("geometry_units", "angstrom")
    program_opts = dict(spec.get("program_options") or {})

    # ---- Memory ----
    memory_mb = int(program_opts.get("memory_mb", 2000))
    blocks: list[str] = []
    blocks.append(f">>> Export MOLCAS_MEM={memory_mb}\n")

    # ---- SEWARD ----
    symmetry = program_opts.get("symmetry")
    n_symmetries = int(program_opts.get("n_symmetries", _default_n_sym(symmetry)))
    seward_text = render_seward_block(
        atoms=atoms,
        basis=basis,
        title=title,
        symmetry=symmetry,
        geometry_units=units,
        pkthrs=program_opts.get("pkthrs"),
        cholesky=bool(program_opts.get("cholesky", False)),
        ricd=bool(program_opts.get("ricd", False)),
        expert=bool(program_opts.get("expert", False)),
        extra_keywords=program_opts.get("seward_extra_keywords"),
        inline_basis=bool(program_opts.get("inline_basis", False)),
    )
    blocks.append(seward_text)

    # ---- SCF ----
    n_electrons = total_electrons(atoms, charge)
    scf_opts = dict(program_opts.get("scf") or {})
    scf_text = render_scf_block(
        n_electrons=n_electrons,
        multiplicity=multiplicity,
        n_symmetries=n_symmetries,
        occupied_per_symmetry=(
            scf_opts.pop("occupied_per_symmetry", None)
            or program_opts.get("occupied_per_symmetry")
        ),
        title=title,
        ksdft_functional=spec.get("functional") if method in {"dft", "ksdft"} else None,
        charge=charge if multiplicity != 1 else None,
        **scf_opts,
    )
    blocks.append(scf_text)

    if method in {"hf", "scf", "dft", "ksdft"}:
        return "\n".join(blocks)

    # ---- RASSCF ----
    if method not in _ACTIVE_SPACE_METHODS:
        raise ValueError(
            f"Unsupported method {method!r}; supported: HF/SCF/DFT/CASSCF/RASSCF/CASPT2/"
            "RASPT2/MS-CASPT2/XMS-CASPT2/RMS-CASPT2/XDW-CASPT2"
        )
    rasscf_opts = dict(program_opts.get("rasscf") or {})
    cas_e = program_opts.get("cas_active_electrons") or rasscf_opts.get("cas_active_electrons")
    cas_o = program_opts.get("cas_active_orbitals") or rasscf_opts.get("cas_active_orbitals")
    if cas_e is None or cas_o is None:
        raise ValueError(
            "Active-space method requires cas_active_electrons and cas_active_orbitals "
            "in program_options."
        )
    partition = compute_active_space_partition(
        n_electrons=n_electrons,
        cas_active_electrons=int(cas_e),
        cas_active_orbitals=int(cas_o),
        n_symmetries=n_symmetries,
        n_basis_per_symmetry=program_opts.get("n_basis_per_symmetry"),
        n_frozen_per_symmetry=rasscf_opts.get("frozen_per_symmetry"),
        n_inactive_per_symmetry=rasscf_opts.get("inactive_per_symmetry"),
        active_per_symmetry=rasscf_opts.get("active_per_symmetry"),
        target_symmetry_for_active=rasscf_opts.get("target_symmetry_for_active"),
        ras1_holes_max=rasscf_opts.get("ras1_holes_max", 0),
        ras1_per_symmetry=rasscf_opts.get("ras1_per_symmetry"),
        ras3_electrons_max=rasscf_opts.get("ras3_electrons_max", 0),
        ras3_per_symmetry=rasscf_opts.get("ras3_per_symmetry"),
    )
    n_roots = int(rasscf_opts.get("n_roots", 1))
    rasscf_text = render_rasscf_block(
        multiplicity=multiplicity,
        state_symmetry=int(rasscf_opts.get("state_symmetry", 1)),
        nactel=partition["nactel"],
        frozen=partition["frozen"],
        inactive=partition["inactive"],
        ras1=partition["ras1"],
        ras2=partition["ras2"],
        ras3=partition["ras3"],
        title=rasscf_opts.get("title", title),
        n_roots=n_roots,
        root_for_optimization=rasscf_opts.get("root_for_optimization"),
        state_average_weights=rasscf_opts.get("state_average_weights"),
        iterations=rasscf_opts.get("iterations", (50, 25)),
        convergence_thresholds=rasscf_opts.get("convergence_thresholds", (1.0e-6, 1.0e-3, 1.0e-3)),
        use_lumorb=rasscf_opts.get("use_lumorb", True),
        out_orbitals=rasscf_opts.get("out_orbitals"),
        extra_keywords=rasscf_opts.get("extra_keywords"),
    )
    blocks.append(rasscf_text)

    if method in {"casscf", "rasscf"}:
        return "\n".join(blocks)

    # ---- CASPT2 ----
    caspt2_opts = dict(program_opts.get("caspt2") or {})
    variant = _method_to_caspt2_variant(method)
    caspt2_text = render_caspt2_block(
        title=caspt2_opts.get("title", title),
        variant=variant,
        n_roots=n_roots,
        target_root=caspt2_opts.get("target_root"),
        frozen_per_symmetry=partition["frozen"],
        ipea_shift=caspt2_opts.get("ipea_shift"),
        real_shift=float(caspt2_opts.get("real_shift", 0.0)),
        imaginary_shift=float(caspt2_opts.get("imaginary_shift", 0.0)),
        sigma_p_regularization=caspt2_opts.get("sigma_p_regularization"),
        max_iter=int(caspt2_opts.get("max_iter", 30)),
        convergence=float(caspt2_opts.get("convergence", 1.0e-8)),
        properties=bool(caspt2_opts.get("properties", False)),
        grdt=bool(caspt2_opts.get("grdt", False)),
        extra_keywords=caspt2_opts.get("extra_keywords"),
    )
    blocks.append(caspt2_text)

    return "\n".join(blocks)


def _method_to_caspt2_variant(method: str) -> str:
    m = method.lower()
    if m in {"caspt2", "raspt2"}:
        return "SS"
    if m in {"ms-caspt2", "ms-raspt2"}:
        return "MS"
    if m in {"xms-caspt2", "xms-raspt2"}:
        return "XMS"
    if m == "rms-caspt2":
        return "RMS"
    if m == "xdw-caspt2":
        return "XDW"
    return "SS"


def _default_n_sym(symmetry: str | None) -> int:
    if not symmetry:
        return 1
    n_gen = len([t for t in symmetry.split() if t])
    # 1 generator → 2 irreps (Cs / C2 / Ci),
    # 2 generators → 4 irreps (C2v / C2h / D2),
    # 3 generators → 8 irreps (D2h)
    return {0: 1, 1: 2, 2: 4, 3: 8}.get(n_gen, 1)

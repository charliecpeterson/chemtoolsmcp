"""SCF (& KSDFT) input block builder.

Renders &SCF blocks for closed-shell, ROHF (Charge keyword), or UHF setups.
For C1 symmetry the Occupied vector is auto-derived from the electron count;
for higher symmetry the caller must pass `occupied_per_symmetry` explicitly
because we cannot reliably split irreps without a starting wave function.
"""

from __future__ import annotations

from typing import Iterable

from chemtools.programs.molcas.input._utils import (
    determine_alpha_beta,
    format_per_symmetry,
)


def render_scf_block(
    *,
    n_electrons: int,
    multiplicity: int = 1,
    occupied_per_symmetry: list[int] | None = None,
    n_symmetries: int = 1,
    title: str | None = None,
    iterations: int = 50,
    uhf: bool = False,
    ksdft_functional: str | None = None,
    charge: int | None = None,
    extra_keywords: list[str] | None = None,
) -> str:
    """Build an &SCF input block.

    Parameters
    ----------
    n_electrons
        Total electron count (computed from atoms + molecular charge).
    multiplicity
        Spin multiplicity (2S+1). 1 = singlet, 3 = triplet, etc.
    occupied_per_symmetry
        Doubly-occupied orbital count per symmetry. Required for n_symmetries>1.
        For C1 (n_symmetries=1), auto-derived from n_electrons.
    n_symmetries
        Number of symmetry irreps from the SEWARD block (1 for C1).
    iterations
        Max SCF iterations (default 50).
    uhf
        If True, emit `UHF` keyword (allows broken-symmetry spin densities).
    ksdft_functional
        If set, emit `KSDFT <functional>` (e.g. "B3LYP", "PBE", "M06").
    charge
        For ROHF: the molecular charge sets the spin density; pass it through
        to the Charge keyword. Use this for open-shell HF on cations/anions.
    extra_keywords
        Free-form lines appended before End of input.
    """
    is_open_shell_rohf = (multiplicity != 1) and (not uhf)

    # `Charge` and `Occupied` are mutually exclusive in Molcas SCF. For ROHF we
    # use `Charge <q> <2S>` and let SCF derive the per-shell occupations. For
    # closed-shell (and explicit-Occupied UHF) we emit `Occupied`.
    if is_open_shell_rohf:
        if occupied_per_symmetry is not None:
            raise ValueError(
                "Open-shell ROHF cannot use Occupied (mutually exclusive with Charge in Molcas). "
                "Either set uhf=True (then Occupied is the alpha-doubly-occupied vector + use "
                "extra_keywords to add Open) or omit occupied_per_symmetry and let SCF derive "
                "from Charge + Spin."
            )
    else:
        if n_symmetries > 1 and occupied_per_symmetry is None:
            raise ValueError(
                "occupied_per_symmetry must be provided when n_symmetries > 1; "
                "Molcas SCF cannot guess the irrep split without a starting wave function."
            )
        if occupied_per_symmetry is not None and len(occupied_per_symmetry) != n_symmetries:
            raise ValueError(
                f"occupied_per_symmetry has {len(occupied_per_symmetry)} entries "
                f"but n_symmetries is {n_symmetries}"
            )
        if occupied_per_symmetry is None:
            # C1 closed-shell: auto-fill
            if n_electrons % 2 != 0:
                raise ValueError(
                    f"Closed-shell SCF requires an even electron count, got {n_electrons}"
                )
            occupied_per_symmetry = [n_electrons // 2]

    body: list[str] = ["&SCF &END"]
    if title:
        body.append("Title")
        body.append(f" {title}")
    if uhf:
        body.append("UHF")
    if ksdft_functional:
        body.append("KSDFT")
        body.append(f" {ksdft_functional}")
    if is_open_shell_rohf:
        # `Charge <q> <2S>` — SCF derives the per-shell occupation from this
        body.append("Charge")
        body.append(f" {charge if charge is not None else 0} {multiplicity - 1}")
    else:
        body.append("Occupied")
        body.append(format_per_symmetry(occupied_per_symmetry))
    body.append("Iterations")
    body.append(f" {iterations}")
    if extra_keywords:
        body.extend(extra_keywords)
    body.append("End of input")
    return "\n".join(body) + "\n"

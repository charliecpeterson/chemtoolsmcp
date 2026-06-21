"""RASSCF / CASSCF input block builder + active-space partition helper.

The active-space helper translates a desired ``CAS(M, N)`` (M active
electrons in N active orbitals) into the per-symmetry partitioning that
RASSCF wants: ``Frozen``, ``Inactive``, ``Ras1``, ``Ras2``, ``Ras3``, plus
the ``Nactel`` triple ``(active_electrons, max_holes_RAS1, max_e_RAS3)``.

For C1 (single irrep) this is straightforward arithmetic. For higher
symmetry the caller must specify either:

  * fully explicit per-symmetry vectors, OR
  * a single target irrep that holds the whole active space (the rest of the
    inactive electrons go in their natural irreps).
"""

from __future__ import annotations

from typing import Iterable

from chemtools.programs.molcas.input._utils import format_per_symmetry


def compute_active_space_partition(
    *,
    n_electrons: int,
    cas_active_electrons: int,
    cas_active_orbitals: int,
    n_symmetries: int = 1,
    n_basis_per_symmetry: list[int] | None = None,
    n_frozen_per_symmetry: list[int] | None = None,
    active_per_symmetry: list[int] | None = None,
    target_symmetry_for_active: int | None = None,
    n_inactive_per_symmetry: list[int] | None = None,
    ras1_holes_max: int = 0,
    ras1_per_symmetry: list[int] | None = None,
    ras3_electrons_max: int = 0,
    ras3_per_symmetry: list[int] | None = None,
) -> dict:
    """Resolve a CAS(M,N) request into the RASSCF directive vectors.

    For C1 (n_symmetries=1) the only required input is the CAS dimensions.
    For higher symmetry the caller must specify either active_per_symmetry
    (fully explicit) or target_symmetry_for_active (place all M,N in one
    irrep) plus n_inactive_per_symmetry (we cannot guess the inactive split).

    Returns
    -------
    dict with keys:
        nactel                : [cas_active_electrons, ras1_holes_max, ras3_electrons_max]
        frozen                : [int per sym]
        inactive              : [int per sym]
        ras1, ras2, ras3      : [int per sym]
        secondary             : [int per sym]   (informational; RASSCF computes it)
        active_orbitals_total : int
        active_electrons_total: int
    """
    if n_basis_per_symmetry is None:
        n_basis_per_symmetry = [None] * n_symmetries  # type: ignore
    if n_frozen_per_symmetry is None:
        n_frozen_per_symmetry = [0] * n_symmetries

    # --- Derive active per symmetry ---
    if active_per_symmetry is not None:
        if len(active_per_symmetry) != n_symmetries:
            raise ValueError(
                f"active_per_symmetry has {len(active_per_symmetry)} entries; "
                f"expected n_symmetries={n_symmetries}"
            )
        if sum(active_per_symmetry) != cas_active_orbitals:
            raise ValueError(
                f"active_per_symmetry sum {sum(active_per_symmetry)} != cas_active_orbitals "
                f"{cas_active_orbitals}"
            )
        active_split = list(active_per_symmetry)
    elif n_symmetries == 1:
        active_split = [cas_active_orbitals]
    else:
        if target_symmetry_for_active is None:
            raise ValueError(
                "For n_symmetries > 1, provide either active_per_symmetry or "
                "target_symmetry_for_active."
            )
        if not (1 <= target_symmetry_for_active <= n_symmetries):
            raise ValueError(
                f"target_symmetry_for_active={target_symmetry_for_active} out of range [1, {n_symmetries}]"
            )
        active_split = [0] * n_symmetries
        active_split[target_symmetry_for_active - 1] = cas_active_orbitals

    # --- Derive RAS1 / RAS2 / RAS3 split ---
    if ras1_holes_max > 0 or ras3_electrons_max > 0:
        # User wants a true RASSCF, not a CASSCF
        ras1 = list(ras1_per_symmetry) if ras1_per_symmetry else [0] * n_symmetries
        ras3 = list(ras3_per_symmetry) if ras3_per_symmetry else [0] * n_symmetries
        ras2 = [active_split[i] - ras1[i] - ras3[i] for i in range(n_symmetries)]
        if any(r < 0 for r in ras2):
            raise ValueError(
                f"RAS1+RAS3 exceed total active in some symmetry; "
                f"ras1={ras1}, ras3={ras3}, active={active_split}"
            )
    else:
        ras1 = [0] * n_symmetries
        ras2 = list(active_split)
        ras3 = [0] * n_symmetries

    # --- Derive inactive ---
    inactive_electrons = n_electrons - cas_active_electrons - 2 * sum(n_frozen_per_symmetry)
    if inactive_electrons % 2 != 0 or inactive_electrons < 0:
        raise ValueError(
            f"Inactive electrons must be a non-negative even number; "
            f"got {inactive_electrons} from n_electrons={n_electrons}, "
            f"cas_active_electrons={cas_active_electrons}, "
            f"frozen={n_frozen_per_symmetry}"
        )
    inactive_orbitals_total = inactive_electrons // 2

    if n_inactive_per_symmetry is not None:
        if len(n_inactive_per_symmetry) != n_symmetries:
            raise ValueError(
                f"n_inactive_per_symmetry has {len(n_inactive_per_symmetry)} entries; "
                f"expected n_symmetries={n_symmetries}"
            )
        if sum(n_inactive_per_symmetry) != inactive_orbitals_total:
            raise ValueError(
                f"n_inactive_per_symmetry sum {sum(n_inactive_per_symmetry)} != "
                f"required inactive total {inactive_orbitals_total}"
            )
        inactive_split = list(n_inactive_per_symmetry)
    elif n_symmetries == 1:
        inactive_split = [inactive_orbitals_total]
    else:
        raise ValueError(
            "For n_symmetries > 1, provide n_inactive_per_symmetry; "
            "the inactive split cannot be guessed without a starting wave function."
        )

    # --- Compute secondary (informational; RASSCF derives it from basis) ---
    secondary_split: list[int] = []
    if all(b is not None for b in n_basis_per_symmetry):
        for i in range(n_symmetries):
            sec = (
                n_basis_per_symmetry[i]  # type: ignore[operator]
                - n_frozen_per_symmetry[i]
                - inactive_split[i]
                - active_split[i]
            )
            if sec < 0:
                raise ValueError(
                    f"sym {i+1}: not enough basis functions ({n_basis_per_symmetry[i]}) "
                    f"to host frozen+inactive+active = "
                    f"{n_frozen_per_symmetry[i]+inactive_split[i]+active_split[i]}"
                )
            secondary_split.append(sec)

    return {
        "nactel": [cas_active_electrons, int(ras1_holes_max), int(ras3_electrons_max)],
        "frozen": list(n_frozen_per_symmetry),
        "inactive": inactive_split,
        "ras1": ras1,
        "ras2": ras2,
        "ras3": ras3,
        "secondary": secondary_split,
        "active_orbitals_total": cas_active_orbitals,
        "active_electrons_total": cas_active_electrons,
    }


def render_rasscf_block(
    *,
    multiplicity: int,
    state_symmetry: int,
    nactel: list[int],
    frozen: list[int],
    inactive: list[int],
    ras2: list[int],
    ras1: list[int] | None = None,
    ras3: list[int] | None = None,
    title: str | None = None,
    n_roots: int = 1,
    root_for_optimization: int | None = None,
    state_average_weights: list[float] | None = None,
    iterations: tuple[int, int] = (50, 25),
    convergence_thresholds: tuple[float, float, float] = (1.0e-6, 1.0e-3, 1.0e-3),
    use_lumorb: bool = True,
    out_orbitals: str | None = None,
    extra_keywords: list[str] | None = None,
) -> str:
    """Render an &RASSCF block.

    nactel: [active_electrons, max_holes_RAS1, max_electrons_RAS3]. For pure CASSCF
    the last two are 0.

    For state-averaged calculations:
      * n_roots=1 + root_for_optimization=None → ground state
      * n_roots>1 → SA-CASSCF over those roots; if state_average_weights is None, uses
        equal weights via CIRoot.
      * root_for_optimization picks which root to optimize with respect to MOs.

    use_lumorb=True emits LumOrb (read MOs from INPORB → typically the SCF orbitals
    or a previous RasOrb).
    """
    body: list[str] = ["&RASSCF &END"]
    if title:
        body.append("Title")
        body.append(f" {title}")
    body.append("Symmetry")
    body.append(f" {int(state_symmetry)}")
    body.append("Spin")
    body.append(f" {int(multiplicity)}")
    body.append("Nactel")
    body.append(format_per_symmetry(nactel))
    if any(frozen):
        body.append("Frozen")
        body.append(format_per_symmetry(frozen))
    body.append("Inactive")
    body.append(format_per_symmetry(inactive))
    if ras1 and any(ras1):
        body.append("Ras1")
        body.append(format_per_symmetry(ras1))
    body.append("Ras2")
    body.append(format_per_symmetry(ras2))
    if ras3 and any(ras3):
        body.append("Ras3")
        body.append(format_per_symmetry(ras3))
    if n_roots > 1:
        # CIRoot N N <weights or sequence>
        body.append("CIRoot")
        # Format: CIRoot <nact> <total_ci_roots> 1; then list of root indices then weights
        # The most common shape (equal weights) is: CIRoot N N 1
        body.append(f" {n_roots} {n_roots} 1")
    if root_for_optimization is not None and n_roots > 1:
        body.append("Rlxroot")
        body.append(f" {int(root_for_optimization)}")
    body.append("Thrs")
    body.append(",".join(f"{t:.1E}" for t in convergence_thresholds))
    body.append("Iteration")
    body.append(f"{iterations[0]},{iterations[1]}")
    if use_lumorb:
        body.append("LumOrb")
    if out_orbitals:
        body.append("OutOrbitals")
        body.append(f" {out_orbitals}")
    if extra_keywords:
        body.extend(extra_keywords)
    body.append("End of input")
    return "\n".join(body) + "\n"

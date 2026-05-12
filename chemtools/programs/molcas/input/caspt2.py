"""CASPT2 input block builder.

Renders &CASPT2 blocks for SS / MS / XMS / RMS / XDW CASPT2 with the most
commonly used knobs: IPEA shift, real or imaginary level shift, sigma-p
regularization, frozen-orbital count, MaxIter, properties.
"""

from __future__ import annotations

from typing import Iterable, Literal

from chemtools.programs.molcas.input._utils import format_per_symmetry


CASPT2Variant = Literal["SS", "MS", "XMS", "RMS", "XDW"]


def render_caspt2_block(
    *,
    title: str | None = None,
    variant: CASPT2Variant = "SS",
    n_roots: int = 1,
    target_root: int | None = None,
    frozen_per_symmetry: list[int] | None = None,
    ipea_shift: float | None = None,
    real_shift: float = 0.0,
    imaginary_shift: float = 0.0,
    sigma_p_regularization: float | None = None,
    max_iter: int = 30,
    convergence: float = 1.0e-8,
    properties: bool = False,
    grdt: bool = False,
    extra_keywords: list[str] | None = None,
) -> str:
    """Render an &CASPT2 block.

    Parameters
    ----------
    variant
        SS = single-state, MS = multistate (Finley/Malmqvist/Roos),
        XMS = extended multistate (Granovsky / Shiozaki),
        RMS = rotated multistate (Battaglia/Lindh),
        XDW = extended dynamically weighted (Battaglia/Lindh).
        For SS with n_roots>1, the same SS-CASPT2 calculation is repeated
        per root (one CASPT2 group per root).
    n_roots
        Number of roots in the preceding RASSCF (must match RASSCF CIRoot).
    target_root
        For SS-CASPT2 on a specific excited state (LRoot keyword). For MS/XMS
        leave None — all `n_roots` are propagated.
    frozen_per_symmetry
        Frozen orbitals (must match RASSCF Frozen for the parser to be happy).
    ipea_shift
        IPEA shift value. None means "use Molcas default" (0.25 since v6.4
        unless MOLCAS_NEW_DEFAULTS=YES). Pass 0.0 to explicitly disable.
    real_shift, imaginary_shift
        Mutually exclusive with sigma_p_regularization. 0.0 = off.
    sigma_p_regularization
        Sigma_p^2 value (e.g. 0.05) — activates SIG2 (or SIG1) keyword.
    properties
        If True, emit `Properties` to compute ⟨n|μ|n⟩ for each root.
    grdt
        If True, emit `GRDT` to precompute quantities for analytic CASPT2 gradients
        (required for ALASKA + MCLR analytic gradient code).
    """
    if real_shift and imaginary_shift:
        raise ValueError("real_shift and imaginary_shift are mutually exclusive")
    if (real_shift or imaginary_shift) and sigma_p_regularization is not None:
        raise ValueError("Level shift and sigma-p regularization are mutually exclusive")
    if target_root is not None and variant != "SS":
        raise ValueError(
            f"target_root only meaningful for variant=SS; for {variant} all roots are propagated"
        )

    body: list[str] = ["&CASPT2 &END"]
    if title:
        body.append("Title")
        body.append(f" {title}")

    if variant == "MS":
        body.append("Multistate")
        body.append(f"{n_roots} " + " ".join(str(i) for i in range(1, n_roots + 1)))
    elif variant == "XMS":
        body.append("XMultistate")
        body.append(f"{n_roots} " + " ".join(str(i) for i in range(1, n_roots + 1)))
    elif variant == "RMS":
        body.append("RMultistate")
        body.append(f"{n_roots} " + " ".join(str(i) for i in range(1, n_roots + 1)))
    elif variant == "XDW":
        body.append("DWMS")
        body.append(f"{n_roots} " + " ".join(str(i) for i in range(1, n_roots + 1)))
    elif target_root is not None:
        body.append("LRoot")
        body.append(f" {int(target_root)}")
    elif n_roots > 1:
        # SS-CASPT2 over multiple roots (one group per root)
        body.append("Multistate")
        body.append(f"{n_roots} " + " ".join(str(i) for i in range(1, n_roots + 1)))
        # Note: with no XMS/RMS/MS keyword, this still produces SS results
        # for each root individually. To explicitly say SS, use "NoMult".
        body.append("NoMult")

    # Only emit Frozen when the user actually wants something frozen. Emitting
    # `Frozen 0` overrides CASPT2's default which is to auto-freeze deep core
    # orbitals — that override forces correlation of tight cores (e.g. C 1s)
    # and can trigger numerical instability (IEEE_UNDERFLOW → segfault).
    if frozen_per_symmetry is not None and any(frozen_per_symmetry):
        body.append("Frozen")
        body.append(format_per_symmetry(frozen_per_symmetry))

    if ipea_shift is not None:
        body.append("IPEAShift")
        body.append(f" {ipea_shift:.3f}")

    if real_shift > 0:
        body.append("Shift")
        body.append(f" {real_shift:.3f}")
    if imaginary_shift > 0:
        body.append("Imaginary")
        body.append(f" {imaginary_shift:.3f}")
    if sigma_p_regularization is not None:
        body.append("SIG2")
        body.append(f" {sigma_p_regularization:.4f}")

    if max_iter != 30:
        body.append("MaxIter")
        body.append(f" {int(max_iter)}")
    if convergence != 1.0e-8:
        body.append("Convergence")
        body.append(f" {convergence:.1E}")

    if properties:
        body.append("Properties")
    if grdt:
        body.append("GRDT")

    if extra_keywords:
        body.extend(extra_keywords)
    body.append("End of input")
    return "\n".join(body) + "\n"

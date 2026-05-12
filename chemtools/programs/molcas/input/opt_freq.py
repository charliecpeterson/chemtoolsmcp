"""ALASKA / SLAPAF / MCKINLEY / MCLR block builders + EMIL loop helpers.

These are the modules a Molcas opt+freq workflow chains around the SCF/RASSCF
core. The blocks are mostly empty — Molcas's defaults are sensible — but each
gets a knob or two for the common overrides.

EMIL helpers:
  - do_while_open() / do_while_close() — wrap a module sequence in `>>> Do while <<<`
    ... `>>> ENDDO <<<`
  - if_iter_one_open() / if_iter_one_close() — `>>> IF ( ITER = 1 ) <<<` ...
    `>>> ENDIF <<<` so SCF only runs on the first opt iteration (after which
    the previous SCF orbitals carry over via Project.ScfOrb)
"""

from __future__ import annotations

from typing import Iterable


def render_alaska_block(*, numerical: bool = False, title: str | None = None) -> str:
    """Render an &ALASKA block (analytical gradients by default).

    `numerical=True` switches to NumGrad — useful when analytic gradients are
    not available for the method or when debugging.
    """
    body: list[str] = ["&ALASKA &END"]
    if title:
        body.append(f"* {title}")
    if numerical:
        body.append("Numerical")
    body.append("End of input")
    return "\n".join(body) + "\n"


def render_slapaf_block(
    *,
    iterations: int | None = None,
    thresholds: tuple[float, float] | None = None,
    transition_state: bool = False,
    constraints: list[str] | None = None,
    title: str | None = None,
) -> str:
    """Render an &SLAPAF block.

    Defaults (empty block) work for minimum optimization with Schlegel
    convergence thresholds and BFGS Hessian update.
    """
    body: list[str] = ["&SLAPAF &END"]
    if title:
        body.append(f"* {title}")
    if transition_state:
        body.append("TS")
    if iterations is not None:
        body.append("Iterations")
        body.append(f" {int(iterations)}")
    if thresholds is not None:
        # SLAPAF Thresholds: gradient / step (both in internal coords)
        body.append("Thresholds")
        body.append(f" {thresholds[0]:.2E} {thresholds[1]:.2E}")
    if constraints:
        body.append("Constraints")
        body.extend(f" {c}" for c in constraints)
        body.append("End of constraints")
    body.append("End of input")
    return "\n".join(body) + "\n"


def render_mckinley_block(*, title: str | None = None) -> str:
    """Render an &MCKINLEY block — emits second-derivative integrals required
    by MCLR for analytical Hessians."""
    body: list[str] = ["&MCKINLEY &END"]
    if title:
        body.append(f"* {title}")
    body.append("End of input")
    return "\n".join(body) + "\n"


def render_mclr_block(
    *,
    iroot: int | None = None,
    title: str | None = None,
) -> str:
    """Render an &MCLR block — multi-configuration linear response → analytic
    Hessian + harmonic frequencies.

    `iroot` picks which RASSCF root to do the freq for in a state-averaged
    case (default is the first / lowest).
    """
    body: list[str] = ["&MCLR &END"]
    if title:
        body.append(f"* {title}")
    if iroot is not None:
        body.append("IRoot")
        body.append(f" {int(iroot)}")
    body.append("End of input")
    return "\n".join(body) + "\n"


def do_while_open() -> str:
    return ">>> Do while <<<\n"


def do_while_close() -> str:
    return ">>> ENDDO <<<\n"


def if_iter_one_open() -> str:
    return ">>> IF ( ITER = 1 ) <<<\n"


def if_iter_one_close() -> str:
    return ">>> ENDIF <<<\n"

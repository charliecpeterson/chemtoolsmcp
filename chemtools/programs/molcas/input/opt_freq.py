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
    irc: bool = False,
    n_irc_points: int | None = None,
    irc_step_size: float | None = None,
    irc_step_size_unit: str = "bohr",
    irc_algorithm: str | None = None,
    reaction_vector: list[list[float]] | None = None,
) -> str:
    """Render an &SLAPAF block.

    Defaults (empty block) work for minimum optimization with Schlegel
    convergence thresholds and BFGS Hessian update.

    IRC parameters (see Molcas SLAPAF docs):
      irc                 If True, emit the IRC keyword. Requires a starting
                          TS geometry + Hessian (from MCKINLEY+MCLR) already
                          in the input.
      n_irc_points        NIRC — max number of IRC points per direction.
                          Default Molcas behaviour follows until E increases.
      irc_step_size       IRCStep / MEPStep — step length (default 0.1 au
                          mass-weighted).
      irc_step_size_unit  "bohr" or "angstrom" — unit suffix for IRCStep.
      irc_algorithm       "GS" (default, González–Schlegel) or "MB" (Müller–Brown).
      reaction_vector     Explicit Cartesian reaction vector for IRC, as a
                          list of [x, y, z] rows (one per atom, in input
                          order). When set, emits the REACtion vector
                          keyword + the row block. Required when the prior
                          RUNFILE is not in the same scratch dir.
    """
    body: list[str] = ["&SLAPAF &END"]
    if title:
        body.append(f"* {title}")
    if transition_state:
        body.append("TS")
    if irc:
        body.append("IRC")
        if n_irc_points is not None:
            body.append("NIRC")
            body.append(f" {int(n_irc_points)}")
        if irc_step_size is not None:
            unit = irc_step_size_unit.strip().lower()
            if unit not in {"bohr", "angstrom"}:
                raise ValueError(
                    f"irc_step_size_unit must be 'bohr' or 'angstrom'; got {unit!r}"
                )
            body.append("IRCStep")
            body.append(f" {irc_step_size:g} {unit.upper()}")
        if irc_algorithm:
            alg = irc_algorithm.strip().upper()
            if alg not in {"GS", "MB"}:
                raise ValueError(
                    f"irc_algorithm must be 'GS' or 'MB'; got {irc_algorithm!r}"
                )
            body.append("IRCAlgorithm")
            body.append(f" {alg}")
        if reaction_vector is not None:
            body.append("REACtion vector")
            for row in reaction_vector:
                if len(row) != 3:
                    raise ValueError(
                        f"reaction_vector rows must be [x, y, z]; got {row!r}"
                    )
                body.append(f" {row[0]:14.8f} {row[1]:14.8f} {row[2]:14.8f}")
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

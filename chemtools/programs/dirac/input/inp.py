"""DIRAC ``.inp`` (job-control) file drafter.

Renders a `.inp` from a structured spec. The drafter handles the most
common DIRAC workflows:

  - SCF or DFT (closed-shell + AOC open-shell)
  - X2C or 4-component Dirac-Coulomb Hamiltonian
  - MULPOP + VECPOP analyze block
  - .REORDER MO inline (or insertion via the reorder strategy module)
  - MP2, CCSD, COSCI as wave function options
  - Properties block (excitations, polarizabilities — leaves the
    program_options open for power users)

For tightly-coupled chains (atomic-start → molecule with --copy),
see ``prepare_dirac_atomic_start`` in this package.
"""

from __future__ import annotations

from typing import Any


_KNOWN_WAVEFUNCTIONS = {"scf", "dft", "mp2", "ccsd", "cosci", "krci"}


def draft_inp(spec: dict[str, Any]) -> str:
    """Render a DIRAC ``.inp`` file from a structured spec.

    Top-level spec keys::

        title              str         human-readable label (.TITLE)
        wave_function      str         "scf", "dft", "mp2", "ccsd", "cosci"
        analyze            list[str]   keywords under **ANALYZE
                                       (mulpop, vecpop, ...)
        properties         bool        include .PROPERTIES under **DIRAC
        hamiltonian        dict        e.g. {"x2c": True, "dft_functional":
                                       "B3LYP", "ecp": True, "spinfree": False}
        integrals          dict        e.g. {"uncontract": True}
        scf                dict        closed_shell, open_shell,
                                       reorder (per-ircop spec list),
                                       max_iter, resolve (bool)
        post_scf           dict        (mp2/ccsd/cosci-specific knobs)
        analyze_vecpop_ranges  list[str]   ircop ranges for *MULPOP /
                                       .VECPOP (default ["1..oo","1..oo"])
        extra_sections     list[(name, body)]
                                       drop-in for power users; each
                                       body is a fully formatted block
                                       (without the **NAME header)

    Returns the file text (terminated with *END OF INPUT).
    """
    wf = spec.get("wave_function", "scf").lower()
    if wf not in _KNOWN_WAVEFUNCTIONS:
        raise ValueError(
            f"Unknown wave_function {wf!r}; expected one of {sorted(_KNOWN_WAVEFUNCTIONS)}"
        )
    ham = spec.get("hamiltonian") or {}
    integrals = spec.get("integrals") or {}
    scf = spec.get("scf") or {}
    analyze = spec.get("analyze") or []
    properties = spec.get("properties", False)

    lines: list[str] = []

    # ----- **DIRAC ---------------------------------------------------
    lines.append("**DIRAC")
    title = spec.get("title")
    if title:
        lines.append(".TITLE")
        lines.append(str(title).strip())
    # Long-form section names DIRAC expects:
    if wf in ("dft",):
        lines.append(".WAVE FUNCTION")
    elif wf in ("scf", "mp2", "ccsd", "cosci", "krci"):
        lines.append(".WAVE FUNCTION")
    if analyze:
        lines.append(".ANALYZE")
    if properties:
        lines.append(".PROPERTIES")

    # ----- **HAMILTONIAN --------------------------------------------
    ham_lines: list[str] = []
    if ham.get("x2c"):
        ham_lines.append(".X2C")
    if ham.get("spinfree"):
        ham_lines.append(".SPINFREE")
    if ham.get("dft_functional") or wf == "dft":
        functional = ham.get("dft_functional") or "B3LYP"
        ham_lines.append(".DFT")
        ham_lines.append(functional)
    if ham.get("ecp"):
        ham_lines.append(".ECP")
    if ham.get("amfi"):
        ham_lines.append(".AMFI")
    # The **HAMILTONIAN section is always required (DIRAC parses it).
    lines.append("**HAMILTONIAN")
    lines.extend(ham_lines)

    # ----- **INTEGRALS ----------------------------------------------
    if integrals.get("uncontract") or integrals.get("readin_uncontract"):
        lines.append("**INTEGRALS")
        lines.append("*READIN")
        lines.append(".UNCONTRACT")

    # ----- **ANALYZE ------------------------------------------------
    if analyze:
        lines.append("**ANALYZE")
        for kw in analyze:
            lines.append(f".{kw.upper()}")
        if "mulpop" in [a.lower() for a in analyze] or "vecpop" in [a.lower() for a in analyze]:
            ranges = spec.get("analyze_vecpop_ranges") or ["1..oo", "1..oo"]
            lines.append("*MULPOP")
            lines.append(".VECPOP")
            for r in ranges:
                lines.append(f" {r}")

    # ----- **WAVE FUNCTION ------------------------------------------
    lines.append("**WAVE FUNCTION")
    if wf == "scf" or wf == "dft":
        lines.append(".SCF")
        scf_block_lines = _build_scf_subsection(scf)
        if scf_block_lines:
            lines.append("*SCF")
            lines.extend(scf_block_lines)
    elif wf == "mp2":
        lines.append(".SCF")
        scf_block_lines = _build_scf_subsection(scf)
        if scf_block_lines:
            lines.append("*SCF")
            lines.extend(scf_block_lines)
        lines.append(".MP2")
    elif wf == "ccsd":
        lines.append(".SCF")
        scf_block_lines = _build_scf_subsection(scf)
        if scf_block_lines:
            lines.append("*SCF")
            lines.extend(scf_block_lines)
        lines.append(".CCSD")
    elif wf == "cosci":
        lines.append(".SCF")
        scf_block_lines = _build_scf_subsection(scf)
        if scf_block_lines:
            lines.append("*SCF")
            lines.extend(scf_block_lines)
        # COSCI is post-SCF; declare under **WAVE FUNCTION + a *COSCI block
        lines.append(".COSCI")

    # ----- Drop-in extra sections (power user) ------------------------
    for name, body in (spec.get("extra_sections") or []):
        lines.append(f"**{name.upper()}")
        if body:
            lines.append(str(body).rstrip())

    lines.append("*END OF INPUT")
    return "\n".join(lines) + "\n"


def _build_scf_subsection(scf: dict[str, Any]) -> list[str]:
    """Build the contents of a *SCF subsection from the SCF spec."""
    lines: list[str] = []
    if scf.get("resolve"):
        lines.append(".RESOLVE")

    closed = scf.get("closed_shell")
    if closed:
        lines.append(".CLOSED SHELL")
        if isinstance(closed, list):
            lines.append(" " + "  ".join(str(int(x)) for x in closed))
        else:
            lines.append(" " + str(int(closed)))

    open_shells = scf.get("open_shell")
    if open_shells:
        lines.append(".OPEN SHELL")
        lines.append(f" {len(open_shells)}")
        for os in open_shells:
            n_e = int(os["n_electrons"])
            # The "spinor spec" is a string like "0,14" or "2/0,14"
            spec_str = os.get("spinors") or os.get("orbital_spec") or ""
            if "/" in str(spec_str):
                lines.append(f" {spec_str}")
            else:
                lines.append(f" {n_e}/{spec_str}")

    reorder = scf.get("reorder")
    if reorder:
        lines.append(".REORDER MO")
        for r in reorder:
            lines.append(f" {r}")

    max_iter = scf.get("max_iter")
    if max_iter is not None:
        lines.append(".MAXITR")
        lines.append(f" {int(max_iter)}")

    evcconv = scf.get("evccnv")
    if evcconv is not None:
        lines.append(".EVCCNV")
        lines.append(f" {evcconv}")

    return lines

"""Failure-mode classifier + recovery suggester for Molcas runs.

`suggest_recovery(output_file)` reads a failed (or suspicious) Molcas .out
and matches against a set of known failure modes — each surfaced by real
dogfooding — emitting a structured Diagnosis envelope with the root cause,
a step-by-step fix recipe, and an agent-actionable `next_actions` list.

Failure modes covered (priority order):

  1.  seward_angstrom_symmetry      — "ANGSTROM is not a keyword!" when
                                       SEWARD has both Symmetry + Angstrom.
                                       Fix: convert geom to bohr.
  2.  missing_basis_in_loop         — "Input does not contain any basis sets"
                                       on an inner-loop SEWARD. Fix: use
                                       &GATEWAY for the initial basis block.
  3.  scf_single_electron           — "Current implementation only allows
                                       double occupations" on 1-electron
                                       species. Fix: drop &SCF block, run
                                       &RASSCF directly from GuessOrb.
  4.  scf_no_convergence            — &SCF Stop with _RC_NOT_CONVERGED_.
                                       Fix: skip SCF for high-spin TM atoms;
                                       or add level shift / bump iters.
  5.  rasscf_no_convergence         — &RASSCF Stop with _RC_NOT_CONVERGED_
                                       and used the full iter budget.
                                       Fix: bump Iteration N,M ceiling.
  6.  rasscf_wrong_active_space     — RASSCF converged but active orbitals
                                       don't match the target character.
                                       Fix: refine_molcas_active_space.
  7.  caspt2_intruder               — &CASPT2 Stop with _RC_NOT_CONVERGED_,
                                       reference weight below 0.7, small
                                       denominators visible. Fix: add
                                       imaginary shift 0.05-0.1.
  8.  caspt2_low_ref_weight         — &CASPT2 converged but ref weight in
                                       caution band. Same fix as 7.
  9.  jobiph_missing                — RASSI or per-group CASPT2 reads
                                       missing JOBxxx. Fix: insert >>COPY
                                       plumbing in the input.
  10. ga_segfault                   — "BAD TERMINATION ... SIGNAL 11" or
                                       SIGSEGV during &CASPT2 on -np > 1.
                                       Fix: rebuild container with GA
                                       --with-mpi-ts, OR force -np 1.
  11. memory_exceeded               — _RC_MEMORY_ERROR_. Fix: bump
                                       MOLCAS_MEM via >>>Export.
  12. slapaf_no_convergence         — SLAPAF reached opt iter limit
                                       without Geometry-converged. Fix:
                                       bump Iterations in &SLAPAF; check
                                       Hessian quality.

Each rule returns a fully-populated recovery dict so the caller can chain
next_actions directly into orchestrators.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from chemtools.core.common import read_text
from chemtools.programs.molcas.parse.output import parse_output_full


# ---------------------------------------------------------------------------
# Helper: regex patterns + light parsing
# ---------------------------------------------------------------------------

_RE_STOP_MODULE = re.compile(
    r"---\s*Stop Module:\s*(\w+)\s*at\s.*?/rc=(\S+)\s*---", re.IGNORECASE,
)
_RE_ANGSTROM_NOT_KEYWORD = re.compile(r"ANGSTROM is not a keyword", re.IGNORECASE)
_RE_NO_BASIS = re.compile(r"Input does not contain any basis sets", re.IGNORECASE)
_RE_DOUBLE_OCC_ONLY = re.compile(
    r"Current implementation only allows double occupations", re.IGNORECASE,
)
_RE_BAD_TERMINATION = re.compile(
    r"BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES|SIGSEGV|SIGNAL 11",
    re.IGNORECASE,
)
_RE_MEMORY_ERROR = re.compile(r"_RC_MEMORY_ERROR_|out of memory|insufficient memory", re.IGNORECASE)
# "No such wave function" is how OpenMolcas (≥25) reports an impossible
# Spin/Nactel(/Ras) combination from RASSCF — the older "Nactel not consistent"
# wording no longer appears.
_RE_NACTEL_PARITY = re.compile(
    r"Nactel.*not consistent|Spin and electrons do not match|No such wave function",
    re.IGNORECASE,
)

_RE_RASSCF_MAX_ITER = re.compile(r"Maximum number of macro iterations\s+(\d+)", re.IGNORECASE)
_RE_RASSCF_CONVERGED_AFTER = re.compile(
    r"Convergence after\s+(\d+)\s+(?:iterations?|Iterations?)", re.IGNORECASE,
)
_RE_CASPT2_REF_WEIGHT = re.compile(
    r"Reference weight:\s+([0-9.E+-]+)", re.IGNORECASE,
)
_RE_DOWHILE_ITER = re.compile(r"DO\s+WHILE\s+iter\s*:\s*(\d+)", re.IGNORECASE)
_RE_JOBIPH_MISSING = re.compile(
    r"JOBI(PH|XX)\s+(file is missing|cannot be opened|not found)",
    re.IGNORECASE,
)
_RE_GEOM_CONVERGED = re.compile(
    r"Geometry is converged in\s+\d+\s+iterations", re.IGNORECASE,
)
_RE_CONVERGENCE_PROBLEM = re.compile(r"^\s*\.+\s*#\s*Convergence problem", re.MULTILINE)


def _scan_stops(text: str) -> list[dict]:
    """Find every `Stop Module` line and its rc code."""
    return [
        {"module": m.group(1).lower(), "rc": m.group(2).upper().rstrip("_-")}
        for m in _RE_STOP_MODULE.finditer(text)
    ]


def _last_stop(text: str) -> dict | None:
    stops = _scan_stops(text)
    return stops[-1] if stops else None


def _excerpt_around(text: str, pattern: re.Pattern, before: int = 0, after: int = 8) -> str:
    """Return the matching line + surrounding context."""
    m = pattern.search(text)
    if not m:
        return ""
    all_lines = text.splitlines()
    idx = text.count("\n", 0, m.start())
    lo = max(0, idx - before)
    hi = min(len(all_lines), idx + after + 1)
    return "\n".join(all_lines[lo:hi])


def _last_caspt2_ref_weight(text: str) -> float | None:
    matches = list(_RE_CASPT2_REF_WEIGHT.finditer(text))
    return float(matches[-1].group(1)) if matches else None


# ---------------------------------------------------------------------------
# Recovery-rule signatures
# ---------------------------------------------------------------------------


def _rule_seward_angstrom_symmetry(text: str, _parsed: dict | None) -> dict | None:
    if not _RE_ANGSTROM_NOT_KEYWORD.search(text):
        return None
    return {
        "failure_class": "seward_angstrom_symmetry",
        "severity": "fatal",
        "module": "seward",
        "root_cause": (
            "SEWARD aborted because the `Angstrom` keyword is incompatible with the "
            "`Symmetry` block in this Molcas build. With Symmetry present, coordinates "
            "must be in bohr."
        ),
        "evidence": _excerpt_around(text, _RE_ANGSTROM_NOT_KEYWORD, before=2, after=4),
        "fix_recipe": [
            "Convert your geometry from Angstrom to bohr (multiply each x,y,z by 1.8897261245650618).",
            "Set geometry_units='bohr' in the drafter call (or omit it; bohr is Molcas's default).",
            "Or use the drafter's auto-conversion: pass geometry_units='angstrom' to the latest draft_molcas_input — the SEWARD block builder converts internally.",
        ],
        "next_actions": [
            {
                "tool": "draft_molcas_input",
                "rationale": "Re-draft with geometry_units='angstrom' — the SEWARD block builder converts Angstrom→bohr automatically and the keyword conflict is sidestepped.",
            }
        ],
    }


def _rule_missing_basis_in_loop(text: str, _parsed: dict | None) -> dict | None:
    if not _RE_NO_BASIS.search(text):
        return None
    return {
        "failure_class": "missing_basis_in_loop",
        "severity": "fatal",
        "module": "seward",
        "root_cause": (
            "SEWARD inside the opt loop couldn't find any basis sets — the initial "
            "block used `&SEWARD` instead of `&GATEWAY`, so the basis spec never "
            "persisted to the RunFile. On iter 2+, the loop's bare `&SEWARD` has "
            "nothing to read."
        ),
        "evidence": _excerpt_around(text, _RE_NO_BASIS, before=2, after=4),
        "fix_recipe": [
            "Replace the first `&SEWARD ... End of input` block with `&GATEWAY` carrying the basis + geometry, followed by a bare `&SEWARD` (or omit the outer SEWARD entirely).",
            "Or use prepare_molcas_opt_freq_workflow — it emits the GATEWAY/SEWARD split correctly when do_optimization=True.",
        ],
        "next_actions": [
            {
                "tool": "prepare_molcas_opt_freq_workflow",
                "args": {"do_optimization": True},
                "rationale": "Regenerate with the proper GATEWAY-for-loop pattern.",
            }
        ],
    }


def _rule_scf_single_electron(text: str, _parsed: dict | None) -> dict | None:
    if not _RE_DOUBLE_OCC_ONLY.search(text):
        return None
    return {
        "failure_class": "scf_single_electron",
        "severity": "fatal",
        "module": "scf",
        "root_cause": (
            "Molcas SCF cannot handle a species with only one electron (or any case "
            "needing a half-occupied orbital). The error 'Current implementation only "
            "allows double occupations' is the giveaway. Common on H atom and similar "
            "one-electron references."
        ),
        "evidence": _excerpt_around(text, _RE_DOUBLE_OCC_ONLY, before=2, after=4),
        "fix_recipe": [
            "Drop the &SCF block from the input entirely.",
            "Let RASSCF start from GuessOrb (Molcas SEWARD's one-electron Hamiltonian guess).",
            "If using prepare_molcas_atomization, the orchestrator now sets skip_scf=True automatically for H.",
        ],
        "next_actions": [
            {
                "tool": "prepare_molcas_atomization",
                "rationale": "The orchestrator's ATOMIC_GROUND_STATES table marks single-electron species as skip_scf=True automatically.",
            }
        ],
    }


def _rule_scf_no_convergence(text: str, _parsed: dict | None) -> dict | None:
    last = _last_stop(text)
    if not last or last["module"] != "scf" or "NOT_CONVERGED" not in last["rc"]:
        return None
    # Try to read the spin/multiplicity context
    mult = None
    m = re.search(r"^\s*Charge\b.*\n\s*0\s+(\d+)", text, re.MULTILINE)
    if m:
        mult = int(m.group(1)) + 1  # Molcas Charge=`q 2S` → mult = 2S+1
    severity = "fatal"
    likely_tm_atom = mult is not None and mult >= 4
    return {
        "failure_class": "scf_no_convergence",
        "severity": severity,
        "module": "scf",
        "root_cause": (
            f"SCF did not converge in the allotted iterations. "
            + ("This is a known issue for high-spin transition-metal atoms (Cr ⁷S, "
               "Mn ⁶S, Fe ⁵D, V ⁴F, Co ⁴F) where Molcas ROHF struggles to converge "
               "from GuessOrb." if likely_tm_atom else
               "Try level-shifting, more iterations, or a better initial guess.")
        ),
        "evidence": f"Stop Module: scf at /rc=_RC_NOT_CONVERGED_  (multiplicity={mult or '?'})",
        "fix_recipe": (
            [
                "Drop the &SCF block entirely; let RASSCF run from GuessOrb (works for high-spin TM atoms).",
                "If the next step is CASSCF/RASSCF, RASSCF only needs an orbital starting point — GuessOrb suffices.",
                "Use prepare_molcas_atomization which sets skip_scf=True for high-spin TMs automatically.",
            ]
            if likely_tm_atom else
            [
                "Increase SCF iterations via `Iterations 200` in the &SCF block.",
                "Add level shifting via `LevShift 0.5` (or larger).",
                "Use a previous run's ScfOrb file via `LumOrb` + `FILEORB` to warm-start.",
                "If on a transition-metal complex, consider DFT (KSDFT B3LYP) which often converges where HF doesn't.",
            ]
        ),
        "next_actions": [
            {
                "tool": "prepare_molcas_atomization" if likely_tm_atom else "draft_molcas_input",
                "rationale": (
                    "Re-draft skipping SCF. The atomization orchestrator handles this automatically for TM atoms."
                    if likely_tm_atom else
                    "Re-draft with higher SCF iter ceiling + level shifting."
                ),
            }
        ],
    }


def _rule_rasscf_no_convergence(text: str, parsed: dict | None) -> dict | None:
    last = _last_stop(text)
    if not last or last["module"] != "rasscf" or "NOT_CONVERGED" not in last["rc"]:
        return None
    max_iter_m = _RE_RASSCF_MAX_ITER.search(text)
    max_iter = int(max_iter_m.group(1)) if max_iter_m else None
    return {
        "failure_class": "rasscf_no_convergence",
        "severity": "fatal",
        "module": "rasscf",
        "root_cause": (
            f"RASSCF reached its iteration budget ({max_iter or '?'} macro iters) "
            "without converging. Common on transition-metal CAS where the orbital "
            "rotation needs 40-60+ iterations from GuessOrb. Subsequent modules "
            "(CASPT2, ALASKA, SLAPAF) will not run."
        ),
        "evidence": f"Stop Module: rasscf at /rc=_RC_NOT_CONVERGED_  (Maximum macro iters: {max_iter})",
        "fix_recipe": [
            "Bump RASSCF iteration ceiling to (100, 50) or higher: set program_options.rasscf.iterations=(100, 50) in draft_molcas_input.",
            "Or use prepare_molcas_atomization — it now defaults to (100, 50) for both molecule and atomic refs.",
            "If still not converging: provide starting orbitals via FILEORB (a converged RASSCF from a smaller CAS or related geometry).",
            "Check the active-space verdict via analyze_molcas_active_space — a 'poor' verdict suggests the CAS itself is wrong.",
        ],
        "next_actions": [
            {
                "tool": "draft_molcas_input",
                "args": {"program_options": {"rasscf": {"iterations": [100, 50]}}},
                "rationale": "Re-run with a roomier RASSCF iteration budget.",
            },
            {
                "tool": "analyze_molcas_active_space",
                "rationale": "Even partial RASSCF iterations can be inspected — verify the active space character isn't grossly wrong.",
            },
        ],
    }


def _rule_caspt2_intruder_or_low_ref_weight(text: str, parsed: dict | None) -> dict | None:
    # Two cases: CASPT2 didn't converge (intruder), or it converged but ref weight is low.
    stops = _scan_stops(text)
    caspt2_stop = next((s for s in reversed(stops) if s["module"] == "caspt2"), None)
    ref_weight = _last_caspt2_ref_weight(text)
    if caspt2_stop is None:
        return None
    caspt2_failed = "NOT_CONVERGED" in caspt2_stop["rc"]
    if not caspt2_failed and (ref_weight is None or ref_weight >= 0.85):
        return None
    failure_class = "caspt2_intruder" if caspt2_failed else "caspt2_low_ref_weight"
    severity = "fatal" if caspt2_failed else "caution"
    fix_recipe = [
        "Add an imaginary shift to suppress intruder denominators: `Imaginary 0.1` (or 0.05 for milder cases) in the &CASPT2 block.",
        "If using prepare_molcas_atomization, the orchestrator now auto-sets imaginary_shift=0.1 when DKH is applied (TM systems).",
        "If using prepare_molcas_caspt2_chain, pass imaginary_shift=0.1 explicitly.",
    ]
    if ref_weight is not None and ref_weight < 0.70:
        fix_recipe.append(
            f"Reference weight is {ref_weight:.3f} — below the 0.70 trust threshold. "
            "Beyond intruder protection, consider whether the active space is too small "
            "or the wrong orbitals are active. Run validate_molcas_caspt2_setup."
        )
    return {
        "failure_class": failure_class,
        "severity": severity,
        "module": "caspt2",
        "root_cause": (
            f"CASPT2 {'failed to converge' if caspt2_failed else 'converged but reference weight is poor'} "
            f"(ref_weight={ref_weight if ref_weight is not None else 'unknown'}). "
            "Small denominators in the perturbation expansion (intruder states) blow up "
            "the amplitudes. For DKH-CASPT2 on TM systems with diffuse virtuals, this is "
            "the default failure mode without an imaginary shift."
        ),
        "evidence": f"CASPT2 stop rc={caspt2_stop['rc']}, ref_weight={ref_weight}",
        "ref_weight": ref_weight,
        "fix_recipe": fix_recipe,
        "next_actions": [
            {
                "tool": "validate_molcas_caspt2_setup",
                "rationale": "Confirm whether the ref weight + intruder pattern is fixable by shift or requires CAS expansion.",
            },
            {
                "tool": "prepare_molcas_caspt2_chain",
                "args": {"imaginary_shift": 0.1},
                "rationale": "Re-run CASPT2 with imaginary shift 0.1.",
            },
        ],
    }


def _rule_jobiph_missing(text: str, _parsed: dict | None) -> dict | None:
    if not _RE_JOBIPH_MISSING.search(text):
        return None
    return {
        "failure_class": "jobiph_missing",
        "severity": "fatal",
        "module": "rassi/caspt2",
        "root_cause": (
            "A downstream module (RASSI, or a per-group CASPT2 in an excited-state "
            "workflow) tried to read a JOBxxx file that wasn't written. Usually means "
            "a missing `>>COPY $Project.JobIph JOBxxx` between the producing RASSCF "
            "and the consuming module, OR a missing `>>COPY JOBxxx $Project.JobIph` "
            "before a per-group CASPT2."
        ),
        "evidence": _excerpt_around(text, _RE_JOBIPH_MISSING, before=2, after=4),
        "fix_recipe": [
            "Use prepare_molcas_excited_states which emits the full COPY-dance correctly.",
            "Manually: after each RASSCF emit `>>COPY $Project.JobIph JOB00X`; before each per-group CASPT2 emit `>>COPY JOB00X $Project.JobIph` to restore the right wave function.",
        ],
        "next_actions": [
            {
                "tool": "prepare_molcas_excited_states",
                "rationale": "Re-generate the input with correct JobIph plumbing.",
            }
        ],
    }


def _rule_ga_segfault(text: str, _parsed: dict | None) -> dict | None:
    if not _RE_BAD_TERMINATION.search(text):
        return None
    in_caspt2 = bool(re.search(r"Start Module: caspt2.*?\Z", text, re.DOTALL))
    return {
        "failure_class": "ga_segfault",
        "severity": "fatal",
        "module": "caspt2" if in_caspt2 else "unknown",
        "root_cause": (
            "Parallel MPI process crashed (SIGSEGV or BAD TERMINATION). "
            + ("In CASPT2 specifically, this is the Global Arrays library segfaulting "
               "because the GA build defaulted to a network mode that doesn't survive "
               "Intel MPI in apptainer/Singularity containers. " if in_caspt2 else "")
            + "Most often a build-time GA configuration issue."
        ),
        "evidence": _excerpt_around(text, _RE_BAD_TERMINATION, before=4, after=4),
        "fix_recipe": [
            "Rebuild the OpenMolcas container with `--with-mpi-ts` added to the GA `./configure` line. "
            "This forces two-sided MPI (most reliable single-node).",
            "Workaround without rebuilding: force `-np 1` for runs containing &CASPT2. The runner profile's `programs.molcas.parallel_caspt2_supported=False` flag does this automatically.",
            "Container fix lives at /home/charlie/projects/mycontainers/docker/molcas/Dockerfile-openmolcas-26.02 (verified working).",
        ],
        "next_actions": [
            {
                "tool": "prepare_molcas_launch",
                "args": {"requested_np": 1},
                "rationale": "Re-run serially as a workaround until the GA build is fixed.",
            }
        ],
    }


def _rule_memory_exceeded(text: str, _parsed: dict | None) -> dict | None:
    if not _RE_MEMORY_ERROR.search(text):
        return None
    # Look for current MOLCAS_MEM
    cur = re.search(r"MOLCAS_MEM=(\d+)", text)
    cur_mb = int(cur.group(1)) if cur else None
    suggested = (cur_mb or 4000) * 2
    return {
        "failure_class": "memory_exceeded",
        "severity": "fatal",
        "module": "unknown",
        "root_cause": (
            f"Molcas ran out of memory "
            + (f"(current MOLCAS_MEM={cur_mb} MB). " if cur_mb else ". ")
            + "CASPT2 / RASSI / large CAS typically need 4-16 GB; integrals + density matrices on TM systems can spike."
        ),
        "evidence": _excerpt_around(text, _RE_MEMORY_ERROR, before=2, after=4),
        "current_memory_mb": cur_mb,
        "fix_recipe": [
            f"Bump MOLCAS_MEM. Try {suggested} MB: set memory_mb={suggested} in program_options.",
            "Use `>>> Export MOLCAS_MEM={suggested}` at the top of the input.",
            "For very large CASPT2: enable Cholesky decomposition (&SEWARD `Cholesky`) to reduce memory footprint at modest accuracy cost.",
        ],
        "next_actions": [
            {
                "tool": "draft_molcas_input",
                "args": {"program_options": {"memory_mb": suggested}},
                "rationale": f"Re-run with MOLCAS_MEM={suggested} MB.",
            }
        ],
    }


def _rule_slapaf_no_convergence(text: str, _parsed: dict | None) -> dict | None:
    if not _RE_CONVERGENCE_PROBLEM.search(text):
        return None
    if _RE_GEOM_CONVERGED.search(text):
        return None
    # How many opt iters ran?
    iters = list(_RE_DOWHILE_ITER.finditer(text))
    n_iters = int(iters[-1].group(1)) if iters else None
    return {
        "failure_class": "slapaf_no_convergence",
        "severity": "fatal",
        "module": "slapaf",
        "root_cause": (
            f"Geometry optimization (SLAPAF) hit the iteration limit "
            + (f"({n_iters} iterations) " if n_iters else "")
            + "without converging. Common when the initial Hessian is poor "
            "(particularly for TS searches) or the starting geometry is far from minimum."
        ),
        "evidence": f"SLAPAF iter limit reached ({n_iters or '?'} iterations), no 'Geometry is converged' marker.",
        "iterations_taken": n_iters,
        "fix_recipe": [
            "Bump SLAPAF Iterations via the orchestrator: `max_opt_iterations=100` (or more).",
            "For TS searches, compute an analytic Hessian first by running &MCKINLEY + &MCLR BEFORE the SLAPAF loop.",
            "Check the SLAPAF output for which coordinates are oscillating — often a single bad mode dominates and constraints help.",
            "If the starting geometry is bad, generate a better one (e.g. by relaxing structural features first).",
        ],
        "next_actions": [
            {
                "tool": "prepare_molcas_opt_freq_workflow",
                "args": {"max_opt_iterations": 100},
                "rationale": "Re-run with a higher iteration ceiling.",
            }
        ],
    }


def _rule_nactel_parity(text: str, _parsed: dict | None) -> dict | None:
    if not _RE_NACTEL_PARITY.search(text):
        return None
    return {
        "failure_class": "nactel_parity",
        "severity": "fatal",
        "module": "rasscf",
        "root_cause": (
            "RASSCF rejected the active-electron count vs spin combination. "
            "The number of unpaired electrons (2S = multiplicity − 1) must have "
            "the same parity as Nactel (i.e. for triplet spin=1, Nactel must be ≥2 and have the same parity as 2)."
        ),
        "evidence": _excerpt_around(text, _RE_NACTEL_PARITY, before=2, after=4),
        "fix_recipe": [
            "Recompute active-electron count: spin (= multiplicity - 1) unpaired + remaining must total Nactel.",
            "Use compute_molcas_active_space_partition for a consistent (M, N) → per-symmetry partition that respects parity.",
            "If using a chemistry_hint='valence_d': prepare_molcas_casscf_setup auto-corrects parity for TM hints.",
        ],
        "next_actions": [
            {
                "tool": "compute_molcas_active_space_partition",
                "rationale": "Validate the (M, N) + multiplicity combination is internally consistent.",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------


# Order matters — most specific signatures first, generic catch-alls last.
_RULES: list[Callable[[str, dict | None], dict | None]] = [
    _rule_seward_angstrom_symmetry,
    _rule_missing_basis_in_loop,
    _rule_scf_single_electron,
    _rule_nactel_parity,
    _rule_jobiph_missing,
    _rule_memory_exceeded,
    _rule_ga_segfault,
    _rule_scf_no_convergence,
    _rule_rasscf_no_convergence,
    _rule_caspt2_intruder_or_low_ref_weight,
    _rule_slapaf_no_convergence,
]


def suggest_recovery(
    output_file: str,
    *,
    return_all_matches: bool = False,
) -> dict[str, Any]:
    """Classify a Molcas .out / .log failure and suggest a recovery plan.

    Thin Molcas-specific dispatcher that:
      1. Reads the file
      2. Parses the output (best-effort; tolerates malformed runs)
      3. Detects Molcas "Stop Module" lines and decides if the run completed
      4. Delegates to ``chemtools.core.recovery.dispatch_rules`` for the
         actual rule-walking + result shaping

    Returns
    -------
    dict with:
      verdict           "success_no_recovery_needed" | "recovery_suggested" | "unknown_failure"
      output_file       path
      ran_to_completion bool
      last_module       str | None — last Stop Module name
      last_rc           str | None — last Stop Module return code
      recovery          dict — primary recovery record (see rule docstrings)
      all_matches       list — only present if return_all_matches=True
    """
    from chemtools.core.recovery import dispatch_rules

    output_path = Path(output_file)
    if not output_path.is_file():
        raise FileNotFoundError(f"Output file not found: {output_file}")
    text = read_text(output_file)

    # Best-effort parse; some failures abort early before tasks are reported.
    try:
        parsed: dict | None = parse_output_full(output_file, text)
    except Exception:  # noqa: BLE001
        parsed = None

    stops = _scan_stops(text)
    last_stop = stops[-1] if stops else None
    # "Ran to completion" heuristic: every Stop Module returned _RC_ALL_IS_WELL_
    # OR _RC_CONTINUE_LOOP_ / _RC_INVOKED_OTHER_MODULE_ (normal opt/control rcs).
    benign_rcs = {"_RC_ALL_IS_WELL", "_RC_CONTINUE_LOOP", "_RC_INVOKED_OTHER_MODULE"}
    ran_clean = bool(stops) and all(
        any(s["rc"].startswith(b) for b in benign_rcs) for s in stops
    )

    return dispatch_rules(
        rules=_RULES,
        text=text,
        parsed=parsed,
        output_file=output_file,
        ran_clean=ran_clean,
        last_module=last_stop["module"] if last_stop else None,
        last_rc=last_stop["rc"] if last_stop else None,
        return_all_matches=return_all_matches,
        unknown_failure_next_actions=[
            {
                "tool": "parse_molcas_tasks",
                "args": {"output_file": output_file},
                "rationale": "Get a coarse task-by-task summary to localize where the run went off the rails.",
            },
            {
                "tool": "parse_molcas_output",
                "args": {"output_file": output_file},
                "rationale": "Get the deep parse — warnings + module-level details often surface the root cause when none of the known patterns match.",
            },
        ],
    )


# ---------------------------------------------------------------------------
# apply_recovery — auto-fix the input file based on a recovery classification
# ---------------------------------------------------------------------------


# Failure classes whose fix is a mechanical regex edit on the input deck.
_MECHANICAL_FIX_CLASSES = {
    "scf_no_convergence",
    "scf_single_electron",
    "rasscf_no_convergence",
    "caspt2_intruder",
    "caspt2_low_ref_weight",
    "memory_exceeded",
    "missing_basis_in_loop",
    "slapaf_no_convergence",
}


def _drop_scf_block(text: str) -> tuple[str, str]:
    """Remove the &SCF ... End of input block. Returns (new_text, message)."""
    new = re.sub(r"&SCF &END.*?End of input\n+", "", text, flags=re.DOTALL, count=1)
    if new == text:
        return text, "no &SCF block found to drop"
    return new, "dropped &SCF block — RASSCF will start from GuessOrb"


def _bump_rasscf_iterations(text: str, new_iters: tuple[int, int] = (100, 50)) -> tuple[str, str]:
    """Increase the `Iteration N,M` line inside &RASSCF."""
    new_line = f" {new_iters[0]},{new_iters[1]}"
    # Pattern: `Iteration\n NN,MM`
    pat = re.compile(r"(Iteration\n)\s*\d+\s*,\s*\d+", re.IGNORECASE)
    if not pat.search(text):
        return text, "no &RASSCF Iteration line found"
    new = pat.sub(rf"\g<1>{new_line}", text, count=1)
    return new, f"bumped &RASSCF Iteration to {new_iters[0]},{new_iters[1]}"


def _add_imaginary_shift(text: str, shift: float = 0.1) -> tuple[str, str]:
    """Add or update the `Imaginary` keyword in &CASPT2."""
    # If already present, update value
    pat_existing = re.compile(r"(Imaginary\n)\s*[0-9.]+", re.IGNORECASE)
    if pat_existing.search(text):
        new = pat_existing.sub(rf"\g<1> {shift:.3f}", text, count=1)
        return new, f"updated Imaginary shift to {shift}"
    # Else inject before the `End of input` of the &CASPT2 block
    pat_caspt2_end = re.compile(
        r"(&CASPT2[\s\S]*?)(End of input\n)", re.IGNORECASE,
    )
    m = pat_caspt2_end.search(text)
    if not m:
        return text, "no &CASPT2 block found to add Imaginary shift to"
    new = (
        text[: m.start(2)]
        + f"Imaginary\n {shift:.3f}\n"
        + text[m.start(2):]
    )
    return new, f"added Imaginary {shift} to &CASPT2 block"


def _bump_molcas_mem(text: str, new_mb: int) -> tuple[str, str]:
    """Bump the `>>> Export MOLCAS_MEM=...` line."""
    pat = re.compile(r"(MOLCAS_MEM\s*=\s*)\d+", re.IGNORECASE)
    if not pat.search(text):
        # Inject at top
        new = f">>> Export MOLCAS_MEM={new_mb}\n" + text
        return new, f"injected >>> Export MOLCAS_MEM={new_mb} at top"
    new = pat.sub(rf"\g<1>{new_mb}", text, count=1)
    return new, f"bumped MOLCAS_MEM to {new_mb} MB"


def _seward_to_gateway(text: str) -> tuple[str, str]:
    """Replace the first `&SEWARD &END` block opening with `&GATEWAY`.

    For an opt loop, the leading basis-carrying block must be &GATEWAY so
    the basis persists to RunFile for inner-loop &SEWARD calls.
    """
    pat = re.compile(r"&SEWARD &END", re.IGNORECASE)
    if not pat.search(text):
        return text, "no &SEWARD &END opening found"
    new = pat.sub("&GATEWAY", text, count=1)
    return new, "replaced opening &SEWARD &END with &GATEWAY"


def _bump_slapaf_iterations(text: str, new_iters: int = 100) -> tuple[str, str]:
    """Add or bump the `Iterations N` keyword inside &SLAPAF."""
    # First check if Iterations already present in SLAPAF block
    pat_existing = re.compile(
        r"(&SLAPAF[\s\S]*?Iterations\n)\s*\d+",
        re.IGNORECASE,
    )
    m = pat_existing.search(text)
    if m:
        new = pat_existing.sub(rf"\g<1> {new_iters}", text, count=1)
        return new, f"bumped &SLAPAF Iterations to {new_iters}"
    # Inject inside SLAPAF before End of input
    pat_slapaf = re.compile(r"(&SLAPAF[\s\S]*?)(End of input)", re.IGNORECASE)
    m = pat_slapaf.search(text)
    if not m:
        return text, "no &SLAPAF block found"
    new = text[: m.start(2)] + f"Iterations\n {new_iters}\n" + text[m.start(2):]
    return new, f"injected Iterations {new_iters} into &SLAPAF block"


def apply_recovery(
    input_file: str,
    *,
    output_file: str | None = None,
    recovery: dict | None = None,
    write_to: str | None = None,
) -> dict[str, Any]:
    """Apply a recovery fix to a Molcas input file.

    Two ways to call:
      apply_recovery(input_file, output_file=path_to_failed_out)
        → auto-classifies the failure via suggest_recovery, then applies the fix.
      apply_recovery(input_file, recovery={'failure_class': ..., ...})
        → use a pre-computed recovery dict (avoids re-parsing).

    Parameters
    ----------
    input_file
        Path to the .input file that produced the failure.
    output_file
        Path to the failed .out/.log to classify. Mutually exclusive with
        ``recovery``.
    recovery
        Pre-computed recovery dict (e.g. from suggest_recovery's `recovery`
        field). Mutually exclusive with ``output_file``.
    write_to
        Path to write the fixed input. If None, defaults to inserting
        ``_recovered`` before the .input suffix.

    Returns dict with:
      verdict           "fix_applied" | "manual_intervention_required" | "no_recovery_needed" | "unknown_failure"
      failure_class     str | None
      changes_applied   list[str] — human-readable per-edit description
      input_path        path to the new (fixed) input file (if fix_applied)
      diagnostics       full recovery dict (preserved for the agent)
      next_actions      ready-to-run launch command(s) if fix_applied
    """
    in_path = Path(input_file)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    input_text = in_path.read_text(encoding="utf-8")
    if re.search(
        (
            r"(?mi)^\s*&(GATEWAY|SEWARD|SCF|RASSCF|CASPT2|"
            r"RASSI|SLAPAF|ALASKA)\b"
        ),
        input_text,
    ) is None:
        return {
            "error": "input_format_mismatch",
            "verdict": "input_format_mismatch",
            "failure_class": None,
            "changes_applied": [],
            "input_path": str(in_path),
            "message": (
                "The recovery patcher requires a recognizable Molcas input "
                "and will not edit this file."
            ),
            "next_actions": [],
        }

    if recovery is None and output_file is None:
        raise ValueError(
            "Pass either output_file (to auto-classify) or recovery (pre-computed dict)."
        )
    if recovery is None:
        diag = suggest_recovery(output_file)  # type: ignore[arg-type]
        if diag["verdict"] == "success_no_recovery_needed":
            return {
                "verdict": "no_recovery_needed",
                "failure_class": None,
                "changes_applied": [],
                "input_path": str(in_path),
                "diagnostics": diag,
                "next_actions": [],
            }
        if diag["verdict"] == "unknown_failure":
            return {
                "verdict": "unknown_failure",
                "failure_class": None,
                "changes_applied": [],
                "input_path": str(in_path),
                "diagnostics": diag,
                "next_actions": diag.get("next_actions", []),
            }
        recovery = diag["recovery"]

    failure_class = recovery.get("failure_class")  # type: ignore[union-attr]
    if failure_class not in _MECHANICAL_FIX_CLASSES:
        return {
            "verdict": "manual_intervention_required",
            "failure_class": failure_class,
            "changes_applied": [],
            "input_path": str(in_path),
            "diagnostics": {"recovery": recovery},
            "reason": (
                f"Failure class '{failure_class}' requires chemistry judgment or a "
                "rebuild step that cannot be automated by a regex edit on the input. "
                "See recovery.fix_recipe for the steps to take manually."
            ),
            "next_actions": recovery.get("next_actions", []),  # type: ignore[union-attr]
        }

    text = input_text
    changes: list[str] = []

    if failure_class in {"scf_no_convergence", "scf_single_electron"}:
        text, msg = _drop_scf_block(text)
        changes.append(msg)
    elif failure_class == "rasscf_no_convergence":
        text, msg = _bump_rasscf_iterations(text, new_iters=(100, 50))
        changes.append(msg)
    elif failure_class in {"caspt2_intruder", "caspt2_low_ref_weight"}:
        text, msg = _add_imaginary_shift(text, shift=0.1)
        changes.append(msg)
    elif failure_class == "memory_exceeded":
        current = recovery.get("current_memory_mb") or 4000  # type: ignore[union-attr]
        new_mb = current * 2
        text, msg = _bump_molcas_mem(text, new_mb=new_mb)
        changes.append(msg)
    elif failure_class == "missing_basis_in_loop":
        text, msg = _seward_to_gateway(text)
        changes.append(msg)
    elif failure_class == "slapaf_no_convergence":
        text, msg = _bump_slapaf_iterations(text, new_iters=100)
        changes.append(msg)

    # Write the fixed input
    if write_to:
        out_path = Path(write_to)
    else:
        suffix = in_path.suffix or ".input"
        stem = in_path.stem
        out_path = in_path.parent / f"{stem}_recovered{suffix}"
    out_path.write_text(text, encoding="utf-8")

    return {
        "verdict": "fix_applied",
        "failure_class": failure_class,
        "changes_applied": changes,
        "input_path": str(out_path),
        "original_input_path": str(in_path),
        "diagnostics": {"recovery": recovery},
        "next_actions": [
            {
                "tool": "prepare_molcas_launch",
                "args": {"input_file": str(out_path)},
                "rationale": (
                    f"Recovered from '{failure_class}' by: {'; '.join(changes)}. "
                    "Re-run the fixed input."
                ),
            },
            {
                "tool": "suggest_molcas_recovery",
                "args": {"output_file": str(out_path).replace(".input", ".log")},
                "rationale": "After the rerun, classify again — some failures are layered.",
            },
        ],
    }


# ---------------------------------------------------------------------------
# try_run_with_recovery — full auto-loop (local-mode tool)
# ---------------------------------------------------------------------------


def try_run_with_recovery(
    input_file: str,
    *,
    pymolcas_command: list[str] | None = None,
    apptainer_sif: str | None = None,
    np_processes: int = 1,
    max_retries: int = 3,
    extra_env: dict[str, str] | None = None,
    timeout_per_attempt: float = 600.0,
) -> dict[str, Any]:
    """Run a Molcas input via pymolcas; on failure, auto-classify + apply
    recovery + re-run, up to ``max_retries`` times.

    This is a local-mode tool — it spawns pymolcas as a subprocess. Use only
    when the agent's environment can execute Molcas directly.

    Parameters
    ----------
    input_file
        Path to the .input deck to run.
    pymolcas_command
        Override the pymolcas invocation (default: ``['pymolcas']`` if no
        apptainer_sif, else apptainer-wrapped).
    apptainer_sif
        Path to an apptainer .sif image. If set, pymolcas is invoked via
        ``apptainer exec <sif> pymolcas``.
    np_processes
        ``-np N`` for pymolcas. Default 1 (safest against GA/MPI issues).
    max_retries
        Max number of recovery cycles before giving up. Default 3.
    extra_env
        Extra environment vars to pass to pymolcas (e.g. MOLCAS_PROJECT for
        scratch isolation).
    timeout_per_attempt
        Seconds per pymolcas attempt before SIGKILL.

    Returns dict with:
      verdict           "converged" | "max_retries_exhausted" | "non_recoverable_failure"
      attempts          list of {attempt, input_path, log_path, suggest, apply, exit_code}
      final_input       path to the input that produced the converged run (if converged)
      final_log         path to the converged .log
      recovery_history  list of failure_class strings applied across attempts
    """
    import os
    import subprocess

    in_path = Path(input_file)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if pymolcas_command is None:
        if apptainer_sif:
            pymolcas_command = ["apptainer", "exec", apptainer_sif, "pymolcas"]
        else:
            pymolcas_command = ["pymolcas"]

    env = {**os.environ, **(extra_env or {})}

    attempts: list[dict[str, Any]] = []
    recovery_history: list[str] = []
    current_input = in_path

    for attempt_idx in range(max_retries + 1):  # +1 because attempt 0 is the initial run
        log_path = Path(str(current_input).replace(".input", ".log"))
        # pymolcas -f writes to <stem>.log automatically
        cmd = pymolcas_command + ["-np", str(np_processes), "-f", str(current_input)]
        try:
            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True,
                timeout=timeout_per_attempt, cwd=str(current_input.parent),
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            exit_code = -1
            proc = None  # type: ignore[assignment]

        record: dict[str, Any] = {
            "attempt": attempt_idx + 1,
            "input_path": str(current_input),
            "log_path": str(log_path),
            "exit_code": exit_code,
        }

        # Classify the result
        if log_path.is_file():
            diag = suggest_recovery(str(log_path))
        else:
            diag = {
                "verdict": "unknown_failure",
                "failure_class": None,
                "last_module": None,
                "last_rc": None,
                "recovery": None,
                "note": "No log file produced — pymolcas likely failed to start or hit timeout.",
            }
        record["suggest_verdict"] = diag["verdict"]
        record["failure_class"] = diag.get("failure_class")

        if diag["verdict"] == "success_no_recovery_needed":
            attempts.append(record)
            return {
                "verdict": "converged",
                "attempts": attempts,
                "final_input": str(current_input),
                "final_log": str(log_path),
                "recovery_history": recovery_history,
            }

        if diag["verdict"] == "unknown_failure":
            attempts.append(record)
            return {
                "verdict": "non_recoverable_failure",
                "attempts": attempts,
                "final_input": str(current_input),
                "final_log": str(log_path),
                "recovery_history": recovery_history,
                "reason": (
                    "Diagnosis returned 'unknown_failure' — the failure pattern "
                    "doesn't match any of the 11 known rules. Inspect the log "
                    "manually."
                ),
            }

        # recovery_suggested — try to apply the mechanical fix
        if attempt_idx >= max_retries:
            attempts.append(record)
            return {
                "verdict": "max_retries_exhausted",
                "attempts": attempts,
                "final_input": str(current_input),
                "final_log": str(log_path),
                "recovery_history": recovery_history,
                "last_recovery": diag["recovery"],
            }

        # Apply the recovery
        apply_result = apply_recovery(
            input_file=str(current_input),
            recovery=diag["recovery"],
        )
        record["apply_verdict"] = apply_result["verdict"]
        record["apply_changes"] = apply_result.get("changes_applied", [])

        if apply_result["verdict"] != "fix_applied":
            attempts.append(record)
            return {
                "verdict": "non_recoverable_failure",
                "attempts": attempts,
                "final_input": str(current_input),
                "final_log": str(log_path),
                "recovery_history": recovery_history,
                "reason": (
                    f"Classification was '{diag['failure_class']}' but the fix "
                    f"requires manual intervention "
                    f"(verdict={apply_result['verdict']}). "
                    + str(apply_result.get("reason") or "")
                ),
            }

        recovery_history.append(diag["failure_class"])
        attempts.append(record)
        current_input = Path(apply_result["input_path"])

    # Shouldn't reach here normally
    return {
        "verdict": "max_retries_exhausted",
        "attempts": attempts,
        "final_input": str(current_input),
        "recovery_history": recovery_history,
    }

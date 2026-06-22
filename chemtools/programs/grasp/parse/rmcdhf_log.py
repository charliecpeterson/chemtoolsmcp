"""Parse ``rmcdhf`` SCF iteration trace from stdout / .alog / run log.

GRASP's rmcdhf prints::

    Iteration number   1
    --------------------
    Average energy =  -3.3054459526D+04 Hartrees
    Average energy =  -3.3054878567D+04 Hartrees   (one per optimized level/block)
    ...
    Iteration number   2
    ...

The Average energy lines repeat per optimized level inside each iteration.
We collect them and report the iter-by-iter average across all blocks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_ITER_RE = re.compile(r"^\s*Iteration number\s+(\d+)\s*$", re.M)
_AVG_E_RE = re.compile(r"Average energy\s*=\s*(-?\d+\.\d+(?:[DdEe][+-]?\d+)?)\s+Hartrees")
_FLOAT_RE = r"-?\d+\.\d+(?:[DdEe][+-]?\d+)?"

# Convergence diagnostic patterns
_MAX_CONV_RE = re.compile(r"maximum convergence parameter is\s*=?\s*(" + _FLOAT_RE + r")")
_NOT_CONVERGED_RE = re.compile(r"SCF not converged|did NOT converge|NOT converged|orbitals diverging", re.I)
_CONVERGED_RE = re.compile(r"Convergence \(latest difference.*\) satisfied|RMCDHF: Execution complete", re.I)

# Hard orbital-solver crash: the radial equation for a given orbital cannot be
# solved (common for a diffuse valence orbital — e.g. 7s in a neutral actinide —
# from a poor Thomas-Fermi start). The "Failure; equation..." line is fatal
# (ERROR STOP follows); the "Method N unable to solve" lines are per-attempt and
# can appear before a successful retry, so they're warnings, not the verdict.
_ORBITAL_FAILED_RE = re.compile(r"equation for orbital\s+(\S+)\s+could not be solved", re.I)
_METHOD_UNABLE_RE = re.compile(r"Method\s+\d+\s+unable to solve for\s+(\S+)\s+orbital", re.I)
_ERROR_STOP_RE = re.compile(r"^\s*ERROR STOP|Error termination", re.M)


def parse_rmcdhf_log(text_or_path: str) -> dict[str, Any]:
    """Parse the rmcdhf SCF iteration trace.

    Returns
    -------
    dict with::

        {
          "iterations": [{iter, avg_energies_hartree, mean_energy_hartree}, ...],
          "n_iterations": int,
          "final_energy_hartree": float | None,
          "energy_change": float | None,
          "converged": bool,
        }
    """
    text = _as_text(text_or_path)
    iter_matches = list(_ITER_RE.finditer(text))

    iterations: list[dict[str, Any]] = []
    # Split energy stream by iteration boundaries
    for i, m in enumerate(iter_matches):
        start = m.end()
        end = iter_matches[i + 1].start() if i + 1 < len(iter_matches) else len(text)
        chunk = text[start:end]
        chunk_energies = [_to_float(em.group(1)) for em in _AVG_E_RE.finditer(chunk)]
        iterations.append({
            "iter": int(m.group(1)),
            "avg_energies_hartree": chunk_energies,
            "mean_energy_hartree": (
                sum(chunk_energies) / len(chunk_energies) if chunk_energies else None
            ),
        })

    # Find the last iteration with a populated mean energy (GRASP sometimes
    # writes "Iteration number N" before the per-block "Average energy" lines
    # land in the log, so the trailing iter may be empty).
    iters_with_energy = [it for it in iterations if it["mean_energy_hartree"] is not None]
    final_e = iters_with_energy[-1]["mean_energy_hartree"] if iters_with_energy else None
    e_change = None
    if len(iters_with_energy) >= 2:
        prev = iters_with_energy[-2]["mean_energy_hartree"]
        curr = iters_with_energy[-1]["mean_energy_hartree"]
        e_change = curr - prev

    converged = bool(_CONVERGED_RE.search(text))
    not_converged = bool(_NOT_CONVERGED_RE.search(text))

    # Orbitals that the radial solver gave up on (de-duplicated, order-preserved).
    failed_orbitals: list[str] = []
    for m in _ORBITAL_FAILED_RE.finditer(text):
        if m.group(1) not in failed_orbitals:
            failed_orbitals.append(m.group(1))
    orbital_solver_failed = bool(failed_orbitals)
    error_stop = bool(_ERROR_STOP_RE.search(text))
    # Orbitals where a method attempt failed (may have recovered on retry).
    struggled_orbitals = sorted({m.group(1) for m in _METHOD_UNABLE_RE.finditer(text)})

    # A hard solver failure or an ERROR STOP overrides any stale "complete" marker.
    if orbital_solver_failed or error_stop or not_converged:
        converged = False

    return {
        "iterations": iterations,
        "n_iterations": len(iterations),
        "final_energy_hartree": final_e,
        "energy_change": e_change,
        "converged": converged,
        "explicitly_not_converged": not_converged,
        "orbital_solver_failed": orbital_solver_failed,
        "failed_orbitals": failed_orbitals,
        "struggled_orbitals": struggled_orbitals,
        "error_stop": error_stop,
    }


def _as_text(path_or_text: str) -> str:
    if "\n" in path_or_text or not Path(path_or_text).exists():
        return path_or_text
    return Path(path_or_text).read_text(encoding="utf-8", errors="replace")


def _to_float(s: str) -> float:
    return float(s.replace("D", "E").replace("d", "e"))

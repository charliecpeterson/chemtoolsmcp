"""Parse the ``rmcdhf`` SCF trace written to stdout.

The ``Average energy`` record is not an ASF total energy. GRASP prints the
state energies separately as ``Level ... Energy`` records. Those level records
are the only stdout values exposed as electronic energies here; ``.sum``
remains the authoritative final result artifact.
"""

from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path
from typing import Any

_ITER_RE = re.compile(r"^\s*Iteration number\s+(\d+)\s*$", re.M)
_AVG_E_RE = re.compile(r"Average energy\s*=\s*(-?\d+\.\d+(?:[DdEe][+-]?\d+)?)\s+Hartrees")
_FLOAT_RE = r"[+-]?\d+\.\d+(?:[DdEe][+-]?\d+)?"
_LEVEL_E_RE = re.compile(
    r"^\s*Level\s+(\d+)\s+Energy\s*=\s*(" + _FLOAT_RE + r")"
    r"(?:\s+Weight\s*=\s*(" + _FLOAT_RE + r"))?\s*$",
    re.M,
)
_ORBITAL_ROW_RE = re.compile(
    r"^\s*(\d+[A-Za-z]+-?)\s+(" + _FLOAT_RE + r")\s+"
    r"(\d+)\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+"
    r"(" + _FLOAT_RE + r")\s+([+-]?\d+(?:\.\d+)?)\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$",
    re.M,
)
_CSF_RE = re.compile(r"There are\s+(\d+)\s+relativistic CSFs")
_SUBSHELL_RE = re.compile(
    r"There are/is\s+(\d+)\s+relativistic subshells"
)
CYCLE_LIMIT_MARKER = "Maximum iterations in SCF Exceeded."

# Convergence diagnostic patterns
_MAX_CONV_RE = re.compile(r"maximum convergence parameter is\s*=?\s*(" + _FLOAT_RE + r")")
_NOT_CONVERGED_RE = re.compile(
    r"SCF not converged|did NOT converge|NOT converged|"
    + re.escape(CYCLE_LIMIT_MARKER)
    + r"|Convergence not reached|Convergence not obtained|orbitals diverging",
    re.I,
)
_ORBITAL_CONVERGED_RE = re.compile(
    r"Convergence \(latest difference.*\) satisfied",
    re.I,
)
_EXECUTION_COMPLETE_RE = re.compile(r"RMCDHF: Execution complete", re.I)

# Hard orbital-solver crash: the radial equation for a given orbital cannot be
# solved (common for a diffuse valence orbital — e.g. 7s in a neutral actinide —
# from a poor Thomas-Fermi start). The "Failure; equation..." line is fatal
# (ERROR STOP follows); the "Method N unable to solve" lines are per-attempt and
# can appear before a successful retry, so they're warnings, not the verdict.
_ORBITAL_FAILED_RE = re.compile(r"equation for orbital\s+(\S+)\s+could not be solved", re.I)
_METHOD_UNABLE_RE = re.compile(r"Method\s+\d+\s+unable to solve for\s+(\S+)\s+orbital", re.I)
_ERROR_STOP_RE = re.compile(r"^\s*ERROR STOP|Error termination", re.M)
_BAD_TERMINATION_RE = re.compile(
    r"BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES",
    re.I,
)


def parse_rmcdhf_log(text_or_path: str) -> dict[str, Any]:
    """Parse the rmcdhf SCF iteration trace.

    Returns
    -------
    dict with::

        {
          "iterations": [{iter, level_energies, ...}, ...],
          "n_iterations": int,
          "final_energy_hartree": float | None,
          "final_energy_source": str | None,
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
        average_energy_records = [
            _to_float(em.group(1))
            for em in _AVG_E_RE.finditer(chunk)
        ]
        level_energies = [
            {
                "level": int(em.group(1)),
                "energy_hartree": _to_float(em.group(2)),
                "weight": (
                    _to_float(em.group(3))
                    if em.group(3) is not None
                    else None
                ),
            }
            for em in _LEVEL_E_RE.finditer(chunk)
        ]
        ground_energy = min(
            (
                level["energy_hartree"]
                for level in level_energies
            ),
            default=None,
        )
        weighted_energy = _weighted_level_energy(level_energies)
        orbitals = _parse_orbitals(chunk)
        iterations.append({
            "iter": int(m.group(1)),
            "average_energy_records_hartree": average_energy_records,
            "level_energies": level_energies,
            "ground_level_energy_hartree": ground_energy,
            "weighted_level_energy_hartree": weighted_energy,
            "orbitals": orbitals,
            "max_orbital_self_consistency": max(
                (
                    orbital["self_consistency"]
                    for orbital in orbitals
                ),
                default=None,
            ),
            # Retain the old keys for callers that used these values as SCF
            # trace diagnostics. They must not be treated as total energies.
            "avg_energies_hartree": average_energy_records,
            "mean_energy_hartree": (
                sum(average_energy_records) / len(average_energy_records)
                if average_energy_records
                else None
            ),
        })

    # GRASP can write the iteration header before the level records reach the
    # log, so use the last iteration that contains an ASF energy.
    iters_with_energy = [
        iteration
        for iteration in iterations
        if iteration["ground_level_energy_hartree"] is not None
    ]
    final_e = (
        iters_with_energy[-1]["ground_level_energy_hartree"]
        if iters_with_energy
        else None
    )
    e_change = None
    if len(iters_with_energy) >= 2:
        prev = iters_with_energy[-2]["ground_level_energy_hartree"]
        curr = iters_with_energy[-1]["ground_level_energy_hartree"]
        e_change = curr - prev

    orbital_convergence_marker = bool(_ORBITAL_CONVERGED_RE.search(text))
    execution_complete = bool(_EXECUTION_COMPLETE_RE.search(text))
    converged = orbital_convergence_marker or execution_complete
    not_converged = bool(_NOT_CONVERGED_RE.search(text))

    # Orbitals that the radial solver gave up on (de-duplicated, order-preserved).
    failed_orbitals: list[str] = []
    for m in _ORBITAL_FAILED_RE.finditer(text):
        if m.group(1) not in failed_orbitals:
            failed_orbitals.append(m.group(1))
    orbital_solver_failed = bool(failed_orbitals)
    error_stop = bool(
        _ERROR_STOP_RE.search(text) or _BAD_TERMINATION_RE.search(text)
    )
    # Orbitals where a method attempt failed (may have recovered on retry).
    struggled_orbitals = sorted({m.group(1) for m in _METHOD_UNABLE_RE.finditer(text)})

    # A hard solver failure or an ERROR STOP overrides any stale "complete" marker.
    if orbital_solver_failed or error_stop or not_converged:
        converged = False

    iterations_with_orbitals = [
        iteration for iteration in iterations if iteration["orbitals"]
    ]
    final_orbitals = (
        iterations_with_orbitals[-1]["orbitals"]
        if iterations_with_orbitals
        else []
    )
    orbital_histories = _orbital_histories(iterations)
    alternating_norm_orbitals = _alternating_norm_orbitals(
        orbital_histories
    )
    growing_alternating_orbitals = [
        label
        for label in alternating_norm_orbitals
        if _residual_grows(orbital_histories[label][-4:])
    ]

    return {
        "iterations": iterations,
        "n_iterations": len(iterations),
        "final_energy_hartree": final_e,
        "final_energy_source": (
            "rmcdhf_level_record" if final_e is not None else None
        ),
        "final_weighted_energy_hartree": (
            iters_with_energy[-1]["weighted_level_energy_hartree"]
            if iters_with_energy
            else None
        ),
        "energy_change": e_change,
        "execution_complete": execution_complete,
        "orbital_convergence_marker": orbital_convergence_marker,
        "convergence_evidence": (
            "explicit_orbital_convergence_marker"
            if orbital_convergence_marker
            else (
                "execution_complete_without_reported_stopping_test"
                if execution_complete
                else None
            )
        ),
        "final_orbitals": final_orbitals,
        "max_final_orbital_self_consistency": max(
            (
                orbital["self_consistency"]
                for orbital in final_orbitals
            ),
            default=None,
        ),
        "alternating_norm_orbitals": alternating_norm_orbitals,
        "growing_alternating_orbitals": growing_alternating_orbitals,
        "node_count_changes": _node_count_changes(orbital_histories),
        "n_csfs": (
            int(csf_match.group(1))
            if (csf_match := _CSF_RE.search(text)) is not None
            else None
        ),
        "n_subshells": (
            int(subshell_match.group(1))
            if (subshell_match := _SUBSHELL_RE.search(text)) is not None
            else None
        ),
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


def _parse_orbitals(chunk: str) -> list[dict[str, Any]]:
    return [
        {
            "label": match.group(1),
            "energy_au": _to_float(match.group(2)),
            "method": int(match.group(3)),
            "p0": _to_float(match.group(4)),
            "self_consistency": _to_float(match.group(5)),
            "norm_minus_one": _to_float(match.group(6)),
            "damping_factor": float(match.group(7)),
            "join_point": int(match.group(8)),
            "max_tabulation_point": int(match.group(9)),
            "inversion_count": int(match.group(10)),
            "node_count": int(match.group(11)),
        }
        for match in _ORBITAL_ROW_RE.finditer(chunk)
    ]


def _orbital_histories(
    iterations: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    histories: dict[str, list[dict[str, Any]]] = {}
    for iteration in iterations:
        for orbital in iteration["orbitals"]:
            histories.setdefault(orbital["label"], []).append({
                "iteration": iteration["iter"],
                **orbital,
            })
    return histories


def _alternating_norm_orbitals(
    histories: dict[str, list[dict[str, Any]]],
) -> list[str]:
    labels = []
    for label, history in histories.items():
        recent = history[-4:]
        if len(recent) < 4:
            continue
        norms = [entry["norm_minus_one"] for entry in recent]
        if all(
            left * right < 0.0
            for left, right in pairwise(norms)
        ):
            labels.append(label)
    return labels


def _residual_grows(history: list[dict[str, Any]]) -> bool:
    if len(history) < 4:
        return False
    residuals = [entry["self_consistency"] for entry in history]
    return (
        residuals[-1] > residuals[0]
        and max(residuals[-2:]) > max(residuals[:2])
    )


def _node_count_changes(
    histories: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    changes = []
    for label, history in histories.items():
        counts = list(dict.fromkeys(entry["node_count"] for entry in history))
        if len(counts) > 1:
            changes.append({"label": label, "node_counts": counts})
    return changes


def _weighted_level_energy(
    levels: list[dict[str, Any]],
) -> float | None:
    if not levels or any(level["weight"] is None for level in levels):
        return None
    total_weight = sum(float(level["weight"]) for level in levels)
    if total_weight == 0.0:
        return None
    return sum(
        float(level["energy_hartree"]) * float(level["weight"])
        for level in levels
    ) / total_weight

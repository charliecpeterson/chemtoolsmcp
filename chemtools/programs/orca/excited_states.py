"""Parse ORCA multireference and electronic excited-state evidence.

This module owns root-resolved CASSCF, perturbation, MRCI, TD-DFT, and
EOM-CCSD records while the general output parser handles run-level evidence.
"""

from __future__ import annotations

import re
from typing import Any


ACTIVE_ORBITAL_FAILURE = "Failed to constrain active orbitals due to rotations:"
_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_ACTIVE_ELECTRONS_RE = re.compile(
    r"^Number of active electrons\s+\.\.\.\s+(\d+)\s*$", re.IGNORECASE
)
_ACTIVE_ORBITALS_RE = re.compile(
    r"^Number of active orbitals\s+\.\.\.\s+(\d+)\s*$", re.IGNORECASE
)
_CASSCF_ENERGY_RE = re.compile(
    rf"^Final CASSCF energy\s*:\s*({_FLOAT})\s+Eh\b", re.IGNORECASE
)
_CASSCF_STATES_RE = re.compile(
    r"^CAS-SCF STATES FOR BLOCK\s+(\d+)\s+MULT=\s*(\d+)\s+NROOTS=\s*(\d+)",
    re.IGNORECASE,
)
_CASSCF_ROOT_RE = re.compile(
    rf"^ROOT\s+(\d+):\s+E=\s*({_FLOAT})\s+Eh"
    rf"(?:\s+({_FLOAT})\s+eV\s+({_FLOAT})\s+cm\*\*-1)?",
    re.IGNORECASE,
)
_PT2_METHOD_RE = re.compile(r"\(PT2\s*=\s*([^)]+)\)", re.IGNORECASE)
_MULT_ROOT_RE = re.compile(
    r"^MULT\s+(\d+),\s*ROOT\s+(\d+)\s*$", re.IGNORECASE
)
_PT2_CORRECTION_RE = re.compile(
    rf"^Total Energy Correction\s*:\s*dE\s*=\s*({_FLOAT})$", re.IGNORECASE
)
_PT2_REFERENCE_RE = re.compile(
    rf"^Reference\s+Energy\s*:\s*E0\s*=\s*({_FLOAT})$", re.IGNORECASE
)
_PT2_WEIGHT_RE = re.compile(
    rf"^Reference Weight\s*:\s*W0\s*=\s*({_FLOAT})$", re.IGNORECASE
)
_PT2_TOTAL_RE = re.compile(
    rf"^Total Energy \(E0\+dE\)\s*:\s*E\s*=\s*({_FLOAT})$", re.IGNORECASE
)
_CASPT2_CONVERGED_RE = re.compile(
    r"^CASPT2 calculation converged in\s+(\d+)\s+iterations$", re.IGNORECASE
)
_DENOMINATOR_RE = re.compile(
    rf"^smallest energy denominator\s+(\S+)\s*=\s*({_FLOAT})$", re.IGNORECASE
)
_MRCI_LOWEST_RE = re.compile(
    rf"^The lowest energy is\s+({_FLOAT})\s+Eh$", re.IGNORECASE
)
_MRCI_TRANSITION_RE = re.compile(
    rf"^(\d+)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+"
    rf"({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})$"
)
_MRCI_STATE_RE = re.compile(
    rf"^STATE\s+\d+:\s+Energy=\s*({_FLOAT})\s+Eh\s+"
    rf"RefWeight=\s*({_FLOAT})\b",
    re.IGNORECASE,
)
_TDDFT_STATE_RE = re.compile(
    rf"^STATE\s+(\d+):\s+E=\s*({_FLOAT})\s+au\s+({_FLOAT})\s+eV\s+"
    rf"({_FLOAT})\s+cm\*\*-1\s+<S\*\*2>\s*=\s*({_FLOAT})\s+Mult\s+(\d+)$",
    re.IGNORECASE,
)
_EOM_ROOT_RE = re.compile(
    rf"^IROOT=\s*(\d+):\s+({_FLOAT})\s+au\s+({_FLOAT})\s+eV\s+"
    rf"({_FLOAT})\s+cm\*\*-1$",
    re.IGNORECASE,
)
_EOM_SINGLES_RE = re.compile(
    rf"^Percentage singles character=\s*({_FLOAT})$", re.IGNORECASE
)
_CC_TOTAL_ENERGY_RE = re.compile(
    rf"^E\(TOT\)\s+\.\.\.\s+({_FLOAT})\s*$", re.IGNORECASE
)


def parse_excited_state_evidence(lines: list[str]) -> dict[str, Any]:
    return {
        "casscf": _parse_casscf(lines),
        "multireference_pt2": _parse_multireference_pt2(lines),
        "mrci": _parse_mrci(lines),
        "tddft": _parse_tddft(lines),
        "eom_ccsd": _parse_eom_ccsd(lines),
    }


def _parse_casscf(lines: list[str]) -> dict[str, Any] | None:
    active_electrons = None
    active_orbitals = None
    final_energy = None
    final_energy_line = None
    roots = []
    current_block = None

    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if match := _ACTIVE_ELECTRONS_RE.match(stripped):
            active_electrons = int(match.group(1))
        if match := _ACTIVE_ORBITALS_RE.match(stripped):
            active_orbitals = int(match.group(1))
        if match := _CASSCF_ENERGY_RE.match(stripped):
            final_energy = _float(match.group(1))
            final_energy_line = number
        if match := _CASSCF_STATES_RE.match(stripped):
            current_block = {
                "block": int(match.group(1)),
                "multiplicity": int(match.group(2)),
            }
            continue
        if current_block is not None and (
            match := _CASSCF_ROOT_RE.match(stripped)
        ):
            roots.append({
                **current_block,
                "root": int(match.group(1)),
                "energy_hartree": _float(match.group(2)),
                "excitation_energy_ev": (
                    _float(match.group(3)) if match.group(3) else 0.0
                ),
                "wavenumber_cm1": (
                    _float(match.group(4)) if match.group(4) else 0.0
                ),
                "line": number,
            })

    if final_energy is None and not roots:
        return None
    return {
        "active_electrons": active_electrons,
        "active_orbitals": active_orbitals,
        "state_average_energy_hartree": final_energy,
        "state_average_energy_line": final_energy_line,
        "roots": roots,
    }


def _parse_multireference_pt2(lines: list[str]) -> dict[str, Any] | None:
    method = None
    results_started = False
    current_state = None
    states = []
    convergence_iterations = []
    root_denominators: list[dict[str, float]] = []
    denominators = None

    for line in lines:
        stripped = line.strip()
        if match := _PT2_METHOD_RE.search(line):
            method = match.group(1).strip()
        if match := _MULT_ROOT_RE.match(stripped):
            if method == "CASPT2" and not results_started:
                denominators = {}
                root_denominators.append(denominators)
            if results_started:
                current_state = {
                    "multiplicity": int(match.group(1)),
                    "root": int(match.group(2)),
                }
            continue
        if match := _DENOMINATOR_RE.match(stripped):
            if denominators is not None:
                denominators[match.group(1)] = _float(match.group(2))
            continue
        if match := _CASPT2_CONVERGED_RE.match(stripped):
            convergence_iterations.append(int(match.group(1)))
            continue
        if stripped in {"NEVPT2 Results", "CASPT2 Results"}:
            results_started = True
            current_state = None
            continue
        if not results_started or current_state is None:
            continue
        if match := _PT2_CORRECTION_RE.match(stripped):
            current_state["correction_energy_hartree"] = _float(match.group(1))
        elif match := _PT2_REFERENCE_RE.match(stripped):
            current_state["reference_energy_hartree"] = _float(match.group(1))
        elif match := _PT2_WEIGHT_RE.match(stripped):
            current_state["reference_weight"] = _float(match.group(1))
        elif match := _PT2_TOTAL_RE.match(stripped):
            current_state["total_energy_hartree"] = _float(match.group(1))
            states.append(current_state)
            current_state = None

    if method is None or not states:
        return None
    for index, state in enumerate(states):
        if index < len(convergence_iterations):
            state["convergence_iterations"] = convergence_iterations[index]
        if index < len(root_denominators) and root_denominators[index]:
            state["smallest_denominators_hartree"] = root_denominators[index]
            state["minimum_denominator_hartree"] = min(
                root_denominators[index].values()
            )
    return {"method": method, "states": states}


def _parse_mrci(lines: list[str]) -> dict[str, Any] | None:
    lowest_energy = None
    lowest_energy_line = None
    transitions_started = False
    reference_weights = []
    states = []

    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if match := _MRCI_STATE_RE.match(stripped):
            reference_weights.append(
                (_float(match.group(1)), _float(match.group(2)))
            )
        if match := _MRCI_LOWEST_RE.match(stripped):
            lowest_energy = _float(match.group(1))
            lowest_energy_line = number
            transitions_started = True
            continue
        if transitions_started and (
            match := _MRCI_TRANSITION_RE.match(stripped)
        ):
            relative_millihartree = _float(match.group(6))
            energy = lowest_energy + relative_millihartree / 1000
            state = {
                "state": int(match.group(1)),
                "multiplicity": int(match.group(2)),
                "irrep": match.group(3),
                "root": int(match.group(4)),
                "block": int(match.group(5)),
                "energy_hartree": energy,
                "excitation_energy_ev": _float(match.group(7)),
                "wavenumber_cm1": _float(match.group(8)),
                "line": number,
            }
            closest = min(
                reference_weights,
                key=lambda item: abs(item[0] - energy),
                default=None,
            )
            if closest is not None and abs(closest[0] - energy) < 1e-6:
                state["energy_hartree"] = closest[0]
                state["reference_weight"] = closest[1]
            states.append(state)

    if lowest_energy is None or not states:
        return None
    return {
        "lowest_energy_hartree": lowest_energy,
        "lowest_energy_line": lowest_energy_line,
        "states": states,
    }


def _parse_tddft(lines: list[str]) -> dict[str, Any] | None:
    states = []
    for number, line in enumerate(lines, start=1):
        if match := _TDDFT_STATE_RE.match(line.strip()):
            states.append({
                "state": int(match.group(1)),
                "energy_hartree": _float(match.group(2)),
                "energy_ev": _float(match.group(3)),
                "wavenumber_cm1": _float(match.group(4)),
                "expectation_s2": _float(match.group(5)),
                "multiplicity": int(match.group(6)),
                "line": number,
            })
    return {"states": states} if states else None


def _parse_eom_ccsd(lines: list[str]) -> dict[str, Any] | None:
    converged_line = None
    ground_state_energy = None
    ground_state_energy_line = None
    roots = []
    current_root = None
    in_rhs_results = False

    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if "The EOM iterations have converged" in line:
            converged_line = number
        if match := _CC_TOTAL_ENERGY_RE.match(stripped):
            ground_state_energy = _float(match.group(1))
            ground_state_energy_line = number
        if stripped == "EOM-CCSD RESULTS (RHS)":
            in_rhs_results = True
            continue
        if in_rhs_results and "Ground State" in stripped and "LHS" in stripped:
            in_rhs_results = False
            current_root = None
        if in_rhs_results and (match := _EOM_ROOT_RE.match(stripped)):
            current_root = {
                "root": int(match.group(1)),
                "energy_hartree": _float(match.group(2)),
                "energy_ev": _float(match.group(3)),
                "wavenumber_cm1": _float(match.group(4)),
                "line": number,
            }
            roots.append(current_root)
        elif in_rhs_results and current_root is not None and (
            match := _EOM_SINGLES_RE.match(stripped)
        ):
            current_root["singles_character_percent"] = _float(match.group(1))

    if converged_line is None and not roots:
        return None
    return {
        "converged_line": converged_line,
        "ground_state_energy_hartree": ground_state_energy,
        "ground_state_energy_line": ground_state_energy_line,
        "roots": roots,
    }


def _float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


__all__ = ["ACTIVE_ORBITAL_FAILURE", "parse_excited_state_evidence"]

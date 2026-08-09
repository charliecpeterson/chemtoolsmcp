"""Parse ORCA_ESD vibronic-spectrum and radiative-rate evidence.

ORCA_ESD is a post-SCF module with completion markers and result files that
are distinct from the electronic-state tables parsed in ``excited_states``.
"""

from __future__ import annotations

import re
from typing import Any


_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_REQUEST_RE = re.compile(
    r"^Requested calculation:\s*\.\.\.(.+?\S)\s*$", re.IGNORECASE
)
_SETTING_RE = re.compile(
    r"^(Lineshape function|Excited state PES):\s*\.\.\.(.+?\S)\s*$",
    re.IGNORECASE,
)
_OPERATOR_DERIVATIVES_RE = re.compile(
    r"^Use operator derivatives:\s*\.\.\.(yes|no)\s*$", re.IGNORECASE
)
_LINEWIDTH_RE = re.compile(
    rf"^Homogeneous linewidth is:\s*({_FLOAT})\s+cm-1\s*$", re.IGNORECASE
)
_INHOMOGENEOUS_LINEWIDTH_RE = re.compile(
    rf"^Inhomogeneous linewidth is:\s*({_FLOAT})\s+cm-1\s*$",
    re.IGNORECASE,
)
_TEMPERATURE_RE = re.compile(
    rf"^Temperature used:\s*({_FLOAT})\s+K\s*$", re.IGNORECASE
)
_ADIABATIC_ENERGY_RE = re.compile(
    rf"^Adiabatic energy difference:\s*({_FLOAT})\s+cm-1\s*$",
    re.IGNORECASE,
)
_ZERO_ZERO_ENERGY_RE = re.compile(
    rf"^0-0 energy difference:\s*({_FLOAT})\s+cm-1\s*$", re.IGNORECASE
)
_LASER_ENERGY_RE = re.compile(
    rf"^The laser energy is:\s*({_FLOAT})\s+cm-1\s*$", re.IGNORECASE
)
_RATE_RE = re.compile(
    rf"^The calculated (fluorescence|phosphorescence) rate constant is\s*"
    rf"({_FLOAT})\s+s-1\s*$",
    re.IGNORECASE,
)
_RATE_COMPONENTS_RE = re.compile(
    rf"^with\s+({_FLOAT})%\s+from FC and\s+({_FLOAT})%\s+from HT\s*$",
    re.IGNORECASE,
)
_SPECTRUM_RE = re.compile(
    r"^The (.+?) spectrum was saved in\s+(.+?\S)\s*$", re.IGNORECASE
)
_FINISHED = "****ORCA ESD FINISHED WITHOUT ERROR****"


def parse_esd_evidence(lines: list[str]) -> dict[str, Any] | None:
    esd: dict[str, Any] | None = None

    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if match := _REQUEST_RE.match(stripped):
            esd = {
                "process": _normalize_process(match.group(1)),
                "started_line": number,
                "finished_line": None,
                "line_shape": None,
                "excited_state_pes": None,
                "operator_derivatives": None,
                "homogeneous_linewidth_cm1": None,
                "inhomogeneous_linewidth_cm1": None,
                "temperature_kelvin": None,
                "adiabatic_energy_cm1": None,
                "zero_zero_energy_cm1": None,
                "laser_energy_cm1": None,
                "rate_constant_s1": None,
                "rate_process": None,
                "franck_condon_percent": None,
                "herzberg_teller_percent": None,
                "spectrum_file": None,
                "spectrum_line": None,
            }
            continue
        if esd is None:
            continue
        if match := _SETTING_RE.match(stripped):
            key = {
                "lineshape function": "line_shape",
                "excited state pes": "excited_state_pes",
            }[match.group(1).casefold()]
            esd[key] = match.group(2)
        elif match := _OPERATOR_DERIVATIVES_RE.match(stripped):
            esd["operator_derivatives"] = match.group(1).casefold() == "yes"
        elif match := _LINEWIDTH_RE.match(stripped):
            esd["homogeneous_linewidth_cm1"] = _float(match.group(1))
        elif match := _INHOMOGENEOUS_LINEWIDTH_RE.match(stripped):
            esd["inhomogeneous_linewidth_cm1"] = _float(match.group(1))
        elif match := _TEMPERATURE_RE.match(stripped):
            esd["temperature_kelvin"] = _float(match.group(1))
        elif match := _ADIABATIC_ENERGY_RE.match(stripped):
            esd["adiabatic_energy_cm1"] = _float(match.group(1))
        elif match := _ZERO_ZERO_ENERGY_RE.match(stripped):
            esd["zero_zero_energy_cm1"] = _float(match.group(1))
        elif match := _LASER_ENERGY_RE.match(stripped):
            esd["laser_energy_cm1"] = _float(match.group(1))
        elif match := _RATE_RE.match(stripped):
            esd["rate_process"] = match.group(1).casefold()
            esd["rate_constant_s1"] = _float(match.group(2))
        elif match := _RATE_COMPONENTS_RE.match(stripped):
            esd["franck_condon_percent"] = _float(match.group(1))
            esd["herzberg_teller_percent"] = _float(match.group(2))
        elif match := _SPECTRUM_RE.match(stripped):
            esd["spectrum_file"] = match.group(2)
            esd["spectrum_line"] = number
        elif stripped == _FINISHED:
            esd["finished_line"] = number

    return esd


def _normalize_process(value: str) -> str:
    return value.casefold().replace("resonant raman", "resonance_raman")


def _float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


__all__ = ["parse_esd_evidence"]

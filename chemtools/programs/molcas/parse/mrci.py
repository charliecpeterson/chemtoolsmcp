"""Parser for the OpenMolcas MRCI module (SD-CI / ACPF / ACPF-2 …).

Extracts per-state total energies and the reference/correlation energies so a
multireference-CI run's actual result is captured, not lost behind an earlier
CASPT2 step.
"""
from __future__ import annotations

import re
from typing import Any

# "  CI State   1     Total energy:   -117.39650711     QDav: ..."
# "  ACPF State   1     Total energy:   -117.44881267"
STATE_RE = re.compile(
    r"^\s*([A-Za-z0-9-]+)\s+State\s+(\d+)\s+Total energy:\s+([-\d.]+)", re.IGNORECASE
)
REFERENCE_RE = re.compile(r"REFERENCE CI ENERGY:\s+([-\d.]+)", re.IGNORECASE)
CORRELATION_RE = re.compile(r"([A-Za-z0-9-]+)\s+CORRELATION ENERGY:\s+([-\d.]+)", re.IGNORECASE)


def parse_mrci(block_text: str) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    variant: str | None = None
    for line in block_text.splitlines():
        match = STATE_RE.match(line)
        if match:
            variant = match.group(1).upper()
            states.append({
                "variant": variant,
                "state": int(match.group(2)),
                "energy_hartree": float(match.group(3)),
            })
    reference = REFERENCE_RE.search(block_text)
    correlation = CORRELATION_RE.search(block_text)
    return {
        "variant": variant,
        "state_energies": states,
        "reference_energy_hartree": float(reference.group(1)) if reference else None,
        "correlation_energy_hartree": float(correlation.group(2)) if correlation else None,
    }

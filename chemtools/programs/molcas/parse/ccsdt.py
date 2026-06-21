"""Parser for the OpenMolcas CCSDT module (CCSD and CCSD(T)).

Pulls the CCSD and CCSD(T) total energies plus the reference and correlation
energies so a coupled-cluster run's result is captured rather than dropped.
"""
from __future__ import annotations

import re
from typing import Any

# "       CCSD     =      -117.4424278671669"
CCSD_RE = re.compile(r"^\s*CCSD\s*=\s*([-\d.]+)", re.IGNORECASE | re.MULTILINE)
# "       CCSD + T3=      -117.4557935064388"   (the CCSD(T) total)
CCSD_T_RE = re.compile(r"^\s*CCSD\s*\+\s*T3\s*=\s*([-\d.]+)", re.IGNORECASE | re.MULTILINE)
CORRELATION_RE = re.compile(r"Correlation energy\s*:\s*([-\d.]+)", re.IGNORECASE)
REFERENCE_RE = re.compile(r"Reference energy\s*:\s*([-\d.]+)", re.IGNORECASE)


def parse_ccsdt(block_text: str) -> dict[str, Any]:
    ccsd = CCSD_RE.search(block_text)
    ccsd_t = CCSD_T_RE.search(block_text)
    correlation = CORRELATION_RE.search(block_text)
    reference = REFERENCE_RE.search(block_text)
    return {
        "ccsd_energy_hartree": float(ccsd.group(1)) if ccsd else None,
        "ccsd_t_energy_hartree": float(ccsd_t.group(1)) if ccsd_t else None,
        "correlation_energy_hartree": float(correlation.group(1)) if correlation else None,
        "reference_energy_hartree": float(reference.group(1)) if reference else None,
        "has_triples": ccsd_t is not None,
    }

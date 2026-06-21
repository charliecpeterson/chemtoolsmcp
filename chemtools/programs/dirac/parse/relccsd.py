"""Parser for the DIRAC RELCCSD module (MP2 / CCSD / CCSD(T)).

Without this a RELCCSD run reads as a plain SCF job — the correlation energies
(the point of the calculation) are never surfaced.
"""
from __future__ import annotations

import re
from typing import Any

# Final summary lines; DIRAC also reprints them with a leading "@". The plain
# float regex tolerates the optional "@".
_SCF_RE = re.compile(r"^@?\s*SCF energy\s*:\s*([-\d.]+)", re.IGNORECASE | re.MULTILINE)
_MP2_RE = re.compile(r"^@?\s*MP2 correlation energy\s*:\s*([-\d.]+)", re.IGNORECASE | re.MULTILINE)
_CCSD_RE = re.compile(r"^@?\s*CCSD correlation energy\s*:\s*([-\d.]+)", re.IGNORECASE | re.MULTILINE)
_T_RE = re.compile(r"^@?\s*5th order triples \(T\) correction\s*:\s*([-\d.]+)", re.IGNORECASE | re.MULTILINE)
_CCSDT_RE = re.compile(r"^@?\s*Total CCSD\(T\) energy\s*:\s*([-\d.]+)", re.IGNORECASE | re.MULTILINE)


def _last(rx: re.Pattern, text: str) -> float | None:
    matches = rx.findall(text)
    return float(matches[-1]) if matches else None


def parse_relccsd(contents: str) -> dict[str, Any]:
    mp2 = _last(_MP2_RE, contents)
    ccsd = _last(_CCSD_RE, contents)
    if mp2 is None and ccsd is None:
        return {"available": False}

    scf = _last(_SCF_RE, contents)
    triples = _last(_T_RE, contents)
    ccsd_t_total = _last(_CCSDT_RE, contents)
    return {
        "available": True,
        "scf_reference_hartree": scf,
        "mp2_correlation_hartree": mp2,
        "ccsd_correlation_hartree": ccsd,
        "triples_t_correction_hartree": triples,
        "mp2_total_hartree": (scf + mp2) if (scf is not None and mp2 is not None) else None,
        "ccsd_total_hartree": (scf + ccsd) if (scf is not None and ccsd is not None) else None,
        "ccsd_t_total_hartree": ccsd_t_total,
    }

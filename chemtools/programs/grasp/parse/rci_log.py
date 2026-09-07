"""Parse RCI execution evidence that is absent from ``.csum`` files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_BLOCK_RE = re.compile(r"Block\s+(\d+)\s*,\s*ncf\s*=\s*(\d+)")
_SUBSHELL_RE = re.compile(r"There are/is\s+(\d+)\s+relativistic subshells")
_LOADED_CSF_RE = re.compile(
    r"There are\s+(\d+)\s+relativistic CSFs\.\.\. load complete"
)
_BREIT_TYPE_RE = re.compile(r"Computing\s+(\d+)\s+Breit integrals of type\s+(\d+)")
_QED_INTERPOLATION_WARNING = "INTERP: Accuracy of interpolation"
_COMPLETE_RE = re.compile(r"RCI:\s*Execution complete", re.I)
_ERROR_RE = re.compile(r"^\s*ERROR STOP|Error termination", re.M)


def parse_rci_log(text_or_path: str) -> dict[str, Any]:
    """Return execution metadata without inventing an RCI energy."""
    text = _as_text(text_or_path)
    blocks = [
        {"block": int(match.group(1)), "n_csfs": int(match.group(2))}
        for match in _BLOCK_RE.finditer(text)
    ]
    loaded_csf_counts = [int(match.group(1)) for match in _LOADED_CSF_RE.finditer(text)]
    breit_integrals = {
        match.group(2): int(match.group(1)) for match in _BREIT_TYPE_RE.finditer(text)
    }
    subshell_match = _SUBSHELL_RE.search(text)
    error_stop = bool(_ERROR_RE.search(text))
    completed = bool(_COMPLETE_RE.search(text)) and not error_stop
    return {
        "blocks": blocks,
        "n_csfs": (
            sum(block["n_csfs"] for block in blocks)
            if blocks
            else sum(loaded_csf_counts) or None
        ),
        "n_subshells": (
            int(subshell_match.group(1)) if subshell_match is not None else None
        ),
        "breit_integrals": breit_integrals,
        "transverse_integrals_computed": bool(breit_integrals),
        "qed_interpolation_warning_count": text.count(_QED_INTERPOLATION_WARNING),
        "completed": completed,
        "error_stop": error_stop,
        "final_energy_hartree": None,
        "energy_source_required": "rci_summary_csum",
    }


def _as_text(path_or_text: str) -> str:
    if "\n" in path_or_text or not Path(path_or_text).exists():
        return path_or_text
    return Path(path_or_text).read_text(
        encoding="utf-8",
        errors="replace",
    )


__all__ = ["parse_rci_log"]

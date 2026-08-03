"""Parse bounded ``pw2qmcpack.x`` output evidence from local examples.

The converter reports its HDF5 product but does not show a PWSCF completion marker.
"""

from __future__ import annotations

import re
from typing import Any

from chemtools.core.common import parse_scientific_float


_BANNER_RE = re.compile(
    r"^\s*Program\s+pw2qmcpack\s+v\.([^\s]+)\s+starts\s+on\b",
    re.IGNORECASE | re.MULTILINE,
)
_HDF5_RE = re.compile(r"^\s*esh5\s+create\s+(\S+)\s*$", re.IGNORECASE)
_COMPUTE_TIME_RE = re.compile(
    r"^\s*inclusive\s+time\s+in\s+compute_qmcpack\s+\(s\)\s+(.+?)\s*$",
    re.IGNORECASE,
)
_WAVEFUNCTIONS_NOT_COLLECTED_RE = re.compile(
    r"read_file_new:\s*Wavefunctions\s+not\s+in\s+collected\s+format",
    re.IGNORECASE,
)
_GAMMA_TRICK_RE = re.compile(
    r"Using\s+gamma\s+trick\s+results\s+a\s+reduced\s+G\s+space[\s\S]{0,500}?"
    r"not\s+supported\s+by\s+QMCPACK",
    re.IGNORECASE,
)
_JOB_DONE = "JOB DONE."


def is_pw2qmcpack_output(text: str) -> bool:
    """Return whether text begins a ``pw2qmcpack.x`` run."""
    return _BANNER_RE.search(text) is not None


def parse_pw2qmcpack_output_text(text: str) -> dict[str, Any]:
    """Extract the emitted HDF5 path and converter timing evidence."""
    artifacts: list[dict[str, Any]] = []
    compute_time: dict[str, Any] | None = None
    errors = _converter_errors(text)
    for line_number, line in enumerate(text.splitlines(), start=1):
        if (artifact_match := _HDF5_RE.match(line)) is not None:
            artifacts.append({"path": artifact_match.group(1), "line": line_number})
        if (time_match := _COMPUTE_TIME_RE.match(line)) is not None:
            value = parse_scientific_float(time_match.group(1))
            if value is not None:
                compute_time = {"seconds": value, "line": line_number}
    banner = _BANNER_RE.search(text)
    job_done_line = next(
        (
            line_number
            for line_number, line in enumerate(text.splitlines(), start=1)
            if line.strip() == _JOB_DONE
        ),
        None,
    )
    return {
        "format": "qe-pw2qmcpack-output/1",
        "program_version": banner.group(1) if banner is not None else None,
        "hdf5_artifacts": artifacts,
        "compute_qmcpack": compute_time,
        "errors": errors,
        "job_done": job_done_line is not None,
        "job_done_line": job_done_line,
        "line_count": len(text.splitlines()),
    }


def _converter_errors(text: str) -> list[dict[str, Any]]:
    markers = (
        (
            "wavefunctions_not_collected",
            _WAVEFUNCTIONS_NOT_COLLECTED_RE,
            "pw2qmcpack could not read wavefunctions in collected format.",
        ),
        (
            "gamma_trick_unsupported",
            _GAMMA_TRICK_RE,
            "pw2qmcpack rejected QE's gamma-only reduced G-space representation.",
        ),
    )
    errors = []
    for kind, pattern, message in markers:
        match = pattern.search(text)
        if match is not None:
            errors.append({
                "kind": kind,
                "message": message,
                "line": text.count("\n", 0, match.start()) + 1,
            })
    return errors


__all__ = ["is_pw2qmcpack_output", "parse_pw2qmcpack_output_text"]

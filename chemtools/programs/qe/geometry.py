"""Select normalized periodic geometries from PWSCF output.

Relaxation output exposes a geometry only when PWSCF prints both BFGS
convergence and a final-coordinate block. Failed relaxations remain explicit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.programs.qe._coordinates import (
    bfgs_converged,
    initial_runtime_geometry,
    is_relaxation,
    last_final_coordinates,
    normalize_final_geometry,
    parse_final_coordinates,
)


def parse_pw_geometry(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_pw_geometry_text(text)


def parse_pw_geometry_text(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    final = last_final_coordinates(lines)
    if final is not None and bfgs_converged(lines):
        return normalize_final_geometry(lines, final)
    if is_relaxation(lines):
        return {
            "status": "unavailable",
            "reason": (
                "PWSCF did not print a converged final-coordinate block for "
                "this relaxation."
            ),
        }
    return initial_runtime_geometry(lines)


__all__ = [
    "parse_final_coordinates",
    "parse_pw_geometry",
    "parse_pw_geometry_text",
]

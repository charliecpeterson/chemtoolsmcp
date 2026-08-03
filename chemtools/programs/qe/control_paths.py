"""Extract QE ``&CONTROL`` paths for post-processing input artifacts.

Consumers must copy these paths rather than infer QE defaults.
"""

from __future__ import annotations

from typing import Any


def inspect_explicit_control_paths(parsed_qe_input: dict[str, Any]) -> dict[str, Any]:
    """Return renderable QE ``prefix`` and ``outdir`` values with provenance."""
    control = parsed_qe_input["namelists"].get("control", {})
    lines = parsed_qe_input["assignment_lines"].get("control", {})
    prefix = _supported_path_value(control.get("prefix"))
    outdir = _supported_path_value(control.get("outdir"))
    return {
        "prefix": prefix,
        "outdir": outdir,
        "checks": [
            _path_check("prefix", control.get("prefix"), lines.get("prefix"), prefix),
            _path_check("outdir", control.get("outdir"), lines.get("outdir"), outdir),
        ],
    }


def _supported_path_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or "\n" in stripped or "\r" in stripped or "'" in stripped:
        return None
    return stripped


def _path_check(
    name: str,
    observed: Any,
    source_line: int | None,
    rendered: str | None,
) -> dict[str, Any]:
    if rendered is not None:
        return {
            "name": name,
            "status": "pass",
            "observed": observed,
            "source_line": source_line,
            "message": f"QE &CONTROL {name} is an explicit renderable path.",
        }
    return {
        "name": name,
        "status": "review_required",
        "observed": observed,
        "source_line": source_line,
        "message": (
            f"QE &CONTROL {name} must be an explicit non-empty path without a "
            "single quote before Chemtools can draft a post-processing input."
        ),
    }


__all__ = ["inspect_explicit_control_paths"]

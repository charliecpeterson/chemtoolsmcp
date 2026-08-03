"""Draft single-q ``ph.x`` input from explicit QE SCF provenance.

Grid, dielectric, Raman, and electron-phonon workflows remain separate.
"""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Any

from chemtools.programs.qe.control_paths import inspect_explicit_control_paths
from chemtools.programs.qe._namelist import (
    has_namelist,
    parse_namelist,
    strip_fortran_comment,
)


def draft_ph_x_input(
    parsed_qe_input: dict[str, Any],
    title: str,
    q_point: list[float],
) -> dict[str, Any]:
    """Draft a single-q ``ph.x`` input using the source QE path settings."""
    paths = inspect_explicit_control_paths(parsed_qe_input)
    normalized_title = _normalize_title(title)
    normalized_q_point = _normalize_q_point(q_point)
    checks = [
        *paths["checks"],
        _title_check(title, normalized_title),
        _q_point_check(q_point, normalized_q_point),
    ]
    if (
        paths["prefix"] is None
        or paths["outdir"] is None
        or normalized_title is None
        or normalized_q_point is None
    ):
        return _draft_result("review_required", None, checks, [])
    input_text = (
        f"{normalized_title}\n"
        "&INPUTPH\n"
        f"  prefix = '{paths['prefix']}'\n"
        f"  outdir = '{paths['outdir']}'\n"
        "/\n"
        f"{' '.join(_format_coordinate(value) for value in normalized_q_point)}\n"
    )
    advisories = _gamma_advisories(normalized_q_point)
    return _draft_result("ready", input_text, checks, advisories)


def _draft_result(
    status: str,
    input_text: str | None,
    checks: list[dict[str, Any]],
    advisories: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "chemtools.ph-x-input-draft/1",
        "status": status,
        "input_text": input_text,
        "expected_artifacts": {"dynamical_matrix": "matdyn"} if input_text else {},
        "checks": checks,
        "advisories": advisories,
        "scope_limit": (
            "This drafts one explicit q-vector calculation only. It does not "
            "select epsil, set a q-point grid, write response potentials, launch "
            "ph.x, or inspect phonon output."
        ),
    }


def _normalize_title(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or "\n" in stripped or "\r" in stripped:
        return None
    return stripped


def _normalize_q_point(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        return None
    normalized = tuple(float(item) for item in value)
    return normalized if all(isfinite(item) for item in normalized) else None


def _title_check(observed: Any, normalized: str | None) -> dict[str, Any]:
    if normalized is not None:
        return {
            "name": "title",
            "status": "pass",
            "observed": normalized,
            "message": "The ph.x job identifier is one renderable line.",
        }
    return {
        "name": "title",
        "status": "review_required",
        "observed": observed,
        "message": "Provide a non-empty one-line ph.x job identifier.",
    }


def _q_point_check(
    observed: Any,
    normalized: tuple[float, float, float] | None,
) -> dict[str, Any]:
    if normalized is not None:
        return {
            "name": "q_point",
            "status": "pass",
            "observed": list(normalized),
            "message": "The draft contains one explicit ph.x q-vector.",
        }
    return {
        "name": "q_point",
        "status": "review_required",
        "observed": observed,
        "message": "Provide exactly three finite q-vector coordinates.",
    }


def _gamma_advisories(
    q_point: tuple[float, float, float],
) -> list[dict[str, Any]]:
    if any(value != 0.0 for value in q_point):
        return []
    return [{
        "name": "gamma_nonanalytic_terms",
        "status": "review_required",
        "message": (
            "At q=0, this draft leaves epsil at its QE default. Decide whether "
            "the non-analytic LO-TO term is needed before running ph.x."
        ),
    }]


def _format_coordinate(value: float) -> str:
    return "0" if value == 0.0 else format(value, ".12g")


def is_ph_x_input(text: str) -> bool:
    """Return whether text declares the PHonon ``&INPUTPH`` namelist."""
    return has_namelist(text, "inputph")


def parse_ph_x_input(path: str) -> dict[str, Any]:
    """Parse the bounded single-q ``ph.x`` input shape used by Chemtools."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        return parse_ph_x_text(handle.read())


def parse_ph_x_text(text: str) -> dict[str, Any]:
    """Parse one PHonon input without resolving its preceding QE calculation."""
    lines = text.splitlines()
    inputph = parse_namelist(text, "inputph")
    if inputph.line is None:
        return {
            "format": "qe-ph-input/1",
            "title": None,
            "inputph_line": None,
            "inputph_closed": False,
            "namelist": {},
            "q_point": None,
            "q_point_line": None,
        }
    title = _first_content_line(lines[:inputph.line - 1])
    q_point, q_point_line = _parse_q_point(lines, inputph.next_line_index)
    return {
        "format": "qe-ph-input/1",
        "title": title,
        "inputph_line": inputph.line,
        "inputph_closed": inputph.closed,
        "namelist": inputph.values,
        "q_point": q_point,
        "q_point_line": q_point_line,
    }


def lint_ph_x_input(text: str) -> list[dict[str, Any]]:
    """Review the explicit single-q portion of a ``ph.x`` input."""
    parsed = parse_ph_x_text(text)
    issues: list[dict[str, Any]] = []
    if parsed["title"] is None:
        issues.append(_issue(
            "error",
            "ph.x requires a one-line job identifier before &INPUTPH.",
            suggested_fix="single-q phonon",
        ))
    if parsed["inputph_line"] is None:
        return [*issues, _issue(
            "error",
            "ph.x requires an &INPUTPH namelist.",
            suggested_fix="&INPUTPH\n/",
        )]
    if not parsed["inputph_closed"]:
        issues.append(_issue(
            "error",
            "&INPUTPH is not closed.",
            line=parsed["inputph_line"],
            suggested_fix="/",
        ))
        return issues
    values = parsed["namelist"]
    for name in ("prefix", "outdir"):
        if not _non_empty_string(values.get(name)):
            issues.append(_issue(
                "warning",
                (
                    f"&INPUTPH {name} is not explicit, so Chemtools cannot "
                    "confirm it matches the preceding pw.x calculation."
                ),
                line=parsed["inputph_line"],
            ))
    if _logical_true(values.get("ldisp")) or _logical_true(values.get("qplot")):
        issues.append(_issue(
            "warning",
            (
                "This ph.x deck requests a q-point grid or list; Chemtools "
                "currently reviews only one explicit q-vector calculation."
            ),
            line=parsed["inputph_line"],
        ))
        return issues
    q_point = parsed["q_point"]
    if q_point is None:
        issues.append(_issue(
            "error",
            "Single-q ph.x input requires exactly three finite q-vector coordinates after &INPUTPH.",
            suggested_fix="0.0 0.0 0.0",
        ))
        return issues
    if _logical_true(values.get("epsil")) and any(value != 0.0 for value in q_point):
        issues.append(_issue(
            "error",
            "epsil=.true. is only supported at q=0 in the documented ph.x scope.",
            line=parsed["q_point_line"],
        ))
    elif all(value == 0.0 for value in q_point) and not _logical_true(values.get("epsil")):
        issues.append(_issue(
            "warning",
            (
                "At q=0, decide whether epsil is needed for the non-analytic "
                "LO-TO term before running ph.x."
            ),
            line=parsed["q_point_line"],
        ))
    return issues


def _first_content_line(lines: list[str]) -> str | None:
    for line in lines:
        stripped = strip_fortran_comment(line).strip()
        if stripped:
            return stripped
    return None


def _parse_q_point(
    lines: list[str],
    start_index: int | None,
) -> tuple[list[float] | None, int | None]:
    if start_index is None:
        return None, None
    for index in range(start_index, len(lines)):
        stripped = strip_fortran_comment(lines[index]).strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) != 3:
            return None, index + 1
        try:
            q_point = [float(token.replace("d", "e").replace("D", "e")) for token in tokens]
        except ValueError:
            return None, index + 1
        return (q_point if all(isfinite(value) for value in q_point) else None), index + 1
    return None, None


def _logical_true(value: Any) -> bool:
    return value is True


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _issue(
    level: str,
    message: str,
    *,
    line: int | None = None,
    suggested_fix: str | None = None,
) -> dict[str, Any]:
    return {
        "level": level,
        "message": message,
        "line": line,
        "suggested_fix": suggested_fix,
    }


__all__ = [
    "draft_ph_x_input",
    "is_ph_x_input",
    "lint_ph_x_input",
    "parse_ph_x_input",
    "parse_ph_x_text",
]

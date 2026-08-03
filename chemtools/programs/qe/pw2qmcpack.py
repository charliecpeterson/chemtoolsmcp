"""Render the narrow pw2qmcpack input form demonstrated by local examples.

The converter deck names the QE prefix and output directory explicitly.
"""

from __future__ import annotations

from typing import Any

from chemtools.programs.qe.control_paths import inspect_explicit_control_paths
from chemtools.programs.qe._namelist import has_namelist, parse_namelist


_REFERENCE_KEYS = frozenset({"prefix", "outdir", "write_psir"})


def draft_pw2qmcpack_input(parsed_qe_input: dict[str, Any]) -> dict[str, Any]:
    """Draft a converter deck from explicit QE ``&CONTROL`` path settings."""
    paths = inspect_explicit_control_paths(parsed_qe_input)
    prefix = paths["prefix"]
    outdir = paths["outdir"]
    checks = paths["checks"]
    if prefix is None or outdir is None:
        return {
            "schema_version": "chemtools.pw2qmcpack-input-draft/1",
            "status": "review_required",
            "input_text": None,
            "checks": checks,
            "scope_limit": (
                "This drafts only the supported inputpp form. It does not infer "
                "QE prefix or outdir defaults, select converter options, launch "
                "pw2qmcpack, or inspect the resulting HDF5 file."
            ),
        }
    return {
        "schema_version": "chemtools.pw2qmcpack-input-draft/1",
        "status": "ready",
        "input_text": (
            "&inputpp\n"
            f"  prefix = '{prefix}'\n"
            f"  outdir = '{outdir}'\n"
            "  write_psir = .false.\n"
            "/\n"
        ),
        "checks": checks,
        "scope_limit": (
            "This drafts only the supported inputpp form. It does not infer QE "
            "defaults, select converter options, launch pw2qmcpack, or inspect "
            "the resulting HDF5 file."
        ),
    }


def is_pw2qmcpack_input(text: str) -> bool:
    """Recognize only the demonstrated ``pw2qmcpack.x`` ``&INPUTPP`` form."""
    if not has_namelist(text, "inputpp"):
        return False
    inputpp = parse_namelist(text, "inputpp")
    return (
        "write_psir" in inputpp.values
        and set(inputpp.values) <= _REFERENCE_KEYS
    )


def parse_pw2qmcpack_input(path: str) -> dict[str, Any]:
    """Parse the bounded converter input shape used by Chemtools."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        return parse_pw2qmcpack_text(handle.read())


def parse_pw2qmcpack_text(text: str) -> dict[str, Any]:
    """Read the demonstrated ``&INPUTPP`` converter namelist from text."""
    inputpp = parse_namelist(text, "inputpp")
    return {
        "format": "qe-pw2qmcpack-input/1",
        "inputpp_line": inputpp.line,
        "inputpp_closed": inputpp.closed,
        "namelist": inputpp.values,
    }


def lint_pw2qmcpack_input(text: str) -> list[dict[str, Any]]:
    """Review the explicit handoff settings in the bounded converter form."""
    parsed = parse_pw2qmcpack_text(text)
    if parsed["inputpp_line"] is None:
        return [_issue(
            "error",
            "pw2qmcpack.x requires an &INPUTPP namelist.",
            suggested_fix="&INPUTPP\n/",
        )]
    if not parsed["inputpp_closed"]:
        return [_issue(
            "error",
            "&INPUTPP is not closed.",
            line=parsed["inputpp_line"],
            suggested_fix="/",
        )]
    values = parsed["namelist"]
    issues: list[dict[str, Any]] = []
    for name in ("prefix", "outdir"):
        if not _non_empty_string(values.get(name)):
            issues.append(_issue(
                "warning",
                (
                    f"&INPUTPP {name} is not explicit, so Chemtools cannot "
                    "confirm the preceding QE calculation."
                ),
                line=parsed["inputpp_line"],
            ))
    if values.get("write_psir") is not False:
        issues.append(_issue(
            "warning",
            (
                "The demonstrated converter form uses write_psir=.false.; "
                "review a different setting before conversion."
            ),
            line=parsed["inputpp_line"],
        ))
    unknown = sorted(set(values) - _REFERENCE_KEYS)
    if unknown:
        issues.append(_issue(
            "warning",
            (
                "This &INPUTPP deck contains options outside the demonstrated "
                f"converter form: {', '.join(unknown)}."
            ),
            line=parsed["inputpp_line"],
        ))
    return issues


def inspect_pw2qmcpack_input_scope(text: str) -> dict[str, Any]:
    """Report whether a converter input stays within Chemtools' supported form."""
    issues = lint_pw2qmcpack_input(text)
    if not issues:
        return {
            "name": "pw2qmcpack_input_scope",
            "status": "pass",
            "observed": {"issues": []},
            "message": "The converter input uses Chemtools' supported inputpp form.",
        }
    if any(issue["level"] == "error" for issue in issues):
        status = "not_ready"
        message = "The converter input is incomplete or malformed."
    else:
        status = "review_required"
        message = "The converter input uses settings outside Chemtools' supported form."
    return {
        "name": "pw2qmcpack_input_scope",
        "status": status,
        "observed": {"issues": issues},
        "message": message,
    }


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
    "draft_pw2qmcpack_input",
    "inspect_pw2qmcpack_input_scope",
    "is_pw2qmcpack_input",
    "lint_pw2qmcpack_input",
    "parse_pw2qmcpack_input",
    "parse_pw2qmcpack_text",
]

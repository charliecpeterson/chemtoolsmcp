"""Build a guided review from one chemistry input artifact.

The service uses only capabilities declared by the selected backend. Missing
parsers or linters remain visible as uncertainty in the returned review.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable, Literal, Mapping

from chemtools.core.artifact_classification import classify_artifact
from chemtools.core.artifacts import ArtifactRole
from chemtools.core.program import (
    PathInputReviewer,
    ProgramBackend,
    ProgramCapability,
)


INPUT_REVIEW_SCHEMA = "chemtools.review-input/1"
InputResolutionMethod = Literal["content", "explicit", "extension"]
_INPUT_IDENTITY_HEAD_BYTES = 64 * 1024


class InputReviewError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        candidates: Iterable[str] = (),
    ) -> None:
        self.code = code
        self.candidates = tuple(candidates)
        super().__init__(message)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": self.code,
            "message": str(self),
        }
        if self.candidates:
            payload["candidates"] = list(self.candidates)
        return payload


def detect_input_backend(
    backends: Iterable[ProgramBackend],
    input_file: str | Path,
) -> tuple[ProgramBackend, InputResolutionMethod]:
    path = _input_path(input_file)
    available = {backend.name: backend for backend in backends}
    content_matches = list(
        detect_input_content_candidates(
            available.values(),
            path,
        )
    )
    if len(content_matches) == 1:
        return available[content_matches[0]], "content"
    if len(content_matches) > 1:
        raise InputReviewError(
            "program_detection_ambiguous",
            f"input content matches multiple programs: {content_matches}",
            candidates=content_matches,
        )

    extension_matches = [
        backend.name
        for backend in available.values()
        if _declares_input_extension(backend, path)
    ]
    if len(extension_matches) == 1:
        return available[extension_matches[0]], "extension"
    if len(extension_matches) > 1:
        raise InputReviewError(
            "program_detection_ambiguous",
            (
                f"input extension {path.suffix!r} matches multiple programs; "
                "pass program explicitly"
            ),
            candidates=extension_matches,
        )
    raise InputReviewError(
        "program_detection_failed",
        f"could not detect a supported chemistry input format from {path}",
        candidates=available,
    )


def detect_input_content_candidates(
    backends: Iterable[ProgramBackend],
    input_file: str | Path,
) -> tuple[str, ...]:
    """Return backends whose input grammar markers match the bounded text."""
    path = _input_path(input_file)
    with path.open(encoding="utf-8", errors="replace") as handle:
        head = handle.read(_INPUT_IDENTITY_HEAD_BYTES)
    return tuple(
        backend.name
        for backend in backends
        if _matches_input_content(backend.name, head)
    )


def review_input(
    backend: ProgramBackend,
    input_file: str | Path,
    *,
    resolved_by: InputResolutionMethod,
) -> dict[str, Any]:
    path = _input_path(input_file)
    text = path.read_text(encoding="utf-8", errors="replace")
    uncertainty: list[dict[str, str]] = []

    parsed, parse_status = _parse_input(
        backend,
        path,
        uncertainty,
    )
    issues, lint_status = _lint_input(
        backend,
        path,
        text,
        uncertainty,
    )
    assessment = _assessment(
        parse_status=parse_status,
        lint_status=lint_status,
        issues=issues,
    )
    next_actions = _next_actions(
        path,
        assessment["verdict"]["label"],
        issues,
        backend.name,
    )
    classification = classify_artifact(backend, path)
    uncertainty.extend(
        _classification_uncertainty(classification.status)
    )

    return {
        "schema_version": INPUT_REVIEW_SCHEMA,
        "program": {
            "name": backend.name,
            "resolved_by": resolved_by,
        },
        "source": {
            "path": str(path),
            "size_bytes": path.stat().st_size,
        },
        "assessment": assessment,
        "evidence": {
            "artifact_classification": {
                "status": classification.status,
                "candidates": [
                    {
                        "kind": candidate.kind,
                        "roles": sorted(
                            role.value for role in candidate.roles
                        ),
                        "evidence": candidate.evidence,
                        "matched_by": candidate.matched_by,
                        "matched_value": candidate.matched_value,
                    }
                    for candidate in classification.candidates
                ],
            },
            "parser": {
                "status": parse_status,
                "result": parsed,
            },
            "lint": {
                "status": lint_status,
                "summary": _issue_summary(issues),
                "issues": issues,
            },
        },
        "uncertainty": uncertainty,
        "next_actions": next_actions,
    }


def _input_path(input_file: str | Path) -> Path:
    path = Path(input_file).expanduser().resolve()
    if not path.is_file():
        raise InputReviewError(
            "source_not_file",
            f"chemistry input is not a readable file: {path}",
        )
    return path


def _matches_input_content(program: str, text: str) -> bool:
    if program == "nwchem":
        has_task = re.search(r"(?mi)^\s*task\s+\S+", text) is not None
        has_geometry = re.search(r"(?mi)^\s*geometry\b", text) is not None
        has_basis = re.search(r"(?mi)^\s*basis\b", text) is not None
        return has_task or (has_geometry and has_basis)
    if program == "molcas":
        return re.search(
            (
                r"(?mi)^\s*&(GATEWAY|SEWARD|SCF|RASSCF|CASPT2|"
                r"RASSI|SLAPAF|ALASKA)\b"
            ),
            text,
        ) is not None
    if program == "dirac":
        return re.search(
            r"(?mi)^\s*\*\*(DIRAC|HAMILTONIAN|WAVE\s+F|PROPERTIES)\b",
            text,
        ) is not None
    if program == "qe":
        has_control = re.search(r"(?mi)^\s*&control\b", text) is not None
        has_system = re.search(r"(?mi)^\s*&system\b", text) is not None
        has_pw_card = re.search(
            r"(?mi)^\s*(ATOMIC_SPECIES|ATOMIC_POSITIONS|K_POINTS)\b",
            text,
        ) is not None
        has_known_companion_namelist = re.search(
            r"(?mi)^\s*&(bands|dos|inputpp|inputph|projwfc)\b",
            text,
        ) is not None
        return (
            has_control and has_system and has_pw_card
        ) or has_known_companion_namelist
    if program == "qmcpack":
        return re.search(
            r"(?is)^\s*(?:<\?xml[^>]*>\s*)?(?:<!--.*?-->\s*)*"
            r"<(?:simulation|qmcsystem)(?:\s|>)",
            text,
        ) is not None
    return False


def _declares_input_extension(
    backend: ProgramBackend,
    path: Path,
) -> bool:
    for spec in backend.artifact_kinds.values():
        if ArtifactRole.PRIMARY_INPUT not in spec.default_roles:
            continue
        if any(path.name.endswith(extension) for extension in spec.extensions):
            return True
        if path.name in spec.filenames:
            return True
    return False


def _parse_input(
    backend: ProgramBackend,
    path: Path,
    uncertainty: list[dict[str, str]],
) -> tuple[dict[str, Any] | None, str]:
    if not backend.supports(ProgramCapability.INPUT_PARSE):
        uncertainty.append({
            "code": "input_parser_unavailable",
            "message": f"{backend.name} has no declared input parser.",
            "impact": "The review cannot report a normalized input summary.",
        })
        return None, "unsupported"
    assert backend.parser is not None
    try:
        parsed = backend.parser.parse_input(str(path))
    except Exception as exc:
        uncertainty.append({
            "code": "input_parse_failed",
            "message": (
                f"{backend.name} input parsing failed with "
                f"{type(exc).__name__}: {exc}"
            ),
            "impact": "Lint evidence remains usable if its check completed.",
        })
        return None, "failed"
    if not isinstance(parsed, Mapping):
        uncertainty.append({
            "code": "invalid_input_parser_result",
            "message": (
                f"{backend.name} input parser returned "
                f"{type(parsed).__name__}, not a mapping."
            ),
            "impact": "The parsed input summary was discarded.",
        })
        return None, "failed"
    return dict(parsed), "completed"


def _lint_input(
    backend: ProgramBackend,
    path: Path,
    text: str,
    uncertainty: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], str]:
    if not backend.supports(ProgramCapability.INPUT_LINT):
        uncertainty.append({
            "code": "input_linter_unavailable",
            "message": f"{backend.name} has no declared input linter.",
            "impact": "Parsing alone does not establish input correctness.",
        })
        return [], "unsupported"
    assert backend.inputs is not None
    try:
        if isinstance(backend.inputs, PathInputReviewer):
            raw_issues = backend.inputs.lint_input_file(str(path))
        else:
            raw_issues = backend.inputs.lint_input(text)
    except Exception as exc:
        uncertainty.append({
            "code": "input_lint_failed",
            "message": (
                f"{backend.name} input linting failed with "
                f"{type(exc).__name__}: {exc}"
            ),
            "impact": "The review cannot make a lint-based judgment.",
        })
        return [], "failed"
    if not isinstance(raw_issues, (list, tuple)):
        uncertainty.append({
            "code": "invalid_input_linter_result",
            "message": (
                f"{backend.name} input linter returned "
                f"{type(raw_issues).__name__}, not a list."
            ),
            "impact": "The linter result was discarded.",
        })
        return [], "failed"
    return [
        _normalize_issue(issue)
        for issue in raw_issues
    ], "completed"


def _normalize_issue(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {
            "level": "warning",
            "message": str(value),
            "line": None,
            "suggested_fix": None,
        }
    level = str(value.get("level") or "warning").lower()
    if level not in {"error", "warning", "info"}:
        level = "warning"
    return {
        "level": level,
        "message": str(value.get("message") or ""),
        "line": value.get("line"),
        "suggested_fix": value.get("suggested_fix"),
    }


def _issue_summary(issues: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "errors": sum(issue["level"] == "error" for issue in issues),
        "warnings": sum(issue["level"] == "warning" for issue in issues),
        "info": sum(issue["level"] == "info" for issue in issues),
    }


def _assessment(
    *,
    parse_status: str,
    lint_status: str,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = _issue_summary(issues)
    if lint_status == "completed" and counts["errors"]:
        verdict = {
            "label": "errors_found",
            "confidence": 0.9,
            "reasons": [
                f"The configured linter found {counts['errors']} error(s)."
            ],
        }
    elif lint_status == "completed" and counts["warnings"]:
        verdict = {
            "label": "review_required",
            "confidence": 0.8,
            "reasons": [
                f"The configured linter found {counts['warnings']} warning(s)."
            ],
        }
    elif lint_status == "completed":
        verdict = {
            "label": "checks_passed",
            "confidence": 0.8,
            "reasons": [
                "The configured linter found no errors or warnings."
            ],
        }
    elif lint_status == "failed":
        verdict = {
            "label": "review_incomplete",
            "confidence": 0.2,
            "reasons": ["The declared input linter did not complete."],
        }
    elif parse_status == "completed":
        verdict = {
            "label": "parsed_unchecked",
            "confidence": 0.45,
            "reasons": [
                "The input parsed, but no declared linter reviewed it."
            ],
        }
    elif parse_status == "failed":
        verdict = {
            "label": "review_failed",
            "confidence": 0.1,
            "reasons": ["Neither parsing nor linting produced usable evidence."],
        }
    else:
        verdict = {
            "label": "unsupported",
            "confidence": 1.0,
            "reasons": [
                "This backend has no declared parser or linter for this input."
            ],
        }
    return {"verdict": verdict}


def _next_actions(
    path: Path,
    verdict: str,
    issues: list[dict[str, Any]],
    program: str,
) -> list[dict[str, Any]]:
    actions = [
        {
            "action": "edit_input",
            "path": str(path),
            "line": issue["line"],
            "suggested_fix": issue["suggested_fix"],
            "reason": issue["message"],
            "priority": 1 if issue["level"] == "error" else 2,
        }
        for issue in issues
        if issue["level"] in {"error", "warning"}
    ]
    if actions:
        return sorted(actions, key=lambda action: action["priority"])
    if verdict == "parsed_unchecked":
        return [{
            "action": "manual_scientific_review",
            "path": str(path),
            "reason": f"{program} has no declared input linter.",
            "priority": 1,
        }]
    if verdict == "unsupported":
        return [{
            "action": "use_program_specific_input_builder",
            "path": str(path),
            "reason": (
                f"{program} does not expose a standalone input review contract."
            ),
            "priority": 1,
        }]
    return []


def _classification_uncertainty(
    status: str,
) -> list[dict[str, str]]:
    if status == "ambiguous":
        return [{
            "code": "artifact_classification_ambiguous",
            "message": "The filename matches more than one input artifact kind.",
            "impact": "Confirm the intended input role before execution.",
        }]
    if status == "unmatched":
        return [{
            "code": "artifact_kind_unmatched",
            "message": "The filename does not match a declared input artifact kind.",
            "impact": "The program was selected from content or an override.",
        }]
    return []


__all__ = [
    "INPUT_REVIEW_SCHEMA",
    "InputReviewError",
    "detect_input_content_candidates",
    "detect_input_backend",
    "review_input",
]

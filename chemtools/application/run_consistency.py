"""Coordinate optional input-output consistency checks for run inspection.

The application layer selects one explicit primary input. Program-specific
comparison logic remains behind the backend capability boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from chemtools.core.program import ProgramBackend, ProgramCapability


def inspect_input_output_consistency(
    backend: ProgramBackend,
    output_path: Path,
    parsed_output: Mapping[str, Any],
    artifacts: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    input_paths = _primary_input_paths(artifacts)
    if not input_paths:
        return {
            "status": "not_checked",
            "reason": (
                "No readable explicit primary input artifact was supplied."
            ),
        }, []
    if len(input_paths) > 1:
        return {
            "status": "not_checked",
            "reason": (
                "More than one explicit primary input artifact was supplied."
            ),
            "candidate_paths": input_paths,
        }, [{
            "code": "primary_input_ambiguous",
            "message": (
                "Input-output consistency was skipped because multiple "
                "primary inputs were supplied."
            ),
            "impact": "Supply only the input that produced this output.",
        }]

    input_path = Path(input_paths[0])
    artifact_paths = tuple(
        artifact["path"]
        for artifact in artifacts
        if artifact.get("entry_type") == "file"
    )
    return compare_explicit_input_output(
        backend,
        output_path,
        input_path,
        parsed_output,
        artifact_paths=artifact_paths,
    )


def compare_explicit_input_output(
    backend: ProgramBackend,
    output_path: Path,
    input_path: Path,
    parsed_output: Mapping[str, Any],
    *,
    artifact_paths: tuple[str, ...] = (),
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Compare one caller-selected input with one parsed output."""
    if not backend.supports(ProgramCapability.INPUT_PARSE):
        return _unsupported(
            backend,
            str(input_path),
            "input parsing",
        )
    if not backend.supports(ProgramCapability.RUN_CONSISTENCY):
        return _unsupported(
            backend,
            str(input_path),
            "input-output consistency checks",
        )
    assert backend.parser is not None
    assert backend.consistency is not None

    try:
        parsed_input = backend.parser.parse_input(str(input_path))
    except Exception as exc:
        return {
            "status": "not_checked",
            "input_path": str(input_path),
            "reason": (
                f"Input parsing failed with {type(exc).__name__}: {exc}"
            ),
        }, [{
            "code": "consistency_input_parse_failed",
            "message": (
                f"{backend.name} could not parse the related input: {exc}"
            ),
            "impact": "No input-output fields were compared.",
        }]
    if not isinstance(parsed_input, Mapping):
        return {
            "status": "not_checked",
            "input_path": str(input_path),
            "reason": (
                "The input parser did not return a structured mapping."
            ),
        }, [{
            "code": "invalid_consistency_input",
            "message": (
                f"{backend.name} input parser returned "
                f"{type(parsed_input).__name__}, not a mapping."
            ),
            "impact": "No input-output fields were compared.",
        }]

    try:
        compared = backend.consistency.compare_input_output(
            str(input_path),
            str(output_path),
            parsed_input,
            parsed_output,
            artifact_paths,
        )
    except Exception as exc:
        return {
            "status": "not_checked",
            "input_path": str(input_path),
            "reason": (
                f"Consistency checking failed with "
                f"{type(exc).__name__}: {exc}"
            ),
        }, [{
            "code": "input_output_consistency_failed",
            "message": (
                f"{backend.name} consistency checking failed with "
                f"{type(exc).__name__}: {exc}"
            ),
            "impact": "The parsed input and output evidence remain usable.",
        }]
    if not isinstance(compared, Mapping):
        return {
            "status": "not_checked",
            "input_path": str(input_path),
            "reason": (
                "The consistency adapter did not return a structured mapping."
            ),
        }, [{
            "code": "invalid_consistency_result",
            "message": (
                f"{backend.name} consistency adapter returned "
                f"{type(compared).__name__}, not a mapping."
            ),
            "impact": "The consistency result was discarded.",
        }]

    result = dict(compared)
    if result.get("status") != "mismatch":
        return result, []
    mismatched = [
        str(check.get("field"))
        for check in result.get("checks") or []
        if (
            isinstance(check, Mapping)
            and check.get("status") == "mismatch"
        )
    ]
    return result, [{
        "code": "input_output_mismatch",
        "message": (
            "The explicit input disagrees with output evidence for: "
            f"{', '.join(mismatched) or 'unknown fields'}."
        ),
        "impact": (
            "Verify that the supplied input and related restart files belong "
            "to this output."
        ),
    }]


def _primary_input_paths(
    artifacts: list[dict[str, Any]],
) -> list[str]:
    paths = []
    for artifact in artifacts:
        if (
            artifact.get("relationship") != "related"
            or artifact.get("entry_type") != "file"
        ):
            continue
        classification = artifact.get("classification") or {}
        if classification.get("status") != "matched":
            continue
        candidates = classification.get("candidates") or []
        if any(
            "primary_input" in (candidate.get("roles") or [])
            for candidate in candidates
        ):
            paths.append(str(artifact["path"]))
    return paths


def _unsupported(
    backend: ProgramBackend,
    input_path: str,
    missing: str,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    return {
        "status": "unsupported",
        "input_path": input_path,
        "reason": f"{backend.name} does not declare {missing}.",
    }, [{
        "code": "input_output_consistency_unavailable",
        "message": (
            f"{backend.name} does not declare {missing}."
        ),
        "impact": "The input and output were inspected separately.",
    }]


__all__ = [
    "compare_explicit_input_output",
    "inspect_input_output_consistency",
]

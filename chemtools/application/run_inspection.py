"""Build a guided inspection from a primary output and explicit artifacts.

Program parsers retain their native detail. This service normalizes the
evidence, scientific assessment, uncertainty, and next-action boundary.
Related artifacts are classified and observed without directory discovery.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from chemtools.core.artifact_classification import (
    ArtifactClassification,
    classify_artifact,
)
from chemtools.core.artifacts import ArtifactRole
from chemtools.core.common import detect_standalone_output_format
from chemtools.core.geometry import inspect_geometry
from chemtools.core.program import ProgramBackend, ProgramCapability
from chemtools.core.units import ANGSTROM_PER_BOHR
from chemtools.application.run_consistency import (
    inspect_input_output_consistency,
)
from chemtools.application.text_evidence import (
    RELATED_TEXT_LIMIT_BYTES,
    RELATED_TEXT_TOTAL_LIMIT_BYTES,
    read_text_excerpt,
)


RUN_INSPECTION_SCHEMA = "chemtools.inspect-run/1"
ResolutionMethod = Literal["content", "explicit"]
_OUTPUT_IDENTITY_HEAD_BYTES = 64 * 1024
PRIMARY_OUTPUT_LIMIT_BYTES = 128 * 1024 * 1024


class RunInspectionError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        program: str | None = None,
        exception_type: str | None = None,
        detected_format: str | None = None,
    ) -> None:
        self.code = code
        self.program = program
        self.exception_type = exception_type
        self.detected_format = detected_format
        super().__init__(message)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": self.code,
            "message": str(self),
        }
        if self.program is not None:
            payload["program"] = self.program
        if self.exception_type is not None:
            payload["exception_type"] = self.exception_type
        if self.detected_format is not None:
            payload["detected_format"] = self.detected_format
        return payload


def inspect_run_geometry(
    backend: ProgramBackend,
    output_file: str | Path,
    *,
    max_bond_length: float = 2.5,
    min_safe_distance: float = 0.6,
    covalent_tolerance: float = 1.20,
    measurements: dict[str, list[list[int]]] | None = None,
) -> dict[str, Any]:
    """Normalize backend geometry and apply the shared molecular checks."""
    assert backend.parser is not None
    raw_geometry = backend.parser.get_geometry(str(output_file))
    if isinstance(raw_geometry, list):
        raw_atoms = raw_geometry
        source_units = "angstrom"
    elif isinstance(raw_geometry, dict):
        raw_atoms = list(raw_geometry.get("atoms") or [])
        source_units = (raw_geometry.get("units") or "angstrom").lower()
    else:
        return {
            "error": "no_geometry",
            "message": (
                f"Plugin {backend.name} returned no geometry for "
                f"{output_file}."
            ),
        }
    if not raw_atoms:
        return {
            "error": "no_geometry",
            "message": (
                f"Plugin {backend.name} returned an empty geometry for "
                f"{output_file}."
            ),
        }

    atoms = [
        atom
        if "symbol" in atom
        else {
            **atom,
            "symbol": atom.get("element") or atom.get("Element"),
        }
        for atom in raw_atoms
    ]
    if source_units == "bohr":
        atoms = [
            {
                **atom,
                "x": atom["x"] * ANGSTROM_PER_BOHR,
                "y": atom["y"] * ANGSTROM_PER_BOHR,
                "z": atom["z"] * ANGSTROM_PER_BOHR,
            }
            for atom in atoms
        ]

    inspected = inspect_geometry(
        atoms,
        max_bond_length=max_bond_length,
        min_safe_distance=min_safe_distance,
        covalent_tolerance=covalent_tolerance,
        measurements=measurements,
        units="angstrom",
    )
    return {"program": backend.name, **inspected}


def validate_primary_output_format(
    output_file: str | Path,
    *,
    program: str | None = None,
) -> None:
    path = Path(output_file).expanduser().resolve()
    detected_format = _detect_standalone_output_format(path)
    if detected_format is None or detected_format == program:
        return

    if detected_format == "nbo" and program is None:
        message = (
            "standalone NBO analysis is not a supported primary run output; "
            "provide the parent quantum-chemistry output that contains the "
            "electronic-structure calculation"
        )
    elif detected_format == "nbo":
        message = (
            f"standalone NBO analysis cannot be inspected as a {program} "
            "run output; provide the parent quantum-chemistry output"
        )
    else:
        qe_program = {
            "qe-bands": "bands.x",
            "qe-dos": "dos.x",
            "qe-projwfc": "projwfc.x",
            "qe-pp": "pp.x",
        }[detected_format]
        message = (
            f"Quantum ESPRESSO {qe_program} output is not a supported "
            "primary run output; inspect the preceding pw.x output instead"
        )
    raise RunInspectionError(
        "unsupported_output_format",
        message,
        program=program,
        detected_format=detected_format,
    )


def inspect_run(
    backend: ProgramBackend,
    output_file: str | Path,
    *,
    resolved_by: ResolutionMethod,
    artifact_files: Iterable[str | Path] = (),
) -> dict[str, Any]:
    path = Path(output_file).expanduser().resolve()
    if not path.is_file():
        raise RunInspectionError(
            "source_not_file",
            f"run output is not a readable file: {path}",
            program=backend.name,
        )
    size_bytes = path.stat().st_size
    if size_bytes > PRIMARY_OUTPUT_LIMIT_BYTES:
        raise RunInspectionError(
            "primary_output_too_large",
            (
                f"run output exceeds the {PRIMARY_OUTPUT_LIMIT_BYTES}-byte "
                f"inspection limit: {path}"
            ),
            program=backend.name,
        )
    validate_primary_output_format(path, program=backend.name)
    if not backend.supports(ProgramCapability.OUTPUT_PARSE):
        raise RunInspectionError(
            "unsupported_capability",
            f"{backend.name!r} does not support output parsing",
            program=backend.name,
        )
    assert backend.parser is not None

    try:
        parsed = backend.parser.parse_output(str(path))
    except Exception as exc:
        raise RunInspectionError(
            "output_parse_failed",
            f"{backend.name} could not parse {path}: {exc}",
            program=backend.name,
            exception_type=type(exc).__name__,
        ) from exc

    if not isinstance(parsed, Mapping):
        raise RunInspectionError(
            "invalid_parser_result",
            f"{backend.name} parser returned {type(parsed).__name__}, not a mapping",
            program=backend.name,
        )

    classification = classify_artifact(backend, path)
    uncertainty = _classification_uncertainty(classification.status)
    artifacts, artifact_uncertainty, text_excerpt_budget = _observe_artifacts(
        backend,
        path,
        classification,
        artifact_files,
    )
    uncertainty.extend(artifact_uncertainty)
    consistency, consistency_uncertainty = (
        inspect_input_output_consistency(
            backend,
            path,
            parsed,
            artifacts,
        )
    )
    uncertainty.extend(consistency_uncertainty)
    parsed_program = parsed.get("program")
    if parsed_program not in (None, backend.name):
        uncertainty.append({
            "code": "parser_program_mismatch",
            "message": (
                f"Resolved backend is {backend.name!r}, but the parser "
                f"reported {parsed_program!r}."
            ),
            "impact": "Verify the program override and source artifact.",
        })

    diagnosis, diagnosis_source, diagnosis_uncertainty = _diagnose(
        backend,
        parsed,
    )
    uncertainty.extend(diagnosis_uncertainty)
    tasks = list(parsed.get("tasks") or [])
    if not tasks:
        uncertainty.append({
            "code": "no_tasks_detected",
            "message": "The parser did not identify a calculation task.",
            "impact": "Scientific completion cannot be established.",
        })

    program_version = parsed.get("program_version")
    if not program_version:
        program_version = _detect_version(backend, path)

    return {
        "schema_version": RUN_INSPECTION_SCHEMA,
        "program": {
            "name": backend.name,
            "version": program_version,
            "resolved_by": resolved_by,
        },
        "source": {
            "path": str(path),
            "size_bytes": size_bytes,
        },
        "assessment": {
            "source": diagnosis_source,
            "verdict": diagnosis["verdict"],
        },
        "evidence": {
            "artifact_classification": _serialize_classification(
                classification
            ),
            "artifacts": artifacts,
            "text_excerpt_budget": text_excerpt_budget,
            "input_output_consistency": consistency,
            "tasks": tasks,
            "derived": dict(parsed.get("derived") or {}),
            "diagnostics": _normalize_diagnostics(
                parsed.get("diagnostics") or [],
                path,
            ),
            "diagnosis_anchors": diagnosis.get("anchors", []),
        },
        "uncertainty": uncertainty,
        "next_actions": diagnosis["next_actions"],
    }


def _observe_artifacts(
    backend: ProgramBackend,
    primary_output: Path,
    primary_classification: ArtifactClassification,
    artifact_files: Iterable[str | Path],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
    dict[str, int],
]:
    requested = [(primary_output, "primary_output")]
    seen = {primary_output}
    for artifact_file in artifact_files:
        path = Path(artifact_file).expanduser().resolve()
        if path not in seen:
            requested.append((path, "related"))
            seen.add(path)

    observations = []
    uncertainty = []
    remaining_text_bytes = RELATED_TEXT_TOTAL_LIMIT_BYTES
    skipped_text_artifacts = 0
    for path, relationship in requested:
        text_excerpt = None
        text_uncertainty = []
        exists = path.exists()
        if path.is_file():
            entry_type = "file"
            size_bytes = path.stat().st_size
            classification = (
                primary_classification
                if relationship == "primary_output"
                else classify_artifact(backend, path)
            )
        elif path.is_dir():
            entry_type = "directory"
            size_bytes = None
            classification = None
        else:
            entry_type = None
            size_bytes = None
            classification = classify_artifact(backend, path)

        text_candidate = (
            classification.candidates[0]
            if (
                classification is not None
                and classification.status == "matched"
                and classification.candidates[0].content_kind == "text"
            )
            else None
        )
        if (
            relationship == "related"
            and text_candidate is not None
            and entry_type == "file"
        ):
            excerpt_limit = min(
                RELATED_TEXT_LIMIT_BYTES,
                remaining_text_bytes,
            )
            if excerpt_limit == 0:
                skipped_text_artifacts += 1
                text_uncertainty = [{
                    "code": "related_artifact_text_budget_exhausted",
                    "message": (
                        f"Text excerpt budget was exhausted before reading: "
                        f"{path}"
                    ),
                    "impact": (
                        "Artifact metadata is present, but its text was "
                        "omitted."
                    ),
                }]
            elif ArtifactRole.STDERR in text_candidate.roles:
                text_excerpt, text_uncertainty = read_text_excerpt(
                    path,
                    size_bytes,
                    excerpt_limit,
                    tail_only=True,
                )
            else:
                text_excerpt, text_uncertainty = read_text_excerpt(
                    path,
                    size_bytes,
                    excerpt_limit,
                    tail_only=False,
                )
            if text_excerpt is not None:
                remaining_text_bytes -= text_excerpt["bytes_read"]

        observation = {
            "path": str(path),
            "relationship": relationship,
            "exists": exists,
            "entry_type": entry_type,
            "size_bytes": size_bytes,
            "classification": (
                _serialize_classification(classification)
                if classification is not None
                else None
            ),
        }
        if text_excerpt is not None:
            observation["text_excerpt"] = text_excerpt
        observations.append(observation)
        uncertainty.extend(text_uncertainty)

        if relationship == "primary_output":
            continue
        if not exists:
            uncertainty.append({
                "code": "related_artifact_missing",
                "message": f"Related artifact does not exist: {path}",
                "impact": "That artifact could not contribute run evidence.",
            })
        elif entry_type != "file":
            uncertainty.append({
                "code": "related_artifact_not_file",
                "message": f"Related artifact is not a file: {path}",
                "impact": (
                    "Directories and special filesystem entries are not "
                    "inspected."
                ),
            })
        elif classification is not None and classification.status == "ambiguous":
            uncertainty.append({
                "code": "related_artifact_classification_ambiguous",
                "message": (
                    f"Related artifact matches more than one kind: {path}"
                ),
                "impact": "Its role requires confirmation before reuse.",
            })
        elif classification is not None and classification.status == "unmatched":
            uncertainty.append({
                "code": "related_artifact_kind_unmatched",
                "message": (
                    f"Related artifact matches no declared kind: {path}"
                ),
                "impact": "Its role in the run could not be established.",
            })
    return observations, uncertainty, {
        "limit_bytes": RELATED_TEXT_TOTAL_LIMIT_BYTES,
        "bytes_read": (
            RELATED_TEXT_TOTAL_LIMIT_BYTES - remaining_text_bytes
        ),
        "remaining_bytes": remaining_text_bytes,
        "skipped_artifacts": skipped_text_artifacts,
    }


def _serialize_classification(
    classification: ArtifactClassification,
) -> dict[str, Any]:
    return {
        "status": classification.status,
        "candidates": [
            {
                "kind": candidate.kind,
                "roles": sorted(
                    role.value for role in candidate.roles
                ),
                "content_kind": candidate.content_kind,
                "evidence": candidate.evidence,
                "matched_by": candidate.matched_by,
                "matched_value": candidate.matched_value,
            }
            for candidate in classification.candidates
        ],
    }


def _diagnose(
    backend: ProgramBackend,
    parsed: Mapping[str, Any],
) -> tuple[dict[str, Any], str, list[dict[str, str]]]:
    if (
        backend.supports(ProgramCapability.DIAGNOSIS_RUN)
        and backend.diagnostics is not None
    ):
        try:
            diagnosis = backend.diagnostics.diagnose(parsed)
        except Exception as exc:
            fallback = _inferred_diagnosis(parsed)
            return fallback, "task_outcomes", [{
                "code": "backend_diagnosis_failed",
                "message": (
                    f"{backend.name} diagnosis failed with "
                    f"{type(exc).__name__}: {exc}"
                ),
                "impact": "The verdict uses parsed task outcomes only.",
            }]
        canonical = _canonical_diagnosis(diagnosis)
        if canonical is not None:
            return canonical, "backend_diagnosis", []
        fallback = _inferred_diagnosis(parsed)
        return fallback, "task_outcomes", [{
            "code": "noncanonical_backend_diagnosis",
            "message": (
                f"{backend.name} diagnosis did not provide a canonical verdict."
            ),
            "impact": "The verdict uses parsed task outcomes only.",
        }]

    embedded = _canonical_diagnosis(parsed.get("diagnosis"))
    if embedded is not None:
        return embedded, "parser_diagnosis", []

    return _inferred_diagnosis(parsed), "task_outcomes", [{
        "code": "scientific_diagnosis_unavailable",
        "message": (
            f"{backend.name} has no guided scientific diagnosis adapter."
        ),
        "impact": "The verdict reflects parser task outcomes, not a full review.",
    }]


def _canonical_diagnosis(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    verdict = value.get("verdict")
    if not isinstance(verdict, Mapping) or not isinstance(
        verdict.get("label"),
        str,
    ):
        return None
    try:
        confidence = float(verdict.get("confidence", 0.5))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    reasons = verdict.get("reasons") or []
    if not isinstance(reasons, (list, tuple)) or not all(
        isinstance(reason, str) for reason in reasons
    ):
        return None
    next_actions = value.get("next_actions") or []
    if not isinstance(next_actions, (list, tuple)):
        return None
    anchors = value.get("anchors") or []
    if not isinstance(anchors, (list, tuple)):
        return None
    return {
        "verdict": {
            "label": verdict["label"],
            "confidence": confidence,
            "reasons": list(reasons),
        },
        "next_actions": _normalize_next_actions(next_actions),
        "anchors": [
            {
                "kind": anchor.get("kind", "info"),
                "message": anchor.get("message", ""),
                "line": anchor.get("line"),
                "file": anchor.get("file"),
            }
            for anchor in anchors
            if isinstance(anchor, Mapping)
            and isinstance(anchor.get("message"), str)
        ],
    }


def _normalize_next_actions(values: Iterable[Any]) -> list[dict[str, Any]]:
    normalized = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        action = value.get("action") or value.get("tool")
        if not isinstance(action, str) or not action:
            continue
        normalized.append({**value, "action": action})
    return normalized


def _inferred_diagnosis(parsed: Mapping[str, Any]) -> dict[str, Any]:
    tasks = [
        task
        for task in parsed.get("tasks") or []
        if isinstance(task, Mapping)
    ]
    outcomes = [
        str(task.get("outcome") or "unknown").lower()
        for task in tasks
    ]
    failed = sum(outcome in {"failed", "error", "aborted"} for outcome in outcomes)
    incomplete = sum(
        outcome in {"incomplete", "running", "unknown"}
        for outcome in outcomes
    )
    successful = sum(
        outcome in {"success", "completed", "converged", "ok"}
        for outcome in outcomes
    )
    derived = parsed.get("derived")
    scf_converged = (
        derived.get("scf_converged")
        if isinstance(derived, Mapping)
        else None
    )

    if failed:
        verdict = {
            "label": "failed",
            "confidence": 0.7,
            "reasons": [f"{failed} parsed task(s) report failure."],
        }
    elif scf_converged is True:
        verdict = {
            "label": "converged",
            "confidence": 0.7,
            "reasons": ["The parser reports SCF convergence."],
        }
    elif incomplete:
        verdict = {
            "label": "incomplete",
            "confidence": 0.6,
            "reasons": [
                f"{incomplete} parsed task(s) have no successful outcome."
            ],
        }
    elif tasks and successful == len(tasks):
        verdict = {
            "label": "completed",
            "confidence": 0.7,
            "reasons": [f"All {len(tasks)} parsed task(s) completed."],
        }
    elif derived:
        verdict = {
            "label": "results_parsed",
            "confidence": 0.45,
            "reasons": [
                "The parser extracted results without a reliable task outcome."
            ],
        }
    else:
        verdict = {
            "label": "unknown",
            "confidence": 0.2,
            "reasons": ["No reliable completion evidence was parsed."],
        }
    return {"verdict": verdict, "next_actions": []}


def _normalize_diagnostics(
    values: Any,
    path: Path,
) -> list[dict[str, Any]]:
    normalized = []
    for value in values:
        if isinstance(value, Mapping):
            normalized.append({
                "kind": value.get("kind", "info"),
                "message": str(value.get("message", "")),
                "line": value.get("line"),
                "file": value.get("file") or str(path),
            })
        else:
            normalized.append({
                "kind": "info",
                "message": str(value),
                "line": None,
                "file": str(path),
            })
    return normalized


def _classification_uncertainty(
    status: str,
) -> list[dict[str, str]]:
    if status == "ambiguous":
        return [{
            "code": "artifact_classification_ambiguous",
            "message": "The filename matches more than one artifact kind.",
            "impact": "Use the parsed evidence to confirm the artifact role.",
        }]
    if status == "unmatched":
        return [{
            "code": "artifact_kind_unmatched",
            "message": "The filename does not match a declared artifact kind.",
            "impact": "The program was resolved from content or an override.",
        }]
    return []


def _detect_version(
    backend: ProgramBackend,
    path: Path,
) -> str | None:
    try:
        with path.open("rb") as handle:
            head = handle.read(32 * 1024).decode(
                "utf-8",
                errors="replace",
            )
    except OSError:
        return None
    return backend.detect_version(head)


def _detect_standalone_output_format(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            head = handle.read(_OUTPUT_IDENTITY_HEAD_BYTES).decode(
                "utf-8",
                errors="replace",
            )
    except OSError:
        return None

    return detect_standalone_output_format(head)


__all__ = [
    "PRIMARY_OUTPUT_LIMIT_BYTES",
    "RELATED_TEXT_LIMIT_BYTES",
    "RELATED_TEXT_TOTAL_LIMIT_BYTES",
    "RUN_INSPECTION_SCHEMA",
    "RunInspectionError",
    "inspect_run",
    "validate_primary_output_format",
]

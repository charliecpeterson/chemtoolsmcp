"""Classify explicit artifact paths from launch expectations and backend declarations.

This module performs no directory discovery or content inspection. Callers
choose the backend and provide the candidate paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from chemtools.core.artifacts import (
    ArtifactRole,
    ExpectedArtifact,
    StepRef,
)
from chemtools.core.program import ProgramBackend


ClassificationEvidence = Literal["declared", "inferred"]
ClassificationMatch = Literal["expectation", "filename", "extension"]
ClassificationStatus = Literal["matched", "ambiguous", "unmatched"]


@dataclass(frozen=True)
class ArtifactCandidate:
    kind: str
    roles: frozenset[ArtifactRole]
    evidence: ClassificationEvidence
    matched_by: ClassificationMatch
    matched_value: str
    content_kind: Literal["text", "binary", "unknown"] = "unknown"
    producing_step: StepRef | None = None
    expectation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "roles": sorted(role.value for role in self.roles),
            "content_kind": self.content_kind,
            "evidence": self.evidence,
            "matched_by": self.matched_by,
            "matched_value": self.matched_value,
            "producing_step": (
                self.producing_step.to_dict()
                if self.producing_step is not None
                else None
            ),
            "expectation_id": self.expectation_id,
        }


@dataclass(frozen=True)
class ArtifactClassification:
    program: str
    path: Path
    candidates: tuple[ArtifactCandidate, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "candidates", tuple(self.candidates))

    @property
    def status(self) -> ClassificationStatus:
        if not self.candidates:
            return "unmatched"
        if len(self.candidates) == 1:
            return "matched"
        return "ambiguous"

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": self.program,
            "path": str(self.path),
            "status": self.status,
            "candidates": [
                candidate.to_dict() for candidate in self.candidates
            ],
        }


def classify_artifact(
    backend: ProgramBackend,
    path: str | Path,
    expectations: Iterable[ExpectedArtifact] = (),
) -> ArtifactClassification:
    candidate_path = Path(path)
    declared = tuple(
        ArtifactCandidate(
            kind=expectation.kind,
            roles=expectation.roles,
            content_kind=(
                backend.artifact_kinds[expectation.kind].content_kind
                if expectation.kind in backend.artifact_kinds
                else "unknown"
            ),
            evidence="declared",
            matched_by="expectation",
            matched_value=str(expectation.location.path),
            producing_step=expectation.producing_step,
            expectation_id=expectation.expectation_id,
        )
        for expectation in expectations
        if expectation.location.path == candidate_path
    )
    if declared:
        return ArtifactClassification(
            program=backend.name,
            path=candidate_path,
            candidates=declared,
        )

    filename_matches = tuple(
        ArtifactCandidate(
            kind=kind,
            roles=spec.default_roles,
            content_kind=spec.content_kind,
            evidence="inferred",
            matched_by="filename",
            matched_value=candidate_path.name,
        )
        for kind, spec in backend.artifact_kinds.items()
        if candidate_path.name in spec.filenames
    )
    if filename_matches:
        return ArtifactClassification(
            program=backend.name,
            path=candidate_path,
            candidates=filename_matches,
        )

    extension_matches = []
    for kind, spec in backend.artifact_kinds.items():
        matching_extensions = tuple(
            extension
            for extension in spec.extensions
            if candidate_path.name.endswith(extension)
        )
        if matching_extensions:
            extension_matches.append(
                ArtifactCandidate(
                    kind=kind,
                    roles=spec.default_roles,
                    content_kind=spec.content_kind,
                    evidence="inferred",
                    matched_by="extension",
                    matched_value=max(matching_extensions, key=len),
                )
            )

    return ArtifactClassification(
        program=backend.name,
        path=candidate_path,
        candidates=tuple(extension_matches),
    )


def classify_artifacts(
    backend: ProgramBackend,
    paths: Iterable[str | Path],
    expectations: Iterable[ExpectedArtifact] = (),
) -> tuple[ArtifactClassification, ...]:
    declared = tuple(expectations)
    return tuple(
        classify_artifact(backend, path, declared)
        for path in paths
    )


__all__ = [
    "ArtifactCandidate",
    "ArtifactClassification",
    "ClassificationEvidence",
    "ClassificationMatch",
    "ClassificationStatus",
    "classify_artifact",
    "classify_artifacts",
]

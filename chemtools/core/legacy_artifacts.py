"""Read-only projection from legacy run columns to artifact collections.

Recorded paths are classification hints, not filesystem observations. Legacy
parent IDs remain separate from snapshot-level provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping
from uuid import NAMESPACE_URL, uuid5

from chemtools.core.artifact_classification import ArtifactClassification
from chemtools.core.artifact_classification import classify_artifact
from chemtools.core.artifacts import ArtifactRef, ArtifactRole, RunArtifacts
from chemtools.core.program import ProgramBackend
from chemtools.core.run_records import get_run_summary


LegacyPathField = Literal["input_file", "output_file"]
LEGACY_ARTIFACT_PROJECTION_SCHEMA = "chemtools.legacy-artifact-projection/1"


@dataclass(frozen=True)
class LegacyPathProjection:
    field: LegacyPathField
    path: Path
    recorded_role: ArtifactRole
    classification: ArtifactClassification
    artifact_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True)
class LegacyParentReference:
    """Recorded run link that does not identify a consumed artifact snapshot."""

    run_id: int
    run_uid: str | None


@dataclass(frozen=True)
class LegacyRunArtifactProjection:
    artifacts: RunArtifacts
    paths: tuple[LegacyPathProjection, ...]
    parent: LegacyParentReference | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "paths", tuple(self.paths))


def _artifact_id(run_uid: str, path: Path) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            f"chemtools:legacy-run-artifact:{run_uid}:{path}",
        )
    )


def _parent_reference(
    run: Mapping[str, Any],
) -> LegacyParentReference | None:
    parent_run_id = run.get("parent_run_id")
    if parent_run_id is None:
        return None
    parent_run_uid = None
    for ancestor in run.get("restart_chain") or ():
        if ancestor.get("run_id") == parent_run_id:
            parent_run_uid = ancestor.get("run_uid")
            break
    return LegacyParentReference(
        run_id=parent_run_id,
        run_uid=parent_run_uid,
    )


def project_legacy_run_artifacts(
    run: Mapping[str, Any],
    backend: ProgramBackend,
) -> LegacyRunArtifactProjection:
    run_uid = run.get("run_uid")
    if not isinstance(run_uid, str) or not run_uid:
        raise ValueError("legacy run projection requires run_uid")
    recorded_program = run.get("program")
    if (
        recorded_program is not None
        and recorded_program.lower() != backend.name
    ):
        raise ValueError(
            f"run program {recorded_program!r} does not match "
            f"backend {backend.name!r}"
        )

    classifications: list[LegacyPathProjection] = []
    artifact_fields: dict[
        Path,
        tuple[str, frozenset[ArtifactRole], list[LegacyPathField]],
    ] = {}
    for field, recorded_role in (
        ("input_file", ArtifactRole.PRIMARY_INPUT),
        ("output_file", ArtifactRole.PRIMARY_OUTPUT),
    ):
        value = run.get(field)
        if not value:
            continue
        path = Path(value)
        classification = classify_artifact(backend, path)
        projected_id = None
        if classification.status == "matched":
            candidate = classification.candidates[0]
            projected_id = _artifact_id(run_uid, path)
            existing = artifact_fields.get(path)
            if existing is None:
                artifact_fields[path] = (
                    candidate.kind,
                    candidate.roles | frozenset({recorded_role}),
                    [field],
                )
            else:
                kind, roles, fields = existing
                if candidate.kind != kind:
                    raise ValueError(
                        f"path {str(path)!r} produced inconsistent kinds"
                    )
                artifact_fields[path] = (
                    kind,
                    roles | frozenset({recorded_role}),
                    [*fields, field],
                )
        classifications.append(
            LegacyPathProjection(
                field=field,
                path=path,
                recorded_role=recorded_role,
                classification=classification,
                artifact_id=projected_id,
            )
        )

    artifacts = tuple(
        ArtifactRef(
            artifact_id=_artifact_id(run_uid, path),
            roles=roles,
            kind=kind,
            metadata={
                "compatibility_source": {
                    "schema": LEGACY_ARTIFACT_PROJECTION_SCHEMA,
                    "fields": fields,
                    "recorded_path": str(path),
                    "evidence": "inferred",
                }
            },
        )
        for path, (kind, roles, fields) in artifact_fields.items()
    )
    return LegacyRunArtifactProjection(
        artifacts=RunArtifacts(
            run_uid=run_uid,
            artifacts=artifacts,
        ),
        paths=tuple(classifications),
        parent=_parent_reference(run),
    )


def project_registered_run_artifacts(
    run_uid: str,
    backend: ProgramBackend,
    db_path: str | Path | None = None,
) -> LegacyRunArtifactProjection | None:
    run = get_run_summary(run_uid=run_uid, db_path=db_path)
    if run is None:
        return None
    return project_legacy_run_artifacts(run, backend)


__all__ = [
    "LEGACY_ARTIFACT_PROJECTION_SCHEMA",
    "LegacyParentReference",
    "LegacyPathField",
    "LegacyPathProjection",
    "LegacyRunArtifactProjection",
    "project_legacy_run_artifacts",
    "project_registered_run_artifacts",
]

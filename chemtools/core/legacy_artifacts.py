"""Compatibility imports for legacy artifact projection."""

from chemtools.application.legacy_artifacts import (
    LEGACY_ARTIFACT_PROJECTION_SCHEMA,
    LegacyParentReference,
    LegacyPathField,
    LegacyPathProjection,
    LegacyRunArtifactProjection,
    project_legacy_run_artifacts,
    project_registered_run_artifacts,
)

__all__ = [
    "LEGACY_ARTIFACT_PROJECTION_SCHEMA",
    "LegacyParentReference",
    "LegacyPathField",
    "LegacyPathProjection",
    "LegacyRunArtifactProjection",
    "project_legacy_run_artifacts",
    "project_registered_run_artifacts",
]

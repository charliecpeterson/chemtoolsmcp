"""Contracts for projecting legacy run columns without invented lineage."""

from __future__ import annotations

from pathlib import Path

import pytest

from chemtools.core.artifacts import ArtifactRole, RunArtifacts
from chemtools.application.legacy_artifacts import (
    LEGACY_ARTIFACT_PROJECTION_SCHEMA,
    LegacyParentReference,
    project_legacy_run_artifacts,
    project_registered_run_artifacts,
)
from chemtools.persistence.runs import register_run
from chemtools.mcp.catalog import BUILTIN_BACKENDS, load_backend


RUN_UID = "626e64b6-9149-44fb-a11d-04d7f207d12f"
PARENT_UID = "fd9d1927-5f2d-4b42-820c-074bb4f41eba"
BACKENDS = {
    spec.name: load_backend(spec)
    for spec in BUILTIN_BACKENDS
}


def test_registered_nwchem_paths_project_without_filesystem_observations(
    tmp_path,
):
    db_path = tmp_path / "registry.db"
    register_run(
        "uo2",
        program="nwchem",
        run_uid=RUN_UID,
        input_file="/does/not/exist/uo2.nw",
        output_file="/does/not/exist/uo2.out",
        db_path=str(db_path),
    )

    projection = project_registered_run_artifacts(
        RUN_UID,
        BACKENDS["nwchem"],
        db_path,
    )

    assert projection is not None
    assert projection == project_registered_run_artifacts(
        RUN_UID,
        BACKENDS["nwchem"],
        db_path,
    )
    assert projection.artifacts.run_uid == RUN_UID
    assert projection.artifacts.observations == ()
    assert projection.artifacts.expectations == ()
    assert projection.artifacts.provenance == ()
    assert tuple(
        artifact.kind for artifact in projection.artifacts.artifacts
    ) == ("nwchem.input", "nwchem.output")
    assert tuple(
        path.classification.status for path in projection.paths
    ) == ("matched", "matched")
    assert all(
        path.classification.candidates[0].evidence == "inferred"
        for path in projection.paths
    )
    assert projection.artifacts.artifacts[0].metadata[
        "compatibility_source"
    ] == {
        "schema": LEGACY_ARTIFACT_PROJECTION_SCHEMA,
        "fields": ("input_file",),
        "recorded_path": "/does/not/exist/uo2.nw",
        "evidence": "inferred",
    }


def test_recorded_field_role_is_added_to_backend_roles():
    projection = project_legacy_run_artifacts(
        {
            "run_uid": RUN_UID,
            "program": "molcas",
            "input_file": "cas.input",
            "output_file": "INPORB",
        },
        BACKENDS["molcas"],
    )

    orbital = projection.artifacts.artifacts[1]

    assert orbital.kind == "molcas.orbitals"
    assert orbital.roles == frozenset({
        ArtifactRole.PRIMARY_OUTPUT,
        ArtifactRole.CHECKPOINT,
        ArtifactRole.ORBITAL,
    })
    assert projection.paths[1].recorded_role == ArtifactRole.PRIMARY_OUTPUT
    assert projection.paths[1].classification.candidates[0].roles == (
        frozenset({ArtifactRole.CHECKPOINT, ArtifactRole.ORBITAL})
    )


def test_specific_grasp_summary_projects_while_unknown_input_remains_unresolved():
    grasp = project_legacy_run_artifacts(
        {
            "run_uid": RUN_UID,
            "program": "grasp",
            "input_file": "interactive.stdin",
            "output_file": "run.sum",
        },
        BACKENDS["grasp"],
    )

    assert len(grasp.artifacts.artifacts) == 1
    assert grasp.artifacts.artifacts[0].kind == "grasp.rmcdhf_summary"
    assert tuple(path.classification.status for path in grasp.paths) == (
        "unmatched",
        "matched",
    )
    assert tuple(
        candidate.kind
        for candidate in grasp.paths[1].classification.candidates
    ) == ("grasp.rmcdhf_summary",)
    assert grasp.paths[0].artifact_id is None
    assert grasp.paths[1].artifact_id == grasp.artifacts.artifacts[0].artifact_id


def test_same_recorded_path_projects_one_multi_role_artifact():
    projection = project_legacy_run_artifacts(
        {
            "run_uid": RUN_UID,
            "program": "nwchem",
            "input_file": "restart.out",
            "output_file": "restart.out",
        },
        BACKENDS["nwchem"],
    )

    assert len(projection.artifacts.artifacts) == 1
    artifact = projection.artifacts.artifacts[0]
    assert artifact.roles == frozenset({
        ArtifactRole.PRIMARY_INPUT,
        ArtifactRole.PRIMARY_OUTPUT,
    })
    assert artifact.metadata["compatibility_source"]["fields"] == (
        "input_file",
        "output_file",
    )
    assert projection.paths[0].artifact_id == projection.paths[1].artifact_id


def test_parent_run_reference_does_not_create_provenance(tmp_path):
    db_path = tmp_path / "registry.db"
    parent = register_run(
        "parent",
        run_uid=PARENT_UID,
        db_path=str(db_path),
    )
    register_run(
        "child",
        run_uid=RUN_UID,
        parent_run_id=parent["run_id"],
        db_path=str(db_path),
    )

    projection = project_registered_run_artifacts(
        RUN_UID,
        BACKENDS["nwchem"],
        db_path,
    )

    assert projection is not None
    assert projection.parent == LegacyParentReference(
        run_id=parent["run_id"],
        run_uid=PARENT_UID,
    )
    assert projection.artifacts.provenance == ()


def test_projection_rejects_backend_mismatch_and_unknown_run(tmp_path):
    with pytest.raises(
        ValueError,
        match="does not match backend",
    ):
        project_legacy_run_artifacts(
            {
                "run_uid": RUN_UID,
                "program": "dirac",
                "output_file": "run.out",
            },
            BACKENDS["nwchem"],
        )

    assert project_registered_run_artifacts(
        RUN_UID,
        BACKENDS["nwchem"],
        tmp_path / "registry.db",
    ) is None

"""Exact contracts for artifact identity, observations, and provenance."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from chemtools.core.artifacts import (
    RUN_ARTIFACTS_SCHEMA,
    ArtifactLocation,
    ArtifactObservation,
    ArtifactRef,
    ArtifactRole,
    ArtifactSnapshotRef,
    ExpectedArtifact,
    FreshnessAssessment,
    ProducerIdentity,
    ProvenanceRecord,
    RunArtifacts,
    StepRef,
)


OBSERVED_AT = datetime(2026, 7, 30, 19, 15, tzinfo=timezone.utc)


def test_artifact_role_values_are_exact():
    assert tuple(role.value for role in ArtifactRole) == (
        "primary_input",
        "primary_output",
        "auxiliary_input",
        "auxiliary_output",
        "stdout",
        "stderr",
        "checkpoint",
        "orbital",
        "wavefunction",
        "wavefunction_seed",
        "pseudopotential",
        "volumetric_data",
        "scheduler_script",
    )


def _artifact(
    artifact_id: str,
    *,
    roles: frozenset[ArtifactRole],
    kind: str,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        roles=roles,
        kind=kind,
        metadata={"configuration": {"shell": "5f", "occupancy": [3, 4]}},
    )


def _observation(
    artifact_id: str,
    observation_id: str,
    path: str,
    digest_character: str,
) -> ArtifactObservation:
    return ArtifactObservation(
        observation_id=observation_id,
        artifact_id=artifact_id,
        observed_at=OBSERVED_AT,
        location=ArtifactLocation(
            path=Path(path),
            entry_type="file",
            root_name="scratch",
            relative_path=Path(path).relative_to("/u/scratch"),
        ),
        exists=True,
        size_bytes=128,
        modified_ns=1_753_900_500_000_000_000,
        sha256=digest_character * 64,
        hash_status="verified",
    )


def test_run_artifacts_round_trip_preserves_grasp_multi_donor_lineage():
    donor_a = _artifact(
        "artifact-donor-a",
        roles=frozenset(
            {ArtifactRole.ORBITAL, ArtifactRole.WAVEFUNCTION_SEED}
        ),
        kind="atsp2k.radial_orbitals",
    )
    donor_b = _artifact(
        "artifact-donor-b",
        roles=frozenset(
            {ArtifactRole.ORBITAL, ArtifactRole.WAVEFUNCTION_SEED}
        ),
        kind="grasp.radial_wfn",
    )
    merged = _artifact(
        "artifact-merged",
        roles=frozenset(
            {
                ArtifactRole.AUXILIARY_INPUT,
                ArtifactRole.ORBITAL,
                ArtifactRole.WAVEFUNCTION_SEED,
            }
        ),
        kind="grasp.radial_wfn",
    )
    observations = (
        _observation(
            donor_a.artifact_id,
            "observation-donor-a",
            "/u/scratch/run-a/seed.w",
            "a",
        ),
        _observation(
            donor_b.artifact_id,
            "observation-donor-b",
            "/u/scratch/run-b/seed.w",
            "b",
        ),
        _observation(
            merged.artifact_id,
            "observation-merged",
            "/u/scratch/run-c/merged.w",
            "c",
        ),
    )
    event = ProvenanceRecord(
        event_id="event-merge",
        event_type="merged",
        occurred_at=OBSERVED_AT,
        actor=ProducerIdentity(
            producer_type="chemtools",
            name="grasp_seed_builder",
            version="1",
        ),
        run_uid="run-c",
        step_id="seed-merge",
        inputs=(
            ArtifactSnapshotRef(
                artifact_id=donor_a.artifact_id,
                observation_id=observations[0].observation_id,
            ),
            ArtifactSnapshotRef(
                artifact_id=donor_b.artifact_id,
                observation_id=observations[1].observation_id,
            ),
        ),
        outputs=(
            ArtifactSnapshotRef(
                artifact_id=merged.artifact_id,
                observation_id=observations[2].observation_id,
            ),
        ),
        evidence="recorded",
        parameters={
            "duplicate_rule": "prefer_lower_energy",
            "donor_priority": ["artifact-donor-a", "artifact-donor-b"],
        },
    )
    expectation = ExpectedArtifact(
        expectation_id="expected-merged-seed",
        roles=frozenset({ArtifactRole.WAVEFUNCTION_SEED}),
        kind="grasp.radial_wfn",
        location=observations[2].location,
        required=True,
        producing_step=StepRef(run_uid="run-c", step_id="seed-merge"),
    )
    collection = RunArtifacts(
        run_uid="run-c",
        artifacts=(donor_a, donor_b, merged),
        observations=observations,
        expectations=(expectation,),
        provenance=(event,),
    )

    payload = collection.to_dict()
    restored = RunArtifacts.from_dict(payload)

    assert restored == collection
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload
    assert len(payload["artifacts"]) == 3
    assert len(payload["observations"]) == 3
    assert payload["provenance"][0]["inputs"] == [
        {
            "artifact_id": "artifact-donor-a",
            "observation_id": "observation-donor-a",
        },
        {
            "artifact_id": "artifact-donor-b",
            "observation_id": "observation-donor-b",
        },
    ]
    assert payload["provenance"][0]["outputs"] == [
        {
            "artifact_id": "artifact-merged",
            "observation_id": "observation-merged",
        }
    ]


def test_empty_run_artifacts_serialization_shape_is_exact():
    assert RunArtifacts(run_uid="run-empty").to_dict() == {
        "schema": RUN_ARTIFACTS_SCHEMA,
        "run": {"run_uid": "run-empty"},
        "artifacts": [],
        "observations": [],
        "expectations": [],
        "provenance": [],
    }


def test_run_artifacts_rejects_observation_without_artifact():
    observation = _observation(
        "artifact-missing",
        "observation-missing",
        "/u/scratch/run/missing.out",
        "d",
    )

    with pytest.raises(
        ValueError,
        match=(
            "^observation 'observation-missing' references unknown artifact "
            "'artifact-missing'$"
        ),
    ):
        RunArtifacts(run_uid="run", observations=(observation,))


def test_run_artifacts_rejects_unobserved_provenance_output():
    artifact = _artifact(
        "artifact-output",
        roles=frozenset({ArtifactRole.PRIMARY_OUTPUT}),
        kind="dirac.output",
    )
    observation = _observation(
        artifact.artifact_id,
        "observation-output",
        "/u/scratch/run/dirac.out",
        "e",
    )
    event = ProvenanceRecord(
        event_id="event-move",
        event_type="moved",
        occurred_at=OBSERVED_AT,
        actor=ProducerIdentity(
            producer_type="manual",
            name="import",
        ),
        inputs=(
            ArtifactSnapshotRef(
                artifact_id=artifact.artifact_id,
                observation_id=observation.observation_id,
            ),
        ),
        outputs=(
            ArtifactSnapshotRef(
                artifact_id=artifact.artifact_id,
                observation_id="observation-after-move",
            ),
        ),
        evidence="recorded",
    )

    with pytest.raises(
        ValueError,
        match=(
            "^provenance event 'event-move' references unknown snapshot "
            "\\('artifact-output', 'observation-after-move'\\)$"
        ),
    ):
        RunArtifacts(
            run_uid="run",
            artifacts=(artifact,),
            observations=(observation,),
            provenance=(event,),
        )


def test_directory_observation_requires_versioned_manifest_pair():
    location = ArtifactLocation(
        path=Path("/u/scratch/qe/feo.save"),
        entry_type="directory",
    )

    with pytest.raises(
        ValueError,
        match=(
            "^directory manifest schema and digest must be provided together$"
        ),
    ):
        ArtifactObservation(
            observation_id="observation-save",
            artifact_id="artifact-save",
            observed_at=OBSERVED_AT,
            location=location,
            exists=True,
            directory_manifest_schema="chemtools.directory-manifest/1",
        )


def test_metadata_is_recursively_immutable():
    artifact = _artifact(
        "artifact-immutable",
        roles=frozenset({ArtifactRole.CHECKPOINT}),
        kind="dirac.checkpoint_h5",
    )

    with pytest.raises(TypeError):
        artifact.metadata["configuration"] = {}
    assert artifact.metadata["configuration"]["occupancy"] == (3, 4)


def test_freshness_requires_evidence_for_non_unknown_verdicts():
    with pytest.raises(
        ValueError,
        match=(
            "^non-unknown freshness verdicts require supporting evidence$"
        ),
    ):
        FreshnessAssessment(
            verdict="current",
            artifact_id="artifact-checkpoint",
            observation_id="observation-checkpoint",
        )

    assessment = FreshnessAssessment(
        verdict="stale",
        artifact_id="artifact-checkpoint",
        observation_id="observation-checkpoint",
        compared_with=("observation-required",),
        evidence=("recorded producer is older than the required snapshot",),
    )
    assert FreshnessAssessment.from_dict(assessment.to_dict()) == assessment

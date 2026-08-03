"""Migration and round-trip contracts for run and artifact persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from uuid import UUID

import pytest

from chemtools.core.artifact_registry import (
    ArtifactPersistenceConflict,
    UnknownRunUidError,
    load_run_artifacts,
    record_run_artifacts,
)
from chemtools.core.artifacts import (
    ArtifactLocation,
    ArtifactObservation,
    ArtifactRef,
    ArtifactRole,
    ArtifactSnapshotRef,
    ExpectedArtifact,
    ProducerIdentity,
    ProvenanceRecord,
    RunArtifacts,
    StepRef,
)
from chemtools.core.registry_db import connect_registry
from chemtools.core.run_registry import (
    get_run_summary,
    list_runs,
    register_run,
)


OBSERVED_AT = datetime(2026, 7, 30, 21, 45, tzinfo=timezone.utc)
RUN_UID = "4bdf4738-1ea9-4ccd-a131-a3b51ab046c6"
CONSUMER_UID = "b233053d-6512-44c0-85a7-cdb043f970e1"


def _collection(
    run_uid: str,
    *,
    include_converted: bool,
) -> RunArtifacts:
    source_step = StepRef(run_uid=RUN_UID, step_id="pwscf")
    source = ArtifactRef(
        artifact_id="artifact-qe-save",
        roles=frozenset(
            {ArtifactRole.CHECKPOINT, ArtifactRole.AUXILIARY_OUTPUT}
        ),
        kind="qe.save_directory",
        producing_step=source_step,
        metadata={"prefix": "uo2", "spin_orbit": True},
    )
    source_observation = ArtifactObservation(
        observation_id="observation-qe-save",
        artifact_id=source.artifact_id,
        observed_at=OBSERVED_AT,
        location=ArtifactLocation(
            path=Path("/u/scratch/qe/uo2.save"),
            entry_type="directory",
            root_name="scratch",
            relative_path=Path("qe/uo2.save"),
        ),
        exists=True,
        size_bytes=4096,
        modified_ns=1_753_911_900_000_000_000,
        hash_status="not_requested",
        directory_manifest_schema="chemtools.directory-manifest/1",
        directory_manifest_sha256="a" * 64,
    )
    if not include_converted:
        return RunArtifacts(
            run_uid=run_uid,
            artifacts=(source,),
            observations=(source_observation,),
        )

    converted_step = StepRef(run_uid=RUN_UID, step_id="pw2qmcpack")
    converted = ArtifactRef(
        artifact_id="artifact-qmcpack-h5",
        roles=frozenset(
            {ArtifactRole.AUXILIARY_INPUT, ArtifactRole.WAVEFUNCTION}
        ),
        kind="qmcpack.wavefunction_h5",
        producing_step=converted_step,
        metadata={"converter": {"name": "pw2qmcpack", "version": "7.4"}},
    )
    converted_observation = ArtifactObservation(
        observation_id="observation-qmcpack-h5",
        artifact_id=converted.artifact_id,
        observed_at=OBSERVED_AT,
        location=ArtifactLocation(
            path=Path("/u/scratch/qmcpack/uo2.pwscf.h5"),
            entry_type="file",
            root_name="scratch",
            relative_path=Path("qmcpack/uo2.pwscf.h5"),
        ),
        exists=True,
        size_bytes=8192,
        modified_ns=1_753_911_901_000_000_000,
        sha256="b" * 64,
        hash_status="verified",
    )
    expectation = ExpectedArtifact(
        expectation_id="expectation-qmcpack-h5",
        roles=converted.roles,
        kind=converted.kind,
        location=converted_observation.location,
        required=True,
        producing_step=converted_step,
    )
    event = ProvenanceRecord(
        event_id="event-pw2qmcpack",
        event_type="converted",
        occurred_at=OBSERVED_AT,
        actor=ProducerIdentity(
            producer_type="program",
            name="pw2qmcpack",
            version="7.4",
        ),
        inputs=(
            ArtifactSnapshotRef(
                artifact_id=source.artifact_id,
                observation_id=source_observation.observation_id,
            ),
        ),
        outputs=(
            ArtifactSnapshotRef(
                artifact_id=converted.artifact_id,
                observation_id=converted_observation.observation_id,
            ),
        ),
        evidence="recorded",
        run_uid=RUN_UID,
        step_id="pw2qmcpack",
        parameters={"write_psir": False, "source_format": "qe.save"},
    )
    return RunArtifacts(
        run_uid=run_uid,
        artifacts=(source, converted),
        observations=(source_observation, converted_observation),
        expectations=(expectation,),
        provenance=(event,),
    )


def test_register_run_adds_portable_uid_and_supports_uid_lookup(tmp_path):
    db_path = tmp_path / "registry.db"

    registered = register_run(
        "uo2-qmcpack",
        program="qmcpack",
        run_uid=RUN_UID,
        db_path=str(db_path),
    )

    assert registered == {
        "run_id": 1,
        "run_uid": RUN_UID,
        "job_name": "uo2-qmcpack",
        "status": "submitted",
        "program": "qmcpack",
    }
    assert get_run_summary(
        run_uid=RUN_UID,
        db_path=str(db_path),
    )["id"] == 1
    assert list_runs(db_path=str(db_path))[0]["run_uid"] == RUN_UID


def test_register_run_generates_canonical_uuid_and_rejects_non_uuid(tmp_path):
    db_path = tmp_path / "registry.db"

    registered = register_run("generated", db_path=str(db_path))

    assert str(UUID(registered["run_uid"])) == registered["run_uid"]
    with pytest.raises(
        ValueError,
        match="^run_uid must be a canonical UUID string$",
    ):
        register_run(
            "invalid",
            run_uid="not-a-uuid",
            db_path=str(db_path),
        )


def test_existing_run_rows_receive_stable_uids_during_migration(tmp_path):
    db_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(db_path)
    legacy.execute(
        """CREATE TABLE runs (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               job_name TEXT NOT NULL,
               status TEXT NOT NULL DEFAULT 'pending',
               campaign_id INTEGER,
               workflow_id INTEGER
           )"""
    )
    legacy.execute(
        "INSERT INTO runs (job_name, status) VALUES ('legacy', 'completed')"
    )
    legacy.commit()
    legacy.close()

    migrated = connect_registry(db_path)
    first_uid = migrated.execute(
        "SELECT run_uid FROM runs WHERE id = 1"
    ).fetchone()["run_uid"]
    migrated.close()

    reopened = connect_registry(db_path)
    second_uid = reopened.execute(
        "SELECT run_uid FROM runs WHERE id = 1"
    ).fetchone()["run_uid"]
    columns = {
        row["name"]
        for row in reopened.execute("PRAGMA table_info(runs)").fetchall()
    }
    reopened.close()

    assert str(UUID(first_uid)) == first_uid
    assert second_uid == first_uid
    assert {"program", "run_uid"} <= columns


def test_artifact_collection_round_trip_is_exact_and_append_only(tmp_path):
    db_path = tmp_path / "registry.db"
    register_run(
        "qe-to-qmcpack",
        program="qmcpack",
        run_uid=RUN_UID,
        db_path=str(db_path),
    )
    initial = _collection(RUN_UID, include_converted=False)
    complete = _collection(RUN_UID, include_converted=True)

    assert record_run_artifacts(initial, db_path) == {
        "run_uid": RUN_UID,
        "artifacts": 1,
        "observations": 1,
        "expectations": 0,
        "provenance_events": 0,
    }
    record_run_artifacts(complete, db_path)
    record_run_artifacts(complete, db_path)

    assert load_run_artifacts(RUN_UID, db_path) == complete
    with pytest.raises(
        ArtifactPersistenceConflict,
        match="membership must be an append-only extension",
    ):
        record_run_artifacts(initial, db_path)
    assert load_run_artifacts(RUN_UID, db_path) == complete


def test_shared_artifact_identity_is_reused_across_run_collections(tmp_path):
    db_path = tmp_path / "registry.db"
    for uid, name in ((RUN_UID, "producer"), (CONSUMER_UID, "consumer")):
        register_run(
            name,
            run_uid=uid,
            db_path=str(db_path),
        )

    producer = _collection(RUN_UID, include_converted=False)
    consumer = _collection(CONSUMER_UID, include_converted=False)
    record_run_artifacts(producer, db_path)
    record_run_artifacts(consumer, db_path)

    conn = connect_registry(db_path)
    counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "artifacts",
            "artifact_observations",
            "run_artifacts",
            "run_artifact_observations",
        )
    }
    foreign_key_violations = conn.execute(
        "PRAGMA foreign_key_check"
    ).fetchall()
    conn.close()

    assert counts == {
        "artifacts": 1,
        "artifact_observations": 1,
        "run_artifacts": 2,
        "run_artifact_observations": 2,
    }
    assert foreign_key_violations == []
    assert load_run_artifacts(CONSUMER_UID, db_path) == consumer


def test_conflicting_artifact_identity_rolls_back_run_membership(tmp_path):
    db_path = tmp_path / "registry.db"
    for uid in (RUN_UID, CONSUMER_UID):
        register_run(uid, run_uid=uid, db_path=str(db_path))
    record_run_artifacts(
        _collection(RUN_UID, include_converted=False),
        db_path,
    )
    conflicting = ArtifactRef(
        artifact_id="artifact-qe-save",
        roles=frozenset({ArtifactRole.CHECKPOINT}),
        kind="qe.different_kind",
    )

    with pytest.raises(
        ArtifactPersistenceConflict,
        match="already has different metadata",
    ):
        record_run_artifacts(
            RunArtifacts(
                run_uid=CONSUMER_UID,
                artifacts=(conflicting,),
            ),
            db_path,
        )

    assert load_run_artifacts(CONSUMER_UID, db_path) == RunArtifacts(
        run_uid=CONSUMER_UID
    )


def test_recording_artifacts_requires_registered_run(tmp_path):
    with pytest.raises(
        UnknownRunUidError,
        match=f"^run {RUN_UID!r} is not registered$",
    ):
        record_run_artifacts(
            RunArtifacts(run_uid=RUN_UID),
            tmp_path / "registry.db",
        )
    assert load_run_artifacts(
        RUN_UID,
        tmp_path / "registry.db",
    ) is None

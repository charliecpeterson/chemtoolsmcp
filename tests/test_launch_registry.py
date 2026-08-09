"""Persistent execution launch records and transition rules."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from chemtools.core.execution import (
    ExecutionLaunchRecord,
    ExecutionRunLink,
    ResourceRequest,
    StagedFile,
)
from chemtools.persistence.sqlite import connect_registry
from chemtools.persistence.runs import (
    get_run_summary,
    list_runs,
    register_run,
)
from chemtools.persistence.launches import (
    LaunchRecordConflict,
    create_launch_record,
    load_execution_run_link,
    load_launch_record,
    update_launch_record,
)


LAUNCH_ID = "00000000-0000-4000-8000-000000000011"
INSTANCE_ID = "00000000-0000-4000-8000-000000000012"
RUN_UID = "00000000-0000-4000-8000-000000000013"
CREATED_AT = datetime(2026, 7, 30, 15, 0, tzinfo=timezone.utc)


def _pending(tmp_path):
    return ExecutionLaunchRecord(
        launch_id=LAUNCH_ID,
        instance_id=INSTANCE_ID,
        target="linux4090",
        executor="local",
        program="nwchem",
        working_directory=tmp_path,
        argv=("nwchem", "water.nw"),
        environment_keys=("OMP_NUM_THREADS",),
        resources=ResourceRequest(mpi_ranks=8, omp_threads=1),
        status="pending",
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
        staged_files=(
            StagedFile(
                source=tmp_path / "basis.dat",
                destination=tmp_path / "staged-basis.dat",
            ),
        ),
        stdout_path=tmp_path / "water.out",
        stderr_path=tmp_path / "water.err",
    )


def test_launch_record_round_trip_and_started_transition(tmp_path):
    db_path = tmp_path / "registry.db"
    pending = _pending(tmp_path)
    started = replace(
        pending,
        status="started",
        process_id=4242,
        updated_at=CREATED_AT + timedelta(seconds=1),
    )

    assert create_launch_record(pending, db_path) == pending
    assert load_launch_record(LAUNCH_ID, db_path) == pending
    assert update_launch_record(started, db_path) == started
    assert load_launch_record(LAUNCH_ID, db_path) == started


def test_launch_registry_rejects_duplicate_and_invalid_transition(tmp_path):
    db_path = tmp_path / "registry.db"
    pending = _pending(tmp_path)
    started = replace(
        pending,
        status="started",
        process_id=4242,
        updated_at=CREATED_AT + timedelta(seconds=1),
    )
    create_launch_record(pending, db_path)

    with pytest.raises(
        LaunchRecordConflict,
        match="already exists",
    ):
        create_launch_record(pending, db_path)

    update_launch_record(started, db_path)
    with pytest.raises(
        LaunchRecordConflict,
        match="invalid launch status transition 'started' to 'launch_failed'",
    ):
        update_launch_record(
            replace(
                started,
                status="launch_failed",
                process_id=None,
                updated_at=CREATED_AT + timedelta(seconds=2),
            ),
            db_path,
        )

    assert load_launch_record(LAUNCH_ID, db_path) == started


def test_launch_record_requires_canonical_ids_and_aware_times(tmp_path):
    with pytest.raises(
        ValueError,
        match="launch_id must be a canonical UUID string",
    ):
        replace(_pending(tmp_path), launch_id="not-a-uuid")

    with pytest.raises(
        ValueError,
        match="created_at must include a UTC offset",
    ):
        replace(
            _pending(tmp_path),
            created_at=datetime(2026, 7, 30, 15, 0),
        )


def test_registry_migrates_existing_launch_table_and_adds_links(tmp_path):
    db_path = tmp_path / "registry.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE execution_launches (
               launch_id TEXT PRIMARY KEY,
               instance_id TEXT NOT NULL,
               target_name TEXT NOT NULL
           )"""
    )
    conn.execute(
        "INSERT INTO execution_launches VALUES (?, ?, ?)",
        (LAUNCH_ID, INSTANCE_ID, "local"),
    )
    conn.commit()
    conn.close()

    migrated = connect_registry(db_path)
    try:
        columns = {
            row[1]
            for row in migrated.execute(
                "PRAGMA table_info(execution_launches)"
            ).fetchall()
        }
        staged_files_json = migrated.execute(
            "SELECT staged_files_json FROM execution_launches "
            "WHERE launch_id = ?",
            (LAUNCH_ID,),
        ).fetchone()[0]
        link_table = migrated.execute(
            """SELECT name FROM sqlite_master
               WHERE type = 'table' AND name = 'execution_run_links'"""
        ).fetchone()["name"]
    finally:
        migrated.close()

    assert "staged_files_json" in columns
    assert "stdin_sha256" in columns
    assert "stdin_size_bytes" in columns
    assert "return_code" in columns
    assert "elapsed_seconds" in columns
    assert staged_files_json == "[]"
    assert link_table == "execution_run_links"


def test_launch_registry_persists_synchronous_terminal_result(tmp_path):
    db_path = tmp_path / "registry.db"
    pending = replace(
        _pending(tmp_path),
        stdin_sha256="a" * 64,
        stdin_size_bytes=12,
        stdout_path=None,
        stderr_path=None,
    )
    completed = replace(
        pending,
        status="completed",
        return_code=0,
        elapsed_seconds=1.25,
        updated_at=CREATED_AT + timedelta(seconds=2),
    )

    create_launch_record(pending, db_path)
    update_launch_record(completed, db_path)

    assert load_launch_record(LAUNCH_ID, db_path) == completed
    with pytest.raises(
        LaunchRecordConflict,
        match="invalid launch status transition 'completed' to 'cancelled'",
    ):
        update_launch_record(
            replace(
                completed,
                status="cancelled",
                updated_at=CREATED_AT + timedelta(seconds=3),
            ),
            db_path,
        )


def test_run_registration_atomically_links_one_launch_to_one_run(tmp_path):
    db_path = tmp_path / "registry.db"
    create_launch_record(_pending(tmp_path), db_path)

    registered = register_run(
        "water",
        program="nwchem",
        run_uid=RUN_UID,
        launch_id=LAUNCH_ID,
        db_path=str(db_path),
    )
    run = get_run_summary(run_uid=RUN_UID, db_path=str(db_path))

    assert registered == {
        "run_id": 1,
        "run_uid": RUN_UID,
        "job_name": "water",
        "status": "submitted",
        "program": "nwchem",
    }
    assert load_execution_run_link(LAUNCH_ID, db_path) == (
        ExecutionRunLink(
            launch_id=LAUNCH_ID,
            run_uid=RUN_UID,
            linked_at=datetime.fromisoformat(run["submitted_at"]),
        )
    )
    conn = connect_registry(db_path)
    try:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()

    with pytest.raises(
        ValueError,
        match="is already linked to run",
    ):
        register_run(
            "duplicate",
            launch_id=LAUNCH_ID,
            db_path=str(db_path),
        )
    assert [item["run_uid"] for item in list_runs(
        db_path=str(db_path),
    )] == [RUN_UID]


def test_missing_launch_rolls_back_scientific_run_registration(tmp_path):
    db_path = tmp_path / "registry.db"

    with pytest.raises(
        ValueError,
        match=f"launch {LAUNCH_ID!r} is not registered",
    ):
        register_run(
            "orphan",
            run_uid=RUN_UID,
            launch_id=LAUNCH_ID,
            db_path=str(db_path),
        )

    assert list_runs(db_path=str(db_path)) == []

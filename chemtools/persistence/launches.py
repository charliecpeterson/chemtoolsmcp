"""SQLite persistence for permission-bound execution launch records."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sqlite3

from chemtools.core.execution import (
    ExecutionLaunchRecord,
    ExecutionRunLink,
    ResourceRequest,
    StagedFile,
)
from chemtools.persistence.sqlite import connect_registry


class UnknownLaunchRecordError(LookupError):
    pass


class LaunchRecordConflict(ValueError):
    pass


class UnknownExecutionRunLinkError(LookupError):
    pass


_STATUS_TRANSITIONS = {
    "pending": frozenset({
        "started",
        "completed",
        "failed",
        "timed_out",
        "submitted",
        "submitted_untracked",
        "submit_failed",
        "launch_failed",
    }),
    "started": frozenset({
        "completed",
        "failed",
        "cancelled",
        "cancel_failed",
    }),
    "submitted": frozenset({
        "completed",
        "failed",
        "timed_out",
        "cancelled",
        "cancel_failed",
    }),
    "cancel_failed": frozenset({
        "completed",
        "failed",
        "timed_out",
        "cancelled",
        "cancel_failed",
    }),
}


def _staged_files_json(staged_files: tuple[StagedFile, ...]) -> str:
    return json.dumps([
        {
            "source": str(staged_file.source),
            "destination": str(staged_file.destination),
            "mode": staged_file.mode,
            "required": staged_file.required,
        }
        for staged_file in staged_files
    ])


def _from_row(row: sqlite3.Row) -> ExecutionLaunchRecord:
    return ExecutionLaunchRecord(
        launch_id=row["launch_id"],
        instance_id=row["instance_id"],
        target=row["target_name"],
        executor=row["executor"],
        program=row["program"],
        working_directory=Path(row["working_directory"]),
        argv=tuple(json.loads(row["argv_json"])),
        environment_keys=tuple(
            json.loads(row["environment_keys_json"])
        ),
        staged_files=tuple(
            StagedFile(
                source=Path(staged_file["source"]),
                destination=Path(staged_file["destination"]),
                mode=staged_file["mode"],
                required=staged_file["required"],
            )
            for staged_file in json.loads(row["staged_files_json"])
        ),
        resources=ResourceRequest(
            nodes=row["nodes"],
            mpi_ranks=row["mpi_ranks"],
            omp_threads=row["omp_threads"],
            memory_mb_per_node=row["memory_mb_per_node"],
            walltime=row["walltime"],
            partition=row["partition_name"],
            account=row["account_name"],
        ),
        status=row["status"],
        stdout_path=(
            Path(row["stdout_path"])
            if row["stdout_path"] is not None
            else None
        ),
        stderr_path=(
            Path(row["stderr_path"])
            if row["stderr_path"] is not None
            else None
        ),
        script_path=(
            Path(row["script_path"])
            if row["script_path"] is not None
            else None
        ),
        process_id=row["process_id"],
        job_id=row["job_id"],
        stdin_sha256=row["stdin_sha256"],
        stdin_size_bytes=row["stdin_size_bytes"],
        return_code=row["return_code"],
        elapsed_seconds=row["elapsed_seconds"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        error=row["error"],
    )


def create_launch_record(
    record: ExecutionLaunchRecord,
    db_path: str | Path | None = None,
) -> ExecutionLaunchRecord:
    if record.status != "pending":
        raise ValueError("new launch records must have pending status")
    conn = connect_registry(db_path)
    try:
        with conn:
            try:
                conn.execute(
                    """INSERT INTO execution_launches (
                           launch_id, instance_id, target_name, executor,
                           program, working_directory, argv_json,
                           environment_keys_json, staged_files_json,
                           nodes, mpi_ranks,
                           omp_threads, memory_mb_per_node, walltime,
                           partition_name, account_name, stdout_path,
                           stderr_path, script_path, status, process_id,
                           job_id, stdin_sha256, stdin_size_bytes,
                           return_code, elapsed_seconds, created_at,
                           updated_at, error
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                                 ?, ?, ?, ?, ?, ?, ?, ?,
                                 ?, ?, ?, ?, ?, ?, ?, ?,
                                 ?, ?, ?, ?)""",
                    (
                        record.launch_id,
                        record.instance_id,
                        record.target,
                        record.executor,
                        record.program,
                        str(record.working_directory),
                        json.dumps(record.argv),
                        json.dumps(record.environment_keys),
                        _staged_files_json(record.staged_files),
                        record.resources.nodes,
                        record.resources.mpi_ranks,
                        record.resources.omp_threads,
                        record.resources.memory_mb_per_node,
                        record.resources.walltime,
                        record.resources.partition,
                        record.resources.account,
                        (
                            str(record.stdout_path)
                            if record.stdout_path is not None
                            else None
                        ),
                        (
                            str(record.stderr_path)
                            if record.stderr_path is not None
                            else None
                        ),
                        (
                            str(record.script_path)
                            if record.script_path is not None
                            else None
                        ),
                        record.status,
                        record.process_id,
                        record.job_id,
                        record.stdin_sha256,
                        record.stdin_size_bytes,
                        record.return_code,
                        record.elapsed_seconds,
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                        record.error,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise LaunchRecordConflict(
                    f"launch record {record.launch_id!r} already exists"
                ) from exc
    finally:
        conn.close()
    return record


def load_launch_record(
    launch_id: str,
    db_path: str | Path | None = None,
) -> ExecutionLaunchRecord:
    conn = connect_registry(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM execution_launches WHERE launch_id = ?",
            (launch_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise UnknownLaunchRecordError(
            f"launch record {launch_id!r} does not exist"
        )
    return _from_row(row)


def load_execution_run_link(
    launch_id: str,
    db_path: str | Path | None = None,
) -> ExecutionRunLink:
    conn = connect_registry(db_path)
    try:
        row = conn.execute(
            """SELECT launch_id, run_uid, linked_at
               FROM execution_run_links WHERE launch_id = ?""",
            (launch_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise UnknownExecutionRunLinkError(
            f"launch {launch_id!r} is not linked to a scientific run"
        )
    return ExecutionRunLink(
        launch_id=row["launch_id"],
        run_uid=row["run_uid"],
        linked_at=datetime.fromisoformat(row["linked_at"]),
    )


def update_launch_record(
    record: ExecutionLaunchRecord,
    db_path: str | Path | None = None,
) -> ExecutionLaunchRecord:
    conn = connect_registry(db_path)
    try:
        with conn:
            row = conn.execute(
                "SELECT * FROM execution_launches WHERE launch_id = ?",
                (record.launch_id,),
            ).fetchone()
            if row is None:
                raise UnknownLaunchRecordError(
                    f"launch record {record.launch_id!r} does not exist"
                )
            current = _from_row(row)
            immutable_current = (
                current.instance_id,
                current.target,
                current.executor,
                current.program,
                current.working_directory,
                current.argv,
                current.environment_keys,
                current.staged_files,
                current.resources,
                current.stdout_path,
                current.stderr_path,
                current.script_path,
                current.stdin_sha256,
                current.stdin_size_bytes,
                current.created_at,
            )
            immutable_incoming = (
                record.instance_id,
                record.target,
                record.executor,
                record.program,
                record.working_directory,
                record.argv,
                record.environment_keys,
                record.staged_files,
                record.resources,
                record.stdout_path,
                record.stderr_path,
                record.script_path,
                record.stdin_sha256,
                record.stdin_size_bytes,
                record.created_at,
            )
            if immutable_incoming != immutable_current:
                raise LaunchRecordConflict(
                    "launch record identity fields cannot change"
                )
            allowed = _STATUS_TRANSITIONS.get(
                current.status,
                frozenset(),
            )
            if record.status not in allowed:
                raise LaunchRecordConflict(
                    f"invalid launch status transition "
                    f"{current.status!r} to {record.status!r}"
                )
            if record.updated_at < current.updated_at:
                raise LaunchRecordConflict(
                    "launch record updated_at cannot move backward"
                )
            conn.execute(
                """UPDATE execution_launches
                   SET status = ?, process_id = ?, job_id = ?,
                       return_code = ?, elapsed_seconds = ?,
                       updated_at = ?, error = ?
                   WHERE launch_id = ?""",
                (
                    record.status,
                    record.process_id,
                    record.job_id,
                    record.return_code,
                    record.elapsed_seconds,
                    record.updated_at.isoformat(),
                    record.error,
                    record.launch_id,
                ),
            )
    finally:
        conn.close()
    return record


__all__ = [
    "LaunchRecordConflict",
    "UnknownExecutionRunLinkError",
    "UnknownLaunchRecordError",
    "create_launch_record",
    "load_execution_run_link",
    "load_launch_record",
    "update_launch_record",
]

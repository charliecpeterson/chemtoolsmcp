"""SQLite persistence for immutable artifact and provenance metadata.

Artifact bytes remain in their original filesystems. This store records the
versioned core models and their ordered membership in a registered run.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

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
from chemtools.persistence.sqlite import connect_registry


class UnknownRunUidError(LookupError):
    pass


class ArtifactPersistenceConflict(ValueError):
    pass


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _step_ref(
    run_uid: str | None,
    step_id: str | None,
) -> StepRef | None:
    if run_uid is None:
        return None
    return StepRef(run_uid=run_uid, step_id=step_id)


def _roles(
    conn: sqlite3.Connection,
    table: str,
    id_column: str,
    identifier: str,
) -> frozenset[ArtifactRole]:
    rows = conn.execute(
        f"SELECT role FROM {table} WHERE {id_column} = ? ORDER BY role",
        (identifier,),
    ).fetchall()
    return frozenset(ArtifactRole(row["role"]) for row in rows)


def _artifact_from_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=row["artifact_id"],
        roles=_roles(
            conn,
            "artifact_roles",
            "artifact_id",
            row["artifact_id"],
        ),
        kind=row["kind"],
        producing_step=_step_ref(
            row["producing_run_uid"],
            row["producing_step_id"],
        ),
        metadata=json.loads(row["metadata_json"]),
    )


def _record_artifact(
    conn: sqlite3.Connection,
    artifact: ArtifactRef,
) -> None:
    existing = conn.execute(
        "SELECT * FROM artifacts WHERE artifact_id = ?",
        (artifact.artifact_id,),
    ).fetchone()
    if existing is not None:
        if _artifact_from_row(conn, existing) != artifact:
            raise ArtifactPersistenceConflict(
                f"artifact {artifact.artifact_id!r} already has different metadata"
            )
        return

    step = artifact.producing_step
    conn.execute(
        """INSERT INTO artifacts (
               artifact_id, kind, producing_run_uid, producing_step_id,
               metadata_json
           ) VALUES (?, ?, ?, ?, ?)""",
        (
            artifact.artifact_id,
            artifact.kind,
            step.run_uid if step is not None else None,
            step.step_id if step is not None else None,
            _json(artifact.to_dict()["metadata"]),
        ),
    )
    conn.executemany(
        "INSERT INTO artifact_roles (artifact_id, role) VALUES (?, ?)",
        (
            (artifact.artifact_id, role.value)
            for role in sorted(artifact.roles, key=lambda item: item.value)
        ),
    )


def _observation_from_row(row: sqlite3.Row) -> ArtifactObservation:
    return ArtifactObservation.from_dict({
        "observation_id": row["observation_id"],
        "artifact_id": row["artifact_id"],
        "observed_at": row["observed_at"],
        "location": {
            "path": row["path"],
            "entry_type": row["entry_type"],
            "root_name": row["root_name"],
            "relative_path": row["relative_path"],
        },
        "exists": bool(row["exists_flag"]),
        "size_bytes": row["size_bytes"],
        "modified_ns": row["modified_ns"],
        "sha256": row["sha256"],
        "hash_status": row["hash_status"],
        "directory_manifest_schema": row["directory_manifest_schema"],
        "directory_manifest_sha256": row[
            "directory_manifest_sha256"
        ],
    })


def _record_observation(
    conn: sqlite3.Connection,
    observation: ArtifactObservation,
) -> None:
    existing = conn.execute(
        "SELECT * FROM artifact_observations WHERE observation_id = ?",
        (observation.observation_id,),
    ).fetchone()
    if existing is not None:
        if _observation_from_row(existing) != observation:
            raise ArtifactPersistenceConflict(
                f"observation {observation.observation_id!r} "
                "already has different metadata"
            )
        return

    location = observation.location
    conn.execute(
        """INSERT INTO artifact_observations (
               observation_id, artifact_id, observed_at, path, entry_type,
               root_name, relative_path, exists_flag, size_bytes, modified_ns,
               sha256, hash_status, directory_manifest_schema,
               directory_manifest_sha256
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            observation.observation_id,
            observation.artifact_id,
            observation.observed_at.isoformat(),
            str(location.path),
            location.entry_type,
            location.root_name,
            (
                str(location.relative_path)
                if location.relative_path is not None
                else None
            ),
            int(observation.exists),
            observation.size_bytes,
            observation.modified_ns,
            observation.sha256,
            observation.hash_status,
            observation.directory_manifest_schema,
            observation.directory_manifest_sha256,
        ),
    )


def _expectation_from_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> ExpectedArtifact:
    return ExpectedArtifact(
        expectation_id=row["expectation_id"],
        roles=_roles(
            conn,
            "artifact_expectation_roles",
            "expectation_id",
            row["expectation_id"],
        ),
        kind=row["kind"],
        location=ArtifactLocation(
            path=Path(row["path"]),
            entry_type=row["entry_type"],
            root_name=row["root_name"],
            relative_path=(
                Path(row["relative_path"])
                if row["relative_path"] is not None
                else None
            ),
        ),
        required=bool(row["required_flag"]),
        producing_step=_step_ref(
            row["producing_run_uid"],
            row["producing_step_id"],
        ),
    )


def _record_expectation(
    conn: sqlite3.Connection,
    run_uid: str,
    position: int,
    expectation: ExpectedArtifact,
) -> None:
    existing = conn.execute(
        "SELECT * FROM artifact_expectations WHERE expectation_id = ?",
        (expectation.expectation_id,),
    ).fetchone()
    if existing is not None:
        if (
            existing["run_uid"] != run_uid
            or existing["position"] != position
            or _expectation_from_row(conn, existing) != expectation
        ):
            raise ArtifactPersistenceConflict(
                f"expectation {expectation.expectation_id!r} "
                "already has different metadata or ordering"
            )
        return

    location = expectation.location
    step = expectation.producing_step
    conn.execute(
        """INSERT INTO artifact_expectations (
               expectation_id, run_uid, position, kind, path, entry_type,
               root_name, relative_path, required_flag, producing_run_uid,
               producing_step_id
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            expectation.expectation_id,
            run_uid,
            position,
            expectation.kind,
            str(location.path),
            location.entry_type,
            location.root_name,
            (
                str(location.relative_path)
                if location.relative_path is not None
                else None
            ),
            int(expectation.required),
            step.run_uid if step is not None else None,
            step.step_id if step is not None else None,
        ),
    )
    conn.executemany(
        """INSERT INTO artifact_expectation_roles (expectation_id, role)
           VALUES (?, ?)""",
        (
            (expectation.expectation_id, role.value)
            for role in sorted(expectation.roles, key=lambda item: item.value)
        ),
    )


def _event_from_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> ProvenanceRecord:
    snapshots = conn.execute(
        """SELECT direction, artifact_id, observation_id
           FROM provenance_snapshots
           WHERE event_id = ?
           ORDER BY direction, position""",
        (row["event_id"],),
    ).fetchall()

    def for_direction(direction: str) -> tuple[ArtifactSnapshotRef, ...]:
        return tuple(
            ArtifactSnapshotRef(
                artifact_id=snapshot["artifact_id"],
                observation_id=snapshot["observation_id"],
            )
            for snapshot in snapshots
            if snapshot["direction"] == direction
        )

    return ProvenanceRecord(
        event_id=row["event_id"],
        event_type=row["event_type"],
        occurred_at=datetime.fromisoformat(row["occurred_at"]),
        actor=ProducerIdentity(
            producer_type=row["actor_type"],
            name=row["actor_name"],
            version=row["actor_version"],
            commit=row["actor_commit"],
        ),
        inputs=for_direction("input"),
        outputs=for_direction("output"),
        evidence=row["evidence"],
        run_uid=row["event_run_uid"],
        step_id=row["step_id"],
        parameters=json.loads(row["parameters_json"]),
    )


def _record_event(
    conn: sqlite3.Connection,
    event: ProvenanceRecord,
) -> None:
    existing = conn.execute(
        "SELECT * FROM provenance_events WHERE event_id = ?",
        (event.event_id,),
    ).fetchone()
    if existing is not None:
        if _event_from_row(conn, existing) != event:
            raise ArtifactPersistenceConflict(
                f"provenance event {event.event_id!r} "
                "already has different metadata"
            )
        return

    actor = event.actor
    conn.execute(
        """INSERT INTO provenance_events (
               event_id, event_type, occurred_at, actor_type, actor_name,
               actor_version, actor_commit, event_run_uid, step_id, evidence,
               parameters_json
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            event.event_id,
            event.event_type,
            event.occurred_at.isoformat(),
            actor.producer_type,
            actor.name,
            actor.version,
            actor.commit,
            event.run_uid,
            event.step_id,
            event.evidence,
            _json(event.to_dict()["parameters"]),
        ),
    )
    for direction, snapshots in (
        ("input", event.inputs),
        ("output", event.outputs),
    ):
        conn.executemany(
            """INSERT INTO provenance_snapshots (
                   event_id, direction, position, artifact_id, observation_id
               ) VALUES (?, ?, ?, ?, ?)""",
            (
                (
                    event.event_id,
                    direction,
                    position,
                    snapshot.artifact_id,
                    snapshot.observation_id,
                )
                for position, snapshot in enumerate(snapshots)
            ),
        )


def _record_memberships(
    conn: sqlite3.Connection,
    table: str,
    id_column: str,
    run_uid: str,
    identifiers: Iterable[str],
) -> None:
    ordered_identifiers = tuple(identifiers)
    existing = _load_ordered(conn, table, id_column, run_uid)
    if existing != ordered_identifiers[:len(existing)]:
        raise ArtifactPersistenceConflict(
            f"{table} membership must be an append-only extension"
        )

    for position, identifier in enumerate(ordered_identifiers):
        by_identifier = conn.execute(
            f"""SELECT position FROM {table}
                WHERE run_uid = ? AND {id_column} = ?""",
            (run_uid, identifier),
        ).fetchone()
        if by_identifier is not None:
            if by_identifier["position"] != position:
                raise ArtifactPersistenceConflict(
                    f"{table} membership for {identifier!r} changed position"
                )
            continue

        by_position = conn.execute(
            f"""SELECT {id_column} FROM {table}
                WHERE run_uid = ? AND position = ?""",
            (run_uid, position),
        ).fetchone()
        if by_position is not None:
            raise ArtifactPersistenceConflict(
                f"{table} position {position} already belongs to "
                f"{by_position[id_column]!r}"
            )
        conn.execute(
            f"""INSERT INTO {table} (run_uid, position, {id_column})
                VALUES (?, ?, ?)""",
            (run_uid, position, identifier),
        )


def _require_run(conn: sqlite3.Connection, run_uid: str) -> None:
    row = conn.execute(
        "SELECT 1 FROM runs WHERE run_uid = ?",
        (run_uid,),
    ).fetchone()
    if row is None:
        raise UnknownRunUidError(f"run {run_uid!r} is not registered")


def record_run_artifacts(
    collection: RunArtifacts,
    db_path: str | Path | None = None,
) -> dict[str, int | str]:
    """Record a complete or append-only extension of a run collection."""
    conn = connect_registry(db_path)
    try:
        with conn:
            _require_run(conn, collection.run_uid)

            for artifact in collection.artifacts:
                _record_artifact(conn, artifact)
            _record_memberships(
                conn,
                "run_artifacts",
                "artifact_id",
                collection.run_uid,
                (
                    artifact.artifact_id
                    for artifact in collection.artifacts
                ),
            )

            for observation in collection.observations:
                _record_observation(conn, observation)
            _record_memberships(
                conn,
                "run_artifact_observations",
                "observation_id",
                collection.run_uid,
                (
                    observation.observation_id
                    for observation in collection.observations
                ),
            )

            existing_expectations = tuple(
                row["expectation_id"]
                for row in conn.execute(
                    """SELECT expectation_id
                       FROM artifact_expectations
                       WHERE run_uid = ? ORDER BY position""",
                    (collection.run_uid,),
                ).fetchall()
            )
            incoming_expectations = tuple(
                item.expectation_id
                for item in collection.expectations
            )
            if (
                existing_expectations
                != incoming_expectations[:len(existing_expectations)]
            ):
                raise ArtifactPersistenceConflict(
                    "artifact_expectations membership must be "
                    "an append-only extension"
                )

            for position, expectation in enumerate(collection.expectations):
                _record_expectation(
                    conn,
                    collection.run_uid,
                    position,
                    expectation,
                )

            for event in collection.provenance:
                _record_event(conn, event)
            _record_memberships(
                conn,
                "run_provenance_events",
                "event_id",
                collection.run_uid,
                (event.event_id for event in collection.provenance),
            )
    finally:
        conn.close()

    return {
        "run_uid": collection.run_uid,
        "artifacts": len(collection.artifacts),
        "observations": len(collection.observations),
        "expectations": len(collection.expectations),
        "provenance_events": len(collection.provenance),
    }


def _load_ordered(
    conn: sqlite3.Connection,
    table: str,
    id_column: str,
    run_uid: str,
) -> tuple[str, ...]:
    rows = conn.execute(
        f"""SELECT {id_column} FROM {table}
            WHERE run_uid = ? ORDER BY position""",
        (run_uid,),
    ).fetchall()
    return tuple(row[id_column] for row in rows)


def load_run_artifacts(
    run_uid: str,
    db_path: str | Path | None = None,
) -> RunArtifacts | None:
    conn = connect_registry(db_path)
    try:
        if conn.execute(
            "SELECT 1 FROM runs WHERE run_uid = ?",
            (run_uid,),
        ).fetchone() is None:
            return None

        artifacts = tuple(
            _artifact_from_row(
                conn,
                conn.execute(
                    "SELECT * FROM artifacts WHERE artifact_id = ?",
                    (artifact_id,),
                ).fetchone(),
            )
            for artifact_id in _load_ordered(
                conn,
                "run_artifacts",
                "artifact_id",
                run_uid,
            )
        )
        observations = tuple(
            _observation_from_row(
                conn.execute(
                    """SELECT * FROM artifact_observations
                       WHERE observation_id = ?""",
                    (observation_id,),
                ).fetchone()
            )
            for observation_id in _load_ordered(
                conn,
                "run_artifact_observations",
                "observation_id",
                run_uid,
            )
        )
        expectation_rows = conn.execute(
            """SELECT * FROM artifact_expectations
               WHERE run_uid = ? ORDER BY position""",
            (run_uid,),
        ).fetchall()
        events = tuple(
            _event_from_row(
                conn,
                conn.execute(
                    "SELECT * FROM provenance_events WHERE event_id = ?",
                    (event_id,),
                ).fetchone(),
            )
            for event_id in _load_ordered(
                conn,
                "run_provenance_events",
                "event_id",
                run_uid,
            )
        )
        return RunArtifacts(
            run_uid=run_uid,
            artifacts=artifacts,
            observations=observations,
            expectations=tuple(
                _expectation_from_row(conn, row)
                for row in expectation_rows
            ),
            provenance=events,
        )
    finally:
        conn.close()


__all__ = [
    "ArtifactPersistenceConflict",
    "UnknownRunUidError",
    "load_run_artifacts",
    "record_run_artifacts",
]

"""Shared SQLite connection and schema migrations for Chemtools persistence.

Run, campaign, workflow, artifact, and provenance stores use this module so
all callers see the same schema and foreign-key settings.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from uuid import uuid4


_BASE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS campaigns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    description TEXT,
    created_at  TEXT NOT NULL,
    tags        TEXT
);

CREATE TABLE IF NOT EXISTS workflows (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER REFERENCES campaigns(id),
    name        TEXT NOT NULL,
    protocol    TEXT,
    state       TEXT NOT NULL DEFAULT 'pending',
    steps_json  TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_uid             TEXT NOT NULL,
    program             TEXT,
    job_name            TEXT NOT NULL,
    input_file          TEXT,
    output_file         TEXT,
    profile             TEXT,
    method              TEXT,
    functional          TEXT,
    basis               TEXT,
    n_atoms             INTEGER,
    elements            TEXT,
    charge              INTEGER,
    multiplicity        INTEGER,
    status              TEXT NOT NULL DEFAULT 'pending',
    submitted_at        TEXT,
    completed_at        TEXT,
    walltime_used_sec   REAL,
    energy_hartree      REAL,
    h_hartree           REAL,
    g_hartree           REAL,
    imaginary_modes     INTEGER,
    mpi_ranks           INTEGER,
    node_memory_mb      INTEGER,
    cpu_arch            TEXT,
    sec_per_gradient    REAL,
    parent_run_id       INTEGER REFERENCES runs(id),
    campaign_id         INTEGER REFERENCES campaigns(id),
    workflow_id         INTEGER REFERENCES workflows(id),
    workflow_step_id    TEXT,
    tags                TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_campaign ON runs(campaign_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_workflow ON runs(workflow_id);
"""


_ARTIFACT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id          TEXT PRIMARY KEY,
    kind                 TEXT NOT NULL,
    producing_run_uid    TEXT,
    producing_step_id    TEXT,
    metadata_json        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_roles (
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    PRIMARY KEY (artifact_id, role)
);

CREATE TABLE IF NOT EXISTS run_artifacts (
    run_uid     TEXT NOT NULL REFERENCES runs(run_uid) ON DELETE CASCADE,
    position    INTEGER NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    PRIMARY KEY (run_uid, artifact_id),
    UNIQUE (run_uid, position)
);

CREATE TABLE IF NOT EXISTS artifact_observations (
    observation_id             TEXT PRIMARY KEY,
    artifact_id                TEXT NOT NULL REFERENCES artifacts(artifact_id),
    observed_at                TEXT NOT NULL,
    path                       TEXT NOT NULL,
    entry_type                 TEXT NOT NULL,
    root_name                  TEXT,
    relative_path              TEXT,
    exists_flag                INTEGER NOT NULL CHECK (exists_flag IN (0, 1)),
    size_bytes                 INTEGER,
    modified_ns                INTEGER,
    sha256                     TEXT,
    hash_status                TEXT NOT NULL,
    directory_manifest_schema  TEXT,
    directory_manifest_sha256  TEXT,
    UNIQUE (artifact_id, observation_id)
);

CREATE TABLE IF NOT EXISTS run_artifact_observations (
    run_uid       TEXT NOT NULL REFERENCES runs(run_uid) ON DELETE CASCADE,
    position      INTEGER NOT NULL,
    observation_id TEXT NOT NULL REFERENCES artifact_observations(observation_id),
    PRIMARY KEY (run_uid, observation_id),
    UNIQUE (run_uid, position)
);

CREATE TABLE IF NOT EXISTS artifact_expectations (
    expectation_id          TEXT PRIMARY KEY,
    run_uid                 TEXT NOT NULL REFERENCES runs(run_uid) ON DELETE CASCADE,
    position                INTEGER NOT NULL,
    kind                    TEXT NOT NULL,
    path                    TEXT NOT NULL,
    entry_type              TEXT NOT NULL,
    root_name               TEXT,
    relative_path           TEXT,
    required_flag           INTEGER NOT NULL CHECK (required_flag IN (0, 1)),
    producing_run_uid       TEXT,
    producing_step_id       TEXT,
    UNIQUE (run_uid, position)
);

CREATE TABLE IF NOT EXISTS artifact_expectation_roles (
    expectation_id TEXT NOT NULL
        REFERENCES artifact_expectations(expectation_id) ON DELETE CASCADE,
    role           TEXT NOT NULL,
    PRIMARY KEY (expectation_id, role)
);

CREATE TABLE IF NOT EXISTS provenance_events (
    event_id          TEXT PRIMARY KEY,
    event_type        TEXT NOT NULL,
    occurred_at       TEXT NOT NULL,
    actor_type        TEXT NOT NULL,
    actor_name        TEXT NOT NULL,
    actor_version     TEXT,
    actor_commit      TEXT,
    event_run_uid     TEXT,
    step_id           TEXT,
    evidence          TEXT NOT NULL,
    parameters_json   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provenance_snapshots (
    event_id       TEXT NOT NULL
        REFERENCES provenance_events(event_id) ON DELETE CASCADE,
    direction      TEXT NOT NULL CHECK (direction IN ('input', 'output')),
    position       INTEGER NOT NULL,
    artifact_id    TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    PRIMARY KEY (event_id, direction, position),
    FOREIGN KEY (artifact_id, observation_id)
        REFERENCES artifact_observations(artifact_id, observation_id)
);

CREATE TABLE IF NOT EXISTS run_provenance_events (
    run_uid  TEXT NOT NULL REFERENCES runs(run_uid) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    event_id TEXT NOT NULL REFERENCES provenance_events(event_id),
    PRIMARY KEY (run_uid, event_id),
    UNIQUE (run_uid, position)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind);
CREATE INDEX IF NOT EXISTS idx_observations_artifact
    ON artifact_observations(artifact_id);
CREATE INDEX IF NOT EXISTS idx_observations_path
    ON artifact_observations(path);
CREATE INDEX IF NOT EXISTS idx_expectations_run
    ON artifact_expectations(run_uid);
"""


_EXECUTION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS execution_launches (
    launch_id          TEXT PRIMARY KEY,
    instance_id        TEXT NOT NULL,
    target_name        TEXT NOT NULL,
    executor           TEXT NOT NULL,
    program            TEXT NOT NULL,
    working_directory  TEXT NOT NULL,
    argv_json          TEXT NOT NULL,
    environment_keys_json TEXT NOT NULL,
    staged_files_json  TEXT NOT NULL DEFAULT '[]',
    nodes              INTEGER NOT NULL,
    mpi_ranks          INTEGER NOT NULL,
    omp_threads        INTEGER NOT NULL,
    memory_mb_per_node INTEGER,
    walltime           TEXT,
    partition_name     TEXT,
    account_name       TEXT,
    stdout_path        TEXT,
    stderr_path        TEXT,
    script_path        TEXT,
    status             TEXT NOT NULL,
    process_id         INTEGER,
    job_id             TEXT,
    stdin_sha256       TEXT,
    stdin_size_bytes   INTEGER,
    return_code        INTEGER,
    elapsed_seconds    REAL,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    error              TEXT
);

CREATE INDEX IF NOT EXISTS idx_execution_launches_instance
    ON execution_launches(instance_id);
CREATE INDEX IF NOT EXISTS idx_execution_launches_target
    ON execution_launches(target_name);

CREATE TABLE IF NOT EXISTS execution_run_links (
    launch_id TEXT PRIMARY KEY
        REFERENCES execution_launches(launch_id) ON DELETE CASCADE,
    run_uid   TEXT NOT NULL UNIQUE
        REFERENCES runs(run_uid) ON DELETE CASCADE,
    linked_at TEXT NOT NULL
);
"""


def _default_db_path() -> Path:
    configured = os.environ.get("CHEMTOOLS_REGISTRY_DB")
    if configured:
        return Path(configured)
    chemtools_dir = Path.home() / ".chemtools"
    chemtools_dir.mkdir(parents=True, exist_ok=True)
    return chemtools_dir / "registry.db"


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _migrate_runs(conn: sqlite3.Connection) -> None:
    columns = _columns(conn, "runs")
    if "program" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN program TEXT")
    if "run_uid" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN run_uid TEXT")

    missing_uids = conn.execute(
        "SELECT id FROM runs WHERE run_uid IS NULL OR TRIM(run_uid) = ''"
    ).fetchall()
    for row in missing_uids:
        conn.execute(
            "UPDATE runs SET run_uid = ? WHERE id = ?",
            (str(uuid4()), row[0]),
        )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_program ON runs(program)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_uid ON runs(run_uid)"
    )


def _migrate_execution_launches(conn: sqlite3.Connection) -> None:
    columns = _columns(conn, "execution_launches")
    if "staged_files_json" not in columns:
        conn.execute(
            "ALTER TABLE execution_launches "
            "ADD COLUMN staged_files_json TEXT NOT NULL DEFAULT '[]'"
        )
    additions = {
        "stdin_sha256": "TEXT",
        "stdin_size_bytes": "INTEGER",
        "return_code": "INTEGER",
        "elapsed_seconds": "REAL",
    }
    for name, sql_type in additions.items():
        if name not in columns:
            conn.execute(
                f"ALTER TABLE execution_launches "
                f"ADD COLUMN {name} {sql_type}"
            )


def ensure_registry_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_BASE_SCHEMA_SQL)
    _migrate_runs(conn)
    conn.executescript(_ARTIFACT_SCHEMA_SQL)
    conn.executescript(_EXECUTION_SCHEMA_SQL)
    _migrate_execution_launches(conn)
    conn.commit()


def connect_registry(
    db_path: str | Path | None = None,
) -> sqlite3.Connection:
    path = Path(db_path) if db_path else _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        ensure_registry_schema(conn)
    except Exception:
        conn.close()
        raise
    return conn


__all__ = ["connect_registry", "ensure_registry_schema"]

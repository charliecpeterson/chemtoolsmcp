"""SQLite persistence for scientific run records and launch links."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
from typing import Any
from uuid import UUID, uuid4

from chemtools.core.registry_db import connect_registry


def _canonical_uuid(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    try:
        normalized = str(UUID(value))
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be a canonical UUID string"
        ) from exc
    if normalized != value:
        raise ValueError(
            f"{field_name} must be a canonical UUID string"
        )
    return normalized


def _run_uid(value: str | None) -> str:
    if value is None:
        return str(uuid4())
    return _canonical_uuid(value, "run_uid")


def register_run(
    job_name: str,
    input_file: str | None = None,
    output_file: str | None = None,
    profile: str | None = None,
    method: str | None = None,
    functional: str | None = None,
    basis: str | None = None,
    n_atoms: int | None = None,
    elements: list[str] | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    mpi_ranks: int | None = None,
    node_memory_mb: int | None = None,
    cpu_arch: str | None = None,
    campaign_id: int | None = None,
    workflow_id: int | None = None,
    workflow_step_id: str | None = None,
    parent_run_id: int | None = None,
    tags: dict[str, Any] | None = None,
    program: str | None = None,
    run_uid: str | None = None,
    db_path: str | None = None,
    launch_id: str | None = None,
) -> dict[str, Any]:
    """Register a new run and return its local and portable IDs.

    ``program`` (since Phase 4b) tags the run with which QC program produced
    it (``'nwchem'``, ``'molcas'``, ``'dirac'``, ``'grasp'``, ...). Pre-Phase-4b runs
    have NULL here; current NWChem registration paths pass ``'nwchem'``
    explicitly.

    ``launch_id`` atomically links a new scientific run to one typed
    execution launch. It is reserved for launch application services; manual
    registry calls normally leave it unset.
    """
    portable_id = _run_uid(run_uid)
    linked_launch_id = (
        _canonical_uuid(launch_id, "launch_id")
        if launch_id is not None
        else None
    )
    now = datetime.now(timezone.utc).isoformat()
    insert_sql = """INSERT INTO runs (
                run_uid, program, job_name, input_file, output_file, profile,
                method, functional, basis, n_atoms, elements,
                charge, multiplicity, status, submitted_at,
                mpi_ranks, node_memory_mb, cpu_arch,
                campaign_id, workflow_id, workflow_step_id,
                parent_run_id, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'submitted', ?,
                      ?, ?, ?, ?, ?, ?, ?, ?)"""
    insert_params = (
        portable_id, program, job_name, input_file, output_file, profile,
        method, functional, basis, n_atoms,
        json.dumps(elements) if elements else None,
        charge, multiplicity, now,
        mpi_ranks, node_memory_mb, cpu_arch,
        campaign_id, workflow_id, workflow_step_id,
        parent_run_id,
        json.dumps(tags) if tags else None,
    )
    conn = connect_registry(db_path)
    try:
        if linked_launch_id is not None:
            launch = conn.execute(
                "SELECT 1 FROM execution_launches WHERE launch_id = ?",
                (linked_launch_id,),
            ).fetchone()
            if launch is None:
                raise ValueError(
                    f"launch {linked_launch_id!r} is not registered"
                )
            existing_link = conn.execute(
                """SELECT run_uid FROM execution_run_links
                   WHERE launch_id = ?""",
                (linked_launch_id,),
            ).fetchone()
            if existing_link is not None:
                raise ValueError(
                    f"launch {linked_launch_id!r} is already linked to "
                    f"run {existing_link['run_uid']!r}"
                )
        try:
            cur = conn.execute(insert_sql, insert_params)
        except sqlite3.OperationalError as exc:
            # A DB created by an older schema (or a connection that predates a
            # migration) can be missing a newer column. Re-run the idempotent
            # schema/migration and retry once before surfacing the error.
            if "no such column" not in str(exc).lower():
                raise
            from chemtools.core.registry_db import ensure_registry_schema

            ensure_registry_schema(conn)
            cur = conn.execute(insert_sql, insert_params)
        if linked_launch_id is not None:
            conn.execute(
                """INSERT INTO execution_run_links (
                       launch_id, run_uid, linked_at
                   ) VALUES (?, ?, ?)""",
                (linked_launch_id, portable_id, now),
            )
        conn.commit()
        run_id = cur.lastrowid
        return {
            "run_id": run_id,
            "run_uid": portable_id,
            "job_name": job_name,
            "status": "submitted",
            "program": program,
        }
    finally:
        conn.close()


def update_run_status(
    run_id: int,
    status: str,
    energy_hartree: float | None = None,
    h_hartree: float | None = None,
    g_hartree: float | None = None,
    imaginary_modes: int | None = None,
    walltime_used_sec: float | None = None,
    sec_per_gradient: float | None = None,
    output_file: str | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Update a run's status and optionally its results."""
    conn = connect_registry(db_path)
    try:
        sets: list[str] = ["status = ?"]
        vals: list[Any] = [status]

        if status in ("completed", "failed", "timelimited", "oom", "cancelled"):
            sets.append("completed_at = ?")
            vals.append(datetime.now(timezone.utc).isoformat())
        if energy_hartree is not None:
            sets.append("energy_hartree = ?")
            vals.append(energy_hartree)
        if h_hartree is not None:
            sets.append("h_hartree = ?")
            vals.append(h_hartree)
        if g_hartree is not None:
            sets.append("g_hartree = ?")
            vals.append(g_hartree)
        if imaginary_modes is not None:
            sets.append("imaginary_modes = ?")
            vals.append(imaginary_modes)
        if walltime_used_sec is not None:
            sets.append("walltime_used_sec = ?")
            vals.append(walltime_used_sec)
        if sec_per_gradient is not None:
            sets.append("sec_per_gradient = ?")
            vals.append(sec_per_gradient)
        if output_file is not None:
            sets.append("output_file = ?")
            vals.append(output_file)

        vals.append(run_id)
        conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()
        return {"run_id": run_id, "status": status}
    finally:
        conn.close()


def list_runs(
    campaign_id: int | None = None,
    workflow_id: int | None = None,
    status: str | None = None,
    method: str | None = None,
    program: str | None = None,
    limit: int = 50,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    """List runs, optionally filtered by campaign, workflow, status, method, or program."""
    conn = connect_registry(db_path)
    try:
        wheres: list[str] = []
        vals: list[Any] = []
        if campaign_id is not None:
            wheres.append("campaign_id = ?")
            vals.append(campaign_id)
        if workflow_id is not None:
            wheres.append("workflow_id = ?")
            vals.append(workflow_id)
        if status is not None:
            wheres.append("status = ?")
            vals.append(status)
        if method is not None:
            wheres.append("UPPER(method) = ?")
            vals.append(method.upper())
        if program is not None:
            wheres.append("LOWER(program) = ?")
            vals.append(program.lower())

        where_clause = (" WHERE " + " AND ".join(wheres)) if wheres else ""
        vals.append(limit)
        rows = conn.execute(
            f"SELECT * FROM runs{where_clause} ORDER BY id DESC LIMIT ?",
            vals,
        ).fetchall()
        return [row_to_dict(row) for row in rows]
    finally:
        conn.close()


def get_run_summary(
    run_id: int | None = None,
    run_uid: str | None = None,
    job_name: str | None = None,
    db_path: str | None = None,
) -> dict[str, Any] | None:
    """Get a single run by local ID, portable UID, or job name."""
    conn = connect_registry(db_path)
    try:
        if run_id is not None:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        elif run_uid is not None:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_uid = ?",
                (run_uid,),
            ).fetchone()
        elif job_name is not None:
            row = conn.execute(
                "SELECT * FROM runs WHERE job_name = ? ORDER BY id DESC LIMIT 1",
                (job_name,),
            ).fetchone()
        else:
            return None
        if row is None:
            return None
        result = row_to_dict(row)

        # Include restart chain
        chain: list[dict[str, Any]] = []
        parent_id = result.get("parent_run_id")
        visited: set[int] = {result["id"]}
        while parent_id and parent_id not in visited:
            visited.add(parent_id)
            parent = conn.execute("SELECT * FROM runs WHERE id = ?", (parent_id,)).fetchone()
            if parent is None:
                break
            chain.append({
                "run_id": parent["id"],
                "run_uid": parent["run_uid"],
                "job_name": parent["job_name"],
                "status": parent["status"],
                "energy_hartree": parent["energy_hartree"],
            })
            parent_id = parent["parent_run_id"]
        result["restart_chain"] = list(reversed(chain))
        return result
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a SQLite row to a plain dictionary, parsing JSON fields."""
    value = dict(row)
    for json_field in ("elements", "tags"):
        if value.get(json_field) and isinstance(value[json_field], str):
            try:
                value[json_field] = json.loads(value[json_field])
            except json.JSONDecodeError:
                pass
    return value


__all__ = [
    "get_run_summary",
    "list_runs",
    "register_run",
    "row_to_dict",
    "update_run_status",
]

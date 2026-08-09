"""Refresh one owned launch and summarize its execution and scientific state."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

from chemtools.application.execution import ExecutionService, LaunchStatusError
from chemtools.application.run_inspection import PRIMARY_OUTPUT_LIMIT_BYTES
from chemtools.core.execution import (
    ExecutionLaunchRecord,
    RecordedLocalStatus,
    RecordedSlurmStatus,
)
from chemtools.core.program import ProgramBackend, ProgramCapability
from chemtools.persistence.launches import UnknownLaunchRecordError


MONITOR_RUN_SCHEMA = "chemtools.monitor-run/1"


class MonitorRunError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        launch_id: str | None = None,
        program: str | None = None,
    ) -> None:
        self.code = code
        self.launch_id = launch_id
        self.program = program
        super().__init__(message)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": self.code,
            "message": str(self),
        }
        if self.launch_id is not None:
            payload["launch_id"] = self.launch_id
        if self.program is not None:
            payload["program"] = self.program
        return payload


def monitor_run(
    backend: ProgramBackend,
    service: ExecutionService,
    *,
    launch_id: str,
) -> dict[str, Any]:
    _validate_launch_id(launch_id)
    try:
        record = service.get_launch_record(launch_id)
    except UnknownLaunchRecordError as exc:
        raise MonitorRunError(
            "launch_not_owned",
            f"Chemtools does not own launch {launch_id!r} in this server process",
            launch_id=launch_id,
        ) from exc
    if record.instance_id != service.instance_id:
        raise MonitorRunError(
            "launch_not_owned",
            f"Chemtools does not own launch {launch_id!r} in this server process",
            launch_id=launch_id,
        )
    if record.program != backend.name:
        raise MonitorRunError(
            "launch_program_mismatch",
            (
                f"launch {launch_id!r} belongs to {record.program!r}, "
                f"not {backend.name!r}"
            ),
            launch_id=launch_id,
            program=backend.name,
        )

    record, refreshed = _refresh_status(service, record)
    execution = _execution_evidence(record, refreshed)
    artifacts = {
        "stdout": _observe_path(record.stdout_path),
        "stderr": _observe_path(record.stderr_path),
        "scheduler_script": _observe_path(record.script_path),
    }
    uncertainty: list[dict[str, str]] = []
    scientific = _scientific_evidence(
        backend,
        record,
        artifacts["stdout"],
        uncertainty,
    )
    execution_state = execution["state"]
    verdict = _verdict(execution_state, scientific["status"])
    uncertainty.extend(
        _execution_uncertainty(record, execution_state, scientific["status"])
    )

    return {
        "schema_version": MONITOR_RUN_SCHEMA,
        "status": execution_state,
        "program": {"name": backend.name},
        "launch": {
            "launch_id": record.launch_id,
            "target": record.target,
            "executor": record.executor,
            "status": record.status,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        },
        "assessment": {"verdict": verdict},
        "evidence": {
            "execution": execution,
            "artifacts": artifacts,
            "scientific": scientific,
        },
        "uncertainty": uncertainty,
        "next_actions": _next_actions(
            launch_id,
            execution_state,
            scientific["status"],
            artifacts["stdout"],
        ),
    }


def _validate_launch_id(launch_id: str) -> None:
    if not isinstance(launch_id, str):
        raise MonitorRunError(
            "invalid_launch_id",
            "launch_id must be a canonical UUID string",
        )
    try:
        normalized = str(UUID(launch_id))
    except ValueError as exc:
        raise MonitorRunError(
            "invalid_launch_id",
            "launch_id must be a canonical UUID string",
            launch_id=launch_id,
        ) from exc
    if normalized != launch_id:
        raise MonitorRunError(
            "invalid_launch_id",
            "launch_id must be a canonical lowercase UUID string",
            launch_id=launch_id,
        )


def _refresh_status(
    service: ExecutionService,
    record: ExecutionLaunchRecord,
) -> tuple[
    ExecutionLaunchRecord,
    RecordedLocalStatus | RecordedSlurmStatus | None,
]:
    try:
        if record.executor == "local" and record.status == "started":
            refreshed = service.refresh_local_status(record.launch_id)
            return refreshed.record, refreshed
        if record.executor == "slurm" and record.status in {
            "submitted",
            "cancel_failed",
        }:
            refreshed = service.refresh_slurm_status(record.launch_id)
            return refreshed.record, refreshed
    except LaunchStatusError as exc:
        details = exc.as_dict()
        raise MonitorRunError(
            details["error"],
            f"launch {record.launch_id!r} cannot be refreshed by this server process",
            launch_id=record.launch_id,
            program=record.program,
        ) from exc
    return record, None


def _execution_evidence(
    record: ExecutionLaunchRecord,
    refreshed: RecordedLocalStatus | RecordedSlurmStatus | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state": _execution_state(record, refreshed),
        "record_status": record.status,
        "return_code": record.return_code,
        "elapsed_seconds": record.elapsed_seconds,
        "error": record.error,
        "resources": asdict(record.resources),
    }
    if isinstance(refreshed, RecordedLocalStatus):
        payload["process"] = {
            "process_id": refreshed.result.process_id,
            "status": refreshed.result.status,
            "return_code": refreshed.result.return_code,
            "checked_at": refreshed.result.checked_at.isoformat(),
        }
    elif record.executor == "local":
        payload["process"] = {
            "process_id": record.process_id,
            "status": record.status,
            "return_code": record.return_code,
            "checked_at": record.updated_at.isoformat(),
        }

    if isinstance(refreshed, RecordedSlurmStatus):
        result = refreshed.result
        payload["scheduler"] = {
            "job_id": result.job_id,
            "status": result.status,
            "raw_state": result.raw_state,
            "source": result.source,
            "query_argv": list(result.query_argv),
            "query_return_code": result.query_return_code,
            "job_exit_code": result.job_exit_code,
            "termination_signal": result.termination_signal,
            "elapsed_seconds": result.elapsed_seconds,
            "checked_at": result.checked_at.isoformat(),
        }
    elif record.executor == "slurm":
        payload["scheduler"] = {
            "job_id": record.job_id,
            "status": record.status,
            "raw_state": None,
            "source": "record",
            "query_argv": [],
            "query_return_code": None,
            "job_exit_code": record.return_code,
            "termination_signal": None,
            "elapsed_seconds": record.elapsed_seconds,
            "checked_at": record.updated_at.isoformat(),
        }
    return payload


def _execution_state(
    record: ExecutionLaunchRecord,
    refreshed: RecordedLocalStatus | RecordedSlurmStatus | None,
) -> str:
    if isinstance(refreshed, RecordedLocalStatus):
        return refreshed.result.status
    if isinstance(refreshed, RecordedSlurmStatus):
        return refreshed.result.status
    if record.status == "started":
        return "running"
    if record.status == "submitted":
        return "queued"
    if record.status in {"launch_failed", "submit_failed"}:
        return "failed"
    if record.status == "submitted_untracked":
        return "unknown"
    return record.status


def _observe_path(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "exists": False,
            "size_bytes": None,
            "modified_utc": None,
        }
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "modified_utc": None,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
    }


def _scientific_evidence(
    backend: ProgramBackend,
    record: ExecutionLaunchRecord,
    stdout: Mapping[str, Any],
    uncertainty: list[dict[str, str]],
) -> dict[str, Any]:
    if not stdout["exists"] or not stdout["size_bytes"]:
        uncertainty.append({
            "code": "scientific_output_not_observed",
            "message": "The recorded primary output is absent or empty.",
            "impact": "Scientific progress and completion cannot be assessed yet.",
        })
        return {
            "status": "not_observed",
            "completion_observed": False,
            "outcome": None,
            "progress": None,
        }
    if stdout["size_bytes"] > PRIMARY_OUTPUT_LIMIT_BYTES:
        uncertainty.append({
            "code": "scientific_output_too_large",
            "message": (
                "The recorded primary output exceeds the guided monitoring "
                f"limit of {PRIMARY_OUTPUT_LIMIT_BYTES} bytes."
            ),
            "impact": (
                "Execution state is current, but scientific progress was not parsed."
            ),
        })
        return {
            "status": "unavailable",
            "completion_observed": None,
            "outcome": None,
            "progress": None,
        }
    if not backend.supports(ProgramCapability.PROGRESS_INSPECT):
        uncertainty.append({
            "code": "scientific_progress_unsupported",
            "message": f"{backend.name} has no declared progress inspector.",
            "impact": "Only owned execution and artifact state are available.",
        })
        return {
            "status": "unavailable",
            "completion_observed": None,
            "outcome": None,
            "progress": None,
        }
    assert backend.progress is not None
    try:
        progress = backend.progress.progress_summary(str(stdout["path"]))
    except Exception as exc:
        uncertainty.append({
            "code": "scientific_progress_failed",
            "message": f"{backend.name} progress inspection failed: {exc}",
            "impact": (
                "Execution state is current, but scientific completion is uncertain."
            ),
        })
        return {
            "status": "unavailable",
            "completion_observed": None,
            "outcome": None,
            "progress": None,
        }
    if not isinstance(progress, Mapping):
        uncertainty.append({
            "code": "invalid_scientific_progress",
            "message": (
                f"{backend.name} progress inspection returned a non-mapping value."
            ),
            "impact": (
                "Execution state is current, but scientific completion is uncertain."
            ),
        })
        return {
            "status": "unavailable",
            "completion_observed": None,
            "outcome": None,
            "progress": None,
        }

    outcome = progress.get("outcome")
    if outcome == "success":
        status = "completed"
        completion_observed: bool | None = True
    elif outcome == "failed":
        status = "failed"
        completion_observed = False
    elif outcome == "incomplete":
        status = "incomplete"
        completion_observed = False
    elif record.status in {"started", "submitted", "cancel_failed"}:
        status = "in_progress"
        completion_observed = False
    else:
        status = "unknown"
        completion_observed = None
    return {
        "status": status,
        "completion_observed": completion_observed,
        "outcome": outcome,
        "progress": _compact_progress(progress),
    }


def _compact_progress(progress: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: progress.get(key)
        for key in (
            "current_task_kind",
            "current_task_label",
            "current_phase",
            "status_line",
            "optimization_status",
            "optimization_step_count",
            "optimization_last_step",
            "optimization_final_energy_hartree",
            "frequency_started",
            "frequency_mode_count",
            "significant_imaginary_mode_count",
            "slow_phase",
            "slow_phase_message",
            "task_count",
        )
    }


def _verdict(execution_state: str, scientific_status: str) -> dict[str, Any]:
    active = execution_state in {
        "pending",
        "queued",
        "running",
        "suspended",
        "completing",
    }
    failed = execution_state in {
        "failed",
        "timed_out",
        "out_of_memory",
        "cancelled",
        "launch_failed",
        "submit_failed",
    }
    if active:
        return {
            "label": "run_active",
            "confidence": 1.0,
            "reasons": [
                f"Owned execution state is {execution_state}.",
                f"Scientific output state is {scientific_status}.",
            ],
        }
    if scientific_status == "completed" and not failed:
        return {
            "label": "completed_success",
            "confidence": 0.95,
            "reasons": [
                f"Owned execution state is {execution_state}.",
                "The progress inspector found a successful scientific outcome.",
            ],
        }
    if failed or scientific_status in {"failed", "incomplete"}:
        return {
            "label": "completed_failed",
            "confidence": 0.9,
            "reasons": [
                f"Owned execution state is {execution_state}.",
                f"Scientific output state is {scientific_status}.",
            ],
        }
    return {
        "label": "completion_unverified",
        "confidence": 0.6,
        "reasons": [
            f"Owned execution state is {execution_state}.",
            f"Scientific output state is {scientific_status}.",
        ],
    }


def _execution_uncertainty(
    record: ExecutionLaunchRecord,
    execution_state: str,
    scientific_status: str,
) -> list[dict[str, str]]:
    uncertainty = []
    if execution_state in {"not_found", "unknown", "submitted_untracked"}:
        uncertainty.append({
            "code": "execution_state_unresolved",
            "message": f"The owned execution state is {execution_state}.",
            "impact": "Do not infer that the calculation finished or resubmit it.",
        })
    if (
        record.status == "completed"
        and scientific_status not in {"completed", "failed", "incomplete"}
    ):
        uncertainty.append({
            "code": "scientific_completion_unverified",
            "message": (
                "The executor completed, but scientific completion was not established."
            ),
            "impact": "Inspect the primary output before accepting the calculation.",
        })
    return uncertainty


def _next_actions(
    launch_id: str,
    execution_state: str,
    scientific_status: str,
    stdout: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if execution_state in {
        "pending",
        "queued",
        "running",
        "suspended",
        "completing",
        "not_found",
    }:
        return [{
            "action": "monitor_run",
            "arguments": {"launch_id": launch_id},
            "reason": "Refresh this owned launch without changing it.",
            "priority": 1,
        }]
    if stdout["exists"] and stdout["size_bytes"]:
        return [{
            "action": "inspect_run",
            "arguments": {"output_file": stdout["path"]},
            "reason": (
                "Run the full scientific inspection before accepting or "
                "recovering the calculation."
            ),
            "priority": 1,
        }]
    if scientific_status == "not_observed":
        return [{
            "action": "inspect_launch_artifacts",
            "reason": "The execution ended without a non-empty primary output.",
            "priority": 1,
        }]
    return []


__all__ = [
    "MONITOR_RUN_SCHEMA",
    "MonitorRunError",
    "monitor_run",
]

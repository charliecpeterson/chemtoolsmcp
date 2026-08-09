"""Project owned execution state into program status responses.

Program adapters keep file parsing and scientific interpretation. This module
owns the shared local and Slurm status fields plus the polling boundary.
"""

from __future__ import annotations

from typing import Any, Callable

from chemtools.application.execution import ExecutionService, LaunchStatusError
from chemtools.core.execution import (
    RecordedLocalStatus,
    RecordedSlurmStatus,
)
from chemtools.core.monitoring import watch_run_status


def refresh_owned_local_status(
    service: ExecutionService,
    process_id: int,
    *,
    program: str,
    program_label: str,
) -> RecordedLocalStatus:
    recorded = service.refresh_local_status_external(process_id)
    if recorded.record.program != program:
        raise ValueError(
            f"local process {process_id} belongs to "
            f"{recorded.record.program!r}, not {program_label}"
        )
    return recorded


def refresh_owned_slurm_status(
    service: ExecutionService,
    job_id: str,
    *,
    profile: str | None,
    program: str,
    program_label: str,
) -> RecordedSlurmStatus | None:
    try:
        recorded = service.refresh_slurm_status_external(
            job_id,
            target_name=profile,
        )
    except LaunchStatusError as exc:
        if exc.as_dict()["error"] == "launch_not_owned":
            return None
        raise
    if recorded.record.program != program:
        raise ValueError(
            f"Slurm job {job_id} belongs to "
            f"{recorded.record.program!r}, not {program_label}"
        )
    return recorded


def project_owned_execution_status(
    status: dict[str, Any],
    *,
    recorded_local: RecordedLocalStatus | None,
    recorded_slurm: RecordedSlurmStatus | None,
    process_id: int | None,
    job_id: str | None,
) -> dict[str, Any]:
    if recorded_local is not None:
        status["process"] = {
            "process_id": process_id,
            "status": recorded_local.result.status,
            "return_code": recorded_local.result.return_code,
        }
        status["execution_record"] = {
            "launch_id": recorded_local.record.launch_id,
            "status": recorded_local.record.status,
            "elapsed_seconds": recorded_local.record.elapsed_seconds,
        }
        if recorded_local.result.status == "running":
            status["overall_status"] = "running"
        elif recorded_local.result.status == "failed":
            status["overall_status"] = "completed_failed"
    if recorded_slurm is None:
        return status

    slurm_result = recorded_slurm.result
    status["scheduler"] = {
        "job_id": job_id,
        "status": slurm_result.status,
        "scheduler_type": "slurm",
        "command": list(slurm_result.query_argv),
        "return_code": slurm_result.query_return_code,
        "raw_state": slurm_result.raw_state,
        "stdout": slurm_result.stdout,
        "stderr": slurm_result.stderr,
        "source": slurm_result.source,
        "job_exit_code": slurm_result.job_exit_code,
        "termination_signal": slurm_result.termination_signal,
        "elapsed_seconds": slurm_result.elapsed_seconds,
    }
    status["execution_record"] = {
        "launch_id": recorded_slurm.record.launch_id,
        "status": recorded_slurm.record.status,
        "elapsed_seconds": recorded_slurm.record.elapsed_seconds,
    }
    status["overall_status"] = _slurm_overall_status(
        slurm_result.status,
        status["overall_status"],
    )
    return status


def watch_owned_execution_status(
    service: ExecutionService,
    status_reader: Callable[[], dict[str, Any]],
    *,
    process_id: int | None,
    job_id: str | None,
    profile: str | None,
    poll_interval_seconds: float,
    adaptive_polling: bool,
    max_poll_interval_seconds: float | None,
    timeout_seconds: float | None,
    max_polls: int | None,
    history_limit: int,
    stall_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    if (process_id is None) == (job_id is None):
        raise ValueError("provide exactly one of process_id or job_id")
    if process_id is not None:
        service.resolve_local_launch_external(process_id)
        identifier = str(process_id)
    else:
        assert job_id is not None
        service.resolve_slurm_launch_external(
            job_id,
            target_name=profile,
        )
        identifier = job_id

    def read_owned_status() -> dict[str, Any]:
        status = status_reader()
        if "execution_record" not in status:
            raise LaunchStatusError({
                "error": "launch_not_owned",
                "identifier": identifier,
            })
        return status

    return watch_run_status(
        read_owned_status,
        poll_interval_seconds=poll_interval_seconds,
        adaptive_polling=adaptive_polling,
        max_poll_interval_seconds=max_poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        max_polls=max_polls,
        history_limit=history_limit,
        stall_timeout_seconds=stall_timeout_seconds,
    )


def _slurm_overall_status(
    slurm_status: str,
    current_status: str,
) -> str:
    if slurm_status == "queued":
        return "queued"
    if slurm_status in ("running", "suspended", "completing"):
        return "running"
    if slurm_status == "cancelled":
        return "cancelled"
    if slurm_status in ("failed", "timed_out", "out_of_memory"):
        return "completed_failed"
    return current_status


__all__ = [
    "project_owned_execution_status",
    "refresh_owned_local_status",
    "refresh_owned_slurm_status",
    "watch_owned_execution_status",
]

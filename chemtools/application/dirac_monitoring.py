"""Combine owned execution status with legacy DIRAC file inspection."""

from __future__ import annotations

from typing import Any

from chemtools.application.execution import ExecutionService
from chemtools.application.execution_monitoring import (
    project_owned_execution_status,
    refresh_owned_local_status,
    refresh_owned_slurm_status,
    watch_owned_execution_status,
)
from chemtools.programs.dirac.scheduler import get_dirac_run_status


def inspect_dirac_status_with_service(
    service: ExecutionService,
    *,
    output_path: str | None = None,
    input_path: str | None = None,
    error_path: str | None = None,
    process_id: int | None = None,
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    recorded_local = (
        refresh_owned_local_status(
            service,
            process_id,
            program="dirac",
            program_label="DIRAC",
        )
        if process_id is not None
        else None
    )
    recorded_slurm = (
        refresh_owned_slurm_status(
            service,
            job_id,
            profile=profile,
            program="dirac",
            program_label="DIRAC",
        )
        if job_id is not None
        else None
    )
    status = get_dirac_run_status(
        output_path=output_path,
        input_path=input_path,
        error_path=error_path,
        process_id=None if recorded_local is not None else process_id,
        profile=None if recorded_slurm is not None else profile,
        job_id=None if recorded_slurm is not None else job_id,
        profiles_path=profiles_path,
    )
    return project_owned_execution_status(
        status,
        recorded_local=recorded_local,
        recorded_slurm=recorded_slurm,
        process_id=process_id,
        job_id=job_id,
    )


def watch_dirac_status_with_service(
    service: ExecutionService,
    *,
    process_id: int | None = None,
    job_id: str | None = None,
    profile: str | None = None,
    output_path: str | None = None,
    input_path: str | None = None,
    error_path: str | None = None,
    profiles_path: str | None = None,
    poll_interval_seconds: float = 10.0,
    adaptive_polling: bool = True,
    max_poll_interval_seconds: float | None = 60.0,
    timeout_seconds: float | None = 3600.0,
    max_polls: int | None = None,
    history_limit: int = 8,
) -> dict[str, Any]:
    def read_status() -> dict[str, Any]:
        return inspect_dirac_status_with_service(
            service,
            output_path=output_path,
            input_path=input_path,
            error_path=error_path,
            process_id=process_id,
            profile=profile if job_id is not None else None,
            job_id=job_id,
            profiles_path=profiles_path,
        )

    watched = watch_owned_execution_status(
        service,
        read_status,
        process_id=process_id,
        job_id=job_id,
        profile=profile,
        poll_interval_seconds=poll_interval_seconds,
        adaptive_polling=adaptive_polling,
        max_poll_interval_seconds=max_poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        max_polls=max_polls,
        history_limit=history_limit,
    )
    watched["overall_status"] = watched["final_status"]["overall_status"]
    return watched


__all__ = [
    "inspect_dirac_status_with_service",
    "watch_dirac_status_with_service",
]

"""Coordinate typed execution status with NWChem progress inspection."""

from __future__ import annotations

from typing import Any

from chemtools.application.execution import ExecutionService
from chemtools.application.execution_monitoring import (
    project_owned_execution_status,
    watch_owned_execution_status,
)
from chemtools.application.nwchem_execution import (
    refresh_nwchem_local_status_with_service,
    refresh_nwchem_slurm_status_with_service,
)
from chemtools.programs.nwchem.runner import (
    check_nwchem_run_status,
    review_nwchem_progress,
)


def inspect_nwchem_status_with_service(
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
    recorded_local = None
    if process_id is not None:
        recorded_local = refresh_nwchem_local_status_with_service(
            service,
            process_id,
        )
    recorded_slurm = None
    if job_id is not None:
        recorded_slurm = refresh_nwchem_slurm_status_with_service(
            service,
            job_id,
            profile=profile,
        )

    inspected_profile = profile
    inspected_job_id = job_id
    if recorded_slurm is not None:
        inspected_profile = None
        inspected_job_id = None

    status = check_nwchem_run_status(
        output_path=output_path,
        input_path=input_path,
        error_path=error_path,
        profile=inspected_profile,
        job_id=inspected_job_id,
        profiles_path=profiles_path,
    )
    return project_owned_execution_status(
        status,
        recorded_local=recorded_local,
        recorded_slurm=recorded_slurm,
        process_id=process_id,
        job_id=job_id,
    )


def watch_nwchem_status_with_service(
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
    stall_timeout_seconds: float | None = 1800.0,
) -> dict[str, Any]:
    def read_status() -> dict[str, Any]:
        return inspect_nwchem_status_with_service(
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
        stall_timeout_seconds=stall_timeout_seconds,
    )
    final_progress = (
        review_nwchem_progress(
            output_path=output_path,
            input_path=input_path,
            error_path=error_path,
        )
        if output_path
        else None
    )
    return {
        "terminal": watched["terminal"],
        "stop_reason": watched["stop_reason"],
        "poll_count": watched["poll_count"],
        "elapsed_seconds": watched["elapsed_seconds"],
        "overall_status": watched["final_status"]["overall_status"],
        "adaptive_polling": watched["adaptive_polling"],
        "max_poll_interval_seconds": watched[
            "max_poll_interval_seconds"
        ],
        "history_limit": watched["history_limit"],
        "last_sleep_seconds": watched["last_sleep_seconds"],
        "history": watched["history"],
        "final_status": watched["final_status"],
        "final_progress": final_progress,
        "summary_text": (final_progress or {}).get("summary_text"),
    }


__all__ = [
    "inspect_nwchem_status_with_service",
    "watch_nwchem_status_with_service",
]

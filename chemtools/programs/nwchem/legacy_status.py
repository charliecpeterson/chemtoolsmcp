"""Compose generic legacy job status with NWChem progress inspection."""

from __future__ import annotations

from typing import Any

from chemtools.execution.legacy_status import inspect_run_status, watch_run
from chemtools.programs.nwchem.strategy.progress import (
    inspect_legacy_status_output,
)


def inspect_nwchem_run_status(
    *,
    output_path: str | None = None,
    input_path: str | None = None,
    error_path: str | None = None,
    process_id: int | None = None,
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
    progress_summary_fn: Any = None,
) -> dict[str, Any]:
    return inspect_run_status(
        output_path=output_path,
        input_path=input_path,
        error_path=error_path,
        process_id=process_id,
        profile=profile,
        job_id=job_id,
        profiles_path=profiles_path,
        output_status_reader=inspect_legacy_status_output,
        progress_summary_fn=progress_summary_fn,
    )


def watch_nwchem_run_status(
    *,
    output_path: str | None = None,
    input_path: str | None = None,
    error_path: str | None = None,
    process_id: int | None = None,
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
    poll_interval_seconds: float = 10.0,
    adaptive_polling: bool = True,
    max_poll_interval_seconds: float | None = 60.0,
    timeout_seconds: float | None = 3600.0,
    max_polls: int | None = None,
    history_limit: int = 8,
    stall_timeout_seconds: float | None = None,
    progress_summary_fn: Any = None,
) -> dict[str, Any]:
    return watch_run(
        output_path=output_path,
        input_path=input_path,
        error_path=error_path,
        process_id=process_id,
        profile=profile,
        job_id=job_id,
        profiles_path=profiles_path,
        poll_interval_seconds=poll_interval_seconds,
        adaptive_polling=adaptive_polling,
        max_poll_interval_seconds=max_poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        max_polls=max_polls,
        history_limit=history_limit,
        stall_timeout_seconds=stall_timeout_seconds,
        output_status_reader=inspect_legacy_status_output,
        progress_summary_fn=progress_summary_fn,
    )


__all__ = ["inspect_nwchem_run_status", "watch_nwchem_run_status"]

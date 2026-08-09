"""Molcas scheduler runner wrappers.

Thin wrappers around the version 1 renderer and launcher plus the focused
external-status owner. The launch path handles scheduler submission and job-ID
persistence. Status combines files with an explicitly identified external
Slurm job; owned launches use the execution service instead.

Profiles consumed here have ``launcher.kind = "scheduler"`` and a
``programs.molcas`` block containing launcher and executable arrays. The
script template can reference ``{program_command}``, ``{job_name}``, and
``{input_file}``; see the TACC Stampede3 example profiles. Previous
program-specific fields remain compatibility inputs.
"""

from __future__ import annotations

from typing import Any

from chemtools.execution.legacy_runner import (
    render_calculation_run,
    run_calculation,
)
from chemtools.execution.external_status import (
    inspect_run_status,
    watch_run,
)


def launch_molcas_run(
    input_path: str,
    profile: str,
    *,
    profiles_path: str | None = None,
    job_name: str | None = None,
    resource_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, str] | None = None,
    write_script: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Submit a Molcas job to the scheduler defined in ``profile``.

    Uses the program-neutral engine in
    ``chemtools/execution/legacy_runner.py``. The
    profile's ``scheduler.script_template`` is rendered with the standard
    placeholders, including the neutral ``{program_command}``.

    Parameters
    ----------
    input_path
        Path to the Molcas ``.input`` file.
    profile
        Runner-profile name (must have ``launcher.kind = "scheduler"``).
    dry_run
        When True, render the submit script but do NOT call sbatch.
    """
    if dry_run:
        result = render_calculation_run(
            input_path=input_path,
            profile=profile,
            profiles_path=profiles_path,
            job_name=job_name,
            resource_overrides=resource_overrides,
            env_overrides=env_overrides,
        )
        result.pop("environment", None)
        return result
    return run_calculation(
        input_path=input_path,
        profile=profile,
        profiles_path=profiles_path,
        job_name=job_name,
        resource_overrides=resource_overrides,
        env_overrides=env_overrides,
        execute=True,
        write_script=write_script,
    )

def get_molcas_run_status(
    output_path: str | None = None,
    input_path: str | None = None,
    error_path: str | None = None,
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    """Inspect Molcas files or an explicitly identified external Slurm job.

    The generic Molcas-output parser path is not yet wired into
    ``inspect_run_status``, so ``progress_summary`` is None; scheduler state
    and file presence still contribute to ``overall_status``.
    """
    return inspect_run_status(
        output_path=output_path,
        input_path=input_path,
        error_path=error_path,
        profile=profile,
        job_id=job_id,
        profiles_path=profiles_path,
    )


def watch_molcas_run(
    output_path: str | None = None,
    input_path: str | None = None,
    error_path: str | None = None,
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
    poll_interval_seconds: float = 10.0,
    adaptive_polling: bool = True,
    max_poll_interval_seconds: float | None = 60.0,
    timeout_seconds: float | None = 3600.0,
    max_polls: int | None = None,
    history_limit: int = 8,
) -> dict[str, Any]:
    """Poll a Molcas job until it reaches a terminal state."""
    return watch_run(
        output_path=output_path,
        input_path=input_path,
        error_path=error_path,
        profile=profile,
        job_id=job_id,
        profiles_path=profiles_path,
        poll_interval_seconds=poll_interval_seconds,
        adaptive_polling=adaptive_polling,
        max_poll_interval_seconds=max_poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        max_polls=max_polls,
        history_limit=history_limit,
    )

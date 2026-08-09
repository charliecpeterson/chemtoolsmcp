"""Molcas scheduler runner wrappers.

Thin wrappers around the version 1 engine in
``chemtools/execution/legacy_runner.py``
that submit, monitor, and cancel Molcas jobs through an HPC scheduler. Mirrors
the NWChem pattern in ``chemtools/programs/nwchem/runner.py`` — the underlying
machinery (sbatch invocation, job-id parsing, ``.jobid`` writing, ``squeue``
polling, output tailing) is generic; this module is just the public Molcas
naming.

Profiles consumed here have ``launcher.kind = "scheduler"`` and a
``programs.molcas`` block containing launcher and executable arrays. The
script template can reference ``{program_command}``, ``{job_name}``, and
``{input_file}``; see the TACC Stampede3 example profiles. Previous
program-specific fields remain compatibility inputs.
"""

from __future__ import annotations

from typing import Any

from chemtools.execution.legacy_runner import (
    cancel_scheduler_job,
    inspect_run_status,
    render_calculation_run,
    run_calculation,
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
    process_id: int | None = None,
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    """Non-blocking status for a Molcas job (HPC or local).

    Reads the ``.jobid`` file alongside the input/output if ``job_id`` is
    not supplied. NOTE: the generic Molcas-output parser path is not yet
    wired into ``inspect_run_status``, so ``progress_summary`` will
    be None for Molcas runs; ``overall_status`` still works via scheduler
    state + file presence.
    """
    return inspect_run_status(
        output_path=output_path,
        input_path=input_path,
        error_path=error_path,
        process_id=process_id,
        profile=profile,
        job_id=job_id,
        profiles_path=profiles_path,
    )


def watch_molcas_run(
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
) -> dict[str, Any]:
    """Poll a Molcas job until it reaches a terminal state."""
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
    )


def terminate_molcas_run(
    job_id: str,
    profile: str,
    *,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    """Cancel a Molcas job via the scheduler's cancel_command (scancel / qdel)."""
    if not job_id:
        raise ValueError("job_id is required to cancel a Molcas scheduler job")
    if not profile:
        raise ValueError("profile is required to resolve the cancel_command")
    return cancel_scheduler_job(
        profile=profile, job_id=job_id, profiles_path=profiles_path,
    )

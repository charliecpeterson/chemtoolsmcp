"""GRASP scheduler runner wrappers.

GRASP is structurally different from NWChem / Molcas / DIRAC: a typical
``input`` is a *shell script* that runs ~50 GRASP executables (rnucleus,
rcsfgenerate, rangular, rwfnestimate, rmcdhf, rsave, jj2lsj, rlevels, ...)
in sequence. The script_template's ``bash {input_file}`` line executes
that shell script inside the apptainer container.

Otherwise the submit / status / watch / cancel pattern is identical to
the other programs and delegates to the program-neutral engine in
``chemtools/core/runner.py``.
"""

from __future__ import annotations

from typing import Any

from chemtools.core.runner import (
    cancel_scheduler_job,
    inspect_nwchem_run_status,
    render_nwchem_run,
    run_nwchem,
    watch_nwchem_run as _watch_nwchem_run,
)


def launch_grasp_workflow_run(
    workflow_script_path: str,
    profile: str,
    *,
    profiles_path: str | None = None,
    job_name: str | None = None,
    resource_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, str] | None = None,
    write_script: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Submit a GRASP workflow shell script to the scheduler defined in ``profile``.

    Parameters
    ----------
    workflow_script_path
        Path to the GRASP workflow shell script (the one the script_template
        invokes via ``bash {input_file}``). Generate this script locally via
        ``plan_grasp_dhf_workflow`` + the heredoc input builders.
    profile
        Runner-profile name (must have ``launcher.kind = "scheduler"``).
    dry_run
        When True, render the submit script but do NOT call sbatch.
    """
    if dry_run:
        result = render_nwchem_run(
            input_path=workflow_script_path,
            profile=profile,
            profiles_path=profiles_path,
            job_name=job_name,
            resource_overrides=resource_overrides,
            env_overrides=env_overrides,
        )
        result.pop("environment", None)
        return result
    return run_nwchem(
        input_path=workflow_script_path,
        profile=profile,
        profiles_path=profiles_path,
        job_name=job_name,
        resource_overrides=resource_overrides,
        env_overrides=env_overrides,
        execute=True,
        write_script=write_script,
    )


def get_grasp_run_status(
    output_path: str | None = None,
    input_path: str | None = None,
    error_path: str | None = None,
    process_id: int | None = None,
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    """Non-blocking status for a GRASP job (HPC or local)."""
    return inspect_nwchem_run_status(
        output_path=output_path,
        input_path=input_path,
        error_path=error_path,
        process_id=process_id,
        profile=profile,
        job_id=job_id,
        profiles_path=profiles_path,
    )


def watch_grasp_run(
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
    """Poll a GRASP job until it reaches a terminal state."""
    return _watch_nwchem_run(
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


def terminate_grasp_run(
    job_id: str,
    profile: str,
    *,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    """Cancel a GRASP job via the scheduler's cancel_command."""
    if not job_id:
        raise ValueError("job_id is required to cancel a GRASP scheduler job")
    if not profile:
        raise ValueError("profile is required to resolve the cancel_command")
    return cancel_scheduler_job(
        profile=profile, job_id=job_id, profiles_path=profiles_path,
    )

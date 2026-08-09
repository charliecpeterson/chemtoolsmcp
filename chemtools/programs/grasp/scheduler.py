"""GRASP scheduler runner wrappers.

GRASP is structurally different from NWChem / Molcas / DIRAC: a typical
``input`` is a *shell script* that runs ~50 GRASP executables (rnucleus,
rcsfgenerate, rangular, rwfnestimate, rmcdhf, rsave, jj2lsj, rlevels, ...)
in sequence. The script_template's ``bash {input_file}`` line executes
that shell script inside the apptainer container.

Rendering and submission delegate to the version 1 engine in
``chemtools/execution/legacy_runner.py``. Read-only file and explicit external
Slurm inspection use the focused external-status owner.
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
        result = render_calculation_run(
            input_path=workflow_script_path,
            profile=profile,
            profiles_path=profiles_path,
            job_name=job_name,
            resource_overrides=resource_overrides,
            env_overrides=env_overrides,
        )
        result.pop("environment", None)
        return result
    return run_calculation(
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
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    """Inspect GRASP files or an explicitly identified external Slurm job."""
    return inspect_run_status(
        output_path=output_path,
        input_path=input_path,
        error_path=error_path,
        profile=profile,
        job_id=job_id,
        profiles_path=profiles_path,
    )


def watch_grasp_run(
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
    """Poll a GRASP job until it reaches a terminal state."""
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

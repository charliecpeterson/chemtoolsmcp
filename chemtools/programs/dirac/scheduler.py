"""DIRAC scheduler runner wrappers.

Submits / monitors / cancels DIRAC ``pam-dirac`` jobs through an HPC scheduler.
DIRAC has one special wrinkle relative to NWChem and Molcas: ``pam-dirac``
needs *both* a ``--inp=`` and ``--mol=`` argument, so the script_template
references both ``{input_file}`` and ``{mol_file}``. ``launch_dirac_run``
takes a ``mol_file`` argument and passes it through ``context_overrides``
so the renderer can populate the placeholder.

Profiles consumed here have ``launcher.kind = "scheduler"`` and read
``container_sif`` from the *top level* of the profile (not nested under
``execution.*``, mirroring how the DIRAC runtime + the example profiles do it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.core.runner import (
    cancel_scheduler_job,
    inspect_nwchem_run_status,
    render_nwchem_run,
    run_nwchem,
    watch_nwchem_run as _watch_nwchem_run,
)


def launch_dirac_run(
    input_path: str,
    mol_file: str,
    profile: str,
    *,
    profiles_path: str | None = None,
    job_name: str | None = None,
    resource_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, str] | None = None,
    write_script: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Submit a DIRAC job to the scheduler defined in ``profile``.

    Parameters
    ----------
    input_path
        Path to the DIRAC ``.inp`` file.
    mol_file
        Path to the matching ``.mol`` file. Only the basename is used in the
        script_template (pam-dirac runs in the working directory).
    profile
        Runner-profile name (must have ``launcher.kind = "scheduler"``).
    dry_run
        When True, render the submit script but do NOT call sbatch.
    """
    if not mol_file:
        raise ValueError("mol_file is required for DIRAC launches (pam-dirac --mol=...)")
    mol_basename = Path(mol_file).name
    overrides = {"mol_file": mol_basename}
    if dry_run:
        result = render_nwchem_run(
            input_path=input_path,
            profile=profile,
            profiles_path=profiles_path,
            job_name=job_name,
            resource_overrides=resource_overrides,
            env_overrides=env_overrides,
            context_overrides=overrides,
        )
        result.pop("environment", None)
        return result
    return run_nwchem(
        input_path=input_path,
        profile=profile,
        profiles_path=profiles_path,
        job_name=job_name,
        resource_overrides=resource_overrides,
        env_overrides=env_overrides,
        execute=True,
        write_script=write_script,
        context_overrides=overrides,
    )


def get_dirac_run_status(
    output_path: str | None = None,
    input_path: str | None = None,
    error_path: str | None = None,
    process_id: int | None = None,
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    """Non-blocking status for a DIRAC job (HPC or local)."""
    return inspect_nwchem_run_status(
        output_path=output_path,
        input_path=input_path,
        error_path=error_path,
        process_id=process_id,
        profile=profile,
        job_id=job_id,
        profiles_path=profiles_path,
    )


def watch_dirac_run(
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
    """Poll a DIRAC job until it reaches a terminal state."""
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


def terminate_dirac_run(
    job_id: str,
    profile: str,
    *,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    """Cancel a DIRAC job via the scheduler's cancel_command."""
    if not job_id:
        raise ValueError("job_id is required to cancel a DIRAC scheduler job")
    if not profile:
        raise ValueError("profile is required to resolve the cancel_command")
    return cancel_scheduler_job(
        profile=profile, job_id=job_id, profiles_path=profiles_path,
    )

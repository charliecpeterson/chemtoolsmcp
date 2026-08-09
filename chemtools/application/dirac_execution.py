"""Translate DIRAC MCP launch calls to typed execution services."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shlex
from typing import Any

from chemtools.application.execution import (
    ExecutionService,
    LaunchCancellationError,
)
from chemtools.application.legacy_execution import (
    apply_legacy_launch_result,
    legacy_slurm_cancellation_result,
)
from chemtools.core.execution import RenderedSlurmScript
from chemtools.execution.legacy_archive import archive_paths
from chemtools.execution.profiles import load_runner_profiles, resource_request
from chemtools.programs.dirac.launch import (
    adapt_legacy_dirac_profile,
    build_dirac_launch_plan,
)
from chemtools.programs.dirac.scheduler import (
    launch_dirac_run as legacy_launch_dirac_run,
)


def launch_dirac_with_service(
    service: ExecutionService,
    *,
    input_path: str,
    mol_file: str,
    profile: str,
    profiles_path: str | None = None,
    job_name: str | None = None,
    resource_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, str] | None = None,
    write_script: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    preview = legacy_launch_dirac_run(
        input_path=input_path,
        mol_file=mol_file,
        profile=profile,
        profiles_path=profiles_path,
        job_name=job_name,
        resource_overrides=resource_overrides,
        env_overrides=env_overrides,
        write_script=write_script,
        dry_run=True,
    )
    if dry_run:
        return preview

    input_file = Path(preview["input_file"]).resolve()
    molecule_file = Path(mol_file).resolve()
    working_directory = Path(preview["working_directory"]).resolve()
    if working_directory != input_file.parent:
        raise ValueError(
            "typed DIRAC execution currently requires the profile working "
            "directory to match the input directory"
        )
    profiles = load_runner_profiles(profiles_path)
    adapted = adapt_legacy_dirac_profile(
        profiles,
        profile,
        allowed_work_roots=(working_directory,),
    )
    if adapted.target.executor == "slurm" and not write_script:
        raise ValueError(
            "write_script=False is not supported for typed Slurm execution"
        )
    requested_resources = resource_request(preview["resources"])
    if not preview["resources"].get("mpi_ranks"):
        requested_resources = replace(
            requested_resources,
            mpi_ranks=adapted.default_resources.mpi_ranks,
        )
    plan = build_dirac_launch_plan(
        input_file,
        molecule_file,
        requested_resources,
        master_memory_mb=adapted.master_memory_mb,
        node_memory_mb=adapted.node_memory_mb,
        job_name=preview["job_name"],
        output_template=adapted.output_template,
        error_template=adapted.error_template,
        environment=env_overrides,
    )

    service.require("launch", adapted.target)
    rendered = service.render(plan, adapted.target)
    if isinstance(rendered, RenderedSlurmScript):
        command = rendered.command
        script_path = rendered.script_path
    else:
        command = rendered
        script_path = None
    archived = archive_paths([
        path
        for path in (
            command.stdout_path,
            command.stderr_path,
            script_path,
        )
        if path is not None
    ])
    launched = service.launch(plan, adapted.target)

    preview["mol_file"] = str(molecule_file)
    preview["resources"]["mpi_ranks"] = plan.resources.mpi_ranks
    preview["master_memory_mb"] = adapted.master_memory_mb
    preview["node_memory_mb"] = adapted.node_memory_mb
    if archived:
        preview["archived_previous_outputs"] = archived
    response = apply_legacy_launch_result(
        preview,
        launched,
        timeout_error="sbatch/qsub timed out after 60 seconds",
    )
    if adapted.target.executor == "local":
        response["command"] = (
            f"{shlex.join(launched.record.argv)} "
            f"> {shlex.quote(str(launched.record.stdout_path))} "
            f"2> {shlex.quote(str(launched.record.stderr_path))}"
        )
    return response


def terminate_dirac_with_service(
    service: ExecutionService,
    *,
    job_id: str,
    profile: str,
) -> dict[str, Any]:
    if not job_id:
        raise ValueError(
            "job_id is required to cancel a DIRAC scheduler job"
        )
    if not profile:
        raise ValueError(
            "profile is required to resolve the cancel_command"
        )
    try:
        cancelled = service.cancel_external(
            job_id=job_id,
            target_name=profile,
        )
    except LaunchCancellationError as exc:
        return {
            "job_id": job_id,
            "cancelled": False,
            "error": exc.as_dict()["error"],
        }
    return legacy_slurm_cancellation_result(cancelled)


__all__ = [
    "launch_dirac_with_service",
    "terminate_dirac_with_service",
]

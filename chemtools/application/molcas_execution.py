"""Translate OpenMolcas MCP launch calls to typed execution services."""

from __future__ import annotations

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
from chemtools.core.runner import (
    archive_paths,
    load_runner_profiles,
)
from chemtools.execution.legacy_profiles import resource_request
from chemtools.programs.molcas.launch import (
    adapt_legacy_molcas_profile,
    build_molcas_launch_plan,
)
from chemtools.programs.molcas.scheduler import (
    launch_molcas_run as legacy_launch_molcas_run,
)


def launch_molcas_with_service(
    service: ExecutionService,
    *,
    input_path: str,
    profile: str,
    profiles_path: str | None = None,
    job_name: str | None = None,
    resource_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, str] | None = None,
    write_script: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    preview = legacy_launch_molcas_run(
        input_path=input_path,
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
    working_directory = Path(preview["working_directory"]).resolve()
    if working_directory != input_file.parent:
        raise ValueError(
            "typed Molcas execution currently requires the profile working "
            "directory to match the input directory"
        )
    profiles = load_runner_profiles(profiles_path)
    adapted = adapt_legacy_molcas_profile(
        profiles,
        profile,
        allowed_work_roots=(working_directory,),
    )
    if adapted.target.executor == "slurm" and not write_script:
        raise ValueError(
            "write_script=False is not supported for typed Slurm execution"
        )
    prepared = build_molcas_launch_plan(
        input_file,
        resource_request(preview["resources"]),
        parallel_caspt2_supported=(
            adapted.parallel_caspt2_supported
        ),
        job_name=preview["job_name"],
        output_template=adapted.output_template,
        error_template=adapted.error_template,
        environment=env_overrides,
    )

    service.require("launch", adapted.target)
    rendered = service.render(prepared.plan, adapted.target)
    if isinstance(rendered, RenderedSlurmScript):
        command = rendered.command
        script_path = rendered.script_path
    else:
        command = rendered
        script_path = None
    archive_candidates = [
        path
        for path in (
            command.stdout_path,
            command.stderr_path,
            script_path,
        )
        if path is not None
    ]
    archived = archive_paths(archive_candidates)
    launched = service.launch(prepared.plan, adapted.target)

    preview["resources"]["nodes"] = prepared.plan.resources.nodes
    preview["resources"]["mpi_ranks"] = (
        prepared.plan.resources.mpi_ranks
    )
    preview["requested_np"] = prepared.requested_mpi_ranks
    preview["effective_np"] = prepared.effective_mpi_ranks
    preview["project"] = prepared.plan.job_name
    preview["has_caspt2"] = prepared.has_caspt2
    preview["parallel_caspt2_supported"] = (
        prepared.parallel_caspt2_supported
    )
    if prepared.warnings:
        preview["warnings"] = list(prepared.warnings)
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


def terminate_molcas_with_service(
    service: ExecutionService,
    *,
    job_id: str,
    profile: str,
) -> dict[str, Any]:
    if not job_id:
        raise ValueError(
            "job_id is required to cancel a Molcas scheduler job"
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
    "launch_molcas_with_service",
    "terminate_molcas_with_service",
]

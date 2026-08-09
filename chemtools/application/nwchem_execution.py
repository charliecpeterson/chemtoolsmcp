"""Translate NWChem runner calls to the typed execution service.

Dry-run responses remain on the legacy renderer. Live calls keep the legacy
response fields while process and scheduler changes pass through the service.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

from chemtools.application.execution import (
    ExecutionService,
    LaunchCancellationError,
    LaunchStatusError,
)
from chemtools.application.legacy_execution import (
    apply_legacy_launch_result,
    legacy_slurm_cancellation_result,
)
from chemtools.persistence.artifacts import (
    load_run_artifacts,
    record_run_artifacts,
)
from chemtools.core.artifacts import (
    ArtifactLocation,
    ArtifactObservation,
    ArtifactRef,
    ArtifactRole,
    RunArtifacts,
    StepRef,
)
from chemtools.core.execution import (
    ExecutionLaunchRecord,
    LocalCancellationResult,
    RecordedLocalStatus,
    RecordedSlurmStatus,
)
from chemtools.execution.legacy_archive import archive_previous_outputs
from chemtools.execution.legacy_runner import render_calculation_run
from chemtools.persistence.runs import (
    get_run_summary,
    register_run,
    update_run_status,
)
from chemtools.persistence.launches import (
    UnknownExecutionRunLinkError,
    load_execution_run_link,
)
from chemtools.programs.nwchem.launch import (
    adapt_legacy_nwchem_profile,
    build_nwchem_launch_plan,
)
from chemtools.programs.nwchem.runner import (
    launch_nwchem_run as legacy_launch_nwchem_run,
)
from chemtools.execution.profiles import (
    load_runner_profiles,
    resolve_runner_profile,
    resource_request,
)


def _environment_overrides(
    profiles: dict[str, Any],
    profile_name: str,
    rendered_environment: Mapping[str, str],
    requested_overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    profile = resolve_runner_profile(profiles, profile_name)
    keys = set((profile.get("env") or {}).keys())
    keys.update((requested_overrides or {}).keys())
    return {
        key: rendered_environment[key]
        for key in keys
        if key in rendered_environment
    }


def launch_nwchem_with_service(
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
    preview = legacy_launch_nwchem_run(
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

    profiles = load_runner_profiles(profiles_path)
    rendered = render_calculation_run(
        input_path=input_path,
        profile=profile,
        profiles=profiles,
        job_name=job_name,
        resource_overrides=resource_overrides,
        env_overrides=env_overrides,
    )
    input_file = Path(preview["input_file"]).resolve()
    working_directory = Path(preview["working_directory"]).resolve()
    if working_directory != input_file.parent:
        raise ValueError(
            "typed NWChem execution currently requires the profile working "
            "directory to match the input directory"
        )
    adapted = adapt_legacy_nwchem_profile(
        profiles,
        profile,
        allowed_work_roots=(working_directory,),
    )
    if adapted.target.executor == "slurm" and not write_script:
        raise ValueError(
            "write_script=False is not supported for typed Slurm execution"
        )
    plan = build_nwchem_launch_plan(
        input_file,
        resource_request(preview["resources"]),
        job_name=preview["job_name"],
        output_template=adapted.output_template,
        error_template=adapted.error_template,
        environment=_environment_overrides(
            profiles,
            profile,
            rendered["environment"],
            env_overrides,
        ),
    )

    service.require("launch", adapted.target)
    service.render(plan, adapted.target)
    archived = archive_previous_outputs(
        str(plan.working_directory),
        plan.job_name,
    )
    launched = service.launch(plan, adapted.target)
    if archived:
        preview["archived_previous_outputs"] = archived
    return apply_legacy_launch_result(
        preview,
        launched,
        timeout_error="sbatch/qsub timed out after 60 seconds",
    )


def register_nwchem_launch_with_service(
    service: ExecutionService,
    *,
    launch_id: str,
    job_name: str,
    input_file: str,
    profile: str,
    campaign_id: int | None = None,
    workflow_id: int | None = None,
    workflow_step_id: str | None = None,
    parent_run_id: int | None = None,
) -> dict[str, Any]:
    record = service.get_launch_record(launch_id)
    if record.instance_id != service.instance_id:
        raise ValueError(
            f"launch {launch_id!r} belongs to another service instance"
        )
    if record.program != "nwchem":
        raise ValueError(
            f"launch {launch_id!r} is for {record.program!r}, not NWChem"
        )
    resolved_input = Path(input_file).resolve()
    if (
        resolved_input.parent != record.working_directory
        or not record.argv
        or record.argv[-1] != resolved_input.name
    ):
        raise ValueError(
            "NWChem registration input does not match the recorded launch"
        )
    return register_run(
        job_name=job_name,
        input_file=str(resolved_input),
        output_file=(
            str(record.stdout_path)
            if record.stdout_path is not None
            else None
        ),
        profile=profile,
        campaign_id=campaign_id,
        workflow_id=workflow_id,
        workflow_step_id=workflow_step_id,
        parent_run_id=parent_run_id,
        mpi_ranks=record.resources.mpi_ranks,
        program="nwchem",
        launch_id=record.launch_id,
        db_path=(
            str(service.registry_db_path)
            if service.registry_db_path is not None
            else None
        ),
    )


def _file_observation(
    record: ExecutionLaunchRecord,
    run_uid: str,
    *,
    role_name: str,
    path: Path,
    roles: frozenset[ArtifactRole],
    kind: str,
) -> tuple[ArtifactRef, ArtifactObservation]:
    artifact_id = f"{record.launch_id}:{role_name}"
    location = ArtifactLocation(path=path, entry_type="file")
    exists = path.is_file()
    size_bytes = None
    modified_ns = None
    sha256 = None
    hash_status = "not_requested"
    if exists:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            hasher = hashlib.sha256()
            while chunk := handle.read(1024 * 1024):
                hasher.update(chunk)
            digest = hasher.hexdigest()
            after = os.fstat(handle.fileno())
        size_bytes = after.st_size
        modified_ns = after.st_mtime_ns
        if (
            before.st_size == after.st_size
            and before.st_mtime_ns == after.st_mtime_ns
        ):
            sha256 = digest
            hash_status = "verified"
        else:
            hash_status = "unavailable"
    artifact = ArtifactRef(
        artifact_id=artifact_id,
        roles=roles,
        kind=kind,
        producing_step=StepRef(
            run_uid=run_uid,
            step_id="execution",
        ),
        metadata={"launch_id": record.launch_id},
    )
    observation = ArtifactObservation(
        observation_id=f"{artifact_id}:terminal",
        artifact_id=artifact_id,
        observed_at=record.updated_at,
        location=location,
        exists=exists,
        size_bytes=size_bytes,
        modified_ns=modified_ns,
        sha256=sha256,
        hash_status=hash_status,
    )
    return artifact, observation


def _record_terminal_nwchem_artifacts(
    record: ExecutionLaunchRecord,
    run_uid: str,
    db_path: str | Path | None,
) -> None:
    current = load_run_artifacts(run_uid, db_path)
    if current is None:
        raise ValueError(f"linked run {run_uid!r} is not registered")
    existing_ids = {
        artifact.artifact_id
        for artifact in current.artifacts
    }
    additions = []
    if record.stdout_path is not None:
        additions.append(_file_observation(
            record,
            run_uid,
            role_name="stdout",
            path=record.stdout_path,
            roles=frozenset({
                ArtifactRole.PRIMARY_OUTPUT,
                ArtifactRole.STDOUT,
            }),
            kind="nwchem.output",
        ))
    if record.stderr_path is not None:
        additions.append(_file_observation(
            record,
            run_uid,
            role_name="stderr",
            path=record.stderr_path,
            roles=frozenset({ArtifactRole.STDERR}),
            kind="nwchem.error",
        ))
    additions = [
        pair
        for pair in additions
        if pair[0].artifact_id not in existing_ids
    ]
    if not additions:
        return
    record_run_artifacts(
        RunArtifacts(
            run_uid=run_uid,
            artifacts=(
                *current.artifacts,
                *(artifact for artifact, _ in additions),
            ),
            observations=(
                *current.observations,
                *(observation for _, observation in additions),
            ),
            expectations=current.expectations,
            provenance=current.provenance,
        ),
        db_path,
    )


def refresh_nwchem_local_status_with_service(
    service: ExecutionService,
    process_id: int,
) -> RecordedLocalStatus | None:
    """Refresh an owned local launch and synchronize its linked NWChem run."""
    try:
        recorded = service.refresh_local_status_external(process_id)
    except LaunchStatusError as exc:
        if exc.as_dict()["error"] == "launch_not_owned":
            return None
        raise
    record = recorded.record
    if record.program != "nwchem":
        raise ValueError(
            f"local process {process_id} belongs to {record.program!r}, "
            "not NWChem"
        )
    try:
        link = load_execution_run_link(
            record.launch_id,
            service.registry_db_path,
        )
    except UnknownExecutionRunLinkError:
        return recorded


    _synchronize_linked_nwchem_run(
        service,
        record,
        link.run_uid,
        (
            "running"
            if recorded.result.status == "running"
            else recorded.result.status
        ),
    )
    return recorded


def _synchronize_linked_nwchem_run(
    service: ExecutionService,
    record: ExecutionLaunchRecord,
    run_uid: str,
    run_status: str,
) -> None:
    db_path = (
        str(service.registry_db_path)
        if service.registry_db_path is not None
        else None
    )
    run = get_run_summary(
        run_uid=run_uid,
        db_path=db_path,
    )
    if run is None:
        raise ValueError(f"linked run {run_uid!r} is not registered")
    if run["status"] != run_status:
        update_run_status(
            run_id=run["id"],
            status=run_status,
            walltime_used_sec=record.elapsed_seconds,
            output_file=(
                str(record.stdout_path)
                if record.stdout_path is not None
                else None
            ),
            db_path=db_path,
        )
    if record.status in (
        "completed",
        "failed",
        "timed_out",
        "cancelled",
    ):
        _record_terminal_nwchem_artifacts(
            record,
            run_uid,
            service.registry_db_path,
        )


def refresh_nwchem_slurm_status_with_service(
    service: ExecutionService,
    job_id: str,
    *,
    profile: str | None = None,
) -> RecordedSlurmStatus | None:
    """Refresh an owned Slurm launch and synchronize its linked NWChem run."""
    try:
        recorded = service.refresh_slurm_status_external(
            job_id,
            target_name=profile,
        )
    except LaunchStatusError as exc:
        if exc.as_dict()["error"] == "launch_not_owned":
            return None
        raise
    record = recorded.record
    if record.program != "nwchem":
        raise ValueError(
            f"Slurm job {job_id} belongs to {record.program!r}, not NWChem"
        )
    try:
        link = load_execution_run_link(
            record.launch_id,
            service.registry_db_path,
        )
    except UnknownExecutionRunLinkError:
        return recorded

    run_status = {
        "queued": "queued",
        "running": "running",
        "suspended": "suspended",
        "completing": "running",
        "completed": "completed",
        "failed": "failed",
        "timed_out": "timelimited",
        "out_of_memory": "oom",
        "cancelled": "cancelled",
    }.get(recorded.result.status)
    if run_status is not None:
        _synchronize_linked_nwchem_run(
            service,
            record,
            link.run_uid,
            run_status,
        )
    return recorded


def _normalized_signal(signal_name: str) -> str:
    normalized = signal_name.strip().lower()
    if normalized in {"term", "sigterm", "terminate"}:
        return "SIGTERM"
    if normalized in {"kill", "sigkill"}:
        return "SIGKILL"
    raise ValueError("signal_name must be one of: term, kill")


def terminate_nwchem_with_service(
    service: ExecutionService,
    *,
    process_id: int | None = None,
    signal_name: str = "term",
    job_id: str | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    if job_id is not None:
        if not profile:
            raise ValueError(
                "profile is required when cancelling a scheduler job by job_id"
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

    if process_id is None:
        raise ValueError(
            "Either process_id (local) or job_id + profile (HPC) "
            "must be provided"
        )
    signal = _normalized_signal(signal_name)
    try:
        cancelled = service.cancel_external(
            process_id=process_id,
            signal_name=signal_name,
        )
    except LaunchCancellationError as exc:
        return {
            "process_id": process_id,
            "signal": signal,
            "sent": False,
            "error": exc.as_dict()["error"],
        }
    result = cancelled.result
    if not isinstance(result, LocalCancellationResult):
        raise TypeError("recorded launch is not a local process")
    return {
        "process_id": process_id,
        "signal": result.signal,
        "sent": result.status == "cancelled",
        "error": result.error,
        "launch_id": cancelled.record.launch_id,
    }


__all__ = [
    "launch_nwchem_with_service",
    "refresh_nwchem_local_status_with_service",
    "refresh_nwchem_slurm_status_with_service",
    "register_nwchem_launch_with_service",
    "terminate_nwchem_with_service",
]

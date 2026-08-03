"""Shared command rendering, path containment, and file staging.

Local and Slurm adapters use these helpers without sharing process-control
state or scheduler behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from typing import Mapping

from chemtools.core.artifacts import ArtifactRole
from chemtools.core.execution import (
    ExecutionTarget,
    LaunchPlan,
    RenderedCommand,
    StagedFile,
)


class WorkRootViolation(ValueError):
    pass


@dataclass(frozen=True)
class _ResolvedStagedFile:
    source: Path
    destination: Path
    mode: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _context(plan: LaunchPlan) -> dict[str, str | int]:
    resources = plan.resources
    return {
        "job_name": plan.job_name,
        "working_directory": str(plan.working_directory),
        "nodes": resources.nodes,
        "mpi_ranks": resources.mpi_ranks,
        "omp_threads": resources.omp_threads,
        "memory_mb_per_node": resources.memory_mb_per_node or "",
        "walltime": resources.walltime or "",
        "partition": resources.partition or "",
        "account": resources.account or "",
    }


def _format(value: str, context: Mapping[str, str | int]) -> str:
    try:
        return value.format_map(context)
    except KeyError as exc:
        raise ValueError(
            f"unknown execution placeholder: {exc.args[0]}"
        ) from exc


def _resolve_under_target(
    path: Path,
    target: ExecutionTarget,
) -> Path:
    candidate = path.resolve(strict=False)
    for root in target.allowed_work_roots:
        resolved_root = root.resolve(strict=False)
        if candidate == resolved_root or resolved_root in candidate.parents:
            return candidate
    raise WorkRootViolation(
        f"path {str(candidate)!r} is outside target {target.name!r} roots"
    )


def _expected_path(
    plan: LaunchPlan,
    target: ExecutionTarget,
    role: ArtifactRole,
) -> Path | None:
    matches = [
        expectation.location.path
        for expectation in plan.expected_artifacts
        if role in expectation.roles
    ]
    if len(matches) > 1:
        raise ValueError(
            f"launch plan has multiple {role.value} artifact locations"
        )
    if not matches:
        return None
    path = matches[0]
    if not path.is_absolute():
        path = plan.working_directory / path
    return _resolve_under_target(path, target)


def _staging_path(path: Path, working_directory: Path) -> Path:
    if path.is_absolute():
        return path
    return working_directory / path


def _resolve_staged_files(
    staged_files: tuple[StagedFile, ...],
    target: ExecutionTarget,
) -> tuple[_ResolvedStagedFile, ...]:
    resolved: list[_ResolvedStagedFile] = []
    for staged_file in staged_files:
        requested_source = staged_file.source
        try:
            source = requested_source.resolve(strict=True)
        except FileNotFoundError:
            if staged_file.required:
                raise FileNotFoundError(
                    f"required staged source does not exist: "
                    f"{requested_source}"
                ) from None
            continue
        _resolve_under_target(source, target)
        if not source.is_file():
            raise ValueError(
                f"staged source is not a regular file: {source}"
            )

        requested_destination = staged_file.destination
        if os.path.lexists(requested_destination):
            raise FileExistsError(
                f"refusing to overwrite staged destination: "
                f"{requested_destination}"
            )
        destination = _resolve_under_target(
            requested_destination,
            target,
        )
        if not destination.parent.is_dir():
            raise ValueError(
                f"staged destination parent does not exist: "
                f"{destination.parent}"
            )
        resolved.append(_ResolvedStagedFile(
            source=source,
            destination=destination,
            mode=staged_file.mode,
        ))
    return tuple(resolved)


def _render_staged_files(
    plan: LaunchPlan,
    target: ExecutionTarget,
    working_directory: Path,
) -> tuple[StagedFile, ...]:
    rendered = []
    destinations: set[Path] = set()
    for staged_file in plan.staged_files:
        source = _resolve_under_target(
            _staging_path(staged_file.source, working_directory),
            target,
        )
        destination = _resolve_under_target(
            _staging_path(staged_file.destination, working_directory),
            target,
        )
        if destination in destinations:
            raise ValueError(
                f"multiple staged files use destination: {destination}"
            )
        destinations.add(destination)
        rendered.append(StagedFile(
            source=source,
            destination=destination,
            mode=staged_file.mode,
            required=staged_file.required,
        ))
    return tuple(rendered)


def _stage_files(
    staged_files: tuple[_ResolvedStagedFile, ...],
    target: ExecutionTarget,
) -> None:
    created: list[Path] = []
    try:
        for staged_file in staged_files:
            destination = staged_file.destination
            if staged_file.mode == "copy":
                with staged_file.source.open("rb") as source_handle:
                    with destination.open("xb") as destination_handle:
                        created.append(destination)
                        shutil.copyfileobj(
                            source_handle,
                            destination_handle,
                        )
            else:
                destination.symlink_to(staged_file.source)
                created.append(destination)
            _resolve_under_target(destination, target)
    except Exception:
        for destination in reversed(created):
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
        raise


def _reject_staging_output_conflicts(
    staged_files: tuple[_ResolvedStagedFile, ...],
    output_paths: tuple[Path | None, ...],
) -> None:
    reserved = {
        path.resolve(strict=False)
        for path in output_paths
        if path is not None
    }
    for staged_file in staged_files:
        if staged_file.destination in reserved:
            raise ValueError(
                "staged destination conflicts with launch output: "
                f"{staged_file.destination}"
            )


def _render_command(
    plan: LaunchPlan,
    target: ExecutionTarget,
) -> RenderedCommand:
    if plan.program not in target.programs:
        raise ValueError(
            f"target {target.name!r} has no {plan.program!r} installation"
        )
    working_directory = _resolve_under_target(
        plan.working_directory,
        target,
    )
    staged_files = _render_staged_files(
        plan,
        target,
        working_directory,
    )

    context = _context(plan)
    installation = target.programs[plan.program]
    if plan.entrypoint is None:
        executable = installation.executable_argv
    else:
        try:
            executable = installation.entrypoints[plan.entrypoint]
        except KeyError as exc:
            raise ValueError(
                f"target {target.name!r} has no {plan.program!r} "
                f"entrypoint {plan.entrypoint!r}"
            ) from exc
    argv = tuple(
        _format(value, context)
        for value in (
            *installation.launcher_argv,
            *executable,
            *plan.program_arguments,
        )
    )
    environment = {
        key: _format(value, context)
        for key, value in installation.environment.items()
    }
    environment.update({
        key: _format(value, context)
        for key, value in plan.environment.items()
    })
    return RenderedCommand(
        target=target.name,
        program=plan.program,
        executor=target.executor,
        argv=argv,
        environment=environment,
        working_directory=working_directory,
        stdout_path=_expected_path(plan, target, ArtifactRole.STDOUT),
        stderr_path=_expected_path(plan, target, ArtifactRole.STDERR),
        staged_files=staged_files,
        stdin_text=plan.stdin_text,
        timeout_seconds=plan.timeout_seconds,
    )


__all__ = ["WorkRootViolation"]

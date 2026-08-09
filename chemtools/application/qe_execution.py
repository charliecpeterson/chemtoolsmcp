"""Launch Quantum ESPRESSO pw.x through the shared typed execution service."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from chemtools.application.execution import ExecutionService
from chemtools.application.legacy_execution import apply_legacy_launch_result
from chemtools.core.execution import RenderedSlurmScript
from chemtools.execution.legacy_archive import archive_paths
from chemtools.execution.legacy_runner import render_calculation_run
from chemtools.execution.profiles import load_runner_profiles, resource_request
from chemtools.programs.qe.launch import (
    adapt_legacy_qe_profile,
    build_qe_launch_plan,
)


def render_qe_launch(
    *,
    input_path: str,
    profile: str,
    profiles_path: str | None = None,
    job_name: str | None = None,
    resource_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, str] | None = None,
) -> tuple[dict[str, Any], Any]:
    preview = render_calculation_run(
        input_path,
        profile,
        profiles_path=profiles_path,
        job_name=job_name,
        resource_overrides=resource_overrides,
        env_overrides=env_overrides,
    )
    input_file = Path(preview["input_file"]).resolve()
    working_directory = Path(preview["working_directory"]).resolve()
    if working_directory != input_file.parent:
        raise ValueError(
            "typed QE execution requires the input directory as the working "
            "directory"
        )
    profiles = load_runner_profiles(profiles_path)
    adapted = adapt_legacy_qe_profile(
        profiles,
        profile,
        allowed_work_roots=(working_directory,),
    )
    plan = build_qe_launch_plan(
        input_file,
        resource_request(preview["resources"]),
        job_name=preview["job_name"],
        output_template=adapted.output_template,
        error_template=adapted.error_template,
        environment=env_overrides,
    )
    rendered = ExecutionService().render(plan, adapted.target)
    command = rendered.command if isinstance(rendered, RenderedSlurmScript) else rendered
    preview["command"] = (
        f"{shlex.join(command.argv)} > "
        f"{shlex.quote(str(command.stdout_path))} 2> "
        f"{shlex.quote(str(command.stderr_path))}"
    )
    configured_keys = {
        *adapted.target.programs["qe"].environment,
        *(env_overrides or {}),
    }
    preview["environment"] = {
        key: command.environment[key]
        for key in configured_keys
        if key in command.environment
    }
    return preview, adapted


def launch_qe_with_service(
    service: ExecutionService,
    *,
    input_path: str,
    profile: str,
    profiles_path: str | None = None,
    job_name: str | None = None,
    resource_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    preview, adapted = render_qe_launch(
        input_path=input_path,
        profile=profile,
        profiles_path=profiles_path,
        job_name=job_name,
        resource_overrides=resource_overrides,
        env_overrides=env_overrides,
    )
    if dry_run:
        return preview
    input_file = Path(preview["input_file"]).resolve()
    plan = build_qe_launch_plan(
        input_file,
        resource_request(preview["resources"]),
        job_name=preview["job_name"],
        output_template=adapted.output_template,
        error_template=adapted.error_template,
        environment=env_overrides,
    )
    service.require("launch", adapted.target)
    rendered = service.render(plan, adapted.target)
    command = rendered.command if isinstance(rendered, RenderedSlurmScript) else rendered
    script_path = rendered.script_path if isinstance(rendered, RenderedSlurmScript) else None
    archived = archive_paths(
        path
        for path in (command.stdout_path, command.stderr_path, script_path)
        if path is not None
    )
    launched = service.launch(plan, adapted.target)
    if archived:
        preview["archived_previous_outputs"] = archived
    response = apply_legacy_launch_result(preview, launched)
    if adapted.target.executor == "local":
        response["command"] = (
            f"{shlex.join(launched.record.argv)} > "
            f"{shlex.quote(str(launched.record.stdout_path))} 2> "
            f"{shlex.quote(str(launched.record.stderr_path))}"
        )
    return response


__all__ = ["launch_qe_with_service", "render_qe_launch"]

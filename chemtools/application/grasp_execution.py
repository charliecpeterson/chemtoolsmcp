"""Translate GRASP MCP execution calls to typed execution services."""

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
from chemtools.programs.grasp.launch import (
    adapt_legacy_grasp_profile,
    build_grasp_interactive_launch_plan,
    build_grasp_interactive_target,
    build_grasp_workflow_launch_plan,
)
from chemtools.programs.grasp.runtime import (
    append_execution_to_session,
)
from chemtools.programs.grasp.scheduler import (
    launch_grasp_workflow_run as legacy_launch_grasp_workflow_run,
)
from chemtools.programs.grasp.strategy.runner import (
    run_workflow as legacy_run_workflow,
)


def launch_grasp_workflow_with_service(
    service: ExecutionService,
    *,
    workflow_script_path: str,
    profile: str,
    profiles_path: str | None = None,
    job_name: str | None = None,
    resource_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, str] | None = None,
    write_script: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    preview = legacy_launch_grasp_workflow_run(
        workflow_script_path=workflow_script_path,
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

    workflow_script = Path(preview["input_file"]).resolve()
    working_directory = Path(preview["working_directory"]).resolve()
    if working_directory != workflow_script.parent:
        raise ValueError(
            "typed GRASP execution currently requires the profile working "
            "directory to match the workflow script directory"
        )
    profiles = load_runner_profiles(profiles_path)
    adapted = adapt_legacy_grasp_profile(
        profiles,
        profile,
        allowed_work_roots=(working_directory,),
    )
    if adapted.target.executor == "slurm" and not write_script:
        raise ValueError(
            "write_script=False is not supported for typed Slurm execution"
        )
    plan = build_grasp_workflow_launch_plan(
        workflow_script,
        resource_request(preview["resources"]),
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

    preview["workflow_script_path"] = str(workflow_script)
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


def terminate_grasp_with_service(
    service: ExecutionService,
    *,
    job_id: str,
    profile: str,
) -> dict[str, Any]:
    if not job_id:
        raise ValueError(
            "job_id is required to cancel a GRASP scheduler job"
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


def run_grasp_exe_with_service(
    service: ExecutionService,
    exe: str,
    *,
    working_dir: str,
    stdin_lines: list[str] | str,
    args: list[str] | None = None,
    timeout_seconds: float = 600.0,
    log_to_session: bool = True,
    capture_log_file: str | None = None,
    container: str | None = None,
) -> dict[str, Any]:
    work = Path(working_dir).resolve()
    target = build_grasp_interactive_target(
        work,
        container=container,
    )
    service.require("launch", target)
    capture_path = None
    if capture_log_file:
        capture_path = (work / capture_log_file).resolve()
        if capture_path == work or work not in capture_path.parents:
            raise ValueError(
                "GRASP capture log must be a file inside the working "
                "directory"
            )
    work.mkdir(parents=True, exist_ok=True)
    if capture_path is not None:
        if not capture_path.parent.is_dir():
            raise ValueError(
                "GRASP capture log parent directory does not exist"
            )
        if capture_path.exists() and not capture_path.is_file():
            raise ValueError(
                "GRASP capture log path is not a regular file"
            )
    plan = build_grasp_interactive_launch_plan(
        exe,
        working_directory=work,
        stdin_lines=stdin_lines,
        args=args,
        timeout_seconds=timeout_seconds,
    )
    recorded = service.run_to_completion(plan, target)
    result = recorded.result
    log_path = None
    if capture_path is not None:
        capture_path.write_text(
            result.stdout
            + (
                "\n--- STDERR ---\n" + result.stderr
                if result.stderr
                else ""
            ),
            encoding="utf-8",
        )
        log_path = str(capture_path)

    command = shlex.join(recorded.record.argv)
    if log_to_session:
        append_execution_to_session(
            work=work,
            exe=exe,
            command=command,
            stdin_text=plan.stdin_text or "",
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.return_code,
            elapsed_seconds=result.elapsed_seconds,
            timed_out=result.status == "timed_out",
            log_file=log_path,
        )
    return {
        "exe": exe,
        "returncode": result.return_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command,
        "working_dir": str(work),
        "log_file": log_path,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "timed_out": result.status == "timed_out",
        "ok": result.status == "completed",
        "launch_id": recorded.record.launch_id,
        "execution_instance_id": recorded.record.instance_id,
    }


def run_grasp_workflow_with_service(
    service: ExecutionService,
    plan: dict[str, Any],
    *,
    working_dir: str,
    stop_on_failure: bool = True,
    timeout_per_step: float = 600.0,
    container: str | None = None,
) -> dict[str, Any]:
    target = build_grasp_interactive_target(
        working_dir,
        container=container,
    )
    service.require("launch", target)

    def run_step(exe: str, **kwargs: Any) -> dict[str, Any]:
        return run_grasp_exe_with_service(
            service,
            exe,
            **kwargs,
        )

    return legacy_run_workflow(
        plan,
        working_dir=working_dir,
        stop_on_failure=stop_on_failure,
        timeout_per_step=timeout_per_step,
        container=container,
        run_step=run_step,
    )


__all__ = [
    "launch_grasp_workflow_with_service",
    "run_grasp_exe_with_service",
    "run_grasp_workflow_with_service",
    "terminate_grasp_with_service",
]

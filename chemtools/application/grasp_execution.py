"""Translate GRASP MCP execution calls to typed execution services."""

from __future__ import annotations

from pathlib import Path
import shlex
from typing import Any

from chemtools.application.execution import ExecutionService
from chemtools.programs.grasp.launch import (
    build_grasp_interactive_launch_plan,
    build_grasp_interactive_target,
)
from chemtools.programs.grasp.runtime import (
    append_execution_to_session,
)
from chemtools.programs.grasp.strategy.runner import (
    run_workflow as legacy_run_workflow,
)


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
    "run_grasp_exe_with_service",
    "run_grasp_workflow_with_service",
]

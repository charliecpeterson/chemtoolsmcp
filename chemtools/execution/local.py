"""Local process execution with retained handles for status and cancellation."""

from __future__ import annotations

import os
import subprocess
import time

from chemtools.core.execution import (
    ExecutionTarget,
    LaunchPlan,
    LocalCancellationResult,
    LocalLaunchResult,
    LocalStatusResult,
    LocalSynchronousResult,
    RenderedCommand,
)
from chemtools.execution._common import (
    _reject_staging_output_conflicts,
    _render_command,
    _resolve_staged_files,
    _stage_files,
    _utc_now,
)


def _timeout_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


class LocalExecutor:
    def __init__(self) -> None:
        self._processes: dict[int, subprocess.Popen] = {}

    def render(
        self,
        plan: LaunchPlan,
        target: ExecutionTarget,
    ) -> RenderedCommand:
        if target.executor != "local":
            raise ValueError("LocalExecutor requires a local target")
        return _render_command(plan, target)

    def launch(
        self,
        plan: LaunchPlan,
        target: ExecutionTarget,
    ) -> LocalLaunchResult:
        command = self.render(plan, target)
        if not command.working_directory.is_dir():
            raise ValueError(
                "launch working directory does not exist: "
                f"{command.working_directory}"
            )
        if command.stdout_path is None or command.stderr_path is None:
            raise ValueError(
                "local launch requires stdout and stderr artifact paths"
            )
        if command.stdout_path == command.stderr_path:
            raise ValueError(
                "local launch requires distinct stdout and stderr paths"
            )
        for path in (command.stdout_path, command.stderr_path):
            if path.exists():
                raise FileExistsError(
                    f"refusing to overwrite launch output: {path}"
                )
        staged_files = _resolve_staged_files(
            command.staged_files,
            target,
        )
        _reject_staging_output_conflicts(
            staged_files,
            (command.stdout_path, command.stderr_path),
        )
        _stage_files(staged_files, target)

        environment = dict(os.environ)
        environment.update(command.environment)
        with command.stdout_path.open("xb") as stdout_handle:
            with command.stderr_path.open("xb") as stderr_handle:
                process = subprocess.Popen(
                    command.argv,
                    cwd=command.working_directory,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    shell=False,
                )
        self._processes[process.pid] = process
        return LocalLaunchResult(
            command=command,
            process_id=process.pid,
            status="started",
            started_at=_utc_now(),
        )

    def status(self, process_id: int) -> LocalStatusResult:
        process = self._processes.get(process_id)
        if process is None:
            raise LookupError(f"local process {process_id} is not owned")
        return_code = process.poll()
        if return_code is None:
            status = "running"
        elif return_code == 0:
            status = "completed"
        else:
            status = "failed"
        return LocalStatusResult(
            process_id=process_id,
            status=status,
            return_code=return_code,
            checked_at=_utc_now(),
        )

    def run_to_completion(
        self,
        plan: LaunchPlan,
        target: ExecutionTarget,
    ) -> LocalSynchronousResult:
        command = self.render(plan, target)
        if not command.working_directory.is_dir():
            raise ValueError(
                "run working directory does not exist: "
                f"{command.working_directory}"
            )
        if (
            command.stdout_path is not None
            or command.stderr_path is not None
        ):
            raise ValueError(
                "synchronous captured execution requires no stdout or "
                "stderr artifact paths"
            )
        staged_files = _resolve_staged_files(
            command.staged_files,
            target,
        )
        _reject_staging_output_conflicts(
            staged_files,
            (command.stdout_path, command.stderr_path),
        )
        _stage_files(staged_files, target)

        environment = dict(os.environ)
        environment.update(command.environment)
        started_at = _utc_now()
        start = time.monotonic()
        run_kwargs = {
            "cwd": command.working_directory,
            "env": environment,
            "capture_output": True,
            "text": True,
            "shell": False,
            "check": False,
            "timeout": command.timeout_seconds,
        }
        if command.stdin_text is None:
            run_kwargs["stdin"] = subprocess.DEVNULL
        else:
            run_kwargs["input"] = command.stdin_text
        try:
            completed = subprocess.run(
                command.argv,
                **run_kwargs,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            return_code = completed.returncode
            status = "completed" if return_code == 0 else "failed"
        except subprocess.TimeoutExpired as exc:
            stdout = _timeout_text(exc.stdout)
            stderr = _timeout_text(exc.stderr)
            return_code = -1
            status = "timed_out"
        elapsed_seconds = time.monotonic() - start
        return LocalSynchronousResult(
            command=command,
            status=status,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            started_at=started_at,
            completed_at=_utc_now(),
            elapsed_seconds=elapsed_seconds,
        )

    def cancel(
        self,
        process_id: int,
        signal_name: str = "term",
    ) -> LocalCancellationResult:
        normalized = signal_name.strip().lower()
        if normalized in {"term", "sigterm", "terminate"}:
            signal = "SIGTERM"
        elif normalized in {"kill", "sigkill"}:
            signal = "SIGKILL"
        else:
            raise ValueError("signal_name must be one of: term, kill")
        process = self._processes.get(process_id)
        if process is None:
            return LocalCancellationResult(
                process_id=process_id,
                status="cancel_failed",
                signal=signal,
                error="process_not_owned",
                cancelled_at=_utc_now(),
            )
        if process.poll() is not None:
            self._processes.pop(process_id, None)
            return LocalCancellationResult(
                process_id=process_id,
                status="cancel_failed",
                signal=signal,
                error="process_exited",
                cancelled_at=_utc_now(),
            )
        try:
            if signal == "SIGTERM":
                process.terminate()
            else:
                process.kill()
        except ProcessLookupError:
            self._processes.pop(process_id, None)
            return LocalCancellationResult(
                process_id=process_id,
                status="cancel_failed",
                signal=signal,
                error="process_not_found",
                cancelled_at=_utc_now(),
            )
        except PermissionError:
            return LocalCancellationResult(
                process_id=process_id,
                status="cancel_failed",
                signal=signal,
                error="permission_denied",
                cancelled_at=_utc_now(),
            )
        self._processes.pop(process_id, None)
        return LocalCancellationResult(
            process_id=process_id,
            status="cancelled",
            signal=signal,
            error=None,
            cancelled_at=_utc_now(),
        )


__all__ = ["LocalExecutor"]

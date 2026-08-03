"""Slurm script rendering, submission, and cancellation."""

from __future__ import annotations

import os
import re
import shlex
import subprocess

from chemtools.core.execution import (
    ExecutionTarget,
    LaunchPlan,
    RenderedSlurmScript,
    SlurmCancellationResult,
    SlurmStatusResult,
    SlurmSubmissionResult,
)
from chemtools.execution._common import (
    _context,
    _format,
    _reject_staging_output_conflicts,
    _render_command,
    _resolve_staged_files,
    _resolve_under_target,
    _stage_files,
    _utc_now,
)


_SLURM_STATE_MAP = {
    "PENDING": "queued",
    "CONFIGURING": "queued",
    "RUNNING": "running",
    "SUSPENDED": "suspended",
    "COMPLETING": "completing",
    "COMPLETED": "completed",
    "CANCELLED": "cancelled",
    "TIMEOUT": "timed_out",
    "OUT_OF_MEMORY": "out_of_memory",
    "BOOT_FAIL": "failed",
    "DEADLINE": "failed",
    "FAILED": "failed",
    "LAUNCH_FAILED": "failed",
    "NODE_FAIL": "failed",
    "PREEMPTED": "failed",
}
_ACTIVE_SLURM_STATUSES = frozenset({
    "queued",
    "running",
    "suspended",
    "completing",
})


def _state_token(value: str) -> str:
    return value.strip().upper().rstrip("+").split()[0]


def _first_line(value: str) -> str | None:
    for line in value.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _status_argv(
    template: tuple[str, ...],
    job_id: str,
) -> tuple[str, ...]:
    return tuple(
        _format(value, {"job_id": job_id})
        for value in template
    )


def _query_failed(
    job_id: str,
    argv: tuple[str, ...],
    source: str,
    *,
    return_code: int | None,
    stdout: str,
    stderr: str,
    error: str,
) -> SlurmStatusResult:
    return SlurmStatusResult(
        job_id=job_id,
        query_argv=argv,
        source=source,
        status="query_failed",
        raw_state=None,
        query_return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        checked_at=_utc_now(),
        error=error,
    )


class SlurmExecutor:
    def render(
        self,
        plan: LaunchPlan,
        target: ExecutionTarget,
    ) -> RenderedSlurmScript:
        if target.executor != "slurm" or target.scheduler is None:
            raise ValueError("SlurmExecutor requires a slurm target")
        command = _render_command(plan, target)
        resources = plan.resources
        context = _context(plan)
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={plan.job_name}",
            f"#SBATCH --nodes={resources.nodes}",
            f"#SBATCH --ntasks={resources.mpi_ranks}",
            f"#SBATCH --cpus-per-task={resources.omp_threads}",
        ]
        if command.stdout_path is not None:
            lines.append(f"#SBATCH --output={command.stdout_path}")
        if command.stderr_path is not None:
            lines.append(f"#SBATCH --error={command.stderr_path}")
        if resources.walltime is not None:
            lines.append(f"#SBATCH --time={resources.walltime}")
        if resources.partition is not None:
            lines.append(f"#SBATCH --partition={resources.partition}")
        if resources.account is not None:
            lines.append(f"#SBATCH --account={resources.account}")
        if resources.memory_mb_per_node is not None:
            lines.append(f"#SBATCH --mem={resources.memory_mb_per_node}M")

        installation = target.programs[plan.program]
        lines.extend(installation.setup_lines)
        lines.extend(
            f"export {key}={shlex.quote(value)}"
            for key, value in command.environment.items()
        )
        lines.append(
            f"cd -- {shlex.quote(str(command.working_directory))}"
        )
        lines.extend(
            _format(line, context)
            for line in installation.pre_run_lines
        )
        lines.append(
            " ".join(shlex.quote(value) for value in command.argv)
        )
        script_text = "\n".join(lines) + "\n"
        script_path = _resolve_under_target(
            command.working_directory
            / f"{plan.job_name}{target.scheduler.script_suffix}",
            target,
        )
        submission_context = {
            **context,
            "script_file": str(script_path),
        }
        submit_argv = tuple(
            _format(value, submission_context)
            for value in target.scheduler.submit_argv
        )
        return RenderedSlurmScript(
            command=command,
            script_path=script_path,
            script_text=script_text,
            submit_argv=submit_argv,
        )

    def submit(
        self,
        plan: LaunchPlan,
        target: ExecutionTarget,
    ) -> SlurmSubmissionResult:
        script = self.render(plan, target)
        if not script.command.working_directory.is_dir():
            raise ValueError(
                "launch working directory does not exist: "
                f"{script.command.working_directory}"
            )
        if script.script_path.exists():
            raise FileExistsError(
                f"refusing to overwrite scheduler script: "
                f"{script.script_path}"
            )
        staged_files = _resolve_staged_files(
            script.command.staged_files,
            target,
        )
        _reject_staging_output_conflicts(
            staged_files,
            (
                script.command.stdout_path,
                script.command.stderr_path,
                script.script_path,
            ),
        )
        _stage_files(staged_files, target)

        with script.script_path.open("x", encoding="utf-8") as handle:
            handle.write(script.script_text)
        _resolve_under_target(script.script_path, target)
        environment = dict(os.environ)
        environment.update(script.command.environment)
        try:
            completed = subprocess.run(
                script.submit_argv,
                cwd=script.command.working_directory,
                env=environment,
                capture_output=True,
                text=True,
                shell=False,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return SlurmSubmissionResult(
                script=script,
                status="submit_failed",
                return_code=-1,
                stdout="",
                stderr="scheduler submission timed out after 60 seconds",
                job_id=None,
                submitted_at=_utc_now(),
            )
        job_id = None
        if completed.returncode == 0 and target.scheduler is not None:
            match = re.search(
                target.scheduler.job_id_regex,
                completed.stdout,
            )
            if match is not None:
                job_id = match.group(1)
        if completed.returncode != 0:
            status = "submit_failed"
        elif job_id is None:
            status = "submitted_untracked"
        else:
            status = "submitted"
        return SlurmSubmissionResult(
            script=script,
            status=status,
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            job_id=job_id,
            submitted_at=_utc_now(),
        )

    def status(
        self,
        job_id: str,
        target: ExecutionTarget,
    ) -> SlurmStatusResult:
        if target.executor != "slurm" or target.scheduler is None:
            raise ValueError("SlurmExecutor requires a slurm target")
        queue_argv = _status_argv(
            target.scheduler.status_argv,
            job_id,
        )
        try:
            queued = subprocess.run(
                queue_argv,
                env=dict(os.environ),
                capture_output=True,
                text=True,
                shell=False,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            queued = None
        if queued is not None and queued.returncode == 0:
            queue_line = _first_line(queued.stdout)
            if queue_line is not None:
                raw_state = _state_token(queue_line)
                status = _SLURM_STATE_MAP.get(raw_state, "unknown")
                if (
                    status in _ACTIVE_SLURM_STATUSES
                    or not target.scheduler.accounting_argv
                ):
                    return SlurmStatusResult(
                        job_id=job_id,
                        query_argv=queue_argv,
                        source="queue",
                        status=status,
                        raw_state=raw_state,
                        query_return_code=queued.returncode,
                        stdout=queued.stdout,
                        stderr=queued.stderr,
                        checked_at=_utc_now(),
                    )

        accounting_template = target.scheduler.accounting_argv
        if not accounting_template:
            if queued is None:
                return _query_failed(
                    job_id,
                    queue_argv,
                    "queue",
                    return_code=None,
                    stdout="",
                    stderr="",
                    error="scheduler status query timed out after 30 seconds",
                )
            if queued.returncode != 0:
                return _query_failed(
                    job_id,
                    queue_argv,
                    "queue",
                    return_code=queued.returncode,
                    stdout=queued.stdout,
                    stderr=queued.stderr,
                    error=queued.stderr or "scheduler status query failed",
                )
            return SlurmStatusResult(
                job_id=job_id,
                query_argv=queue_argv,
                source="queue",
                status="not_found",
                raw_state=None,
                query_return_code=queued.returncode,
                stdout=queued.stdout,
                stderr=queued.stderr,
                checked_at=_utc_now(),
            )

        accounting_argv = _status_argv(accounting_template, job_id)
        try:
            accounted = subprocess.run(
                accounting_argv,
                env=dict(os.environ),
                capture_output=True,
                text=True,
                shell=False,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return _query_failed(
                job_id,
                accounting_argv,
                "accounting",
                return_code=None,
                stdout="",
                stderr="",
                error="scheduler accounting query timed out after 30 seconds",
            )
        if accounted.returncode != 0:
            return _query_failed(
                job_id,
                accounting_argv,
                "accounting",
                return_code=accounted.returncode,
                stdout=accounted.stdout,
                stderr=accounted.stderr,
                error=accounted.stderr or "scheduler accounting query failed",
            )
        accounting_line = _first_line(accounted.stdout)
        if accounting_line is None:
            return SlurmStatusResult(
                job_id=job_id,
                query_argv=accounting_argv,
                source="accounting",
                status="not_found",
                raw_state=None,
                query_return_code=accounted.returncode,
                stdout=accounted.stdout,
                stderr=accounted.stderr,
                checked_at=_utc_now(),
            )
        fields = tuple(
            field.strip()
            for field in accounting_line.split("|")
        )
        if not fields[0]:
            return SlurmStatusResult(
                job_id=job_id,
                query_argv=accounting_argv,
                source="accounting",
                status="unknown",
                raw_state=None,
                query_return_code=accounted.returncode,
                stdout=accounted.stdout,
                stderr=accounted.stderr,
                checked_at=_utc_now(),
                error="scheduler accounting state is empty",
            )
        raw_state = _state_token(fields[0])
        status = _SLURM_STATE_MAP.get(raw_state, "unknown")
        job_exit_code = None
        termination_signal = None
        if len(fields) > 1 and ":" in fields[1]:
            exit_code, signal = fields[1].split(":", maxsplit=1)
            if exit_code.isdigit() and signal.isdigit():
                job_exit_code = int(exit_code)
                termination_signal = int(signal)
        elapsed_seconds = None
        if len(fields) > 2 and fields[2].isdigit():
            elapsed_seconds = float(fields[2])
        return SlurmStatusResult(
            job_id=job_id,
            query_argv=accounting_argv,
            source="accounting",
            status=status,
            raw_state=raw_state,
            query_return_code=accounted.returncode,
            stdout=accounted.stdout,
            stderr=accounted.stderr,
            checked_at=_utc_now(),
            job_exit_code=job_exit_code,
            termination_signal=termination_signal,
            elapsed_seconds=elapsed_seconds,
        )

    def cancel(
        self,
        job_id: str,
        target: ExecutionTarget,
    ) -> SlurmCancellationResult:
        if target.executor != "slurm" or target.scheduler is None:
            raise ValueError("SlurmExecutor requires a slurm target")
        argv = tuple(
            _format(value, {"job_id": job_id})
            for value in target.scheduler.cancel_argv
        )
        completed = subprocess.run(
            argv,
            env=dict(os.environ),
            capture_output=True,
            text=True,
            shell=False,
            check=False,
            timeout=30,
        )
        return SlurmCancellationResult(
            job_id=job_id,
            argv=argv,
            status=(
                "cancelled"
                if completed.returncode == 0
                else "cancel_failed"
            ),
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            cancelled_at=_utc_now(),
        )


__all__ = ["SlurmExecutor"]

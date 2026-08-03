"""Project typed execution results into legacy MCP response dictionaries."""

from __future__ import annotations

from typing import Any

from chemtools.core.execution import (
    LocalLaunchResult,
    RecordedCancellation,
    RecordedLaunch,
    SlurmCancellationResult,
    SlurmSubmissionResult,
)


def apply_legacy_launch_result(
    preview: dict[str, Any],
    launched: RecordedLaunch,
    *,
    timeout_error: str | None = None,
) -> dict[str, Any]:
    preview["executed"] = True
    preview["launch_id"] = launched.record.launch_id
    preview["execution_instance_id"] = launched.record.instance_id
    preview["effective_argv"] = list(launched.record.argv)

    if isinstance(launched.result, LocalLaunchResult):
        preview["process_id"] = launched.result.process_id
        preview["status"] = "started"
        return preview
    if not isinstance(launched.result, SlurmSubmissionResult):
        raise TypeError("execution service returned an unknown launch result")

    submission = launched.result
    preview.update({
        "submit_script_path": str(submission.script.script_path),
        "submit_script_text": submission.script.script_text,
        "submit_command": list(submission.script.submit_argv),
        "status": (
            "submitted"
            if submission.status == "submitted_untracked"
            else submission.status
        ),
        "return_code": submission.return_code,
        "stdout": submission.stdout,
        "stderr": (
            timeout_error
            if submission.return_code == -1 and timeout_error is not None
            else submission.stderr
        ),
        "job_id": submission.job_id,
    })
    if submission.job_id is not None:
        jobid_path = submission.script.script_path.with_suffix(".jobid")
        try:
            jobid_path.write_text(submission.job_id, encoding="utf-8")
            preview["jobid_file"] = str(jobid_path)
        except OSError as exc:
            preview["jobid_file_error"] = str(exc)
    return preview


def legacy_slurm_cancellation_result(
    cancelled: RecordedCancellation,
) -> dict[str, Any]:
    result = cancelled.result
    if not isinstance(result, SlurmCancellationResult):
        raise TypeError("recorded launch is not a scheduler job")
    return {
        "job_id": result.job_id,
        "cancelled": result.status == "cancelled",
        "command": list(result.argv),
        "return_code": result.return_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "launch_id": cancelled.record.launch_id,
    }


__all__ = [
    "apply_legacy_launch_result",
    "legacy_slurm_cancellation_result",
]

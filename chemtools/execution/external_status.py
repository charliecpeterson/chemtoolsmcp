"""Inspect files or attach read-only status to an external Slurm job.

Owned launches use the execution service. This module covers calculations
started elsewhere without probing arbitrary local processes or cancelling jobs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
from typing import Any

from chemtools.core.common import read_text
from chemtools.core.monitoring import watch_run_status
from chemtools.execution.profiles import (
    _format_template,
    _resolve_profile,
    load_runner_profiles,
)


UTC = timezone.utc


def inspect_run_status(
    *,
    output_path: str | None = None,
    input_path: str | None = None,
    error_path: str | None = None,
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
    output_status_reader: Any = None,
    progress_summary_fn: Any = None,
) -> dict[str, Any]:
    output_info = _file_info(output_path)
    input_info = _file_info(input_path)
    error_info = _file_info(error_path)
    if (profile is None) != (job_id is None):
        raise ValueError(
            "external Slurm inspection requires both profile and job_id"
        )

    scheduler_status = None
    if profile and job_id:
        scheduler_status = _scheduler_status(
            profile=profile,
            job_id=job_id,
            profiles_path=profiles_path,
        )

    input_raw_text: str | None = None
    if input_path:
        try:
            input_raw_text = read_text(input_path)
        except Exception:  # pragma: no cover
            input_raw_text = None

    parsed_output = None
    compact_summary = None
    progress_summary = None
    task_preview = None
    if output_info["exists"] and output_status_reader is not None:
        try:
            contents = read_text(output_info["path"])
            output_status = output_status_reader(
                contents,
                output_info["path"],
                input_path=input_path,
                input_raw_text=input_raw_text,
                progress_summary_fn=progress_summary_fn,
            )
            parsed_output = output_status.get("parsed_output")
            progress_summary = output_status.get("progress_summary")
            compact_summary = output_status.get("compact_summary")
            task_preview = output_status.get("task_preview")
        except Exception as exc:  # pragma: no cover
            parsed_output = {
                "error": str(exc),
                "incomplete": True,
            }

    overall_status = "unknown"
    scheduler_state = (scheduler_status or {}).get("status")
    outcome = (
        (parsed_output or {}).get("program_summary", {}).get("outcome")
    )
    if scheduler_state == "queued":
        overall_status = "queued"
    elif scheduler_state == "running":
        overall_status = "running"
    elif outcome == "success":
        overall_status = "completed_success"
    elif outcome == "failed":
        overall_status = "completed_failed"
    elif outcome == "incomplete":
        overall_status = "completed_incomplete"
    elif scheduler_state == "failed":
        overall_status = "completed_failed"
    elif scheduler_state == "cancelled":
        overall_status = "cancelled"
    elif error_info["exists"] and error_info["size_bytes"] > 0:
        overall_status = "error_only"
    elif output_info["exists"]:
        overall_status = "output_present_unknown"
    else:
        overall_status = "not_started"

    status = {
        "output_file": output_info,
        "input_file": {
            "path": input_info["path"],
            "exists": input_info["exists"],
        },
        "error_file": error_info,
        "process": {
            "process_id": None,
            "status": "unknown",
        },
        "scheduler": scheduler_status,
        "output_summary": compact_summary,
        "progress_summary": progress_summary,
        "task_preview": task_preview,
        "parsed_tasks": parsed_output if not compact_summary else None,
        "overall_status": overall_status,
    }
    return status


def watch_run(
    *,
    output_path: str | None = None,
    input_path: str | None = None,
    error_path: str | None = None,
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
    poll_interval_seconds: float = 10.0,
    adaptive_polling: bool = True,
    max_poll_interval_seconds: float | None = 60.0,
    timeout_seconds: float | None = 3600.0,
    max_polls: int | None = None,
    history_limit: int = 8,
    stall_timeout_seconds: float | None = None,
    output_status_reader: Any = None,
    progress_summary_fn: Any = None,
) -> dict[str, Any]:
    def read_status() -> dict[str, Any]:
        return inspect_run_status(
            output_path=output_path,
            input_path=input_path,
            error_path=error_path,
            profile=profile,
            job_id=job_id,
            profiles_path=profiles_path,
            output_status_reader=output_status_reader,
            progress_summary_fn=progress_summary_fn,
        )

    return watch_run_status(
        read_status,
        poll_interval_seconds=poll_interval_seconds,
        adaptive_polling=adaptive_polling,
        max_poll_interval_seconds=max_poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        max_polls=max_polls,
        history_limit=history_limit,
        stall_timeout_seconds=stall_timeout_seconds,
    )


def tail_text_file(
    path: str,
    lines: int = 30,
    max_characters: int = 4000,
) -> dict[str, Any]:
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise ValueError(f"file does not exist: {path}")
    contents = file_path.read_text(encoding="utf-8", errors="replace")
    all_lines = contents.splitlines()
    excerpt_lines = all_lines[-lines:] if lines > 0 else all_lines
    excerpt = "\n".join(excerpt_lines)
    if len(excerpt) > max_characters:
        excerpt = excerpt[-max_characters:]
    last_nonempty_line = next(
        (
            line
            for line in reversed(excerpt_lines)
            if line.strip()
        ),
        None,
    )
    return {
        "path": str(file_path),
        "requested_lines": lines,
        "returned_line_count": len(excerpt_lines),
        "total_line_count": len(all_lines),
        "tail_text": excerpt,
        "last_nonempty_line": last_nonempty_line,
    }


def _file_info(path: str | None) -> dict[str, Any]:
    if not path:
        return {
            "path": None,
            "exists": False,
            "size_bytes": None,
            "modified_utc": None,
        }
    file_path = Path(path).resolve()
    if not file_path.exists():
        return {
            "path": str(file_path),
            "exists": False,
            "size_bytes": None,
            "modified_utc": None,
        }
    stat = file_path.stat()
    return {
        "path": str(file_path),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime,
            tz=UTC,
        ).isoformat(),
    }


_SLURM_STATE_MAP = {
    "PENDING": "queued",
    "CONFIGURING": "queued",
    "SUSPENDED": "queued",
    "RUNNING": "running",
    "COMPLETING": "running",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "NODE_FAIL": "failed",
    "TIMEOUT": "failed",
    "OUT_OF_MEMORY": "failed",
    "PREEMPTED": "failed",
    "REVOKED": "failed",
    "DEADLINE": "failed",
    "BOOT_FAIL": "failed",
    "SPECIAL_EXIT": "failed",
    "CANCELLED": "cancelled",
}


def _scheduler_status(
    profile: str,
    job_id: str,
    profiles_path: str | None,
) -> dict[str, Any]:
    profiles = load_runner_profiles(profiles_path)
    profile_payload = _resolve_profile(profiles, profile)
    launcher = profile_payload.get("launcher", {})
    scheduler = profile_payload.get("scheduler", {})
    status_template = launcher.get("status_command")
    scheduler_type = (
        scheduler.get("system")
        or launcher.get("scheduler_type", "slurm")
    ).lower()
    if scheduler_type != "slurm":
        raise ValueError(
            "external scheduler inspection supports only Slurm profiles"
        )

    if not status_template:
        return {
            "job_id": job_id,
            "status": "unsupported",
            "scheduler_type": scheduler_type,
            "command": None,
            "return_code": None,
            "raw_state": None,
            "stdout": None,
            "stderr": None,
        }

    command = shlex.split(
        _format_template(status_template, {"job_id": job_id})
    )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception as exc:
        return {
            "job_id": job_id,
            "status": "error",
            "scheduler_type": scheduler_type,
            "command": command,
            "return_code": None,
            "raw_state": None,
            "error": str(exc),
            "stdout": None,
            "stderr": None,
        }

    stdout = completed.stdout.strip()
    raw_state = None
    normalized, raw_state = _slurm_status(stdout)

    return {
        "job_id": job_id,
        "status": normalized,
        "scheduler_type": scheduler_type,
        "raw_state": raw_state,
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _slurm_status(stdout: str) -> tuple[str, str | None]:
    raw_state = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if (
            stripped
            and not stripped.upper().startswith("JOBID")
            and not stripped.startswith("-")
        ):
            raw_state = stripped.split()[-1].upper()
            break
    if raw_state:
        return _SLURM_STATE_MAP.get(raw_state, "unknown"), raw_state
    return "not_found", None


__all__ = [
    "inspect_run_status",
    "tail_text_file",
    "watch_run",
]

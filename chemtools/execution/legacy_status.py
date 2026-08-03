"""Legacy process, scheduler, file, and NWChem progress inspection.

Typed execution owns new launch status. These functions preserve the version 1
profile behavior used for unowned identifiers and direct Python callers.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

from chemtools.core.common import detect_program, read_text
from chemtools.core.monitoring import watch_run_status
from chemtools.execution.legacy_profiles import (
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
    process_id: int | None = None,
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
    progress_summary_fn: Any = None,
) -> dict[str, Any]:
    output_info = _file_info(output_path)
    error_info = _file_info(error_path)
    process_status = _process_status(process_id)
    jobid_stale_warning = None
    if job_id is None:
        jobid_file = _auto_jobid_path(output_path, input_path)
        if jobid_file is not None:
            try:
                job_id = (
                    jobid_file.read_text(encoding="utf-8").strip()
                    or None
                )
            except Exception:
                pass
    if job_id:
        resolved_error_path = error_path
        if not resolved_error_path:
            for candidate_path in (output_path, input_path):
                if not candidate_path:
                    continue
                base = re.sub(
                    r"\.(out|nw|log|nwout)$",
                    "",
                    str(Path(candidate_path).resolve()),
                    flags=re.IGNORECASE,
                )
                candidate = Path(base + ".err")
                if candidate.exists():
                    resolved_error_path = str(candidate)
                    break
        error_job_id = _extract_job_id_from_err(resolved_error_path)
        if error_job_id and error_job_id != job_id:
            jobid_stale_warning = (
                f"Stale .jobid detected: file says {job_id} but .err "
                f"references job {error_job_id}. Using {error_job_id} "
                "from .err (more recent)."
            )
            job_id = error_job_id

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

    input_summary = None
    parsed_output = None
    compact_summary = None
    progress_summary = None
    task_preview = None
    if output_info["exists"]:
        try:
            contents = read_text(output_info["path"])
            if detect_program(contents) == "nwchem":
                from chemtools.programs.nwchem.strategy.progress import (
                    build_progress_summary,
                    compact_program_summary,
                    load_input_summary,
                    parse_progress_state,
                )

                if input_path:
                    input_summary = load_input_summary(
                        input_path,
                        raw_text=input_raw_text,
                    )
                parsed_output = parse_progress_state(
                    contents,
                    output_info["path"],
                )
                build_progress = (
                    progress_summary_fn
                    or build_progress_summary
                )
                progress_summary = build_progress(
                    contents,
                    parsed_output,
                    input_summary=input_summary,
                )
                compact_summary = compact_program_summary(
                    parsed_output,
                    progress_summary=progress_summary,
                )
                task_preview = parsed_output.get("generic_tasks", [])[:5]
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
    elif scheduler_state == "running" or process_status == "running":
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
            "path": (
                str(Path(input_path).resolve())
                if input_path
                else None
            ),
            "exists": bool(input_summary),
        },
        "error_file": error_info,
        "process": {
            "process_id": process_id,
            "status": process_status,
        },
        "scheduler": scheduler_status,
        "output_summary": compact_summary,
        "progress_summary": progress_summary,
        "task_preview": task_preview,
        "parsed_tasks": parsed_output if not compact_summary else None,
        "overall_status": overall_status,
    }
    if jobid_stale_warning:
        status["jobid_stale_warning"] = jobid_stale_warning
    return status


def watch_run(
    *,
    output_path: str | None = None,
    input_path: str | None = None,
    error_path: str | None = None,
    process_id: int | None = None,
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
    progress_summary_fn: Any = None,
) -> dict[str, Any]:
    def read_status() -> dict[str, Any]:
        return inspect_run_status(
            output_path=output_path,
            input_path=input_path,
            error_path=error_path,
            process_id=process_id,
            profile=profile,
            job_id=job_id,
            profiles_path=profiles_path,
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


def cancel_scheduler_job(
    profile: str,
    job_id: str,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    """Send the configured cancel command for a legacy scheduler job."""
    profiles = load_runner_profiles(profiles_path)
    profile_payload = _resolve_profile(profiles, profile)
    launcher = profile_payload.get("launcher", {})
    cancel_template = launcher.get("cancel_command")
    if not cancel_template:
        return {
            "job_id": job_id,
            "cancelled": False,
            "error": "no cancel_command configured in profile",
        }
    command = shlex.split(
        _format_template(cancel_template, {"job_id": job_id})
    )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        return {
            "job_id": job_id,
            "cancelled": completed.returncode == 0,
            "command": command,
            "return_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:
        return {
            "job_id": job_id,
            "cancelled": False,
            "command": command,
            "error": str(exc),
        }


def _auto_jobid_path(
    output_path: str | None,
    input_path: str | None,
) -> Path | None:
    for path in (output_path, input_path):
        if not path:
            continue
        base = re.sub(
            r"\.(out|nw|log|nwout|err)$",
            "",
            str(Path(path).resolve()),
            flags=re.IGNORECASE,
        )
        candidate = Path(base + ".jobid")
        if candidate.exists():
            return candidate
    return None


def _extract_job_id_from_err(error_path: str | None) -> str | None:
    if not error_path:
        return None
    try:
        path = Path(error_path)
        if not path.exists() or path.stat().st_size == 0:
            return None
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )[-4096:]
        matches = re.findall(r"JOB\s+(\d+)\s+ON", text)
        if matches:
            return matches[-1]
    except Exception:
        pass
    return None


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


def _process_status(process_id: int | None) -> str:
    if process_id is None:
        return "unknown"
    try:
        waited_pid, _ = os.waitpid(process_id, os.WNOHANG)
        if waited_pid == process_id:
            return "exited"
    except ChildProcessError:
        pass
    except OSError:  # pragma: no cover
        pass
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return "not_found"
    except PermissionError:
        return "permission_denied"
    try:
        completed = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(process_id)],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if completed.returncode == 0:
            state = completed.stdout.strip()
            if not state:
                return "not_found"
            if "Z" in state.upper():
                return "zombie"
    except Exception:  # pragma: no cover
        pass
    return "running"


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

_PBS_STATE_MAP = {
    "Q": "queued",
    "H": "queued",
    "W": "queued",
    "T": "queued",
    "S": "queued",
    "R": "running",
    "E": "running",
    "C": "completed",
    "F": "failed",
}

_LSF_STATE_MAP = {
    "PEND": "queued",
    "PSUSP": "queued",
    "USUSP": "queued",
    "SSUSP": "queued",
    "RUN": "running",
    "DONE": "completed",
    "EXIT": "failed",
    "ZOMBI": "failed",
    "UNKWN": "unknown",
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
    if scheduler_type == "slurm":
        normalized, raw_state = _slurm_status(stdout)
    elif scheduler_type == "pbs":
        normalized, raw_state = _pbs_status(
            stdout,
            completed.returncode,
        )
    elif scheduler_type == "lsf":
        normalized, raw_state = _lsf_status(
            stdout,
            completed.returncode,
        )
    else:
        normalized = (
            "running"
            if completed.returncode == 0 and stdout
            else "not_found"
        )

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


def _pbs_status(
    stdout: str,
    return_code: int,
) -> tuple[str, str | None]:
    match = re.search(
        r"job_state\s*=\s*(\w+)",
        stdout,
        re.IGNORECASE,
    )
    if match:
        raw_state = match.group(1).upper()
        return _PBS_STATE_MAP.get(raw_state, "unknown"), raw_state
    if not stdout and return_code != 0:
        return "not_found", None
    return "unknown", None


def _lsf_status(
    stdout: str,
    return_code: int,
) -> tuple[str, str | None]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) >= 2:
        header = lines[0].split()
        values = lines[1].split()
        if "STAT" in header and len(values) > header.index("STAT"):
            raw_state = values[header.index("STAT")].upper()
            return _LSF_STATE_MAP.get(raw_state, "unknown"), raw_state
        return "unknown", None
    if not stdout and return_code != 0:
        return "not_found", None
    return "unknown", None


__all__ = [
    "cancel_scheduler_job",
    "inspect_run_status",
    "tail_text_file",
    "watch_run",
]

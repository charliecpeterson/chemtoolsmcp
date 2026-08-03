"""Legacy GRASP executor, container lookup, and per-run session log.

GRASP2018 ships ~50 small executables. Each takes a fixed sequence of
prompted answers via stdin (no input file). This compatibility module:
  * Resolves the container path from CHEMTOOLS_GRASP_CONTAINER env var
    (default ~/mycontainers/grasp2018.sif).
  * Retains the direct Python executor during the compatibility window.
  * Formats capture files and ``grasp_session.md`` entries.
  * Appends a markdown entry to ``grasp_session.md`` in the working dir
    so the user can replay the workflow manually.

MCP handlers use the typed execution service and call only the session-log
formatter from this module.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_CONTAINER = "~/mycontainers/grasp2018.sif"
SESSION_LOG_NAME = "grasp_session.md"


def resolve_container() -> str:
    """Return the GRASP container path (env var or default), expanded."""
    raw = os.environ.get("CHEMTOOLS_GRASP_CONTAINER", DEFAULT_CONTAINER)
    return os.path.expanduser(raw)


def container_available() -> bool:
    """True if the container file exists."""
    return Path(resolve_container()).exists()


def build_command(exe: str, *, container: str | None = None) -> list[str]:
    """Build the apptainer exec argv for a GRASP executable."""
    container_path = container or resolve_container()
    return ["apptainer", "exec", container_path, exe]


def run_grasp_exe(
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
    """Run one GRASP executable through the legacy direct Python API.

    Parameters
    ----------
    exe
        Name of the GRASP binary (e.g. ``rmcdhf``, ``rcsfgenerate``, ``hf``).
    working_dir
        Directory to ``cd`` into before running. Created if missing.
    stdin_lines
        Lines to feed to the binary's stdin. Joined with ``\n``.
    args
        Optional extra positional args (e.g. ``["5f10.m"]`` for ``rlevels``).
    timeout_seconds
        Hard timeout. Default 10 min — atomic GRASP runs are typically seconds.
    log_to_session
        If True, append a markdown block to ``working_dir/grasp_session.md``.
    capture_log_file
        If given (relative to working_dir), also write stdout there.
    container
        Override the container path. Default uses CHEMTOOLS_GRASP_CONTAINER.

    Returns a dict with: exe, returncode, stdout, stderr, command,
    working_dir, log_file (if any), elapsed_seconds, timed_out, ok.
    """
    work = Path(working_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    if isinstance(stdin_lines, list):
        stdin_text = "\n".join(stdin_lines) + "\n"
    else:
        stdin_text = stdin_lines if stdin_lines.endswith("\n") else stdin_lines + "\n"

    cmd = build_command(exe, container=container)
    if args:
        cmd.extend(args)

    start = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            input=stdin_text,
            capture_output=True,
            text=True,
            cwd=str(work),
            timeout=timeout_seconds,
            check=False,
        )
        stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        returncode = -1
        timed_out = True
    elapsed = time.time() - start

    log_path = None
    if capture_log_file:
        log_path = str(work / capture_log_file)
        Path(log_path).write_text(stdout + ("\n--- STDERR ---\n" + stderr if stderr else ""))

    ok = (returncode == 0) and not timed_out
    pretty_cmd = " ".join(shlex.quote(p) for p in cmd)

    if log_to_session:
        append_execution_to_session(
            work=work,
            exe=exe,
            command=pretty_cmd,
            stdin_text=stdin_text,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            elapsed_seconds=elapsed,
            timed_out=timed_out,
            log_file=log_path,
        )

    return {
        "exe": exe,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "command": pretty_cmd,
        "working_dir": str(work),
        "log_file": log_path,
        "elapsed_seconds": round(elapsed, 3),
        "timed_out": timed_out,
        "ok": ok,
    }


def append_execution_to_session(
    *,
    work: Path,
    exe: str,
    command: str,
    stdin_text: str,
    stdout: str,
    stderr: str,
    returncode: int,
    elapsed_seconds: float,
    timed_out: bool,
    log_file: str | None,
) -> None:
    """Append a markdown entry to grasp_session.md in the working dir."""
    log_path = work / SESSION_LOG_NAME
    is_new = not log_path.exists()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts: list[str] = []
    if is_new:
        parts.append(f"# GRASP session log — {work.name}\n")
        parts.append(f"_Started {timestamp}_\n")

    status_emoji = "✓" if returncode == 0 and not timed_out else "✗"
    parts.append(f"\n## {timestamp} — `{exe}` {status_emoji}\n")
    if timed_out:
        parts.append(f"**Status**: TIMED OUT after {elapsed_seconds:.1f}s\n")
    else:
        parts.append(f"**Return code**: {returncode}  |  **Elapsed**: {elapsed_seconds:.2f}s\n")

    parts.append(f"\n**Command**:\n```\n{command}\n```\n")
    parts.append(f"\n**Stdin**:\n```\n{stdin_text.rstrip()}\n```\n")

    # Compact stdout — show first 30 + last 30 lines if large
    stdout_lines = stdout.strip().splitlines()
    if len(stdout_lines) > 80:
        head = "\n".join(stdout_lines[:30])
        tail = "\n".join(stdout_lines[-30:])
        stdout_show = f"{head}\n... [{len(stdout_lines) - 60} lines truncated] ...\n{tail}"
    else:
        stdout_show = stdout.rstrip()

    if stdout_show:
        parts.append(f"\n**Stdout**:\n```\n{stdout_show}\n```\n")
    if stderr.strip():
        parts.append(f"\n**Stderr**:\n```\n{stderr.rstrip()}\n```\n")
    if log_file:
        parts.append(f"\n_Full log saved to_: `{Path(log_file).name}`\n")

    with log_path.open("a") as f:
        f.write("".join(parts))


def append_session_note(working_dir: str, note: str, *, title: str | None = None) -> dict[str, Any]:
    """Manually append a freeform note to grasp_session.md."""
    work = Path(working_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    log_path = work / SESSION_LOG_NAME
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n## {timestamp} — Note: {title}\n" if title else f"\n## {timestamp} — Note\n"
    with log_path.open("a") as f:
        f.write(header)
        f.write(note.rstrip() + "\n")
    return {"log_file": str(log_path), "appended": True}


def read_session_log(working_dir: str) -> dict[str, Any]:
    """Return the contents of grasp_session.md if it exists."""
    log_path = Path(working_dir) / SESSION_LOG_NAME
    if not log_path.exists():
        return {"exists": False, "log_file": str(log_path), "contents": None}
    return {
        "exists": True,
        "log_file": str(log_path),
        "contents": log_path.read_text(encoding="utf-8", errors="replace"),
    }

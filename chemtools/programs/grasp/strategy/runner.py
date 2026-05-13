"""Execute a GRASP workflow plan end-to-end (local mode only).

Takes the step list from ``workflows.py`` and runs each step via
``run_grasp_exe``. Stops on first failure and returns the partial transcript.
``post`` shell commands (e.g. ``cp rcsf.out rcsf.inp``) are executed in the
working directory between steps.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

from chemtools.programs.grasp.runtime import run_grasp_exe, append_session_note


def run_workflow(
    plan: dict[str, Any],
    *,
    working_dir: str,
    stop_on_failure: bool = True,
    timeout_per_step: float = 600.0,
    container: str | None = None,
) -> dict[str, Any]:
    """Execute every step in a workflow plan.

    Returns a transcript: list of {step_index, exe, ok, returncode, ...}
    plus an overall ``ok`` flag.
    """
    work = Path(working_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    # Title note in session log
    append_session_note(
        str(work),
        f"Starting **{plan['workflow']}** workflow with {plan['n_steps']} steps. "
        f"Target name: `{plan.get('name')}`.",
        title=f"Workflow start: {plan['workflow']}",
    )

    # Run preamble shell commands if present (e.g., copy a previous *.w file)
    preamble = plan.get("preamble") or []
    for shell_cmd in preamble:
        _run_shell(shell_cmd, work)

    transcript: list[dict[str, Any]] = []
    overall_ok = True
    for i, step in enumerate(plan["steps"], start=1):
        exe = step["exe"]
        result = run_grasp_exe(
            exe,
            working_dir=str(work),
            stdin_lines=step["stdin"],
            args=step.get("args") or None,
            timeout_seconds=timeout_per_step,
            container=container,
        )
        transcript.append({
            "step_index": i,
            "exe": exe,
            "description": step.get("description"),
            "ok": result["ok"],
            "returncode": result["returncode"],
            "elapsed_seconds": result["elapsed_seconds"],
            "stdout_tail": "\n".join(result["stdout"].splitlines()[-15:]),
            "stderr_tail": "\n".join(result["stderr"].splitlines()[-15:]) if result["stderr"] else "",
        })
        if not result["ok"]:
            overall_ok = False
            if stop_on_failure:
                append_session_note(
                    str(work),
                    f"Step {i} (`{exe}`) failed with returncode {result['returncode']}. "
                    f"Stopping workflow.",
                    title="Workflow halted on failure",
                )
                break
        # Run post commands
        for shell_cmd in step.get("post", []):
            _run_shell(shell_cmd, work)

    if overall_ok:
        append_session_note(
            str(work),
            f"Workflow `{plan['workflow']}` completed successfully ({len(transcript)} steps).",
            title="Workflow complete",
        )

    return {
        "workflow": plan["workflow"],
        "name": plan.get("name"),
        "working_dir": str(work),
        "n_steps_attempted": len(transcript),
        "n_steps_total": plan["n_steps"],
        "ok": overall_ok,
        "transcript": transcript,
        "session_log": str(work / "grasp_session.md"),
    }


def _run_shell(cmd: str, cwd: Path) -> None:
    """Execute a simple shell command in cwd. Used for cp/mv between steps."""
    subprocess.run(cmd, shell=True, cwd=str(cwd), check=False)

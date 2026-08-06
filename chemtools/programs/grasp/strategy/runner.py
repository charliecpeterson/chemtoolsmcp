"""Execute a GRASP workflow plan step by step in local mode.

The MCP application injects its permission-checked synchronous runner. Direct
Python callers retain the legacy runner during the compatibility window.
Preamble and post-step actions accept only three-token ``cp`` operations, and
copy destinations must remain in the working directory.
"""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path
from typing import Any, Callable

from chemtools.programs.grasp.parse.csf import load_grasp_csf_list
from chemtools.programs.grasp.runtime import run_grasp_exe, append_session_note


def run_workflow(
    plan: dict[str, Any],
    *,
    working_dir: str,
    stop_on_failure: bool = True,
    timeout_per_step: float = 600.0,
    container: str | None = None,
    run_step: Callable[..., dict[str, Any]] = run_grasp_exe,
) -> dict[str, Any]:
    """Execute every step in a workflow plan.

    Returns a transcript: list of {step_index, exe, ok, returncode, ...}
    plus an overall ``ok`` flag.
    """
    work = Path(working_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    append_session_note(
        str(work),
        f"Starting **{plan['workflow']}** workflow with {plan['n_steps']} steps. "
        f"Target name: `{plan.get('name')}`.",
        title=f"Workflow start: {plan['workflow']}",
    )

    preamble = plan.get("preamble") or []
    for action in preamble:
        _run_file_action(action, work)

    transcript: list[dict[str, Any]] = []
    overall_ok = True
    for i, step in enumerate(plan["steps"], start=1):
        exe = step["exe"]
        result = run_step(
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
            "launch_id": result.get("launch_id"),
            "execution_instance_id": result.get(
                "execution_instance_id"
            ),
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
        for action in step.get("post", []):
            _run_file_action(action, work)
        expected_blocks = step.get("expected_csf_blocks")
        if expected_blocks is not None:
            try:
                validate_csf_block_contract(
                    work / "rcsf.inp",
                    expected_blocks,
                )
            except ValueError as error:
                transcript[-1]["ok"] = False
                transcript[-1]["contract_error"] = str(error)
                overall_ok = False
                break

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


def validate_csf_block_contract(
    csf_path: str | Path,
    expected_blocks: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Require generated CSF blocks to match labeled RMCDHF selections."""
    document = load_grasp_csf_list(csf_path)
    expected: list[dict[str, object]] = []
    for index, block in enumerate(expected_blocks, start=1):
        required = {"two_j", "parity", "ncsf"}
        if set(block) != required:
            raise ValueError(
                f"expected CSF block {index} requires exactly {sorted(required)}"
            )
        parity = block["parity"]
        if parity not in {"+", "-"}:
            raise ValueError(
                f"expected CSF block {index} parity must be '+' or '-'"
            )
        expected.append({
            "two_j": int(block["two_j"]),
            "parity": parity,
            "ncsf": int(block["ncsf"]),
        })
    actual = [
        {
            "two_j": block.two_j,
            "parity": block.parity,
            "ncsf": len(block.entries),
        }
        for block in document.blocks
    ]
    if actual != expected:
        raise ValueError(
            "generated CSF block order does not match the labeled RMCDHF "
            f"selection contract: expected {expected}, found {actual}"
        )
    return actual


def _run_file_action(command: str, cwd: Path) -> None:
    """Apply the copy actions emitted by the built-in workflow planners."""
    tokens = shlex.split(command, comments=True)
    if len(tokens) != 3 or tokens[0] != "cp":
        raise ValueError(
            f"unsupported GRASP workflow file action: {command!r}"
        )
    source = Path(tokens[1])
    if not source.is_absolute():
        source = cwd / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"GRASP workflow copy source does not exist: {source}"
        )
    destination = (cwd / tokens[2]).resolve()
    if destination != cwd and cwd not in destination.parents:
        raise ValueError(
            f"GRASP workflow copy destination escapes working directory: "
            f"{destination}"
        )
    shutil.copy2(source, destination)

"""Typed NWChem scheduler monitoring through the MCP launch path."""

import hashlib
from pathlib import Path
import subprocess

import chemtools.core.monitoring as core_monitoring
import chemtools.execution.local as local_execution
import chemtools.execution.slurm as slurm_execution
import chemtools.application.nwchem_monitoring as nwchem_monitoring
import chemtools.mcp.tools.nwchem_jobs as nwchem_jobs
import pytest

from chemtools.core.artifact_registry import load_run_artifacts
from chemtools.core.run_registry import get_run_summary
from chemtools.mcp.decorator import set_active_mode
from chemtools.mcp.tools.nwchem_jobs import (
    _handle_launch_nwchem_run,
    _handle_watch_nwchem_run,
)


PROFILE_PATH = (
    Path(__file__).parents[1]
    / "chemtools"
    / "runner_profiles.example.json"
)


@pytest.mark.parametrize("auto_watch", (True, False))
def test_mcp_slurm_watch_paths_use_owned_typed_status(
    tmp_path,
    monkeypatch,
    auto_watch,
):
    input_path = tmp_path / "water.nw"
    input_path.write_text(
        "start water\ngeometry\nO 0 0 0\nend\ntask scf energy\n",
        encoding="utf-8",
    )
    stdout_bytes = (
        b"Northwest Computational Chemistry Package (NWChem)\n"
        b"Total SCF energy = -75.000000\n"
    )
    stderr_bytes = b""
    calls = []
    sleeps = []
    queue_queries = 0

    def fake_run(argv, **kwargs):
        nonlocal queue_queries
        calls.append(tuple(argv))
        if argv[0] == "sbatch":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="Submitted batch job 9292\n",
                stderr="",
            )
        if argv[0] == "squeue":
            queue_queries += 1
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="RUNNING\n" if queue_queries == 1 else "",
                stderr="",
            )
        (tmp_path / "water.out").write_bytes(stdout_bytes)
        (tmp_path / "water.err").write_bytes(stderr_bytes)
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="COMPLETED|0:0|42\n",
            stderr="",
        )

    db_path = tmp_path / "registry.db"
    monkeypatch.setenv("CHEMTOOLS_REGISTRY_DB", str(db_path))
    monkeypatch.setattr(
        slurm_execution.subprocess,
        "run",
        fake_run,
    )
    monkeypatch.setattr(
        core_monitoring.time,
        "sleep",
        sleeps.append,
    )
    set_active_mode("hpc")
    try:
        launched = _handle_launch_nwchem_run({
            "input_file": str(input_path),
            "profile": "slurm_cpu",
            "profiles_path": str(PROFILE_PATH),
            "auto_watch": auto_watch,
        })
        watch = (
            launched["watch"]
            if auto_watch
            else _handle_watch_nwchem_run({
                "output_file": launched["output_file"],
                "input_file": str(input_path),
                "error_file": launched["error_file"],
                "profile": "slurm_cpu",
                "job_id": launched["job_id"],
                "profiles_path": str(PROFILE_PATH),
                "poll_interval_seconds": 30.0,
                "max_poll_interval_seconds": 120.0,
                "timeout_seconds": None,
            })
        )
    finally:
        set_active_mode("analysis")

    final_status = watch["final_status"]
    run_uid = launched["registry"]["run_uid"]
    run = get_run_summary(run_uid=run_uid, db_path=str(db_path))
    artifacts = load_run_artifacts(run_uid, db_path)

    assert watch["terminal"] is True
    assert watch["stop_reason"] == "terminal_status"
    assert watch["poll_count"] == 2
    assert watch["overall_status"] == "completed_incomplete"
    assert tuple(item["overall_status"] for item in watch["history"]) == (
        "running",
        "completed_incomplete",
    )
    assert final_status["scheduler"]["status"] == "completed"
    assert final_status["scheduler"]["source"] == "accounting"
    assert final_status["scheduler"]["job_exit_code"] == 0
    assert final_status["scheduler"]["termination_signal"] == 0
    assert final_status["execution_record"]["status"] == "completed"
    assert run["status"] == "completed"
    assert run["walltime_used_sec"] == 42.0
    assert artifacts is not None
    assert tuple(
        observation.sha256
        for observation in artifacts.observations
    ) == (
        hashlib.sha256(stdout_bytes).hexdigest(),
        hashlib.sha256(stderr_bytes).hexdigest(),
    )
    assert sleeps == [30.0]
    assert tuple(call[0] for call in calls) == (
        "sbatch",
        "squeue",
        "squeue",
        "sacct",
    )
    if not auto_watch:
        assert watch["next_actions"][0]["tool"] == "analyze_nwchem_case"


def test_explicit_local_watch_uses_owned_process_handle(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "water.nw"
    input_path.write_text(
        "start water\ngeometry\nO 0 0 0\nend\ntask scf energy\n",
        encoding="utf-8",
    )
    stdout_bytes = (
        b"Northwest Computational Chemistry Package (NWChem)\n"
        b"Total SCF energy = -75.000000\n"
    )
    stderr_bytes = b""

    class Process:
        pid = 9393
        polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0

    process = Process()

    def fake_popen(*args, **kwargs):
        kwargs["stdout"].write(stdout_bytes)
        kwargs["stdout"].flush()
        kwargs["stderr"].write(stderr_bytes)
        kwargs["stderr"].flush()
        return process

    db_path = tmp_path / "registry.db"
    monkeypatch.setenv("CHEMTOOLS_REGISTRY_DB", str(db_path))
    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        fake_popen,
    )
    set_active_mode("local")
    try:
        launched = _handle_launch_nwchem_run({
            "input_file": str(input_path),
            "profile": "local",
            "profiles_path": str(PROFILE_PATH),
        })
        watched = _handle_watch_nwchem_run({
            "output_file": launched["output_file"],
            "input_file": str(input_path),
            "error_file": launched["error_file"],
            "process_id": process.pid,
            "poll_interval_seconds": 0,
        })
    finally:
        set_active_mode("analysis")

    final_status = watched["final_status"]
    run_uid = launched["registry"]["run_uid"]
    run = get_run_summary(run_uid=run_uid, db_path=str(db_path))
    artifacts = load_run_artifacts(run_uid, db_path)

    assert watched["terminal"] is True
    assert watched["poll_count"] == 2
    assert tuple(item["overall_status"] for item in watched["history"]) == (
        "running",
        "completed_incomplete",
    )
    assert final_status["process"] == {
        "process_id": 9393,
        "status": "completed",
        "return_code": 0,
    }
    assert final_status["execution_record"]["status"] == "completed"
    assert run["status"] == "completed"
    assert artifacts is not None
    assert tuple(
        observation.sha256
        for observation in artifacts.observations
    ) == (
        hashlib.sha256(stdout_bytes).hexdigest(),
        hashlib.sha256(stderr_bytes).hexdigest(),
    )
    assert process.polls == 2
    assert watched["next_actions"][0]["tool"] == "analyze_nwchem_case"


@pytest.mark.parametrize(
    "identifier",
    (
        {"profile": "legacy_cluster", "job_id": "8181"},
        {"process_id": 8282},
    ),
)
def test_explicit_watch_keeps_unowned_identifier_on_legacy_path(
    monkeypatch,
    identifier,
):
    legacy_calls = []

    def fake_legacy_watch(**kwargs):
        legacy_calls.append(kwargs)
        return {
            "terminal": False,
            "overall_status": "running",
            "final_status": {"overall_status": "running"},
        }

    def reject_typed_poll(*args, **kwargs):
        raise AssertionError("unowned identifier reached typed polling")

    monkeypatch.setattr(
        nwchem_jobs,
        "watch_nwchem_run",
        fake_legacy_watch,
    )
    monkeypatch.setattr(
        nwchem_monitoring,
        "inspect_nwchem_status_with_service",
        reject_typed_poll,
    )
    set_active_mode("hpc")
    try:
        arguments = {
            "output_file": "/work/legacy.out",
            "input_file": "/work/legacy.nw",
            "poll_interval_seconds": 0,
            "max_polls": 1,
        }
        arguments.update(identifier)
        watched = _handle_watch_nwchem_run(arguments)
    finally:
        set_active_mode("analysis")

    assert len(legacy_calls) == 1
    for key, value in identifier.items():
        assert legacy_calls[0][key] == value
    assert watched["overall_status"] == "running"
    assert watched["next_actions"][0]["tool"] == "watch_nwchem_run"

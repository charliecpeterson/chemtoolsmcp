"""Typed Slurm queue, accounting, and persisted status contracts."""

from pathlib import Path
import subprocess

import chemtools.execution.slurm as slurm_execution
from chemtools.application.execution import ExecutionService
from chemtools.core.execution import SlurmStatusResult
from chemtools.execution.legacy_runner import load_runner_profiles
from chemtools.execution import SlurmExecutor
from chemtools.programs.nwchem.launch import (
    adapt_legacy_nwchem_profile,
    build_nwchem_launch_plan,
)


PROFILE_PATH = (
    Path(__file__).parents[1]
    / "chemtools"
    / "runner_profiles.example.json"
)


def _plan_and_target(tmp_path: Path):
    input_path = tmp_path / "water.nw"
    input_path.write_text(
        "start water\ngeometry\nO 0 0 0\nend\ntask scf energy\n",
        encoding="utf-8",
    )
    profiles = load_runner_profiles(str(PROFILE_PATH))
    adapted = adapt_legacy_nwchem_profile(
        profiles,
        "slurm_cpu",
        allowed_work_roots=(tmp_path,),
    )
    return (
        build_nwchem_launch_plan(
            input_path,
            adapted.default_resources,
        ),
        adapted.target,
    )


def test_active_job_status_comes_from_squeue(tmp_path, monkeypatch):
    _, target = _plan_and_target(tmp_path)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="RUNNING\n",
            stderr="",
        )

    monkeypatch.setattr(slurm_execution.subprocess, "run", fake_run)

    result = SlurmExecutor().status("81", target)

    assert result == SlurmStatusResult(
        job_id="81",
        query_argv=("squeue", "-j", "81", "-h", "-o", "%T"),
        source="queue",
        status="running",
        raw_state="RUNNING",
        query_return_code=0,
        stdout="RUNNING\n",
        stderr="",
        checked_at=result.checked_at,
    )
    assert calls == [("squeue", "-j", "81", "-h", "-o", "%T")]


def test_terminal_job_falls_back_to_accounting(
    tmp_path,
    monkeypatch,
):
    _, target = _plan_and_target(tmp_path)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="",
                stderr="",
            )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="COMPLETED|0:0|125\n",
            stderr="",
        )

    monkeypatch.setattr(slurm_execution.subprocess, "run", fake_run)

    result = SlurmExecutor().status("82", target)

    assert result.status == "completed"
    assert result.source == "accounting"
    assert result.raw_state == "COMPLETED"
    assert result.job_exit_code == 0
    assert result.termination_signal == 0
    assert result.elapsed_seconds == 125.0
    assert calls == [
        ("squeue", "-j", "82", "-h", "-o", "%T"),
        (
            "sacct",
            "-n",
            "-X",
            "-j",
            "82",
            "-o",
            "State%30,ExitCode,ElapsedRaw",
            "-P",
        ),
    ]


def test_empty_queue_and_accounting_results_are_not_completion(
    tmp_path,
    monkeypatch,
):
    _, target = _plan_and_target(tmp_path)

    monkeypatch.setattr(
        slurm_execution.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv,
            0,
            stdout="",
            stderr="",
        ),
    )

    result = SlurmExecutor().status("83", target)

    assert result.status == "not_found"
    assert result.source == "accounting"
    assert result.raw_state is None
    assert result.job_exit_code is None


def test_service_persists_accounted_terminal_state_once(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan_and_target(tmp_path)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[0] == "sbatch":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="Submitted batch job 84\n",
                stderr="",
            )
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="",
                stderr="",
            )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="COMPLETED|0:0|125\n",
            stderr="",
        )

    monkeypatch.setattr(slurm_execution.subprocess, "run", fake_run)
    service = ExecutionService(
        enable_execution=True,
        registry_db_path=tmp_path / "registry.db",
    )
    launched = service.launch(plan, target)

    completed = service.refresh_slurm_status_external(
        "84",
        target_name="slurm_cpu",
    )
    repeated = service.refresh_slurm_status(launched.record.launch_id)

    assert completed.result.status == "completed"
    assert completed.record.status == "completed"
    assert completed.record.return_code == 0
    assert completed.record.elapsed_seconds == 125.0
    assert repeated.record == completed.record
    assert repeated.result.source == "record"
    assert repeated.result.status == "completed"
    assert len(calls) == 3


def test_service_does_not_complete_job_missing_from_both_sources(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan_and_target(tmp_path)

    def fake_run(argv, **kwargs):
        if argv[0] == "sbatch":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="Submitted batch job 85\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(slurm_execution.subprocess, "run", fake_run)
    service = ExecutionService(
        enable_execution=True,
        registry_db_path=tmp_path / "registry.db",
    )
    launched = service.launch(plan, target)

    missing = service.refresh_slurm_status(launched.record.launch_id)

    assert missing.result.status == "not_found"
    assert missing.record.status == "submitted"
    assert service.get_launch_record(
        launched.record.launch_id
    ).status == "submitted"

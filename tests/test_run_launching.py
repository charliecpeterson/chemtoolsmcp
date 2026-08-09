"""Guided launches require approval of one exact rendered plan."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from chemtools.application.execution import ExecutionService
from chemtools.application.run_launching import (
    LAUNCH_RUN_SCHEMA,
    LaunchRunError,
    launch_run,
)
from chemtools.core.program import (
    InvalidProgramBackend,
    ProgramCapability,
    validate_backend,
)
from chemtools.core.execution import SlurmSubmissionResult
from chemtools.execution.slurm import SlurmExecutor
from chemtools.mcp.catalog import BUILTIN_BACKENDS, load_backend
import chemtools.execution.local as local_execution


PROFILE_PATH = (
    Path(__file__).parents[1]
    / "chemtools"
    / "runner_profiles.example.json"
)
NWCHEM = load_backend(BUILTIN_BACKENDS[0])
MOLCAS = load_backend(BUILTIN_BACKENDS[1])


def _input(tmp_path: Path) -> Path:
    path = tmp_path / "uo2.nw"
    path.write_text(
        "start uo2\n"
        "geometry\n"
        "U 0 0 0\n"
        "O 0 0 1.8\n"
        "O 0 0 -1.8\n"
        "end\n"
        "basis\n"
        "* library sto-3g\n"
        "end\n"
        "dft\n"
        "mult 3\n"
        "xc pbe0\n"
        "end\n"
        "task dft optimize\n",
        encoding="utf-8",
    )
    return path


def _prepare(
    tmp_path: Path,
    *,
    service: ExecutionService | None = None,
    profile: str = "local_mpirun",
    resources: dict | None = None,
    approval_token: str | None = None,
) -> dict:
    return launch_run(
        NWCHEM,
        service or ExecutionService(),
        input_file=tmp_path / "uo2.nw",
        profile=profile,
        profiles_path=PROFILE_PATH,
        resources=resources,
        approval_token=approval_token,
    )


def test_slurm_launch_preparation_is_read_only_and_exact(tmp_path):
    input_path = _input(tmp_path)
    before = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
    }

    prepared = _prepare(
        tmp_path,
        profile="slurm_cpu",
        resources={"mpi_ranks": 32},
    )

    assert prepared["schema_version"] == LAUNCH_RUN_SCHEMA
    assert prepared["status"] == "awaiting_approval"
    assert prepared["program"] == {"name": "nwchem"}
    assert prepared["input"] == {
        "path": str(input_path),
        "size_bytes": input_path.stat().st_size,
        "sha256": prepared["input"]["sha256"],
    }
    assert len(prepared["input"]["sha256"]) == 64
    plan = prepared["evidence"]["plan"]
    assert plan["executor"] == "slurm"
    assert plan["resources"]["mpi_ranks"] == 32
    assert plan["argv"] == ["srun", "nwchem", "uo2.nw"]
    assert len(plan["environment_sha256"]) == 64
    assert plan["scheduler"]["submit_argv"] == [
        "sbatch",
        str(tmp_path / "uo2.job"),
    ]
    assert len(plan["scheduler"]["script_sha256"]) == 64
    assert prepared["approval"]["token"].startswith("sha256:")
    assert {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
    } == before


def test_same_plan_returns_same_approval_token(tmp_path):
    _input(tmp_path)

    first = _prepare(tmp_path, resources={"mpi_ranks": 2})
    second = _prepare(tmp_path, resources={"mpi_ranks": 2})

    assert first["approval"]["token"] == second["approval"]["token"]


def test_changed_input_invalidates_prior_approval(tmp_path):
    input_path = _input(tmp_path)
    prepared = _prepare(tmp_path)
    input_path.write_text(
        input_path.read_text(encoding="utf-8") + "# reviewed edit\n",
        encoding="utf-8",
    )

    changed = _prepare(
        tmp_path,
        approval_token=prepared["approval"]["token"],
    )

    assert changed["status"] == "approval_invalidated"
    assert changed["assessment"]["verdict"]["label"] == (
        "launch_requires_new_approval"
    )
    assert changed["approval"]["token"] != prepared["approval"]["token"]
    assert not (tmp_path / "uo2.out").exists()


def test_existing_artifact_blocks_preparation_without_archiving(tmp_path):
    _input(tmp_path)
    output_path = tmp_path / "uo2.out"
    output_path.write_text("existing result\n", encoding="utf-8")

    blocked = _prepare(tmp_path)

    assert blocked["status"] == "blocked"
    assert blocked["approval"] == {"required": True, "token": None}
    assert blocked["evidence"]["conflicts"] == [{
        "role": "stdout",
        "path": str(output_path),
    }]
    assert isinstance(blocked["evidence"]["input_review"]["issues"], list)
    assert output_path.read_text(encoding="utf-8") == "existing result\n"


def test_approved_plan_stays_disabled_in_analysis_mode(tmp_path):
    _input(tmp_path)
    prepared = _prepare(tmp_path)

    refused = _prepare(
        tmp_path,
        approval_token=prepared["approval"]["token"],
    )

    assert refused["status"] == "execution_disabled"
    assert refused["approval"]["accepted"] is True
    assert refused["next_actions"][0]["action"] == (
        "restart_with_execution_enabled"
    )
    assert not (tmp_path / "uo2.out").exists()


def test_explicitly_approved_local_plan_creates_owned_launch(
    tmp_path,
    monkeypatch,
):
    _input(tmp_path)
    registry_path = tmp_path / "registry.db"
    service = ExecutionService(
        enable_execution=True,
        registry_db_path=registry_path,
    )
    prepared = _prepare(tmp_path, service=service, resources={"mpi_ranks": 2})
    observed = {}

    class Process:
        pid = 4242

    def fake_popen(argv, **kwargs):
        observed["argv"] = argv
        observed["cwd"] = kwargs["cwd"]
        observed["shell"] = kwargs["shell"]
        return Process()

    monkeypatch.setattr(local_execution.subprocess, "Popen", fake_popen)

    launched = _prepare(
        tmp_path,
        service=service,
        resources={"mpi_ranks": 2},
        approval_token=prepared["approval"]["token"],
    )

    assert launched["status"] == "launched"
    assert launched["launch"]["status"] == "started"
    assert launched["launch"]["process_id"] == 4242
    assert launched["next_actions"][0]["launch_id"] == (
        launched["launch"]["launch_id"]
    )
    assert observed == {
        "argv": ("mpirun", "-np", "2", "nwchem", "uo2.nw"),
        "cwd": tmp_path,
        "shell": False,
    }
    assert registry_path.is_file()
    assert (tmp_path / "uo2.out").is_file()
    assert (tmp_path / "uo2.err").is_file()


def test_launch_run_refuses_unimplemented_program_provider(tmp_path):
    input_path = _input(tmp_path)

    with pytest.raises(LaunchRunError) as caught:
        launch_run(
            MOLCAS,
            ExecutionService(),
            input_file=input_path,
            profile="local_mpirun",
            profiles_path=PROFILE_PATH,
        )

    assert caught.value.as_dict() == {
        "error": "unsupported_capability",
        "message": "'molcas' does not support guided launch planning",
        "program": "molcas",
    }


def test_scheduler_submission_failure_is_not_reported_as_launched(
    tmp_path,
    monkeypatch,
):
    _input(tmp_path)
    service = ExecutionService(
        enable_execution=True,
        registry_db_path=tmp_path / "registry.db",
    )
    prepared = _prepare(
        tmp_path,
        service=service,
        profile="slurm_cpu",
        resources={"mpi_ranks": 32},
    )

    def failed_submit(executor, plan, target):
        return SlurmSubmissionResult(
            script=executor.render(plan, target),
            status="submit_failed",
            return_code=1,
            stdout="",
            stderr="account is invalid",
            job_id=None,
            submitted_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr(SlurmExecutor, "submit", failed_submit)

    failed = _prepare(
        tmp_path,
        service=service,
        profile="slurm_cpu",
        resources={"mpi_ranks": 32},
        approval_token=prepared["approval"]["token"],
    )

    assert failed["status"] == "launch_failed"
    assert failed["assessment"]["verdict"]["label"] == "submit_failed"
    assert failed["launch"]["submission"] == {
        "return_code": 1,
        "stdout": "",
        "stderr": "account is invalid",
    }
    assert failed["next_actions"][0]["action"] == "inspect_launch_failure"


def test_execution_plan_capability_requires_provider():
    broken = replace(NWCHEM, launches=None)

    assert ProgramCapability.EXECUTION_PLAN in broken.capabilities
    with pytest.raises(
        InvalidProgramBackend,
        match=(
            "declares 'execution.plan' but launches.prepare_launch is unavailable"
        ),
    ):
        validate_backend(broken)

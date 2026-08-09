"""Guided monitoring stays bound to owned launches and recorded artifacts."""

from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

import chemtools.execution.local as local_execution
import chemtools.execution.slurm as slurm_execution
from chemtools.application.execution import ExecutionService
from chemtools.application.run_monitoring import (
    MONITOR_RUN_SCHEMA,
    MonitorRunError,
    monitor_run,
)
from chemtools.execution.legacy_runner import load_runner_profiles
from chemtools.core.program import ProgramCapability
from chemtools.mcp.catalog import BUILTIN_BACKENDS, load_backend
from chemtools.mcp.dispatch import dispatch_tool
from chemtools.mcp.tools import guided as guided_tools
from chemtools.programs.nwchem.launch import (
    adapt_legacy_nwchem_profile,
    build_nwchem_launch_plan,
)


PROFILE_PATH = (
    Path(__file__).parents[1]
    / "chemtools"
    / "runner_profiles.example.json"
)
SUCCESS_OUTPUT = (
    Path(__file__).parent
    / "fixtures"
    / "nwchem_pyscf"
    / "h2o_rhf_sto3g.out"
)
NWCHEM = load_backend(BUILTIN_BACKENDS[0])


def _plan_and_target(tmp_path: Path, profile: str = "local"):
    input_path = tmp_path / "water.nw"
    input_path.write_text(
        "start water\n"
        "geometry\n"
        "O 0 0 0\n"
        "H 0 0.7 0.5\n"
        "H 0 -0.7 0.5\n"
        "end\n"
        "basis\n"
        "* library sto-3g\n"
        "end\n"
        "task scf energy\n",
        encoding="utf-8",
    )
    profiles = load_runner_profiles(str(PROFILE_PATH))
    adapted = adapt_legacy_nwchem_profile(
        profiles,
        profile,
        allowed_work_roots=(tmp_path,),
    )
    return (
        build_nwchem_launch_plan(
            input_path,
            adapted.default_resources,
        ),
        adapted.target,
    )


def _service(tmp_path: Path) -> ExecutionService:
    return ExecutionService(
        enable_execution=True,
        registry_db_path=tmp_path / "registry.db",
    )


def test_running_launch_reports_execution_and_scientific_completion(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan_and_target(tmp_path)

    class Process:
        pid = 4242

        def poll(self):
            return None

    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        lambda *args, **kwargs: Process(),
    )
    service = _service(tmp_path)
    launched = service.launch(plan, target)
    assert launched.record.stdout_path is not None
    launched.record.stdout_path.write_bytes(SUCCESS_OUTPUT.read_bytes())

    monitored = monitor_run(
        NWCHEM,
        service,
        launch_id=launched.record.launch_id,
    )

    assert monitored["schema_version"] == MONITOR_RUN_SCHEMA
    assert monitored["status"] == "running"
    assert monitored["program"] == {"name": "nwchem"}
    assert monitored["launch"]["status"] == "started"
    assert monitored["evidence"]["execution"]["process"] == {
        "process_id": 4242,
        "status": "running",
        "return_code": None,
        "checked_at": monitored["evidence"]["execution"]["process"][
            "checked_at"
        ],
    }
    scientific = monitored["evidence"]["scientific"]
    assert scientific["status"] == "completed"
    assert scientific["completion_observed"] is True
    assert scientific["outcome"] == "success"
    assert scientific["progress"]["task_count"] == 1
    assert monitored["assessment"]["verdict"]["label"] == "run_active"
    assert monitored["next_actions"] == [{
        "action": "monitor_run",
        "arguments": {"launch_id": launched.record.launch_id},
        "reason": "Refresh this owned launch without changing it.",
        "priority": 1,
    }]


def test_completed_launch_requires_full_inspection_before_acceptance(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan_and_target(tmp_path)

    class Process:
        pid = 4343

        def poll(self):
            return 0

    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        lambda *args, **kwargs: Process(),
    )
    service = _service(tmp_path)
    launched = service.launch(plan, target)
    assert launched.record.stdout_path is not None
    launched.record.stdout_path.write_bytes(SUCCESS_OUTPUT.read_bytes())

    monitored = monitor_run(
        NWCHEM,
        service,
        launch_id=launched.record.launch_id,
    )

    assert monitored["status"] == "completed"
    assert monitored["launch"]["status"] == "completed"
    assert monitored["assessment"]["verdict"]["label"] == (
        "completed_success"
    )
    assert monitored["evidence"]["scientific"]["status"] == "completed"
    assert monitored["next_actions"] == [{
        "action": "inspect_run",
        "arguments": {"output_file": str(launched.record.stdout_path)},
        "reason": (
            "Run the full scientific inspection before accepting or "
            "recovering the calculation."
        ),
        "priority": 1,
    }]
    assert service.get_launch_record(launched.record.launch_id).status == (
        "completed"
    )


def test_guided_mcp_monitor_uses_process_owned_service(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan_and_target(tmp_path)

    class Process:
        pid = 4399

        def poll(self):
            return None

    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        lambda *args, **kwargs: Process(),
    )
    service = _service(tmp_path)
    launched = service.launch(plan, target)
    monkeypatch.setattr(
        guided_tools,
        "get_execution_service",
        lambda: service,
    )

    monitored = dispatch_tool(
        "monitor_run",
        {"launch_id": launched.record.launch_id},
    )

    assert monitored["schema_version"] == MONITOR_RUN_SCHEMA
    assert monitored["status"] == "running"
    assert monitored["launch"]["launch_id"] == launched.record.launch_id


def test_failed_launch_does_not_infer_scientific_completion(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan_and_target(tmp_path)

    class Process:
        pid = 4444

        def poll(self):
            return 17

    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        lambda *args, **kwargs: Process(),
    )
    service = _service(tmp_path)
    launched = service.launch(plan, target)

    monitored = monitor_run(
        NWCHEM,
        service,
        launch_id=launched.record.launch_id,
    )

    assert monitored["status"] == "failed"
    assert monitored["assessment"]["verdict"]["label"] == "completed_failed"
    assert monitored["evidence"]["scientific"] == {
        "status": "not_observed",
        "completion_observed": False,
        "outcome": None,
        "progress": None,
    }
    assert {item["code"] for item in monitored["uncertainty"]} == {
        "scientific_output_not_observed",
    }


def test_backend_without_progress_still_reports_owned_execution(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan_and_target(tmp_path)

    class Process:
        pid = 4499

        def poll(self):
            return None

    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        lambda *args, **kwargs: Process(),
    )
    service = _service(tmp_path)
    launched = service.launch(plan, target)
    assert launched.record.stdout_path is not None
    launched.record.stdout_path.write_bytes(SUCCESS_OUTPUT.read_bytes())
    execution_only = replace(
        NWCHEM,
        capabilities=NWCHEM.capabilities - {
            ProgramCapability.PROGRESS_INSPECT
        },
        progress=None,
    )

    monitored = monitor_run(
        execution_only,
        service,
        launch_id=launched.record.launch_id,
    )

    assert monitored["status"] == "running"
    assert monitored["evidence"]["scientific"] == {
        "status": "unavailable",
        "completion_observed": None,
        "outcome": None,
        "progress": None,
    }
    assert [item["code"] for item in monitored["uncertainty"]] == [
        "scientific_progress_unsupported"
    ]


def test_missing_slurm_job_remains_unresolved(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan_and_target(tmp_path, "slurm_cpu")

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
    service = _service(tmp_path)
    launched = service.launch(plan, target)

    monitored = monitor_run(
        NWCHEM,
        service,
        launch_id=launched.record.launch_id,
    )

    assert monitored["status"] == "not_found"
    assert monitored["launch"]["status"] == "submitted"
    scheduler = monitored["evidence"]["execution"]["scheduler"]
    assert scheduler["job_id"] == "85"
    assert scheduler["status"] == "not_found"
    assert scheduler["source"] == "accounting"
    assert {item["code"] for item in monitored["uncertainty"]} == {
        "execution_state_unresolved",
        "scientific_output_not_observed",
    }
    assert service.get_launch_record(launched.record.launch_id).status == (
        "submitted"
    )


def test_monitor_rejects_launch_from_another_service_instance(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan_and_target(tmp_path)

    class Process:
        pid = 4545

        def poll(self):
            return None

    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        lambda *args, **kwargs: Process(),
    )
    owner = _service(tmp_path)
    launched = owner.launch(plan, target)
    other = ExecutionService(
        enable_execution=True,
        registry_db_path=tmp_path / "registry.db",
    )

    with pytest.raises(MonitorRunError) as caught:
        monitor_run(
            NWCHEM,
            other,
            launch_id=launched.record.launch_id,
        )

    assert caught.value.as_dict() == {
        "error": "launch_not_owned",
        "message": (
            f"Chemtools does not own launch {launched.record.launch_id!r} "
            "in this server process"
        ),
        "launch_id": launched.record.launch_id,
    }


@pytest.mark.parametrize(
    "launch_id",
    ["not-a-uuid", "7C9A2D1E-0000-4000-8000-000000000000"],
)
def test_monitor_requires_canonical_launch_id(tmp_path, launch_id):
    with pytest.raises(MonitorRunError) as caught:
        monitor_run(
            NWCHEM,
            _service(tmp_path),
            launch_id=launch_id,
        )

    assert caught.value.code == "invalid_launch_id"


def test_monitor_rejects_unknown_launch_id(tmp_path):
    launch_id = "7c9a2d1e-0000-4000-8000-000000000000"

    with pytest.raises(MonitorRunError) as caught:
        monitor_run(
            NWCHEM,
            _service(tmp_path),
            launch_id=launch_id,
        )

    assert caught.value.as_dict() == {
        "error": "launch_not_owned",
        "message": (
            f"Chemtools does not own launch {launch_id!r} in this server process"
        ),
        "launch_id": launch_id,
    }

"""Owned local process polling and persisted launch-state transitions."""

from pathlib import Path

import pytest

import chemtools.execution.local as local_execution
from chemtools.application.execution import (
    ExecutionService,
    LaunchStatusError,
)
from chemtools.core.execution import LocalStatusResult
from chemtools.core.runner import load_runner_profiles
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
        "local",
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


def test_owned_local_status_moves_from_running_to_completed(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan_and_target(tmp_path)

    class Process:
        pid = 4242
        polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0

    process = Process()
    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    service = _service(tmp_path)
    launched = service.launch(plan, target)

    running = service.refresh_local_status(launched.record.launch_id)
    completed = service.refresh_local_status_external(process.pid)
    repeated = service.refresh_local_status(launched.record.launch_id)

    assert running.result == LocalStatusResult(
        process_id=4242,
        status="running",
        return_code=None,
        checked_at=running.result.checked_at,
    )
    assert running.record.status == "started"
    assert completed.result.status == "completed"
    assert completed.result.return_code == 0
    assert completed.record.status == "completed"
    assert completed.record.return_code == 0
    assert completed.record.elapsed_seconds is not None
    assert completed.record.elapsed_seconds >= 0
    assert completed.record.error is None
    assert service.get_launch_record(
        launched.record.launch_id
    ) == completed.record
    assert repeated.record == completed.record
    assert repeated.result.status == "completed"
    assert process.polls == 2


def test_nonzero_local_exit_is_persisted_as_failed(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan_and_target(tmp_path)

    class Process:
        pid = 4343

        def poll(self):
            return 17

    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        lambda *args, **kwargs: Process(),
    )
    service = _service(tmp_path)
    launched = service.launch(plan, target)

    failed = service.refresh_local_status(launched.record.launch_id)

    assert failed.result.status == "failed"
    assert failed.result.return_code == 17
    assert failed.record.status == "failed"
    assert failed.record.return_code == 17
    assert failed.record.error == "process exited with return code 17"


def test_local_status_rejects_unowned_process_id(tmp_path):
    service = _service(tmp_path)

    with pytest.raises(LaunchStatusError) as raised:
        service.refresh_local_status_external(999999)

    assert raised.value.as_dict() == {
        "error": "launch_not_owned",
        "identifier": "999999",
    }

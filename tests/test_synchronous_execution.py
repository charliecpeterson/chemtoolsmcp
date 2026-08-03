"""Contracts for permission-checked synchronous local execution."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import pytest

import chemtools.application.execution as application_execution
import chemtools.execution.local as local_execution
from chemtools.application.execution import ExecutionService
from chemtools.core.execution import (
    LocalSynchronousResult,
    RecordedSynchronousRun,
)
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
INSTANCE_ID = "00000000-0000-4000-8000-000000000001"


def test_service_closes_unused_stdin_and_records_completion(
    tmp_path,
    monkeypatch,
):
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
    plan = replace(
        build_nwchem_launch_plan(
            input_path,
            adapted.default_resources,
        ),
        expected_artifacts=(),
    )
    completed_at = datetime(
        2026,
        7,
        30,
        12,
        40,
        tzinfo=timezone.utc,
    )
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    ticks = iter((10.0, 12.5))

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="total energy = -75.0\n",
            stderr="",
        )

    monkeypatch.setattr(local_execution.subprocess, "run", fake_run)
    monkeypatch.setattr(
        local_execution.time,
        "monotonic",
        lambda: next(ticks),
    )
    monkeypatch.setattr(
        local_execution,
        "_utc_now",
        lambda: completed_at,
    )
    monkeypatch.setattr(
        application_execution,
        "_utc_now",
        lambda: completed_at,
    )
    service = ExecutionService(
        enable_execution=True,
        registry_db_path=tmp_path / "registry.db",
        instance_id=INSTANCE_ID,
    )

    recorded = service.run_to_completion(plan, adapted.target)

    assert isinstance(recorded, RecordedSynchronousRun)
    assert isinstance(recorded.result, LocalSynchronousResult)
    assert recorded.result.status == "completed"
    assert recorded.result.return_code == 0
    assert recorded.result.stdout == "total energy = -75.0\n"
    assert recorded.result.stderr == ""
    assert recorded.result.elapsed_seconds == 2.5
    assert recorded.record.status == "completed"
    assert recorded.record.return_code == 0
    assert recorded.record.elapsed_seconds == 2.5
    assert recorded.record.stdin_sha256 is None
    assert recorded.record.stdin_size_bytes is None
    assert calls[0][0] == ("nwchem", "water.nw")
    assert calls[0][1]["stdin"] == subprocess.DEVNULL
    assert "input" not in calls[0][1]
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["text"] is True
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["check"] is False
    assert calls[0][1]["timeout"] is None
    with pytest.raises(
        ValueError,
        match="terminal synchronous records require return_code",
    ):
        replace(recorded.record, return_code=None)
    with pytest.raises(
        ValueError,
        match="terminal synchronous records require elapsed_seconds",
    ):
        replace(recorded.record, elapsed_seconds=None)

"""Synchronous GRASP entrypoint and structured-workflow execution."""

import hashlib
from pathlib import Path
import subprocess

import pytest

import chemtools.execution.local as local_execution
from chemtools.application.execution import ExecutionService
from chemtools.application.grasp_execution import (
    run_grasp_exe_with_service,
    run_grasp_workflow_with_service,
)
from chemtools.execution import LocalExecutor
from chemtools.programs.grasp.launch import (
    build_grasp_interactive_launch_plan,
    build_grasp_interactive_target,
)


def _service(tmp_path: Path) -> ExecutionService:
    return ExecutionService(
        enable_execution=True,
        registry_db_path=tmp_path / "registry.db",
    )


def test_grasp_interactive_plan_selects_reviewed_entrypoint(tmp_path):
    target = build_grasp_interactive_target(
        tmp_path,
        container="/containers/grasp.sif",
    )
    plan = build_grasp_interactive_launch_plan(
        "rlevels",
        working_directory=tmp_path,
        stdin_lines="",
        args=["levels.m"],
        timeout_seconds=30,
    )

    rendered = LocalExecutor().render(plan, target)

    assert rendered.argv == (
        "apptainer",
        "exec",
        "/containers/grasp.sif",
        "rlevels",
        "levels.m",
    )
    assert rendered.stdin_text == "\n"
    assert rendered.timeout_seconds == 30
    assert rendered.stdout_path is None
    assert rendered.stderr_path is None
    with pytest.raises(TypeError):
        target.programs["grasp"].entrypoints["other"] = ("other",)


def test_grasp_interactive_plan_rejects_unknown_executable(tmp_path):
    with pytest.raises(
        ValueError,
        match="unsupported GRASP executable 'sh'",
    ):
        build_grasp_interactive_launch_plan(
            "sh",
            working_directory=tmp_path,
            stdin_lines="echo unsafe",
        )


def test_grasp_interactive_success_preserves_response_and_session_log(
    tmp_path,
    monkeypatch,
):
    observed: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed["input"] = kwargs["input"]
        observed["cwd"] = kwargs["cwd"]
        observed["capture_output"] = kwargs["capture_output"]
        observed["text"] = kwargs["text"]
        observed["shell"] = kwargs["shell"]
        observed["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="Nuclear data written\n",
            stderr="",
        )

    monkeypatch.setattr(local_execution.subprocess, "run", fake_run)
    service = _service(tmp_path)

    response = run_grasp_exe_with_service(
        service,
        "rnucleus",
        working_dir=str(tmp_path),
        stdin_lines=["90", "232"],
        container="/containers/grasp.sif",
        capture_log_file="rnucleus.log",
    )

    assert response["exe"] == "rnucleus"
    assert response["returncode"] == 0
    assert response["stdout"] == "Nuclear data written\n"
    assert response["stderr"] == ""
    assert response["command"] == (
        "apptainer exec /containers/grasp.sif rnucleus"
    )
    assert response["working_dir"] == str(tmp_path)
    assert response["log_file"] == str(tmp_path / "rnucleus.log")
    assert response["timed_out"] is False
    assert response["ok"] is True
    assert observed == {
        "argv": (
            "apptainer",
            "exec",
            "/containers/grasp.sif",
            "rnucleus",
        ),
        "input": "90\n232\n",
        "cwd": tmp_path,
        "capture_output": True,
        "text": True,
        "shell": False,
        "timeout": 600.0,
    }
    record = service.get_launch_record(response["launch_id"])
    assert record.status == "completed"
    assert record.return_code == 0
    assert record.stdin_sha256 == hashlib.sha256(
        b"90\n232\n"
    ).hexdigest()
    assert record.stdin_size_bytes == 7
    assert record.elapsed_seconds is not None
    assert (tmp_path / "rnucleus.log").read_text(
        encoding="utf-8"
    ) == "Nuclear data written\n"
    session = (tmp_path / "grasp_session.md").read_text(
        encoding="utf-8"
    )
    assert "`rnucleus`" in session
    assert "90\n232" in session


def test_grasp_interactive_failure_is_persisted(tmp_path, monkeypatch):
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            7,
            stdout="partial output\n",
            stderr="bad orbital selection\n",
        )

    monkeypatch.setattr(local_execution.subprocess, "run", fake_run)
    service = _service(tmp_path)

    response = run_grasp_exe_with_service(
        service,
        "rmcdhf",
        working_dir=str(tmp_path),
        stdin_lines="y",
        container="/containers/grasp.sif",
        log_to_session=False,
    )

    assert response["returncode"] == 7
    assert response["stdout"] == "partial output\n"
    assert response["stderr"] == "bad orbital selection\n"
    assert response["timed_out"] is False
    assert response["ok"] is False
    record = service.get_launch_record(response["launch_id"])
    assert record.status == "failed"
    assert record.return_code == 7
    assert record.error == "bad orbital selection\n"
    assert not (tmp_path / "grasp_session.md").exists()


def test_grasp_interactive_timeout_preserves_partial_output(
    tmp_path,
    monkeypatch,
):
    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(
            argv,
            2,
            output=b"SCF iteration 4\n",
            stderr=b"still running\n",
        )

    monkeypatch.setattr(local_execution.subprocess, "run", timeout)
    service = _service(tmp_path)

    response = run_grasp_exe_with_service(
        service,
        "rmcdhf",
        working_dir=str(tmp_path),
        stdin_lines="y",
        timeout_seconds=2,
        container="/containers/grasp.sif",
        log_to_session=False,
    )

    assert response["returncode"] == -1
    assert response["stdout"] == "SCF iteration 4\n"
    assert response["stderr"] == "still running\n"
    assert response["timed_out"] is True
    assert response["ok"] is False
    record = service.get_launch_record(response["launch_id"])
    assert record.status == "timed_out"
    assert record.return_code == -1
    assert record.error == "still running\n"


def test_grasp_interactive_disabled_before_directory_or_process(
    tmp_path,
    monkeypatch,
):
    working_directory = tmp_path / "new-run"

    def unexpected_run(*args, **kwargs):
        raise AssertionError("disabled execution reached subprocess")

    monkeypatch.setattr(
        local_execution.subprocess,
        "run",
        unexpected_run,
    )

    with pytest.raises(PermissionError, match="execution is disabled"):
        run_grasp_exe_with_service(
            ExecutionService(),
            "rnucleus",
            working_dir=str(working_directory),
            stdin_lines=["90", "232"],
            container="/containers/grasp.sif",
        )

    assert not working_directory.exists()


def test_grasp_capture_log_cannot_escape_before_execution(
    tmp_path,
    monkeypatch,
):
    def unexpected_run(*args, **kwargs):
        raise AssertionError("unsafe capture path reached subprocess")

    monkeypatch.setattr(
        local_execution.subprocess,
        "run",
        unexpected_run,
    )

    with pytest.raises(
        ValueError,
        match="capture log must be a file inside",
    ):
        run_grasp_exe_with_service(
            _service(tmp_path),
            "rnucleus",
            working_dir=str(tmp_path),
            stdin_lines=["90", "232"],
            container="/containers/grasp.sif",
            capture_log_file="../outside.log",
        )

    assert not (tmp_path.parent / "outside.log").exists()


def test_grasp_capture_log_rejects_working_directory_before_execution(
    tmp_path,
    monkeypatch,
):
    def unexpected_run(*args, **kwargs):
        raise AssertionError("invalid capture path reached subprocess")

    monkeypatch.setattr(
        local_execution.subprocess,
        "run",
        unexpected_run,
    )

    with pytest.raises(
        ValueError,
        match="capture log must be a file inside",
    ):
        run_grasp_exe_with_service(
            _service(tmp_path),
            "rnucleus",
            working_dir=str(tmp_path),
            stdin_lines=["90", "232"],
            container="/containers/grasp.sif",
            capture_log_file=".",
        )


def test_grasp_structured_workflow_uses_service_and_typed_copies(
    tmp_path,
    monkeypatch,
):
    calls: list[tuple[str, ...]] = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[-1] == "rcsfgenerate":
            (tmp_path / "rcsf.out").write_text(
                "generated CSFs\n",
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=f"{argv[-1]} complete\n",
            stderr="",
        )

    monkeypatch.setattr(local_execution.subprocess, "run", fake_run)
    service = _service(tmp_path)
    plan = {
        "workflow": "test",
        "name": "Th",
        "n_steps": 2,
        "steps": [
            {
                "exe": "rcsfgenerate",
                "stdin": ["0"],
                "args": [],
                "post": ["cp rcsf.out rcsf.inp"],
            },
            {
                "exe": "rangular",
                "stdin": ["y"],
                "args": [],
                "post": [],
            },
        ],
    }

    response = run_grasp_workflow_with_service(
        service,
        plan,
        working_dir=str(tmp_path),
        container="/containers/grasp.sif",
    )

    assert response["ok"] is True
    assert response["n_steps_attempted"] == 2
    assert (tmp_path / "rcsf.inp").read_text(
        encoding="utf-8"
    ) == "generated CSFs\n"
    assert calls == [
        (
            "apptainer",
            "exec",
            "/containers/grasp.sif",
            "rcsfgenerate",
        ),
        (
            "apptainer",
            "exec",
            "/containers/grasp.sif",
            "rangular",
        ),
    ]
    records = [
        service.get_launch_record(step["launch_id"])
        for step in response["transcript"]
    ]
    assert [record.status for record in records] == [
        "completed",
        "completed",
    ]

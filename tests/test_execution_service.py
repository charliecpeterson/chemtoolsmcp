"""Permission, launch recording, rendering, and cancellation contracts."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from uuid import UUID

import pytest

import chemtools.application.execution as application_execution
import chemtools.execution.local as local_execution
import chemtools.execution.slurm as slurm_execution
from chemtools.application.execution import (
    ExecutionDecision,
    ExecutionDisabledError,
    ExecutionService,
    LaunchCancellationError,
)
from chemtools.core.execution import (
    LocalCancellationResult,
    LocalLaunchResult,
    RecordedCancellation,
    RecordedLaunch,
    RenderedCommand,
    RenderedSlurmScript,
    SlurmCancellationResult,
    SlurmSubmissionResult,
    StagedFile,
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
LAUNCH_ID = "00000000-0000-4000-8000-000000000002"


def _adapted_target(tmp_path: Path, profile_name: str):
    profiles = load_runner_profiles(str(PROFILE_PATH))
    return adapt_legacy_nwchem_profile(
        profiles,
        profile_name,
        allowed_work_roots=(tmp_path,),
    )


def _plan(tmp_path: Path, profile_name: str):
    input_path = tmp_path / "water.nw"
    input_path.write_text(
        "start water\ngeometry\nO 0 0 0\nend\ntask scf energy\n",
        encoding="utf-8",
    )
    adapted = _adapted_target(tmp_path, profile_name)
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
        instance_id=INSTANCE_ID,
    )


def test_execution_permission_defaults_to_disabled(tmp_path):
    target = _adapted_target(tmp_path, "local").target

    decision = ExecutionService().check("launch", target)

    assert decision == ExecutionDecision(
        allowed=False,
        operation="launch",
        target="local",
        error="execution_disabled",
    )
    assert decision.as_dict() == {
        "allowed": False,
        "error": "execution_disabled",
        "operation": "launch",
        "target": "local",
    }


def test_disabled_permission_returns_exact_structured_error(tmp_path):
    target = _adapted_target(tmp_path, "slurm_cpu").target

    with pytest.raises(ExecutionDisabledError) as raised:
        ExecutionService().require("cancel", target)

    assert str(raised.value) == (
        "execution is disabled for cancel on target 'slurm_cpu'"
    )
    assert raised.value.as_dict() == {
        "error": "execution_disabled",
        "operation": "cancel",
        "target": "slurm_cpu",
    }


def test_enabled_permission_allows_launch_and_cancel(tmp_path):
    target = _adapted_target(tmp_path, "local").target
    service = ExecutionService(enable_execution=True)

    assert service.require("launch", target) == ExecutionDecision(
        allowed=True,
        operation="launch",
        target="local",
    )
    assert service.require("cancel", target) == ExecutionDecision(
        allowed=True,
        operation="cancel",
        target="local",
    )


def test_rendering_remains_available_when_execution_is_disabled(tmp_path):
    local_plan, local_target = _plan(tmp_path, "local_mpirun")
    slurm_plan, slurm_target = _plan(tmp_path, "slurm_cpu")
    service = ExecutionService()

    local = service.render(local_plan, local_target)
    slurm = service.render(slurm_plan, slurm_target)

    assert isinstance(local, RenderedCommand)
    assert local.argv == (
        "mpirun",
        "-np",
        "8",
        "nwchem",
        "water.nw",
    )
    assert isinstance(slurm, RenderedSlurmScript)
    assert slurm.command.argv == ("srun", "nwchem", "water.nw")
    assert slurm.submit_argv == (
        "sbatch",
        str(tmp_path / "water.job"),
    )


def test_permission_service_rejects_unknown_operation(tmp_path):
    target = _adapted_target(tmp_path, "local").target

    with pytest.raises(
        ValueError,
        match="execution operation must be 'launch' or 'cancel'",
    ):
        ExecutionService().check("status", target)  # type: ignore[arg-type]


def test_execution_decisions_are_immutable(tmp_path):
    target = _adapted_target(tmp_path, "local").target
    decision = ExecutionService().check("launch", target)

    with pytest.raises(AttributeError):
        decision.allowed = True


def test_disabled_service_refuses_launch_before_process_call(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan(tmp_path, "local")

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("disabled execution reached subprocess.Popen")

    monkeypatch.setattr(local_execution.subprocess, "Popen", unexpected_popen)

    with pytest.raises(ExecutionDisabledError) as raised:
        ExecutionService().launch(plan, target)

    assert raised.value.as_dict() == {
        "error": "execution_disabled",
        "operation": "launch",
        "target": "local",
    }
    assert not (tmp_path / "water.out").exists()
    assert not (tmp_path / "water.err").exists()


def test_enabled_service_launches_local_command_without_shell(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan(tmp_path, "local_mpirun")
    launched: dict[str, object] = {}
    started_at = datetime(2026, 7, 30, 12, 30, tzinfo=timezone.utc)

    class StartedProcess:
        pid = 4242

    def fake_popen(argv, **kwargs):
        launched["argv"] = argv
        launched["cwd"] = kwargs["cwd"]
        launched["omp_threads"] = kwargs["env"]["OMP_NUM_THREADS"]
        launched["stdin"] = kwargs["stdin"]
        launched["stdout"] = Path(kwargs["stdout"].name)
        launched["stderr"] = Path(kwargs["stderr"].name)
        launched["shell"] = kwargs["shell"]
        return StartedProcess()

    monkeypatch.setattr(local_execution.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(local_execution, "_utc_now", lambda: started_at)

    service = _service(tmp_path)
    recorded = service.launch(plan, target)
    result = recorded.result

    assert isinstance(recorded, RecordedLaunch)
    assert isinstance(result, LocalLaunchResult)
    assert result.process_id == 4242
    assert result.status == "started"
    assert result.started_at == started_at
    assert str(UUID(recorded.record.launch_id)) == recorded.record.launch_id
    assert recorded.record.instance_id == INSTANCE_ID
    assert recorded.record.target == "local_mpirun"
    assert recorded.record.executor == "local"
    assert recorded.record.program == "nwchem"
    assert recorded.record.working_directory == tmp_path
    assert recorded.record.argv == (
        "mpirun",
        "-np",
        "8",
        "nwchem",
        "water.nw",
    )
    assert recorded.record.environment_keys == ("OMP_NUM_THREADS",)
    assert recorded.record.resources.mpi_ranks == 8
    assert recorded.record.stdout_path == tmp_path / "water.out"
    assert recorded.record.stderr_path == tmp_path / "water.err"
    assert recorded.record.script_path is None
    assert recorded.record.status == "started"
    assert recorded.record.process_id == 4242
    assert recorded.record.job_id is None
    assert service.get_launch_record(recorded.record.launch_id) == (
        recorded.record
    )
    assert launched == {
        "argv": (
            "mpirun",
            "-np",
            "8",
            "nwchem",
            "water.nw",
        ),
        "cwd": tmp_path,
        "omp_threads": "1",
        "stdin": subprocess.DEVNULL,
        "stdout": tmp_path / "water.out",
        "stderr": tmp_path / "water.err",
        "shell": False,
    }


def test_enabled_service_submits_slurm_script_without_shell(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan(tmp_path, "slurm_cpu")
    submitted: dict[str, object] = {}
    submitted_at = datetime(2026, 7, 30, 12, 45, tzinfo=timezone.utc)

    def fake_run(argv, **kwargs):
        submitted["argv"] = argv
        submitted["cwd"] = kwargs["cwd"]
        submitted["capture_output"] = kwargs["capture_output"]
        submitted["text"] = kwargs["text"]
        submitted["shell"] = kwargs["shell"]
        submitted["check"] = kwargs["check"]
        submitted["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="Submitted batch job 8675309\n",
            stderr="",
        )

    monkeypatch.setattr(slurm_execution.subprocess, "run", fake_run)
    monkeypatch.setattr(slurm_execution, "_utc_now", lambda: submitted_at)

    service = _service(tmp_path)
    recorded = service.launch(plan, target)
    result = recorded.result

    assert isinstance(recorded, RecordedLaunch)
    assert isinstance(result, SlurmSubmissionResult)
    assert result.status == "submitted"
    assert result.return_code == 0
    assert result.stdout == "Submitted batch job 8675309\n"
    assert result.stderr == ""
    assert result.job_id == "8675309"
    assert result.submitted_at == submitted_at
    assert recorded.record.instance_id == INSTANCE_ID
    assert recorded.record.target == "slurm_cpu"
    assert recorded.record.executor == "slurm"
    assert recorded.record.status == "submitted"
    assert recorded.record.argv == ("srun", "nwchem", "water.nw")
    assert recorded.record.environment_keys == ()
    assert recorded.record.resources.mpi_ranks == 16
    assert recorded.record.stdout_path == tmp_path / "water.out"
    assert recorded.record.stderr_path == tmp_path / "water.err"
    assert recorded.record.script_path == tmp_path / "water.job"
    assert recorded.record.process_id is None
    assert recorded.record.job_id == "8675309"
    assert service.get_launch_record(recorded.record.launch_id) == (
        recorded.record
    )
    assert submitted == {
        "argv": ("sbatch", str(tmp_path / "water.job")),
        "cwd": tmp_path,
        "capture_output": True,
        "text": True,
        "shell": False,
        "check": False,
        "timeout": 60,
    }
    assert (tmp_path / "water.job").read_text(encoding="utf-8") == (
        result.script.script_text
    )


def test_slurm_submit_failure_is_a_typed_result(tmp_path, monkeypatch):
    plan, target = _plan(tmp_path, "slurm_cpu")
    submitted_at = datetime(2026, 7, 30, 13, 0, tzinfo=timezone.utc)

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            1,
            stdout="",
            stderr="sbatch: invalid account\n",
        )

    monkeypatch.setattr(slurm_execution.subprocess, "run", fake_run)
    monkeypatch.setattr(slurm_execution, "_utc_now", lambda: submitted_at)

    recorded = _service(tmp_path).launch(plan, target)
    result = recorded.result

    assert isinstance(recorded, RecordedLaunch)
    assert isinstance(result, SlurmSubmissionResult)
    assert result.status == "submit_failed"
    assert result.return_code == 1
    assert result.stdout == ""
    assert result.stderr == "sbatch: invalid account\n"
    assert result.job_id is None
    assert result.submitted_at == submitted_at
    assert recorded.record.status == "submit_failed"
    assert recorded.record.error == "sbatch: invalid account\n"


def test_state_changing_launch_refuses_existing_files(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan(tmp_path, "local")
    output_path = tmp_path / "water.out"
    output_path.write_text("prior result\n", encoding="utf-8")

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("overwrite refusal reached subprocess.Popen")

    monkeypatch.setattr(local_execution.subprocess, "Popen", unexpected_popen)
    monkeypatch.setattr(
        application_execution,
        "uuid4",
        lambda: UUID(LAUNCH_ID),
    )
    service = _service(tmp_path)

    with pytest.raises(
        FileExistsError,
        match="refusing to overwrite launch output",
    ):
        service.launch(plan, target)

    assert output_path.read_text(encoding="utf-8") == "prior result\n"
    failed = service.get_launch_record(LAUNCH_ID)
    assert failed.status == "launch_failed"
    assert failed.process_id is None
    assert failed.job_id is None
    assert failed.error == (
        f"FileExistsError: refusing to overwrite launch output: "
        f"{output_path}"
    )


def test_slurm_submission_refuses_existing_script(tmp_path, monkeypatch):
    plan, target = _plan(tmp_path, "slurm_cpu")
    script_path = tmp_path / "water.job"
    script_path.write_text("prior script\n", encoding="utf-8")

    def unexpected_run(*args, **kwargs):
        raise AssertionError("overwrite refusal reached subprocess.run")

    monkeypatch.setattr(slurm_execution.subprocess, "run", unexpected_run)

    with pytest.raises(
        FileExistsError,
        match="refusing to overwrite scheduler script",
    ):
        _service(tmp_path).launch(plan, target)

    assert script_path.read_text(encoding="utf-8") == "prior script\n"


def test_state_changing_launch_copies_staged_file_before_process(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan(tmp_path, "local")
    auxiliary = tmp_path / "basis.dat"
    auxiliary.write_text("basis\n", encoding="utf-8")
    staged_plan = replace(
        plan,
        staged_files=(
            StagedFile(
                source=auxiliary,
                destination=Path("staged-basis.dat"),
            ),
        ),
    )

    class StartedProcess:
        pid = 4343

    def fake_popen(*args, **kwargs):
        assert (tmp_path / "staged-basis.dat").read_text(
            encoding="utf-8"
        ) == "basis\n"
        return StartedProcess()

    monkeypatch.setattr(local_execution.subprocess, "Popen", fake_popen)

    launched = _service(tmp_path).launch(staged_plan, target)

    assert launched.record.process_id == 4343
    assert launched.record.staged_files == (
        StagedFile(
            source=auxiliary,
            destination=tmp_path / "staged-basis.dat",
        ),
    )
    assert (tmp_path / "staged-basis.dat").read_bytes() == b"basis\n"


def test_local_cancellation_requires_record_and_live_process_handle(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan(tmp_path, "local")
    cancelled_at = datetime(2026, 7, 30, 14, 0, tzinfo=timezone.utc)

    class RunningProcess:
        pid = 4242
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    process = RunningProcess()
    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        local_execution,
        "_utc_now",
        lambda: cancelled_at,
    )
    monkeypatch.setattr(
        application_execution,
        "uuid4",
        lambda: UUID(LAUNCH_ID),
    )
    service = _service(tmp_path)
    launched = service.launch(plan, target)

    cancelled = service.cancel(launched.record.launch_id, target)

    assert isinstance(cancelled, RecordedCancellation)
    assert isinstance(cancelled.result, LocalCancellationResult)
    assert cancelled.result == LocalCancellationResult(
        process_id=4242,
        status="cancelled",
        signal="SIGTERM",
        error=None,
        cancelled_at=cancelled_at,
    )
    assert process.terminated is True
    assert cancelled.record.status == "cancelled"
    assert cancelled.record.process_id == 4242
    assert service.get_launch_record(LAUNCH_ID) == cancelled.record

    with pytest.raises(LaunchCancellationError) as raised:
        service.cancel(LAUNCH_ID, target)
    assert raised.value.as_dict() == {
        "error": "launch_not_cancelable",
        "launch_id": LAUNCH_ID,
        "target": "local",
        "status": "cancelled",
    }


def test_slurm_cancellation_uses_recorded_job_and_target_command(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan(tmp_path, "slurm_cpu")
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    cancelled_at = datetime(2026, 7, 30, 14, 15, tzinfo=timezone.utc)

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[0] == "sbatch":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="Submitted batch job 8675309\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(slurm_execution.subprocess, "run", fake_run)
    monkeypatch.setattr(
        slurm_execution,
        "_utc_now",
        lambda: cancelled_at,
    )
    monkeypatch.setattr(
        application_execution,
        "uuid4",
        lambda: UUID(LAUNCH_ID),
    )
    service = _service(tmp_path)
    launched = service.launch(plan, target)

    cancelled = service.cancel(launched.record.launch_id, target)

    assert isinstance(cancelled.result, SlurmCancellationResult)
    assert cancelled.result == SlurmCancellationResult(
        job_id="8675309",
        argv=("scancel", "8675309"),
        status="cancelled",
        return_code=0,
        stdout="",
        stderr="",
        cancelled_at=cancelled_at,
    )
    assert cancelled.record.status == "cancelled"
    assert cancelled.record.job_id == "8675309"
    assert calls[1][0] == ("scancel", "8675309")
    assert calls[1][1]["capture_output"] is True
    assert calls[1][1]["text"] is True
    assert calls[1][1]["shell"] is False
    assert calls[1][1]["check"] is False
    assert calls[1][1]["timeout"] == 30


def test_different_service_instance_cannot_cancel_persisted_launch(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan(tmp_path, "local")

    class RunningProcess:
        pid = 4242
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    process = RunningProcess()
    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    owner = _service(tmp_path)
    launched = owner.launch(plan, target)
    other = ExecutionService(
        enable_execution=True,
        registry_db_path=tmp_path / "registry.db",
        instance_id="00000000-0000-4000-8000-000000000003",
    )

    with pytest.raises(LaunchCancellationError) as raised:
        other.cancel(launched.record.launch_id, target)

    assert raised.value.as_dict() == {
        "error": "launch_not_owned",
        "launch_id": launched.record.launch_id,
        "target": "local",
    }
    assert process.terminated is False


def test_cancellation_rejects_target_mismatch(tmp_path, monkeypatch):
    plan, target = _plan(tmp_path, "local")
    other_target = _adapted_target(tmp_path, "local_mpirun").target

    class RunningProcess:
        pid = 4242
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    process = RunningProcess()
    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    service = _service(tmp_path)
    launched = service.launch(plan, target)

    with pytest.raises(LaunchCancellationError) as raised:
        service.cancel(launched.record.launch_id, other_target)

    assert raised.value.as_dict() == {
        "error": "launch_target_mismatch",
        "launch_id": launched.record.launch_id,
        "target": "local_mpirun",
        "recorded_target": "local",
    }
    assert process.terminated is False


def test_failed_submission_is_recorded_but_not_cancelable(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan(tmp_path, "slurm_cpu")

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            1,
            stdout="",
            stderr="sbatch: invalid account\n",
        )

    monkeypatch.setattr(slurm_execution.subprocess, "run", fake_run)
    service = _service(tmp_path)
    launched = service.launch(plan, target)

    with pytest.raises(LaunchCancellationError) as raised:
        service.cancel(launched.record.launch_id, target)

    assert raised.value.as_dict() == {
        "error": "launch_not_cancelable",
        "launch_id": launched.record.launch_id,
        "target": "slurm_cpu",
        "status": "submit_failed",
    }

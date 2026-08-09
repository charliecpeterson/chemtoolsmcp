"""NWChem compatibility responses backed by the typed execution service."""

import hashlib
from pathlib import Path
import subprocess

import chemtools.execution.local as local_execution
import chemtools.execution.slurm as slurm_execution
import pytest
from chemtools.application.execution import ExecutionService
from chemtools.application.nwchem_execution import (
    launch_nwchem_with_service,
    register_nwchem_launch_with_service,
    render_nwchem_job_script,
    terminate_nwchem_with_service,
)
from chemtools.persistence.artifacts import load_run_artifacts
from chemtools.persistence.launches import load_execution_run_link
from chemtools.persistence.runs import get_run_summary
from chemtools.mcp.decorator import set_active_mode
from chemtools.mcp.tools.nwchem_jobs import (
    _handle_get_nwchem_run_status,
    _handle_launch_nwchem_run,
    _handle_terminate_nwchem_run,
)


PROFILE_PATH = (
    Path(__file__).parents[1]
    / "chemtools"
    / "runner_profiles.example.json"
)


def _input_file(tmp_path: Path) -> Path:
    input_path = tmp_path / "water.nw"
    input_path.write_text(
        "start water\ngeometry\nO 0 0 0\nend\ntask scf energy\n",
        encoding="utf-8",
    )
    return input_path


def _service(tmp_path: Path) -> ExecutionService:
    return ExecutionService(
        enable_execution=True,
        registry_db_path=tmp_path / "registry.db",
    )


def test_dry_run_uses_the_typed_command_without_registry_writes(tmp_path):
    input_path = _input_file(tmp_path)
    actual = launch_nwchem_with_service(
        ExecutionService(),
        input_path=str(input_path),
        profile="local_mpirun",
        profiles_path=str(PROFILE_PATH),
        resource_overrides={"mpi_ranks": 3},
        env_overrides={"OMP_NUM_THREADS": "2"},
        dry_run=True,
    )

    assert actual == {
        "profile": "local_mpirun",
        "profiles_path": str(PROFILE_PATH),
        "launcher_kind": "direct",
        "input_file": str(input_path),
        "job_name": "water",
        "working_directory": str(tmp_path),
        "shell": "/bin/bash",
        "output_file": str(tmp_path / "water.out"),
        "error_file": str(tmp_path / "water.err"),
        "restart_prefix": "water",
        "resources": {
            "mpi_ranks": 3,
            "omp_threads": 1,
            "memory": None,
            "walltime": None,
            "node_memory_mb": None,
            "max_walltime": None,
            "cores_per_node": None,
            "max_nodes": None,
            "cpu_arch": None,
        },
        "executed": False,
        "launcher_command": "mpirun -np 3 nwchem",
        "command": (
            "mpirun -np 3 nwchem water.nw > water.out 2> water.err"
        ),
    }
    assert not (tmp_path / "registry.db").exists()


def test_environment_overrides_reach_the_typed_local_executor(
    tmp_path,
    monkeypatch,
):
    input_path = _input_file(tmp_path)
    observed = {}

    class RunningProcess:
        pid = 4141

        def poll(self):
            return None

    def fake_popen(*args, **kwargs):
        observed["environment"] = kwargs["env"]
        return RunningProcess()

    monkeypatch.setattr(local_execution.subprocess, "Popen", fake_popen)

    launch_nwchem_with_service(
        _service(tmp_path),
        input_path=str(input_path),
        profile="local_mpirun",
        profiles_path=str(PROFILE_PATH),
        resource_overrides={"mpi_ranks": 3},
        env_overrides={"OMP_NUM_THREADS": "2", "GA_MEMORY": "1GiB"},
    )

    assert observed["environment"]["OMP_NUM_THREADS"] == "2"
    assert observed["environment"]["GA_MEMORY"] == "1GiB"


def test_slurm_dry_run_reports_the_typed_script_without_writing_it(tmp_path):
    input_path = _input_file(tmp_path)

    preview = launch_nwchem_with_service(
        ExecutionService(registry_db_path=tmp_path / "registry.db"),
        input_path=str(input_path),
        profile="slurm_cpu",
        profiles_path=str(PROFILE_PATH),
        resource_overrides={"mpi_ranks": 3},
        env_overrides={"OMP_NUM_THREADS": "2"},
        dry_run=True,
    )

    assert preview["submit_command"] == [
        "sbatch",
        str(tmp_path / "water.job"),
    ]
    assert preview["submit_script_text"] == (
        "#!/bin/bash\n"
        "#SBATCH --job-name=water\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --ntasks=3\n"
        "#SBATCH --cpus-per-task=1\n"
        f"#SBATCH --output={tmp_path / 'water.out'}\n"
        f"#SBATCH --error={tmp_path / 'water.err'}\n"
        "#SBATCH --time=24:00:00\n"
        "#SBATCH --partition=compute\n"
        "module purge\n"
        "module load nwchem\n"
        "export OMP_NUM_THREADS=2\n"
        f"cd -- {tmp_path}\n"
        "srun nwchem water.nw\n"
    )
    assert not (tmp_path / "water.job").exists()
    assert not (tmp_path / "registry.db").exists()


def test_job_script_projection_uses_the_typed_slurm_preview(tmp_path):
    input_path = _input_file(tmp_path)

    rendered = render_nwchem_job_script(
        input_path=str(input_path),
        profile="slurm_cpu",
        profiles_path=str(PROFILE_PATH),
        resource_overrides={"mpi_ranks": 3},
    )

    assert rendered["profile"] == "slurm_cpu"
    assert rendered["launcher_kind"] == "scheduler"
    assert rendered["scheduler_type"] == "slurm"
    assert rendered["script_path"] == str(tmp_path / "water.job")
    assert rendered["submit_command"] == [
        "sbatch",
        str(tmp_path / "water.job"),
    ]
    assert "#SBATCH --ntasks=3\n" in rendered["script_text"]
    assert rendered["resources"]["mpi_ranks"] == 3


def test_job_script_projection_rejects_a_direct_profile(tmp_path):
    input_path = _input_file(tmp_path)

    with pytest.raises(ValueError, match="direct/local launcher"):
        render_nwchem_job_script(
            input_path=str(input_path),
            profile="local",
            profiles_path=str(PROFILE_PATH),
        )


def test_local_launch_and_kill_keep_legacy_response_fields(
    tmp_path,
    monkeypatch,
):
    input_path = _input_file(tmp_path)

    class RunningProcess:
        pid = 4242
        killed = False

        def poll(self):
            return None

        def kill(self):
            self.killed = True

    process = RunningProcess()
    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    service = _service(tmp_path)

    launched = launch_nwchem_with_service(
        service,
        input_path=str(input_path),
        profile="local_mpirun",
        profiles_path=str(PROFILE_PATH),
        resource_overrides={"mpi_ranks": 3},
        env_overrides={"OMP_NUM_THREADS": "2"},
    )

    assert launched["profile"] == "local_mpirun"
    assert launched["launcher_kind"] == "direct"
    assert launched["input_file"] == str(input_path)
    assert launched["output_file"] == str(tmp_path / "water.out")
    assert launched["error_file"] == str(tmp_path / "water.err")
    assert launched["executed"] is True
    assert launched["process_id"] == 4242
    assert launched["status"] == "started"
    assert launched["effective_argv"] == [
        "mpirun",
        "-np",
        "3",
        "nwchem",
        "water.nw",
    ]
    assert service.get_launch_record(launched["launch_id"]).process_id == 4242

    cancelled = terminate_nwchem_with_service(
        service,
        process_id=4242,
        signal_name="kill",
    )

    assert cancelled == {
        "process_id": 4242,
        "signal": "SIGKILL",
        "sent": True,
        "error": None,
        "launch_id": launched["launch_id"],
    }
    assert process.killed is True


def test_local_termination_rejects_unrecorded_pid(tmp_path):
    result = terminate_nwchem_with_service(
        _service(tmp_path),
        process_id=999999,
    )

    assert result == {
        "process_id": 999999,
        "signal": "SIGTERM",
        "sent": False,
        "error": "launch_not_owned",
    }


def test_launch_validates_paths_before_archiving(tmp_path):
    input_path = _input_file(tmp_path)
    outside_name = f"{tmp_path.name}-outside"
    outside_output = tmp_path.parent / f"{outside_name}.out"
    outside_output.write_text("must remain\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid job_name"):
        launch_nwchem_with_service(
            _service(tmp_path),
            input_path=str(input_path),
            profile="local",
            profiles_path=str(PROFILE_PATH),
            job_name=f"../{outside_name}",
        )

    assert outside_output.read_text(encoding="utf-8") == "must remain\n"


def test_registration_rejects_launch_from_another_service_instance(
    tmp_path,
    monkeypatch,
):
    input_path = _input_file(tmp_path)

    class RunningProcess:
        pid = 4343

        def poll(self):
            return None

    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        lambda *args, **kwargs: RunningProcess(),
    )
    owner = _service(tmp_path)
    launched = launch_nwchem_with_service(
        owner,
        input_path=str(input_path),
        profile="local",
        profiles_path=str(PROFILE_PATH),
    )
    other = ExecutionService(
        enable_execution=True,
        registry_db_path=tmp_path / "registry.db",
    )

    with pytest.raises(
        ValueError,
        match="belongs to another service instance",
    ):
        register_nwchem_launch_with_service(
            other,
            launch_id=launched["launch_id"],
            job_name="water",
            input_file=str(input_path),
            profile="local",
        )

    assert get_run_summary(
        job_name="water",
        db_path=str(tmp_path / "registry.db"),
    ) is None


def test_slurm_launch_and_cancel_keep_legacy_response_fields(
    tmp_path,
    monkeypatch,
):
    input_path = _input_file(tmp_path)
    calls: list[tuple[str, ...]] = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
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
    service = _service(tmp_path)

    launched = launch_nwchem_with_service(
        service,
        input_path=str(input_path),
        profile="slurm_cpu",
        profiles_path=str(PROFILE_PATH),
        write_script=True,
    )

    script_path = tmp_path / "water.job"
    assert launched["profile"] == "slurm_cpu"
    assert launched["launcher_kind"] == "scheduler"
    assert launched["executed"] is True
    assert launched["status"] == "submitted"
    assert launched["job_id"] == "8675309"
    assert launched["return_code"] == 0
    assert launched["submit_command"] == ["sbatch", str(script_path)]
    assert launched["submit_script_path"] == str(script_path)
    assert launched["submit_script_text"] == script_path.read_text(
        encoding="utf-8"
    )
    assert launched["jobid_file"] == str(tmp_path / "water.jobid")
    assert (tmp_path / "water.jobid").read_text(
        encoding="utf-8"
    ) == "8675309"
    assert calls[0] == ("sbatch", str(script_path))

    cancelled = terminate_nwchem_with_service(
        service,
        job_id="8675309",
        profile="slurm_cpu",
    )

    assert cancelled == {
        "job_id": "8675309",
        "cancelled": True,
        "command": ["scancel", "8675309"],
        "return_code": 0,
        "stdout": "",
        "stderr": "",
        "launch_id": launched["launch_id"],
    }
    assert calls[1] == ("scancel", "8675309")


def test_slurm_timeout_returns_legacy_submission_failure(
    tmp_path,
    monkeypatch,
):
    input_path = _input_file(tmp_path)

    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 60)

    monkeypatch.setattr(slurm_execution.subprocess, "run", timeout)
    service = _service(tmp_path)

    launched = launch_nwchem_with_service(
        service,
        input_path=str(input_path),
        profile="slurm_cpu",
        profiles_path=str(PROFILE_PATH),
    )

    assert launched["executed"] is True
    assert launched["status"] == "submit_failed"
    assert launched["return_code"] == -1
    assert launched["stdout"] == ""
    assert launched["stderr"] == (
        "sbatch/qsub timed out after 60 seconds"
    )
    assert launched["job_id"] is None
    assert service.get_launch_record(
        launched["launch_id"]
    ).status == "submit_failed"


def test_mcp_handlers_share_launch_ownership(tmp_path, monkeypatch):
    input_path = _input_file(tmp_path)

    class RunningProcess:
        pid = 5252
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    process = RunningProcess()
    monkeypatch.setenv(
        "CHEMTOOLS_REGISTRY_DB",
        str(tmp_path / "registry.db"),
    )
    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    set_active_mode("local")
    try:
        launched = _handle_launch_nwchem_run({
            "input_file": str(input_path),
            "profile": "local",
            "profiles_path": str(PROFILE_PATH),
            "auto_watch": False,
            "auto_register": False,
        })
        cancelled = _handle_terminate_nwchem_run({
            "process_id": launched["process_id"],
        })
    finally:
        set_active_mode("analysis")

    assert launched["process_id"] == 5252
    assert cancelled["sent"] is True
    assert cancelled["launch_id"] == launched["launch_id"]
    assert process.terminated is True


def test_mcp_auto_registration_links_nwchem_run_to_launch(
    tmp_path,
    monkeypatch,
):
    input_path = _input_file(tmp_path)

    class RunningProcess:
        pid = 6262

        def poll(self):
            return None

    db_path = tmp_path / "registry.db"
    monkeypatch.setenv("CHEMTOOLS_REGISTRY_DB", str(db_path))
    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        lambda *args, **kwargs: RunningProcess(),
    )
    set_active_mode("local")
    try:
        launched = _handle_launch_nwchem_run({
            "input_file": str(input_path),
            "profile": "local",
            "profiles_path": str(PROFILE_PATH),
            "auto_watch": False,
        })
    finally:
        set_active_mode("analysis")

    registration = launched["registry"]
    link = load_execution_run_link(launched["launch_id"], db_path)
    run = get_run_summary(
        run_uid=registration["run_uid"],
        db_path=str(db_path),
    )

    assert registration["program"] == "nwchem"
    assert link.launch_id == launched["launch_id"]
    assert link.run_uid == registration["run_uid"]
    assert run["program"] == "nwchem"
    assert run["input_file"] == str(input_path)
    assert run["output_file"] == str(tmp_path / "water.out")
    assert run["mpi_ranks"] == 1


def test_mcp_local_status_completes_linked_run_and_observes_outputs(
    tmp_path,
    monkeypatch,
):
    input_path = _input_file(tmp_path)
    stdout_bytes = (
        b"Northwest Computational Chemistry Package (NWChem)\n"
        b"Total SCF energy = -75.000000\n"
    )
    stderr_bytes = b"launcher diagnostic\n"

    class CompletedProcess:
        pid = 7373
        polls = 0

        def poll(self):
            self.polls += 1
            return 0

    process = CompletedProcess()

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
            "auto_watch": False,
        })
        arguments = {
            "output_file": launched["output_file"],
            "input_file": str(input_path),
            "error_file": launched["error_file"],
            "process_id": process.pid,
        }
        status = _handle_get_nwchem_run_status(arguments)
        repeated = _handle_get_nwchem_run_status(arguments)
    finally:
        set_active_mode("analysis")

    run_uid = launched["registry"]["run_uid"]
    run = get_run_summary(run_uid=run_uid, db_path=str(db_path))
    artifacts = load_run_artifacts(run_uid, db_path)

    assert status["process"] == {
        "process_id": 7373,
        "status": "completed",
        "return_code": 0,
    }
    assert status["execution_record"]["launch_id"] == launched["launch_id"]
    assert status["execution_record"]["status"] == "completed"
    assert run["status"] == "completed"
    assert run["walltime_used_sec"] >= 0
    assert artifacts is not None
    assert tuple(
        (artifact.kind, sorted(role.value for role in artifact.roles))
        for artifact in artifacts.artifacts
    ) == (
        ("nwchem.output", ["primary_output", "stdout"]),
        ("nwchem.error", ["stderr"]),
    )
    assert tuple(
        (
            observation.exists,
            observation.size_bytes,
            observation.sha256,
            observation.hash_status,
        )
        for observation in artifacts.observations
    ) == (
        (
            True,
            len(stdout_bytes),
            hashlib.sha256(stdout_bytes).hexdigest(),
            "verified",
        ),
        (
            True,
            len(stderr_bytes),
            hashlib.sha256(stderr_bytes).hexdigest(),
            "verified",
        ),
    )
    assert repeated["execution_record"] == status["execution_record"]
    assert process.polls == 1
    assert len(artifacts.artifacts) == 2
    assert len(artifacts.observations) == 2


def test_mcp_slurm_status_uses_accounting_and_preserves_oom(
    tmp_path,
    monkeypatch,
):
    input_path = _input_file(tmp_path)
    stdout_bytes = b"Northwest Computational Chemistry Package (NWChem)\n"
    stderr_bytes = b"slurmstepd: error: Detected oom_kill event\n"
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[0] == "sbatch":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="Submitted batch job 9191\n",
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
            stdout="OUT_OF_MEMORY|0:9|77\n",
            stderr="",
        )

    db_path = tmp_path / "registry.db"
    monkeypatch.setenv("CHEMTOOLS_REGISTRY_DB", str(db_path))
    monkeypatch.setattr(
        slurm_execution.subprocess,
        "run",
        fake_run,
    )
    set_active_mode("hpc")
    try:
        launched = _handle_launch_nwchem_run({
            "input_file": str(input_path),
            "profile": "slurm_cpu",
            "profiles_path": str(PROFILE_PATH),
            "auto_watch": False,
        })
        Path(launched["output_file"]).write_bytes(stdout_bytes)
        Path(launched["error_file"]).write_bytes(stderr_bytes)
        arguments = {
            "output_file": launched["output_file"],
            "input_file": str(input_path),
            "error_file": launched["error_file"],
            "job_id": launched["job_id"],
            "profile": "slurm_cpu",
            "profiles_path": str(PROFILE_PATH),
        }
        status = _handle_get_nwchem_run_status(arguments)
        repeated = _handle_get_nwchem_run_status(arguments)
    finally:
        set_active_mode("analysis")

    run_uid = launched["registry"]["run_uid"]
    run = get_run_summary(run_uid=run_uid, db_path=str(db_path))
    artifacts = load_run_artifacts(run_uid, db_path)

    assert status["scheduler"]["status"] == "out_of_memory"
    assert status["scheduler"]["raw_state"] == "OUT_OF_MEMORY"
    assert status["scheduler"]["source"] == "accounting"
    assert status["scheduler"]["job_exit_code"] == 0
    assert status["scheduler"]["termination_signal"] == 9
    assert status["execution_record"]["status"] == "failed"
    assert status["overall_status"] == "completed_failed"
    assert run["status"] == "oom"
    assert run["walltime_used_sec"] == 77.0
    assert artifacts is not None
    assert tuple(
        observation.sha256
        for observation in artifacts.observations
    ) == (
        hashlib.sha256(stdout_bytes).hexdigest(),
        hashlib.sha256(stderr_bytes).hexdigest(),
    )
    assert repeated["scheduler"]["status"] == "out_of_memory"
    assert repeated["scheduler"]["source"] == "record"
    assert len(calls) == 3

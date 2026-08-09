"""OpenMolcas launch-plan, target, and compatibility contracts."""

import json
from pathlib import Path
import subprocess

import pytest

import chemtools.core.monitoring as core_monitoring
import chemtools.execution.legacy_runner as core_runner
import chemtools.execution.local as local_execution
import chemtools.execution.slurm as slurm_execution
from chemtools.application.execution import ExecutionService
from chemtools.application.molcas_execution import (
    launch_molcas_with_service,
    terminate_molcas_with_service,
)
from chemtools.execution.legacy_runner import load_runner_profiles
from chemtools.execution import LocalExecutor, SlurmExecutor
from chemtools.persistence.launches import load_launch_record
from chemtools.mcp.decorator import set_active_mode
from chemtools.mcp.tools.molcas import (
    _handle_get_molcas_run_status,
    _handle_launch_molcas_run,
    _handle_watch_molcas_run,
)
from chemtools.programs.molcas.launch import (
    adapt_legacy_molcas_profile,
    build_molcas_launch_plan,
)
from chemtools.programs.molcas.runtime import prepare_launch
from chemtools.programs.molcas.scheduler import launch_molcas_run


PROFILE_PATH = (
    Path(__file__).parents[1]
    / "chemtools"
    / "runner_profiles.example.yaml"
)
STAMPEDE_PROFILE_PATH = (
    Path(__file__).parents[1]
    / "examples"
    / "tacc_stampede3"
    / "runner_profiles.yaml"
)


def _input(tmp_path: Path, *, caspt2: bool = False) -> Path:
    input_path = tmp_path / "complex.input"
    modules = "&GATEWAY\nTitle=Complex\n&SEWARD\n&RASSCF\n"
    if caspt2:
        modules += "&CASPT2\n"
    input_path.write_text(modules, encoding="utf-8")
    return input_path


def _service(tmp_path: Path) -> ExecutionService:
    return ExecutionService(
        enable_execution=True,
        registry_db_path=tmp_path / "registry.db",
    )


def _slurm_profile_path(tmp_path: Path) -> Path:
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(json.dumps({
        "schema_version": "1.0",
        "profiles": {
            "molcas_slurm": {
                "launcher": {
                    "kind": "scheduler",
                    "scheduler_type": "slurm",
                    "submit_command": "sbatch",
                    "status_command": "squeue -j {job_id} -h -o %T",
                    "cancel_command": "scancel {job_id}",
                    "job_id_regex": "Submitted batch job (\\d+)",
                },
                "scheduler": {
                    "system": "slurm",
                    "submit_script_name": "{job_name}.job",
                    "script_template": "",
                },
                "execution": {
                    "apptainer_sif": "/containers/molcas.sif",
                    "pymolcas_command": "pymolcas",
                    "parallel_caspt2_supported": False,
                },
                "resources": {
                    "nodes": 2,
                    "mpi_ranks": 48,
                    "omp_threads": 1,
                    "walltime": "02:00:00",
                    "partition": "compute",
                },
                "modules": {"load": ["tacc-apptainer"]},
                "hooks": {
                    "pre_run": [
                        'export MOLCAS_WORKDIR="$SCRATCH/molcas/'
                        '$MOLCAS_PROJECT"',
                        'mkdir -p "$MOLCAS_WORKDIR"',
                    ],
                },
                "file_rules": {
                    "output_file": "{job_name}.log",
                    "error_file": "{job_name}.err",
                },
            },
        },
    }), encoding="utf-8")
    return profile_path


def test_local_molcas_plan_matches_safe_runtime_boundary(tmp_path):
    input_path = _input(tmp_path)
    profiles = load_runner_profiles(str(PROFILE_PATH))
    adapted = adapt_legacy_molcas_profile(
        profiles,
        "molcas_local_native",
        allowed_work_roots=(tmp_path,),
    )
    prepared = build_molcas_launch_plan(
        input_path,
        adapted.default_resources,
        parallel_caspt2_supported=(
            adapted.parallel_caspt2_supported
        ),
        output_template=adapted.output_template,
        error_template=adapted.error_template,
    )

    rendered = LocalExecutor().render(
        prepared.plan,
        adapted.target,
    )
    legacy = prepare_launch(
        input_path,
        profile={
            "execution": {
                "parallel_caspt2_supported": True,
            }
        },
        requested_np=2,
    )

    assert rendered.argv == (
        "pymolcas",
        "-np",
        "2",
        "complex.input",
    )
    assert rendered.argv[:-1] == tuple(legacy["command"][:-1])
    assert rendered.environment == {
        "MOLCAS_PROJECT": "complex",
        "MOLCAS_NPROCS": "2",
    }
    assert rendered.stdout_path == tmp_path / "complex.out"
    assert rendered.stderr_path == tmp_path / "complex.err"
    assert prepared.requested_mpi_ranks == 2
    assert prepared.effective_mpi_ranks == 2
    assert prepared.has_caspt2 is False
    assert prepared.warnings == ()


def test_molcas_caspt2_guard_changes_plan_resources_and_command(tmp_path):
    input_path = _input(tmp_path, caspt2=True)
    profiles = load_runner_profiles(str(PROFILE_PATH))
    adapted = adapt_legacy_molcas_profile(
        profiles,
        "molcas_apptainer_broken_caspt2",
        allowed_work_roots=(tmp_path,),
    )
    prepared = build_molcas_launch_plan(
        input_path,
        adapted.default_resources,
        parallel_caspt2_supported=(
            adapted.parallel_caspt2_supported
        ),
        output_template=adapted.output_template,
        error_template=adapted.error_template,
    )

    rendered = LocalExecutor().render(
        prepared.plan,
        adapted.target,
    )

    assert prepared.requested_mpi_ranks == 4
    assert prepared.effective_mpi_ranks == 1
    assert prepared.plan.resources.nodes == 1
    assert prepared.plan.resources.mpi_ranks == 1
    assert prepared.has_caspt2 is True
    assert prepared.parallel_caspt2_supported is False
    assert len(prepared.warnings) == 1
    assert "forcing -np 1 (requested 4)" in prepared.warnings[0]
    assert rendered.argv == (
        "apptainer",
        "exec",
        "/path/to/openmolcas-25.02.sif",
        "pymolcas",
        "-np",
        "1",
        "complex.input",
    )
    assert rendered.environment["MOLCAS_NPROCS"] == "1"


def test_slurm_molcas_plan_keeps_runtime_rules_out_of_scheduler(
    tmp_path,
):
    input_path = _input(tmp_path, caspt2=True)
    profiles = {
        "schema_version": "1.0",
        "profiles": {
            "molcas_slurm": {
                "launcher": {
                    "kind": "scheduler",
                    "scheduler_type": "slurm",
                    "submit_command": "sbatch",
                    "status_command": (
                        "squeue -j {job_id} -h -o %T"
                    ),
                    "cancel_command": "scancel {job_id}",
                },
                "scheduler": {
                    "system": "slurm",
                    "submit_script_name": "{job_name}.job",
                },
                "execution": {
                    "apptainer_sif": "/containers/molcas.sif",
                    "pymolcas_command": "pymolcas",
                    "parallel_caspt2_supported": False,
                    "env": {"MOLCAS_COLOR": "NO"},
                },
                "resources": {
                    "nodes": 2,
                    "mpi_ranks": 48,
                    "omp_threads": 1,
                    "walltime": "02:00:00",
                    "partition": "compute",
                },
                "modules": {
                    "load": ["tacc-apptainer"],
                },
                "hooks": {
                    "pre_run": [
                        'export MOLCAS_WORKDIR="$SCRATCH/molcas/'
                        '$MOLCAS_PROJECT"',
                        'mkdir -p "$MOLCAS_WORKDIR"',
                    ],
                },
                "file_rules": {
                    "output_file": "{job_name}.log",
                    "error_file": "{job_name}.err",
                },
            },
        },
    }
    adapted = adapt_legacy_molcas_profile(
        profiles,
        "molcas_slurm",
        allowed_work_roots=(tmp_path,),
    )
    prepared = build_molcas_launch_plan(
        input_path,
        adapted.default_resources,
        parallel_caspt2_supported=False,
        output_template=adapted.output_template,
        error_template=adapted.error_template,
    )

    rendered = SlurmExecutor().render(
        prepared.plan,
        adapted.target,
    )

    assert prepared.plan.resources.nodes == 1
    assert prepared.plan.resources.mpi_ranks == 1
    assert rendered.command.argv == (
        "apptainer",
        "exec",
        "/containers/molcas.sif",
        "pymolcas",
        "-np",
        "1",
        "complex.input",
    )
    assert rendered.submit_argv == (
        "sbatch",
        str(tmp_path / "complex.job"),
    )
    assert "module load tacc-apptainer\n" in rendered.script_text
    assert "export MOLCAS_PROJECT=complex\n" in rendered.script_text
    assert "export MOLCAS_NPROCS=1\n" in rendered.script_text
    assert (
        "export MOLCAS_PROJECT=complex_$SLURM_JOB_ID\n"
        in rendered.script_text
    )
    assert (
        'export MOLCAS_WORKDIR="$SCRATCH/molcas/$MOLCAS_PROJECT"\n'
        in rendered.script_text
    )
    assert (
        "apptainer exec /containers/molcas.sif "
        "pymolcas -np 1 complex.input\n"
        in rendered.script_text
    )
    assert rendered.command.stdout_path == tmp_path / "complex.log"


def test_molcas_adapter_rejects_unresolved_container_variable(tmp_path):
    profiles = {
        "schema_version": "1.0",
        "profiles": {
            "molcas_slurm": {
                "launcher": {
                    "kind": "scheduler",
                    "scheduler_type": "slurm",
                },
                "scheduler": {
                    "system": "slurm",
                    "submit_script_name": "{job_name}.job",
                },
                "execution": {
                    "apptainer_sif": "$UNSET_MOLCAS_ROOT/molcas.sif",
                },
            },
        },
    }

    with pytest.raises(
        ValueError,
        match="contains an unresolved variable",
    ):
        adapt_legacy_molcas_profile(
            profiles,
            "molcas_slurm",
            allowed_work_roots=(tmp_path,),
        )


def test_stampede3_molcas_profile_declares_typed_runtime_setup(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("WORK", "/work/charlie")
    profiles = load_runner_profiles(str(STAMPEDE_PROFILE_PATH))

    adapted = adapt_legacy_molcas_profile(
        profiles,
        "stampede3_molcas_skx",
        allowed_work_roots=(tmp_path,),
    )
    installation = adapted.target.programs["molcas"]

    assert installation.launcher_argv == (
        "apptainer",
        "exec",
        "/work/charlie/containers/openmolcas-26.02.sif",
    )
    assert installation.setup_lines == (
        "module load tacc-apptainer",
    )
    assert installation.pre_run_lines == (
        "export MOLCAS_PROJECT={job_name}_$SLURM_JOB_ID",
        'export MOLCAS_WORKDIR="$SCRATCH/molcas/$MOLCAS_PROJECT"',
        'mkdir -p "$MOLCAS_WORKDIR"',
        "export MOLCAS_MEM=8000",
    )


def test_molcas_dry_run_remains_legacy_preview(tmp_path):
    input_path = _input(tmp_path)
    expected = launch_molcas_run(
        input_path=str(input_path),
        profile="molcas_local_native",
        profiles_path=str(PROFILE_PATH),
        dry_run=True,
    )

    actual = launch_molcas_with_service(
        ExecutionService(),
        input_path=str(input_path),
        profile="molcas_local_native",
        profiles_path=str(PROFILE_PATH),
        dry_run=True,
    )

    assert actual == expected
    assert actual["executed"] is False
    assert not (tmp_path / "registry.db").exists()


def test_molcas_local_launch_applies_caspt2_guard(
    tmp_path,
    monkeypatch,
):
    input_path = _input(tmp_path, caspt2=True)
    launched_process: dict[str, object] = {}

    class StartedProcess:
        pid = 7171

    def fake_popen(argv, **kwargs):
        launched_process["argv"] = argv
        launched_process["project"] = kwargs["env"]["MOLCAS_PROJECT"]
        launched_process["nprocs"] = kwargs["env"]["MOLCAS_NPROCS"]
        launched_process["shell"] = kwargs["shell"]
        return StartedProcess()

    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        fake_popen,
    )
    service = _service(tmp_path)

    launched = launch_molcas_with_service(
        service,
        input_path=str(input_path),
        profile="molcas_apptainer_broken_caspt2",
        profiles_path=str(PROFILE_PATH),
        env_overrides={
            "MOLCAS_NPROCS": "99",
            "MOLCAS_COLOR": "NO",
        },
    )

    assert launched["executed"] is True
    assert launched["process_id"] == 7171
    assert launched["status"] == "started"
    assert launched["requested_np"] == 4
    assert launched["effective_np"] == 1
    assert launched["resources"]["mpi_ranks"] == 1
    assert launched["has_caspt2"] is True
    assert len(launched["warnings"]) == 1
    assert launched["effective_argv"] == [
        "apptainer",
        "exec",
        "/path/to/openmolcas-25.02.sif",
        "pymolcas",
        "-np",
        "1",
        "complex.input",
    ]
    assert service.get_launch_record(
        launched["launch_id"]
    ).resources.mpi_ranks == 1
    assert launched_process == {
        "argv": tuple(launched["effective_argv"]),
        "project": "complex",
        "nprocs": "1",
        "shell": False,
    }
    record = service.get_launch_record(launched["launch_id"])
    assert record.environment_keys == (
        "MOLCAS_COLOR",
        "MOLCAS_NPROCS",
        "MOLCAS_PROJECT",
    )


def test_molcas_slurm_launch_and_cancel_use_same_service(
    tmp_path,
    monkeypatch,
):
    input_path = _input(tmp_path, caspt2=True)
    profile_path = _slurm_profile_path(tmp_path)
    previous_log = tmp_path / "complex.log"
    previous_log.write_text("prior Molcas output\n", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[0] == "sbatch":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="Submitted batch job 90210\n",
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

    launched = launch_molcas_with_service(
        service,
        input_path=str(input_path),
        profile="molcas_slurm",
        profiles_path=str(profile_path),
    )

    assert launched["status"] == "submitted"
    assert launched["job_id"] == "90210"
    assert launched["requested_np"] == 96
    assert launched["effective_np"] == 1
    assert launched["resources"]["nodes"] == 1
    assert launched["resources"]["mpi_ranks"] == 1
    assert launched["output_file"] == str(tmp_path / "complex.log")
    assert len(launched["archived_previous_outputs"]) == 1
    archived_log = Path(launched["archived_previous_outputs"][0])
    assert archived_log.name.startswith("complex.log.")
    assert archived_log.read_text(
        encoding="utf-8"
    ) == "prior Molcas output\n"
    assert launched["submit_command"] == [
        "sbatch",
        str(tmp_path / "complex.job"),
    ]
    assert (tmp_path / "complex.jobid").read_text(
        encoding="utf-8"
    ) == "90210"
    assert (
        "export MOLCAS_PROJECT=complex_$SLURM_JOB_ID\n"
        in launched["submit_script_text"]
    )
    assert (
        "apptainer exec /containers/molcas.sif "
        "pymolcas -np 1 complex.input\n"
        in launched["submit_script_text"]
    )

    cancelled = terminate_molcas_with_service(
        service,
        job_id="90210",
        profile="molcas_slurm",
    )

    assert cancelled == {
        "job_id": "90210",
        "cancelled": True,
        "command": ["scancel", "90210"],
        "return_code": 0,
        "stdout": "",
        "stderr": "",
        "launch_id": launched["launch_id"],
    }
    assert calls == [
        ("sbatch", str(tmp_path / "complex.job")),
        ("scancel", "90210"),
    ]


def test_molcas_cancel_rejects_unrecorded_job(tmp_path):
    result = terminate_molcas_with_service(
        _service(tmp_path),
        job_id="90210",
        profile="molcas_slurm",
    )

    assert result == {
        "job_id": "90210",
        "cancelled": False,
        "error": "launch_not_owned",
    }


def test_mcp_local_status_uses_owned_process_handle(
    tmp_path,
    monkeypatch,
):
    input_path = _input(tmp_path)
    output_bytes = b"OpenMolcasOP\n"

    class CompletedProcess:
        pid = 7272

        def poll(self):
            return 0

    def fake_popen(*args, **kwargs):
        kwargs["stdout"].write(output_bytes)
        kwargs["stdout"].flush()
        return CompletedProcess()

    def reject_pid_probe(*args, **kwargs):
        raise AssertionError("owned process reached legacy PID probe")

    db_path = tmp_path / "registry.db"
    monkeypatch.setenv("CHEMTOOLS_REGISTRY_DB", str(db_path))
    monkeypatch.setattr(local_execution.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(core_runner.os, "kill", reject_pid_probe)
    set_active_mode("local")
    try:
        launched = _handle_launch_molcas_run({
            "input_file": str(input_path),
            "profile": "molcas_local_native",
            "profiles_path": str(PROFILE_PATH),
        })
        status = _handle_get_molcas_run_status({
            "output_file": launched["output_file"],
            "input_file": str(input_path),
            "error_file": launched["error_file"],
            "process_id": launched["process_id"],
        })
    finally:
        set_active_mode("analysis")

    assert status["process"] == {
        "process_id": 7272,
        "status": "completed",
        "return_code": 0,
    }
    assert status["execution_record"]["launch_id"] == launched["launch_id"]
    assert status["execution_record"]["status"] == "completed"
    assert load_launch_record(launched["launch_id"], db_path).status == (
        "completed"
    )


def test_mcp_slurm_watch_uses_owned_accounting_status(
    tmp_path,
    monkeypatch,
):
    input_path = _input(tmp_path)
    profile_path = _slurm_profile_path(tmp_path)
    calls = []
    queue_queries = 0

    def fake_run(argv, **kwargs):
        nonlocal queue_queries
        calls.append(tuple(argv))
        if argv[0] == "sbatch":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="Submitted batch job 90300\n",
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
        (tmp_path / "complex.log").write_text(
            "OpenMolcasOP\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="COMPLETED|0:0|51\n",
            stderr="",
        )

    db_path = tmp_path / "registry.db"
    monkeypatch.setenv("CHEMTOOLS_REGISTRY_DB", str(db_path))
    monkeypatch.setattr(slurm_execution.subprocess, "run", fake_run)
    monkeypatch.setattr(core_monitoring.time, "sleep", lambda _: None)
    set_active_mode("hpc")
    try:
        launched = _handle_launch_molcas_run({
            "input_file": str(input_path),
            "profile": "molcas_slurm",
            "profiles_path": str(profile_path),
        })
        watched = _handle_watch_molcas_run({
            "output_file": launched["output_file"],
            "input_file": str(input_path),
            "error_file": launched["error_file"],
            "profile": "molcas_slurm",
            "job_id": launched["job_id"],
            "profiles_path": str(profile_path),
            "poll_interval_seconds": 1,
        })
    finally:
        set_active_mode("analysis")

    final_status = watched["final_status"]
    assert watched["terminal"] is True
    assert watched["poll_count"] == 2
    assert final_status["scheduler"]["status"] == "completed"
    assert final_status["scheduler"]["source"] == "accounting"
    assert final_status["scheduler"]["elapsed_seconds"] == 51.0
    assert final_status["execution_record"]["status"] == "completed"
    assert load_launch_record(launched["launch_id"], db_path).status == (
        "completed"
    )
    assert tuple(call[0] for call in calls) == (
        "sbatch",
        "squeue",
        "squeue",
        "sacct",
    )

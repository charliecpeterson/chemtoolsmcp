"""GRASP workflow launch-plan, target, and compatibility contracts."""

import json
from pathlib import Path
import subprocess

import pytest

import chemtools.core.monitoring as core_monitoring
import chemtools.execution.legacy_runner as core_runner
import chemtools.execution.local as local_execution
import chemtools.execution.slurm as slurm_execution
from chemtools.application.execution import ExecutionService
from chemtools.application.grasp_execution import (
    launch_grasp_workflow_with_service,
    terminate_grasp_with_service,
)
from chemtools.execution.legacy_runner import load_runner_profiles
from chemtools.execution import LocalExecutor, SlurmExecutor
from chemtools.persistence.launches import load_launch_record
from chemtools.mcp.decorator import set_active_mode
from chemtools.mcp.tools.grasp import (
    _handle_get_grasp_run_status,
    _handle_launch_grasp_workflow_run,
    _handle_watch_grasp_run,
)
from chemtools.programs.grasp.launch import (
    adapt_legacy_grasp_profile,
    build_grasp_workflow_launch_plan,
)
from chemtools.programs.grasp.scheduler import (
    launch_grasp_workflow_run,
)


STAMPEDE_PROFILE_PATH = (
    Path(__file__).parents[1]
    / "examples"
    / "tacc_stampede3"
    / "runner_profiles.yaml"
)


def _workflow(tmp_path: Path) -> Path:
    script_path = tmp_path / "run_th.sh"
    script_path.write_text(
        "#!/bin/bash\n"
        "set -e\n"
        "rnucleus <<'EOF'\n"
        "90\n"
        "232\n"
        "EOF\n"
        "rmcdhf <<'EOF'\n"
        "y\n"
        "EOF\n",
        encoding="utf-8",
    )
    return script_path


def _profiles() -> dict:
    return {
        "schema_version": "1.0",
        "profiles": {
            "grasp_local": {
                "launcher": {
                    "kind": "direct",
                    "command": "bash",
                },
                "apptainer_sif": "/containers/grasp.sif",
                "resources": {
                    "nodes": 1,
                    "mpi_ranks": 1,
                    "omp_threads": 1,
                },
                "env": {"GRASP_TMPDIR": "/scratch/grasp"},
                "file_rules": {
                    "output_file": "{job_name}.out",
                    "error_file": "{job_name}.err",
                },
            },
            "grasp_slurm": {
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
                    "script_template": (
                        "#!/bin/bash\n"
                        "apptainer exec {apptainer_sif} "
                        "bash {input_file}\n"
                    ),
                },
                "apptainer_sif": "/containers/grasp.sif",
                "resources": {
                    "nodes": 1,
                    "mpi_ranks": 8,
                    "omp_threads": 1,
                    "walltime": "02:00:00",
                    "partition": "compute",
                },
                "modules": {"load": ["tacc-apptainer"]},
                "hooks": {
                    "pre_run": [
                        'export GRASP_TMPDIR="$SCRATCH/grasp"',
                    ],
                },
                "file_rules": {
                    "output_file": "{job_name}.out",
                    "error_file": "{job_name}.err",
                },
            },
        },
    }


def _profile_path(tmp_path: Path) -> Path:
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        json.dumps(_profiles()),
        encoding="utf-8",
    )
    return profile_path


def _service(tmp_path: Path) -> ExecutionService:
    return ExecutionService(
        enable_execution=True,
        registry_db_path=tmp_path / "registry.db",
    )


def test_local_grasp_plan_runs_whole_workflow_inside_container(
    tmp_path,
):
    workflow = _workflow(tmp_path)
    adapted = adapt_legacy_grasp_profile(
        _profiles(),
        "grasp_local",
        allowed_work_roots=(tmp_path,),
    )
    plan = build_grasp_workflow_launch_plan(
        workflow,
        adapted.default_resources,
    )

    rendered = LocalExecutor().render(plan, adapted.target)

    assert rendered.argv == (
        "apptainer",
        "exec",
        "/containers/grasp.sif",
        "bash",
        "run_th.sh",
    )
    assert rendered.environment == {
        "GRASP_TMPDIR": "/scratch/grasp",
    }
    assert rendered.stdout_path == tmp_path / "run_th.out"
    assert rendered.stderr_path == tmp_path / "run_th.err"
    assert plan.program_arguments == ("run_th.sh",)


def test_slurm_grasp_plan_keeps_workflow_as_one_ordered_script(
    tmp_path,
):
    workflow = _workflow(tmp_path)
    adapted = adapt_legacy_grasp_profile(
        _profiles(),
        "grasp_slurm",
        allowed_work_roots=(tmp_path,),
    )
    plan = build_grasp_workflow_launch_plan(
        workflow,
        adapted.default_resources,
    )

    rendered = SlurmExecutor().render(plan, adapted.target)

    assert rendered.command.argv == (
        "apptainer",
        "exec",
        "/containers/grasp.sif",
        "bash",
        "run_th.sh",
    )
    assert "module load tacc-apptainer\n" in rendered.script_text
    assert 'export GRASP_TMPDIR="$SCRATCH/grasp"\n' in (
        rendered.script_text
    )
    assert (
        "apptainer exec /containers/grasp.sif bash run_th.sh\n"
        in rendered.script_text
    )
    assert rendered.submit_argv == (
        "sbatch",
        str(tmp_path / "run_th.job"),
    )


def test_grasp_adapter_requires_explicit_container(tmp_path):
    profiles = _profiles()
    profiles["profiles"]["grasp_local"].pop("apptainer_sif")

    with pytest.raises(
        ValueError,
        match="requires apptainer_sif",
    ):
        adapt_legacy_grasp_profile(
            profiles,
            "grasp_local",
            allowed_work_roots=(tmp_path,),
        )


def test_grasp_adapter_rejects_unresolved_container_variable(tmp_path):
    profiles = _profiles()
    profiles["profiles"]["grasp_slurm"]["apptainer_sif"] = (
        "$UNSET_GRASP_ROOT/grasp.sif"
    )

    with pytest.raises(
        ValueError,
        match="contains an unresolved variable",
    ):
        adapt_legacy_grasp_profile(
            profiles,
            "grasp_slurm",
            allowed_work_roots=(tmp_path,),
        )


def test_grasp_plan_requires_existing_workflow_script(tmp_path):
    with pytest.raises(
        ValueError,
        match="workflow script does not exist",
    ):
        build_grasp_workflow_launch_plan(
            tmp_path / "missing.sh",
            adapt_legacy_grasp_profile(
                _profiles(),
                "grasp_local",
                allowed_work_roots=(tmp_path,),
            ).default_resources,
        )


def test_stampede3_grasp_profile_runs_script_inside_container(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("WORK", "/work/charlie")
    profiles = load_runner_profiles(str(STAMPEDE_PROFILE_PATH))

    adapted = adapt_legacy_grasp_profile(
        profiles,
        "stampede3_grasp_skx",
        allowed_work_roots=(tmp_path,),
    )
    installation = adapted.target.programs["grasp"]

    assert installation.launcher_argv == (
        "apptainer",
        "exec",
        "/work/charlie/containers/grasp2018.sif",
    )
    assert installation.executable_argv == ("bash",)
    assert installation.setup_lines == (
        "module load tacc-apptainer",
    )


def test_grasp_dry_run_remains_legacy_preview(tmp_path):
    workflow = _workflow(tmp_path)
    profile_path = _profile_path(tmp_path)
    expected = launch_grasp_workflow_run(
        workflow_script_path=str(workflow),
        profile="grasp_slurm",
        profiles_path=str(profile_path),
        dry_run=True,
    )

    actual = launch_grasp_workflow_with_service(
        ExecutionService(),
        workflow_script_path=str(workflow),
        profile="grasp_slurm",
        profiles_path=str(profile_path),
        dry_run=True,
    )

    assert actual == expected
    assert actual["executed"] is False
    assert not (tmp_path / "registry.db").exists()


def test_grasp_local_launch_uses_tracked_container_command(
    tmp_path,
    monkeypatch,
):
    workflow = _workflow(tmp_path)
    profile_path = _profile_path(tmp_path)
    observed: dict[str, object] = {}

    class StartedProcess:
        pid = 9191

    def fake_popen(argv, **kwargs):
        observed["argv"] = argv
        observed["tmpdir"] = kwargs["env"]["GRASP_TMPDIR"]
        observed["shell"] = kwargs["shell"]
        return StartedProcess()

    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        fake_popen,
    )
    service = _service(tmp_path)

    launched = launch_grasp_workflow_with_service(
        service,
        workflow_script_path=str(workflow),
        profile="grasp_local",
        profiles_path=str(profile_path),
        env_overrides={"GRASP_TMPDIR": "/custom/scratch"},
    )

    assert launched["executed"] is True
    assert launched["process_id"] == 9191
    assert launched["status"] == "started"
    assert launched["workflow_script_path"] == str(workflow)
    assert launched["effective_argv"] == [
        "apptainer",
        "exec",
        "/containers/grasp.sif",
        "bash",
        "run_th.sh",
    ]
    assert observed == {
        "argv": tuple(launched["effective_argv"]),
        "tmpdir": "/custom/scratch",
        "shell": False,
    }
    assert service.get_launch_record(
        launched["launch_id"]
    ).process_id == 9191


def test_grasp_slurm_launch_archives_output_and_cancels_owned_job(
    tmp_path,
    monkeypatch,
):
    workflow = _workflow(tmp_path)
    profile_path = _profile_path(tmp_path)
    previous_output = tmp_path / "run_th.out"
    previous_output.write_text("prior GRASP output\n", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[0] == "sbatch":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="Submitted batch job 13579\n",
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

    launched = launch_grasp_workflow_with_service(
        service,
        workflow_script_path=str(workflow),
        profile="grasp_slurm",
        profiles_path=str(profile_path),
    )

    assert launched["status"] == "submitted"
    assert launched["job_id"] == "13579"
    assert launched["submit_command"] == [
        "sbatch",
        str(tmp_path / "run_th.job"),
    ]
    assert launched["jobid_file"] == str(tmp_path / "run_th.jobid")
    assert len(launched["archived_previous_outputs"]) == 1
    archived_output = Path(
        launched["archived_previous_outputs"][0]
    )
    assert archived_output.name.startswith("run_th.out.")
    assert archived_output.read_text(
        encoding="utf-8"
    ) == "prior GRASP output\n"
    assert (
        "apptainer exec /containers/grasp.sif bash run_th.sh\n"
        in launched["submit_script_text"]
    )

    cancelled = terminate_grasp_with_service(
        service,
        job_id="13579",
        profile="grasp_slurm",
    )

    assert cancelled == {
        "job_id": "13579",
        "cancelled": True,
        "command": ["scancel", "13579"],
        "return_code": 0,
        "stdout": "",
        "stderr": "",
        "launch_id": launched["launch_id"],
    }
    assert calls == [
        ("sbatch", str(tmp_path / "run_th.job")),
        ("scancel", "13579"),
    ]


def test_grasp_cancel_rejects_unrecorded_job(tmp_path):
    result = terminate_grasp_with_service(
        _service(tmp_path),
        job_id="13579",
        profile="grasp_slurm",
    )

    assert result == {
        "job_id": "13579",
        "cancelled": False,
        "error": "launch_not_owned",
    }


def test_mcp_grasp_local_status_uses_owned_process_handle(
    tmp_path,
    monkeypatch,
):
    workflow = _workflow(tmp_path)
    profile_path = _profile_path(tmp_path)

    class CompletedProcess:
        pid = 9292

        def poll(self):
            return 0

    def fake_popen(*args, **kwargs):
        kwargs["stdout"].write(b"GRASP workflow output\n")
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
        launched = _handle_launch_grasp_workflow_run({
            "workflow_script_path": str(workflow),
            "profile": "grasp_local",
            "profiles_path": str(profile_path),
        })
        status = _handle_get_grasp_run_status({
            "output_file": launched["output_file"],
            "input_file": str(workflow),
            "error_file": launched["error_file"],
            "process_id": launched["process_id"],
        })
    finally:
        set_active_mode("analysis")

    assert status["process"] == {
        "process_id": 9292,
        "status": "completed",
        "return_code": 0,
    }
    persisted = load_launch_record(launched["launch_id"], db_path)
    assert status["execution_record"] == {
        "launch_id": launched["launch_id"],
        "status": "completed",
        "elapsed_seconds": persisted.elapsed_seconds,
    }
    assert persisted.status == "completed"


def test_mcp_grasp_slurm_watch_uses_owned_accounting_status(
    tmp_path,
    monkeypatch,
):
    workflow = _workflow(tmp_path)
    profile_path = _profile_path(tmp_path)
    calls = []
    queue_queries = 0

    def fake_run(argv, **kwargs):
        nonlocal queue_queries
        calls.append(tuple(argv))
        if argv[0] == "sbatch":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="Submitted batch job 13580\n",
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
        (tmp_path / "run_th.out").write_text(
            "GRASP workflow output\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="COMPLETED|0:0|74\n",
            stderr="",
        )

    db_path = tmp_path / "registry.db"
    monkeypatch.setenv("CHEMTOOLS_REGISTRY_DB", str(db_path))
    monkeypatch.setattr(slurm_execution.subprocess, "run", fake_run)
    monkeypatch.setattr(core_monitoring.time, "sleep", lambda _: None)
    set_active_mode("hpc")
    try:
        launched = _handle_launch_grasp_workflow_run({
            "workflow_script_path": str(workflow),
            "profile": "grasp_slurm",
            "profiles_path": str(profile_path),
        })
        watched = _handle_watch_grasp_run({
            "output_file": launched["output_file"],
            "input_file": str(workflow),
            "error_file": launched["error_file"],
            "profile": "grasp_slurm",
            "job_id": launched["job_id"],
            "profiles_path": str(profile_path),
            "poll_interval_seconds": 0,
        })
    finally:
        set_active_mode("analysis")

    final_status = watched["final_status"]
    assert watched["terminal"] is True
    assert watched["poll_count"] == 2
    assert final_status["scheduler"]["status"] == "completed"
    assert final_status["scheduler"]["source"] == "accounting"
    assert final_status["scheduler"]["elapsed_seconds"] == 74.0
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

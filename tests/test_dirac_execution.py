"""DIRAC launch-plan, target, and compatibility contracts."""

import json
from pathlib import Path
import subprocess

import pytest

import chemtools.core.monitoring as core_monitoring
import chemtools.core.runner as core_runner
import chemtools.execution.local as local_execution
import chemtools.execution.slurm as slurm_execution
from chemtools.application.dirac_execution import (
    launch_dirac_with_service,
    terminate_dirac_with_service,
)
from chemtools.application.execution import ExecutionService
from chemtools.core.runner import load_runner_profiles
from chemtools.execution import LocalExecutor, SlurmExecutor
from chemtools.execution.launch_registry import load_launch_record
from chemtools.mcp.decorator import set_active_mode
from chemtools.mcp.tools.dirac import (
    _handle_get_dirac_run_status,
    _handle_launch_dirac_run,
    _handle_watch_dirac_run,
)
from chemtools.programs.dirac.launch import (
    adapt_legacy_dirac_profile,
    build_dirac_launch_plan,
)
from chemtools.programs.dirac.runtime import prepare_launch
from chemtools.programs.dirac.scheduler import launch_dirac_run


STAMPEDE_PROFILE_PATH = (
    Path(__file__).parents[1]
    / "examples"
    / "tacc_stampede3"
    / "runner_profiles.yaml"
)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    input_path = tmp_path / "molecule.inp"
    molecule_path = tmp_path / "geometry.mol"
    input_path.write_text(
        "**DIRAC\n.WAVE FUNCTION\n**HAMILTONIAN\n.X2C\n",
        encoding="utf-8",
    )
    molecule_path.write_text(
        "INTGRL\nMolecule\nC 1\n1.0 1\nH 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    return input_path, molecule_path


def _profiles() -> dict:
    return {
        "schema_version": "1.0",
        "profiles": {
            "dirac_local": {
                "launcher": {
                    "kind": "direct",
                    "command": "pam-dirac",
                },
                "resources": {
                    "nodes": 1,
                    "mpi_ranks": 4,
                    "omp_threads": 1,
                },
                "default_mw": 512,
                "default_nw": 256,
                "env": {"DIRAC_TMPDIR": "/scratch/dirac"},
                "file_rules": {
                    "output_file": "{job_name}.out",
                    "error_file": "{job_name}.err",
                },
            },
            "dirac_slurm": {
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
                        "apptainer exec {container_sif} "
                        "pam-dirac --mpi={mpi_ranks} "
                        "--inp={input_file} --mol={mol_file}\n"
                    ),
                },
                "container_sif": "/containers/dirac.sif",
                "pam_dirac_binary": "pam-dirac",
                "apptainer_binary": "apptainer",
                "default_mw": 1024,
                "default_nw": 512,
                "resources": {
                    "nodes": 1,
                    "mpi_ranks": 48,
                    "omp_threads": 1,
                    "walltime": "02:00:00",
                    "partition": "compute",
                },
                "modules": {"load": ["tacc-apptainer"]},
                "hooks": {
                    "pre_run": [
                        'export DIRAC_TMPDIR="$SCRATCH/dirac"',
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


def test_local_dirac_plan_matches_read_only_argument_builder(tmp_path):
    input_path, molecule_path = _inputs(tmp_path)
    adapted = adapt_legacy_dirac_profile(
        _profiles(),
        "dirac_local",
        allowed_work_roots=(tmp_path,),
    )
    plan = build_dirac_launch_plan(
        input_path,
        molecule_path,
        adapted.default_resources,
        master_memory_mb=adapted.master_memory_mb,
        node_memory_mb=adapted.node_memory_mb,
    )

    rendered = LocalExecutor().render(plan, adapted.target)
    preview = prepare_launch(
        str(input_path),
        str(molecule_path),
        mpi=4,
        mw=512,
        nw=256,
        work_dir=str(tmp_path),
    )

    assert rendered.argv == tuple(preview["command"])
    assert rendered.argv == (
        "pam-dirac",
        "--mpi=4",
        "--inp=molecule.inp",
        "--mol=geometry.mol",
        "--mw=512",
        "--nw=256",
    )
    assert rendered.environment == {
        "DIRAC_TMPDIR": "/scratch/dirac",
    }
    assert rendered.stdout_path == tmp_path / "molecule.out"
    assert rendered.stderr_path == tmp_path / "molecule.err"


def test_dirac_adapter_uses_runtime_default_mpi_without_resources(
    tmp_path,
):
    profiles = {
        "schema_version": "1.0",
        "profiles": {
            "dirac_local": {
                "launcher": {
                    "kind": "direct",
                    "command": "pam-dirac",
                },
                "default_mpi": 10,
            },
        },
    }

    adapted = adapt_legacy_dirac_profile(
        profiles,
        "dirac_local",
        allowed_work_roots=(tmp_path,),
    )

    assert adapted.default_resources.mpi_ranks == 10


def test_slurm_dirac_plan_keeps_pam_mpi_out_of_scheduler_launcher(
    tmp_path,
):
    input_path, molecule_path = _inputs(tmp_path)
    adapted = adapt_legacy_dirac_profile(
        _profiles(),
        "dirac_slurm",
        allowed_work_roots=(tmp_path,),
    )
    plan = build_dirac_launch_plan(
        input_path,
        molecule_path,
        adapted.default_resources,
        master_memory_mb=adapted.master_memory_mb,
        node_memory_mb=adapted.node_memory_mb,
    )

    rendered = SlurmExecutor().render(plan, adapted.target)

    assert rendered.command.argv == (
        "apptainer",
        "exec",
        "/containers/dirac.sif",
        "pam-dirac",
        "--mpi=48",
        "--inp=molecule.inp",
        "--mol=geometry.mol",
        "--mw=1024",
        "--nw=512",
    )
    assert "ibrun" not in rendered.script_text
    assert "module load tacc-apptainer\n" in rendered.script_text
    assert 'export DIRAC_TMPDIR="$SCRATCH/dirac"\n' in (
        rendered.script_text
    )
    assert rendered.submit_argv == (
        "sbatch",
        str(tmp_path / "molecule.job"),
    )


def test_dirac_plan_requires_paired_files_in_one_working_directory(
    tmp_path,
):
    input_path, _ = _inputs(tmp_path)
    other_directory = tmp_path / "other"
    other_directory.mkdir()
    molecule_path = other_directory / "geometry.mol"
    molecule_path.write_text("INTGRL\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="\\.inp and \\.mol files to use the same working directory",
    ):
        build_dirac_launch_plan(
            input_path,
            molecule_path,
            adapt_legacy_dirac_profile(
                _profiles(),
                "dirac_local",
                allowed_work_roots=(tmp_path,),
            ).default_resources,
        )


def test_dirac_adapter_rejects_unresolved_container_variable(tmp_path):
    profiles = _profiles()
    profiles["profiles"]["dirac_slurm"]["container_sif"] = (
        "$UNSET_DIRAC_ROOT/dirac.sif"
    )

    with pytest.raises(
        ValueError,
        match="contains an unresolved variable",
    ):
        adapt_legacy_dirac_profile(
            profiles,
            "dirac_slurm",
            allowed_work_roots=(tmp_path,),
        )


def test_stampede3_dirac_profile_declares_typed_runtime_setup(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("WORK", "/work/charlie")
    profiles = load_runner_profiles(str(STAMPEDE_PROFILE_PATH))

    adapted = adapt_legacy_dirac_profile(
        profiles,
        "stampede3_dirac_skx",
        allowed_work_roots=(tmp_path,),
    )
    installation = adapted.target.programs["dirac"]

    assert installation.launcher_argv == (
        "apptainer",
        "exec",
        "/work/charlie/containers/dirac-25.0.sif",
    )
    assert installation.executable_argv == ("pam-dirac",)
    assert installation.setup_lines == (
        "module load tacc-apptainer",
    )


def test_dirac_dry_run_remains_legacy_preview(tmp_path):
    input_path, molecule_path = _inputs(tmp_path)
    profile_path = _profile_path(tmp_path)
    expected = launch_dirac_run(
        input_path=str(input_path),
        mol_file=str(molecule_path),
        profile="dirac_slurm",
        profiles_path=str(profile_path),
        dry_run=True,
    )

    actual = launch_dirac_with_service(
        ExecutionService(),
        input_path=str(input_path),
        mol_file=str(molecule_path),
        profile="dirac_slurm",
        profiles_path=str(profile_path),
        dry_run=True,
    )

    assert actual == expected
    assert actual["executed"] is False
    assert not (tmp_path / "registry.db").exists()


def test_dirac_local_launch_uses_exact_typed_command(
    tmp_path,
    monkeypatch,
):
    input_path, molecule_path = _inputs(tmp_path)
    profile_path = _profile_path(tmp_path)
    observed: dict[str, object] = {}

    class StartedProcess:
        pid = 8181

    def fake_popen(argv, **kwargs):
        observed["argv"] = argv
        observed["tmpdir"] = kwargs["env"]["DIRAC_TMPDIR"]
        observed["shell"] = kwargs["shell"]
        return StartedProcess()

    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        fake_popen,
    )
    service = _service(tmp_path)

    launched = launch_dirac_with_service(
        service,
        input_path=str(input_path),
        mol_file=str(molecule_path),
        profile="dirac_local",
        profiles_path=str(profile_path),
        env_overrides={"DIRAC_TMPDIR": "/custom/scratch"},
    )

    assert launched["executed"] is True
    assert launched["process_id"] == 8181
    assert launched["status"] == "started"
    assert launched["mol_file"] == str(molecule_path)
    assert launched["master_memory_mb"] == 512
    assert launched["node_memory_mb"] == 256
    assert launched["effective_argv"] == [
        "pam-dirac",
        "--mpi=4",
        "--inp=molecule.inp",
        "--mol=geometry.mol",
        "--mw=512",
        "--nw=256",
    ]
    assert observed == {
        "argv": tuple(launched["effective_argv"]),
        "tmpdir": "/custom/scratch",
        "shell": False,
    }
    assert service.get_launch_record(
        launched["launch_id"]
    ).process_id == 8181


def test_dirac_slurm_launch_archives_output_and_cancels_owned_job(
    tmp_path,
    monkeypatch,
):
    input_path, molecule_path = _inputs(tmp_path)
    profile_path = _profile_path(tmp_path)
    previous_output = tmp_path / "molecule.out"
    previous_output.write_text("prior DIRAC output\n", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[0] == "sbatch":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="Submitted batch job 24680\n",
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

    launched = launch_dirac_with_service(
        service,
        input_path=str(input_path),
        mol_file=str(molecule_path),
        profile="dirac_slurm",
        profiles_path=str(profile_path),
    )

    assert launched["status"] == "submitted"
    assert launched["job_id"] == "24680"
    assert launched["submit_command"] == [
        "sbatch",
        str(tmp_path / "molecule.job"),
    ]
    assert launched["jobid_file"] == str(
        tmp_path / "molecule.jobid"
    )
    assert len(launched["archived_previous_outputs"]) == 1
    archived_output = Path(
        launched["archived_previous_outputs"][0]
    )
    assert archived_output.name.startswith("molecule.out.")
    assert archived_output.read_text(
        encoding="utf-8"
    ) == "prior DIRAC output\n"
    assert (
        "pam-dirac --mpi=48 --inp=molecule.inp "
        "--mol=geometry.mol --mw=1024 --nw=512\n"
        in launched["submit_script_text"]
    )

    cancelled = terminate_dirac_with_service(
        service,
        job_id="24680",
        profile="dirac_slurm",
    )

    assert cancelled == {
        "job_id": "24680",
        "cancelled": True,
        "command": ["scancel", "24680"],
        "return_code": 0,
        "stdout": "",
        "stderr": "",
        "launch_id": launched["launch_id"],
    }
    assert calls == [
        ("sbatch", str(tmp_path / "molecule.job")),
        ("scancel", "24680"),
    ]


def test_dirac_cancel_rejects_unrecorded_job(tmp_path):
    result = terminate_dirac_with_service(
        _service(tmp_path),
        job_id="24680",
        profile="dirac_slurm",
    )

    assert result == {
        "job_id": "24680",
        "cancelled": False,
        "error": "launch_not_owned",
    }


def test_mcp_dirac_local_status_uses_owned_process_handle(
    tmp_path,
    monkeypatch,
):
    input_path, molecule_path = _inputs(tmp_path)
    profile_path = _profile_path(tmp_path)

    class CompletedProcess:
        pid = 8282

        def poll(self):
            return 0

    def fake_popen(*args, **kwargs):
        kwargs["stdout"].write(b"DIRAC execution output\n")
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
        launched = _handle_launch_dirac_run({
            "input_file": str(input_path),
            "mol_file": str(molecule_path),
            "profile": "dirac_local",
            "profiles_path": str(profile_path),
        })
        status = _handle_get_dirac_run_status({
            "output_file": launched["output_file"],
            "input_file": str(input_path),
            "error_file": launched["error_file"],
            "process_id": launched["process_id"],
        })
    finally:
        set_active_mode("analysis")

    assert status["process"] == {
        "process_id": 8282,
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


def test_mcp_dirac_slurm_watch_uses_owned_accounting_status(
    tmp_path,
    monkeypatch,
):
    input_path, molecule_path = _inputs(tmp_path)
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
                stdout="Submitted batch job 24681\n",
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
        (tmp_path / "molecule.out").write_text(
            "DIRAC execution output\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="COMPLETED|0:0|63\n",
            stderr="",
        )

    db_path = tmp_path / "registry.db"
    monkeypatch.setenv("CHEMTOOLS_REGISTRY_DB", str(db_path))
    monkeypatch.setattr(slurm_execution.subprocess, "run", fake_run)
    monkeypatch.setattr(core_monitoring.time, "sleep", lambda _: None)
    set_active_mode("hpc")
    try:
        launched = _handle_launch_dirac_run({
            "input_file": str(input_path),
            "mol_file": str(molecule_path),
            "profile": "dirac_slurm",
            "profiles_path": str(profile_path),
        })
        watched = _handle_watch_dirac_run({
            "output_file": launched["output_file"],
            "input_file": str(input_path),
            "error_file": launched["error_file"],
            "profile": "dirac_slurm",
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
    assert final_status["scheduler"]["elapsed_seconds"] == 63.0
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

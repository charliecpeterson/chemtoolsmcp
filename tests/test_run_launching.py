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
from chemtools.core.execution import (
    ExecutionTarget,
    HardwareDescription,
    ProgramInstallation,
    ResourceRequest,
    SchedulerDefaults,
)
from chemtools.execution.slurm import SlurmExecutor
from chemtools.execution.profiles import load_runner_profiles
from chemtools.mcp.catalog import BUILTIN_BACKENDS, load_backend
from chemtools.programs.nwchem.launch import adapt_legacy_nwchem_profile
import chemtools.execution.local as local_execution


PROFILE_PATH = (
    Path(__file__).parents[1]
    / "chemtools"
    / "runner_profiles.example.json"
)
NWCHEM = load_backend(BUILTIN_BACKENDS[0])
GRASP = load_backend(BUILTIN_BACKENDS[3])


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


def _named_service(
    tmp_path: Path,
    *,
    executor: str = "local",
) -> ExecutionService:
    scheduler = None
    installation = ProgramInstallation(
        launcher_argv=(
            ("mpirun", "-np", "{mpi_ranks}")
            if executor == "local"
            else ("srun",)
        ),
        executable_argv=("nwchem",),
        environment={"OMP_NUM_THREADS": "{omp_threads}"},
        setup_lines=(
            ()
            if executor == "local"
            else ("module purge", "module load nwchem")
        ),
    )
    if executor == "slurm":
        scheduler = SchedulerDefaults(
            submit_argv=("sbatch", "{script_file}"),
            status_argv=("squeue", "-j", "{job_id}"),
            cancel_argv=("scancel", "{job_id}"),
        )
    target = ExecutionTarget(
        name=f"named_{executor}",
        executor=executor,
        allowed_work_roots=(tmp_path,),
        hardware=HardwareDescription(cores_per_node=20),
        programs={"nwchem": installation},
        scheduler=scheduler,
        default_resources=ResourceRequest(
            mpi_ranks=4,
            omp_threads=2,
            walltime="01:00:00" if executor == "slurm" else None,
            partition="compute" if executor == "slurm" else None,
        ),
    )
    return ExecutionService(
        configured_targets={target.name: target},
        default_target=target.name,
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


def test_named_target_prepares_without_loading_runner_profiles(tmp_path):
    input_path = _input(tmp_path)
    service = _named_service(tmp_path)

    prepared = launch_run(
        NWCHEM,
        service,
        input_file=input_path,
        resources={"mpi_ranks": 6},
    )

    plan = prepared["evidence"]["plan"]
    assert prepared["status"] == "awaiting_approval"
    assert plan["profile"] is None
    assert plan["profiles_path"] is None
    assert plan["target"] == "named_local"
    assert plan["argv"] == [
        "mpirun",
        "-np",
        "6",
        "nwchem",
        "uo2.nw",
    ]
    assert plan["resources"] == {
        "nodes": 1,
        "mpi_ranks": 6,
        "omp_threads": 2,
        "memory_mb_per_node": None,
        "walltime": None,
        "partition": None,
        "account": None,
    }


def test_named_slurm_target_uses_target_owned_scheduler_commands(tmp_path):
    _input(tmp_path)
    service = _named_service(tmp_path, executor="slurm")

    prepared = launch_run(
        NWCHEM,
        service,
        input_file=tmp_path / "uo2.nw",
        target="named_slurm",
    )

    plan = prepared["evidence"]["plan"]
    assert plan["argv"] == ["srun", "nwchem", "uo2.nw"]
    assert plan["resources"]["mpi_ranks"] == 4
    assert plan["scheduler"]["submit_argv"] == [
        "sbatch",
        str(tmp_path / "uo2.job"),
    ]


@pytest.mark.parametrize("profile", ["local_mpirun", "slurm_cpu"])
def test_named_target_render_matches_profile_migration_adapter(
    tmp_path,
    profile,
):
    _input(tmp_path)
    profiles = load_runner_profiles(str(PROFILE_PATH))
    adapted = adapt_legacy_nwchem_profile(
        profiles,
        profile,
        allowed_work_roots=(tmp_path,),
    )
    named_service = ExecutionService(
        configured_targets={profile: adapted.target},
        default_target=profile,
    )

    migrated = launch_run(
        NWCHEM,
        ExecutionService(),
        input_file=tmp_path / "uo2.nw",
        profile=profile,
        profiles_path=PROFILE_PATH,
    )
    named = launch_run(
        NWCHEM,
        named_service,
        input_file=tmp_path / "uo2.nw",
    )

    migrated_plan = dict(migrated["evidence"]["plan"])
    named_plan = dict(named["evidence"]["plan"])
    migrated_plan.pop("profile")
    migrated_plan.pop("profiles_path")
    named_plan.pop("profile")
    named_plan.pop("profiles_path")
    assert named_plan == migrated_plan
    assert named["approval"]["token"] == migrated["approval"]["token"]


def test_launch_selection_rejects_ambiguous_or_missing_configuration(tmp_path):
    _input(tmp_path)
    with pytest.raises(LaunchRunError, match="provide profile or target"):
        launch_run(
            NWCHEM,
            _named_service(tmp_path),
            input_file=tmp_path / "uo2.nw",
            profile="local_mpirun",
            target="named_local",
        )

    with pytest.raises(LaunchRunError, match="no default_target"):
        launch_run(
            NWCHEM,
            ExecutionService(),
            input_file=tmp_path / "uo2.nw",
        )

    with pytest.raises(LaunchRunError, match="supported only for qmcpack"):
        launch_run(
            NWCHEM,
            _named_service(tmp_path),
            input_file=tmp_path / "uo2.nw",
            initialization_only=True,
        )


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
            GRASP,
            ExecutionService(),
            input_file=input_path,
            profile="local_mpirun",
            profiles_path=PROFILE_PATH,
        )

    assert caught.value.as_dict() == {
        "error": "unsupported_capability",
        "message": "'grasp' does not support guided launch planning",
        "program": "grasp",
    }


def test_launch_run_rejects_dirac_molecule_for_other_program(tmp_path):
    input_path = _input(tmp_path)

    with pytest.raises(LaunchRunError) as caught:
        launch_run(
            NWCHEM,
            ExecutionService(),
            input_file=input_path,
            molecule_file=input_path,
            profile="local_mpirun",
            profiles_path=PROFILE_PATH,
        )

    assert caught.value.as_dict() == {
        "error": "invalid_launch_request",
        "message": "molecule_file is supported only for dirac",
        "program": "nwchem",
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

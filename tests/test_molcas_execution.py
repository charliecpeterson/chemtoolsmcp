"""OpenMolcas launch-plan, target, and compatibility contracts."""

import json
from pathlib import Path

import pytest

from chemtools.application.execution import ExecutionService
from chemtools.application.run_launching import launch_run
from chemtools.execution import LocalExecutor, SlurmExecutor
from chemtools.execution.profiles import load_runner_profiles
from chemtools.programs.molcas.launch import (
    adapt_legacy_molcas_profile,
    build_molcas_launch_plan,
)
from chemtools.programs.molcas import MOLCAS
from chemtools.programs.molcas.runtime import prepare_launch


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


def _guided_profile_path(tmp_path: Path) -> Path:
    profile_path = tmp_path / "guided_profiles.json"
    profile_path.write_text(json.dumps({
        "schema_version": "1.0",
        "profiles": {
            "molcas_local": {
                "launcher": {"kind": "direct"},
                "programs": {
                    "molcas": {
                        "launcher_argv": [
                            "apptainer",
                            "exec",
                            "/containers/molcas.sif",
                        ],
                        "executable_argv": ["pymolcas"],
                        "parallel_caspt2_supported": False,
                    },
                },
                "execution": {"working_directory": "{job_dir}"},
                "resources": {"mpi_ranks": 4, "omp_threads": 1},
                "file_rules": {
                    "output_file": "{job_name}.out",
                    "error_file": "{job_name}.err",
                },
            },
            "molcas_slurm": {
                "launcher": {
                    "kind": "scheduler",
                    "scheduler_type": "slurm",
                    "submit_command": "sbatch",
                    "status_command": "squeue -j {job_id}",
                    "cancel_command": "scancel {job_id}",
                },
                "scheduler": {
                    "system": "slurm",
                    "submit_script_name": "{job_name}.job",
                },
                "programs": {
                    "molcas": {
                        "launcher_argv": [
                            "apptainer",
                            "exec",
                            "/containers/molcas.sif",
                        ],
                        "executable_argv": ["pymolcas"],
                        "parallel_caspt2_supported": False,
                    },
                },
                "execution": {
                    "working_directory": "{job_dir}",
                    "env": {"MOLCAS_COLOR": "NO"},
                },
                "resources": {
                    "nodes": 1,
                    "mpi_ranks": 8,
                    "omp_threads": 1,
                    "walltime": "02:00:00",
                    "partition": "compute",
                },
                "modules": {"load": ["openmolcas"]},
                "hooks": {
                    "pre_run": [
                        'export MOLCAS_WORKDIR="$SCRATCH/molcas/'
                        '$MOLCAS_PROJECT"',
                    ],
                },
                "file_rules": {
                    "output_file": "{job_name}.out",
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


@pytest.mark.parametrize("caspt2", [False, True])
@pytest.mark.parametrize("executor", ["local", "slurm"])
def test_guided_molcas_named_target_matches_safe_profile(
    tmp_path,
    caspt2,
    executor,
):
    input_path = _input(tmp_path, caspt2=caspt2)
    profile_name = f"molcas_{executor}"
    profile_path = _guided_profile_path(tmp_path)
    profiles = load_runner_profiles(str(profile_path))
    adapted = adapt_legacy_molcas_profile(
        profiles,
        profile_name,
        allowed_work_roots=(tmp_path,),
    )
    named_service = ExecutionService(
        configured_targets={profile_name: adapted.target},
        default_target=profile_name,
    )

    migrated = launch_run(
        MOLCAS,
        ExecutionService(),
        input_file=input_path,
        profile=profile_name,
        profiles_path=profile_path,
    )
    named = launch_run(
        MOLCAS,
        named_service,
        input_file=input_path,
    )

    migrated_plan = dict(migrated["evidence"]["plan"])
    named_plan = dict(named["evidence"]["plan"])
    migrated_plan.pop("profile")
    migrated_plan.pop("profiles_path")
    named_plan.pop("profile")
    named_plan.pop("profiles_path")
    assert named_plan == migrated_plan
    assert named["approval"]["token"] == migrated["approval"]["token"]
    if caspt2:
        assert named_plan["resources"]["mpi_ranks"] == 1
        assert len(named_plan["adjustments"]) == 1
        adjustment = named_plan["adjustments"][0]
        assert adjustment["code"] == (
            "molcas_parallel_caspt2_serialized"
        )
        assert adjustment["requested_mpi_ranks"] == (
            adapted.default_resources.mpi_ranks
        )
        assert adjustment["effective_mpi_ranks"] == 1
        assert "forcing -np 1" in adjustment["reason"]
    else:
        assert named_plan["adjustments"] == []


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

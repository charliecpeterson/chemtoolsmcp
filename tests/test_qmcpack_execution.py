"""Regression coverage for QMCPACK typed launch plans."""

import json
from pathlib import Path

import pytest

from chemtools.application.execution import ExecutionService
from chemtools.application.run_launching import launch_run
from chemtools.execution import LocalExecutor
from chemtools.execution.profiles import load_runner_profiles
from chemtools.programs.qmcpack import QMCPACK
from chemtools.programs.qmcpack.launch import (
    adapt_legacy_qmcpack_profile,
    build_qmcpack_launch_plan,
)


def _qmcpack_input(tmp_path: Path) -> Path:
    input_path = tmp_path / "hydrogen.xml"
    input_path.write_text(
        "<simulation>\n"
        '  <project id="hydrogen" series="0"/>\n'
        '  <qmc method="vmc">\n'
        '    <parameter name="blocks">10</parameter>\n'
        "  </qmc>\n"
        "</simulation>\n",
        encoding="utf-8",
    )
    return input_path


def _guided_profiles(tmp_path: Path) -> Path:
    profile_path = tmp_path / "guided_profiles.json"
    profile_path.write_text(json.dumps({
        "schema_version": "1.0",
        "profiles": {
            "qmcpack_local": {
                "launcher": {"kind": "direct"},
                "programs": {
                    "qmcpack": {
                        "launcher_argv": ["mpirun", "-np", "{mpi_ranks}"],
                        "executable_argv": ["/opt/qmcpack/bin/qmcpack"],
                    },
                },
                "execution": {"working_directory": "{job_dir}"},
                "resources": {"mpi_ranks": 2, "omp_threads": 4},
                "env": {"OMP_NUM_THREADS": "{omp_threads}"},
                "file_rules": {
                    "output_file": "{job_name}.out",
                    "error_file": "{job_name}.err",
                },
            },
            "qmcpack_slurm": {
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
                    "qmcpack": {
                        "launcher_argv": ["srun"],
                        "executable_argv": ["/opt/qmcpack/bin/qmcpack"],
                    },
                },
                "resources": {
                    "nodes": 1,
                    "mpi_ranks": 8,
                    "omp_threads": 2,
                    "walltime": "02:00:00",
                    "partition": "compute",
                },
                "env": {"OMP_NUM_THREADS": "{omp_threads}"},
                "file_rules": {
                    "output_file": "{job_name}.out",
                    "error_file": "{job_name}.err",
                },
            },
        },
    }), encoding="utf-8")
    return profile_path


def test_local_qmcpack_plan_uses_profile_installation_and_artifacts(tmp_path):
    input_path = tmp_path / "hydrogen.xml"
    input_path.write_text("<simulation/>", encoding="utf-8")
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(json.dumps({
        "schema_version": "1.0",
        "profiles": {
            "qmcpack_local": {
                "launcher": {"kind": "direct"},
                "programs": {
                    "qmcpack": {"executable_argv": ["/opt/qmcpack/bin/qmcpack"]},
                },
                "resources": {"mpi_ranks": 1, "omp_threads": 4},
                "env": {"OMP_NUM_THREADS": "{omp_threads}"},
                "file_rules": {
                    "output_file": "{job_name}.out",
                    "error_file": "{job_name}.err",
                },
            },
        },
    }), encoding="utf-8")

    profiles = load_runner_profiles(str(profile_path))
    adapted = adapt_legacy_qmcpack_profile(
        profiles,
        "qmcpack_local",
        allowed_work_roots=(tmp_path,),
    )
    plan = build_qmcpack_launch_plan(
        input_path,
        adapted.default_resources,
        output_template=adapted.output_template,
        error_template=adapted.error_template,
        qmcpack_dry_run=True,
    )

    rendered = LocalExecutor().render(plan, adapted.target)

    assert rendered.argv == (
        "/opt/qmcpack/bin/qmcpack",
        "hydrogen.xml",
        "--dryrun",
    )
    assert rendered.environment == {"OMP_NUM_THREADS": "4"}
    assert rendered.stdout_path == tmp_path / "hydrogen.out"
    assert rendered.stderr_path == tmp_path / "hydrogen.err"
    assert [artifact.kind for artifact in plan.expected_artifacts] == [
        "qmcpack.output",
        "qmcpack.error",
    ]


def test_guided_qmcpack_named_target_preserves_initialization_only(tmp_path):
    input_path = _qmcpack_input(tmp_path)
    profile_path = _guided_profiles(tmp_path)
    profiles = load_runner_profiles(str(profile_path))
    adapted = adapt_legacy_qmcpack_profile(
        profiles,
        "qmcpack_local",
        allowed_work_roots=(tmp_path,),
    )
    service = ExecutionService(
        configured_targets={"qmcpack_local": adapted.target},
        default_target="qmcpack_local",
    )

    ordinary = launch_run(QMCPACK, service, input_file=input_path)
    initialization = launch_run(
        QMCPACK,
        service,
        input_file=input_path,
        initialization_only=True,
    )

    ordinary_plan = ordinary["evidence"]["plan"]
    initialization_plan = initialization["evidence"]["plan"]
    assert ordinary["status"] == "awaiting_approval"
    assert ordinary_plan["argv"] == [
        "mpirun",
        "-np",
        "2",
        "/opt/qmcpack/bin/qmcpack",
        "hydrogen.xml",
    ]
    assert initialization_plan["argv"] == [
        *ordinary_plan["argv"],
        "--dryrun",
    ]
    assert ordinary["approval"]["token"] != initialization["approval"]["token"]


@pytest.mark.parametrize("profile_name", ["qmcpack_local", "qmcpack_slurm"])
@pytest.mark.parametrize("initialization_only", [False, True])
def test_guided_qmcpack_named_target_matches_profile_adapter(
    tmp_path,
    profile_name,
    initialization_only,
):
    input_path = _qmcpack_input(tmp_path)
    profile_path = _guided_profiles(tmp_path)
    profiles = load_runner_profiles(str(profile_path))
    adapted = adapt_legacy_qmcpack_profile(
        profiles,
        profile_name,
        allowed_work_roots=(tmp_path,),
    )
    named_service = ExecutionService(
        configured_targets={profile_name: adapted.target},
        default_target=profile_name,
    )

    migrated = launch_run(
        QMCPACK,
        ExecutionService(),
        input_file=input_path,
        profile=profile_name,
        profiles_path=profile_path,
        initialization_only=initialization_only,
    )
    named = launch_run(
        QMCPACK,
        named_service,
        input_file=input_path,
        initialization_only=initialization_only,
    )

    migrated_plan = dict(migrated["evidence"]["plan"])
    named_plan = dict(named["evidence"]["plan"])
    migrated_plan.pop("profile")
    migrated_plan.pop("profiles_path")
    named_plan.pop("profile")
    named_plan.pop("profiles_path")
    assert named_plan == migrated_plan
    assert named["approval"]["token"] == migrated["approval"]["token"]

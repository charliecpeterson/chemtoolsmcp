"""Regression coverage for Quantum ESPRESSO typed launch plans."""

import json
from pathlib import Path

import pytest

from chemtools.application.execution import ExecutionService
from chemtools.application.qe_execution import render_qe_launch
from chemtools.application.run_launching import launch_run
from chemtools.execution.legacy_runner import load_runner_profiles
from chemtools.execution import LocalExecutor
from chemtools.programs.qe import QE
from chemtools.programs.qe.launch import (
    adapt_legacy_qe_profile,
    build_qe_launch_plan,
)


FIXTURES = Path(__file__).parent / "golden" / "mcp" / "fixtures"


def _qe_input(tmp_path: Path) -> Path:
    input_path = tmp_path / "silicon.in"
    input_path.write_bytes((FIXTURES / "qe_si_scf.in").read_bytes())
    (tmp_path / "Si.pbe.UPF").write_bytes(
        (FIXTURES / "Si.pbe.UPF").read_bytes()
    )
    return input_path


def _guided_profiles(tmp_path: Path) -> Path:
    profile_path = tmp_path / "guided_profiles.json"
    profile_path.write_text(json.dumps({
        "schema_version": "1.0",
        "profiles": {
            "qe_local": {
                "launcher": {"kind": "direct"},
                "programs": {
                    "qe": {
                        "launcher_argv": ["mpirun", "-np", "{mpi_ranks}"],
                        "executable_argv": ["/opt/qe/bin/pw.x"],
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
            "qe_slurm": {
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
                    "qe": {
                        "launcher_argv": ["srun"],
                        "executable_argv": ["/opt/qe/bin/pw.x"],
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


def test_local_qe_plan_uses_profile_installation_and_artifacts(tmp_path):
    input_path = tmp_path / "silicon.in"
    input_path.write_text("&CONTROL\n/\n", encoding="utf-8")
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(json.dumps({
        "schema_version": "1.0",
        "profiles": {
            "qe_local": {
                "launcher": {"kind": "direct"},
                "programs": {
                    "qe": {"executable_argv": ["/opt/qe/bin/pw.x"]},
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
    adapted = adapt_legacy_qe_profile(
        profiles,
        "qe_local",
        allowed_work_roots=(tmp_path,),
    )
    plan = build_qe_launch_plan(
        input_path,
        adapted.default_resources,
        output_template=adapted.output_template,
        error_template=adapted.error_template,
    )

    rendered = LocalExecutor().render(plan, adapted.target)

    assert rendered.argv == ("/opt/qe/bin/pw.x", "-in", "silicon.in")
    assert rendered.environment == {"OMP_NUM_THREADS": "4"}
    assert rendered.stdout_path == tmp_path / "silicon.out"
    assert rendered.stderr_path == tmp_path / "silicon.err"
    assert [artifact.kind for artifact in plan.expected_artifacts] == [
        "qe.output",
        "qe.error",
    ]

    preview, _ = render_qe_launch(
        input_path=str(input_path),
        profile="qe_local",
        profiles_path=str(profile_path),
        env_overrides={"QE_TRACE": "1"},
    )

    assert preview["environment"] == {
        "OMP_NUM_THREADS": "4",
        "QE_TRACE": "1",
    }
    assert preview["command"] == (
        "/opt/qe/bin/pw.x -in silicon.in > "
        f"{tmp_path / 'silicon.out'} 2> {tmp_path / 'silicon.err'}"
    )


def test_guided_qe_named_target_prepares_without_profile_loading(tmp_path):
    input_path = _qe_input(tmp_path)
    profile_path = _guided_profiles(tmp_path)
    profiles = load_runner_profiles(str(profile_path))
    adapted = adapt_legacy_qe_profile(
        profiles,
        "qe_local",
        allowed_work_roots=(tmp_path,),
    )
    service = ExecutionService(
        configured_targets={"qe_local": adapted.target},
        default_target="qe_local",
    )

    prepared = launch_run(QE, service, input_file=input_path)

    plan = prepared["evidence"]["plan"]
    assert prepared["status"] == "awaiting_approval"
    assert prepared["program"] == {"name": "qe"}
    assert plan["profile"] is None
    assert plan["profiles_path"] is None
    assert plan["target"] == "qe_local"
    assert plan["argv"] == [
        "mpirun",
        "-np",
        "2",
        "/opt/qe/bin/pw.x",
        "-in",
        "silicon.in",
    ]
    assert [item["kind"] for item in plan["expected_artifacts"]] == [
        "qe.output",
        "qe.error",
    ]


@pytest.mark.parametrize("profile_name", ["qe_local", "qe_slurm"])
def test_guided_qe_named_target_matches_profile_adapter(
    tmp_path,
    profile_name,
):
    input_path = _qe_input(tmp_path)
    profile_path = _guided_profiles(tmp_path)
    profiles = load_runner_profiles(str(profile_path))
    adapted = adapt_legacy_qe_profile(
        profiles,
        profile_name,
        allowed_work_roots=(tmp_path,),
    )
    named_service = ExecutionService(
        configured_targets={profile_name: adapted.target},
        default_target=profile_name,
    )

    migrated = launch_run(
        QE,
        ExecutionService(),
        input_file=input_path,
        profile=profile_name,
        profiles_path=profile_path,
    )
    named = launch_run(QE, named_service, input_file=input_path)

    migrated_plan = dict(migrated["evidence"]["plan"])
    named_plan = dict(named["evidence"]["plan"])
    migrated_plan.pop("profile")
    migrated_plan.pop("profiles_path")
    named_plan.pop("profile")
    named_plan.pop("profiles_path")
    assert named_plan == migrated_plan
    assert named["approval"]["token"] == migrated["approval"]["token"]

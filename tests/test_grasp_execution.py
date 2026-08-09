"""GRASP workflow launch-plan, target, and guided-provider contracts."""

import json
from pathlib import Path

import pytest

from chemtools.application.execution import ExecutionService
from chemtools.application.run_launching import launch_run
from chemtools.execution import LocalExecutor, SlurmExecutor
from chemtools.execution.profiles import load_runner_profiles
from chemtools.programs.grasp import GRASP
from chemtools.programs.grasp.launch import (
    adapt_legacy_grasp_profile,
    build_grasp_workflow_launch_plan,
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
    profile_path.write_text(json.dumps(_profiles()), encoding="utf-8")
    return profile_path


def test_local_grasp_plan_runs_whole_workflow_inside_container(tmp_path):
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
    assert rendered.environment == {"GRASP_TMPDIR": "/scratch/grasp"}
    assert rendered.stdout_path == tmp_path / "run_th.out"
    assert rendered.stderr_path == tmp_path / "run_th.err"
    assert plan.program_arguments == ("run_th.sh",)


def test_slurm_grasp_plan_keeps_workflow_as_one_ordered_script(tmp_path):
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
    assert 'export GRASP_TMPDIR="$SCRATCH/grasp"\n' in rendered.script_text
    assert (
        "apptainer exec /containers/grasp.sif bash run_th.sh\n"
        in rendered.script_text
    )
    assert rendered.submit_argv == (
        "sbatch",
        str(tmp_path / "run_th.job"),
    )


@pytest.mark.parametrize("profile_name", ["grasp_local", "grasp_slurm"])
def test_guided_grasp_named_target_matches_profile(tmp_path, profile_name):
    workflow = _workflow(tmp_path)
    profile_path = _profile_path(tmp_path)
    profiles = load_runner_profiles(str(profile_path))
    adapted = adapt_legacy_grasp_profile(
        profiles,
        profile_name,
        allowed_work_roots=(tmp_path,),
    )
    named_service = ExecutionService(
        configured_targets={profile_name: adapted.target},
        default_target=profile_name,
    )

    migrated = launch_run(
        GRASP,
        ExecutionService(),
        input_file=workflow,
        profile=profile_name,
        profiles_path=profile_path,
    )
    named = launch_run(GRASP, named_service, input_file=workflow)

    migrated_plan = dict(migrated["evidence"]["plan"])
    named_plan = dict(named["evidence"]["plan"])
    migrated_plan.pop("profile")
    migrated_plan.pop("profiles_path")
    named_plan.pop("profile")
    named_plan.pop("profiles_path")
    assert named_plan == migrated_plan
    assert named["approval"]["token"] == migrated["approval"]["token"]
    assert named["evidence"]["input_review"]["verdict"]["label"] == (
        "unsupported"
    )
    assert [item["code"] for item in named["uncertainty"]] == [
        "input_parser_unavailable",
        "input_linter_unavailable",
        "artifact_kind_unmatched",
    ]


def test_grasp_adapter_requires_explicit_container(tmp_path):
    profiles = _profiles()
    profiles["profiles"]["grasp_local"].pop("apptainer_sif")

    with pytest.raises(ValueError, match="requires apptainer_sif"):
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

    with pytest.raises(ValueError, match="contains an unresolved variable"):
        adapt_legacy_grasp_profile(
            profiles,
            "grasp_slurm",
            allowed_work_roots=(tmp_path,),
        )


def test_grasp_plan_requires_existing_workflow_script(tmp_path):
    with pytest.raises(ValueError, match="workflow script does not exist"):
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
    assert installation.setup_lines == ("module load tacc-apptainer",)

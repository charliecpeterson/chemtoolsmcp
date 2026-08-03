"""Exact contracts for launch plans, targets, and render-only executors."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import chemtools.execution.executors as compatibility_executors
from chemtools.core.execution import (
    ResourceRequest,
)
from chemtools.execution import LocalExecutor, SlurmExecutor, WorkRootViolation
from chemtools.core.runner import (
    load_runner_profiles,
    render_nwchem_run,
)
from chemtools.programs.nwchem.launch import (
    adapt_legacy_nwchem_profile,
    build_nwchem_launch_plan,
)


PROFILE_PATH = (
    Path(__file__).parents[1]
    / "chemtools"
    / "runner_profiles.example.json"
)
STAMPEDE3_PROFILE_PATH = (
    Path(__file__).parents[1]
    / "examples"
    / "tacc_stampede3"
    / "runner_profiles.yaml"
)


def _input(tmp_path: Path) -> Path:
    input_path = tmp_path / "water.nw"
    input_path.write_text(
        "start water\ngeometry\nO 0 0 0\nend\ntask scf energy\n",
        encoding="utf-8",
    )
    return input_path


def test_split_executor_compatibility_imports_are_direct():
    assert compatibility_executors.LocalExecutor is LocalExecutor
    assert compatibility_executors.SlurmExecutor is SlurmExecutor
    assert compatibility_executors.WorkRootViolation is WorkRootViolation


def test_same_nwchem_plan_renders_for_local_mpi_and_slurm(tmp_path):
    input_path = _input(tmp_path)
    profiles = load_runner_profiles(str(PROFILE_PATH))
    local_profile = adapt_legacy_nwchem_profile(
        profiles,
        "local_mpirun",
        allowed_work_roots=(tmp_path,),
    )
    slurm_profile = adapt_legacy_nwchem_profile(
        profiles,
        "slurm_cpu",
        allowed_work_roots=(tmp_path,),
    )
    plan = build_nwchem_launch_plan(
        input_path,
        slurm_profile.default_resources,
    )

    local = LocalExecutor().render(plan, local_profile.target)
    slurm = SlurmExecutor().render(plan, slurm_profile.target)

    assert local.argv == (
        "mpirun",
        "-np",
        "16",
        "nwchem",
        "water.nw",
    )
    assert local.environment == {"OMP_NUM_THREADS": "1"}
    assert local.stdout_path == tmp_path / "water.out"
    assert local.stderr_path == tmp_path / "water.err"
    assert slurm.command.argv == ("srun", "nwchem", "water.nw")
    assert slurm.submit_argv == (
        "sbatch",
        str(tmp_path / "water.job"),
    )
    assert slurm.script_text == (
        "#!/bin/bash\n"
        "#SBATCH --job-name=water\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --ntasks=16\n"
        "#SBATCH --cpus-per-task=1\n"
        f"#SBATCH --output={tmp_path / 'water.out'}\n"
        f"#SBATCH --error={tmp_path / 'water.err'}\n"
        "#SBATCH --time=24:00:00\n"
        "#SBATCH --partition=compute\n"
        "module purge\n"
        "module load nwchem\n"
        f"cd -- {tmp_path}\n"
        "srun nwchem water.nw\n"
    )


def test_new_render_matches_legacy_nwchem_command_boundaries(tmp_path):
    input_path = _input(tmp_path)
    profiles = load_runner_profiles(str(PROFILE_PATH))
    adapted = adapt_legacy_nwchem_profile(
        profiles,
        "local_mpirun",
        allowed_work_roots=(tmp_path,),
    )
    resources = ResourceRequest(mpi_ranks=8, omp_threads=1)
    plan = build_nwchem_launch_plan(input_path, resources)

    rendered = LocalExecutor().render(plan, adapted.target)
    legacy = render_nwchem_run(
        str(input_path),
        "local_mpirun",
        profiles=profiles,
    )

    assert rendered.argv == (
        *tuple(legacy["launcher_command"].split()),
        input_path.name,
    )
    assert str(rendered.stdout_path) == legacy["output_file"]
    assert str(rendered.stderr_path) == legacy["error_file"]
    assert rendered.working_directory == Path(legacy["working_directory"])
    assert rendered.environment["OMP_NUM_THREADS"] == (
        legacy["environment"]["OMP_NUM_THREADS"]
    )


def test_legacy_slurm_profile_preserves_target_owned_commands(tmp_path):
    profiles = load_runner_profiles(str(PROFILE_PATH))
    adapted = adapt_legacy_nwchem_profile(
        profiles,
        "slurm_cpu",
        allowed_work_roots=(tmp_path,),
    )
    target = adapted.target

    assert target.executor == "slurm"
    assert target.programs["nwchem"].launcher_argv == ("srun",)
    assert target.programs["nwchem"].executable_argv == ("nwchem",)
    assert target.programs["nwchem"].setup_lines == (
        "module purge",
        "module load nwchem",
    )
    assert target.scheduler is not None
    assert target.scheduler.status_argv == (
        "squeue",
        "-j",
        "{job_id}",
        "-h",
        "-o",
        "%T",
    )
    assert target.scheduler.accounting_argv == (
        "sacct",
        "-n",
        "-X",
        "-j",
        "{job_id}",
        "-o",
        "State%30,ExitCode,ElapsedRaw",
        "-P",
    )
    assert target.scheduler.cancel_argv == ("scancel", "{job_id}")
    with pytest.raises(TypeError):
        target.programs["other"] = target.programs["nwchem"]


def test_stampede3_hardware_memory_is_not_a_slurm_memory_request(tmp_path):
    input_path = _input(tmp_path)
    profiles = load_runner_profiles(str(STAMPEDE3_PROFILE_PATH))
    adapted = adapt_legacy_nwchem_profile(
        profiles,
        "stampede3_skx_dev",
        allowed_work_roots=(tmp_path,),
    )
    resources = replace(
        adapted.default_resources,
        mpi_ranks=2,
        walltime="00:05:00",
    )
    plan = build_nwchem_launch_plan(input_path, resources)

    rendered = SlurmExecutor().render(plan, adapted.target)

    assert adapted.target.hardware.memory_mb_per_node == 192000
    assert resources.memory_mb_per_node is None
    assert "#SBATCH --mem=" not in rendered.script_text
    assert "#SBATCH --partition=skx-dev\n" in rendered.script_text
    assert "#SBATCH --nodes=1\n" in rendered.script_text
    assert "#SBATCH --ntasks=2\n" in rendered.script_text
    assert "#SBATCH --time=00:05:00\n" in rendered.script_text
    assert rendered.command.argv == (
        "ibrun",
        "/home1/01775/charlesp/apps/nwchem/7.2.3/bin/nwchem",
        "water.nw",
    )


def test_explicit_memory_request_renders_slurm_mem_directive(tmp_path):
    input_path = _input(tmp_path)
    profiles = {
        "schema_version": "1.0",
        "profiles": {
            "slurm": {
                "launcher": {
                    "kind": "scheduler",
                    "scheduler_type": "slurm",
                },
                "programs": {
                    "nwchem": {
                        "launcher_argv": ["srun"],
                        "executable_argv": ["nwchem"],
                    }
                },
                "resources": {
                    "memory_mb_per_node": 64000,
                },
            }
        },
    }
    adapted = adapt_legacy_nwchem_profile(
        profiles,
        "slurm",
        allowed_work_roots=(tmp_path,),
    )
    plan = build_nwchem_launch_plan(
        input_path,
        adapted.default_resources,
    )

    rendered = SlurmExecutor().render(plan, adapted.target)

    assert adapted.target.hardware.memory_mb_per_node is None
    assert adapted.default_resources.memory_mb_per_node == 64000
    assert "#SBATCH --mem=64000M\n" in rendered.script_text


def test_render_rejects_working_directory_symlink_escape(tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    input_path = _input(outside)
    escape = allowed / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    profiles = load_runner_profiles(str(PROFILE_PATH))
    adapted = adapt_legacy_nwchem_profile(
        profiles,
        "local",
        allowed_work_roots=(allowed,),
    )
    plan = build_nwchem_launch_plan(
        input_path,
        adapted.default_resources,
    )

    with pytest.raises(
        WorkRootViolation,
        match="is outside target 'local' roots",
    ):
        LocalExecutor().render(
            replace(plan, working_directory=escape),
            adapted.target,
        )


def test_profile_adapter_rejects_unversioned_and_non_slurm_profiles(tmp_path):
    with pytest.raises(
        ValueError,
        match="unsupported legacy runner profile schema",
    ):
        adapt_legacy_nwchem_profile(
            {"profiles": {"local": {}}},
            "local",
            allowed_work_roots=(tmp_path,),
        )

    profiles = {
        "schema_version": "1.0",
        "profiles": {
            "pbs": {
                "launcher": {
                    "kind": "scheduler",
                    "scheduler_type": "pbs",
                },
                "scheduler": {"system": "pbs"},
            }
        },
    }
    with pytest.raises(
        ValueError,
        match="legacy scheduler 'pbs' is not canonical",
    ):
        adapt_legacy_nwchem_profile(
            profiles,
            "pbs",
            allowed_work_roots=(tmp_path,),
        )

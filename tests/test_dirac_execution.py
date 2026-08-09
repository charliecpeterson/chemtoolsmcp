"""DIRAC launch-plan, target, and compatibility contracts."""

import json
from pathlib import Path

import pytest

from chemtools.application.execution import ExecutionService
from chemtools.application.run_launching import LaunchRunError, launch_run
from chemtools.execution import LocalExecutor, SlurmExecutor
from chemtools.execution.profiles import load_runner_profiles
from chemtools.programs.dirac.launch import (
    adapt_legacy_dirac_profile,
    build_dirac_launch_plan,
)
from chemtools.programs.dirac import DIRAC
from chemtools.programs.dirac.runtime import prepare_launch


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


def _guided_profile_path(tmp_path: Path) -> Path:
    profiles = _profiles()
    for profile in profiles["profiles"].values():
        profile.pop("default_mw", None)
        profile.pop("default_nw", None)
    profile_path = tmp_path / "guided_profiles.json"
    profile_path.write_text(json.dumps(profiles), encoding="utf-8")
    return profile_path


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


@pytest.mark.parametrize("profile_name", ["dirac_local", "dirac_slurm"])
def test_guided_dirac_named_target_matches_profile(
    tmp_path,
    profile_name,
):
    input_path, molecule_path = _inputs(tmp_path)
    profile_path = _guided_profile_path(tmp_path)
    profiles = load_runner_profiles(str(profile_path))
    adapted = adapt_legacy_dirac_profile(
        profiles,
        profile_name,
        allowed_work_roots=(tmp_path,),
    )
    named_service = ExecutionService(
        configured_targets={profile_name: adapted.target},
        default_target=profile_name,
    )

    migrated = launch_run(
        DIRAC,
        ExecutionService(),
        input_file=input_path,
        molecule_file=molecule_path,
        profile=profile_name,
        profiles_path=profile_path,
    )
    named = launch_run(
        DIRAC,
        named_service,
        input_file=input_path,
        molecule_file=molecule_path,
    )

    migrated_plan = dict(migrated["evidence"]["plan"])
    named_plan = dict(named["evidence"]["plan"])
    migrated_plan.pop("profile")
    migrated_plan.pop("profiles_path")
    named_plan.pop("profile")
    named_plan.pop("profiles_path")
    assert named_plan == migrated_plan
    assert named["approval"]["token"] == migrated["approval"]["token"]
    assert named["input"]["auxiliary_inputs"] == [{
        "role": "molecule",
        "path": str(molecule_path),
        "size_bytes": 40,
        "sha256": (
            "e757ba0fe7251e203b0fbcfdf592baba"
            "52862179658fed024af725303e7628e6"
        ),
    }]


def test_guided_dirac_approval_binds_molecule_contents(tmp_path):
    input_path, molecule_path = _inputs(tmp_path)
    profile_path = _guided_profile_path(tmp_path)
    profiles = load_runner_profiles(str(profile_path))
    adapted = adapt_legacy_dirac_profile(
        profiles,
        "dirac_local",
        allowed_work_roots=(tmp_path,),
    )
    service = ExecutionService(
        configured_targets={"dirac_local": adapted.target},
        default_target="dirac_local",
    )
    prepared = launch_run(
        DIRAC,
        service,
        input_file=input_path,
        molecule_file=molecule_path,
    )
    original_sha256 = prepared["input"]["auxiliary_inputs"][0]["sha256"]

    molecule_path.write_text(
        molecule_path.read_text(encoding="utf-8") + "# changed\n",
        encoding="utf-8",
    )
    invalidated = launch_run(
        DIRAC,
        service,
        input_file=input_path,
        molecule_file=molecule_path,
        approval_token=prepared["approval"]["token"],
    )

    assert invalidated["status"] == "approval_invalidated"
    assert invalidated["input"]["auxiliary_inputs"][0]["sha256"] != (
        original_sha256
    )


def test_guided_dirac_requires_molecule_file(tmp_path):
    input_path, _ = _inputs(tmp_path)

    with pytest.raises(LaunchRunError) as caught:
        launch_run(
            DIRAC,
            ExecutionService(),
            input_file=input_path,
            profile="dirac_local",
            profiles_path=_guided_profile_path(tmp_path),
        )

    assert caught.value.as_dict() == {
        "error": "invalid_launch_request",
        "message": "molecule_file is required for dirac",
        "program": "dirac",
    }


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

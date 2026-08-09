"""Compatibility tests for program-scoped version 1 installations.

The standard program block wins while legacy field locations remain readable.
"""

from pathlib import Path

import pytest

from chemtools.execution import LocalExecutor, SlurmExecutor
from chemtools.execution.profiles import (
    declared_program_installation,
    load_runner_profiles,
    merge_profile_resources,
    render_profile_value,
    resolve_runner_profile,
)
from chemtools.programs.dirac.launch import adapt_legacy_dirac_profile
from chemtools.programs.grasp.launch import adapt_legacy_grasp_profile
from chemtools.programs.molcas.launch import adapt_legacy_molcas_profile
from chemtools.programs.molcas.runtime import prepare_launch
from chemtools.programs.nwchem.launch import (
    adapt_legacy_nwchem_profile,
    build_nwchem_launch_plan,
)
from chemtools.programs.nwchem.runner import inspect_runner_profiles


ADAPTERS = {
    "dirac": adapt_legacy_dirac_profile,
    "grasp": adapt_legacy_grasp_profile,
    "molcas": adapt_legacy_molcas_profile,
    "nwchem": adapt_legacy_nwchem_profile,
}


def test_guided_profile_helpers_preserve_version_1_rendering_rules():
    profile = {
        "resources": {
            "nodes": 2,
            "mpi_ranks": 8,
            "cores_per_node": 48,
            "account": None,
        },
    }

    resources = merge_profile_resources(profile, {"omp_threads": 2})

    assert resources == {
        "nodes": 2,
        "mpi_ranks": 96,
        "cores_per_node": 48,
        "omp_threads": 2,
        "account": None,
    }
    assert render_profile_value(
        "{mpi_ranks}-{omp_threads}-{account}",
        resources,
    ) == "96-2-"


def _scheduler_profiles(
    program: str,
    *,
    settings: dict,
) -> dict:
    return {
        "schema_version": "1.0",
        "__source__": "test",
        "profiles": {
            "target": {
                "launcher": {
                    "kind": "scheduler",
                    "submit_command": "sbatch",
                },
                "scheduler": {"system": "slurm"},
                "programs": {program: settings},
                "execution": {
                    "nwchem_executable": "/legacy/nwchem",
                    "mpi_launch": "legacy-mpi",
                    "apptainer_sif": "/legacy/molcas.sif",
                    "pymolcas_command": "legacy-pymolcas",
                },
                "container_sif": "/legacy/dirac.sif",
                "pam_dirac_binary": "legacy-pam-dirac",
                "apptainer_sif": "/legacy/grasp.sif",
                "resources": {"mpi_ranks": 8},
            },
        },
    }


@pytest.mark.parametrize("program", sorted(ADAPTERS))
def test_program_block_takes_precedence_for_every_adapter(
    tmp_path: Path,
    program: str,
):
    settings = {
        "launcher_argv": ["srun", "--ntasks={mpi_ranks}"],
        "executable_argv": [f"/apps/{program}"],
    }
    adapted = ADAPTERS[program](
        _scheduler_profiles(program, settings=settings),
        "target",
        allowed_work_roots=(tmp_path,),
    )
    installation = adapted.target.programs[program]

    assert installation.launcher_argv == (
        "srun",
        "--ntasks={mpi_ranks}",
    )
    assert installation.executable_argv == (f"/apps/{program}",)


def test_program_block_owns_program_runtime_defaults(tmp_path: Path):
    molcas = adapt_legacy_molcas_profile(
        _scheduler_profiles(
            "molcas",
            settings={
                "executable_argv": ["pymolcas"],
                "parallel_caspt2_supported": False,
            },
        ),
        "target",
        allowed_work_roots=(tmp_path,),
    )
    dirac_profiles = _scheduler_profiles(
        "dirac",
        settings={
            "executable_argv": ["pam-dirac"],
            "default_mpi": 12,
            "default_mw": 384,
            "default_nw": 768,
        },
    )
    dirac_profiles["profiles"]["target"]["resources"].pop("mpi_ranks")
    dirac_profiles["profiles"]["target"].update({
        "default_mpi": 3,
        "default_mw": 4,
        "default_nw": 5,
    })
    dirac = adapt_legacy_dirac_profile(
        dirac_profiles,
        "target",
        allowed_work_roots=(tmp_path,),
    )

    assert molcas.parallel_caspt2_supported is False
    assert dirac.default_resources.mpi_ranks == 12
    assert dirac.master_memory_mb == 384
    assert dirac.node_memory_mb == 768


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            {"executable_argv": "nwchem"},
            "executable_argv must be an array of strings",
        ),
        (
            {"launcher_argv": [], "executable_argv": []},
            "executable_argv must be a non-empty array",
        ),
    ],
)
def test_program_argument_arrays_are_validated(settings, message):
    with pytest.raises(ValueError, match=message):
        declared_program_installation(
            {"programs": {"nwchem": settings}},
            "nwchem",
        )


def test_profile_adapter_uses_standard_direct_program_command(
    tmp_path: Path,
):
    input_path = tmp_path / "water.nw"
    input_path.write_text("task scf energy\n", encoding="utf-8")
    profiles = {
        "schema_version": "1.0",
        "__source__": "test",
        "profiles": {
            "local": {
                "launcher": {
                    "kind": "direct",
                    "command": "legacy-nwchem",
                },
                "programs": {
                    "nwchem": {
                        "launcher_argv": [
                            "mpirun",
                            "-np",
                            "{mpi_ranks}",
                        ],
                        "executable_argv": ["/apps/nwchem"],
                    },
                },
                "resources": {"mpi_ranks": 6},
            },
        },
    }

    adapted = adapt_legacy_nwchem_profile(
        profiles,
        "local",
        allowed_work_roots=(tmp_path,),
    )
    plan = build_nwchem_launch_plan(
        input_path,
        adapted.default_resources,
    )
    rendered = LocalExecutor().render(plan, adapted.target)

    assert rendered.argv == (
        "mpirun",
        "-np",
        "6",
        "/apps/nwchem",
        "water.nw",
    )


def test_profile_adapter_exposes_standard_scheduler_program_command(
    tmp_path: Path,
):
    input_path = tmp_path / "water.nw"
    input_path.write_text("task scf energy\n", encoding="utf-8")
    profiles = {
        "schema_version": "1.0",
        "__source__": "test",
        "profiles": {
            "slurm": {
                "launcher": {"kind": "scheduler"},
                "scheduler": {
                    "system": "slurm",
                    "script_template": "{program_command} {input_file}\n",
                },
                "programs": {
                    "nwchem": {
                        "launcher_argv": ["srun"],
                        "executable_argv": ["/apps/nwchem"],
                    },
                },
            },
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

    assert rendered.command.argv == ("srun", "/apps/nwchem", "water.nw")
    assert rendered.script_text.endswith("srun /apps/nwchem water.nw\n")


def test_molcas_preview_reads_standard_program_block(tmp_path: Path):
    input_path = tmp_path / "caspt2.input"
    input_path.write_text("&CASPT2\n", encoding="utf-8")

    preview = prepare_launch(
        input_path,
        profile={
            "programs": {
                "molcas": {
                    "launcher_argv": [
                        "apptainer",
                        "exec",
                        "/containers/molcas.sif",
                    ],
                    "executable_argv": ["/opt/molcas/pymolcas"],
                    "parallel_caspt2_supported": False,
                },
            },
            "execution": {
                "apptainer_sif": "/legacy/molcas.sif",
                "pymolcas_command": "legacy-pymolcas",
                "parallel_caspt2_supported": True,
            },
        },
        requested_np=8,
    )

    assert preview["command"] == [
        "apptainer",
        "exec",
        "/containers/molcas.sif",
        "/opt/molcas/pymolcas",
        "-np",
        "1",
        str(input_path),
    ]
    assert preview["effective_np"] == 1
    assert preview["apptainer_sif"] == "/containers/molcas.sif"


def test_profile_inspection_reports_standard_nwchem_arrays():
    profile_path = (
        Path(__file__).parents[1]
        / "chemtools"
        / "runner_profiles.example.json"
    )

    inspected = inspect_runner_profiles(str(profile_path))

    assert inspected["profiles"]["local"]["nwchem_executable"] == (
        "nwchem"
    )
    assert inspected["profiles"]["local"]["mpi_launch"] is None
    assert inspected["profiles"]["local_mpirun"]["mpi_launch"] == (
        "mpirun -np '{mpi_ranks}'"
    )


def test_bundled_slurm_example_is_portable_and_resolves():
    profile_path = (
        Path(__file__).parents[1]
        / "chemtools"
        / "runner_profiles.slurm.example.yaml"
    )

    profiles = load_runner_profiles(str(profile_path))
    profile = resolve_runner_profile(profiles, "slurm")

    assert profile["launcher"] == {
        "kind": "scheduler",
        "submit_command": "sbatch",
        "scheduler_type": "slurm",
        "job_id_regex": "Submitted batch job (\\d+)",
        "status_command": "squeue -j {job_id} -h -o %T",
        "cancel_command": "scancel {job_id}",
    }
    assert profile["scheduler"]["system"] == "slurm"
    assert profile["programs"]["nwchem"] == {
        "launcher_argv": ["srun"],
        "executable_argv": ["nwchem"],
    }
    text = profile_path.read_text(encoding="utf-8")
    assert "/home/" not in text
    assert "/home1/" not in text
    assert "/Users/" not in text

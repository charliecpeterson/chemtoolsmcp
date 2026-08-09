"""Schema-2 target files define trusted commands, roots, and defaults."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import pytest

from chemtools.execution.targets import (
    TARGET_CONFIG_SCHEMA,
    load_target_catalog,
    parse_target_catalog,
)


def _payload(root: str) -> dict:
    return {
        "schema_version": TARGET_CONFIG_SCHEMA,
        "chemtools": {
            "enable_execution": False,
            "default_target": "workstation",
        },
        "targets": {
            "workstation": {
                "executor": "local",
                "allowed_work_roots": [root],
                "hardware": {
                    "cores_per_node": 20,
                    "memory_mb_per_node": 64000,
                    "cpu_arch": "x86_64",
                },
                "resources": {
                    "mpi_ranks": 8,
                    "omp_threads": 2,
                },
                "programs": {
                    "nwchem": {
                        "launcher_argv": [
                            "mpirun",
                            "-np",
                            "{mpi_ranks}",
                        ],
                        "executable_argv": ["/apps/nwchem"],
                        "environment": {
                            "OMP_NUM_THREADS": "{omp_threads}",
                        },
                    },
                },
            },
            "slurm_cpu": {
                "executor": "slurm",
                "allowed_work_roots": [root],
                "resources": {
                    "nodes": 2,
                    "mpi_ranks": 96,
                    "omp_threads": 1,
                    "walltime": "02:00:00",
                    "partition": "compute",
                },
                "scheduler": {
                    "submit_argv": ["sbatch", "{script_file}"],
                    "status_argv": [
                        "squeue",
                        "-j",
                        "{job_id}",
                        "-h",
                        "-o",
                        "%T",
                    ],
                    "accounting_argv": [
                        "sacct",
                        "-n",
                        "-X",
                        "-j",
                        "{job_id}",
                        "-o",
                        "State%30,ExitCode,ElapsedRaw",
                        "-P",
                    ],
                    "cancel_argv": ["scancel", "{job_id}"],
                },
                "programs": {
                    "nwchem": {
                        "launcher_argv": ["srun"],
                        "executable_argv": ["nwchem"],
                        "setup_lines": [
                            "module purge",
                            "module load nwchem",
                        ],
                    },
                },
            },
        },
    }


def test_catalog_builds_complete_local_and_slurm_targets(tmp_path):
    catalog = parse_target_catalog(
        _payload(str(tmp_path)),
        source=tmp_path / "targets.yaml",
    )

    local = catalog.resolve(program="nwchem")
    slurm = catalog.resolve("slurm_cpu", program="nwchem")

    assert catalog.enable_execution is False
    assert catalog.default_target == "workstation"
    assert local.allowed_work_roots == (tmp_path,)
    assert local.hardware.cores_per_node == 20
    assert local.default_resources.mpi_ranks == 8
    assert local.default_resources.omp_threads == 2
    assert local.programs["nwchem"].launcher_argv == (
        "mpirun",
        "-np",
        "{mpi_ranks}",
    )
    assert slurm.default_resources.nodes == 2
    assert slurm.default_resources.mpi_ranks == 96
    assert slurm.scheduler is not None
    assert slurm.scheduler.submit_argv == (
        "sbatch",
        "{script_file}",
    )
    assert slurm.programs["nwchem"].setup_lines == (
        "module purge",
        "module load nwchem",
    )


def test_catalog_loads_json_from_the_environment(tmp_path, monkeypatch):
    config = tmp_path / "targets.json"
    config.write_text(
        json.dumps(_payload(str(tmp_path))),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHEMTOOLS_TARGETS", str(config))

    catalog = load_target_catalog()

    assert catalog.source == config
    assert catalog.resolve().name == "workstation"


def test_catalog_expands_host_environment_in_trusted_paths(
    tmp_path,
    monkeypatch,
):
    payload = _payload("$CHEMTOOLS_TEST_WORK")
    payload["targets"]["workstation"]["programs"]["nwchem"][
        "executable_argv"
    ] = ["$CHEMTOOLS_TEST_APPS/nwchem"]
    monkeypatch.setenv("CHEMTOOLS_TEST_WORK", str(tmp_path))
    monkeypatch.setenv("CHEMTOOLS_TEST_APPS", "/opt/chemistry")

    target = parse_target_catalog(
        payload,
        source=tmp_path / "targets.yaml",
    ).resolve()

    assert target.allowed_work_roots == (tmp_path,)
    assert target.programs["nwchem"].executable_argv == (
        "/opt/chemistry/nwchem",
    )


def test_catalog_never_selects_a_target_by_guessing(tmp_path):
    payload = _payload(str(tmp_path))
    payload["chemtools"].pop("default_target")
    payload["targets"] = {
        "workstation": payload["targets"]["workstation"],
    }
    catalog = parse_target_catalog(
        payload,
        source=tmp_path / "targets.yaml",
    )

    with pytest.raises(ValueError, match="target name is required"):
        catalog.resolve()


def test_catalog_rejects_unknown_fields_and_unresolved_variables(tmp_path):
    unknown = _payload(str(tmp_path))
    unknown["targets"]["workstation"]["shell_command"] = "nwchem"
    with pytest.raises(ValueError, match="unknown fields: shell_command"):
        parse_target_catalog(
            unknown,
            source=tmp_path / "targets.yaml",
        )

    unresolved = _payload("$CHEMTOOLS_MISSING_ROOT")
    with pytest.raises(ValueError, match="unresolved environment variable"):
        parse_target_catalog(
            unresolved,
            source=tmp_path / "targets.yaml",
        )


def test_catalog_rejects_missing_program_and_scheduler_mismatch(tmp_path):
    catalog = parse_target_catalog(
        _payload(str(tmp_path)),
        source=tmp_path / "targets.yaml",
    )
    with pytest.raises(ValueError, match="has no 'molcas' installation"):
        catalog.resolve(program="molcas")

    invalid = _payload(str(tmp_path))
    invalid["targets"]["workstation"]["scheduler"] = {
        "submit_argv": ["sbatch", "{script_file}"],
        "status_argv": ["squeue", "-j", "{job_id}"],
        "cancel_argv": ["scancel", "{job_id}"],
    }
    with pytest.raises(
        ValueError,
        match="local targets cannot define scheduler defaults",
    ):
        parse_target_catalog(
            invalid,
            source=tmp_path / "targets.yaml",
        )


def test_bundled_target_example_is_portable_and_execution_disabled():
    path = files("chemtools").joinpath("execution_targets.example.yaml")

    catalog = load_target_catalog(str(path))

    assert catalog.enable_execution is False
    assert tuple(catalog.targets) == ("workstation", "slurm_cpu")
    assert catalog.resolve().programs["nwchem"].executable_argv == (
        "/absolute/path/to/nwchem",
    )
    assert catalog.resolve(program="qe").programs["qe"].executable_argv == (
        "/absolute/path/to/pw.x",
    )

"""Contracts for read-only file and external Slurm status inspection."""

import json
import inspect
from pathlib import Path
import subprocess

import pytest

import chemtools.execution.external_status as external_status

from chemtools.application.execution import ExecutionService, LaunchStatusError
from chemtools.application.execution_monitoring import refresh_owned_local_status
from chemtools.execution.external_status import inspect_run_status
from chemtools.programs.nwchem.external_status import inspect_nwchem_run_status
from chemtools.programs.nwchem.runner import watch_multiple_nwchem_runs


NWCHEM_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "nwchem_pyscf"
    / "h2o_rhf_sto3g.out"
)


def _profile_path(tmp_path):
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        json.dumps({
            "schema_version": "1.0",
            "profiles": {
                "legacy_slurm": {
                    "launcher": {
                        "kind": "scheduler",
                        "scheduler_type": "slurm",
                        "status_command": (
                            "squeue -j {job_id} -h -o %T"
                        ),
                    },
                },
            },
        }),
        encoding="utf-8",
    )
    return profile_path


def test_external_slurm_status_keeps_public_response_shape(
    tmp_path,
    monkeypatch,
):
    profile_path = _profile_path(tmp_path)

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="RUNNING\n",
            stderr="",
        )

    monkeypatch.setattr(external_status.subprocess, "run", fake_run)

    status = inspect_run_status(
        profile="legacy_slurm",
        job_id="41200",
        profiles_path=str(profile_path),
    )

    assert status == {
        "output_file": {
            "path": None,
            "exists": False,
            "size_bytes": None,
            "modified_utc": None,
        },
        "input_file": {
            "path": None,
            "exists": False,
        },
        "error_file": {
            "path": None,
            "exists": False,
            "size_bytes": None,
            "modified_utc": None,
        },
        "process": {
            "process_id": None,
            "status": "unknown",
        },
        "scheduler": {
            "job_id": "41200",
            "status": "running",
            "scheduler_type": "slurm",
            "raw_state": "RUNNING",
            "command": [
                "squeue",
                "-j",
                "41200",
                "-h",
                "-o",
                "%T",
            ],
            "return_code": 0,
            "stdout": "RUNNING\n",
            "stderr": "",
        },
        "output_summary": None,
        "progress_summary": None,
        "task_preview": None,
        "parsed_tasks": None,
        "overall_status": "running",
    }


def test_generic_status_does_not_choose_a_program_parser():
    status = inspect_run_status(output_path=str(NWCHEM_FIXTURE))

    assert status["overall_status"] == "output_present_unknown"
    assert status["output_summary"] is None
    assert status["progress_summary"] is None
    assert status["task_preview"] is None


def test_generic_status_reports_input_file_presence(tmp_path):
    input_path = tmp_path / "external.inp"
    input_path.write_text("external input\n", encoding="utf-8")

    status = inspect_run_status(input_path=str(input_path))

    assert status["input_file"] == {
        "path": str(input_path.resolve()),
        "exists": True,
    }


def test_nwchem_adapter_preserves_legacy_progress_payload():
    status = inspect_nwchem_run_status(output_path=str(NWCHEM_FIXTURE))

    assert status["overall_status"] == "completed_success"
    assert status["output_summary"] == {
        "kind": "nwchem",
        "outcome": "success",
        "task_count": 1,
        "diagnostics": [],
        "current_task_kind": "single_point",
        "current_phase": "single_point_task",
        "status_line": "Single Point · SCF is success.",
    }
    assert status["progress_summary"]["optimization_status"] == "not_detected"
    assert status["task_preview"][0]["energy_hartree"] == -74.962991605334


def test_external_status_requires_an_explicit_profile_and_job_id():
    with pytest.raises(
        ValueError,
        match="requires both profile and job_id",
    ):
        inspect_run_status(job_id="41200")


def test_external_status_has_no_arbitrary_process_contract():
    assert "process_id" not in inspect.signature(inspect_run_status).parameters


def test_owned_local_status_rejects_an_unrecorded_process(tmp_path):
    service = ExecutionService(
        enable_execution=True,
        registry_db_path=tmp_path / "registry.db",
    )

    with pytest.raises(LaunchStatusError) as captured:
        refresh_owned_local_status(
            service,
            4242,
            program="molcas",
            program_label="Molcas",
        )

    assert captured.value.as_dict()["error"] == "launch_not_owned"


def test_external_status_rejects_non_slurm_profiles(tmp_path):
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        json.dumps({
            "schema_version": "1.0",
            "profiles": {
                "pbs": {
                    "launcher": {
                        "status_command": "qstat -f {job_id}",
                    },
                    "scheduler": {"system": "pbs"},
                },
            },
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="supports only Slurm"):
        inspect_run_status(
            profile="pbs",
            job_id="42",
            profiles_path=str(profile_path),
        )


def test_multiple_file_inspection_needs_no_scheduler_identifier():
    watched = watch_multiple_nwchem_runs([
        {"output_file": str(NWCHEM_FIXTURE), "label": "water"},
    ])

    assert watched["stop_reason"] == "all_terminal"
    assert watched["poll_count"] == 1
    assert watched["jobs"] == [{
        "label": "water",
        "status": "completed_success",
        "terminal": True,
        "input_file": None,
        "output_file": str(NWCHEM_FIXTURE),
    }]


def test_multiple_external_slurm_jobs_require_profile_and_job_id():
    with pytest.raises(ValueError, match="requires both profile and job_id"):
        watch_multiple_nwchem_runs([
            {"output_file": str(NWCHEM_FIXTURE), "job_id": "42"},
        ])

"""Exact response contracts for the split legacy status adapter."""

import json
from pathlib import Path
import subprocess

import chemtools.execution.legacy_status as legacy_status

from chemtools.execution.legacy_runner import inspect_run_status
from chemtools.programs.nwchem.legacy_status import inspect_nwchem_run_status


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


def test_legacy_slurm_status_keeps_public_response_shape(
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

    monkeypatch.setattr(legacy_status.subprocess, "run", fake_run)

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

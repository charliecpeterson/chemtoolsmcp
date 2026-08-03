"""Exact response contracts for the split legacy status adapter."""

import json
import subprocess

import chemtools.execution.legacy_status as legacy_status

from chemtools.core.runner import inspect_run_status


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

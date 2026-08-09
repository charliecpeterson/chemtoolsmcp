"""External Slurm fallbacks for Molcas status and watch handlers."""

import chemtools.application.molcas_monitoring as molcas_monitoring
import chemtools.application.execution_monitoring as execution_monitoring
import chemtools.mcp.tools.molcas as molcas_tools

from chemtools.mcp.decorator import set_active_mode
from chemtools.mcp.tools.molcas import (
    _handle_get_molcas_run_status,
    _handle_watch_molcas_run,
)


def test_unowned_molcas_job_uses_external_status_and_watch(monkeypatch):
    status_calls = []
    watch_calls = []

    def fake_status(**kwargs):
        status_calls.append(kwargs)
        return {"overall_status": "queued"}

    def fake_watch(**kwargs):
        watch_calls.append(kwargs)
        return {
            "terminal": False,
            "final_status": {"overall_status": "queued"},
        }

    def reject_typed_poll(*args, **kwargs):
        raise AssertionError("unowned job reached typed polling")

    monkeypatch.setattr(
        molcas_monitoring,
        "get_molcas_run_status",
        fake_status,
    )
    monkeypatch.setattr(
        execution_monitoring,
        "watch_run_status",
        reject_typed_poll,
    )
    monkeypatch.setattr(
        molcas_tools,
        "_watch_molcas_run",
        fake_watch,
    )
    arguments = {
        "output_file": "/work/legacy.log",
        "input_file": "/work/legacy.input",
        "profile": "legacy_cluster",
        "job_id": "90400",
        "poll_interval_seconds": 0,
        "max_polls": 1,
    }

    set_active_mode("hpc")
    try:
        status = _handle_get_molcas_run_status(arguments)
        watched = _handle_watch_molcas_run(arguments)
    finally:
        set_active_mode("analysis")

    assert status == {"overall_status": "queued"}
    assert len(status_calls) == 1
    assert status_calls[0]["job_id"] == "90400"
    assert len(watch_calls) == 1
    assert watch_calls[0]["job_id"] == "90400"
    assert watched["final_status"]["overall_status"] == "queued"

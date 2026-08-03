"""Compatibility fallbacks for DIRAC status and watch handlers."""

import chemtools.application.dirac_monitoring as dirac_monitoring
import chemtools.application.execution_monitoring as execution_monitoring
import chemtools.mcp.tools.dirac as dirac_tools

from chemtools.mcp.decorator import set_active_mode
from chemtools.mcp.tools.dirac import (
    _handle_get_dirac_run_status,
    _handle_watch_dirac_run,
)


def test_unowned_dirac_job_uses_legacy_status_and_watch(monkeypatch):
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
        dirac_monitoring,
        "get_dirac_run_status",
        fake_status,
    )
    monkeypatch.setattr(
        execution_monitoring,
        "watch_run_status",
        reject_typed_poll,
    )
    monkeypatch.setattr(
        dirac_tools,
        "_watch_dirac_run",
        fake_watch,
    )
    arguments = {
        "output_file": "/work/legacy.out",
        "input_file": "/work/legacy.inp",
        "profile": "legacy_cluster",
        "job_id": "24700",
        "poll_interval_seconds": 0,
        "max_polls": 1,
    }

    set_active_mode("hpc")
    try:
        status = _handle_get_dirac_run_status(arguments)
        watched = _handle_watch_dirac_run(arguments)
    finally:
        set_active_mode("analysis")

    assert status == {"overall_status": "queued"}
    assert len(status_calls) == 1
    assert status_calls[0]["job_id"] == "24700"
    assert len(watch_calls) == 1
    assert watch_calls[0]["job_id"] == "24700"
    assert watched["final_status"]["overall_status"] == "queued"

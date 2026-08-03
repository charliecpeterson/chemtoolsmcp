"""Compatibility fallbacks for GRASP status and watch handlers."""

import chemtools.application.execution_monitoring as execution_monitoring
import chemtools.application.grasp_monitoring as grasp_monitoring
import chemtools.mcp.tools.grasp as grasp_tools

from chemtools.mcp.decorator import set_active_mode
from chemtools.mcp.tools.grasp import (
    _handle_get_grasp_run_status,
    _handle_watch_grasp_run,
)


def test_unowned_grasp_job_uses_legacy_status_and_watch(monkeypatch):
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
        grasp_monitoring,
        "get_grasp_run_status",
        fake_status,
    )
    monkeypatch.setattr(
        execution_monitoring,
        "watch_run_status",
        reject_typed_poll,
    )
    monkeypatch.setattr(
        grasp_tools,
        "_watch_grasp_run",
        fake_watch,
    )
    arguments = {
        "output_file": "/work/legacy.out",
        "input_file": "/work/legacy.sh",
        "profile": "legacy_cluster",
        "job_id": "13600",
        "poll_interval_seconds": 0,
        "max_polls": 1,
    }

    set_active_mode("hpc")
    try:
        status = _handle_get_grasp_run_status(arguments)
        watched = _handle_watch_grasp_run(arguments)
    finally:
        set_active_mode("analysis")

    assert status == {"overall_status": "queued"}
    assert len(status_calls) == 1
    assert status_calls[0]["job_id"] == "13600"
    assert len(watch_calls) == 1
    assert watch_calls[0]["job_id"] == "13600"
    assert watched["final_status"]["overall_status"] == "queued"

"""Shared polling behavior for external and typed status readers."""

from chemtools.core.monitoring import watch_run_status


def _status(scheduler: dict[str, str]) -> dict:
    return {
        "overall_status": "output_present_unknown",
        "output_summary": None,
        "progress_summary": None,
        "process": {"status": "unknown"},
        "output_file": {"exists": True, "size_bytes": 12},
        "scheduler": scheduler,
    }


def test_typed_not_found_does_not_infer_terminal_state():
    watched = watch_run_status(
        lambda: _status({
            "status": "not_found",
            "source": "accounting",
        }),
        poll_interval_seconds=0,
        max_polls=1,
    )

    assert watched["terminal"] is False
    assert watched["stop_reason"] == "max_polls_reached"
    assert watched["poll_count"] == 1


def test_external_not_found_with_output_uses_file_completion_evidence():
    watched = watch_run_status(
        lambda: _status({"status": "not_found"}),
        poll_interval_seconds=0,
        max_polls=1,
    )

    assert watched["terminal"] is True
    assert watched["stop_reason"] == "terminal_status"
    assert watched["poll_count"] == 1


def test_typed_local_exit_is_terminal_without_parsed_output():
    watched = watch_run_status(
        lambda: {
            "overall_status": "not_started",
            "output_summary": None,
            "progress_summary": None,
            "process": {
                "process_id": 99,
                "status": "completed",
                "return_code": 0,
            },
            "output_file": {"exists": False, "size_bytes": None},
            "scheduler": None,
        },
        poll_interval_seconds=0,
        max_polls=1,
    )

    assert watched["terminal"] is True
    assert watched["stop_reason"] == "terminal_status"
    assert watched["poll_count"] == 1

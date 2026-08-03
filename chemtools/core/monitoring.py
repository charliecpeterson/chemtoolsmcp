"""Poll calculation status readers and retain a compact progress history.

Scheduler and process ownership stay with application services. This module
only controls polling, terminal detection, adaptive intervals, and timeouts.
"""

from __future__ import annotations

import time
from typing import Any, Callable


def watch_run_status(
    status_reader: Callable[[], dict[str, Any]],
    *,
    poll_interval_seconds: float = 10.0,
    adaptive_polling: bool = True,
    max_poll_interval_seconds: float | None = 60.0,
    timeout_seconds: float | None = 3600.0,
    max_polls: int | None = None,
    history_limit: int = 8,
    stall_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    if poll_interval_seconds < 0:
        raise ValueError("poll_interval_seconds must be non-negative")
    if max_poll_interval_seconds is not None and max_poll_interval_seconds < 0:
        raise ValueError(
            "max_poll_interval_seconds must be non-negative when provided"
        )
    if timeout_seconds is not None and timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative when provided")
    if stall_timeout_seconds is not None and stall_timeout_seconds < 0:
        raise ValueError(
            "stall_timeout_seconds must be non-negative when provided"
        )
    if max_polls is not None and max_polls <= 0:
        raise ValueError("max_polls must be positive when provided")
    if history_limit <= 0:
        raise ValueError("history_limit must be positive")

    started = time.monotonic()
    poll_count = 0
    snapshots: list[dict[str, Any]] = []
    final_status: dict[str, Any] | None = None
    stop_reason = "unknown"
    terminal = False
    previous_signature: tuple[Any, ...] | None = None
    stable_poll_count = 0
    last_progress_time = started
    last_sleep_seconds = 0.0

    while True:
        final_status = status_reader()
        poll_count += 1
        elapsed_seconds = time.monotonic() - started
        snapshot = {
            "poll": poll_count,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "overall_status": final_status["overall_status"],
            "current_phase": (
                final_status.get("output_summary") or {}
            ).get("current_phase"),
            "status_line": (
                (final_status.get("output_summary") or {}).get("status_line")
                or (final_status.get("progress_summary") or {}).get(
                    "status_line"
                )
            ),
        }
        signature = (
            snapshot["overall_status"],
            snapshot["current_phase"],
            snapshot["status_line"],
            (final_status.get("process") or {}).get("status"),
            (final_status.get("output_file") or {}).get("size_bytes"),
        )
        if not snapshots or snapshot != snapshots[-1]:
            snapshots.append(snapshot)
            if len(snapshots) > history_limit:
                snapshots = snapshots[-history_limit:]

        if previous_signature is None or signature != previous_signature:
            stable_poll_count = 0
            last_progress_time = time.monotonic()
        else:
            stable_poll_count += 1
        previous_signature = signature

        if _is_terminal_status(final_status):
            terminal = True
            stop_reason = "terminal_status"
            break
        if max_polls is not None and poll_count >= max_polls:
            stop_reason = "max_polls_reached"
            break
        if timeout_seconds is not None and elapsed_seconds >= timeout_seconds:
            stop_reason = "timeout_reached"
            break
        if (
            stall_timeout_seconds is not None
            and (time.monotonic() - last_progress_time)
            >= stall_timeout_seconds
        ):
            stop_reason = "stalled_no_progress"
            break
        if poll_interval_seconds > 0:
            last_sleep_seconds = _watch_sleep_seconds(
                base_interval_seconds=poll_interval_seconds,
                stable_poll_count=stable_poll_count,
                adaptive_polling=adaptive_polling,
                max_poll_interval_seconds=max_poll_interval_seconds,
            )
            time.sleep(last_sleep_seconds)

    assert final_status is not None
    return {
        "terminal": terminal,
        "stop_reason": stop_reason,
        "poll_count": poll_count,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "adaptive_polling": adaptive_polling,
        "max_poll_interval_seconds": max_poll_interval_seconds,
        "history_limit": history_limit,
        "last_sleep_seconds": round(last_sleep_seconds, 3),
        "final_status": final_status,
        "history": snapshots,
    }


def _is_terminal_status(status: dict[str, Any]) -> bool:
    overall_status = status.get("overall_status")
    if overall_status in {
        "completed_success",
        "completed_failed",
        "completed_incomplete",
        "cancelled",
        "error_only",
    }:
        return True

    scheduler = status.get("scheduler") or {}
    sched_status = scheduler.get("status")
    typed_scheduler = scheduler.get("source") in {
        "queue",
        "accounting",
        "record",
    }
    process = status.get("process") or {}
    output_file = status.get("output_file") or {}

    if (
        "return_code" in process
        and process.get("status") in {"completed", "failed"}
    ):
        return True
    if sched_status in {
        "failed",
        "timed_out",
        "out_of_memory",
        "cancelled",
    }:
        return True
    if (
        sched_status == "completed"
        or (sched_status == "not_found" and not typed_scheduler)
    ) and output_file.get("exists"):
        return True

    return (
        overall_status == "output_present_unknown"
        and sched_status not in {"queued", "running"}
        and not (
            typed_scheduler
            and sched_status
            in {
                "suspended",
                "completing",
                "not_found",
                "unknown",
                "query_failed",
            }
        )
        and process.get("status") != "running"
        and output_file.get("exists")
    )


def _watch_sleep_seconds(
    *,
    base_interval_seconds: float,
    stable_poll_count: int,
    adaptive_polling: bool,
    max_poll_interval_seconds: float | None,
) -> float:
    if base_interval_seconds <= 0:
        return 0.0
    if not adaptive_polling:
        return base_interval_seconds
    interval = base_interval_seconds * min(2**stable_poll_count, 8)
    if max_poll_interval_seconds is not None:
        interval = min(interval, max_poll_interval_seconds)
    return interval


__all__ = ["watch_run_status"]

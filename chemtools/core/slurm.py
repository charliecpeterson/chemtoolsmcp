"""Typed results returned by the Slurm status adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from chemtools.core.execution import ExecutionLaunchRecord


SlurmStatusState = Literal[
    "queued",
    "running",
    "suspended",
    "completing",
    "completed",
    "failed",
    "timed_out",
    "out_of_memory",
    "cancelled",
    "not_found",
    "unknown",
    "query_failed",
]


@dataclass(frozen=True)
class SlurmStatusResult:
    job_id: str
    query_argv: tuple[str, ...]
    source: Literal["queue", "accounting", "record"]
    status: SlurmStatusState
    raw_state: str | None
    query_return_code: int | None
    stdout: str
    stderr: str
    checked_at: datetime
    job_exit_code: int | None = None
    termination_signal: int | None = None
    elapsed_seconds: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("job_id must be a non-empty string")
        object.__setattr__(self, "query_argv", tuple(self.query_argv))
        if self.source not in ("queue", "accounting", "record"):
            raise ValueError("invalid Slurm status source")
        if self.raw_state is not None and not self.raw_state:
            raise ValueError("raw_state must be non-empty when provided")
        if self.query_return_code is not None and (
            isinstance(self.query_return_code, bool)
            or not isinstance(self.query_return_code, int)
        ):
            raise TypeError("query_return_code must be an integer")
        for field_name in ("job_exit_code", "termination_signal"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a non-negative integer"
                )
        if self.elapsed_seconds is not None and (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or self.elapsed_seconds < 0
        ):
            raise ValueError(
                "elapsed_seconds must be a non-negative number"
            )
        if (
            self.checked_at.tzinfo is None
            or self.checked_at.utcoffset() is None
        ):
            raise ValueError("checked_at must include a UTC offset")


@dataclass(frozen=True)
class RecordedSlurmStatus:
    record: ExecutionLaunchRecord
    result: SlurmStatusResult


__all__ = [
    "RecordedSlurmStatus",
    "SlurmStatusResult",
    "SlurmStatusState",
]

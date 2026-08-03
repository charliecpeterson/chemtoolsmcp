"""Execution permission decisions and public application-service errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ExecutionOperation = Literal["launch", "cancel"]
EXECUTION_OPERATIONS = frozenset({"launch", "cancel"})


@dataclass(frozen=True)
class ExecutionDecision:
    allowed: bool
    operation: ExecutionOperation
    target: str
    error: str | None = None

    def as_dict(self) -> dict[str, str | bool | None]:
        return {
            "allowed": self.allowed,
            "error": self.error,
            "operation": self.operation,
            "target": self.target,
        }


class ExecutionDisabledError(PermissionError):
    def __init__(self, decision: ExecutionDecision) -> None:
        self.decision = decision
        super().__init__(
            f"execution is disabled for {decision.operation} on "
            f"target {decision.target!r}"
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "error": "execution_disabled",
            "operation": self.decision.operation,
            "target": self.decision.target,
        }


class LaunchCancellationError(PermissionError):
    def __init__(self, payload: dict[str, str]) -> None:
        self.payload = dict(payload)
        super().__init__(payload["error"])

    def as_dict(self) -> dict[str, str]:
        return dict(self.payload)


class LaunchStatusError(LookupError):
    def __init__(self, payload: dict[str, str]) -> None:
        self.payload = dict(payload)
        super().__init__(payload["error"])

    def as_dict(self) -> dict[str, str]:
        return dict(self.payload)


__all__ = [
    "EXECUTION_OPERATIONS",
    "ExecutionDecision",
    "ExecutionDisabledError",
    "ExecutionOperation",
    "LaunchCancellationError",
    "LaunchStatusError",
]

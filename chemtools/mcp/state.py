"""Runtime configuration and launch ownership for one MCP server process.

The CLI creates this object once and passes it through request dispatch so
tool filtering and execution ownership stay tied to the same server instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, Optional

from chemtools.application.execution import ExecutionService


@dataclass(frozen=True)
class ServerState:
    mode: str = "analysis"
    programs: Optional[FrozenSet[str]] = None
    toolset: Optional[FrozenSet[str]] = None
    execution_service: ExecutionService = field(default_factory=ExecutionService)

    @classmethod
    def create(
        cls,
        *,
        mode: str = "analysis",
        programs: Optional[Iterable[str]] = None,
        toolset: Optional[Iterable[str]] = None,
    ) -> "ServerState":
        return cls(
            mode=mode,
            programs=frozenset(programs) if programs is not None else None,
            toolset=frozenset(toolset) if toolset is not None else None,
            execution_service=ExecutionService(
                enable_execution=mode in {"local", "hpc"},
            ),
        )


__all__ = ["ServerState"]

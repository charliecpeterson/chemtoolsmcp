"""Runtime configuration and launch ownership for one MCP server process.

The CLI creates this object once and passes it through request dispatch so
tool filtering and execution ownership stay tied to the same server instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, Optional

from chemtools.application.execution import ExecutionService
from chemtools.execution.targets import TargetCatalog


@dataclass(frozen=True)
class ServerState:
    mode: str = "analysis"
    programs: Optional[FrozenSet[str]] = None
    toolset: Optional[FrozenSet[str]] = None
    target_catalog: TargetCatalog | None = None
    execution_service: ExecutionService = field(default_factory=ExecutionService)

    @classmethod
    def create(
        cls,
        *,
        mode: str = "analysis",
        programs: Optional[Iterable[str]] = None,
        toolset: Optional[Iterable[str]] = None,
        target_catalog: TargetCatalog | None = None,
        enable_execution: bool | None = None,
    ) -> "ServerState":
        enabled = (
            enable_execution
            if enable_execution is not None
            else (
                target_catalog.enable_execution
                if target_catalog is not None
                else mode in {"local", "hpc"}
            )
        )
        return cls(
            mode=mode,
            programs=frozenset(programs) if programs is not None else None,
            toolset=frozenset(toolset) if toolset is not None else None,
            target_catalog=target_catalog,
            execution_service=ExecutionService(
                enable_execution=enabled,
                configured_targets=(
                    target_catalog.targets
                    if target_catalog is not None
                    else {}
                ),
                default_target=(
                    target_catalog.default_target
                    if target_catalog is not None
                    else None
                ),
            ),
        )


__all__ = ["ServerState"]

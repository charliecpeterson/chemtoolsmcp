"""Compatibility imports for the split local and Slurm execution adapters."""

from chemtools.execution._common import WorkRootViolation
from chemtools.execution.local import LocalExecutor
from chemtools.execution.slurm import SlurmExecutor


__all__ = [
    "LocalExecutor",
    "SlurmExecutor",
    "WorkRootViolation",
]

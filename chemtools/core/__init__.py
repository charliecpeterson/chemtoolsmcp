"""Program-agnostic core for chemtoolsmcp.

This package will hold the cross-program plumbing: data shapes, plugin
protocol, runner/registry/session/workflow glue. Per-program code lives in
`chemtools.programs.<name>`.

This is the seed of the multi-program refactor — code is still being moved in.
"""

from chemtools.core import (
    artifact_classification,
    artifact_registry,
    artifacts,
    execution,
    hdf5,
    legacy_artifacts,
    monitoring,
    registry_db,
    slurm,
    systems,
    types,
    program,
    registry,
)

__all__ = [
    "artifact_classification",
    "artifact_registry",
    "artifacts",
    "execution",
    "hdf5",
    "legacy_artifacts",
    "monitoring",
    "registry_db",
    "slurm",
    "systems",
    "types",
    "program",
    "registry",
]

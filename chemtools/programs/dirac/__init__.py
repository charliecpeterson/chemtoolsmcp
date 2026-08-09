"""Validated DIRAC backend declaration."""

from __future__ import annotations

from chemtools.core.program import (
    ArtifactKindSpec,
    ProgramBackend,
    ProgramCapability,
    validate_backend,
)
from chemtools.programs.dirac._plugin_binary import DIRAC_BINARY
from chemtools.programs.dirac._plugin_launcher import DIRAC_LAUNCH_PLANNER
from chemtools.programs.dirac._plugin_parser import DIRAC_PARSER
from chemtools.programs.dirac.parse.output import (
    looks_like_dirac,
    parse_version,
)


class _DiracDetector:
    def detect(self, output_head: str) -> bool:
        return looks_like_dirac(output_head[:8192])

    def detect_version(self, output_head: str) -> str | None:
        return parse_version(output_head[:5000])


DIRAC = validate_backend(
    ProgramBackend(
        name="dirac",
        capabilities=frozenset(
            {
                ProgramCapability.OUTPUT_PARSE,
                ProgramCapability.OUTPUT_TASK_INDEX,
                ProgramCapability.OUTPUT_GEOMETRY,
                ProgramCapability.INPUT_PARSE,
                ProgramCapability.BINARY_READ,
                ProgramCapability.EXECUTION_PLAN,
            }
        ),
        artifact_kinds={
            "dirac.input": ArtifactKindSpec(
                extensions=(".inp",),
                default_roles=frozenset({"primary_input"}),
                content_kind="text",
            ),
            "dirac.molecule": ArtifactKindSpec(
                extensions=(".mol",),
                default_roles=frozenset({"auxiliary_input"}),
                content_kind="text",
            ),
            "dirac.output": ArtifactKindSpec(
                extensions=(".out", ".log"),
                default_roles=frozenset({"primary_output"}),
                content_kind="text",
            ),
            "dirac.error": ArtifactKindSpec(
                extensions=(".err",),
                default_roles=frozenset({"stderr"}),
                content_kind="text",
            ),
            "dirac.checkpoint": ArtifactKindSpec(
                extensions=(".h5",),
                default_roles=frozenset({"checkpoint", "wavefunction"}),
                content_kind="binary",
            ),
            "dirac.orbitals": ArtifactKindSpec(
                filenames=("DFCOEF", "DFPCMO", "DFACMO"),
                default_roles=frozenset({"checkpoint", "orbital"}),
                content_kind="binary",
            ),
        },
        detector=_DiracDetector(),
        parser=DIRAC_PARSER,
        binary=DIRAC_BINARY,
        launches=DIRAC_LAUNCH_PLANNER,
    )
)


__all__ = ["DIRAC"]

"""Validated OpenMolcas backend declaration."""

from __future__ import annotations

import re

from chemtools.core.program import (
    ArtifactKindSpec,
    ProgramBackend,
    ProgramCapability,
    validate_backend,
)
from chemtools.programs.molcas._plugin_binary import MOLCAS_BINARY
from chemtools.programs.molcas._plugin_drafter import MOLCAS_DRAFTER
from chemtools.programs.molcas._plugin_parser import MOLCAS_PARSER


class _MolcasDetector:
    _BANNER_RE = re.compile(
        r"^\s*OpenMolcas(?:\s|$)",
        re.IGNORECASE | re.MULTILINE,
    )

    def detect(self, output_head: str) -> bool:
        upper = output_head[:8192].upper()
        return (
            self._BANNER_RE.search(output_head[:8192]) is not None
            or "THIS RUN OF MOLCAS IS USING THE PYMOLCAS DRIVER" in upper
            or "OPENMOLCASOP" in upper
            or "DEFINITIONS: _MOLCAS_" in upper
        )

    def detect_version(self, output_head: str) -> str | None:
        return None


MOLCAS = validate_backend(
    ProgramBackend(
        name="molcas",
        capabilities=frozenset(
            {
                ProgramCapability.OUTPUT_PARSE,
                ProgramCapability.OUTPUT_TASK_INDEX,
                ProgramCapability.OUTPUT_GEOMETRY,
                ProgramCapability.OUTPUT_ORBITALS,
                ProgramCapability.OUTPUT_FREQUENCIES,
                ProgramCapability.OUTPUT_TRAJECTORY,
                ProgramCapability.OUTPUT_THERMOCHEMISTRY,
                ProgramCapability.INPUT_DRAFT,
                ProgramCapability.INPUT_LINT,
                ProgramCapability.BINARY_READ,
                ProgramCapability.BINARY_WRITE,
            }
        ),
        artifact_kinds={
            "molcas.input": ArtifactKindSpec(
                extensions=(".input", ".inp"),
                default_roles=frozenset({"primary_input"}),
                content_kind="text",
            ),
            "molcas.output": ArtifactKindSpec(
                extensions=(".out", ".log"),
                default_roles=frozenset({"primary_output"}),
                content_kind="text",
            ),
            "molcas.error": ArtifactKindSpec(
                extensions=(".err",),
                default_roles=frozenset({"stderr"}),
                content_kind="text",
            ),
            "molcas.runfile": ArtifactKindSpec(
                extensions=(".RunFile",),
                default_roles=frozenset({"checkpoint"}),
                content_kind="binary",
            ),
            "molcas.orbitals": ArtifactKindSpec(
                filenames=("INPORB",),
                default_roles=frozenset({"checkpoint", "orbital"}),
                content_kind="text",
            ),
            "molcas.jobiph": ArtifactKindSpec(
                filenames=("JOBIPH",),
                default_roles=frozenset({
                    "checkpoint",
                    "orbital",
                    "wavefunction",
                }),
                content_kind="binary",
            ),
        },
        detector=_MolcasDetector(),
        parser=MOLCAS_PARSER,
        inputs=MOLCAS_DRAFTER,
        binary=MOLCAS_BINARY,
    )
)


__all__ = ["MOLCAS"]

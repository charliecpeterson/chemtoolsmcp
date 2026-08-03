"""Validated QMCPACK backend declaration for XML input review."""

from __future__ import annotations

import re

from chemtools.core.program import (
    ArtifactKindSpec,
    ProgramBackend,
    ProgramCapability,
    validate_backend,
)
from chemtools.programs.qmcpack._plugin_inputs import QMCPACK_INPUTS
from chemtools.programs.qmcpack._plugin_parser import QMCPACK_PARSER
from chemtools.programs.qmcpack.consistency import QMCPACK_RUN_CONSISTENCY


class _QmcpackDetector:
    _BANNER_RE = re.compile(r"^\s*QMCPACK\s+([^\s]+)", re.MULTILINE)

    def detect(self, output_head: str) -> bool:
        return self._BANNER_RE.search(output_head[:8192]) is not None

    def detect_version(self, output_head: str) -> str | None:
        match = self._BANNER_RE.search(output_head[:8192])
        return match.group(1) if match else None


QMCPACK = validate_backend(
    ProgramBackend(
        name="qmcpack",
        capabilities=frozenset({
            ProgramCapability.OUTPUT_PARSE,
            ProgramCapability.OUTPUT_TASK_INDEX,
            ProgramCapability.INPUT_PARSE,
            ProgramCapability.INPUT_LINT,
            ProgramCapability.RUN_CONSISTENCY,
        }),
        artifact_kinds={
            "qmcpack.input": ArtifactKindSpec(
                extensions=(".xml",),
                default_roles=frozenset({"primary_input"}),
                content_kind="text",
            ),
            "qmcpack.output": ArtifactKindSpec(
                extensions=(".out",),
                default_roles=frozenset({"primary_output"}),
                content_kind="text",
            ),
            "qmcpack.error": ArtifactKindSpec(
                extensions=(".err",),
                default_roles=frozenset({"stderr"}),
                content_kind="text",
            ),
            "qmcpack.wavefunction_hdf5": ArtifactKindSpec(
                extensions=(".h5",),
                default_roles=frozenset({"checkpoint", "wavefunction"}),
                content_kind="binary",
            ),
            "qmcpack.scalar": ArtifactKindSpec(
                extensions=(".scalar.dat",),
                default_roles=frozenset({"auxiliary_output"}),
                content_kind="text",
            ),
            "qmcpack.dmc": ArtifactKindSpec(
                extensions=(".dmc.dat",),
                default_roles=frozenset({"auxiliary_output"}),
                content_kind="text",
            ),
        },
        detector=_QmcpackDetector(),
        parser=QMCPACK_PARSER,
        inputs=QMCPACK_INPUTS,
        consistency=QMCPACK_RUN_CONSISTENCY,
    )
)


__all__ = ["QMCPACK"]

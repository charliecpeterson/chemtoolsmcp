"""Validated Quantum ESPRESSO backend declaration."""

from __future__ import annotations

import re

from chemtools.core.program import (
    ArtifactKindSpec,
    ProgramBackend,
    ProgramCapability,
    validate_backend,
)
from chemtools.programs.qe._plugin_inputs import QE_INPUTS
from chemtools.programs.qe._plugin_diagnostics import QE_DIAGNOSTICS
from chemtools.programs.qe._plugin_parser import QE_PARSER
from chemtools.programs.qe.consistency import QE_RUN_CONSISTENCY


class _QeDetector:
    _PWSCF_BANNER_RE = re.compile(
        r"^\s*Program\s+PWSCF\s+v\.([^\s]+)\s+starts\s+on\b",
        re.IGNORECASE | re.MULTILINE,
    )
    _PW2QMCPACK_BANNER_RE = re.compile(
        r"^\s*Program\s+pw2qmcpack\s+v\.([^\s]+)\s+starts\s+on\b",
        re.IGNORECASE | re.MULTILINE,
    )

    def detect(self, output_head: str) -> bool:
        head = output_head[:8192]
        return any(pattern.search(head) is not None for pattern in (
            self._PWSCF_BANNER_RE,
            self._PW2QMCPACK_BANNER_RE,
        ))

    def detect_version(self, output_head: str) -> str | None:
        head = output_head[:8192]
        for pattern in (self._PWSCF_BANNER_RE, self._PW2QMCPACK_BANNER_RE):
            if (match := pattern.search(head)) is not None:
                return match.group(1)
        return None


QE = validate_backend(
    ProgramBackend(
        name="qe",
        capabilities=frozenset({
            ProgramCapability.OUTPUT_PARSE,
            ProgramCapability.OUTPUT_TASK_INDEX,
            ProgramCapability.OUTPUT_GEOMETRY,
            ProgramCapability.OUTPUT_TRAJECTORY,
            ProgramCapability.INPUT_PARSE,
            ProgramCapability.INPUT_LINT,
            ProgramCapability.DIAGNOSIS_RUN,
            ProgramCapability.RUN_CONSISTENCY,
        }),
        artifact_kinds={
            "qe.input": ArtifactKindSpec(
                extensions=(".in",),
                default_roles=frozenset({"primary_input"}),
                content_kind="text",
            ),
            "qe.output": ArtifactKindSpec(
                extensions=(".out",),
                default_roles=frozenset({"primary_output"}),
                content_kind="text",
            ),
            "qe.error": ArtifactKindSpec(
                extensions=(".err",),
                default_roles=frozenset({"stderr"}),
                content_kind="text",
            ),
            "qe.pw2qmcpack_hdf5": ArtifactKindSpec(
                extensions=(".pwscf.h5",),
                default_roles=frozenset({"checkpoint", "wavefunction"}),
                content_kind="binary",
            ),
        },
        detector=_QeDetector(),
        parser=QE_PARSER,
        inputs=QE_INPUTS,
        diagnostics=QE_DIAGNOSTICS,
        consistency=QE_RUN_CONSISTENCY,
    )
)


__all__ = ["QE"]

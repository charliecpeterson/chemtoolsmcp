"""Validated NWChem backend declaration."""

from __future__ import annotations

import re

from chemtools.core.program import (
    ArtifactKindSpec,
    ProgramBackend,
    ProgramCapability,
    validate_backend,
)
from chemtools.programs.nwchem._plugin_binary import NWCHEM_BINARY
from chemtools.programs.nwchem._plugin_drafter import NWCHEM_DRAFTER
from chemtools.programs.nwchem._plugin_examples import NWCHEM_EXAMPLES
from chemtools.programs.nwchem._plugin_parser import NWCHEM_PARSER
from chemtools.programs.nwchem._plugin_planner import NWCHEM_CALCULATION_PLANNER
from chemtools.programs.nwchem._plugin_launcher import NWCHEM_LAUNCH_PLANNER
from chemtools.programs.nwchem._plugin_strategist import NWCHEM_STRATEGIST
from chemtools.programs.nwchem.consistency import NWCHEM_RUN_CONSISTENCY


class _NwchemDetector:
    _SHORT_BANNER_RE = re.compile(
        r"^\s*NWChem\)?(?:\s+version)?\s+\d+\.\d+",
        re.IGNORECASE | re.MULTILINE,
    )
    _VERSION_RE = re.compile(
        r"NWChem\)?(?:\s+version)?\s+(\d+\.\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )

    def detect(self, output_head: str) -> bool:
        upper = output_head.upper()
        return (
            "NORTHWEST COMPUTATIONAL CHEMISTRY PACKAGE" in upper
            or self._SHORT_BANNER_RE.search(output_head) is not None
        )

    def detect_version(self, output_head: str) -> str | None:
        match = self._VERSION_RE.search(output_head)
        return match.group(1) if match else None


NWCHEM = validate_backend(
    ProgramBackend(
        name="nwchem",
        capabilities=frozenset(ProgramCapability),
        artifact_kinds={
            "nwchem.input": ArtifactKindSpec(
                extensions=(".nw", ".nwi"),
                default_roles=frozenset({"primary_input"}),
                content_kind="text",
            ),
            "nwchem.output": ArtifactKindSpec(
                extensions=(".out", ".nwo", ".log"),
                default_roles=frozenset({"primary_output"}),
                content_kind="text",
            ),
            "nwchem.error": ArtifactKindSpec(
                extensions=(".err",),
                default_roles=frozenset({"stderr"}),
                content_kind="text",
            ),
            "nwchem.movecs": ArtifactKindSpec(
                extensions=(".movecs",),
                default_roles=frozenset({"checkpoint", "orbital"}),
                content_kind="binary",
            ),
            "nwchem.hessian": ArtifactKindSpec(
                extensions=(".hess",),
                default_roles=frozenset({"auxiliary_output"}),
                content_kind="text",
            ),
            "nwchem.freq_restart": ArtifactKindSpec(
                extensions=(".fdrst",),
                default_roles=frozenset({"checkpoint"}),
                content_kind="unknown",
            ),
            "nwchem.trajectory": ArtifactKindSpec(
                extensions=(".xyz",),
                default_roles=frozenset({"auxiliary_output"}),
                content_kind="text",
            ),
            "nwchem.jobid": ArtifactKindSpec(
                extensions=(".jobid",),
                default_roles=frozenset({"auxiliary_output"}),
                content_kind="text",
            ),
            "nwchem.scratch": ArtifactKindSpec(
                extensions=(".db", ".rmd"),
                default_roles=frozenset({"auxiliary_output"}),
                content_kind="binary",
            ),
            "nwchem.normal_modes": ArtifactKindSpec(
                extensions=(".normal", ".nmode"),
                default_roles=frozenset({"auxiliary_output"}),
                content_kind="unknown",
            ),
        },
        detector=_NwchemDetector(),
        parser=NWCHEM_PARSER,
        inputs=NWCHEM_DRAFTER,
        binary=NWCHEM_BINARY,
        diagnostics=NWCHEM_STRATEGIST,
        resources=NWCHEM_STRATEGIST,
        progress=NWCHEM_STRATEGIST,
        consistency=NWCHEM_RUN_CONSISTENCY,
        planning=NWCHEM_CALCULATION_PLANNER,
        launches=NWCHEM_LAUNCH_PLANNER,
        examples=NWCHEM_EXAMPLES,
    )
)


__all__ = ["NWCHEM"]

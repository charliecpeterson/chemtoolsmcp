"""Validated ORCA backend for the experimental reference cases."""

from __future__ import annotations

from chemtools.core.program import (
    ArtifactKindSpec,
    ProgramBackend,
    ProgramCapability,
    validate_backend,
)
from chemtools.programs.orca._plugin_parser import ORCA_PARSER
from chemtools.programs.orca.output import looks_like_orca, parse_version


class _OrcaDetector:
    def detect(self, output_head: str) -> bool:
        return looks_like_orca(output_head)

    def detect_version(self, output_head: str) -> str | None:
        return parse_version(output_head)


ORCA = validate_backend(
    ProgramBackend(
        name="orca",
        capabilities=frozenset({
            ProgramCapability.OUTPUT_PARSE,
            ProgramCapability.OUTPUT_TASK_INDEX,
            ProgramCapability.OUTPUT_GEOMETRY,
            ProgramCapability.OUTPUT_FREQUENCIES,
            ProgramCapability.INPUT_PARSE,
        }),
        artifact_kinds={
            "orca.input": ArtifactKindSpec(
                extensions=(".inp",),
                default_roles=frozenset({"primary_input"}),
                content_kind="text",
            ),
            "orca.output": ArtifactKindSpec(
                extensions=(".out",),
                default_roles=frozenset({"primary_output"}),
                content_kind="text",
            ),
            "orca.error": ArtifactKindSpec(
                extensions=(".err",),
                default_roles=frozenset({"stderr"}),
                content_kind="text",
            ),
            "orca.wavefunction": ArtifactKindSpec(
                extensions=(".gbw",),
                default_roles=frozenset({"checkpoint", "wavefunction"}),
                content_kind="binary",
            ),
            "orca.hessian": ArtifactKindSpec(
                extensions=(".hess",),
                default_roles=frozenset({"auxiliary_output"}),
                content_kind="text",
            ),
            "orca.gradient": ArtifactKindSpec(
                extensions=(".engrad",),
                default_roles=frozenset({"auxiliary_output"}),
                content_kind="text",
            ),
            "orca.geometry": ArtifactKindSpec(
                extensions=(".xyz",),
                default_roles=frozenset({"auxiliary_output"}),
                content_kind="text",
            ),
            "orca.properties": ArtifactKindSpec(
                extensions=(".property.txt",),
                default_roles=frozenset({"auxiliary_output"}),
                content_kind="text",
            ),
            "orca.bibliography": ArtifactKindSpec(
                extensions=(".bibtex",),
                default_roles=frozenset({"auxiliary_output"}),
                content_kind="text",
            ),
            "orca.densities": ArtifactKindSpec(
                extensions=(".densities", ".densitiesinfo"),
                default_roles=frozenset({"auxiliary_output"}),
                content_kind="binary",
            ),
            "orca.optimization_state": ArtifactKindSpec(
                extensions=(".opt",),
                default_roles=frozenset({"checkpoint"}),
                content_kind="binary",
            ),
        },
        detector=_OrcaDetector(),
        parser=ORCA_PARSER,
    )
)


__all__ = ["ORCA"]

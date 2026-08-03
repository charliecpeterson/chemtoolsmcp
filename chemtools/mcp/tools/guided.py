"""Guided chemistry workflow handlers and their public MCP schemas.

These tools coordinate application services across program backends. The
program-specific and low-level generic tools remain separate providers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.application.run_inspection import (
    RunInspectionError,
    inspect_run,
    validate_primary_output_format,
)
from chemtools.application.input_review import (
    InputReviewError,
    detect_input_content_candidates,
    detect_input_backend,
    review_input,
)
from chemtools.core import registry
from chemtools.core.program import (
    ProgramBackend,
    ProgramCapability,
    UnsupportedCapabilityError,
)
from chemtools.mcp.decorator import _tool


@_tool("review_input", program="generic")
def _handle_review_input(arguments: dict[str, Any]) -> dict[str, Any]:
    input_file = arguments["input_file"]
    path = Path(input_file).expanduser().resolve()
    if not path.is_file():
        return {
            "error": "source_not_file",
            "message": f"chemistry input is not a readable file: {path}",
        }
    program = arguments.get("program")
    if program:
        try:
            backend = registry.get(program)
        except registry.ProgramNotRegistered as exc:
            return {
                "error": "program_not_registered",
                "message": str(exc),
                "registered_programs": registry.list_programs(),
            }
        if isinstance(backend, ProgramBackend):
            detected_programs = detect_input_content_candidates(
                (
                    item
                    for item in registry.iter_programs()
                    if isinstance(item, ProgramBackend)
                ),
                path,
            )
            if detected_programs and program not in detected_programs:
                return {
                    "error": "program_content_mismatch",
                    "message": (
                        "chemistry input content matches "
                        f"{', '.join(detected_programs)}, but program "
                        f"override selected {program}"
                    ),
                    "program": program,
                    "detected_programs": list(detected_programs),
                }
        resolved_by = "explicit"
    else:
        try:
            backend, resolved_by = detect_input_backend(
                (
                    item
                    for item in registry.iter_programs()
                    if isinstance(item, ProgramBackend)
                ),
                path,
            )
        except InputReviewError as exc:
            return exc.as_dict()
    if not isinstance(backend, ProgramBackend):
        return {
            "error": "unsupported_backend_contract",
            "program": backend.name,
            "message": (
                "review_input requires a capability-declared program backend"
            ),
        }
    try:
        return review_input(
            backend,
            path,
            resolved_by=resolved_by,
        )
    except InputReviewError as exc:
        return exc.as_dict()


@_tool("inspect_run", program="generic")
def _handle_inspect_run(arguments: dict[str, Any]) -> dict[str, Any]:
    output_file = arguments["output_file"]
    path = Path(output_file).expanduser().resolve()
    if not path.is_file():
        return {
            "error": "source_not_file",
            "message": f"run output is not a readable file: {path}",
        }
    artifact_files = arguments.get("artifact_files", [])
    if (
        not isinstance(artifact_files, list)
        or len(artifact_files) > 64
        or any(
            not isinstance(item, str) or not item.strip()
            for item in artifact_files
        )
    ):
        return {
            "error": "invalid_artifact_files",
            "message": (
                "artifact_files must be an array of at most 64 non-empty "
                "path strings."
            ),
        }
    try:
        backend = registry.resolve(
            program=arguments.get("program"),
            path=str(path),
        )
    except registry.ProgramDetectionAmbiguous as exc:
        return {
            "error": "program_detection_ambiguous",
            "message": str(exc),
            "candidates": list(exc.candidates),
        }
    except registry.ProgramContentMismatch as exc:
        return {
            "error": "program_content_mismatch",
            "message": str(exc),
            "program": exc.program,
            "detected_programs": list(exc.candidates),
        }
    except registry.ProgramDetectorError as exc:
        return {
            "error": "program_detector_error",
            "message": str(exc),
            "candidates": list(exc.candidates),
            "detector_failures": [
                {
                    "program": failure.program,
                    "error_type": failure.error_type,
                    "message": failure.message,
                }
                for failure in exc.failures
            ],
        }
    except registry.ProgramDetectionSourceError as exc:
        return {
            "error": "program_source_error",
            "message": str(exc),
            "path": exc.path,
            "source_failure": {
                "error_type": exc.failure.error_type,
                "message": exc.failure.message,
                "errno": exc.failure.errno,
            },
        }
    except registry.ProgramDetectionFailed as exc:
        try:
            validate_primary_output_format(path)
        except RunInspectionError as format_exc:
            return format_exc.as_dict()
        return {
            "error": "program_detection_failed",
            "message": str(exc),
            "registered_programs": registry.list_programs(),
        }
    except registry.ProgramNotRegistered as exc:
        return {
            "error": "program_not_registered",
            "message": str(exc),
            "registered_programs": registry.list_programs(),
        }

    if not isinstance(backend, ProgramBackend):
        return {
            "error": "unsupported_backend_contract",
            "program": backend.name,
            "message": (
                "inspect_run requires a capability-declared program backend"
            ),
        }
    try:
        backend.require(ProgramCapability.OUTPUT_PARSE)
    except UnsupportedCapabilityError as exc:
        return {
            "error": "unsupported_capability",
            "program": exc.program,
            "capability": exc.capability.value,
            "available_capabilities": list(exc.available_capabilities),
        }
    try:
        return inspect_run(
            backend,
            path,
            resolved_by=(
                "explicit" if arguments.get("program") else "content"
            ),
            artifact_files=artifact_files,
        )
    except RunInspectionError as exc:
        return exc.as_dict()


def guided_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "review_input",
            "description": (
                "Guided pre-execution review of one chemistry input file. "
                "Detects or uses the selected program, runs only its declared "
                "input parser and linter, and returns parsed evidence, exact "
                "issues, uncertainty about missing checks, and concrete edit "
                "actions. A clean result means the configured checks passed; "
                "it does not claim that every program rule was validated. "
                "Supported backends may attach bounded structural evidence; "
                "QE reviews explicit ibrav=0 input geometries."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to one chemistry input file.",
                    },
                    "program": {
                        "type": "string",
                        "enum": ["nwchem", "molcas", "dirac", "grasp", "qe", "qmcpack"],
                        "description": (
                            "Optional program override. Without it, Chemtools "
                            "uses conservative input-content and extension "
                            "detection."
                        ),
                    },
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "inspect_run",
            "description": (
                "Guided inspection of one primary chemistry output and an "
                "optional explicit set of related artifacts. Detects or uses "
                "the selected program. Recognized standalone NBO analysis "
                "is rejected before parsing so the caller can provide the "
                "parent calculation output. An explicit program override is "
                "rejected when another registered backend positively matches "
                "the content; detector-negative fragments remain eligible for "
                "explicit parsing. Automatic detection returns an ambiguity "
                "error when multiple backends match instead of choosing by "
                "registration order. Parses only the primary output and "
                "classifies related files without scanning directories. A "
                "related artifact declared as text contributes bounded "
                "evidence: stderr uses its tail, while other text files use "
                "whole or head-and-tail segments. Per-file and total byte "
                "limits prevent unbounded responses; binary and unknown "
                "artifacts remain metadata only. When exactly one primary "
                "input is supplied, a backend may compare it with output "
                "evidence and declared restart artifacts. Unsupported or "
                "ambiguous fields remain not checked. Returns a normalized "
                "scientific verdict, explicit uncertainty, and supported "
                "next actions. Supported backends may attach bounded, compact "
                "program-specific evidence; QE relaxation inspection includes "
                "trajectory structural findings without copying every frame. "
                "A task-outcome verdict is labeled as a fallback when the "
                "program lacks a full diagnosis adapter."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {
                        "type": "string",
                        "description": "Path to one primary output artifact.",
                    },
                    "artifact_files": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "maxItems": 64,
                        "description": (
                            "Optional paths to related inputs, stderr, "
                            "checkpoints, orbitals, or other run artifacts. "
                            "Paths are classified and observed in the supplied "
                            "order. Directories are never scanned."
                        ),
                    },
                    "program": {
                        "type": "string",
                        "enum": ["nwchem", "molcas", "dirac", "grasp", "qe", "qmcpack"],
                        "description": (
                            "Optional program override. Without it, Chemtools "
                            "detects the program from file content."
                        ),
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
    ]


__all__ = ["guided_tool_definitions"]

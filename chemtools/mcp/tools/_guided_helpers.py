"""Shared backend-resolution helpers for guided MCP handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.application.run_inspection import (
    RunInspectionError,
    validate_primary_output_format,
)
from chemtools.core import registry


def resolve_output_backend(
    path: Path,
    program: str | None,
) -> tuple[Any | None, dict[str, Any] | None]:
    try:
        return registry.resolve(program=program, path=str(path)), None
    except registry.ProgramDetectionAmbiguous as exc:
        return None, {
            "error": "program_detection_ambiguous",
            "message": str(exc),
            "candidates": list(exc.candidates),
        }
    except registry.ProgramContentMismatch as exc:
        return None, {
            "error": "program_content_mismatch",
            "message": str(exc),
            "program": exc.program,
            "detected_programs": list(exc.candidates),
        }
    except registry.ProgramDetectorError as exc:
        return None, {
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
        return None, {
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
            return None, format_exc.as_dict()
        return None, {
            "error": "program_detection_failed",
            "message": str(exc),
            "registered_programs": registry.list_programs(),
        }
    except registry.ProgramNotRegistered as exc:
        return None, {
            "error": "program_not_registered",
            "message": str(exc),
            "registered_programs": registry.list_programs(),
        }


__all__ = ["resolve_output_backend"]

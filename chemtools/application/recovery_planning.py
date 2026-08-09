"""Build a read-only recovery plan from one run and its intended state."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from chemtools.application.run_consistency import (
    compare_explicit_input_output,
)
from chemtools.application.input_review import (
    InputReviewError,
    detect_input_content_candidates,
)
from chemtools.application.run_inspection import PRIMARY_OUTPUT_LIMIT_BYTES
from chemtools.core.program import ProgramBackend, ProgramCapability


RECOVERY_PLAN_SCHEMA = "chemtools.plan-recovery/1"
_LEGACY_APPLY_RECOVERY_PROGRAMS = ("molcas", "nwchem")


class RecoveryPlanError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        program: str,
        exception_type: str | None = None,
    ) -> None:
        self.code = code
        self.program = program
        self.exception_type = exception_type
        super().__init__(message)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "error": self.code,
            "message": str(self),
            "program": self.program,
        }
        if self.exception_type is not None:
            payload["exception_type"] = self.exception_type
        return payload


class ApplyRecoveryResolutionError(ValueError):
    """Report source-resolution failures for the legacy apply operation."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        program: str | None = None,
        candidates: tuple[str, ...] = (),
    ) -> None:
        self.code = code
        self.program = program
        self.candidates = candidates
        super().__init__(message)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": self.code,
            "message": str(self),
        }
        if self.program is not None:
            payload["program"] = self.program
        if self.candidates:
            payload["detected_programs"] = list(self.candidates)
        return payload


def resolve_apply_recovery_program(
    backends: Iterable[ProgramBackend],
    *,
    input_file: str | Path | None = None,
    selected_program: str | None = None,
) -> str:
    """Resolve and cross-check the program for legacy recovery application."""
    candidates: tuple[str, ...] = ()
    if input_file is not None:
        try:
            candidates = detect_input_content_candidates(
                backends,
                input_file,
            )
        except InputReviewError:
            if selected_program is not None:
                raise

    if selected_program is None:
        selected_program = next(
            (
                program
                for program in _LEGACY_APPLY_RECOVERY_PROGRAMS
                if program in candidates
            ),
            None,
        )
    if selected_program is None:
        raise ApplyRecoveryResolutionError(
            "program_detection_failed",
            (
                "Could not auto-detect program for apply_recovery. Pass "
                "`program='nwchem'` or `program='molcas'` explicitly, or "
                "provide an output_file that hints at the program."
            ),
        )
    if candidates and selected_program not in candidates:
        raise ApplyRecoveryResolutionError(
            "program_content_mismatch",
            (
                "recovery input content matches "
                f"{', '.join(candidates)}, but selected program is "
                f"{selected_program}"
            ),
            program=selected_program,
            candidates=candidates,
        )
    return selected_program


def plan_recovery(
    backend: ProgramBackend,
    output_file: str | Path,
    *,
    input_file: str | Path | None = None,
    target: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_file).expanduser().resolve()
    input_path = (
        Path(input_file).expanduser().resolve()
        if input_file is not None
        else None
    )
    _validate_sources(backend, output_path, input_path)
    normalized_target = _normalize_target(
        backend.name,
        target if target is not None else {},
    )

    if not backend.supports(ProgramCapability.DIAGNOSIS_RECOVERY):
        raise RecoveryPlanError(
            "unsupported_capability",
            f"{backend.name!r} does not support recovery planning",
            program=backend.name,
        )
    assert backend.diagnostics is not None

    consistency, consistency_uncertainty = _check_source_consistency(
        backend,
        output_path,
        input_path,
    )

    try:
        planned = backend.diagnostics.plan_recovery(
            str(output_path),
            str(input_path) if input_path is not None else None,
            normalized_target,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RecoveryPlanError(
            "invalid_recovery_request",
            f"{backend.name} rejected the recovery request: {exc}",
            program=backend.name,
            exception_type=type(exc).__name__,
        ) from exc
    except Exception as exc:
        raise RecoveryPlanError(
            "recovery_planning_failed",
            f"{backend.name} could not plan recovery: {exc}",
            program=backend.name,
            exception_type=type(exc).__name__,
        ) from exc

    required = {"assessment", "evidence", "uncertainty", "next_actions"}
    if not isinstance(planned, Mapping) or not required.issubset(planned):
        raise RecoveryPlanError(
            "invalid_recovery_provider_result",
            f"{backend.name} recovery provider returned an invalid result",
            program=backend.name,
        )

    evidence = dict(planned["evidence"])
    evidence["input_output_consistency"] = consistency
    uncertainty = [
        *consistency_uncertainty,
        *list(planned["uncertainty"]),
    ]
    assessment = dict(planned["assessment"])
    next_actions = list(planned["next_actions"])
    if consistency.get("status") == "mismatch" and _prepares_candidate(
        evidence,
    ):
        assessment, evidence, next_actions = _block_mismatched_preparation(
            evidence,
        )

    return {
        "schema_version": RECOVERY_PLAN_SCHEMA,
        "program": {"name": backend.name},
        "source": {
            "output_path": str(output_path),
            "input_path": str(input_path) if input_path is not None else None,
        },
        "target": normalized_target,
        "assessment": assessment,
        "evidence": evidence,
        "uncertainty": uncertainty,
        "next_actions": next_actions,
    }


def _check_source_consistency(
    backend: ProgramBackend,
    output_path: Path,
    input_path: Path | None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if input_path is None:
        return {
            "status": "not_checked",
            "reason": "No explicit input file was supplied.",
        }, []
    if not backend.supports(ProgramCapability.OUTPUT_PARSE):
        return {
            "status": "unsupported",
            "input_path": str(input_path),
            "reason": f"{backend.name} does not declare output parsing.",
        }, [{
            "code": "input_output_consistency_unavailable",
            "message": f"{backend.name} does not declare output parsing.",
            "impact": (
                "Recovery planning proceeded without a source-pair check."
            ),
        }]
    assert backend.parser is not None
    try:
        parsed_output = backend.parser.parse_output(str(output_path))
    except Exception as exc:
        return {
            "status": "not_checked",
            "input_path": str(input_path),
            "reason": (
                f"Output parsing failed with {type(exc).__name__}: {exc}"
            ),
        }, [{
            "code": "consistency_output_parse_failed",
            "message": (
                f"{backend.name} could not parse the output for a source-pair "
                f"check: {exc}"
            ),
            "impact": (
                "Recovery planning proceeded without a source-pair check."
            ),
        }]
    if not isinstance(parsed_output, Mapping):
        return {
            "status": "not_checked",
            "input_path": str(input_path),
            "reason": (
                "The output parser did not return a structured mapping."
            ),
        }, [{
            "code": "invalid_consistency_output",
            "message": (
                f"{backend.name} output parser returned "
                f"{type(parsed_output).__name__}, not a mapping."
            ),
            "impact": (
                "Recovery planning proceeded without a source-pair check."
            ),
        }]
    return compare_explicit_input_output(
        backend,
        output_path,
        input_path,
        parsed_output,
        artifact_paths=(str(input_path),),
    )


def _prepares_candidate(evidence: Mapping[str, Any]) -> bool:
    if evidence.get("can_prepare") is True:
        return True
    return any(
        prepared.get("candidate_drafts")
        for prepared in evidence.get("prepared_artifacts") or []
        if isinstance(prepared, Mapping)
    )


def _block_mismatched_preparation(
    evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    proposed_plan_kind = evidence.get("plan_kind")
    blocked_evidence = {
        **evidence,
        "can_prepare": False,
        "proposed_plan_kind": proposed_plan_kind,
        "prepared_artifacts": [],
    }
    return (
        {
            "verdict": {
                "label": "source_consistency_required",
                "confidence": 0.95,
                "reasons": [
                    "The supplied input and output have a confirmed mismatch; "
                    "candidate preparation is blocked until their provenance "
                    "is resolved."
                ],
            }
        },
        blocked_evidence,
        [{
            "action": "confirm_source_artifacts",
            "reason": (
                "Confirm the input, output, and referenced restart artifacts "
                "belong to the same calculation before preparing a retry."
            ),
            "priority": 1,
        }],
    )


def _validate_sources(
    backend: ProgramBackend,
    output_path: Path,
    input_path: Path | None,
) -> None:
    if not output_path.is_file():
        raise RecoveryPlanError(
            "source_not_file",
            f"run output is not a readable file: {output_path}",
            program=backend.name,
        )
    if output_path.stat().st_size > PRIMARY_OUTPUT_LIMIT_BYTES:
        raise RecoveryPlanError(
            "primary_output_too_large",
            (
                f"run output exceeds the {PRIMARY_OUTPUT_LIMIT_BYTES}-byte "
                f"recovery-planning limit: {output_path}"
            ),
            program=backend.name,
        )
    if input_path is not None and not input_path.is_file():
        raise RecoveryPlanError(
            "source_not_file",
            f"run input is not a readable file: {input_path}",
            program=backend.name,
        )


def _normalize_target(
    program: str,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(target, Mapping):
        raise RecoveryPlanError(
            "invalid_recovery_target",
            "recovery target must be an object",
            program=program,
        )
    allowed = {
        "expected_charge",
        "expected_multiplicity",
        "expected_metal_elements",
        "expected_somo_count",
    }
    unknown = sorted(set(target) - allowed)
    if unknown:
        raise RecoveryPlanError(
            "invalid_recovery_target",
            "unsupported recovery target fields: " + ", ".join(unknown),
            program=program,
        )

    charge = target.get("expected_charge")
    if charge is not None and (
        isinstance(charge, bool) or not isinstance(charge, int)
    ):
        _invalid_target(program, "expected_charge must be an integer")
    multiplicity = target.get("expected_multiplicity")
    if multiplicity is not None and (
        isinstance(multiplicity, bool)
        or not isinstance(multiplicity, int)
        or multiplicity < 1
    ):
        _invalid_target(
            program,
            "expected_multiplicity must be a positive integer",
        )
    somo_count = target.get("expected_somo_count")
    if somo_count is not None and (
        isinstance(somo_count, bool)
        or not isinstance(somo_count, int)
        or somo_count < 0
    ):
        _invalid_target(
            program,
            "expected_somo_count must be a nonnegative integer",
        )
    metals = target.get("expected_metal_elements")
    if metals is not None and (
        not isinstance(metals, list)
        or len(metals) > 32
        or any(
            not isinstance(element, str) or not element.strip()
            for element in metals
        )
    ):
        _invalid_target(
            program,
            "expected_metal_elements must contain at most 32 element symbols",
        )

    if somo_count is None and multiplicity is not None:
        somo_count = multiplicity - 1
    return {
        "expected_charge": charge,
        "expected_multiplicity": multiplicity,
        "expected_metal_elements": (
            [element.strip().capitalize() for element in metals]
            if metals is not None
            else []
        ),
        "expected_somo_count": somo_count,
        "somo_count_source": (
            "derived_from_multiplicity"
            if target.get("expected_somo_count") is None
            and multiplicity is not None
            else "explicit"
            if somo_count is not None
            else None
        ),
    }


def _invalid_target(program: str, message: str) -> None:
    raise RecoveryPlanError(
        "invalid_recovery_target",
        message,
        program=program,
    )


__all__ = [
    "RECOVERY_PLAN_SCHEMA",
    "RecoveryPlanError",
    "plan_recovery",
]

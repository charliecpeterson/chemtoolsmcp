"""Prepare an exact launch for approval, then start only that plan."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from chemtools.application.execution import ExecutionService
from chemtools.application.input_review import review_input
from chemtools.core.execution import (
    PreparedLaunch,
    RecordedLaunch,
    RenderedCommand,
    RenderedSlurmScript,
    SlurmSubmissionResult,
)
from chemtools.core.program import ProgramBackend, ProgramCapability


LAUNCH_RUN_SCHEMA = "chemtools.launch-run/1"
_RESOURCE_FIELDS = {
    "nodes",
    "mpi_ranks",
    "omp_threads",
    "memory_mb_per_node",
    "walltime",
    "partition",
    "account",
}
_APPROVAL_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class LaunchRunError(ValueError):
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


def launch_run(
    backend: ProgramBackend,
    service: ExecutionService,
    *,
    input_file: str | Path,
    profile: str | None = None,
    profiles_path: str | Path | None = None,
    target: str | None = None,
    job_name: str | None = None,
    resources: Mapping[str, Any] | None = None,
    initialization_only: bool = False,
    approval_token: str | None = None,
) -> dict[str, Any]:
    input_path = Path(input_file).expanduser().resolve()
    if not input_path.is_file():
        raise LaunchRunError(
            "source_not_file",
            f"chemistry input is not a readable file: {input_path}",
            program=backend.name,
        )
    if profile is not None and (
        not isinstance(profile, str) or not profile.strip()
    ):
        _invalid(backend.name, "profile must be a non-empty string")
    if target is not None and (
        not isinstance(target, str) or not target.strip()
    ):
        _invalid(backend.name, "target must be a non-empty string")
    if profile is not None and target is not None:
        _invalid(backend.name, "provide profile or target, not both")
    if profiles_path is not None and profile is None:
        _invalid(backend.name, "profiles_path requires profile")
    if profiles_path is not None:
        configured_profiles = Path(profiles_path).expanduser().resolve()
        if not configured_profiles.is_file():
            raise LaunchRunError(
                "profiles_not_file",
                f"runner profiles are not a readable file: {configured_profiles}",
                program=backend.name,
            )
        normalized_profiles_path = str(configured_profiles)
    else:
        normalized_profiles_path = None
    normalized_resources = _normalize_resources(backend.name, resources)
    if not isinstance(initialization_only, bool):
        _invalid(backend.name, "initialization_only must be a boolean")
    if initialization_only and backend.name != "qmcpack":
        _invalid(
            backend.name,
            "initialization_only is supported only for qmcpack",
        )
    if approval_token is not None and (
        not isinstance(approval_token, str)
        or _APPROVAL_RE.fullmatch(approval_token) is None
    ):
        _invalid(
            backend.name,
            "approval_token must use the returned sha256:<digest> form",
        )
    if not backend.supports(ProgramCapability.EXECUTION_PLAN):
        raise LaunchRunError(
            "unsupported_capability",
            f"{backend.name!r} does not support guided launch planning",
            program=backend.name,
        )
    assert backend.launches is not None

    request: dict[str, Any] = {
        "input_file": str(input_path),
        "resources": normalized_resources,
        "initialization_only": initialization_only,
    }
    if profile is not None:
        request["profile"] = profile.strip()
        if normalized_profiles_path is not None:
            request["profiles_path"] = normalized_profiles_path
    else:
        try:
            configured_target = service.resolve_target(
                target.strip() if target is not None else None,
                program=backend.name,
            )
        except ValueError as exc:
            _invalid(backend.name, str(exc))
        request["target"] = configured_target.name
        request["execution_target"] = configured_target
    if job_name is not None:
        if not isinstance(job_name, str) or not job_name.strip():
            _invalid(backend.name, "job_name must be a non-empty string")
        request["job_name"] = job_name.strip()

    prepared = _prepare(backend, request)
    if prepared.plan.program != backend.name:
        raise LaunchRunError(
            "invalid_launch_provider_result",
            (
                f"{backend.name} launch provider returned a plan for "
                f"{prepared.plan.program!r}"
            ),
            program=backend.name,
        )
    try:
        rendered = service.render(prepared.plan, prepared.target)
    except Exception as exc:
        raise LaunchRunError(
            "launch_render_failed",
            f"{backend.name} could not render the configured launch: {exc}",
            program=backend.name,
            exception_type=type(exc).__name__,
        ) from exc

    input_review = review_input(
        backend,
        input_path,
        resolved_by="explicit",
    )
    input_identity = {
        "path": str(input_path),
        "size_bytes": input_path.stat().st_size,
        "sha256": _sha256_file(input_path),
    }
    plan_summary = _plan_summary(prepared, rendered)
    current_token = _approval_token(
        prepared,
        rendered,
        input_identity,
    )
    conflicts = _launch_conflicts(rendered)
    review_label = input_review["assessment"]["verdict"]["label"]
    blocked_by_review = review_label == "errors_found"
    uncertainty = list(input_review["uncertainty"])

    if conflicts or blocked_by_review:
        reasons = []
        if blocked_by_review:
            reasons.append("The guided input review found errors.")
        if conflicts:
            reasons.append(
                "The launch would overwrite or reuse existing artifacts."
            )
        return _response(
            backend,
            status="blocked",
            verdict={
                "label": "launch_blocked",
                "confidence": 0.95,
                "reasons": reasons,
            },
            input_identity=input_identity,
            input_review=input_review,
            plan_summary=plan_summary,
            conflicts=conflicts,
            uncertainty=uncertainty,
            approval={"required": True, "token": None},
            next_actions=_blocked_actions(input_path, conflicts, blocked_by_review),
        )

    approval = {
        "required": True,
        "token": current_token,
        "scope": (
            "This token approves only the displayed input identity, target, "
            "resources, command, configured-environment fingerprint, and "
            "artifact paths."
        ),
    }
    if approval_token is None:
        return _response(
            backend,
            status="awaiting_approval",
            verdict={
                "label": "launch_ready_for_approval",
                "confidence": 0.9,
                "reasons": [
                    "The input has no review errors and the rendered launch "
                    "has no existing-artifact conflicts."
                ],
            },
            input_identity=input_identity,
            input_review=input_review,
            plan_summary=plan_summary,
            conflicts=[],
            uncertainty=uncertainty,
            approval=approval,
            next_actions=[{
                "action": "request_launch_approval",
                "reason": (
                    "Show the displayed plan to the user. Call launch_run again "
                    "with the token only after explicit approval."
                ),
                "priority": 1,
            }],
        )

    if not hmac.compare_digest(approval_token, current_token):
        return _response(
            backend,
            status="approval_invalidated",
            verdict={
                "label": "launch_requires_new_approval",
                "confidence": 1.0,
                "reasons": [
                    "The supplied token does not match the current rendered plan."
                ],
            },
            input_identity=input_identity,
            input_review=input_review,
            plan_summary=plan_summary,
            conflicts=[],
            uncertainty=uncertainty,
            approval=approval,
            next_actions=[{
                "action": "request_launch_approval",
                "reason": "Review and approve the newly displayed plan.",
                "priority": 1,
            }],
        )

    decision = service.check("launch", prepared.target)
    if not decision.allowed:
        return _response(
            backend,
            status="execution_disabled",
            verdict={
                "label": "launch_not_started",
                "confidence": 1.0,
                "reasons": [
                    "The exact plan was approved, but this server process has "
                    "execution disabled."
                ],
            },
            input_identity=input_identity,
            input_review=input_review,
            plan_summary=plan_summary,
            conflicts=[],
            uncertainty=uncertainty,
            approval={**approval, "accepted": True},
            next_actions=[{
                "action": "restart_with_execution_enabled",
                "target_executor": prepared.target.executor,
                "reason": (
                    "Restart Chemtools in local or hpc mode, prepare the plan "
                    "again, and obtain a fresh approval token."
                ),
                "priority": 1,
            }],
        )

    try:
        launched = service.launch(prepared.plan, prepared.target)
    except Exception as exc:
        raise LaunchRunError(
            "launch_failed",
            f"{backend.name} launch failed: {exc}",
            program=backend.name,
            exception_type=type(exc).__name__,
        ) from exc
    return _launched_response(
        backend,
        launched,
        input_identity=input_identity,
        input_review=input_review,
        plan_summary=plan_summary,
        uncertainty=uncertainty,
        approval=approval,
    )


def _prepare(
    backend: ProgramBackend,
    request: Mapping[str, Any],
) -> PreparedLaunch:
    assert backend.launches is not None
    try:
        prepared = backend.launches.prepare_launch(request)
    except (KeyError, TypeError, ValueError) as exc:
        raise LaunchRunError(
            "invalid_launch_request",
            f"{backend.name} rejected the launch request: {exc}",
            program=backend.name,
            exception_type=type(exc).__name__,
        ) from exc
    except Exception as exc:
        raise LaunchRunError(
            "launch_preparation_failed",
            f"{backend.name} could not prepare the launch: {exc}",
            program=backend.name,
            exception_type=type(exc).__name__,
        ) from exc
    if not isinstance(prepared, PreparedLaunch):
        raise LaunchRunError(
            "invalid_launch_provider_result",
            f"{backend.name} launch provider returned an invalid result",
            program=backend.name,
        )
    return prepared


def _normalize_resources(
    program: str,
    resources: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if resources is None:
        return {}
    if not isinstance(resources, Mapping):
        _invalid(program, "resources must be an object")
    unknown = sorted(set(resources) - _RESOURCE_FIELDS)
    if unknown:
        _invalid(program, "unsupported resource fields: " + ", ".join(unknown))
    return dict(resources)


def _plan_summary(
    prepared: PreparedLaunch,
    rendered: RenderedCommand | RenderedSlurmScript,
) -> dict[str, Any]:
    command = rendered.command if isinstance(rendered, RenderedSlurmScript) else rendered
    summary = {
        "profile": prepared.metadata.get("profile"),
        "profiles_path": prepared.metadata.get("profiles_path"),
        "target": prepared.target.name,
        "executor": prepared.target.executor,
        "job_name": prepared.plan.job_name,
        "working_directory": str(command.working_directory),
        "argv": list(command.argv),
        "environment_keys": sorted(command.environment),
        "environment_sha256": _json_sha256(
            dict(sorted(command.environment.items()))
        ),
        "resources": asdict(prepared.plan.resources),
        "stdout_path": (
            str(command.stdout_path) if command.stdout_path is not None else None
        ),
        "stderr_path": (
            str(command.stderr_path) if command.stderr_path is not None else None
        ),
        "staged_files": [
            {
                "source": str(item.source),
                "destination": str(item.destination),
                "mode": item.mode,
                "required": item.required,
            }
            for item in command.staged_files
        ],
        "expected_artifacts": [
            {
                "kind": artifact.kind,
                "path": str(artifact.location.path),
                "roles": sorted(role.value for role in artifact.roles),
                "required": artifact.required,
            }
            for artifact in prepared.plan.expected_artifacts
        ],
    }
    if isinstance(rendered, RenderedSlurmScript):
        summary["scheduler"] = {
            "script_path": str(rendered.script_path),
            "script_sha256": hashlib.sha256(
                rendered.script_text.encode("utf-8")
            ).hexdigest(),
            "submit_argv": list(rendered.submit_argv),
        }
    else:
        summary["scheduler"] = None
    if "adjustments" in prepared.metadata:
        summary["adjustments"] = [
            dict(item)
            for item in prepared.metadata["adjustments"]
        ]
    return summary


def _approval_token(
    prepared: PreparedLaunch,
    rendered: RenderedCommand | RenderedSlurmScript,
    input_identity: Mapping[str, Any],
) -> str:
    command = rendered.command if isinstance(rendered, RenderedSlurmScript) else rendered
    snapshot = {
        "schema": LAUNCH_RUN_SCHEMA,
        "input": dict(input_identity),
        "target": prepared.target.name,
        "executor": prepared.target.executor,
        "program": prepared.plan.program,
        "job_name": prepared.plan.job_name,
        "argv": list(command.argv),
        "environment": dict(sorted(command.environment.items())),
        "working_directory": str(command.working_directory),
        "stdout_path": (
            str(command.stdout_path) if command.stdout_path is not None else None
        ),
        "stderr_path": (
            str(command.stderr_path) if command.stderr_path is not None else None
        ),
        "resources": asdict(prepared.plan.resources),
        "staged_files": [
            {
                "source": str(item.source),
                "destination": str(item.destination),
                "mode": item.mode,
                "required": item.required,
            }
            for item in command.staged_files
        ],
        "scheduler": (
            {
                "script_path": str(rendered.script_path),
                "script_text": rendered.script_text,
                "submit_argv": list(rendered.submit_argv),
            }
            if isinstance(rendered, RenderedSlurmScript)
            else None
        ),
    }
    encoded = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _launch_conflicts(
    rendered: RenderedCommand | RenderedSlurmScript,
) -> list[dict[str, str]]:
    command = rendered.command if isinstance(rendered, RenderedSlurmScript) else rendered
    candidates = [
        ("stdout", command.stdout_path),
        ("stderr", command.stderr_path),
    ]
    if isinstance(rendered, RenderedSlurmScript):
        candidates.append(("scheduler_script", rendered.script_path))
    candidates.extend(
        ("staged_destination", item.destination)
        for item in command.staged_files
    )
    return [
        {"role": role, "path": str(path)}
        for role, path in candidates
        if path is not None and os.path.lexists(path)
    ]


def _blocked_actions(
    input_path: Path,
    conflicts: list[dict[str, str]],
    blocked_by_review: bool,
) -> list[dict[str, Any]]:
    actions = []
    if blocked_by_review:
        actions.append({
            "action": "revise_input",
            "path": str(input_path),
            "reason": "Resolve the guided input-review errors before launch.",
            "priority": 1,
        })
    if conflicts:
        actions.append({
            "action": "choose_new_job_name_or_archive_artifacts",
            "paths": [item["path"] for item in conflicts],
            "reason": "Guided launch never overwrites or silently archives artifacts.",
            "priority": 1,
        })
    return actions


def _response(
    backend: ProgramBackend,
    *,
    status: str,
    verdict: dict[str, Any],
    input_identity: dict[str, Any],
    input_review: Mapping[str, Any],
    plan_summary: dict[str, Any],
    conflicts: list[dict[str, str]],
    uncertainty: list[dict[str, Any]],
    approval: dict[str, Any],
    next_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": LAUNCH_RUN_SCHEMA,
        "status": status,
        "program": {"name": backend.name},
        "input": input_identity,
        "assessment": {"verdict": verdict},
        "evidence": {
            "input_review": {
                "verdict": input_review["assessment"]["verdict"],
                "lint": input_review["evidence"]["lint"]["summary"],
                "issues": input_review["evidence"]["lint"]["issues"],
            },
            "plan": plan_summary,
            "conflicts": conflicts,
        },
        "approval": approval,
        "uncertainty": uncertainty,
        "next_actions": next_actions,
    }


def _launched_response(
    backend: ProgramBackend,
    launched: RecordedLaunch,
    *,
    input_identity: dict[str, Any],
    input_review: Mapping[str, Any],
    plan_summary: dict[str, Any],
    uncertainty: list[dict[str, Any]],
    approval: dict[str, Any],
) -> dict[str, Any]:
    record = launched.record
    if record.status in {"started", "submitted"}:
        status = "launched"
        reasons = [
            f"The approved plan was handed to the {record.executor} executor."
        ]
        next_actions = [{
            "action": "monitor_owned_launch",
            "launch_id": record.launch_id,
            "reason": "Track execution and scientific status using this launch ID.",
            "priority": 1,
        }]
    elif record.status == "submitted_untracked":
        status = "launch_untracked"
        reasons = [
            "The scheduler command succeeded, but no job ID was recognized."
        ]
        next_actions = [{
            "action": "inspect_scheduler_submission",
            "reason": (
                "Confirm whether the scheduler accepted the job before "
                "attempting another submission."
            ),
            "priority": 1,
        }]
    else:
        status = "launch_failed"
        reasons = [
            "The executor did not start or submit the approved calculation."
        ]
        next_actions = [{
            "action": "inspect_launch_failure",
            "reason": "Review the recorded submission evidence before retrying.",
            "priority": 1,
        }]
    payload = _response(
        backend,
        status=status,
        verdict={
            "label": record.status,
            "confidence": 1.0,
            "reasons": reasons,
        },
        input_identity=input_identity,
        input_review=input_review,
        plan_summary=plan_summary,
        conflicts=[],
        uncertainty=uncertainty,
        approval={**approval, "accepted": True},
        next_actions=next_actions,
    )
    payload["launch"] = {
        "launch_id": record.launch_id,
        "instance_id": record.instance_id,
        "target": record.target,
        "executor": record.executor,
        "status": record.status,
        "process_id": record.process_id,
        "job_id": record.job_id,
        "stdout_path": (
            str(record.stdout_path) if record.stdout_path is not None else None
        ),
        "stderr_path": (
            str(record.stderr_path) if record.stderr_path is not None else None
        ),
        "script_path": (
            str(record.script_path) if record.script_path is not None else None
        ),
        "created_at": record.created_at.isoformat(),
    }
    if isinstance(launched.result, SlurmSubmissionResult):
        payload["launch"]["submission"] = {
            "return_code": launched.result.return_code,
            "stdout": launched.result.stdout,
            "stderr": launched.result.stderr,
        }
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _invalid(program: str, message: str) -> None:
    raise LaunchRunError(
        "invalid_launch_request",
        message,
        program=program,
    )


__all__ = [
    "LAUNCH_RUN_SCHEMA",
    "LaunchRunError",
    "launch_run",
]

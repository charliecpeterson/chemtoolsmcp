"""MCP adapters for the guided chemistry workflow tools.

Public contracts live in `_guided_definitions.py`. The binding below validates
that every contract has exactly one `_handle_<tool name>` implementation.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from chemtools.application.run_inspection import (
    RunInspectionError,
    inspect_run,
)
from chemtools.application.input_review import (
    InputReviewError,
    detect_input_content_candidates,
    detect_input_backend,
    review_input,
)
from chemtools.application.run_comparison import compare_runs
from chemtools.application.calculation_planning import (
    CalculationPlanError,
    plan_calculation,
)
from chemtools.application.run_launching import (
    LaunchRunError,
    launch_run,
)
from chemtools.application.run_monitoring import (
    MonitorRunError,
    monitor_run,
)
from chemtools.application.input_drafting import (
    InputDraftError,
    draft_input,
)
from chemtools.application.recovery_planning import (
    RecoveryPlanError,
    plan_recovery,
)
from chemtools.core import registry
from chemtools.core.program import (
    ProgramBackend,
    ProgramCapability,
    UnsupportedCapabilityError,
)
from chemtools.mcp.decorator import _tool, get_execution_service
from chemtools.mcp.tools._guided_definitions import (
    guided_tool_definitions as _declarative_guided_tool_definitions,
)
from chemtools.mcp.tools._guided_helpers import resolve_output_backend
from chemtools.persistence.launches import UnknownLaunchRecordError


_GuidedHandler = Callable[[dict[str, Any]], dict[str, Any]]


class _GuidedToolBindings:
    def __init__(self, definitions: list[dict[str, Any]]) -> None:
        self._definitions = tuple(definitions)
        self._definitions_by_name: dict[str, dict[str, Any]] = {}
        self._handlers: dict[str, _GuidedHandler] = {}
        for definition in self._definitions:
            if not isinstance(definition, dict):
                raise TypeError("guided tool definitions must be dictionaries")
            name = definition.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("guided tool definition has no valid name")
            if name in self._definitions_by_name:
                raise ValueError(f"duplicate guided tool definition: {name!r}")
            self._definitions_by_name[name] = definition

    def handler(self, function: _GuidedHandler) -> _GuidedHandler:
        prefix = "_handle_"
        if not function.__name__.startswith(prefix):
            raise ValueError(
                "guided handler names must use the _handle_<tool name> form"
            )
        name = function.__name__[len(prefix):]
        if name not in self._definitions_by_name:
            raise ValueError(
                f"guided handler {function.__name__!r} has no tool definition"
            )
        if name in self._handlers:
            raise ValueError(f"duplicate guided tool handler: {name!r}")
        registered = _tool(name, program="generic")(function)
        self._handlers[name] = registered
        return registered

    def definitions(self) -> list[dict[str, Any]]:
        missing = [
            name
            for name in self._definitions_by_name
            if name not in self._handlers
        ]
        if missing:
            raise ValueError(
                f"guided tool definitions have no handlers: {missing}"
            )
        return deepcopy(list(self._definitions))


_GUIDED_TOOLS = _GuidedToolBindings(_declarative_guided_tool_definitions())
_guided_tool = _GUIDED_TOOLS.handler


@_guided_tool
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


@_guided_tool
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
    backend, resolution_error = resolve_output_backend(
        path,
        arguments.get("program"),
    )
    if resolution_error is not None:
        return resolution_error

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


@_guided_tool
def _handle_compare_runs(arguments: dict[str, Any]) -> dict[str, Any]:
    reference_output = Path(
        arguments["reference_output_file"]
    ).expanduser().resolve()
    candidate_output = Path(
        arguments["candidate_output_file"]
    ).expanduser().resolve()
    for role, path in (
        ("reference", reference_output),
        ("candidate", candidate_output),
    ):
        if not path.is_file():
            return {
                "error": "source_not_file",
                "message": f"{role} run output is not a readable file: {path}",
            }

    input_paths = {}
    for role, argument_name in (
        ("reference", "reference_input_file"),
        ("candidate", "candidate_input_file"),
    ):
        value = arguments.get(argument_name)
        if value is None:
            input_paths[role] = None
            continue
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            return {
                "error": "source_not_file",
                "message": f"{role} run input is not a readable file: {path}",
            }
        input_paths[role] = path

    program = arguments.get("program")
    reference_backend, resolution_error = resolve_output_backend(
        reference_output,
        program,
    )
    if resolution_error is not None:
        return resolution_error
    candidate_backend, resolution_error = resolve_output_backend(
        candidate_output,
        program,
    )
    if resolution_error is not None:
        return resolution_error
    if reference_backend.name != candidate_backend.name:
        return {
            "error": "program_mismatch",
            "message": (
                "run comparison requires outputs from the same program; "
                f"detected {reference_backend.name!r} and "
                f"{candidate_backend.name!r}"
            ),
            "reference_program": reference_backend.name,
            "candidate_program": candidate_backend.name,
        }
    if not isinstance(reference_backend, ProgramBackend):
        return {
            "error": "unsupported_backend_contract",
            "program": reference_backend.name,
            "message": (
                "compare_runs requires a capability-declared program backend"
            ),
        }
    try:
        reference_backend.require(ProgramCapability.OUTPUT_PARSE)
    except UnsupportedCapabilityError as exc:
        return {
            "error": "unsupported_capability",
            "program": exc.program,
            "capability": exc.capability.value,
            "available_capabilities": list(exc.available_capabilities),
        }
    try:
        return compare_runs(
            reference_backend,
            reference_output,
            candidate_output,
            reference_input_file=input_paths["reference"],
            candidate_input_file=input_paths["candidate"],
        )
    except RunInspectionError as exc:
        return exc.as_dict()


@_guided_tool
def _handle_plan_calculation(arguments: dict[str, Any]) -> dict[str, Any]:
    program = arguments["program"]
    try:
        backend = registry.get(program)
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
                "plan_calculation requires a capability-declared program backend"
            ),
        }
    request = {
        key: value
        for key, value in arguments.items()
        if key != "program"
    }
    try:
        return plan_calculation(backend, request)
    except CalculationPlanError as exc:
        return exc.as_dict()


@_guided_tool
def _handle_launch_run(arguments: dict[str, Any]) -> dict[str, Any]:
    program = arguments["program"]
    try:
        backend = registry.get(program)
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
                "launch_run requires a capability-declared program backend"
            ),
        }
    input_path = Path(arguments["input_file"]).expanduser().resolve()
    if input_path.is_file():
        detected_programs = detect_input_content_candidates(
            (
                item
                for item in registry.iter_programs()
                if isinstance(item, ProgramBackend)
            ),
            input_path,
        )
        if detected_programs and program not in detected_programs:
            return {
                "error": "program_content_mismatch",
                "message": (
                    "chemistry input content matches "
                    f"{', '.join(detected_programs)}, but program selected "
                    f"{program}"
                ),
                "program": program,
                "detected_programs": list(detected_programs),
            }
    try:
        return launch_run(
            backend,
            get_execution_service(),
            input_file=input_path,
            profile=arguments.get("profile"),
            profiles_path=arguments.get("profiles_path"),
            target=arguments.get("target"),
            job_name=arguments.get("job_name"),
            resources=arguments.get("resources"),
            approval_token=arguments.get("approval_token"),
        )
    except LaunchRunError as exc:
        return exc.as_dict()


@_guided_tool
def _handle_monitor_run(arguments: dict[str, Any]) -> dict[str, Any]:
    launch_id = arguments["launch_id"]
    try:
        normalized = str(UUID(launch_id))
    except (TypeError, ValueError):
        return {
            "error": "invalid_launch_id",
            "message": "launch_id must be a canonical UUID string",
        }
    if normalized != launch_id:
        return {
            "error": "invalid_launch_id",
            "message": "launch_id must be a canonical lowercase UUID string",
            "launch_id": launch_id,
        }

    service = get_execution_service()
    try:
        record = service.get_launch_record(launch_id)
    except UnknownLaunchRecordError:
        return {
            "error": "launch_not_owned",
            "message": (
                f"Chemtools does not own launch {launch_id!r} in this "
                "server process"
            ),
            "launch_id": launch_id,
        }
    try:
        backend = registry.get(record.program)
    except registry.ProgramNotRegistered as exc:
        return {
            "error": "program_not_registered",
            "message": str(exc),
            "program": record.program,
            "registered_programs": registry.list_programs(),
        }
    if not isinstance(backend, ProgramBackend):
        return {
            "error": "unsupported_backend_contract",
            "program": backend.name,
            "message": (
                "monitor_run requires a capability-declared program backend"
            ),
        }
    try:
        return monitor_run(
            backend,
            service,
            launch_id=launch_id,
        )
    except MonitorRunError as exc:
        return exc.as_dict()


@_guided_tool
def _handle_plan_recovery(arguments: dict[str, Any]) -> dict[str, Any]:
    output_path = Path(arguments["output_file"]).expanduser().resolve()
    if not output_path.is_file():
        return {
            "error": "source_not_file",
            "message": f"run output is not a readable file: {output_path}",
        }
    input_value = arguments.get("input_file")
    input_path = (
        Path(input_value).expanduser().resolve()
        if input_value is not None
        else None
    )
    if input_path is not None and not input_path.is_file():
        return {
            "error": "source_not_file",
            "message": f"run input is not a readable file: {input_path}",
        }

    backend, resolution_error = resolve_output_backend(
        output_path,
        arguments.get("program"),
    )
    if resolution_error is not None:
        return resolution_error
    if not isinstance(backend, ProgramBackend):
        return {
            "error": "unsupported_backend_contract",
            "program": backend.name,
            "message": (
                "plan_recovery requires a capability-declared program backend"
            ),
        }
    if input_path is not None:
        detected_programs = detect_input_content_candidates(
            (
                item
                for item in registry.iter_programs()
                if isinstance(item, ProgramBackend)
            ),
            input_path,
        )
        if detected_programs and backend.name not in detected_programs:
            return {
                "error": "program_content_mismatch",
                "message": (
                    "run input content matches "
                    f"{', '.join(detected_programs)}, but the output uses "
                    f"{backend.name}"
                ),
                "program": backend.name,
                "detected_programs": list(detected_programs),
            }
    target = {
        key: arguments[key]
        for key in (
            "expected_charge",
            "expected_multiplicity",
            "expected_metal_elements",
            "expected_somo_count",
        )
        if key in arguments
    }
    try:
        return plan_recovery(
            backend,
            output_path,
            input_file=input_path,
            target=target,
        )
    except RecoveryPlanError as exc:
        return exc.as_dict()


@_guided_tool
def _handle_draft_input(arguments: dict[str, Any]) -> dict[str, Any]:
    program = arguments["program"]
    try:
        backend = registry.get(program)
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
                "draft_input requires a capability-declared program backend"
            ),
        }
    specification = {
        key: value
        for key, value in arguments.items()
        if key != "program"
    }
    try:
        return draft_input(backend, specification)
    except InputDraftError as exc:
        return exc.as_dict()


def guided_tool_definitions() -> list[dict[str, Any]]:
    return _GUIDED_TOOLS.definitions()


__all__ = ["guided_tool_definitions"]

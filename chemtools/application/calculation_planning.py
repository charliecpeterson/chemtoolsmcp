"""Validate and normalize read-only calculation-strategy plans."""

from __future__ import annotations

from typing import Any, Mapping

from chemtools.core.common import ELEMENT_TO_Z
from chemtools.core.program import ProgramBackend, ProgramCapability


CALCULATION_PLAN_SCHEMA = "chemtools.plan-calculation/1"
MAX_PLAN_ELEMENTS = 32
MAX_PLAN_STAGES = 8
_STAGE_KINDS = {
    "energy",
    "optimize",
    "frequency",
}
_OPTIONAL_FIELDS = {
    "method",
    "functional",
    "basis",
    "ecp",
    "relativistic",
    "geometry_source",
    "solvent",
    "state_strategy",
}


class CalculationPlanError(ValueError):
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


def plan_calculation(
    backend: ProgramBackend,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        _invalid(backend.name, "calculation request must be an object")
    normalized = _normalize_request(backend.name, request)
    if not backend.supports(ProgramCapability.CALCULATION_PLAN):
        raise CalculationPlanError(
            "unsupported_capability",
            f"{backend.name!r} does not support calculation planning",
            program=backend.name,
        )
    assert backend.planning is not None
    try:
        planned = backend.planning.plan_calculation(normalized)
    except (KeyError, TypeError, ValueError) as exc:
        raise CalculationPlanError(
            "invalid_calculation_request",
            f"{backend.name} rejected the calculation request: {exc}",
            program=backend.name,
            exception_type=type(exc).__name__,
        ) from exc
    except Exception as exc:
        raise CalculationPlanError(
            "calculation_planning_failed",
            f"{backend.name} could not plan the calculation: {exc}",
            program=backend.name,
            exception_type=type(exc).__name__,
        ) from exc

    required = {
        "protocol",
        "stages",
        "required_decisions",
        "assumptions",
        "verdict",
    }
    if not isinstance(planned, Mapping) or not required.issubset(planned):
        raise CalculationPlanError(
            "invalid_calculation_provider_result",
            f"{backend.name} calculation planner returned an invalid result",
            program=backend.name,
        )

    decisions = list(planned["required_decisions"])
    return {
        "schema_version": CALCULATION_PLAN_SCHEMA,
        "program": {"name": backend.name},
        "request": normalized,
        "assessment": {"verdict": dict(planned["verdict"])},
        "evidence": {
            "protocol": dict(planned["protocol"]),
            "stages": list(planned["stages"]),
            "required_decisions": decisions,
        },
        "uncertainty": list(planned["assumptions"]),
        "next_actions": _next_actions(normalized, decisions),
    }


def _normalize_request(
    program: str,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    allowed = {
        "system",
        "elements",
        "charge",
        "multiplicity",
        "stages",
        *_OPTIONAL_FIELDS,
    }
    unknown = sorted(set(request) - allowed)
    if unknown:
        _invalid(program, "unsupported planning fields: " + ", ".join(unknown))

    system = request.get("system")
    if not isinstance(system, str) or not system.strip():
        _invalid(program, "system must be a non-empty string")
    elements = request.get("elements")
    if (
        not isinstance(elements, list)
        or not 1 <= len(elements) <= MAX_PLAN_ELEMENTS
        or any(not isinstance(item, str) for item in elements)
    ):
        _invalid(
            program,
            f"elements must contain 1 to {MAX_PLAN_ELEMENTS} atomic symbols",
        )
    normalized_elements = []
    for item in elements:
        symbol = item.strip().capitalize()
        if symbol not in ELEMENT_TO_Z:
            _invalid(program, f"unknown atomic symbol: {item!r}")
        if symbol not in normalized_elements:
            normalized_elements.append(symbol)

    charge = request.get("charge")
    if isinstance(charge, bool) or not isinstance(charge, int):
        _invalid(program, "charge must be an integer")
    multiplicity = request.get("multiplicity")
    if (
        isinstance(multiplicity, bool)
        or not isinstance(multiplicity, int)
        or multiplicity < 1
    ):
        _invalid(program, "multiplicity must be a positive integer")
    stages = request.get("stages")
    if (
        not isinstance(stages, list)
        or not 1 <= len(stages) <= MAX_PLAN_STAGES
        or any(stage not in _STAGE_KINDS for stage in stages)
    ):
        _invalid(
            program,
            "stages must contain 1 to "
            f"{MAX_PLAN_STAGES} values from: {', '.join(sorted(_STAGE_KINDS))}",
        )

    normalized = {
        "system": system.strip(),
        "elements": normalized_elements,
        "charge": charge,
        "multiplicity": multiplicity,
        "stages": list(stages),
    }
    for field in _OPTIONAL_FIELDS:
        if field not in request:
            continue
        value = request[field]
        if field in {"basis", "ecp"}:
            _validate_basis_like(program, field, value)
        elif not isinstance(value, str) or not value.strip():
            _invalid(program, f"{field} must be a non-empty string")
        normalized[field] = (
            dict(value) if isinstance(value, Mapping) else value.strip()
        )
    return normalized


def _validate_basis_like(program: str, field: str, value: Any) -> None:
    if isinstance(value, str) and value.strip():
        return
    if isinstance(value, Mapping) and value and all(
        isinstance(element, str)
        and element.strip()
        and isinstance(name, str)
        and name.strip()
        for element, name in value.items()
    ):
        return
    _invalid(
        program,
        f"{field} must be a non-empty string or element-to-name mapping",
    )


def _next_actions(
    request: Mapping[str, Any],
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if decisions:
        return [{
            "action": "resolve_scientific_decisions",
            "fields": [decision["field"] for decision in decisions],
            "reason": (
                "Set these choices before asking Chemtools to draft input syntax."
            ),
            "priority": 1,
        }]
    return [{
        "action": "draft_stage_inputs",
        "tool": "draft_input",
        "reason": (
            "Draft and review each stage separately, preserving the optimized "
            "geometry and electronic-state choices between dependent stages."
        ),
        "stage_kinds": list(request["stages"]),
        "priority": 1,
    }]


def _invalid(program: str, message: str) -> None:
    raise CalculationPlanError(
        "invalid_calculation_request",
        message,
        program=program,
    )


__all__ = [
    "CALCULATION_PLAN_SCHEMA",
    "MAX_PLAN_ELEMENTS",
    "MAX_PLAN_STAGES",
    "CalculationPlanError",
    "plan_calculation",
]

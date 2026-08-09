"""Render and check one program-neutral chemistry input specification."""

from __future__ import annotations

import math
from typing import Any, Mapping

from chemtools.core.program import ProgramBackend, ProgramCapability
from chemtools.core.types import InputSpec


INPUT_DRAFT_SCHEMA = "chemtools.draft-input/1"
MAX_DRAFT_ATOMS = 2048
_REQUIRED_FIELDS = (
    "atoms",
    "charge",
    "multiplicity",
    "method",
    "basis",
    "task",
)
_TASKS = {
    "energy",
    "gradient",
    "optimize",
    "saddle",
    "frequency",
    "property",
    "dynamics",
}
_METHODS_BY_PROGRAM = {
    "nwchem": {"dft", "hf", "scf"},
    "molcas": {
        "caspt2",
        "casscf",
        "dft",
        "hf",
        "ksdft",
        "ms-caspt2",
        "ms-raspt2",
        "raspt2",
        "rasscf",
        "rms-caspt2",
        "scf",
        "xdw-caspt2",
        "xms-caspt2",
        "xms-raspt2",
    },
}
_TASKS_BY_PROGRAM = {
    "molcas": {"energy"},
    "nwchem": _TASKS,
}


class InputDraftError(ValueError):
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


def draft_input(
    backend: ProgramBackend,
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    if not backend.supports(ProgramCapability.INPUT_DRAFT):
        raise InputDraftError(
            "unsupported_capability",
            f"{backend.name!r} does not support input drafting",
            program=backend.name,
        )
    if not isinstance(specification, Mapping):
        raise InputDraftError(
            "invalid_input_specification",
            "input specification must be a mapping",
            program=backend.name,
        )
    _validate_specification(backend.name, specification)
    assert backend.inputs is not None

    try:
        rendered = backend.inputs.draft_input(
            InputSpec(**dict(specification)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InputDraftError(
            "invalid_input_specification",
            f"{backend.name} rejected the input specification: {exc}",
            program=backend.name,
            exception_type=type(exc).__name__,
        ) from exc
    except Exception as exc:
        raise InputDraftError(
            "input_draft_failed",
            f"{backend.name} could not draft the requested input: {exc}",
            program=backend.name,
            exception_type=type(exc).__name__,
        ) from exc
    if not isinstance(rendered, str) or not rendered.strip():
        raise InputDraftError(
            "invalid_input_drafter_result",
            f"{backend.name} input drafter returned no input text",
            program=backend.name,
        )

    issues, lint_status, uncertainty = _lint_rendered_input(
        backend,
        rendered,
    )
    issue_counts = _issue_counts(issues)
    verdict = _verdict(lint_status, issue_counts)

    return {
        "schema_version": INPUT_DRAFT_SCHEMA,
        "program": {"name": backend.name},
        "assessment": {"verdict": verdict},
        "evidence": {
            "request": _request_summary(specification),
            "rendered_input": {
                "text": rendered,
                "size_bytes": len(rendered.encode("utf-8")),
                "line_count": len(rendered.splitlines()),
            },
            "lint": {
                "status": lint_status,
                "summary": issue_counts,
                "issues": issues,
            },
        },
        "uncertainty": uncertainty,
        "next_actions": _next_actions(
            backend,
            lint_status,
            issues,
        ),
    }


def _validate_specification(
    program: str,
    specification: Mapping[str, Any],
) -> None:
    missing = [
        field for field in _REQUIRED_FIELDS if field not in specification
    ]
    if missing:
        _invalid(
            program,
            "input specification is missing required fields: "
            + ", ".join(missing),
        )

    atoms = specification["atoms"]
    if not isinstance(atoms, list) or not 1 <= len(atoms) <= MAX_DRAFT_ATOMS:
        _invalid(
            program,
            f"atoms must contain between 1 and {MAX_DRAFT_ATOMS} entries",
        )
    for index, atom in enumerate(atoms):
        if not isinstance(atom, Mapping):
            _invalid(program, f"atoms[{index}] must be an object")
        element = atom.get("element")
        if not isinstance(element, str) or not element.strip():
            _invalid(program, f"atoms[{index}].element must be a non-empty string")
        for axis in ("x", "y", "z"):
            value = atom.get(axis)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                _invalid(program, f"atoms[{index}].{axis} must be finite")

    charge = specification["charge"]
    if isinstance(charge, bool) or not isinstance(charge, int):
        _invalid(program, "charge must be an integer")
    multiplicity = specification["multiplicity"]
    if (
        isinstance(multiplicity, bool)
        or not isinstance(multiplicity, int)
        or multiplicity < 1
    ):
        _invalid(program, "multiplicity must be a positive integer")

    method = specification["method"]
    if not isinstance(method, str) or not method.strip():
        _invalid(program, "method must be a non-empty string")
    method_key = method.strip().casefold()
    supported_methods = _METHODS_BY_PROGRAM.get(program)
    if supported_methods is not None and method_key not in supported_methods:
        _invalid(
            program,
            f"{program} guided drafting supports methods: "
            + ", ".join(sorted(supported_methods)),
        )
    functional = specification.get("functional")
    is_dft = method_key in {"dft", "ksdft"}
    if is_dft and (
        not isinstance(functional, str) or not functional.strip()
    ):
        _invalid(program, "DFT input drafting requires a functional")
    if functional is not None and not is_dft:
        _invalid(program, "functional is accepted only for DFT methods")
    basis = specification["basis"]
    if isinstance(basis, str):
        if not basis.strip():
            _invalid(program, "basis must be a non-empty string")
    elif not isinstance(basis, Mapping) or not basis or any(
        not isinstance(element, str)
        or not element.strip()
        or not isinstance(name, str)
        or not name.strip()
        for element, name in basis.items()
    ):
        _invalid(
            program,
            "basis must be a non-empty string or element-to-basis mapping",
        )

    task = specification["task"]
    if task not in _TASKS:
        _invalid(
            program,
            "task must be one of: " + ", ".join(sorted(_TASKS)),
        )
    supported_tasks = _TASKS_BY_PROGRAM.get(program)
    if supported_tasks is not None and task not in supported_tasks:
        _invalid(
            program,
            f"{program} guided drafting supports tasks: "
            + ", ".join(sorted(supported_tasks)),
        )
    units = specification.get("geometry_units", "angstrom")
    if units not in {"angstrom", "bohr"}:
        _invalid(program, "geometry_units must be 'angstrom' or 'bohr'")
    options = specification.get("program_options", {})
    if not isinstance(options, Mapping):
        _invalid(program, "program_options must be an object")
    if "geometry_path" in options:
        _invalid(
            program,
            "guided input drafting requires inline atoms; geometry_path is "
            "not accepted",
        )


def _invalid(program: str, message: str) -> None:
    raise InputDraftError(
        "invalid_input_specification",
        message,
        program=program,
    )


def _lint_rendered_input(
    backend: ProgramBackend,
    rendered: str,
) -> tuple[list[dict[str, Any]], str, list[dict[str, str]]]:
    if not backend.supports(ProgramCapability.INPUT_LINT):
        return [], "unsupported", [{
            "code": "input_linter_unavailable",
            "message": f"{backend.name} has no declared input linter.",
            "impact": "The rendered input requires manual review.",
        }]
    assert backend.inputs is not None
    try:
        raw_issues = backend.inputs.lint_input(rendered)
    except Exception as exc:
        return [], "failed", [{
            "code": "input_lint_failed",
            "message": (
                f"{backend.name} input linting failed with "
                f"{type(exc).__name__}: {exc}"
            ),
            "impact": "The rendered input requires manual review.",
        }]
    if not isinstance(raw_issues, (list, tuple)):
        return [], "failed", [{
            "code": "invalid_input_linter_result",
            "message": (
                f"{backend.name} input linter returned "
                f"{type(raw_issues).__name__}, not a list."
            ),
            "impact": "The rendered input requires manual review.",
        }]
    return [
        dict(issue)
        for issue in raw_issues
        if isinstance(issue, Mapping)
    ], "completed", []


def _issue_counts(issues: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "errors": sum(issue.get("level") == "error" for issue in issues),
        "warnings": sum(
            issue.get("level") == "warning" for issue in issues
        ),
        "info": sum(issue.get("level") == "info" for issue in issues),
    }


def _verdict(
    lint_status: str,
    issue_counts: Mapping[str, int],
) -> dict[str, Any]:
    if lint_status == "completed" and issue_counts["errors"]:
        return {
            "label": "draft_has_errors",
            "confidence": 0.9,
            "reasons": [
                f"The configured linter found {issue_counts['errors']} error(s)."
            ],
        }
    if lint_status == "completed" and issue_counts["warnings"]:
        return {
            "label": "draft_requires_review",
            "confidence": 0.8,
            "reasons": [
                "The configured linter found "
                f"{issue_counts['warnings']} warning(s)."
            ],
        }
    if lint_status == "completed":
        return {
            "label": "draft_ready",
            "confidence": 0.8,
            "reasons": [
                "The input was rendered and the configured linter found no "
                "errors or warnings."
            ],
        }
    return {
        "label": "draft_unchecked",
        "confidence": 0.35,
        "reasons": [
            "The input was rendered, but the configured linter did not "
            "complete."
        ],
    }


def _request_summary(specification: Mapping[str, Any]) -> dict[str, Any]:
    atoms = specification.get("atoms")
    options = specification.get("program_options")
    return {
        "atom_count": len(atoms) if isinstance(atoms, list) else None,
        "charge": specification.get("charge"),
        "multiplicity": specification.get("multiplicity"),
        "method": specification.get("method"),
        "basis": specification.get("basis"),
        "task": specification.get("task"),
        "functional": specification.get("functional"),
        "geometry_units": specification.get("geometry_units", "angstrom"),
        "program_option_keys": (
            sorted(options) if isinstance(options, Mapping) else []
        ),
    }


def _next_actions(
    backend: ProgramBackend,
    lint_status: str,
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actions = [
        {
            "action": "revise_specification",
            "reason": str(issue.get("message") or "Input lint issue."),
            "suggested_fix": issue.get("suggested_fix"),
            "priority": 1 if issue.get("level") == "error" else 2,
        }
        for issue in issues
        if issue.get("level") in {"error", "warning"}
    ]
    if actions:
        return sorted(actions, key=lambda action: action["priority"])
    if lint_status != "completed":
        return [{
            "action": "manual_scientific_review",
            "reason": (
                f"The {backend.name} draft was not checked by its linter."
            ),
            "priority": 1,
        }]
    return [{
        "action": "save_input",
        "reason": "Save the rendered text to a new input file after review.",
        "priority": 1,
    }]


__all__ = [
    "INPUT_DRAFT_SCHEMA",
    "MAX_DRAFT_ATOMS",
    "InputDraftError",
    "draft_input",
]

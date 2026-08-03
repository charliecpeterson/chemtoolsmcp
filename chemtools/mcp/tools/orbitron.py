"""MCP access to bounded Orbitron inspection, analysis, and rendering."""

from __future__ import annotations

from typing import Any, Callable, Literal

from chemtools.core import registry
from chemtools.core.artifact_classification import classify_artifact
from chemtools.core.artifacts import ProducerIdentity
from chemtools.core.program import ProgramBackend
from chemtools.integrations.orbitron import (
    OrbitronClient,
    OrbitronCommandError,
    OrbitronProtocolError,
    OrbitronRender,
    OrbitronResponse,
    OrbitronUnavailableError,
)
from chemtools.mcp.decorator import _tool
from chemtools.mcp.server import ImageToolResult


_ARGUMENTS = frozenset({"path"})
_ENVELOPE_FIELDS = frozenset({"schema", "producer", "warnings"})
_MAX_ERROR_CHARACTERS = 4_096
_COMPANION_PROGRAMS = {
    "movecs": "nwchem",
    "hessian": "nwchem",
    "civecs": "nwchem",
    "dirac_checkpoint": "dirac",
}
_INSPECTION_SCHEMA = "chemtools.orbitron-inspection/1"
_GEOMETRY_ANALYSIS_SCHEMA = "chemtools.orbitron-geometry-analysis/3"
_ORBITAL_ANALYSIS_SCHEMA = "chemtools.orbitron-orbital-analysis/2"
_POPULATION_ANALYSIS_SCHEMA = "chemtools.orbitron-population-analysis/2"
_VIBRATION_ANALYSIS_SCHEMA = "chemtools.orbitron-vibration-analysis/3"
_RENDER_SCHEMA = "chemtools.orbitron-render/1"
_Operation = Literal[
    "inspect",
    "analyze_geometry",
    "analyze_orbitals",
    "analyze_populations",
    "analyze_vibrations",
]


@_tool("inspect_with_orbitron", program="generic")
def _handle_inspect_with_orbitron(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return _run_read_only_operation(
        arguments,
        operation="inspect",
        schema_version=_INSPECTION_SCHEMA,
        success_builder=_inspection_success,
    )


@_tool("analyze_geometry_with_orbitron", program="generic")
def _handle_analyze_geometry_with_orbitron(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return _run_read_only_operation(
        arguments,
        operation="analyze_geometry",
        schema_version=_GEOMETRY_ANALYSIS_SCHEMA,
        success_builder=_geometry_analysis_success,
    )


@_tool("analyze_orbitals_with_orbitron", program="generic")
def _handle_analyze_orbitals_with_orbitron(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return _run_read_only_operation(
        arguments,
        operation="analyze_orbitals",
        schema_version=_ORBITAL_ANALYSIS_SCHEMA,
        success_builder=_orbital_analysis_success,
    )


@_tool("analyze_populations_with_orbitron", program="generic")
def _handle_analyze_populations_with_orbitron(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return _run_read_only_operation(
        arguments,
        operation="analyze_populations",
        schema_version=_POPULATION_ANALYSIS_SCHEMA,
        success_builder=_population_analysis_success,
    )


@_tool("analyze_vibrations_with_orbitron", program="generic")
def _handle_analyze_vibrations_with_orbitron(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return _run_read_only_operation(
        arguments,
        operation="analyze_vibrations",
        schema_version=_VIBRATION_ANALYSIS_SCHEMA,
        success_builder=_vibration_analysis_success,
    )


@_tool("render_with_orbitron", program="generic")
def _handle_render_with_orbitron(
    arguments: dict[str, Any],
) -> dict[str, Any] | ImageToolResult:
    unknown = sorted(set(arguments) - _ARGUMENTS)
    if unknown:
        raise ValueError(f"unknown Orbitron render arguments: {unknown}")
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")

    try:
        response = OrbitronClient().render(path)
    except OrbitronUnavailableError as error:
        return _render_error("unavailable", "orbitron_unavailable", error)
    except OrbitronProtocolError as error:
        return _render_error("incompatible", "orbitron_protocol_error", error)
    except OrbitronCommandError as error:
        result = _render_error("tool_refused", "orbitron_command_error", error)
        result["returncode"] = error.returncode
        result["stderr"] = _bounded_text(error.stderr)
        return result
    return _render_success(response)


def _run_read_only_operation(
    arguments: dict[str, Any],
    *,
    operation: _Operation,
    schema_version: str,
    success_builder: Callable[[OrbitronResponse], dict[str, Any]],
) -> dict[str, Any]:
    unknown = sorted(set(arguments) - _ARGUMENTS)
    if unknown:
        raise ValueError(f"unknown Orbitron {operation} arguments: {unknown}")
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")

    try:
        client = OrbitronClient()
        if operation == "inspect":
            response = client.inspect(path)
        elif operation == "analyze_geometry":
            response = client.analyze_geometry(path)
        elif operation == "analyze_orbitals":
            response = client.analyze_orbitals(path)
        elif operation == "analyze_populations":
            response = client.analyze_populations(path)
        else:
            response = client.analyze_vibrations(path)
    except OrbitronUnavailableError as error:
        return {
            "schema_version": schema_version,
            "status": "unavailable",
            "error": "orbitron_unavailable",
            "message": str(error),
        }
    except OrbitronProtocolError as error:
        return {
            "schema_version": schema_version,
            "status": "incompatible",
            "error": "orbitron_protocol_error",
            "message": str(error),
        }
    except OrbitronCommandError as error:
        return {
            "schema_version": schema_version,
            "status": "tool_refused",
            "error": "orbitron_command_error",
            "message": str(error),
            "returncode": error.returncode,
            "stderr": _bounded_text(error.stderr),
        }
    return success_builder(response)


def _render_error(
    status: str,
    code: str,
    error: OrbitronUnavailableError | OrbitronProtocolError | OrbitronCommandError,
) -> dict[str, Any]:
    return {
        "schema_version": _RENDER_SCHEMA,
        "status": status,
        "error": code,
        "message": str(error),
    }


def _render_success(response: OrbitronRender) -> ImageToolResult:
    payload = {
        "schema_version": _RENDER_SCHEMA,
        "status": "ok",
        "operation": "render",
        "source": response.source,
        "producer": {
            "name": "orbitron",
            "version": response.version.version,
            "commit": response.version.commit,
        },
        "image": {
            "mime_type": "image/png",
            "width": response.width,
            "height": response.height,
            "size_bytes": len(response.image),
        },
    }
    if response.stderr:
        payload["stderr"] = _bounded_text(response.stderr)
    return ImageToolResult(payload=payload, image=response.image)


def _inspection_success(response: OrbitronResponse) -> dict[str, Any]:
    payload = _evidence_payload(response)

    result = {
        "schema_version": _INSPECTION_SCHEMA,
        "status": "ok",
        "operation": response.operation,
        "source": response.source,
        "orbitron_schema": response.schema,
        "producer": response.producer,
        "warnings": list(response.warnings),
        "evidence": payload,
        "canonical_mapping": {
            "producer": _producer_identity(response),
            "artifact": _map_artifact(response.source, payload),
            "scientific_system": _map_scientific_system(payload),
        },
    }
    if response.stderr:
        result["stderr"] = _bounded_text(response.stderr)
    return result


def _geometry_analysis_success(
    response: OrbitronResponse,
) -> dict[str, Any]:
    payload = _evidence_payload(response)
    uncertainty = []
    if payload["geometry_role"] == "last_attempted":
        uncertainty.append(
            {
                "code": "orbitron_geometry_not_converged",
                "message": payload["geometry_source"],
                "impact": (
                    "Use this geometry to diagnose the failed run, not as a "
                    "converged structure."
                ),
            }
        )
    result = {
        "schema_version": _GEOMETRY_ANALYSIS_SCHEMA,
        "status": "ok",
        "operation": response.operation,
        "source": response.source,
        "orbitron_schema": response.schema,
        "producer": response.producer,
        "warnings": list(response.warnings),
        "uncertainty": uncertainty,
        "evidence": payload,
        "canonical_mapping": {
            "producer": _producer_identity(response),
            "scientific_system": {
                "status": "insufficient_evidence",
                "reason": (
                    f"{response.schema} reports geometry summaries "
                    "but not atom identities with coordinates."
                ),
            },
        },
    }
    if response.stderr:
        result["stderr"] = _bounded_text(response.stderr)
    return result


def _orbital_analysis_success(
    response: OrbitronResponse,
) -> dict[str, Any]:
    payload = _evidence_payload(response)
    result = {
        "schema_version": _ORBITAL_ANALYSIS_SCHEMA,
        "status": "ok",
        "operation": response.operation,
        "source": response.source,
        "orbitron_schema": response.schema,
        "producer": response.producer,
        "parameters": {"frontier_count": 3},
        "warnings": list(response.warnings),
        "uncertainty": [],
        "evidence": payload,
        "canonical_mapping": {
            "producer": _producer_identity(response),
            "electronic_structure": {
                "status": "not_mapped",
                "reason": (
                    "Chemtools does not yet define a canonical molecular-"
                    "orbital summary model."
                ),
            },
        },
    }
    if response.stderr:
        result["stderr"] = _bounded_text(response.stderr)
    return result


def _population_analysis_success(
    response: OrbitronResponse,
) -> dict[str, Any]:
    payload = _evidence_payload(response)
    result = {
        "schema_version": _POPULATION_ANALYSIS_SCHEMA,
        "status": "ok",
        "operation": response.operation,
        "source": response.source,
        "orbitron_schema": response.schema,
        "producer": response.producer,
        "parameters": {"top_count": 8},
        "warnings": list(response.warnings),
        "uncertainty": _population_uncertainty(payload),
        "evidence": payload,
        "canonical_mapping": {
            "producer": _producer_identity(response),
            "electronic_structure": {
                "status": "not_mapped",
                "reason": (
                    "Chemtools does not yet define a canonical atomic-"
                    "population summary model."
                ),
            },
        },
    }
    if response.stderr:
        result["stderr"] = _bounded_text(response.stderr)
    return result


def _population_uncertainty(payload: dict[str, Any]) -> list[dict[str, Any]]:
    uncertainty = []
    for method in payload["methods"]:
        if method["expected_total_charge"] is None:
            uncertainty.append(
                {
                    "code": "orbitron_population_expected_charge_unknown",
                    "method": method["method"],
                    "message": (
                        "The source does not establish the expected total "
                        "charge for this population analysis."
                    ),
                    "impact": (
                        "The partial-charge sum cannot be checked against the "
                        "charge of the calculated system."
                    ),
                }
            )
        for warning in method["warnings"]:
            uncertainty.append(
                {
                    "code": "orbitron_population_method_warning",
                    "method": method["method"],
                    "message": warning,
                    "impact": (
                        "Review the population data before using this method's "
                        "atomic charges."
                    ),
                }
            )
    return uncertainty


def _vibration_analysis_success(
    response: OrbitronResponse,
) -> dict[str, Any]:
    payload = _evidence_payload(response)
    uncertainty = []
    if payload["geometry_role"] == "last_attempted":
        uncertainty.append(
            {
                "code": "orbitron_vibration_geometry_not_converged",
                "message": (
                    "Orbitron identified the frequency geometry as the last "
                    f"attempted structure: {payload['geometry_source']}"
                ),
                "impact": (
                    "Do not interpret these frequencies as stationary-point "
                    "modes until the geometry is converged."
                ),
            }
        )
    thermochemistry = payload["thermochemistry"]
    if thermochemistry is not None and thermochemistry["pressure_atm"] is None:
        uncertainty.append(
            {
                "code": "orbitron_thermochemistry_standard_state_unknown",
                "message": (
                    "The source does not report the pressure standard state "
                    "for its thermochemistry."
                ),
                "impact": (
                    "Do not compare its Gibbs correction with a value using a "
                    "different gas- or solution-phase standard state."
                ),
            }
        )
    sampled_imaginary = [
        mode for mode in payload["modes"] if mode["frequency"] < 0
    ]
    missing_imaginary_displacement = any(
        not mode["has_displacement"] for mode in sampled_imaginary
    )
    unsampled_imaginary = payload["imaginary_count"] > len(sampled_imaginary)
    if missing_imaginary_displacement or (
        unsampled_imaginary and not payload["has_displacements"]
    ):
        uncertainty.append(
            {
                "code": "orbitron_vibration_displacements_unavailable",
                "message": (
                    "The source reports imaginary frequencies without normal-"
                    "mode displacement vectors."
                ),
                "impact": (
                    "Frequency sign alone cannot establish whether the mode "
                    "follows the intended reaction coordinate."
                ),
            }
        )

    result = {
        "schema_version": _VIBRATION_ANALYSIS_SCHEMA,
        "status": "ok",
        "operation": response.operation,
        "source": response.source,
        "orbitron_schema": response.schema,
        "producer": response.producer,
        "parameters": {"mode_set": "raw", "top_count": 10},
        "warnings": list(response.warnings),
        "uncertainty": uncertainty,
        "evidence": payload,
        "canonical_mapping": {
            "producer": _producer_identity(response),
            "vibrations": {
                "status": "not_mapped",
                "reason": (
                    "Chemtools does not yet define a program-neutral "
                    "vibration-analysis summary model."
                ),
            },
        },
    }
    if response.stderr:
        result["stderr"] = _bounded_text(response.stderr)
    return result


def _evidence_payload(response: OrbitronResponse) -> dict[str, Any]:
    return {
        key: value
        for key, value in response.payload.items()
        if key not in _ENVELOPE_FIELDS
    }


def _producer_identity(response: OrbitronResponse) -> dict[str, Any]:
    return ProducerIdentity(
        producer_type="external_tool",
        name=response.producer["name"],
        version=response.producer.get("version"),
        commit=response.producer.get("commit"),
    ).to_dict()


def _bounded_text(value: str) -> str:
    if len(value) <= _MAX_ERROR_CHARACTERS:
        return value
    return value[:_MAX_ERROR_CHARACTERS] + "\n[truncated by Chemtools]"


def _map_artifact(source: str, payload: dict[str, Any]) -> dict[str, Any]:
    subject = payload.get("subject")
    if not isinstance(subject, str) or not subject:
        return {
            "status": "unsupported_subject",
            "orbitron_subject": subject,
            "reason": "Orbitron did not report a supported artifact subject.",
        }

    if subject == "output":
        program = payload.get("program")
        if not isinstance(program, str) or not program:
            return {
                "status": "program_unresolved",
                "orbitron_subject": subject,
                "reason": "Orbitron did not identify the output program.",
            }
    elif subject in _COMPANION_PROGRAMS:
        program = _COMPANION_PROGRAMS[subject]
    else:
        return {
            "status": "unsupported_subject",
            "orbitron_subject": subject,
            "reason": (
                "Chemtools has no canonical artifact owner for this "
                "Orbitron subject."
            ),
        }

    if not registry.has(program):
        return {
            "status": "unsupported_program",
            "orbitron_subject": subject,
            "program": program,
            "reason": "Chemtools has no registered backend for this program.",
        }

    backend = registry.get(program)
    if not isinstance(backend, ProgramBackend):
        return {
            "status": "unsupported_backend_contract",
            "orbitron_subject": subject,
            "program": program,
            "reason": "The registered program has no artifact-kind contract.",
        }

    classification = classify_artifact(backend, source).to_dict()
    classification["orbitron_subject"] = subject
    return classification


def _map_scientific_system(payload: dict[str, Any]) -> dict[str, Any]:
    subject = payload.get("subject")
    if subject != "output":
        return {
            "status": "not_applicable",
            "reason": (
                "Orbitron inspected a companion artifact rather than a "
                "complete scientific system."
            ),
        }

    detected = payload.get("detected")
    if detected == "unparsed":
        reason = payload.get("parse_error")
        return {
            "status": "unavailable",
            "detected": detected,
            "reason": (
                reason
                if isinstance(reason, str) and reason
                else "Orbitron could not parse a geometry from the output."
            ),
        }
    if detected in {"scene", "trajectory"}:
        return {
            "status": "insufficient_evidence",
            "detected": detected,
            "reason": (
                "orbitron.inspect/2 reports atom and bond counts, but not "
                "atom coordinates or a complete periodic-system specification."
            ),
        }
    return {
        "status": "unsupported_evidence",
        "detected": detected,
        "reason": "Orbitron reported an unrecognized output detection state.",
    }


def orbitron_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "render_with_orbitron",
            "description": (
                "Render one local chemistry file through Orbitron's fixed "
                "headless PNG operation. Chemtools writes only to an "
                "ephemeral sibling directory, fixes the image at 1024 by "
                "768 pixels, validates the PNG and its size, then returns "
                "the image directly in the MCP result. Source, output, "
                "camera, appearance, diagram, and dimension overrides are "
                "not accepted. Orbitron is optional; unavailable, "
                "incompatible, and refused outcomes are reported explicitly."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Path to one local file supported by Orbitron. "
                            "Directories, remote targets, output paths, and "
                            "render arguments are not accepted."
                        ),
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "inspect_with_orbitron",
            "description": (
                "Inspect one local chemistry file through the configured "
                "Orbitron CLI. Calls only Orbitron's read-only inspect JSON "
                "operation and verifies its schema, producer version, commit, "
                "and structured warnings. Returns parsed structural, "
                "trajectory, task, orbital, vibrational, or periodic evidence "
                "plus canonical producer and artifact classification where "
                "the current Chemtools backend supports it. Geometry counts "
                "are not promoted into a scientific-system model. "
                "Orbitron is optional; unavailable, incompatible, and refused "
                "outcomes are reported explicitly."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Path to one local file supported by Orbitron. "
                            "Directories and remote targets are not accepted."
                        ),
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "analyze_geometry_with_orbitron",
            "description": (
                "Analyze one local chemistry file through Orbitron's fixed, "
                "read-only analyze geometry JSON operation. Returns validated "
                "atom and bond counts, element and coordination summaries, "
                "bond-length statistics, bounds, and unit-cell evidence. "
                "Contradictory counts or invalid numeric fields are reported "
                "as an incompatible Orbitron response. The angstrom distance "
                "unit is validated. The summary does not contain atom "
                "coordinates and is not promoted into a Chemtools scientific-"
                "system model."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Path to one local file supported by Orbitron. "
                            "Directories, remote targets, output paths, and "
                            "render arguments are not accepted."
                        ),
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "analyze_orbitals_with_orbitron",
            "description": (
                "Analyze molecular-orbital frontier evidence through "
                "Orbitron's fixed, read-only analyze orbitals JSON operation. "
                "Uses a fixed frontier count of three and validates orbital "
                "counts, finite energies, Hartree-to-eV conversions, the "
                "HOMO-LUMO gap, frontier membership, occupancy policy, and "
                "restricted or alpha/beta spin-channel partitions."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Path to one local file with molecular-orbital "
                            "data supported by Orbitron. Commands, remote "
                            "targets, output paths, and frontier overrides are "
                            "not accepted."
                        ),
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "analyze_populations_with_orbitron",
            "description": (
                "Analyze atomic population charges through Orbitron's fixed, "
                "read-only analyze populations JSON operation. Uses a fixed "
                "top count of eight and validates atom counts, finite charges, "
                "derived totals and extrema, charge ordering, per-atom maps, "
                "top-charge membership, expected system charge, and charge "
                "residuals. A missing expected charge remains explicit."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Path to one local file with atomic population "
                            "data supported by Orbitron. Commands, remote "
                            "targets, output paths, and top-count overrides "
                            "are not accepted."
                        ),
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "analyze_vibrations_with_orbitron",
            "description": (
                "Analyze vibrational frequencies through Orbitron's fixed, "
                "read-only raw-mode JSON operation. Uses a fixed top count of "
                "ten and validates frequency counts and statistics, sorted "
                "mode evidence, units, scaling policy, displacement "
                "counts and per-mode availability, and unit-labelled "
                "thermochemistry. Missing thermochemistry standard-state or "
                "imaginary-mode displacement evidence remains explicit."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Path to one local file with vibrational data "
                            "supported by Orbitron. Commands, remote targets, "
                            "output paths, projected-mode requests, and top-"
                            "count overrides are not accepted."
                        ),
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    ]


__all__ = ["orbitron_tool_definitions"]

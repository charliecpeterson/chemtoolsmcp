"""Public descriptions and input schemas for the guided MCP tools.

`guided.py` binds every definition to one `_handle_<tool name>` implementation
before the catalog can expose these contracts.
"""

from __future__ import annotations

from typing import Any

from chemtools.application.calculation_planning import (
    CALCULATION_PLAN_SCHEMA,
    MAX_PLAN_ELEMENTS,
    MAX_PLAN_STAGES,
)
from chemtools.application.input_drafting import INPUT_DRAFT_SCHEMA, MAX_DRAFT_ATOMS
from chemtools.application.input_review import INPUT_REVIEW_SCHEMA
from chemtools.application.recovery_planning import RECOVERY_PLAN_SCHEMA
from chemtools.application.run_comparison import RUN_COMPARISON_SCHEMA
from chemtools.application.run_inspection import RUN_INSPECTION_SCHEMA
from chemtools.application.run_launching import LAUNCH_RUN_SCHEMA
from chemtools.application.run_monitoring import MONITOR_RUN_SCHEMA
from chemtools.mcp.tools._output_schemas import (
    ARRAY,
    OBJECT,
    STRING,
    versioned_output_schema,
)


def _read_only_annotations(title: str) -> dict[str, Any]:
    return {
        "title": title,
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }


def _monitor_annotations() -> dict[str, Any]:
    return {
        "title": "Monitor an owned calculation",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }


def guided_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "review_input",
            "annotations": _read_only_annotations("Review chemistry input"),
            "outputSchema": versioned_output_schema(
                INPUT_REVIEW_SCHEMA,
                {
                    "program": OBJECT,
                    "source": OBJECT,
                    "assessment": OBJECT,
                    "evidence": OBJECT,
                    "uncertainty": ARRAY,
                    "next_actions": ARRAY,
                },
            ),
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
            "annotations": _read_only_annotations("Inspect chemistry run"),
            "outputSchema": versioned_output_schema(
                RUN_INSPECTION_SCHEMA,
                {
                    "program": OBJECT,
                    "source": OBJECT,
                    "assessment": OBJECT,
                    "evidence": OBJECT,
                    "uncertainty": ARRAY,
                    "next_actions": ARRAY,
                },
            ),
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
        {
            "name": "compare_runs",
            "annotations": _read_only_annotations("Compare chemistry runs"),
            "outputSchema": versioned_output_schema(
                RUN_COMPARISON_SCHEMA,
                {
                    "program": STRING,
                    "sources": OBJECT,
                    "assessment": OBJECT,
                    "evidence": OBJECT,
                    "uncertainty": ARRAY,
                    "next_actions": ARRAY,
                },
            ),
            "description": (
                "Compare two completed or partial outputs from the same "
                "chemistry program. Reports parsed energy arithmetic "
                "separately from checks of task, method, charge, composition, "
                "functional, basis, and multiplicity. A lower energy is "
                "reported conditionally when required settings or geometries "
                "could not be checked; it is never presented as a ground-state "
                "assignment by itself. Supplying the corresponding input "
                "files enables stronger comparability checks."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_output_file": {
                        "type": "string",
                        "description": "Path to the reference run output.",
                    },
                    "candidate_output_file": {
                        "type": "string",
                        "description": "Path to the candidate run output.",
                    },
                    "reference_input_file": {
                        "type": "string",
                        "description": "Optional input corresponding to the reference output.",
                    },
                    "candidate_input_file": {
                        "type": "string",
                        "description": "Optional input corresponding to the candidate output.",
                    },
                    "program": {
                        "type": "string",
                        "enum": ["nwchem", "molcas", "dirac", "grasp", "qe", "qmcpack"],
                        "description": (
                            "Optional shared program override. Without it, "
                            "both outputs are detected independently."
                        ),
                    },
                },
                "required": [
                    "reference_output_file",
                    "candidate_output_file",
                ],
                "additionalProperties": False,
            },
        },
        {
            "name": "plan_recovery",
            "annotations": _read_only_annotations("Plan run recovery"),
            "outputSchema": versioned_output_schema(
                RECOVERY_PLAN_SCHEMA,
                {
                    "program": OBJECT,
                    "source": OBJECT,
                    "target": OBJECT,
                    "assessment": OBJECT,
                    "evidence": OBJECT,
                    "uncertainty": ARRAY,
                    "next_actions": ARRAY,
                },
            ),
            "description": (
                "Build a read-only recovery plan for one NWChem run. The "
                "matching input enables candidate retry drafts. Explicit "
                "target charge, multiplicity, metal elements, and SOMO count "
                "keep electronic-state recovery separate from ordinary SCF "
                "repair. If the input charge or multiplicity differs from the "
                "target, the tool requires a fresh state-specific input and "
                "will not propose an orbital swap. Candidate text is returned "
                "for review; no files are written or calculations launched."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {
                        "type": "string",
                        "description": "Path to one NWChem output file.",
                    },
                    "input_file": {
                        "type": "string",
                        "description": (
                            "Optional matching NWChem input. Required when a "
                            "candidate recovery input should be prepared."
                        ),
                    },
                    "program": {
                        "type": "string",
                        "enum": ["nwchem"],
                        "description": (
                            "Optional NWChem override. Without it, Chemtools "
                            "detects the output program from content."
                        ),
                    },
                    "expected_charge": {"type": "integer"},
                    "expected_multiplicity": {
                        "type": "integer",
                        "minimum": 1,
                    },
                    "expected_metal_elements": {
                        "type": "array",
                        "maxItems": 32,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "expected_somo_count": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "Optional explicit SOMO target. When omitted, it "
                            "is derived as multiplicity minus one if a target "
                            "multiplicity was supplied."
                        ),
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "plan_calculation",
            "annotations": _read_only_annotations("Plan calculation strategy"),
            "outputSchema": versioned_output_schema(
                CALCULATION_PLAN_SCHEMA,
                {
                    "program": OBJECT,
                    "request": OBJECT,
                    "assessment": OBJECT,
                    "evidence": OBJECT,
                    "uncertainty": ARRAY,
                    "next_actions": ARRAY,
                },
            ),
            "description": (
                "Plan calculation stages and expose unresolved scientific "
                "choices before any program input is rendered. Returns a "
                "normalized verdict, an ordered dependency plan, required "
                "decisions, assumptions, and the next planning or drafting "
                "action. It does not read or write files and cannot launch a "
                "calculation. NWChem currently supplies the planning provider; "
                "other registered programs return an explicit unsupported "
                "capability result."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "program": {
                        "type": "string",
                        "enum": [
                            "nwchem",
                            "molcas",
                            "dirac",
                            "grasp",
                            "qe",
                            "qmcpack",
                        ],
                    },
                    "system": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Chemical formula or concise system label.",
                    },
                    "elements": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_PLAN_ELEMENTS,
                        "items": {"type": "string", "minLength": 1},
                        "description": (
                            "Distinct element symbols used to identify "
                            "element-dependent decisions."
                        ),
                    },
                    "charge": {"type": "integer"},
                    "multiplicity": {"type": "integer", "minimum": 1},
                    "stages": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_PLAN_STAGES,
                        "items": {
                            "type": "string",
                            "enum": [
                                "energy",
                                "optimize",
                                "frequency",
                            ],
                        },
                        "description": "Ordered scientific stages to plan.",
                    },
                    "method": {"type": "string", "minLength": 1},
                    "functional": {"type": "string", "minLength": 1},
                    "basis": {
                        "oneOf": [
                            {"type": "string", "minLength": 1},
                            {
                                "type": "object",
                                "minProperties": 1,
                                "additionalProperties": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                            },
                        ],
                    },
                    "ecp": {
                        "oneOf": [
                            {"type": "string", "minLength": 1},
                            {
                                "type": "object",
                                "minProperties": 1,
                                "additionalProperties": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                            },
                        ],
                    },
                    "relativistic": {"type": "string", "minLength": 1},
                    "geometry_source": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Description of the accepted coordinate source and "
                            "units; no file is read."
                        ),
                    },
                    "solvent": {"type": "string", "minLength": 1},
                    "state_strategy": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "How the requested electronic state will be "
                            "initialized and checked across stages."
                        ),
                    },
                },
                "required": [
                    "program",
                    "system",
                    "elements",
                    "charge",
                    "multiplicity",
                    "stages",
                ],
                "additionalProperties": False,
            },
        },
        {
            "name": "launch_run",
            "description": (
                "Prepare an exact launch for a reviewed chemistry input and "
                "require a second, explicitly approved call before starting "
                "anything. The first call returns a SHA-256 approval token "
                "bound to the input contents, configured target, resources, "
                "command, configured environment, output paths, and scheduler "
                "script. Environment values remain private; the plan returns "
                "their keys and a stable fingerprint. Show the plan to the "
                "user. Pass the token back only after explicit approval. Any "
                "changed input or plan invalidates the token. Existing artifacts "
                "block launch rather than being overwritten or silently archived. "
                "Select a schema-2 named target, use the configured default, "
                "or provide a version 1 profile during migration. NWChem and "
                "Quantum ESPRESSO supply guided launch providers."
            ),
            "annotations": {
                "title": "Prepare or launch an approved calculation",
                "readOnlyHint": False,
                "destructiveHint": True,
                "idempotentHint": False,
                "openWorldHint": True,
            },
            "outputSchema": versioned_output_schema(
                LAUNCH_RUN_SCHEMA,
                {
                    "status": STRING,
                    "program": OBJECT,
                    "input": OBJECT,
                    "assessment": OBJECT,
                    "evidence": OBJECT,
                    "approval": OBJECT,
                    "uncertainty": ARRAY,
                    "next_actions": ARRAY,
                },
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "program": {
                        "type": "string",
                        "enum": [
                            "nwchem",
                            "molcas",
                            "dirac",
                            "grasp",
                            "qe",
                            "qmcpack",
                        ],
                    },
                    "input_file": {
                        "type": "string",
                        "description": "Path to the reviewed chemistry input.",
                    },
                    "profile": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Configured runner-profile name.",
                    },
                    "target": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Configured schema-2 target name. Omit to use "
                            "the server's default target."
                        ),
                    },
                    "profiles_path": {
                        "type": "string",
                        "description": (
                            "Optional explicit runner-profile YAML or JSON path."
                        ),
                    },
                    "job_name": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Optional output and scheduler job stem.",
                    },
                    "resources": {
                        "type": "object",
                        "properties": {
                            "nodes": {"type": "integer", "minimum": 1},
                            "mpi_ranks": {"type": "integer", "minimum": 1},
                            "omp_threads": {"type": "integer", "minimum": 1},
                            "memory_mb_per_node": {
                                "type": "integer",
                                "minimum": 1,
                            },
                            "walltime": {
                                "type": "string",
                                "pattern": "^\\d+:[0-5]\\d:[0-5]\\d$",
                            },
                            "partition": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "account": {
                                "type": "string",
                                "minLength": 1,
                            },
                        },
                        "additionalProperties": False,
                    },
                    "approval_token": {
                        "type": "string",
                        "pattern": "^sha256:[0-9a-f]{64}$",
                        "description": (
                            "Token from the immediately preceding preparation. "
                            "Provide it only after the user explicitly approves "
                            "the displayed plan."
                        ),
                    },
                },
                "required": ["program", "input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "monitor_run",
            "annotations": _monitor_annotations(),
            "outputSchema": versioned_output_schema(
                MONITOR_RUN_SCHEMA,
                {
                    "status": STRING,
                    "program": OBJECT,
                    "launch": OBJECT,
                    "assessment": OBJECT,
                    "evidence": OBJECT,
                    "uncertainty": ARRAY,
                    "next_actions": ARRAY,
                },
            ),
            "description": (
                "Refresh one calculation launched and owned by this Chemtools "
                "server process. The launch ID resolves the recorded program, "
                "target, process or Slurm job, and artifact paths; arbitrary "
                "PIDs, scheduler IDs, and output paths are not accepted. "
                "Returns current execution state, recorded artifact metadata, "
                "and scientific progress when the backend declares a progress "
                "inspector. It never submits, restarts, cancels, or edits a run."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "launch_id": {
                        "type": "string",
                        "pattern": (
                            "^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
                            "[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
                        ),
                        "description": (
                            "Launch ID returned by launch_run in this server "
                            "process."
                        ),
                    },
                },
                "required": ["launch_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_input",
            "annotations": _read_only_annotations("Draft chemistry input"),
            "outputSchema": versioned_output_schema(
                INPUT_DRAFT_SCHEMA,
                {
                    "program": OBJECT,
                    "assessment": OBJECT,
                    "evidence": OBJECT,
                    "uncertainty": ARRAY,
                    "next_actions": ARRAY,
                },
            ),
            "description": (
                "Draft one chemistry input from a program-neutral molecular "
                "specification. The selected backend renders the native input "
                "text and, when available, immediately checks that text with "
                "its declared linter. Returns the exact rendered input, lint "
                "evidence, uncertainty, and the next review or save action. "
                "This tool does not write a file or launch a calculation. "
                "NWChem and OpenMolcas currently declare this capability; "
                "program_options carries settings that have no common form."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "program": {
                        "type": "string",
                        "enum": ["nwchem", "molcas"],
                        "description": "Program whose native input should be rendered.",
                    },
                    "atoms": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_DRAFT_ATOMS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "element": {"type": "string", "minLength": 1},
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"},
                            },
                            "required": ["element", "x", "y", "z"],
                            "additionalProperties": False,
                        },
                        "description": "Inline molecular geometry.",
                    },
                    "charge": {"type": "integer"},
                    "multiplicity": {"type": "integer", "minimum": 1},
                    "method": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "NWChem guided drafting accepts DFT, HF, and SCF. "
                            "OpenMolcas also accepts its CASSCF, RASSCF, and "
                            "CASPT2-family methods."
                        ),
                    },
                    "basis": {
                        "oneOf": [
                            {"type": "string", "minLength": 1},
                            {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                            },
                        ],
                    },
                    "task": {
                        "type": "string",
                        "enum": [
                            "energy",
                            "gradient",
                            "optimize",
                            "saddle",
                            "frequency",
                            "property",
                            "dynamics",
                        ],
                        "description": (
                            "OpenMolcas common drafting currently accepts "
                            "energy only."
                        ),
                    },
                    "title": {"type": "string"},
                    "geometry_units": {
                        "type": "string",
                        "enum": ["angstrom", "bohr"],
                        "default": "angstrom",
                    },
                    "functional": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Required for DFT and rejected for non-DFT methods."
                        ),
                    },
                    "program_options": {
                        "type": "object",
                        "description": (
                            "Program-specific settings that have no common "
                            "cross-program field. Unknown keys are rejected."
                        ),
                    },
                },
                "required": [
                    "program",
                    "atoms",
                    "charge",
                    "multiplicity",
                    "method",
                    "basis",
                    "task",
                ],
                "additionalProperties": False,
            },
        },
    ]


__all__ = ["guided_tool_definitions"]

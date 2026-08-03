"""MCP access to the optional companion scientific Python runtime probe."""

from __future__ import annotations

from typing import Any

from chemtools.application.pyscf_execution import (
    render_pyscf_single_point,
    run_pyscf_single_point,
)
from chemtools.core.pyscf_comparison import compare_pyscf_reference_calculation
from chemtools.integrations.nist_asd import (
    NIST_ASD_REFERENCE_SCHEMA,
    NistAsdError,
    fetch_nist_asd_reference,
)
from chemtools.integrations.science_runtime import (
    SCIENCE_RUNTIME_PYTHON_ENV,
    SCIENCE_RUNTIME_PROBE_SCHEMA,
    ScienceRuntimeClient,
    ScienceRuntimeCommandError,
    ScienceRuntimeProtocolError,
    ScienceRuntimeUnavailableError,
)
from chemtools.mcp.decorator import _tool, get_execution_service
from chemtools.science_runner import (
    BSE_RENDER_REQUEST_SCHEMA,
    OPENBABEL_CONVERSION_REQUEST_SCHEMA,
    ORBITRON_NBO_REQUEST_SCHEMA,
    ORBITRON_PERIODIC_REQUEST_SCHEMA,
    ORBITRON_STRUCTURE_IDENTITY_REQUEST_SCHEMA,
    RDKIT_PREFLIGHT_REQUEST_SCHEMA,
)


@_tool("inspect_science_runtime", program="generic")
def _handle_inspect_science_runtime(arguments: dict[str, Any]) -> dict[str, Any]:
    if arguments:
        raise ValueError("inspect_science_runtime does not accept arguments")
    try:
        probe = ScienceRuntimeClient().probe()
    except ScienceRuntimeUnavailableError as error:
        return _error("unavailable", "science_runtime_unavailable", error)
    except ScienceRuntimeProtocolError as error:
        return _error("incompatible", "science_runtime_protocol_error", error)
    except ScienceRuntimeCommandError as error:
        response = _error("tool_refused", "science_runtime_probe_error", error)
        response["returncode"] = error.returncode
        if error.stderr:
            response["stderr"] = error.stderr
        return response
    return {
        "schema_version": SCIENCE_RUNTIME_PROBE_SCHEMA,
        "status": "ok",
        "python": probe.python,
        "packages": probe.packages,
    }


@_tool("preflight_molecule_with_rdkit", program="generic")
def _handle_preflight_molecule_with_rdkit(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    request = {
        "schema_version": RDKIT_PREFLIGHT_REQUEST_SCHEMA,
        "format": arguments["format"],
        "source": arguments["source"],
    }
    try:
        return ScienceRuntimeClient().rdkit_preflight(request)
    except ScienceRuntimeUnavailableError as error:
        return _error("unavailable", "science_runtime_unavailable", error)
    except ScienceRuntimeProtocolError as error:
        return _error("incompatible", "science_runtime_protocol_error", error)
    except ScienceRuntimeCommandError as error:
        response = _error("tool_refused", "rdkit_preflight_error", error)
        response["returncode"] = error.returncode
        if error.stderr:
            response["stderr"] = error.stderr
        return response


@_tool("convert_molecule_with_openbabel", program="generic")
def _handle_convert_molecule_with_openbabel(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    request = {
        "schema_version": OPENBABEL_CONVERSION_REQUEST_SCHEMA,
        "format": arguments["format"],
        "source": arguments["source"],
        "output_format": arguments["output_format"],
    }
    try:
        return ScienceRuntimeClient().openbabel_convert(request)
    except ScienceRuntimeUnavailableError as error:
        return _error("unavailable", "science_runtime_unavailable", error)
    except ScienceRuntimeProtocolError as error:
        return _error("incompatible", "science_runtime_protocol_error", error)
    except ScienceRuntimeCommandError as error:
        response = _error("tool_refused", "openbabel_conversion_error", error)
        response["returncode"] = error.returncode
        if error.stderr:
            response["stderr"] = error.stderr
        return response


@_tool("render_basis_set_with_bse", program="generic")
def _handle_render_basis_set_with_bse(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    request = {
        "schema_version": BSE_RENDER_REQUEST_SCHEMA,
        "basis": arguments["basis"],
        "elements": arguments["elements"],
        "program_format": arguments["program_format"],
    }
    try:
        return ScienceRuntimeClient().bse_render(request)
    except ScienceRuntimeUnavailableError as error:
        return _error("unavailable", "science_runtime_unavailable", error)
    except ScienceRuntimeProtocolError as error:
        return _error("incompatible", "science_runtime_protocol_error", error)
    except ScienceRuntimeCommandError as error:
        response = _error("tool_refused", "bse_render_error", error)
        response["returncode"] = error.returncode
        if error.stderr:
            response["stderr"] = error.stderr
        return response


@_tool("fetch_nist_atomic_reference", program="generic")
def _handle_fetch_nist_atomic_reference(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    try:
        return fetch_nist_asd_reference(
            arguments["kind"],
            arguments["spectrum"],
            row_limit=arguments.get("row_limit", 200),
            refresh=arguments.get("refresh", False),
        )
    except (NistAsdError, ValueError) as error:
        return {
            "schema_version": NIST_ASD_REFERENCE_SCHEMA,
            "status": "tool_refused",
            "message": str(error),
        }


@_tool("inspect_periodic_electronic_structure_with_orbitron", program="generic")
def _handle_inspect_periodic_electronic_structure_with_orbitron(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    request = {
        "schema_version": ORBITRON_PERIODIC_REQUEST_SCHEMA,
        "path": arguments["path"],
    }
    try:
        return ScienceRuntimeClient().orbitron_periodic_electronic_structure(
            request
        )
    except ScienceRuntimeUnavailableError as error:
        return _error("unavailable", "science_runtime_unavailable", error)
    except ScienceRuntimeProtocolError as error:
        return _error("incompatible", "science_runtime_protocol_error", error)
    except ScienceRuntimeCommandError as error:
        response = _error("tool_refused", "orbitron_periodic_inspection_error", error)
        response["returncode"] = error.returncode
        if error.stderr:
            response["stderr"] = error.stderr
    return response


@_tool("inspect_structure_identity_with_orbitron", program="generic")
def _handle_inspect_structure_identity_with_orbitron(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    request = {
        "schema_version": ORBITRON_STRUCTURE_IDENTITY_REQUEST_SCHEMA,
        "path": arguments["path"],
    }
    try:
        return ScienceRuntimeClient().orbitron_structure_identity(request)
    except ScienceRuntimeUnavailableError as error:
        return _error("unavailable", "science_runtime_unavailable", error)
    except ScienceRuntimeProtocolError as error:
        return _error("incompatible", "science_runtime_protocol_error", error)
    except ScienceRuntimeCommandError as error:
        response = _error("tool_refused", "orbitron_structure_identity_error", error)
        response["returncode"] = error.returncode
        if error.stderr:
            response["stderr"] = error.stderr
        return response


@_tool("inspect_nbo_with_orbitron", program="generic")
def _handle_inspect_nbo_with_orbitron(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    request = {
        "schema_version": ORBITRON_NBO_REQUEST_SCHEMA,
        "path": arguments["path"],
    }
    try:
        return ScienceRuntimeClient().orbitron_nbo(request)
    except ScienceRuntimeUnavailableError as error:
        return _error("unavailable", "science_runtime_unavailable", error)
    except ScienceRuntimeProtocolError as error:
        return _error("incompatible", "science_runtime_protocol_error", error)
    except ScienceRuntimeCommandError as error:
        response = _error("tool_refused", "orbitron_nbo_inspection_error", error)
        response["returncode"] = error.returncode
        if error.stderr:
            response["stderr"] = error.stderr
        return response


@_tool("run_pyscf_single_point", needs="executable", program="generic")
def _handle_run_pyscf_single_point(arguments: dict[str, Any]) -> dict[str, Any]:
    run_arguments = dict(arguments)
    dry_run = run_arguments.pop("dry_run", False)
    if dry_run:
        _, _, preview = render_pyscf_single_point(**run_arguments)
        return preview
    return run_pyscf_single_point(get_execution_service(), **run_arguments)


@_tool("compare_pyscf_reference_calculation", program="generic")
def _handle_compare_pyscf_reference_calculation(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return compare_pyscf_reference_calculation(
        arguments["pyscf_result"],
        arguments["reference"],
        pyscf_orbital_cube=arguments.get("pyscf_orbital_cube"),
    )


def _error(status: str, code: str, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": SCIENCE_RUNTIME_PROBE_SCHEMA,
        "status": status,
        "error": code,
        "message": str(error),
    }


def science_runtime_tool_definitions() -> list[dict[str, Any]]:
    pyscf_properties = {
        "atoms": {
            "type": "array",
            "minItems": 1,
            "maxItems": 500,
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
            "description": "Cartesian atoms in angstroms.",
        },
        "charge": {"type": "integer"},
        "multiplicity": {"type": "integer", "minimum": 1},
        "method": {
            "type": "string",
            "enum": ["rhf", "uhf", "rks", "uks"],
        },
        "basis": {"type": "string", "minLength": 1},
        "xc": {
            "type": ["string", "null"],
            "description": "Required for RKS and UKS; null for RHF and UHF.",
        },
        "density_fit": {"type": "boolean", "default": False},
        "max_cycles": {
            "type": "integer",
            "minimum": 1,
            "maximum": 500,
            "default": 100,
        },
        "convergence_tolerance": {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": 0.0001,
            "default": 1e-9,
        },
        "max_memory_mb": {
            "type": "integer",
            "minimum": 64,
            "maximum": 262144,
            "default": 2048,
        },
        "density_cube_grid_points": {
            "type": ["integer", "null"],
            "minimum": 20,
            "maximum": 120,
            "default": None,
            "description": "Write one converged PySCF total-density CUBE on this cubic grid; omit or set null to disable artifacts.",
        },
        "orbital_cube_grid_points": {
            "type": ["integer", "null"],
            "minimum": 20,
            "maximum": 120,
            "default": None,
            "description": "Cubic grid size for selected orbital CUBEs; requires orbital_cube_requests.",
        },
        "orbital_cube_requests": {
            "type": ["array", "null"],
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "spin": {
                        "type": "string",
                        "enum": ["restricted", "alpha", "beta"],
                    },
                    "orbital_index": {"type": "integer", "minimum": 0},
                },
                "required": ["spin", "orbital_index"],
                "additionalProperties": False,
            },
            "default": None,
            "description": "Up to eight zero-based PySCF MO selectors. Restricted methods require restricted; unrestricted methods require alpha or beta.",
        },
        "omp_threads": {
            "type": "integer",
            "minimum": 1,
            "maximum": 128,
            "default": 1,
        },
        "timeout_seconds": {
            "type": "number",
            "minimum": 1,
            "maximum": 3600,
            "default": 120,
        },
        "working_directory": {
            "type": "string",
            "minLength": 1,
            "description": "Existing local directory used for the process and PySCF temporary files.",
        },
        "job_name": {"type": "string", "default": "pyscf_single_point"},
    }
    return [
        {
            "name": "inspect_science_runtime",
            "description": (
                "Inspect the configured optional companion scientific Python "
                "runtime through a fixed, read-only probe. Reports the "
                "interpreter and availability and version evidence for PySCF, "
                "RDKit, Open Babel, and Orbitron's Python API. The tool does "
                "not install packages, run chemistry, import caller-selected "
                "modules, or accept a Python path. Configure the interpreter "
                f"with {SCIENCE_RUNTIME_PYTHON_ENV}."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "preflight_molecule_with_rdkit",
            "description": (
                "Validate one SMILES string or MDL mol block through the "
                "configured RDKit companion runtime. Returns canonical RDKit "
                "evidence, formula, atom and bond counts, formal charge, "
                "fragment count, radical-electron evidence, and warnings. "
                "The submitted source is preserved; RDKit's chemical "
                "perception is reported as evidence and is not silently "
                "applied to a calculation input."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["smiles", "molblock"],
                    },
                    "source": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1_000_000,
                    },
                },
                "required": ["format", "source"],
                "additionalProperties": False,
            },
        },
        {
            "name": "convert_molecule_with_openbabel",
            "description": (
                "Convert one SMILES string or MDL mol block through the "
                "configured Open Babel companion runtime, then independently "
                "inspect both forms with RDKit. Returns converted text, "
                "hashes, Open Babel and RDKit version evidence, and explicit "
                "matched or different molecular evidence for connectivity, "
                "charge, aromaticity, and stereochemistry. SMILES-to-mol "
                "conversion does not generate coordinates; its zero-coordinate "
                "mol block is connectivity evidence only. The tool refuses "
                "outputs that RDKit cannot independently inspect."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["smiles", "molblock"],
                    },
                    "source": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 131_072,
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["smiles", "molblock"],
                    },
                },
                "required": ["format", "source", "output_format"],
                "additionalProperties": False,
            },
        },
        {
            "name": "render_basis_set_with_bse",
            "description": (
                "Render an explicit orbital basis block from the installed "
                "Basis Set Exchange data through the configured companion "
                "runtime. Returns the exact emitted text, content hash, and "
                "Basis Set Exchange version. It uses bundled BSE data only; "
                "it does not fetch a basis, use a custom overlay, or decide "
                "whether the basis, ECP, relativistic Hamiltonian, or angular "
                "convention is scientifically suitable."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "basis": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                        "description": "Basis Set Exchange basis name.",
                    },
                    "elements": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 32,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 3,
                        },
                        "description": "Element symbols or atomic numbers to include.",
                    },
                    "program_format": {
                        "type": "string",
                        "enum": [
                            "nwchem",
                            "molcas",
                            "orca",
                            "gaussian94",
                            "qchem",
                            "psi4",
                            "cp2k",
                            "turbomole",
                        ],
                    },
                },
                "required": ["basis", "elements", "program_format"],
                "additionalProperties": False,
            },
        },
        {
            "name": "fetch_nist_atomic_reference",
            "description": (
                "Fetch one NIST Atomic Spectra Database energy-level or "
                "ionization-energy table in NIST's tab-delimited form. The "
                "tool calls fixed NIST endpoints, caches the exact response "
                "locally with URL, retrieval time, and SHA-256 provenance, "
                "and returns at most 500 rows. It does not bulk-mirror ASD, "
                "assign calculated states to experimental levels, or treat "
                "an energy match as a unique state assignment."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["energy_levels", "ionization_energies"],
                    },
                    "spectrum": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 64,
                        "description": "NIST ASD spectrum notation, for example O I, U III, or Mg Li-like.",
                    },
                    "row_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "default": 200,
                    },
                    "refresh": {
                        "type": "boolean",
                        "default": False,
                        "description": "Ignore a matching local cache entry and query NIST again.",
                    },
                },
                "required": ["kind", "spectrum"],
                "additionalProperties": False,
            },
        },
        {
            "name": "inspect_periodic_electronic_structure_with_orbitron",
            "description": (
                "Inspect bounded periodic band-structure and density-of-states "
                "evidence with Orbitron's configured Python API. Returns source "
                "hash and package provenance, Fermi energy, band-gap evidence, "
                "band sampling and dimensions, and DOS energy range and "
                "dimensions. It omits raw curves and projections, does not run "
                "a calculation, and does not decide whether the calculation "
                "method or k-point sampling is scientifically adequate."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Absolute local path to a supported periodic electronic-structure output.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "inspect_structure_identity_with_orbitron",
            "description": (
                "Inspect one molecular or coordination structure with Orbitron's "
                "configured Python API. Returns source and package provenance, "
                "atom and bond counts, Orbitron bond-order counts including "
                "Dative when assigned during canonical conversion, and formula, "
                "InChI, InChIKey, and SMILES evidence when available. It does "
                "not change the input or decide whether a coordination model, "
                "bond order, or chemical identity is scientifically correct."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Absolute local path to a supported molecular or coordination-structure file.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "inspect_nbo_with_orbitron",
            "description": (
                "Inspect Natural Bond Orbital evidence with Orbitron's configured "
                "Python API. Returns source and package provenance, orbital-type "
                "counts, occupancy range, per-atom entry counts, and at most twelve "
                "BD, BD*, LP, or LP* orbital samples with bounded atom weights and "
                "coefficient signs. It does not return raw NBO tables, change the "
                "input, or determine a unique bonding model or oxidation state."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Absolute local path to a supported output containing Natural Bond Orbital data.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "run_pyscf_single_point",
            "description": (
                "Run a bounded molecular PySCF RHF, UHF, RKS, or UKS "
                "single-point calculation through the configured companion "
                "interpreter. The operation accepts typed atoms in angstroms "
                "and fixed SCF controls only. It records execution evidence "
                "separately from SCF convergence and does not accept Python "
                "source, custom method objects, periodic cells, geometry "
                "optimization, or multireference methods."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    **pyscf_properties,
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Render the fixed companion command without starting it.",
                    },
                },
                "required": [
                    "atoms",
                    "charge",
                    "multiplicity",
                    "method",
                    "basis",
                    "working_directory",
                ],
                "additionalProperties": False,
            },
        },
        {
            "name": "compare_pyscf_reference_calculation",
            "description": (
                "Compare a completed bounded PySCF single-point result with "
                "caller-declared reference-calculation evidence. Reports "
                "matched and different geometry, settings, electron count, "
                "SCF outcome, energy difference, and optional same-grid "
                "density or phase-aligned orbital CUBE evidence. It does not "
                "select a correct calculation from an energy difference."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pyscf_result": {
                        "type": "object",
                        "description": "The result object returned by run_pyscf_single_point after successful execution.",
                    },
                    "reference": {
                        "type": "object",
                        "description": "Caller-declared reference evidence: label, Cartesian geometry, calculation settings, SCF state, total energy, electron count, and optional CUBE artifacts.",
                    },
                    "pyscf_orbital_cube": {
                        "type": ["object", "null"],
                        "description": "Optional externally written PySCF orbital CUBE with path and caller-declared orbital_label.",
                    },
                },
                "required": ["pyscf_result", "reference"],
                "additionalProperties": False,
            },
        },
    ]


__all__ = ["science_runtime_tool_definitions"]

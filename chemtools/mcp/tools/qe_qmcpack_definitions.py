"""Static MCP schemas for QE-to-QMCPACK conversion inspection."""

from __future__ import annotations

from typing import Any


def qe_tool_definitions() -> list[dict[str, Any]]:
    return [{
        "name": "check_qe_qmcpack_conversion_ready",
        "description": (
            "Check one QE pw.x input against the documented preconditions for "
            "pw2qmcpack conversion: SCF mode, wavefunction-preserving disk_io, "
            "an explicit crystal gamma point, and compatible isolation "
            "conventions. It does not run QE or pw2qmcpack, inspect an output, "
            "or establish an energy comparison."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "qe_input": {
                    "type": "string",
                    "description": "Path to one Quantum ESPRESSO pw.x input file.",
                },
            },
            "required": ["qe_input"],
            "additionalProperties": False,
        },
    }, {
        "name": "plan_qe_qmcpack_conversion",
        "description": (
            "Declare the QE SCF, pw2qmcpack, and QMCPACK-deck artifact "
            "handoff for a caller-supplied .pwscf.h5 path. Includes the QE "
            "conversion preflight but deliberately does not generate converter "
            "input or command-line options, launch programs, or decode HDF5."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "qe_input": {
                    "type": "string",
                    "description": "Path to one Quantum ESPRESSO pw.x input file.",
                },
                "pwscf_h5": {
                    "type": "string",
                    "description": "Planned path for the pw2qmcpack .pwscf.h5 artifact.",
                },
            },
            "required": ["qe_input", "pwscf_h5"],
            "additionalProperties": False,
        },
    }, {
        "name": "draft_pw2qmcpack_input",
        "description": (
            "Draft the demonstrated pw2qmcpack inputpp namelist from explicit "
            "QE &CONTROL prefix and outdir values, with write_psir disabled. "
            "Returns review rather than guessing missing QE defaults. It does "
            "not launch the converter or inspect HDF5 output."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "qe_input": {
                    "type": "string",
                    "description": "Path to one Quantum ESPRESSO pw.x input file.",
                },
            },
            "required": ["qe_input"],
            "additionalProperties": False,
        },
    }, {
        "name": "draft_ph_x_input",
        "description": (
            "Draft a single-q Quantum ESPRESSO ph.x input from explicit QE "
            "&CONTROL prefix and outdir values, a one-line title, and one "
            "caller-supplied q-vector. Returns review rather than guessing QE "
            "paths or phonon settings. It does not create q-point grids, choose "
            "epsil, launch ph.x, or inspect phonon output."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "qe_input": {
                    "type": "string",
                    "description": "Path to one Quantum ESPRESSO pw.x SCF input file.",
                },
                "title": {
                    "type": "string",
                    "minLength": 1,
                    "description": "One-line ph.x job identifier.",
                },
                "q_point": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "One phonon q-vector in ph.x units of 2pi/a0.",
                },
            },
            "required": ["qe_input", "title", "q_point"],
            "additionalProperties": False,
        },
    }, {
        "name": "inspect_qe_qmcpack_conversion",
        "description": (
            "Run Chemtools' bounded QE-to-QMCPACK conversion inspection in "
            "one request: QE preconditions and completed SCF evidence, HDF5 "
            "lineage, XML references, QMCPACK pseudopotentials, species, "
            "valence, electron, atom, ion-species, periodic geometry, "
            "fixed-moment spin, charge accounting, and optional fixed-layout "
            "pw2qmcpack HDF5 metadata against the QMCPACK deck. It does not "
            "decode coefficients or arbitrary HDF5 datasets, run a converter or "
            "QMCPACK, establish potential-family equivalence, "
            "or validate an energy comparison."
        ),
        "inputSchema": _conversion_input_schema(),
    }, {
        "name": "inspect_qe_qmcpack_conversion_artifacts",
        "description": (
            "Inspect declared QE-to-QMCPACK conversion artifacts. Rechecks the "
            "QE input preconditions, requires a completed converged QE SCF output, "
            "and checks that an explicitly supplied .pwscf.h5 artifact has an "
            "HDF5 signature and is current. It does not decode HDF5 contents or "
            "run pw2qmcpack."
        ),
        "inputSchema": _artifact_input_schema(),
    }, {
        "name": "inspect_qe_qmcpack_conversion_execution",
        "description": (
            "Inspect the declared QE-to-QMCPACK chain, including a supplied "
            "pw2qmcpack input and completed converter output. It composes the "
            "existing QE, HDF5-path, and QMCPACK-deck checks with converter "
            "input-output evidence. It does not decode HDF5, establish "
            "pseudopotential-family equivalence, or validate an energy comparison."
        ),
        "inputSchema": _conversion_execution_input_schema(),
    }, {
        "name": "inspect_qe_qmcpack_conversion_deck",
        "description": (
            "Inspect declared QE conversion artifacts and confirm that a QMCPACK "
            "XML deck resolves an HDF5 reference to the exact supplied .pwscf.h5 "
            "artifact, including bounded nested XML includes. It does not decode "
            "HDF5, merge XML, compare physics settings, or run a converter."
        ),
        "inputSchema": _conversion_input_schema(),
    }, {
        "name": "inspect_qe_qmcpack_conversion_pseudopotentials",
        "description": (
            "Inspect declared QE conversion artifacts, the QMCPACK XML deck, and "
            "each QMCPACK pseudopotential referenced from the bounded XML include "
            "graph. Reports supported semilocal structural evidence only. It does "
            "not establish pseudopotential transferability, QE-UPF equivalence, or "
            "a valid energy comparison."
        ),
        "inputSchema": _conversion_input_schema(),
    }, {
        "name": "inspect_qe_qmcpack_conversion_species",
        "description": (
            "Inspect declared QE conversion artifacts and compare QE atomic-"
            "species elements with QMCPACK pseudopotential elementType "
            "declarations, including bounded XML includes. It does not establish "
            "pseudopotential family or valence equivalence, spin state, or "
            "coordinate consistency."
        ),
        "inputSchema": _conversion_input_schema(),
    }, {
        "name": "inspect_qe_qmcpack_conversion_valence",
        "description": (
            "Inspect declared QE conversion artifacts and compare parsed QE UPF "
            "z_valence values with QMCPACK pseudopotential XML zval headers, "
            "including bounded XML includes. It does not establish potential "
            "family or scattering equivalence, spin state, or coordinate "
            "consistency."
        ),
        "inputSchema": _conversion_input_schema(),
    }, {
        "name": "inspect_qe_qmcpack_conversion_projectors",
        "description": (
            "Inspect bounded QE UPF projector evidence for a QMCPACK deck that "
            "declares DMC. Reports per-angular-channel counts only when the UPF "
            "preamble contains the complete declared projector set, and flags "
            "multi-projector channels for semilocal-QMC review. It does not prove "
            "pseudopotential family equivalence or DMC compatibility."
        ),
        "inputSchema": _conversion_input_schema(),
    }, {
        "name": "inspect_qe_qmcpack_conversion_electrons",
        "description": (
            "Inspect declared QE conversion artifacts and compare QE runtime or "
            "complete UPF valence-electron evidence with the electron-particle "
            "groups selected by the QMCPACK Hamiltonian. It does not establish a "
            "physical charge state, spin state, orbital occupancy, or potential "
            "equivalence."
        ),
        "inputSchema": _conversion_input_schema(),
    }, {
        "name": "inspect_qe_qmcpack_conversion_atoms",
        "description": (
            "Inspect declared QE conversion artifacts and compare QE's nat value "
            "with the sizes of QMCPACK particle sets other than Hamiltonian "
            "targets, including bounded XML includes. It does not compare element "
            "identities, cell parameters, or coordinates."
        ),
        "inputSchema": _conversion_input_schema(),
    }, {
        "name": "inspect_qe_qmcpack_conversion_ion_species",
        "description": (
            "Inspect declared QE conversion artifacts and compare QE atomic "
            "element counts with explicitly sized QMCPACK non-electron ion "
            "groups, including bounded XML includes. It does not compare "
            "coordinates, pseudopotential identity, charge, or spin."
        ),
        "inputSchema": _conversion_input_schema(),
    }, {
        "name": "inspect_qe_qmcpack_conversion_geometry",
        "description": (
            "Inspect declared QE conversion artifacts and compare an explicit "
            "QE periodic cell and atomic positions with an explicit QMCPACK "
            "bohr simulationcell and ion particle positions, including bounded "
            "XML includes. It requires one Hamiltonian target and one ion "
            "geometry, and does not establish pseudopotential, spin, or energy "
            "equivalence."
        ),
        "inputSchema": _conversion_input_schema(),
    }, {
        "name": "inspect_qe_qmcpack_conversion_spin",
        "description": (
            "Inspect declared QE conversion artifacts and compare an explicit "
            "QE nspin=2 fixed total magnetization with the QMCPACK u-d "
            "electron-particle imbalance, including bounded XML includes. It "
            "does not establish a physical spin state, compare spin densities, "
            "or support noncollinear or spin-orbit calculations."
        ),
        "inputSchema": _conversion_input_schema(),
    }, {
        "name": "inspect_qe_qmcpack_conversion_charge",
        "description": (
            "Inspect declared QE conversion artifacts and compare QE UPF "
            "valence accounting and total charge with QMCPACK ion valence "
            "parameters and selected electron-particle counts, including "
            "bounded XML includes. It does not establish potential-family, "
            "physical charge-state, or spin-state equivalence."
        ),
        "inputSchema": _conversion_input_schema(),
    }]


def _artifact_input_schema() -> dict[str, Any]:
    schema = _conversion_input_schema()
    schema["properties"].pop("qmcpack_input")
    schema["required"].remove("qmcpack_input")
    return schema


def _conversion_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "qe_input": {"type": "string", "description": "Path to the QE pw.x input."},
            "qe_output": {"type": "string", "description": "Path to the matching QE pw.x output."},
            "pwscf_h5": {"type": "string", "description": "Path to the declared pw2qmcpack .pwscf.h5 artifact."},
            "qmcpack_input": {"type": "string", "description": "Path to the QMCPACK XML deck."},
        },
        "required": ["qe_input", "qe_output", "pwscf_h5", "qmcpack_input"],
        "additionalProperties": False,
    }


def _conversion_execution_input_schema() -> dict[str, Any]:
    schema = _conversion_input_schema()
    schema["properties"].update({
        "pw2qmcpack_input": {
            "type": "string",
            "description": "Path to the pw2qmcpack.x input file.",
        },
        "pw2qmcpack_output": {
            "type": "string",
            "description": "Path to the matching pw2qmcpack.x output file.",
        },
    })
    schema["required"].extend(["pw2qmcpack_input", "pw2qmcpack_output"])
    return schema

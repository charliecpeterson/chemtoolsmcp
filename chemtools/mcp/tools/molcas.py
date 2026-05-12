"""Molcas MCP tool definitions and handlers.

Importing this module registers every @_tool handler with the shared
`_TOOL_REGISTRY` in chemtools.mcp.decorator. The accompanying
`molcas_tool_definitions()` is appended into the multi-program tool list
exposed by `chemtools/mcp/tools/nwchem.py:tool_definitions()`.

Tools provided (21):

  Input drafting + linting:
  draft_molcas_input             full SEWARD+SCF+(RASSCF)+(CASPT2) input deck
  lint_molcas_input              structural + semantic checks on an input string
  compute_molcas_active_space_partition   CAS(M,N) → per-symmetry RASSCF directives
  list_molcas_basis_sets         enumerate bundled basis sets (filterable by element)

  Output parsing:
  parse_molcas_output            full deep parse + active-space summary + warnings
  parse_molcas_tasks             cheap module-boundary task index
  get_molcas_orbitals            dump the LAST '++ Molecular orbitals:' block
  parse_molcas_inporb            parse INPORB / RasOrb / ScfOrb / GssOrb / LprOrb
  parse_molcas_frequencies       MCLR / numerical-grad harmonic frequencies + normal modes
  parse_molcas_thermochem        per-temperature ZPVE + S + U + H + G
  extract_molcas_geometry        SLAPAF converged geometry (or specific Cartesian block)
  parse_molcas_trajectory        SLAPAF Energy Statistics + per-iteration geometries
  parse_molcas_rassi             RASSI: input states + SF / SO eigenstates + SOC matrix + composition + osc strengths

  Runtime / launch:
  prepare_molcas_launch          safe pymolcas command + env (CASPT2 -np 1 guard, scratch isolation)

  Strategy:
  analyze_molcas_active_space    active-space quality verdict + recommendations
  validate_molcas_caspt2_setup   CASPT2 quality verdict (refweight, intruders, shift)

  Documentation:
  list_molcas_docs               list bundled OpenMolcas docs
  search_molcas_docs             search bundled OpenMolcas docs
  lookup_molcas_module           pull the docs page for a given module
  read_molcas_doc_excerpt        read a slice of a bundled doc
  get_molcas_topic_guide         curated guidance for high-value topics
"""

from __future__ import annotations

from typing import Any

from chemtools.core.common import read_text
from chemtools.mcp.decorator import _tool
from chemtools.programs.molcas.parse.output import (
    parse_tasks as _parse_tasks,
    parse_output_full as _parse_output_full,
)
from chemtools.programs.molcas.parse.mos import parse_last_mo_block as _parse_last_mo_block
from chemtools.programs.molcas.parse.freq import parse_last_freq_block as _parse_last_freq_block
from chemtools.programs.molcas.parse.thermochem import parse_thermochem_block as _parse_thermochem_block
from chemtools.programs.molcas.parse.geometry import (
    parse_cartesian_blocks as _parse_cartesian_blocks,
    parse_final_geometry as _parse_final_geometry,
    parse_trajectory as _parse_trajectory,
)
from chemtools.programs.molcas.parse.rassi import parse_rassi as _parse_rassi
from chemtools.programs.molcas.binary.orbitals import (
    parse_inporb as _parse_inporb,
    swap_orbitals_in_inporb as _swap_orbitals_in_inporb,
)
from chemtools.programs.molcas.runtime import prepare_launch as _prepare_molcas_launch
from chemtools.programs.molcas.strategy.active_space import (
    analyze_active_space as _analyze_active_space,
    validate_caspt2_setup as _validate_caspt2_setup,
    suggest_orbital_swaps_by_character as _suggest_swaps_by_character,
)
from chemtools.programs.molcas.strategy.orchestrators import (
    refine_active_space as _refine_active_space,
    prepare_casscf_setup as _prepare_casscf_setup,
    prepare_caspt2_chain as _prepare_caspt2_chain,
    prepare_excited_states_workflow as _prepare_excited_states_workflow,
    prepare_opt_freq_workflow as _prepare_opt_freq_workflow,
    prepare_irc_workflow as _prepare_irc_workflow,
    prepare_scan_workflow as _prepare_scan_workflow,
)
from chemtools.programs.molcas.strategy.reaction_energy import (
    compute_reaction_energy as _compute_reaction_energy,
    check_active_space_consistency as _check_active_space_consistency,
)
from chemtools.programs.molcas.strategy.atomization import (
    prepare_atomization_calculation as _prepare_atomization_calculation,
)
from chemtools.programs.molcas.strategy.recovery import (
    suggest_recovery as _suggest_recovery,
    apply_recovery as _apply_recovery,
)
from chemtools.programs.molcas.docs import (
    list_docs as _list_docs,
    search_docs as _search_docs,
    lookup_module_syntax as _lookup_module_syntax,
    read_doc_excerpt as _read_doc_excerpt,
    get_topic_guide as _get_topic_guide,
)
from chemtools.programs.molcas.input.draft import draft_molcas_input as _draft_input
from chemtools.programs.molcas.input.lint import lint_molcas_input as _lint_input
from chemtools.programs.molcas.input.rasscf import (
    compute_active_space_partition as _compute_active_space_partition,
)
from chemtools.programs.molcas.input.basis_library import (
    list_basis_sets as _list_basis_sets,
    list_elements_for_basis as _list_elements_for_basis,
    list_contractions_for as _list_contractions_for,
    default_contraction as _default_contraction,
    basis_label as _basis_label,
)


def molcas_tool_definitions() -> list[dict[str, Any]]:
    return [
        # ----- Input drafting -----
        {
            "name": "draft_molcas_input",
            "description": (
                "Draft a complete Molcas input deck (MOLCAS_MEM + SEWARD + SCF + RASSCF + "
                "CASPT2 chain as appropriate for the requested method). Methods: HF, SCF, "
                "DFT/KSDFT, CASSCF, RASSCF, CASPT2, RASPT2, MS-CASPT2, XMS-CASPT2, "
                "RMS-CASPT2, XDW-CASPT2. Active-space methods require "
                "program_options.cas_active_electrons and cas_active_orbitals. For runs "
                "with non-trivial symmetry, supply n_basis_per_symmetry, "
                "occupied_per_symmetry, and rasscf.{inactive,active}_per_symmetry."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "atoms": {
                        "type": "array",
                        "description": "Geometry atoms with symbol + x/y/z (and optional label).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"},
                                "label": {"type": "string"},
                            },
                            "required": ["symbol", "x", "y", "z"],
                        },
                    },
                    "charge": {"type": "integer", "default": 0},
                    "multiplicity": {"type": "integer", "default": 1},
                    "method": {"type": "string", "description": "HF/SCF/DFT/CASSCF/RASSCF/CASPT2/MS-CASPT2/etc."},
                    "basis": {
                        "description": "Basis set name (e.g. 'ANO-S') or per-element dict (e.g. {'C': 'ANO-S', 'Fe': 'ANO-RCC'}).",
                        "oneOf": [
                            {"type": "string"},
                            {"type": "object", "additionalProperties": {"type": "string"}},
                        ],
                    },
                    "task": {"type": "string", "default": "energy"},
                    "title": {"type": "string"},
                    "geometry_units": {"type": "string", "enum": ["angstrom", "bohr"], "default": "angstrom"},
                    "functional": {"type": "string", "description": "DFT functional (B3LYP, PBE, M06, ...)."},
                    "program_options": {
                        "type": "object",
                        "description": (
                            "Molcas-specific knobs. Recognized keys: memory_mb, symmetry, "
                            "n_symmetries, n_basis_per_symmetry, occupied_per_symmetry, pkthrs, "
                            "cholesky, ricd, expert, scf{}, cas_active_electrons, cas_active_orbitals, "
                            "rasscf{}, caspt2{}, seward_extra_keywords[]."
                        ),
                    },
                },
                "required": ["atoms", "method", "basis"],
                "additionalProperties": False,
            },
        },
        {
            "name": "lint_molcas_input",
            "description": (
                "Validate a Molcas input string. Returns a list of issues with level "
                "(error / warning / info), line number, message, and copy-paste suggested fix "
                "where applicable. Checks include block-pair (`&MODULE` / `End of input`) "
                "consistency, basis-label library/element existence, RASSCF/CASPT2 Frozen "
                "consistency, Nactel sanity, LumOrb provenance, and Spin sanity."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_text": {"type": "string", "description": "Full Molcas input file content."},
                },
                "required": ["input_text"],
                "additionalProperties": False,
            },
        },
        {
            "name": "compute_molcas_active_space_partition",
            "description": (
                "Translate a desired CAS(M, N) into the per-symmetry RASSCF directives: "
                "Nactel, Frozen, Inactive, Ras1/2/3, Secondary. For C1, the function fills "
                "in everything from the CAS dimensions; for higher symmetry it requires "
                "explicit n_basis_per_symmetry, n_inactive_per_symmetry, and "
                "active_per_symmetry (or target_symmetry_for_active)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_electrons": {"type": "integer", "description": "Total electrons in the molecule (= sum atomic numbers - charge)."},
                    "cas_active_electrons": {"type": "integer"},
                    "cas_active_orbitals": {"type": "integer"},
                    "n_symmetries": {"type": "integer", "default": 1},
                    "n_basis_per_symmetry": {"type": "array", "items": {"type": "integer"}},
                    "n_frozen_per_symmetry": {"type": "array", "items": {"type": "integer"}},
                    "active_per_symmetry": {"type": "array", "items": {"type": "integer"}},
                    "target_symmetry_for_active": {"type": "integer", "description": "1-indexed irrep that hosts the entire active space (for higher symmetry)."},
                    "n_inactive_per_symmetry": {"type": "array", "items": {"type": "integer"}},
                    "ras1_holes_max": {"type": "integer", "default": 0},
                    "ras1_per_symmetry": {"type": "array", "items": {"type": "integer"}},
                    "ras3_electrons_max": {"type": "integer", "default": 0},
                    "ras3_per_symmetry": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["n_electrons", "cas_active_electrons", "cas_active_orbitals"],
                "additionalProperties": False,
            },
        },
        {
            "name": "list_molcas_basis_sets",
            "description": (
                "List bundled Molcas basis-set names. Optionally filter to those that "
                "support a given element, or return the available contractions for a "
                "specific (basis, element) pair."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "element": {"type": "string", "description": "Filter to bases supporting this element."},
                    "basis_name": {"type": "string", "description": "If set with element, return per-element contractions."},
                },
                "additionalProperties": False,
            },
        },
        # ----- Output parsing -----
        {
            "name": "parse_molcas_output",
            "description": (
                "Deep parse of a Molcas .out file. Returns module-by-module results "
                "(SCF / RASSCF / CASPT2 details), an energy roll-up (primary energy "
                "selection follows MS-CASPT2 > CASPT2 > RASSCF > SCF), an "
                "active-space summary with NO occupation classification, and a list "
                "of cross-task warnings (low reference weight, intruder states, etc.)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the Molcas .out file."},
                    "include_mo_coefficients": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include AO coefficients in the per-task MO blocks (heavy; default off).",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_molcas_tasks",
            "description": (
                "Cheap pass — emit only the module-boundary task list. Use when you "
                "only need to know which Molcas modules ran and their return codes."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_molcas_orbitals",
            "description": (
                "Return the LAST '++ Molecular orbitals:' block from a Molcas .out "
                "file (or from a chosen task). For RASSCF tasks this returns the "
                "natural-orbital block with occupations + dominant AO contributions. "
                "For SCF tasks it returns the canonical SCF MOs. Use this to label "
                "active orbitals (π, π*, lone pair, etc.) before tweaking the active "
                "space."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "task_index": {
                        "type": ["integer", "null"],
                        "default": None,
                        "description": "0-indexed task; null = auto (prefer last RASSCF, fall back to last SCF).",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        # ----- Orbital file (greenfield) -----
        {
            "name": "parse_molcas_inporb",
            "description": (
                "Parse a Molcas orbital file in the INPORB / RasOrb / ScfOrb / GssOrb / "
                "LprOrb / SpdOrb format. Returns per-symmetry MO coefficients, "
                "occupation numbers, orbital energies, and the typeindex partitioning "
                "(frozen / inactive / RAS1 / RAS2 / RAS3 / secondary / deleted), plus "
                "an active-space signature derived from the file alone."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "orbital_file": {"type": "string", "description": "Path to a RasOrb / ScfOrb / GssOrb / LprOrb / SpdOrb / INPORB file."},
                    "include_coefficients": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include MO coefficients in the response (large; set False for a quick metadata read).",
                    },
                },
                "required": ["orbital_file"],
                "additionalProperties": False,
            },
        },
        # ----- Frequency / thermochem / geometry / trajectory -----
        {
            "name": "parse_molcas_frequencies",
            "description": (
                "Parse the LAST 'Harmonic frequencies in cm-1' block from a Molcas .out "
                "file (MCLR analytical or numerical-gradient driven). Returns per-symmetry "
                "modes with frequency, IR intensity, reduced mass, and per-atom-direction "
                "displacements; flat list across symmetries; counts of imaginary modes "
                "(stored as negative floats)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_molcas_thermochem",
            "description": (
                "Parse the per-temperature thermochemistry block emitted by MCLR. Returns "
                "ZPVE (kcal/mol + au), ZPVE-corrected energy, molecular mass, rotational "
                "constants (cm-1 + GHz), rotational symmetry factor, and one row per "
                "temperature with entropy / U / H / G in both kcal/mol and au. The "
                "298.15 K row is duplicated under `standard_298_15` for convenience."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "extract_molcas_geometry",
            "description": (
                "Return a single geometry snapshot from a Molcas .out file. By default "
                "returns SLAPAF's converged geometry if present, else the LAST "
                "'Cartesian coordinates in angstrom:' block. Pass block_index to pick "
                "a specific 'Cartesian coordinates' emission (0-indexed in source order)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "block_index": {"type": ["integer", "null"], "default": None},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_molcas_trajectory",
            "description": (
                "Walk the SLAPAF Energy Statistics + per-iteration geometry blocks. "
                "Returns one row per opt iteration (energy, gradient_norm, gradient_max, "
                "step_max, geometry snapshot) plus convergence verdict. The Energy "
                "Statistics table is cumulative — only the LAST emission is consumed so "
                "duplicates aren't appended."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_molcas_rassi",
            "description": (
                "Parse the &RASSI (RAS State Interaction) module block from a Molcas .out "
                "file. Returns input-state mapping (which JobIph + root each state came "
                "from), spin-free eigenstate energies (relative + absolute, au + eV + cm-1), "
                "spin-orbit eigenstate energies, SO composition (top spin-free contributors "
                "per SO state with weights), SOC matrix elements above the SOCOupling "
                "threshold, dipole oscillator strengths in spin-free and spin-orbit bases, "
                "natural-orbital occupations per RASSI eigenstate. Includes a roll-up "
                "energy summary with SOC stabilization in cm-1."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        # ----- Runtime / launch helper -----
        {
            "name": "prepare_molcas_launch",
            "description": (
                "Build a safe pymolcas launch command + environment for a given input. "
                "Guards against two known Molcas runtime pitfalls: (1) parallel CASPT2 "
                "may segfault on builds where GA wasn't compiled with --with-mpi-ts — "
                "if the profile sets execution.parallel_caspt2_supported=False, the "
                "launcher auto-downgrades to -np 1 for any input containing &CASPT2 and "
                "emits a warning; (2) Molcas refuses to mix runs with different nProcs "
                "in the same scratch dir — the launcher always sets MOLCAS_PROJECT to a "
                "unique per-input value (default = input file stem) to isolate scratch. "
                "Returns the full command, env-var dict, and any warnings. Does NOT "
                "execute — the caller (or agent) runs the returned command themselves."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "requested_np": {"type": "integer", "default": 1, "minimum": 1},
                    "profile": {
                        "type": ["object", "null"],
                        "default": None,
                        "description": "Optional runner-profile dict (see chemtools/runner_profiles.example.yaml). Fields consulted: execution.parallel_caspt2_supported, execution.apptainer_sif, execution.pymolcas_command, execution.env.",
                    },
                    "job_name": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": "Overrides MOLCAS_PROJECT. Default: input file stem.",
                    },
                    "apptainer_sif": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": "Path to a .sif container image. If supplied, wraps the command with apptainer exec.",
                    },
                    "extra_env": {
                        "type": ["object", "null"],
                        "default": None,
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "swap_molcas_inporb_orbitals",
            "description": (
                "Swap pairs of orbitals within a symmetry block of an INPORB / "
                "RasOrb / ScfOrb file. Writes a new file with the swapped MO "
                "coefficients, occupations, orbital energies, and typeindex. "
                "Use this to move orbitals between inactive/active/secondary "
                "spaces (typical TM-complex active-space tuning workflow): "
                "inspect the current active space with get_molcas_orbitals or "
                "parse_molcas_inporb, identify the wrong-class orbital, find a "
                "candidate replacement, swap them, re-run RASSCF with the new "
                "INPORB (via FILEORB keyword). Does NOT modify the input file."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_orbital_file": {"type": "string", "description": "Source INPORB / RasOrb file."},
                    "output_orbital_file": {"type": "string", "description": "Destination path for the modified file."},
                    "swaps": {
                        "type": "array",
                        "description": "List of (orbital_a, orbital_b) 1-indexed orbital pairs to swap.",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "symmetry": {
                        "type": "integer",
                        "default": 1,
                        "minimum": 1,
                        "description": "1-indexed symmetry irrep. Default 1 (correct for C1).",
                    },
                },
                "required": ["input_orbital_file", "output_orbital_file", "swaps"],
                "additionalProperties": False,
            },
        },
        # ----- Strategy -----
        {
            "name": "analyze_molcas_active_space",
            "description": (
                "Diagnose a Molcas active space using NO occupation thresholds (per-root if "
                "available). Accepts either a Molcas .out file or a RasOrb-format orbital "
                "file. Returns: signature (e.g. CAS(8,7)), a 'healthy' / 'marginal' / "
                "'poor' verdict, per-root quality with promote/demote orbital indices, "
                "and a deterministic next_actions envelope. Use this BEFORE drafting a "
                "CASPT2 input to avoid wasting cycles on an unhealthy reference."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Molcas .out file (preferred — gives per-root NOs)."},
                    "orbital_file": {"type": "string", "description": "RasOrb-format file (state-averaged NOs only)."},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "validate_molcas_caspt2_setup",
            "description": (
                "Inspect a CASPT2 result and emit a 'healthy' / 'caution' / 'unreliable' "
                "verdict with structured checks (per-group reference weight, IPEA/SHIFT/"
                "IMAGINARY SHIFT/SIG2 setup, real intruder excitations, multi-state "
                "consistency hint). Includes a next_actions envelope so the agent knows "
                "whether to trust the energies or to redraft the input."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the Molcas .out file containing the CASPT2 task."},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_molcas_orbital_swaps",
            "description": (
                "Character-aware orbital-swap suggester for active-space tuning. "
                "Walks the LAST '++ Molecular orbitals:' block in a Molcas output, "
                "classifies each orbital's space (inactive / active / secondary) using the "
                "RASSCF orbital_specs, matches each orbital's dominant AO against a "
                "target pattern (e.g. 'Cr' + '3d'), and proposes (active_orbital, "
                "swap_with) pairs that bring target-character orbitals INTO the active "
                "space. The output is ready to feed into swap_molcas_inporb_orbitals."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the Molcas .out file (must have a RASSCF block)."},
                    "target_atom_pattern": {
                        "type": "string",
                        "description": "Case-insensitive prefix to match an atom label, e.g. 'Cr' (matches 'CR1', 'Cr1', etc.).",
                    },
                    "target_ao_pattern": {
                        "type": "string",
                        "description": "Case-insensitive prefix to match an AO label, e.g. '3d' (matches '3d2-', '3d2+', etc.).",
                    },
                    "symmetry": {"type": "integer", "default": 1, "minimum": 1},
                    "top_dominant_aos": {
                        "type": "integer",
                        "default": 1,
                        "minimum": 1,
                        "description": "Number of dominant AOs to inspect when judging an orbital's character.",
                    },
                },
                "required": ["output_file", "target_atom_pattern", "target_ao_pattern"],
                "additionalProperties": False,
            },
        },
        # ----- Thick orchestrators -----
        {
            "name": "refine_molcas_active_space",
            "description": (
                "Thick orchestrator that closes the active-space-tuning loop in ONE call: "
                "parse an existing RASSCF output, run occupation-based analysis, run "
                "character-aware swap suggester for a target AO pattern (e.g. 'Cr' + '3d'), "
                "apply the suggested swaps to the RasOrb file, write a refined input that "
                "uses FILEORB to read the swapped orbitals, prepare a safe launch plan. "
                "Returns a Diagnosis envelope with next_actions. Replaces the manual chain "
                "parse_molcas_output → suggest_molcas_orbital_swaps → "
                "swap_molcas_inporb_orbitals → text-edit input → prepare_molcas_launch."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Existing Molcas .out file with a RASSCF task."},
                    "target_atom_pattern": {"type": "string"},
                    "target_ao_pattern": {"type": "string"},
                    "rasorb_file": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": "Source RasOrb file. Default: <output_file_stem>.RasOrb next to the output.",
                    },
                    "input_file": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": "Source .input file to clone. Default: <output_file_stem>.input next to the output.",
                    },
                    "output_dir": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": "Directory for refined files. Default: same as output_file.",
                    },
                    "refined_job_name": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": "Base name for refined files. Default: '<stem>_refined'.",
                    },
                    "apply_swaps": {
                        "type": "boolean",
                        "default": True,
                        "description": "If False, only return suggestions (dry-run).",
                    },
                    "symmetry": {"type": "integer", "default": 1, "minimum": 1},
                    "max_swaps": {"type": "integer", "default": 5},
                    "apptainer_sif": {"type": ["string", "null"], "default": None},
                    "profile": {"type": ["object", "null"], "default": None},
                    "requested_np": {"type": "integer", "default": 1, "minimum": 1},
                },
                "required": ["output_file", "target_atom_pattern", "target_ao_pattern"],
                "additionalProperties": False,
            },
        },
        {
            "name": "prepare_molcas_casscf_setup",
            "description": (
                "Greenfield thick orchestrator for a fresh CASSCF / CASPT2 / MS-CASPT2 "
                "calculation. Takes a molecule + method spec + EITHER explicit "
                "(cas_active_electrons, cas_active_orbitals) OR chemistry_hint ('valence_d' "
                "for TM, 'frontier_pair' for closed-shell). Drafts the input, lints, "
                "optionally writes it to disk + builds a launch plan. Returns a Diagnosis "
                "envelope with the active-space rationale, lint issues, launch command, "
                "and next_actions. Mirrors NWChem's prepare_nwchem_mcscf_setup."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "atoms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"},
                                "label": {"type": "string"},
                            },
                            "required": ["symbol", "x", "y", "z"],
                        },
                    },
                    "charge": {"type": "integer", "default": 0},
                    "multiplicity": {"type": "integer", "default": 1},
                    "basis": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "object", "additionalProperties": {"type": "string"}},
                        ]
                    },
                    "title": {"type": ["string", "null"], "default": None},
                    "method": {"type": "string", "default": "CASSCF"},
                    "geometry_units": {"type": "string", "enum": ["angstrom", "bohr"], "default": "angstrom"},
                    "chemistry_hint": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": "Auto-derive CAS: 'valence_d' (all TM d-orbitals + neutral-state d electrons - charge) or 'frontier_pair' (HOMO/LUMO 2,2).",
                    },
                    "cas_active_electrons": {"type": ["integer", "null"], "default": None},
                    "cas_active_orbitals": {"type": ["integer", "null"], "default": None},
                    "program_options": {"type": ["object", "null"], "default": None},
                    "job_name": {"type": ["string", "null"], "default": None},
                    "write_input_to": {"type": ["string", "null"], "default": None},
                    "apptainer_sif": {"type": ["string", "null"], "default": None},
                    "profile": {"type": ["object", "null"], "default": None},
                    "requested_np": {"type": "integer", "default": 1, "minimum": 1},
                },
                "required": ["atoms", "basis"],
                "additionalProperties": False,
            },
        },
        {
            "name": "prepare_molcas_caspt2_chain",
            "description": (
                "Continuation orchestrator: given a converged RASSCF .out, generate a "
                "ready-to-launch CASPT2 follow-up. Automatically picks SS- vs MS-CASPT2 "
                "based on RASSCF n_roots, sets IPEA = 0.25 by default, emits imaginary "
                "shift 0.1 if the RASSCF active-space verdict is 'marginal', mirrors "
                "the RASSCF Frozen vector when non-empty. Short-circuits with verdict="
                "'needs_active_space_refinement' if RASSCF is 'poor' — points at "
                "refine_molcas_active_space first."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "rasscf_output_file": {"type": "string"},
                    "rasorb_file": {"type": ["string", "null"], "default": None},
                    "input_file": {"type": ["string", "null"], "default": None},
                    "output_dir": {"type": ["string", "null"], "default": None},
                    "job_name": {"type": ["string", "null"], "default": None},
                    "variant": {
                        "type": ["string", "null"],
                        "enum": [None, "SS", "MS", "XMS", "RMS", "XDW"],
                        "default": None,
                        "description": "Override CASPT2 variant. Default: SS if n_roots=1, MS if n_roots>1.",
                    },
                    "ipea_shift": {"type": ["number", "null"], "default": None},
                    "real_shift": {"type": ["number", "null"], "default": None},
                    "imaginary_shift": {"type": ["number", "null"], "default": None},
                    "sigma_p_regularization": {"type": ["number", "null"], "default": None},
                    "target_root": {"type": ["integer", "null"], "default": None},
                    "properties": {"type": "boolean", "default": False},
                    "grdt": {"type": "boolean", "default": False, "description": "Emit GRDT for analytic CASPT2 gradients (ALASKA).",},
                    "apptainer_sif": {"type": ["string", "null"], "default": None},
                    "profile": {"type": ["object", "null"], "default": None},
                    "requested_np": {"type": "integer", "default": 1, "minimum": 1},
                },
                "required": ["rasscf_output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "prepare_molcas_excited_states",
            "description": (
                "Multi-state excited-states orchestrator. Generates a full input chain "
                "of SEWARD + SCF + RASSCF over n_singlets singlets + RASSCF over "
                "n_triplets triplets + per-group MS-CASPT2 + optional RASSI for SOC. "
                "Required: at least one of n_singlets / n_triplets > 0. Assumes a "
                "closed-shell singlet SCF starting point (even total electrons). "
                "Inserts the right EMIL `>>COPY $Project.JobIph JOB00X` between "
                "RASSCF blocks so RASSI can find both wave functions."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "atoms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"},
                                "label": {"type": "string"},
                            },
                            "required": ["symbol", "x", "y", "z"],
                        },
                    },
                    "charge": {"type": "integer", "default": 0},
                    "basis": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "object", "additionalProperties": {"type": "string"}},
                        ],
                    },
                    "cas_active_electrons": {"type": "integer"},
                    "cas_active_orbitals": {"type": "integer"},
                    "n_singlets": {"type": "integer", "default": 0, "minimum": 0},
                    "n_triplets": {"type": "integer", "default": 0, "minimum": 0},
                    "method": {"type": "string", "default": "MS-CASPT2"},
                    "compute_soc": {"type": "boolean", "default": False},
                    "properties": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                        "description": "Property labels like 'MLTPL  1' (8-char) for transition dipole moments.",
                    },
                    "title": {"type": ["string", "null"], "default": None},
                    "geometry_units": {"type": "string", "enum": ["angstrom", "bohr"], "default": "angstrom"},
                    "symmetry": {"type": ["string", "null"], "default": None},
                    "n_symmetries": {"type": "integer", "default": 1},
                    "occupied_per_symmetry": {"type": ["array", "null"], "items": {"type": "integer"}, "default": None},
                    "n_basis_per_symmetry": {"type": ["array", "null"], "items": {"type": "integer"}, "default": None},
                    "rasscf_inactive_per_symmetry": {"type": ["array", "null"], "items": {"type": "integer"}, "default": None},
                    "rasscf_active_per_symmetry": {"type": ["array", "null"], "items": {"type": "integer"}, "default": None},
                    "ipea_shift": {"type": "number", "default": 0.25},
                    "imaginary_shift": {"type": "number", "default": 0.1},
                    "inline_basis": {"type": "boolean", "default": True},
                    "memory_mb": {"type": "integer", "default": 4000},
                    "job_name": {"type": ["string", "null"], "default": None},
                    "write_input_to": {"type": ["string", "null"], "default": None},
                    "apptainer_sif": {"type": ["string", "null"], "default": None},
                    "profile": {"type": ["object", "null"], "default": None},
                    "requested_np": {"type": "integer", "default": 1, "minimum": 1},
                },
                "required": ["atoms", "basis", "cas_active_electrons", "cas_active_orbitals"],
                "additionalProperties": False,
            },
        },
        {
            "name": "prepare_molcas_opt_freq_workflow",
            "description": (
                "Geometry-optimization + analytic-frequency orchestrator. Wraps "
                "SEWARD + (SCF on iter 1 only) + RASSCF (if CASSCF) + ALASKA + SLAPAF "
                "in an EMIL `>>> Do while <<<` ... `>>> ENDDO <<<` loop, followed by "
                "MCKINLEY + MCLR for analytic second derivatives. Supports SCF/HF/"
                "CASSCF/RASSCF methods, minimum or transition-state search, frequency-"
                "only single points, numerical-gradient fallback, and iroot picking "
                "for state-averaged frequencies."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "atoms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"},
                                "label": {"type": "string"},
                            },
                            "required": ["symbol", "x", "y", "z"],
                        },
                    },
                    "charge": {"type": "integer", "default": 0},
                    "multiplicity": {"type": "integer", "default": 1, "minimum": 1},
                    "basis": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "object", "additionalProperties": {"type": "string"}},
                        ],
                    },
                    "method": {"type": "string", "enum": ["SCF", "HF", "CASSCF", "RASSCF"], "default": "CASSCF"},
                    "cas_active_electrons": {"type": ["integer", "null"], "default": None},
                    "cas_active_orbitals": {"type": ["integer", "null"], "default": None},
                    "do_optimization": {"type": "boolean", "default": True},
                    "do_frequency": {"type": "boolean", "default": True},
                    "transition_state": {"type": "boolean", "default": False, "description": "Add `TS` to SLAPAF for transition-state search."},
                    "title": {"type": ["string", "null"], "default": None},
                    "geometry_units": {"type": "string", "enum": ["angstrom", "bohr"], "default": "angstrom"},
                    "symmetry": {"type": ["string", "null"], "default": None},
                    "n_symmetries": {"type": "integer", "default": 1},
                    "occupied_per_symmetry": {"type": ["array", "null"], "items": {"type": "integer"}, "default": None},
                    "n_basis_per_symmetry": {"type": ["array", "null"], "items": {"type": "integer"}, "default": None},
                    "rasscf_inactive_per_symmetry": {"type": ["array", "null"], "items": {"type": "integer"}, "default": None},
                    "rasscf_active_per_symmetry": {"type": ["array", "null"], "items": {"type": "integer"}, "default": None},
                    "inline_basis": {"type": "boolean", "default": True},
                    "memory_mb": {"type": "integer", "default": 2000},
                    "max_opt_iterations": {"type": ["integer", "null"], "default": None},
                    "numerical_gradients": {"type": "boolean", "default": False, "description": "Use NumGrad in ALASKA instead of analytic gradients."},
                    "iroot_freq": {"type": ["integer", "null"], "default": None, "description": "Pick which RASSCF root MCLR computes the Hessian for (state-averaged cases)."},
                    "job_name": {"type": ["string", "null"], "default": None},
                    "write_input_to": {"type": ["string", "null"], "default": None},
                    "apptainer_sif": {"type": ["string", "null"], "default": None},
                    "profile": {"type": ["object", "null"], "default": None},
                    "requested_np": {"type": "integer", "default": 1, "minimum": 1},
                },
                "required": ["atoms", "basis"],
                "additionalProperties": False,
            },
        },
        {
            "name": "prepare_molcas_scan_workflow",
            "description": (
                "Constrained-geometry PES scan orchestrator. For each value in "
                "scan_coordinate.values, generates a Molcas input that optimizes "
                "the molecule with the chosen bond/angle/dihedral locked at that "
                "value (via GATEWAY Constraint + SLAPAF). Supports SCF/HF/"
                "CASSCF/RASSCF. Optional orbital chaining: each point after the "
                "first reads the previous point's converged orbitals via FILEORB, "
                "giving faster convergence + smoother PES (no orbital flipping). "
                "Returns N inputs + a sequential launch plan. Warning: scans "
                "that traverse a bent↔linear transition (e.g. H-C-N as r(C-H) "
                "grows) trip SLAPAF's BMtrx error — use angle or non-collinear "
                "scans for those cases."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "atoms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"},
                                "label": {"type": "string"},
                            },
                            "required": ["symbol", "x", "y", "z"],
                        },
                    },
                    "charge": {"type": "integer", "default": 0},
                    "multiplicity": {"type": "integer", "minimum": 1},
                    "basis": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "object", "additionalProperties": {"type": "string"}},
                        ],
                    },
                    "method": {"type": "string", "enum": ["SCF", "HF", "CASSCF", "RASSCF"], "default": "SCF"},
                    "cas_active_electrons": {"type": ["integer", "null"], "default": None},
                    "cas_active_orbitals": {"type": ["integer", "null"], "default": None},
                    "scan_coordinate": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": ["bond", "angle", "dihedral"]},
                            "atom_labels": {
                                "type": "array", "items": {"type": "string"},
                                "description": "Auto-generated atom labels (e.g. ['C1', 'H1']). Use atom_indices instead if you don't know the labels.",
                            },
                            "atom_indices": {
                                "type": "array", "items": {"type": "integer", "minimum": 1},
                                "description": "1-based indices into atoms list. Mutually exclusive with atom_labels.",
                            },
                            "values": {"type": "array", "items": {"type": "number"}, "minItems": 1},
                            "unit": {"type": "string", "enum": ["angstrom", "bohr", "degree", "radian"]},
                        },
                        "required": ["kind", "values"],
                    },
                    "title": {"type": ["string", "null"], "default": None},
                    "geometry_units": {"type": "string", "enum": ["angstrom", "bohr"], "default": "angstrom"},
                    "symmetry": {"type": ["string", "null"], "default": None},
                    "n_symmetries": {"type": "integer", "default": 1},
                    "occupied_per_symmetry": {"type": ["array", "null"], "items": {"type": "integer"}, "default": None},
                    "n_basis_per_symmetry": {"type": ["array", "null"], "items": {"type": "integer"}, "default": None},
                    "rasscf_inactive_per_symmetry": {"type": ["array", "null"], "items": {"type": "integer"}, "default": None},
                    "rasscf_active_per_symmetry": {"type": ["array", "null"], "items": {"type": "integer"}, "default": None},
                    "inline_basis": {"type": "boolean", "default": True},
                    "memory_mb": {"type": "integer", "default": 2000},
                    "max_opt_iterations": {"type": ["integer", "null"], "default": None},
                    "chain_orbitals": {"type": "boolean", "default": True, "description": "If True, each scan point after the first uses FILEORB from the previous point's RasOrb/ScfOrb for warm-start convergence."},
                    "base_job_name": {"type": "string", "default": "scan"},
                    "output_dir": {"type": "string", "default": "."},
                    "apptainer_sif": {"type": ["string", "null"], "default": None},
                    "profile": {"type": ["object", "null"], "default": None},
                    "requested_np": {"type": "integer", "default": 1, "minimum": 1},
                },
                "required": ["atoms", "multiplicity", "basis", "scan_coordinate"],
                "additionalProperties": False,
            },
        },
        {
            "name": "prepare_molcas_irc_workflow",
            "description": (
                "Intrinsic reaction coordinate (IRC) orchestrator. Takes a "
                "converged TS geometry + reaction vector (parsed from a prior "
                "TS opt+freq .log via ts_output_file, OR passed explicitly) and "
                "generates a Molcas input that walks the reaction coordinate in "
                "both directions from the TS until energy rises or NIRC points "
                "are reached. Output: $Project.mep.molden trajectory + per-point "
                "geometries. Validates the TS connects the right reactant + product. "
                "Note: bohr coordinates are recommended (the prior TS log's "
                "'Nuclear coordinates for the next iteration' section is in bohr — "
                "so set geometry_units='bohr' to pass them verbatim)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ts_atoms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"},
                                "label": {"type": "string"},
                            },
                            "required": ["symbol", "x", "y", "z"],
                        },
                    },
                    "charge": {"type": "integer", "default": 0},
                    "multiplicity": {"type": "integer", "minimum": 1},
                    "basis": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "object", "additionalProperties": {"type": "string"}},
                        ],
                    },
                    "method": {"type": "string", "enum": ["SCF", "HF", "CASSCF", "RASSCF"], "default": "SCF"},
                    "cas_active_electrons": {"type": ["integer", "null"], "default": None},
                    "cas_active_orbitals": {"type": ["integer", "null"], "default": None},
                    "reaction_vector": {
                        "type": ["array", "null"], "default": None,
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Explicit Cartesian reaction vector — list of [x, y, z] rows, one per atom in ts_atoms order. Mutually exclusive with ts_output_file.",
                    },
                    "ts_output_file": {
                        "type": ["string", "null"], "default": None,
                        "description": "Path to a prior TS opt+freq .log. The orchestrator parses 'The Cartesian Reaction vector' section. Mutually exclusive with reaction_vector.",
                    },
                    "n_irc_points": {"type": "integer", "default": 20, "minimum": 1, "description": "NIRC — max points per direction."},
                    "irc_step_size": {"type": ["number", "null"], "default": None, "description": "IRCStep — step length in mass-weighted coordinates. Default Molcas value is 0.1 au."},
                    "irc_step_size_unit": {"type": "string", "enum": ["bohr", "angstrom"], "default": "bohr"},
                    "irc_algorithm": {"type": "string", "enum": ["GS", "MB"], "default": "GS"},
                    "title": {"type": ["string", "null"], "default": None},
                    "geometry_units": {"type": "string", "enum": ["angstrom", "bohr"], "default": "angstrom"},
                    "symmetry": {"type": ["string", "null"], "default": None},
                    "n_symmetries": {"type": "integer", "default": 1},
                    "occupied_per_symmetry": {"type": ["array", "null"], "items": {"type": "integer"}, "default": None},
                    "n_basis_per_symmetry": {"type": ["array", "null"], "items": {"type": "integer"}, "default": None},
                    "rasscf_inactive_per_symmetry": {"type": ["array", "null"], "items": {"type": "integer"}, "default": None},
                    "rasscf_active_per_symmetry": {"type": ["array", "null"], "items": {"type": "integer"}, "default": None},
                    "inline_basis": {"type": "boolean", "default": True},
                    "memory_mb": {"type": "integer", "default": 2000},
                    "job_name": {"type": ["string", "null"], "default": None},
                    "write_input_to": {"type": ["string", "null"], "default": None},
                    "apptainer_sif": {"type": ["string", "null"], "default": None},
                    "profile": {"type": ["object", "null"], "default": None},
                    "requested_np": {"type": "integer", "default": 1, "minimum": 1},
                },
                "required": ["ts_atoms", "multiplicity", "basis"],
                "additionalProperties": False,
            },
        },
        {
            "name": "prepare_molcas_atomization",
            "description": (
                "Thick orchestrator for atomization-energy / binding-energy workflows. "
                "Takes a molecule + chemistry context and generates: (1) the molecule "
                "input at consistent CAS (auto-summed from atomic fragments by default), "
                "(2) one input per unique atomic element at its bundled ground state + "
                "recommended CAS, all at uniform theory level (same basis, same DKH "
                "setting). Returns launch plans for every species PLUS a post-hoc "
                "next_actions list calling check_molcas_active_space_consistency + "
                "compute_molcas_reaction_energy. Handles three CrO-class workflow "
                "traps: (a) auto-sums molecule CAS to span atomic fragments; (b) "
                "applies Relativistic R02O02 uniformly when any element needs it (TM "
                "atoms); (c) skips the &SCF block on high-spin TM atoms (Cr ⁷S, Mn ⁶S, "
                "Fe ⁵D, etc.) because Molcas ROHF doesn't converge from GuessOrb for "
                "them. Atomic CAS comes from a bundled ground-state table (Z=1..30)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "atoms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"},
                                "label": {"type": "string"},
                            },
                            "required": ["symbol", "x", "y", "z"],
                        },
                    },
                    "charge": {"type": "integer", "default": 0},
                    "multiplicity": {"type": "integer", "minimum": 1},
                    "basis": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "object", "additionalProperties": {"type": "string"}},
                        ],
                        "description": "Either a single basis name (all elements) or per-element dict like {'Cr': 'ANO-RCC', 'O': 'ANO-S'}.",
                    },
                    "method": {"type": "string", "enum": ["CASSCF", "CASPT2", "MS-CASPT2"], "default": "CASSCF"},
                    "cas_active_electrons": {"type": ["integer", "null"], "default": None, "description": "Molecule CAS active electrons. If null AND cas_active_orbitals is null, auto-summed from atomic fragments."},
                    "cas_active_orbitals": {"type": ["integer", "null"], "default": None},
                    "atomic_cas_strategy": {"type": "string", "enum": ["minimal", "valence"], "default": "minimal", "description": "'minimal' = SOMOs + open valence shell (matches what the molecule usually treats as active); 'valence' = full valence shell."},
                    "relativistic": {"type": "string", "enum": ["auto", "always", "never"], "default": "auto", "description": "'auto' = enable Relativistic R02O02 if any element needs DKH per the bundled table."},
                    "output_dir": {"type": "string", "default": "."},
                    "base_job_name": {"type": ["string", "null"], "default": None},
                    "inline_basis": {"type": "boolean", "default": True},
                    "memory_mb": {"type": "integer", "default": 4000},
                    "title": {"type": ["string", "null"], "default": None},
                    "geometry_units": {"type": "string", "enum": ["angstrom", "bohr"], "default": "angstrom"},
                    "apptainer_sif": {"type": ["string", "null"], "default": None},
                    "profile": {"type": ["object", "null"], "default": None},
                    "requested_np": {"type": "integer", "default": 1, "minimum": 1},
                },
                "required": ["atoms", "multiplicity", "basis"],
                "additionalProperties": False,
            },
        },
        {
            "name": "compute_molcas_reaction_energy",
            "description": (
                "Compute a reaction energy from converged Molcas outputs. "
                "ΔE = Σ_products(coef × E) − Σ_reactants(coef × E). "
                "For atomization / dissociation (1 reactant molecule, N atomic "
                "products) the result is the binding/dissociation energy D_e "
                "(positive = bound). Pair with check_molcas_active_space_consistency "
                "before trusting CASSCF reaction energies on multireference systems."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "products": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "output_file": {"type": "string"},
                                "coefficient": {"type": "number", "default": 1},
                                "label": {"type": "string"},
                            },
                            "required": ["output_file"],
                        },
                    },
                    "reactants": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "output_file": {"type": "string"},
                                "coefficient": {"type": "number", "default": 1},
                                "label": {"type": "string"},
                            },
                            "required": ["output_file"],
                        },
                    },
                    "energy_kind": {
                        "type": "string",
                        "enum": ["primary", "scf", "rasscf", "caspt2", "ms_caspt2", "rassi_sf", "rassi_so"],
                        "default": "primary",
                        "description": "Which energy field to use. 'primary' follows the parser hierarchy (CASPT2 > RASSCF > SCF). For reaction energies, force a consistent level (e.g. 'rasscf' or 'caspt2') across all species.",
                    },
                    "include_thermochem": {
                        "type": "boolean",
                        "default": False,
                        "description": "If True, also compute ΔZPVE, D_0, ΔH(T), ΔG(T), ΔS(T). Pulls ZPVE + thermal corrections from each species' Molcas thermochem block (requires MCLR freq calc). For monoatomic species without parsed thermochem, falls back to ideal-gas Sackur-Tetrode + electronic-degeneracy entropy.",
                    },
                    "temperature_k": {"type": "number", "default": 298.15},
                    "pressure_atm": {"type": "number", "default": 1.0},
                    "label": {"type": ["string", "null"], "default": None},
                },
                "required": ["products", "reactants"],
                "additionalProperties": False,
            },
        },
        {
            "name": "check_molcas_active_space_consistency",
            "description": (
                "Compare a molecule's CAS spec to the sum of its dissociation "
                "fragments' CAS specs. Flags the 'molecule CAS too small to "
                "span fragments' trap — when this is wrong, CASSCF reaction "
                "energies are unphysical (often negative for clearly bound "
                "molecules). Optionally also counts active orbitals with "
                "specific atomic character (e.g. how many Cr 3d orbitals)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "molecule_output": {"type": "string"},
                    "fragments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "output_file": {"type": "string"},
                                "stoichiometry": {"type": "integer", "default": 1},
                                "label": {"type": "string"},
                            },
                            "required": ["output_file"],
                        },
                    },
                    "target_character_atom": {"type": ["string", "null"], "default": None, "description": "If set with target_character_ao, count active orbitals with this character in molecule vs fragments (e.g. 'Cr' + '3d')."},
                    "target_character_ao": {"type": ["string", "null"], "default": None},
                },
                "required": ["molecule_output", "fragments"],
                "additionalProperties": False,
            },
        },
        {
            "name": "apply_molcas_recovery",
            "description": (
                "Apply a recovery fix to a failed Molcas input. Pairs with "
                "suggest_molcas_recovery to close the auto-fix loop. Two ways "
                "to call: pass output_file to auto-classify + apply the fix, "
                "or pass a pre-computed recovery dict. Handles mechanical fixes "
                "(drop &SCF block; bump RASSCF Iteration; add Imaginary 0.1 to "
                "&CASPT2; bump MOLCAS_MEM; replace opening &SEWARD with &GATEWAY; "
                "bump SLAPAF Iterations). Returns verdict=fix_applied with the new "
                "input path + a ready-to-run next_actions chain, OR "
                "verdict=manual_intervention_required for failure classes that "
                "need chemistry judgment (jobiph_missing, ga_segfault, nactel_parity, "
                "seward_angstrom_symmetry — those route back to draft_molcas_input "
                "or prepare_molcas_excited_states)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "output_file": {
                        "type": ["string", "null"], "default": None,
                        "description": "Path to the failed .out/.log — passed through suggest_molcas_recovery to classify. Mutually exclusive with `recovery`.",
                    },
                    "recovery": {
                        "type": ["object", "null"], "default": None,
                        "description": "Pre-computed recovery dict (from suggest_molcas_recovery's `recovery` field). Mutually exclusive with `output_file`.",
                    },
                    "write_to": {
                        "type": ["string", "null"], "default": None,
                        "description": "Output path for the fixed input. Defaults to inserting '_recovered' before the .input suffix.",
                    },
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_molcas_recovery",
            "description": (
                "Classify a Molcas failure (or suspicious success) and emit a "
                "step-by-step recovery plan. Walks a priority-ordered rule engine "
                "against the .out / .log text + parsed metadata; covers 11 failure "
                "modes surfaced by real dogfooding: seward_angstrom_symmetry, "
                "missing_basis_in_loop (Do-while SEWARD without GATEWAY), "
                "scf_single_electron (H-atom-style ROHF abort), scf_no_convergence "
                "(TM atoms from GuessOrb), rasscf_no_convergence (iter budget too "
                "tight), caspt2_intruder (small denominators on diffuse virtuals), "
                "caspt2_low_ref_weight (caution band), jobiph_missing (excited-states "
                "COPY plumbing), ga_segfault (parallel CASPT2 on broken GA build), "
                "memory_exceeded, slapaf_no_convergence, nactel_parity. Returns "
                "failure_class + severity + root_cause + step-by-step fix_recipe + "
                "agent-actionable next_actions chained into the right orchestrator."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "return_all_matches": {
                        "type": "boolean", "default": False,
                        "description": "If True, return ALL matching rules under all_matches (still picks the highest-priority as primary). Debug aid for the rule engine itself.",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        # ----- Documentation -----
        {
            "name": "list_molcas_docs",
            "description": "List bundled OpenMolcas documentation files (programs, tutorials, users_guide, advanced_examples, etc.).",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "search_molcas_docs",
            "description": "Search the bundled OpenMolcas docs for keywords or directives. Returns ranked excerpts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 8},
                    "context_lines": {"type": "integer", "default": 2},
                    "subdir": {"type": ["string", "null"], "default": None, "description": "Restrict to a subdirectory: programs, tutorials, users_guide, advanced_examples, installation, overview."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "lookup_molcas_module",
            "description": "Return the bundled docs page for a Molcas module (e.g. 'rasscf', 'caspt2', 'rassi', 'alaska', 'mclr').",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "module_name": {"type": "string"},
                },
                "required": ["module_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "read_molcas_doc_excerpt",
            "description": "Read an excerpt from a bundled Molcas doc by relative path (e.g. 'programs/caspt2.md') and line range, or around the first match for a query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "doc_name": {"type": "string"},
                    "start_line": {"type": ["integer", "null"], "default": None},
                    "end_line": {"type": ["integer", "null"], "default": None},
                    "query": {"type": ["string", "null"], "default": None},
                    "context_lines": {"type": "integer", "default": 8},
                },
                "required": ["doc_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_molcas_topic_guide",
            "description": (
                "Curated guidance for high-value Molcas topics. Recognized topics: "
                "rasscf_active_space, caspt2_setup, ipea_shift, xms_caspt2, "
                "alaska_gradients, mclr_freq, rassi_state_interaction, inporb_format, "
                "scf_setup. Returns a short summary plus relevant docs excerpts and "
                "the linked module page."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
    ]


# ----- Handlers -----------------------------------------------------------------

@_tool("draft_molcas_input")
def _handle_draft_molcas_input(arguments: dict[str, Any]) -> dict[str, Any]:
    text = _draft_input(arguments)
    issues = _lint_input(text)
    return {
        "input_text": text,
        "lint_issues": issues,
        "n_lint_issues": len(issues),
    }


@_tool("lint_molcas_input")
def _handle_lint_molcas_input(arguments: dict[str, Any]) -> dict[str, Any]:
    text = arguments["input_text"]
    issues = _lint_input(text)
    return {
        "issues": issues,
        "n_issues": len(issues),
        "n_errors": sum(1 for i in issues if i.get("level") == "error"),
        "n_warnings": sum(1 for i in issues if i.get("level") == "warning"),
    }


@_tool("compute_molcas_active_space_partition")
def _handle_compute_active_space_partition(arguments: dict[str, Any]) -> dict[str, Any]:
    return _compute_active_space_partition(
        n_electrons=int(arguments["n_electrons"]),
        cas_active_electrons=int(arguments["cas_active_electrons"]),
        cas_active_orbitals=int(arguments["cas_active_orbitals"]),
        n_symmetries=int(arguments.get("n_symmetries", 1)),
        n_basis_per_symmetry=arguments.get("n_basis_per_symmetry"),
        n_frozen_per_symmetry=arguments.get("n_frozen_per_symmetry"),
        active_per_symmetry=arguments.get("active_per_symmetry"),
        target_symmetry_for_active=arguments.get("target_symmetry_for_active"),
        n_inactive_per_symmetry=arguments.get("n_inactive_per_symmetry"),
        ras1_holes_max=int(arguments.get("ras1_holes_max", 0)),
        ras1_per_symmetry=arguments.get("ras1_per_symmetry"),
        ras3_electrons_max=int(arguments.get("ras3_electrons_max", 0)),
        ras3_per_symmetry=arguments.get("ras3_per_symmetry"),
    )


@_tool("list_molcas_basis_sets")
def _handle_list_molcas_basis_sets(arguments: dict[str, Any]) -> dict[str, Any]:
    basis_name = arguments.get("basis_name")
    element = arguments.get("element")
    if basis_name and element:
        # Return contractions for a specific (basis, element)
        return {
            "basis_name": basis_name,
            "element": element,
            "default_contraction": _default_contraction(basis_name, element),
            "default_label": _basis_label(basis_name, element)
            if _default_contraction(basis_name, element)
            else None,
            "available_contractions": _list_contractions_for(basis_name, element),
        }
    if element:
        # List all bases that support this element
        all_bases = _list_basis_sets()
        supporting = [
            b for b in all_bases if element[0].upper() + element[1:].lower() in _list_elements_for_basis(b)
        ]
        return {
            "element": element,
            "supporting_basis_sets": supporting,
            "n_supporting": len(supporting),
        }
    if basis_name:
        return {
            "basis_name": basis_name,
            "elements": _list_elements_for_basis(basis_name),
            "n_elements": len(_list_elements_for_basis(basis_name)),
        }
    return {"basis_sets": _list_basis_sets(), "n_basis_sets": len(_list_basis_sets())}


@_tool("parse_molcas_output")
def _handle_parse_molcas_output(arguments: dict[str, Any]) -> dict[str, Any]:
    path = arguments["output_file"]
    contents = read_text(path)
    return _parse_output_full(
        path,
        contents,
        parse_mo_coefficients=bool(arguments.get("include_mo_coefficients", False)),
    )


@_tool("parse_molcas_tasks")
def _handle_parse_molcas_tasks(arguments: dict[str, Any]) -> dict[str, Any]:
    path = arguments["output_file"]
    contents = read_text(path)
    return _parse_tasks(path, contents)


@_tool("get_molcas_orbitals")
def _handle_get_molcas_orbitals(arguments: dict[str, Any]) -> dict[str, Any]:
    path = arguments["output_file"]
    contents = read_text(path)
    tasks_result = _parse_tasks(path, contents)
    generic_tasks = tasks_result.get("generic_tasks") or []
    if not generic_tasks:
        return {"error": "no_tasks", "message": f"No tasks found in {path}"}
    task_index = arguments.get("task_index")
    if task_index is None:
        preferred = None
        for i, t in enumerate(generic_tasks):
            if t["extra"]["module"] in {"RASSCF", "SCF"}:
                preferred = i
        if preferred is None:
            return {"error": "no_orbital_task", "message": f"No SCF/RASSCF tasks found in {path}"}
        task_index = preferred
    if task_index < 0 or task_index >= len(generic_tasks):
        return {"error": "task_index_out_of_range", "message": f"task_index={task_index} out of range (have {len(generic_tasks)} tasks)"}
    task = generic_tasks[task_index]
    lines = contents.splitlines()
    block_text = "\n".join(lines[task["line_start"] - 1 : task["line_end"]])
    mo = _parse_last_mo_block(block_text, parse_coefficients=True)
    if mo is None:
        return {
            "error": "no_mo_block",
            "message": f"No '++ Molecular orbitals:' block in task {task_index} ({task['extra']['module']})",
        }
    return {
        "task_index": task_index,
        "module": task["extra"]["module"],
        "mo_block": mo,
    }


@_tool("parse_molcas_frequencies")
def _handle_parse_molcas_frequencies(arguments: dict[str, Any]) -> dict[str, Any]:
    contents = read_text(arguments["output_file"])
    block = _parse_last_freq_block(contents)
    if block is None:
        return {
            "error": "no_freq_block",
            "message": f"No 'Harmonic frequencies in cm-1' block found in {arguments['output_file']}",
        }
    return block


@_tool("parse_molcas_thermochem")
def _handle_parse_molcas_thermochem(arguments: dict[str, Any]) -> dict[str, Any]:
    contents = read_text(arguments["output_file"])
    block = _parse_thermochem_block(contents)
    if block is None:
        return {
            "error": "no_thermochem_block",
            "message": f"No thermochemistry block found in {arguments['output_file']}",
        }
    return block


@_tool("extract_molcas_geometry")
def _handle_extract_molcas_geometry(arguments: dict[str, Any]) -> dict[str, Any]:
    contents = read_text(arguments["output_file"])
    block_index = arguments.get("block_index")
    if block_index is not None:
        blocks = _parse_cartesian_blocks(contents)
        if not blocks:
            return {"error": "no_geometry", "message": f"No 'Cartesian coordinates' blocks in {arguments['output_file']}"}
        if block_index < 0 or block_index >= len(blocks):
            return {
                "error": "block_index_out_of_range",
                "message": f"block_index={block_index} out of range; have {len(blocks)} blocks",
            }
        return blocks[block_index]
    final = _parse_final_geometry(contents)
    if final is None:
        return {"error": "no_geometry", "message": f"No geometry found in {arguments['output_file']}"}
    return final


@_tool("parse_molcas_trajectory")
def _handle_parse_molcas_trajectory(arguments: dict[str, Any]) -> dict[str, Any]:
    contents = read_text(arguments["output_file"])
    return _parse_trajectory(contents)


@_tool("swap_molcas_inporb_orbitals")
def _handle_swap_molcas_inporb_orbitals(arguments: dict[str, Any]) -> dict[str, Any]:
    swaps = [tuple(pair) for pair in arguments["swaps"]]
    return _swap_orbitals_in_inporb(
        arguments["input_orbital_file"],
        arguments["output_orbital_file"],
        swaps=swaps,
        symmetry=int(arguments.get("symmetry", 1)),
    )


@_tool("parse_molcas_rassi")
def _handle_parse_molcas_rassi(arguments: dict[str, Any]) -> dict[str, Any]:
    contents = read_text(arguments["output_file"])
    start = contents.find("--- Start Module: rassi")
    if start == -1:
        return {
            "error": "no_rassi_module",
            "message": f"No '--- Start Module: rassi' marker found in {arguments['output_file']}",
        }
    end = contents.find("--- Stop Module: rassi", start)
    if end == -1:
        end = len(contents)
    return _parse_rassi(contents[start:end])


@_tool("prepare_molcas_launch")
def _handle_prepare_molcas_launch(arguments: dict[str, Any]) -> dict[str, Any]:
    return _prepare_molcas_launch(
        arguments["input_file"],
        profile=arguments.get("profile"),
        requested_np=int(arguments.get("requested_np", 1)),
        job_name=arguments.get("job_name"),
        apptainer_sif=arguments.get("apptainer_sif"),
        extra_env=arguments.get("extra_env"),
    )


@_tool("parse_molcas_inporb")
def _handle_parse_molcas_inporb(arguments: dict[str, Any]) -> dict[str, Any]:
    return _parse_inporb(
        arguments["orbital_file"],
        parse_coefficients=bool(arguments.get("include_coefficients", True)),
    )


@_tool("analyze_molcas_active_space")
def _handle_analyze_molcas_active_space(arguments: dict[str, Any]) -> dict[str, Any]:
    output_file = arguments.get("output_file")
    orbital_file = arguments.get("orbital_file")
    if not output_file and not orbital_file:
        return {
            "error": "missing_input",
            "message": "Provide either output_file or orbital_file.",
        }
    if output_file:
        contents = read_text(output_file)
        full = _parse_output_full(output_file, contents)
        rasscf_payload = next(
            (p["details"] for p in full["task_payloads"] if p["module"] == "RASSCF"),
            None,
        )
        if not rasscf_payload:
            return {
                "error": "no_rasscf_task",
                "message": f"No RASSCF task found in {output_file}; supply orbital_file instead.",
            }
        return _analyze_active_space(rasscf_payload)
    return _analyze_active_space(_parse_inporb(orbital_file, parse_coefficients=False))


@_tool("validate_molcas_caspt2_setup")
def _handle_validate_molcas_caspt2_setup(arguments: dict[str, Any]) -> dict[str, Any]:
    contents = read_text(arguments["output_file"])
    full = _parse_output_full(arguments["output_file"], contents)
    caspt2_payload = next(
        (p["details"] for p in full["task_payloads"] if p["module"] == "CASPT2"),
        None,
    )
    if not caspt2_payload:
        return {
            "error": "no_caspt2_task",
            "message": f"No CASPT2 task found in {arguments['output_file']}",
        }
    return _validate_caspt2_setup(caspt2_payload)


@_tool("suggest_molcas_orbital_swaps")
def _handle_suggest_molcas_orbital_swaps(arguments: dict[str, Any]) -> dict[str, Any]:
    output_file = arguments["output_file"]
    contents = read_text(output_file)
    full = _parse_output_full(output_file, contents)
    rasscf_payload = next(
        (p["details"] for p in full["task_payloads"] if p["module"] == "RASSCF"),
        None,
    )
    if not rasscf_payload:
        return {"error": "no_rasscf_task", "message": f"No RASSCF task in {output_file}"}

    # Slice the RASSCF block and pull the LAST MO block (which has dominant_aos)
    sym = int(arguments.get("symmetry", 1))
    rasscf_task = next(
        (p for p in full["task_payloads"] if p["module"] == "RASSCF"),
        None,
    )
    line_start, line_end = rasscf_task["line_range"]
    lines = contents.splitlines()
    block_text = "\n".join(lines[line_start - 1: line_end])
    mo_block = _parse_last_mo_block(block_text, parse_coefficients=True)
    if mo_block is None:
        return {"error": "no_mo_block", "message": f"No MO block in RASSCF task"}

    return _suggest_swaps_by_character(
        mo_block=mo_block,
        rasscf_orbital_specs=rasscf_payload.get("orbital_specs", {}),
        target_atom_pattern=arguments["target_atom_pattern"],
        target_ao_pattern=arguments["target_ao_pattern"],
        symmetry=sym,
        top_dominant_aos=int(arguments.get("top_dominant_aos", 1)),
    )


@_tool("refine_molcas_active_space")
def _handle_refine_molcas_active_space(arguments: dict[str, Any]) -> dict[str, Any]:
    return _refine_active_space(
        arguments["output_file"],
        target_atom_pattern=arguments["target_atom_pattern"],
        target_ao_pattern=arguments["target_ao_pattern"],
        rasorb_file=arguments.get("rasorb_file"),
        input_file=arguments.get("input_file"),
        output_dir=arguments.get("output_dir"),
        refined_job_name=arguments.get("refined_job_name"),
        apply_swaps=bool(arguments.get("apply_swaps", True)),
        symmetry=int(arguments.get("symmetry", 1)),
        max_swaps=int(arguments.get("max_swaps", 5)),
        apptainer_sif=arguments.get("apptainer_sif"),
        profile=arguments.get("profile"),
        requested_np=int(arguments.get("requested_np", 1)),
    )


@_tool("prepare_molcas_casscf_setup")
def _handle_prepare_molcas_casscf_setup(arguments: dict[str, Any]) -> dict[str, Any]:
    return _prepare_casscf_setup(
        atoms=arguments["atoms"],
        charge=int(arguments.get("charge", 0)),
        multiplicity=int(arguments.get("multiplicity", 1)),
        basis=arguments["basis"],
        title=arguments.get("title"),
        method=arguments.get("method", "CASSCF"),
        geometry_units=arguments.get("geometry_units", "angstrom"),
        chemistry_hint=arguments.get("chemistry_hint"),
        cas_active_electrons=arguments.get("cas_active_electrons"),
        cas_active_orbitals=arguments.get("cas_active_orbitals"),
        program_options=arguments.get("program_options"),
        job_name=arguments.get("job_name"),
        write_input_to=arguments.get("write_input_to"),
        apptainer_sif=arguments.get("apptainer_sif"),
        profile=arguments.get("profile"),
        requested_np=int(arguments.get("requested_np", 1)),
    )


@_tool("prepare_molcas_caspt2_chain")
def _handle_prepare_molcas_caspt2_chain(arguments: dict[str, Any]) -> dict[str, Any]:
    return _prepare_caspt2_chain(
        arguments["rasscf_output_file"],
        rasorb_file=arguments.get("rasorb_file"),
        input_file=arguments.get("input_file"),
        output_dir=arguments.get("output_dir"),
        job_name=arguments.get("job_name"),
        variant=arguments.get("variant"),
        ipea_shift=arguments.get("ipea_shift"),
        real_shift=arguments.get("real_shift"),
        imaginary_shift=arguments.get("imaginary_shift"),
        sigma_p_regularization=arguments.get("sigma_p_regularization"),
        target_root=arguments.get("target_root"),
        properties=bool(arguments.get("properties", False)),
        grdt=bool(arguments.get("grdt", False)),
        apptainer_sif=arguments.get("apptainer_sif"),
        profile=arguments.get("profile"),
        requested_np=int(arguments.get("requested_np", 1)),
    )


@_tool("prepare_molcas_excited_states")
def _handle_prepare_molcas_excited_states(arguments: dict[str, Any]) -> dict[str, Any]:
    return _prepare_excited_states_workflow(
        atoms=arguments["atoms"],
        charge=int(arguments.get("charge", 0)),
        basis=arguments["basis"],
        cas_active_electrons=int(arguments["cas_active_electrons"]),
        cas_active_orbitals=int(arguments["cas_active_orbitals"]),
        n_singlets=int(arguments.get("n_singlets", 0)),
        n_triplets=int(arguments.get("n_triplets", 0)),
        method=arguments.get("method", "MS-CASPT2"),
        compute_soc=bool(arguments.get("compute_soc", False)),
        properties=arguments.get("properties"),
        title=arguments.get("title"),
        geometry_units=arguments.get("geometry_units", "angstrom"),
        symmetry=arguments.get("symmetry"),
        n_symmetries=int(arguments.get("n_symmetries", 1)),
        occupied_per_symmetry=arguments.get("occupied_per_symmetry"),
        n_basis_per_symmetry=arguments.get("n_basis_per_symmetry"),
        rasscf_inactive_per_symmetry=arguments.get("rasscf_inactive_per_symmetry"),
        rasscf_active_per_symmetry=arguments.get("rasscf_active_per_symmetry"),
        ipea_shift=float(arguments.get("ipea_shift", 0.25)),
        imaginary_shift=float(arguments.get("imaginary_shift", 0.1)),
        inline_basis=bool(arguments.get("inline_basis", True)),
        memory_mb=int(arguments.get("memory_mb", 4000)),
        job_name=arguments.get("job_name"),
        write_input_to=arguments.get("write_input_to"),
        apptainer_sif=arguments.get("apptainer_sif"),
        profile=arguments.get("profile"),
        requested_np=int(arguments.get("requested_np", 1)),
    )


@_tool("prepare_molcas_scan_workflow")
def _handle_prepare_molcas_scan_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    return _prepare_scan_workflow(
        atoms=arguments["atoms"],
        charge=int(arguments.get("charge", 0)),
        multiplicity=int(arguments["multiplicity"]),
        basis=arguments["basis"],
        method=arguments.get("method", "SCF"),
        cas_active_electrons=arguments.get("cas_active_electrons"),
        cas_active_orbitals=arguments.get("cas_active_orbitals"),
        scan_coordinate=arguments["scan_coordinate"],
        title=arguments.get("title"),
        geometry_units=arguments.get("geometry_units", "angstrom"),
        symmetry=arguments.get("symmetry"),
        n_symmetries=int(arguments.get("n_symmetries", 1)),
        occupied_per_symmetry=arguments.get("occupied_per_symmetry"),
        n_basis_per_symmetry=arguments.get("n_basis_per_symmetry"),
        rasscf_inactive_per_symmetry=arguments.get("rasscf_inactive_per_symmetry"),
        rasscf_active_per_symmetry=arguments.get("rasscf_active_per_symmetry"),
        inline_basis=bool(arguments.get("inline_basis", True)),
        memory_mb=int(arguments.get("memory_mb", 2000)),
        max_opt_iterations=arguments.get("max_opt_iterations"),
        chain_orbitals=bool(arguments.get("chain_orbitals", True)),
        base_job_name=arguments.get("base_job_name", "scan"),
        output_dir=arguments.get("output_dir", "."),
        apptainer_sif=arguments.get("apptainer_sif"),
        profile=arguments.get("profile"),
        requested_np=int(arguments.get("requested_np", 1)),
    )


@_tool("prepare_molcas_irc_workflow")
def _handle_prepare_molcas_irc_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    return _prepare_irc_workflow(
        ts_atoms=arguments["ts_atoms"],
        charge=int(arguments.get("charge", 0)),
        multiplicity=int(arguments["multiplicity"]),
        basis=arguments["basis"],
        method=arguments.get("method", "SCF"),
        cas_active_electrons=arguments.get("cas_active_electrons"),
        cas_active_orbitals=arguments.get("cas_active_orbitals"),
        reaction_vector=arguments.get("reaction_vector"),
        ts_output_file=arguments.get("ts_output_file"),
        n_irc_points=int(arguments.get("n_irc_points", 20)),
        irc_step_size=arguments.get("irc_step_size"),
        irc_step_size_unit=arguments.get("irc_step_size_unit", "bohr"),
        irc_algorithm=arguments.get("irc_algorithm", "GS"),
        title=arguments.get("title"),
        geometry_units=arguments.get("geometry_units", "angstrom"),
        symmetry=arguments.get("symmetry"),
        n_symmetries=int(arguments.get("n_symmetries", 1)),
        occupied_per_symmetry=arguments.get("occupied_per_symmetry"),
        n_basis_per_symmetry=arguments.get("n_basis_per_symmetry"),
        rasscf_inactive_per_symmetry=arguments.get("rasscf_inactive_per_symmetry"),
        rasscf_active_per_symmetry=arguments.get("rasscf_active_per_symmetry"),
        inline_basis=bool(arguments.get("inline_basis", True)),
        memory_mb=int(arguments.get("memory_mb", 2000)),
        job_name=arguments.get("job_name"),
        write_input_to=arguments.get("write_input_to"),
        apptainer_sif=arguments.get("apptainer_sif"),
        profile=arguments.get("profile"),
        requested_np=int(arguments.get("requested_np", 1)),
    )


@_tool("prepare_molcas_opt_freq_workflow")
def _handle_prepare_molcas_opt_freq_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    return _prepare_opt_freq_workflow(
        atoms=arguments["atoms"],
        charge=int(arguments.get("charge", 0)),
        multiplicity=int(arguments.get("multiplicity", 1)),
        basis=arguments["basis"],
        method=arguments.get("method", "CASSCF"),
        cas_active_electrons=arguments.get("cas_active_electrons"),
        cas_active_orbitals=arguments.get("cas_active_orbitals"),
        do_optimization=bool(arguments.get("do_optimization", True)),
        do_frequency=bool(arguments.get("do_frequency", True)),
        transition_state=bool(arguments.get("transition_state", False)),
        title=arguments.get("title"),
        geometry_units=arguments.get("geometry_units", "angstrom"),
        symmetry=arguments.get("symmetry"),
        n_symmetries=int(arguments.get("n_symmetries", 1)),
        occupied_per_symmetry=arguments.get("occupied_per_symmetry"),
        n_basis_per_symmetry=arguments.get("n_basis_per_symmetry"),
        rasscf_inactive_per_symmetry=arguments.get("rasscf_inactive_per_symmetry"),
        rasscf_active_per_symmetry=arguments.get("rasscf_active_per_symmetry"),
        inline_basis=bool(arguments.get("inline_basis", True)),
        memory_mb=int(arguments.get("memory_mb", 2000)),
        max_opt_iterations=arguments.get("max_opt_iterations"),
        numerical_gradients=bool(arguments.get("numerical_gradients", False)),
        iroot_freq=arguments.get("iroot_freq"),
        job_name=arguments.get("job_name"),
        write_input_to=arguments.get("write_input_to"),
        apptainer_sif=arguments.get("apptainer_sif"),
        profile=arguments.get("profile"),
        requested_np=int(arguments.get("requested_np", 1)),
    )


@_tool("prepare_molcas_atomization")
def _handle_prepare_molcas_atomization(arguments: dict[str, Any]) -> dict[str, Any]:
    return _prepare_atomization_calculation(
        atoms=arguments["atoms"],
        charge=int(arguments.get("charge", 0)),
        multiplicity=int(arguments["multiplicity"]),
        basis=arguments["basis"],
        method=arguments.get("method", "CASSCF"),
        cas_active_electrons=arguments.get("cas_active_electrons"),
        cas_active_orbitals=arguments.get("cas_active_orbitals"),
        atomic_cas_strategy=arguments.get("atomic_cas_strategy", "minimal"),
        relativistic=arguments.get("relativistic", "auto"),
        output_dir=arguments.get("output_dir", "."),
        base_job_name=arguments.get("base_job_name"),
        inline_basis=bool(arguments.get("inline_basis", True)),
        memory_mb=int(arguments.get("memory_mb", 4000)),
        title=arguments.get("title"),
        geometry_units=arguments.get("geometry_units", "angstrom"),
        apptainer_sif=arguments.get("apptainer_sif"),
        profile=arguments.get("profile"),
        requested_np=int(arguments.get("requested_np", 1)),
    )


@_tool("compute_molcas_reaction_energy")
def _handle_compute_molcas_reaction_energy(arguments: dict[str, Any]) -> dict[str, Any]:
    return _compute_reaction_energy(
        products=arguments["products"],
        reactants=arguments["reactants"],
        energy_kind=arguments.get("energy_kind", "primary"),
        label=arguments.get("label"),
        include_thermochem=bool(arguments.get("include_thermochem", False)),
        temperature_k=float(arguments.get("temperature_k", 298.15)),
        pressure_atm=float(arguments.get("pressure_atm", 1.0)),
    )


@_tool("check_molcas_active_space_consistency")
def _handle_check_molcas_active_space_consistency(arguments: dict[str, Any]) -> dict[str, Any]:
    return _check_active_space_consistency(
        molecule_output=arguments["molecule_output"],
        fragments=arguments["fragments"],
        target_character_atom=arguments.get("target_character_atom"),
        target_character_ao=arguments.get("target_character_ao"),
    )


@_tool("suggest_molcas_recovery")
def _handle_suggest_molcas_recovery(arguments: dict[str, Any]) -> dict[str, Any]:
    return _suggest_recovery(
        arguments["output_file"],
        return_all_matches=bool(arguments.get("return_all_matches", False)),
    )


@_tool("apply_molcas_recovery")
def _handle_apply_molcas_recovery(arguments: dict[str, Any]) -> dict[str, Any]:
    return _apply_recovery(
        arguments["input_file"],
        output_file=arguments.get("output_file"),
        recovery=arguments.get("recovery"),
        write_to=arguments.get("write_to"),
    )


@_tool("list_molcas_docs")
def _handle_list_molcas_docs(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"files": _list_docs()}


@_tool("search_molcas_docs")
def _handle_search_molcas_docs(arguments: dict[str, Any]) -> dict[str, Any]:
    return _search_docs(
        arguments["query"],
        max_results=int(arguments.get("max_results", 8)),
        context_lines=int(arguments.get("context_lines", 2)),
        subdir=arguments.get("subdir"),
    )


@_tool("lookup_molcas_module")
def _handle_lookup_molcas_module(arguments: dict[str, Any]) -> dict[str, Any]:
    return _lookup_module_syntax(arguments["module_name"])


@_tool("read_molcas_doc_excerpt")
def _handle_read_molcas_doc_excerpt(arguments: dict[str, Any]) -> dict[str, Any]:
    return _read_doc_excerpt(
        arguments["doc_name"],
        start_line=arguments.get("start_line"),
        end_line=arguments.get("end_line"),
        query=arguments.get("query"),
        context_lines=int(arguments.get("context_lines", 8)),
    )


@_tool("get_molcas_topic_guide")
def _handle_get_molcas_topic_guide(arguments: dict[str, Any]) -> dict[str, Any]:
    return _get_topic_guide(arguments["topic"])

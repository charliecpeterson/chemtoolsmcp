"""NWChem MCP tool schema definitions — the static input-schema data.

Split out of mcp/tools/nwchem.py to keep the handler module navigable. This
is pure data (no logic): _nwchem_tool_definitions() returns the list, which
chemtools.mcp.dispatch aggregates with the other programs' definitions.
"""
from __future__ import annotations

from typing import Any  # noqa: F401  (used in the return type hint)


def _nwchem_tool_definitions() -> list[dict[str, Any]]:
    """NWChem-specific tool definitions.

    Generic (cross-program) tool definitions live in
    ``chemtools/mcp/tools/generic.py`` and are aggregated alongside this
    list by ``chemtools.mcp.dispatch.tool_definitions()``.
    """
    return [
        # ------------------------------------------------------------------
        # Server introspection
        # ------------------------------------------------------------------
        {
            "name": "prepare_nwchem_mcscf_setup",
            "description": (
                "Thick orchestrator for MCSCF / CASSCF active-space setup. Given a "
                "converged SCF reference output, this tool parses the MOs, picks a "
                "recommended CAS(M,N) window, checks frontier orbital character, "
                "and returns a Diagnosis envelope with next_actions so the agent "
                "knows whether to draft MCSCF directly, inspect more orbitals "
                "first, or fix a state mismatch via vectors swap. Companion to "
                "prepare_nwchem_tce_setup for the multireference branch."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scf_output_path": {
                        "type": "string",
                        "description": "Path to the converged SCF (or DFT) reference output.",
                    },
                    "input_path": {
                        "type": "string",
                        "description": "Optional path to the SCF input file. Improves expected_somo_count inference when multiplicity > 1.",
                    },
                    "expected_metal_elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Transition metals expected to host the active electrons.",
                    },
                    "expected_somo_count": {
                        "type": "integer",
                        "description": "Expected number of singly-occupied orbitals; overrides multiplicity-based inference.",
                    },
                    "prefer_expanded": {
                        "type": "boolean",
                        "default": False,
                        "description": "When true, route the agent toward the expanded CAS window (typically larger). Default minimal.",
                    },
                },
                "required": ["scf_output_path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "prepare_nwchem_tce_setup",
            "description": (
                "Thick orchestrator for TCE (CCSD / CCSD(T) / MP2) setup. Given a "
                "converged SCF/DFT output, this tool parses the MOs, computes the "
                "freeze count for the target method, checks orbital ordering, and "
                "suggests any vectors swaps needed before launching TCE. Returns a "
                "Diagnosis envelope with next_actions so a small LLM can chain "
                "swap_nwchem_movecs -> draft_nwchem_tce_input deterministically. "
                "Replaces ~5 manual reasoning steps (parse MOs, decide freeze, "
                "check ordering, suggest swaps, draft input) with a single call."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scf_output_path": {
                        "type": "string",
                        "description": "Path to the converged SCF/DFT output file.",
                    },
                    "target_method": {
                        "type": "string",
                        "default": "ccsd(t)",
                        "description": "TCE method tag: 'ccsd', 'ccsd(t)', 'mp2'.",
                    },
                    "elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional explicit element list. Inferred from MO data when omitted.",
                    },
                    "charge": {"type": "integer", "default": 0},
                    "multiplicity": {"type": "integer", "default": 1},
                    "expected_metal_elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "For open-shell systems: transition metals expected to host the SOMO(s).",
                    },
                    "expected_somo_count": {
                        "type": "integer",
                        "description": "Expected number of singly-occupied orbitals (M-1 where M is multiplicity).",
                    },
                    "ecp_core_electrons": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                        "description": "Per-element ECP core counts, e.g. {'Au': 60}. Adjusts freeze count.",
                    },
                },
                "required": ["scf_output_path"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Generic auto-detect parsers — Phase 4 / 5
        # These dispatch to the appropriate program plugin via
        # registry.resolve(path=output_file). Each program-prefixed sibling
        # (extract_molcas_geometry, parse_nwchem_thermochem, etc.) still
        # exists for callers that want the program-specific shape; the
        # generic version returns whatever the plugin's parser protocol
        # method emits.
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Generic case-analysis / recovery dispatchers (Phase 6a)
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Workflow planning and geometry setup (start here for new jobs)
        # ------------------------------------------------------------------
        {
            "name": "plan_nwchem_workflow",
            "description": (
                "Return a concrete step-by-step tool call plan for a NWChem workflow. "
                "Call this FIRST when starting any new calculation to get the exact sequence "
                "of tools and parameters to use. Eliminates guesswork about workflow order."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "enum": ["opt_freq", "opt_freq_ccsd", "opt_freq_mp2",
                                 "single_point_dft", "single_point_ccsd", "single_point_mp2"],
                        "description": "What you want to compute.",
                    },
                    "elements": {"type": "array", "items": {"type": "string"},
                                 "description": "Element symbols in the molecule, e.g. ['Fe', 'Cl']."},
                    "charge": {"type": "integer"},
                    "multiplicity": {"type": "integer"},
                    "basis": {"type": "string", "description": "Basis set name, e.g. '6-31gs'. If omitted, suggest_basis_set is included as step 1 of the plan."},
                    "method": {"type": "string", "default": "ccsd",
                               "description": "TCE method: 'ccsd', 'mp2', or 'ccsd(t)'."},
                    "xc_functional": {"type": "string", "default": "b3lyp",
                                      "description": "DFT exchange-correlation functional for opt/freq step, e.g. 'b3lyp', 'pbe0', 'm06', 'tpss'. Default: 'b3lyp'."},
                    "has_geometry_file": {"type": "boolean", "default": False},
                    "has_dft_output": {"type": "boolean", "default": False},
                    "has_scf_output": {"type": "boolean", "default": False},
                },
                "required": ["goal", "elements", "charge", "multiplicity"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_atom_input",
            "description": (
                "Generate a NWChem input for a single atom (for atomization energies, ionization "
                "potentials, electron affinities). Automatically looks up the neutral ground-state "
                "multiplicity for common elements (H–Xe plus 5d metals). Use before "
                "compute_reaction_energy to run each atom at the same level of theory as the molecule. "
                "Charged atoms require an explicit multiplicity. Always uses symmetry c1, does not "
                "constrain orbital occupation, and places the atom at the origin. It cannot reproduce "
                "a cataloged f-block configuration without separate occupation control and validation."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "element": {
                        "type": "string",
                        "description": "Element symbol, e.g. 'Fe', 'O', 'C'.",
                    },
                    "basis": {
                        "type": "string",
                        "description": "Basis set name, e.g. '6-31gs', 'def2-tzvp', 'cc-pvtz'.",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["scf", "dft", "mp2"],
                        "default": "scf",
                        "description": "NWChem module to use.",
                    },
                    "charge": {
                        "type": "integer",
                        "default": 0,
                        "description": "Total charge (0 = neutral atom).",
                    },
                    "multiplicity": {
                        "type": "integer",
                        "description": (
                            "Spin multiplicity. Auto-looked-up only for neutral atoms; "
                            "required explicitly when charge is nonzero."
                        ),
                    },
                    "xc_functional": {
                        "type": "string",
                        "default": "m06",
                        "description": "XC functional when method=dft.",
                    },
                    "memory": {"type": "string", "description": "NWChem memory directive, e.g. '2000 mb'."},
                    "start_name": {"type": "string", "description": "NWChem start name. Defaults to '{element}_atom'."},
                    "output_dir": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                    "basis_library": {"type": "string"},
                },
                "required": ["element", "basis"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_nwchem_thermochem",
            "description": (
                "Extract thermochemistry from frequency output: ZPE, H(T), G(T), S, Cv "
                "in Hartree and kcal/mol. Call after check_nwchem_freq_plausibility."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {
                        "type": "string",
                        "description": "Path to the NWChem frequency output file.",
                    },
                    "T": {
                        "type": "number",
                        "description": "Temperature in Kelvin (default 298.15). Note: NWChem computes corrections at the temperature in the input; this is for reporting only.",
                    },
                    "P": {
                        "type": "number",
                        "description": "Pressure in atm (default 1.0). For reporting only.",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "summarize_nwchem_electronic_structure",
            "description": (
                "Produce a compact electronic structure summary from an NWChem output: "
                "HOMO-LUMO gap (Hartree and eV), frontier orbital character, SOMO count, "
                "Mulliken charges and spin densities on metal centers, "
                "spin-state consistency check, and top charge/spin sites. "
                "Use after a DFT or SCF calculation to verify the electronic state is "
                "physically reasonable before proceeding to the next step."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {
                        "type": "string",
                        "description": "Path to the NWChem output file.",
                    },
                    "input_file": {
                        "type": "string",
                        "description": "Optional path to the NWChem input file (used to read charge/multiplicity).",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "track_nwchem_spin_state",
            "description": (
                "Track <S²> and energy across optimization steps to detect spin-state "
                "changes during geometry optimization. Parses per-step DFT energies and "
                "<S²> values, detects discontinuities that suggest spin flips, state "
                "crossings, or broken-symmetry collapse. Reports spin contamination. "
                "Call this after an optimization completes (especially for open-shell "
                "transition-metal or f-element systems) to verify the electronic state "
                "remained consistent throughout."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {
                        "type": "string",
                        "description": "Path to the NWChem optimization output file.",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Frontier orbital / vectors-swap workflow
        # ------------------------------------------------------------------
        {
            "name": "analyze_nwchem_frontier_orbitals",
            "description": "Analyze NWChem frontier orbitals and SOMOs to estimate metal-centered vs ligand-centered open-shell character.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_nwchem_vectors_swaps",
            "description": "Suggest NWChem vectors swap operations to move buried metal-centered orbitals into the SOMO window.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                    "vectors_input": {"type": "string"},
                    "vectors_output": {"type": "string"},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_vectors_swap_input",
            "description": "Draft a NWChem restart input that applies explicit vectors swaps to move buried metal-centered orbitals into the SOMO window.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                    "vectors_input": {"type": "string"},
                    "vectors_output": {"type": "string"},
                    "task_operation": {"type": "string", "default": "energy"},
                    "iterations": {"type": "integer", "default": 500},
                    "smear": {"type": "number", "default": 0.001},
                    "convergence_damp": {"type": "integer", "default": 30},
                    "convergence_ncydp": {"type": "integer", "default": 30},
                    "population_print": {"type": "string", "default": "mulliken"},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "title": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["output_file", "input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_property_check_input",
            "description": "Draft a one-step NWChem property input around a chosen movecs file for Mulliken or Lowdin inspection.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "reference_output_file": {"type": "string"},
                    "vectors_input": {"type": "string"},
                    "vectors_output": {"type": "string"},
                    "property_keywords": {"type": "array", "items": {"type": "string"}},
                    "task_strategy": {"type": "string", "enum": ["auto", "property", "energy"], "default": "auto"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                    "iterations": {"type": "integer", "default": 1},
                    "convergence_energy": {"type": "string", "default": "1e-3"},
                    "smear": {"type": "number"},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "title": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_scf_stabilization_input",
            "description": "Draft a safer SCF stabilization restart input from an existing movecs path when a state-check or SCF retry still fails.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "reference_output_file": {"type": "string"},
                    "vectors_input": {"type": "string"},
                    "vectors_output": {"type": "string"},
                    "task_operation": {"type": "string", "default": "energy"},
                    "iterations": {"type": "integer"},
                    "smear": {"type": "number"},
                    "convergence_damp": {"type": "integer"},
                    "convergence_ncydp": {"type": "integer"},
                    "population_print": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "title": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_optimization_followup_input",
            "description": "Draft a NWChem follow-up input from the last optimized geometry, either to continue optimization or to run frequency only.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "task_strategy": {
                        "type": "string",
                        "enum": ["auto", "optimize_only", "freq_only", "optimize_then_freq"],
                        "default": "auto",
                    },
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "title": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["output_file", "input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "extract_nwchem_geometry",
            "description": (
                "Extract a geometry from a NWChem optimization output as XYZ and NWChem geometry block text. "
                "For converged or incomplete runs returns the last frame; for failed/diverged runs returns the "
                "lowest-energy frame as the best restart guess. Use frame='best' (default) for automatic smart "
                "selection, or 'last', 'first', 'min_energy', or an integer step number for explicit control."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string", "description": "Optional: used to preserve geometry block header/directives"},
                    "frame": {
                        "type": "string",
                        "description": "'best' (smart selection), 'last', 'first', 'min_energy', or integer step",
                        "default": "best",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_cube_input",
            "description": "Draft a NWChem dplot input for orbital, density, or spin-density cube generation from a chosen movecs file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "vectors_input": {"type": "string"},
                    "orbital_vectors": {"type": "array", "items": {"type": "integer"}},
                    "density_modes": {"type": "array", "items": {"type": "string"}},
                    "orbital_spin": {"type": "string", "default": "total"},
                    "extent_angstrom": {"type": "number", "default": 6.0},
                    "grid_points": {"type": "integer", "default": 120},
                    "pyscf_compatible_grid_points": {
                        "type": "integer",
                        "minimum": 20,
                        "maximum": 120,
                        "description": "Derive a PySCF-compatible limitxyz grid from one explicit-unit Cartesian geometry. Overrides the symmetric extent_angstrom/grid_points box.",
                    },
                    "gaussian": {"type": "boolean", "default": True},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "title": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["input_file", "vectors_input"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_frontier_cube_input",
            "description": "Draft a NWChem dplot input for SOMO, HOMO, and LUMO cubes inferred from parsed frontier orbitals.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "vectors_input": {"type": "string"},
                    "include_somos": {"type": "boolean", "default": True},
                    "include_homo": {"type": "boolean", "default": True},
                    "include_lumo": {"type": "boolean", "default": True},
                    "include_density_modes": {"type": "array", "items": {"type": "string"}},
                    "extent_angstrom": {"type": "number", "default": 6.0},
                    "grid_points": {"type": "integer", "default": 120},
                    "pyscf_compatible_grid_points": {
                        "type": "integer",
                        "minimum": 20,
                        "maximum": 120,
                        "description": "Derive a PySCF-compatible limitxyz grid from one explicit-unit Cartesian geometry. Overrides the symmetric extent_angstrom/grid_points box.",
                    },
                    "gaussian": {"type": "boolean", "default": True},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "title": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["output_file", "input_file"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Output parsers
        # ------------------------------------------------------------------
        {
            "name": "parse_nwchem_output",
            "description": "Parse a NWChem output file into structured sections. Defaults to tasks section only. Use sections=['tasks','mos'] for SCF review, add 'freq' for frequency, 'trajectory' for optimization path.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["tasks", "mos", "freq", "mcscf", "population", "trajectory", "tddft"],
                        },
                        "description": "Sections to parse. Omit for tasks-only (default). Each section adds to response size.",
                    },
                    "top_n": {"type": "integer", "default": 5, "description": "Frontier orbitals to return per spin channel (mos section)."},
                    "include_coefficients": {"type": "boolean", "default": False, "description": "Include MO coefficients (mos section). Only use for small systems (<50 basis functions)."},
                    "include_displacements": {"type": "boolean", "default": False},
                    "include_positions": {"type": "boolean", "default": False},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_nwchem_mos",
            "description": "Parse NWChem molecular orbitals. Returns frontier orbital window (SOMOs + top_n around HOMO/LUMO) by default. Set include_all_orbitals=true only when you need to inspect the full orbital spectrum (e.g. to verify core ordering for TCE freeze count — prefer parse_nwchem_movecs for that instead).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "top_n": {"type": "integer", "default": 5},
                    "include_coefficients": {"type": "boolean", "default": False, "description": "Include MO coefficient vectors. Only use for small systems (<50 basis functions) — large systems produce very large responses."},
                    "include_all_orbitals": {"type": "boolean", "default": False},
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_nwchem_mcscf_output",
            "description": "Parse a NWChem MCSCF output for settings, iteration energies, CI convergence, natural occupations, and active-space Mulliken summaries.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_nwchem_population_analysis",
            "description": "Parse NWChem Mulliken and Lowdin population analysis blocks, including total and spin density tables.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_nwchem_scf",
            "description": "Parse the NWChem SCF/DFT iteration table and identify the convergence pattern.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Input inspection and linting
        # ------------------------------------------------------------------
        {
            "name": "inspect_nwchem_input",
            "description": "Inspect a NWChem input file for geometry elements, transition metals, charge, multiplicity, and tasks.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "lint_nwchem_input",
            "description": "Lint a NWChem input for task/module consistency, basis/ECP coverage, and movecs output policy issues.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "library_path": {"type": "string"},
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "find_nwchem_restart_assets",
            "description": "Discover restart-relevant files in a job directory, including movecs, db, xyz, cubes, inputs, and outputs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Runner / job management
        # ------------------------------------------------------------------
        {
            "name": "inspect_nwchem_runner_profiles",
            "description": "List available NWChem runner profiles and their launcher kinds.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "profiles_path": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "launch_nwchem_run",
            "description": (
                "Launch NWChem via a runner profile. auto_watch=true (default) blocks until "
                "completion. Set auto_watch=false for parallel submissions. dry_run=true to preview."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "profile": {"type": "string"},
                    "profiles_path": {"type": "string"},
                    "job_name": {"type": "string"},
                    "resource_overrides": {"type": "object"},
                    "env_overrides": {"type": "object"},
                    "write_script": {"type": "boolean", "default": True},
                    "dry_run": {"type": "boolean", "default": False},
                    "auto_watch": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "When true (default), automatically polls until the job completes. "
                            "For scheduler jobs this blocks until squeue reports a terminal state. "
                            "Set false to return immediately after submission."
                        ),
                    },
                    "auto_register": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "When true (default), auto-registers the run in the SQLite registry. "
                            "If auto_watch is also true, auto-updates status on completion."
                        ),
                    },
                    "campaign_id": {"type": "integer", "description": "Link this run to a campaign (requires auto_register)."},
                    "workflow_id": {"type": "integer", "description": "Link this run to a workflow."},
                    "workflow_step_id": {"type": "string", "description": "Workflow step ID."},
                    "parent_run_id": {"type": "integer", "description": "Previous run in a restart chain."},
                },
                "required": ["input_file", "profile"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_nwchem_run_status",
            "description": (
                "Check the status of a NWChem run. For HPC jobs the scheduler job ID is auto-detected "
                "from {job_name}.jobid alongside the input/output file. Returns scheduler state "
                "(queued/running/completed/failed/cancelled), parsed output outcome, and a compact "
                "progress summary for running jobs."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "error_file": {"type": "string"},
                    "process_id": {"type": "integer"},
                    "profile": {"type": "string"},
                    "job_id": {"type": "string"},
                    "profiles_path": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "watch_nwchem_run",
            "description": (
                "Poll NWChem status until terminal state or timeout. "
                "For HPC jobs, omit timeout_seconds to block until scheduler completion. "
                "Detects output-silent phases (SAD, X2C, freq displacements) as expected. "
                "Only call directly for local runs or jobs launched with auto_watch=false."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "error_file": {"type": "string"},
                    "process_id": {"type": "integer"},
                    "profile": {"type": "string"},
                    "job_id": {"type": "string"},
                    "profiles_path": {"type": "string"},
                    "poll_interval_seconds": {"type": "number", "default": 10.0},
                    "adaptive_polling": {"type": "boolean", "default": True},
                    "max_poll_interval_seconds": {"type": "number", "default": 60.0},
                    "timeout_seconds": {
                        "type": ["number", "null"],
                        "default": 3600.0,
                        "description": "Seconds before timing out. Set null for HPC jobs to wait indefinitely.",
                    },
                    "max_polls": {"type": "integer"},
                    "history_limit": {"type": "integer", "default": 8},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "tail_nwchem_output",
            "description": "Return the tail of a NWChem output file for quick inspection. Capped at 10000 characters.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "lines": {"type": "integer", "default": 30},
                    "max_characters": {"type": "integer", "default": 4000, "maximum": 10000, "description": "Maximum characters to return (hard cap: 10000)."},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "terminate_nwchem_run",
            "description": (
                "Stop a running NWChem job. "
                "For local runs: provide process_id and optionally signal_name (term or kill). "
                "For HPC scheduler jobs: provide job_id and profile (calls scancel/qdel/bkill). "
                "Only call after intervention review has determined the run should stop."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "process_id": {"type": "integer", "description": "Local process ID (direct/local runs)."},
                    "signal_name": {"type": "string", "default": "term", "description": "term or kill (local only)."},
                    "job_id": {"type": "string", "description": "Scheduler job ID (HPC runs)."},
                    "profile": {"type": "string", "description": "Runner profile name (required with job_id)."},
                    "profiles_path": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Run comparison and follow-up
        # ------------------------------------------------------------------
        {
            "name": "compare_nwchem_runs",
            "description": "Compare two NWChem runs by diagnosis, task outcome, and energy change. Optionally writes a follow-up artifact when output_dir is provided.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_output_file": {"type": "string"},
                    "candidate_output_file": {"type": "string"},
                    "reference_input_file": {"type": "string"},
                    "candidate_input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                },
                "required": ["reference_output_file", "candidate_output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "review_nwchem_mcscf_case",
            "description": "Review a NWChem MCSCF run for convergence quality, active-space health, and likely next action.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "review_nwchem_mcscf_followup_outcome",
            "description": "Compare a follow-up MCSCF run against a reference MCSCF run and summarize whether convergence or active-space quality improved.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_output_file": {"type": "string"},
                    "candidate_output_file": {"type": "string"},
                    "reference_input_file": {"type": "string"},
                    "candidate_input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                },
                "required": ["reference_output_file", "candidate_output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "prepare_nwchem_next_step",
            "description": "Diagnose a NWChem output and prepare the most likely next artifact, such as a swap restart or imaginary-mode follow-up inputs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "write_files": {"type": "boolean", "default": False},
                    "include_property_check": {"type": "boolean", "default": True},
                    "include_frontier_cubes": {"type": "boolean", "default": False},
                    "include_density_modes": {"type": "array", "items": {"type": "string"}},
                    "cube_extent_angstrom": {"type": "number", "default": 6.0},
                    "cube_grid_points": {"type": "integer", "default": 120},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Basis and ECP
        # ------------------------------------------------------------------
        {
            "name": "render_nwchem_basis_block",
            "description": "Render an explicit per-element NWChem basis block from the local library. Provide elements list or input_file (geometry source). Set check_only=true to validate existence without rendering.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "basis_name": {"type": "string"},
                    "elements": {"type": "array", "items": {"type": "string"}},
                    "input_file": {"type": "string"},
                    "library_path": {"type": "string"},
                    "block_name": {"type": "string", "default": "ao basis"},
                    "mode": {"type": "string"},
                    "check_only": {"type": "boolean", "default": False},
                },
                "required": ["basis_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "render_nwchem_ecp_block",
            "description": "Render an explicit per-element NWChem ECP block from the local library. Set check_only=true to validate existence without rendering.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ecp_name": {"type": "string"},
                    "elements": {"type": "array", "items": {"type": "string"}},
                    "library_path": {"type": "string"},
                    "check_only": {"type": "boolean", "default": False},
                },
                "required": ["ecp_name", "elements"],
                "additionalProperties": False,
            },
        },
        {
            "name": "render_nwchem_basis_setup",
            "description": "Render mixed per-element NWChem basis and ECP blocks from the local basis library.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "geometry_file": {"type": "string"},
                    "basis_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Per-element basis set assignments, e.g. {\"Fe\": \"def2-svp\", \"Cl\": \"def2-svp\"}. Use suggest_basis_set to get these.",
                    },
                    "ecp_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Per-element ECP assignments for heavy elements (Z>36), e.g. {\"U\": \"def2-ecp\"}. Only needed for post-Kr elements.",
                    },
                    "default_basis": {"type": "string"},
                    "default_ecp": {"type": "string"},
                    "block_name": {"type": "string", "default": "ao basis"},
                    "basis_mode": {"type": "string"},
                    "library_path": {"type": "string"},
                },
                "required": ["geometry_file", "basis_assignments"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Input creation
        # ------------------------------------------------------------------
        {
            "name": "create_nwchem_input",
            "description": "Create a new NWChem input with mixed explicit basis/ECP assignments and automatic movecs output for SCF/DFT.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "geometry_file": {"type": "string"},
                    "basis_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Per-element basis set assignments, e.g. {\"Fe\": \"def2-svp\", \"Cl\": \"def2-svp\"}. Use suggest_basis_set to generate these.",
                    },
                    "ecp_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Per-element ECP assignments for heavy elements (Z>36), e.g. {\"U\": \"def2-ecp\"}.",
                    },
                    "default_basis": {"type": "string"},
                    "default_ecp": {"type": "string"},
                    "block_name": {"type": "string", "default": "ao basis"},
                    "basis_mode": {"type": "string"},
                    "module": {"type": "string"},
                    "task_operation": {"type": "string"},
                    "charge": {"type": "integer"},
                    "multiplicity": {"type": "integer"},
                    "scf_type": {
                        "type": "string",
                        "enum": ["rhf", "uhf", "rohf"],
                        "description": "SCF wavefunction type for module='scf'. 'rohf' required for open-shell (multiplicity>1). Omit for DFT.",
                    },
                    "nopen": {
                        "type": "integer",
                        "description": "Number of open-shell (singly occupied) orbitals for ROHF. Equal to multiplicity-1. Required when scf_type='rohf'.",
                    },
                    "maxiter": {
                        "type": "integer",
                        "description": "Maximum SCF iterations (default: NWChem uses 30). Increase to 100+ for difficult convergence.",
                    },
                    "thresh": {
                        "type": "number",
                        "description": "SCF convergence threshold (e.g. 1e-6). Leave unset for NWChem default.",
                    },
                    "extra_blocks": {"type": "array", "items": {"type": "string"}},
                    "memory": {"type": "string"},
                    "title": {"type": "string"},
                    "start_name": {"type": "string"},
                    "vectors_input": {"type": "string"},
                    "vectors_output": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                    "library_path": {"type": "string"},
                },
                "required": ["geometry_file", "basis_assignments", "module"],
                "additionalProperties": False,
            },
        },
        {
            "name": "create_nwchem_dft_workflow_input",
            "description": "Create a standard NWChem DFT workflow input, such as optimize+freq, with explicit basis/ECP blocks and automatic movecs output.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "geometry_file": {"type": "string"},
                    "basis_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Per-element basis set assignments, e.g. {\"Fe\": \"def2-svp\", \"Cl\": \"def2-svp\"}. Use suggest_basis_set to generate these.",
                    },
                    "ecp_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Per-element ECP assignments for heavy elements (Z>36), e.g. {\"U\": \"def2-ecp\"}.",
                    },
                    "default_basis": {"type": "string"},
                    "default_ecp": {"type": "string"},
                    "block_name": {"type": "string", "default": "ao basis"},
                    "basis_mode": {"type": "string"},
                    "xc_functional": {"type": "string"},
                    "task_operations": {"type": "array", "items": {"type": "string"}},
                    "charge": {"type": "integer"},
                    "multiplicity": {"type": "integer"},
                    "dft_settings": {"type": "array", "items": {"type": "string"}},
                    "extra_blocks": {"type": "array", "items": {"type": "string"}, "description": "Extra NWChem blocks to append. Do NOT use for geometry options — use geometry_options instead."},
                    "geometry_options": {"type": "array", "items": {"type": "string"}, "description": "Geometry block options (e.g. ['noautosym', 'noautoz'])."},
                    "memory": {"type": "string"},
                    "title": {"type": "string"},
                    "start_name": {"type": "string"},
                    "vectors_input": {"type": "string"},
                    "vectors_output": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                    "library_path": {"type": "string"},
                },
                "required": ["geometry_file", "basis_assignments", "xc_functional", "task_operations"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Case analysis and recovery
        # ------------------------------------------------------------------
        {
            "name": "draft_nwchem_pyscf_reference",
            "description": (
                "Extract defensible NWChem input/output evidence into a draft "
                "reference for compare_pyscf_reference_calculation. It never "
                "guesses the PySCF SCF flavour, density fitting, or electron "
                "count: declare those fields explicitly and inspect "
                "missing_required_fields before comparing."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to the NWChem input file.",
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Optional completed NWChem output for SCF outcome and final energy evidence.",
                    },
                    "label": {"type": "string"},
                    "pyscf_method": {
                        "type": "string",
                        "enum": ["rhf", "uhf", "rks", "uks"],
                        "description": "Caller-declared PySCF method corresponding to the NWChem calculation.",
                    },
                    "pyscf_xc": {
                        "type": ["string", "null"],
                        "description": "Caller-declared PySCF XC functional. Required for RKS and UKS; null for RHF and UHF. The NWChem xc line is retained separately as evidence.",
                    },
                    "density_fit": {
                        "type": "boolean",
                        "description": "Caller-declared PySCF density_fit setting.",
                    },
                    "electron_total": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Caller-declared effective electron count. Required because ECPs and center charges can change it.",
                    },
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "run_nwchem_pyscf_matched_reference",
            "description": (
                "Run one bounded PySCF single point from a completed NWChem "
                "input/output pair, then return the strict evidence-only "
                "comparison report. The PySCF method, functional for DFT, "
                "density fitting, and effective electron count are explicit "
                "caller declarations. The tool refuses to start PySCF when "
                "the NWChem reference is incomplete."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "output_file": {"type": "string"},
                    "working_directory": {
                        "type": "string",
                        "description": "Existing local directory for PySCF temporary files.",
                    },
                    "label": {"type": "string"},
                    "pyscf_method": {
                        "type": "string",
                        "enum": ["rhf", "uhf", "rks", "uks"],
                    },
                    "pyscf_xc": {
                        "type": ["string", "null"],
                        "description": "Required caller-declared PySCF functional for RKS and UKS; null for RHF and UHF.",
                    },
                    "density_fit": {"type": "boolean"},
                    "electron_total": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Caller-declared effective electron count.",
                    },
                    "reference_density_cube": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "minLength": 1},
                            "density_value_unit": {
                                "type": "string",
                                "enum": [
                                    "electron_per_bohr3",
                                    "electron_per_angstrom3",
                                ],
                            },
                        },
                        "required": ["path", "density_value_unit"],
                        "additionalProperties": False,
                        "description": "Caller-declared NWChem total-density CUBE. Requires density_cube_grid_points.",
                    },
                    "density_cube_grid_points": {
                        "type": "integer",
                        "minimum": 20,
                        "maximum": 120,
                        "description": "PySCF total-density CUBE grid size. Requires reference_density_cube.",
                    },
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
                    "job_name": {"type": "string", "default": "nwchem_pyscf_match"},
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Render the PySCF launch after validating NWChem evidence, without running it.",
                    },
                },
                "required": [
                    "input_file",
                    "output_file",
                    "working_directory",
                    "pyscf_method",
                    "density_fit",
                    "electron_total",
                ],
                "additionalProperties": False,
            },
        },
        {
            "name": "analyze_nwchem_case",
            "description": "One-shot NWChem case analysis: diagnosis, input lint, restart assets, spin-state review, and next-step planning. Automatically reads the .err file (same basename) for crash classification. Use detail='compact' for the agent-facing triage payload, 'full' for the human-readable summary.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "err_file": {"type": "string", "description": "Path to the .err file. Auto-detected from output_file if omitted."},
                    "library_path": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "detail": {"type": "string", "enum": ["compact", "full"], "default": "compact"},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_nwchem_recovery",
            "description": "Suggest ranked recovery strategies for a failed or suspicious NWChem run. Use mode='scf' for convergence failures, 'state' for wrong electronic state, 'auto' to check both.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                    "mode": {"type": "string", "enum": ["scf", "state", "auto"], "default": "auto"},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # MCSCF
        # ------------------------------------------------------------------
        {
            "name": "suggest_nwchem_mcscf_active_space",
            "description": "Suggest minimal and expanded MCSCF active spaces from the current MO and spin-state picture.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_mcscf_input",
            "description": "Draft a NWChem MCSCF input using the recommended active space and a vectors reordering plan.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "expected_somos": {"type": "integer"},
                    "active_space_mode": {"type": "string", "enum": ["minimal", "expanded"], "default": "minimal"},
                    "vectors_input": {"type": "string"},
                    "vectors_output": {"type": "string"},
                    "state_label": {"type": "string"},
                    "symmetry": {"type": "integer"},
                    "hessian": {"type": "string", "enum": ["exact", "onel"], "default": "exact"},
                    "maxiter": {"type": "integer", "default": 80},
                    "thresh": {"type": "number", "default": 1.0e-5},
                    "level": {"type": "number", "default": 0.6},
                    "lock_vectors": {"type": "boolean", "default": True},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "title": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["reference_output_file", "input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_mcscf_retry_input",
            "description": "Draft a refined NWChem MCSCF retry input after reviewing a failed or stiff prior MCSCF run.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "expected_metals": {"type": "array", "items": {"type": "string"}},
                    "active_space_mode": {"type": "string", "enum": ["auto", "minimal", "expanded"], "default": "auto"},
                    "vectors_input": {"type": "string"},
                    "vectors_output": {"type": "string"},
                    "state_label": {"type": "string"},
                    "symmetry": {"type": "integer"},
                    "hessian": {"type": "string", "enum": ["exact", "onel"]},
                    "maxiter": {"type": "integer"},
                    "thresh": {"type": "number"},
                    "level": {"type": "number"},
                    "lock_vectors": {"type": "boolean", "default": True},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "title": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["output_file", "input_file"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Geometry and frequency plausibility checks
        # ------------------------------------------------------------------
        {
            "name": "check_nwchem_geometry_plausibility",
            "description": (
                "Check whether an optimized geometry is chemically plausible. "
                "Catches clashes, broken bonds, bad angles, and coordination errors. "
                "Run after any optimization."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the NWChem optimisation output."},
                    "input_file": {"type": "string", "description": "Optional: input file for element labels."},
                    "frame": {
                        "type": "string",
                        "description": "'best' (smart selection), 'last', 'first', 'min_energy', or integer step.",
                        "default": "best",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "check_nwchem_freq_plausibility",
            "description": (
                "Check frequency results for plausibility: imaginary mode count, "
                "missing X-H stretches, abnormal ZPE, suspicious frequencies. Run after freq jobs."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the NWChem frequency output."},
                    "input_file": {"type": "string", "description": "Optional: input file for element list."},
                    "expect_minimum": {
                        "type": "boolean",
                        "description": "True (default) if expecting a local minimum (0 imaginary modes). False for TS.",
                        "default": True,
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_nwchem_freq_progress",
            "description": (
                "Report progress of a finite-difference frequency (Hessian) job. "
                "Returns: displacements done vs total, percentage complete, pace (sec/gradient), "
                "estimated remaining time, number of additional 48h runs needed, and fdrst checkpoint info. "
                "Essential for multi-restart freq jobs on HPC with walltime limits."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the NWChem freq output file."},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "prepare_nwchem_freq_restart",
            "description": (
                "Validate that a frequency restart is ready and return a submit-ready report. "
                "Checks: 'restart' keyword in input, .fdrst checkpoint exists, .db exists, "
                "and reports freq progress from the previous output. "
                "Does NOT submit — use launch_nwchem_run after confirming ready_to_restart=true."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the NWChem .nw input file."},
                    "output_file": {"type": "string", "description": "Path to the previous NWChem output file."},
                    "profile": {"type": "string", "description": "Runner profile name for resubmission."},
                },
                "required": ["input_file", "output_file"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Imaginary modes
        # ------------------------------------------------------------------
        {
            "name": "analyze_nwchem_imaginary_modes",
            "description": (
                "Analyze significant imaginary modes in a NWChem frequency output and identify the dominant moving atoms. "
                "Default detail='compact' strips displacement vectors (~3 KB output). Use detail='full' to include "
                "full Cartesian displacements (needed for displace_nwchem_geometry_along_mode)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "significant_threshold_cm1": {"type": "number", "default": 20.0},
                    "top_atoms": {"type": "integer", "default": 4},
                    "detail": {
                        "type": "string",
                        "enum": ["compact", "full"],
                        "default": "compact",
                        "description": "compact: omit displacement vectors (default, much smaller). full: include all Cartesian displacements.",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "displace_nwchem_geometry_along_mode",
            "description": "Generate plus/minus displaced geometries along an imaginary mode from a NWChem output.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "mode_number": {"type": "integer"},
                    "amplitude_angstrom": {"type": "number", "default": 0.15},
                    "significant_threshold_cm1": {"type": "number", "default": 20.0},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_imaginary_mode_inputs",
            "description": "Create plus/minus displaced NWChem input texts by replacing the input geometry along an imaginary mode.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "input_file": {"type": "string"},
                    "mode_number": {"type": "integer"},
                    "amplitude_angstrom": {"type": "number", "default": 0.15},
                    "significant_threshold_cm1": {"type": "number", "default": 20.0},
                    "task_strategy": {
                        "type": "string",
                        "enum": ["auto", "optimize_only", "optimize_then_freq"],
                        "default": "auto",
                    },
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "write_files": {"type": "boolean", "default": False},
                    "noautosym": {"type": "boolean", "default": True},
                    "symmetry_c1": {"type": "boolean", "default": True},
                },
                "required": ["output_file", "input_file"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # TCE (Tensor Contraction Engine) tools
        # ------------------------------------------------------------------
        {
            "name": "parse_nwchem_tce_output",
            "description": (
                "Parse TCE output: method, correlation/total energy, frozen core count, "
                "convergence, and MR diagnostics (T1/D1/T2 from amplitude files if available)."
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
            "name": "parse_nwchem_tce_amplitudes",
            "description": (
                "Compute multireference diagnostics (T1, D1, T2 norm, MR verdict) from "
                "saved TCE amplitude files. Requires 'set tce:save_t T T' in input "
                "(added by draft_nwchem_tce_input). Call after parse_nwchem_tce_output."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {
                        "type": "string",
                        "description": "Path to the NWChem TCE .out file.",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_tce_input",
            "description": (
                "Draft NWChem TCE input (MP2/CCSD/CCSD(T)). Call AFTER SCF: reads orbitals "
                "to set explicit freeze count, auto-detects ECP, checks orbital ordering. "
                "Never uses 'freeze atomic'. Warns if movecs swap is needed first."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scf_output_file": {
                        "type": "string",
                        "description": "Path to the completed SCF output file (must contain MO analysis).",
                    },
                    "input_file": {
                        "type": "string",
                        "description": "Path to the SCF input file (for geometry/ECP metadata).",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["mp2", "ccsd", "ccsd(t)"],
                        "default": "mp2",
                        "description": "TCE correlation method.",
                    },
                    "freeze_count": {
                        "type": "integer",
                        "description": "Override freeze count. If omitted, computed from chemistry + ECP.",
                    },
                    "ecp_core_electrons": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                        "description": (
                            "Override ECP nelec values per element, e.g. {\"Zn\": 10, \"I\": 28}. "
                            "If omitted, nelec is auto-detected from the input file ECP block."
                        ),
                    },
                    "basis_library": {
                        "type": "string",
                        "description": "Path to basis library for ECP nelec lookup (library-assigned ECPs).",
                    },
                    "movecs_file": {
                        "type": "string",
                        "description": "Path to the movecs file. Inferred from SCF output if omitted.",
                    },
                    "swap_pairs": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "description": "List of [i,j] MO swap pairs already applied via swap_nwchem_movecs.",
                    },
                    "start_name": {"type": "string"},
                    "memory": {"type": "string", "default": "2000 mb"},
                    "output_dir": {"type": "string"},
                    "base_name": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["scf_output_file", "input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "draft_nwchem_tce_restart_input",
            "description": (
                "Generate a NWChem TCE restart input for a stalled or timed-out CCSD/MP2 run. "
                "Finds saved amplitude files (.t1amp.* or .t1_copy.*), copies them to the "
                "{start_name}.t1/.t2 restart names, and builds a 'restart' input with "
                "'set tce:read_ta .true.' and 'set tce:save_t T T'. Use when CCSD iterations "
                "stall before convergence (e.g. residual ~0.001 at iter 100). "
                "Returns the restart input text and a report of which amplitude files were copied."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tce_output_file": {
                        "type": "string",
                        "description": "Path to the incomplete TCE output (.out) file.",
                    },
                    "tce_input_file": {
                        "type": "string",
                        "description": "Path to the previous TCE input (.nw). Auto-inferred if omitted.",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "default": 200,
                        "description": "Max CCSD iterations for the restart run.",
                    },
                    "thresh": {
                        "type": "number",
                        "default": 1e-5,
                        "description": "CCSD residual threshold (default 1e-5, 10× looser than NWChem default).",
                    },
                    "copy_amplitudes": {
                        "type": "boolean",
                        "default": True,
                        "description": "If true, copy .t1amp/.t1_copy files to the restart names automatically.",
                    },
                    "output_dir": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "required": ["tce_output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "validate_nwchem_tce_setup",
            "description": (
                "Validate a NWChem TCE input file before submitting. "
                "Catches common errors: missing symmetry c1, wrong SCF reference for open-shell, "
                "'freeze atomic' (forbidden), missing vectors file, freeze count out of range. "
                "Call after draft_nwchem_tce_input and lint_nwchem_input, before launch_nwchem_run."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tce_input_file": {
                        "type": "string",
                        "description": "Path to the NWChem TCE input file to validate.",
                    },
                    "scf_output_file": {
                        "type": "string",
                        "description": "Optional: path to the SCF output that will be used as reference. Used to verify the vectors file exists.",
                    },
                },
                "required": ["tce_input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_nwchem_movecs",
            "description": (
                "Read orbital eigenvalues and occupations from a binary NWChem movecs file. "
                "Use this to inspect orbital ordering BEFORE designing a TCE freeze count. "
                "Returns all MO indices (1-based), energies in Hartree, and occupancies. "
                "Identifies which orbitals are occupied vs virtual."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "movecs_file": {
                        "type": "string",
                        "description": "Path to the binary NWChem .movecs file.",
                    },
                },
                "required": ["movecs_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "compute_nwchem_harmonic_frequencies",
            "description": (
                "Diagonalize a mass-weighted NWChem .hess Hessian and return "
                "vibrational frequencies in cm^-1. Useful for: "
                "(1) verifying a .hess produces the same frequencies an agent "
                "saw in the .out file; (2) re-deriving frequencies with "
                "different isotope labels without re-running NWChem; (3) "
                "detecting imaginary modes when the .out file is unavailable. "
                "Returns frequencies, eigenvalues in atomic units, and explicit "
                "imaginary_modes (frequency < -50 cm^-1, real TS modes — does "
                "not flag numerical near-zero noise from translations/rotations). "
                "Conversion factor verified bit-exact against NWChem's printed "
                "frequencies on water / methane / CO2 / H2O2 TS fixtures."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "hessian_file": {
                        "type": "string",
                        "description": "Path to the .hess file.",
                    },
                    "elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Element symbol per atom in the SAME ORDER as the geometry that produced the Hessian. Length must equal n_atoms.",
                    },
                    "masses_amu": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Optional explicit masses in amu, overriding the NWChem-default-isotope table. Length must equal n_atoms. Useful for deuterium / heavy-isotope studies.",
                    },
                },
                "required": ["hessian_file", "elements"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_nwchem_hessian",
            "description": (
                "Parse an NWChem .hess file (ASCII lower-triangle Cartesian Hessian in "
                "atomic units, Eh/bohr^2). Use this to validate a previous frequency "
                "run's Hessian before seeding a TS optimization (`driver; inhess 2; "
                "reuse <file.hess>`), or to inspect basic Hessian stats. "
                "Returns n_atoms, n_dof, the full symmetric matrix, and quick sanity "
                "stats (max|H|, Frobenius norm, diagonal min/max). For larger systems, "
                "set return_matrix=false to skip the n_dof*n_dof matrix and keep only "
                "the lower-triangle entries + stats."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "hessian_file": {
                        "type": "string",
                        "description": "Path to the .hess file.",
                    },
                    "return_matrix": {
                        "type": "boolean",
                        "default": True,
                        "description": "When true (default), include the full n_dof x n_dof symmetric matrix. Set false for large systems to keep only the flat triangle entries + stats.",
                    },
                },
                "required": ["hessian_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "swap_nwchem_movecs",
            "description": (
                "Swap two MOs in a binary NWChem movecs file. This is the solution when "
                "orbital ordering is wrong for TCE freezing (e.g., O 1s at position 1 "
                "but Zn 3s/3p at positions 2-5). The RTDB is NOT modified, so if SCF "
                "was already converged, NWChem will use the swapped vectors directly on "
                "restart without re-running SCF. Always call parse_nwchem_movecs before "
                "and after to verify the swap worked correctly."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "movecs_file": {
                        "type": "string",
                        "description": "Path to the binary NWChem .movecs file to modify.",
                    },
                    "i": {
                        "type": "integer",
                        "description": "1-based index of the first MO to swap.",
                    },
                    "j": {
                        "type": "integer",
                        "description": "1-based index of the second MO to swap.",
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Output path. If omitted, overwrites the input file in-place.",
                    },
                },
                "required": ["movecs_file", "i", "j"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_nwchem_tce_freeze",
            "description": (
                "Suggest a freeze count for a NWChem TCE calculation from element list and ECP info. "
                "Returns a per-element breakdown and the total freeze count. "
                "This is a starting estimate — the agent must always verify against the actual "
                "SCF orbital eigenvalues using parse_nwchem_movecs or parse_nwchem_mos."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of element symbols (repeats allowed), e.g. [\"C\",\"C\",\"H\",\"H\"].",
                    },
                    "ecp_core_electrons": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                        "description": "ECP nelec values per element, e.g. {\"Zn\": 10}.",
                    },
                },
                "required": ["elements"],
                "additionalProperties": False,
            },
        },
        # ------------------------------------------------------------------
        # Parallel job monitoring
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Session log — running Markdown doc for context preservation
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Input file versioning
        # ------------------------------------------------------------------
        {
            "name": "create_nwchem_input_variant",
            "description": (
                "Create a versioned copy (_v2.nw, _v3.nw) of an input with changes applied. "
                "Keys: 'memory', 'charge', 'mult', 'task', 'block.keyword' (e.g. 'dft.xc')."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_input": {"type": "string", "description": "Path to the original .nw input file."},
                    "changes": {
                        "type": "object",
                        "description": (
                            "Key-value pairs of changes to apply. Keys are directive names "
                            "like 'memory', 'charge', 'mult', 'task', or 'block.keyword' "
                            "patterns like 'dft.iterations', 'dft.xc'."
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why these changes are being made (e.g. 'OOM at 2000 mb on SPR nodes').",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Explicit output path. If omitted, auto-versioned from source.",
                    },
                },
                "required": ["source_input", "changes"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_nwchem_workflow_state",
            "description": (
                "Determine workflow state and return the next tool call to advance it. "
                "Returns state enum + pre-filled next_action. Loop: call → execute next_action → repeat. "
                "input_file is optional (parsed from output echo if missing)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the NWChem .nw input file. Optional — parsed from output echo if missing."},
                    "output_file": {"type": "string", "description": "Path to the NWChem .out output file."},
                    "profile": {"type": "string", "description": "Runner profile name (e.g. 'stampede3_skx')."},
                    "error_file": {"type": "string", "description": "Path to .err file. Auto-derived from output_file if omitted."},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "plan_nwchem_calculation",
            "description": (
                "Generate a step-by-step calculation plan from a pre-baked protocol. "
                "Available protocols: single_point_dft, geometry_opt_dft, thermochem_dft, "
                "opt_then_tce, basis_set_convergence, spin_state_scan. "
                "Returns step IDs, dependencies, and the exact tool calls for each step. "
                "The model follows the plan — no NWChem expertise needed."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the NWChem .nw input file."},
                    "protocol": {
                        "type": "string",
                        "description": "Protocol name.",
                        "enum": ["single_point_dft", "geometry_opt_dft", "thermochem_dft",
                                 "opt_then_tce", "basis_set_convergence", "spin_state_scan"],
                    },
                    "profile": {"type": "string", "description": "Runner profile name."},
                    "output_dir": {"type": "string", "description": "Directory for output files. Defaults to input file directory."},
                    "overrides": {
                        "type": "object",
                        "description": "Optional overrides (e.g. multiplicities for spin_state_scan).",
                        "additionalProperties": True,
                    },
                },
                "required": ["input_file", "protocol"],
                "additionalProperties": False,
            },
        },
        {
            "name": "list_nwchem_protocols",
            "description": (
                "List all available pre-baked calculation protocols with descriptions. "
                "Protocols encode multi-step NWChem workflows (opt→freq, opt→TCE, etc.) "
                "so the model can plan calculations without NWChem expertise."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        # --- Phase 6: Eval + smart input creation ---
        {
            "name": "evaluate_nwchem_case",
            "description": (
                "Evaluate an NWChem test case against expected outcomes. "
                "Reads a case.json file that defines input/output paths and expectations "
                "(failure_class, recommended_next_action, workflow). Returns pass/fail checks. "
                "Use for automated validation of tool quality and regression testing."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string", "description": "Path to a case.json or *.case.json file."},
                },
                "required": ["case_path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "evaluate_nwchem_cases",
            "description": (
                "Batch-evaluate all NWChem test cases in a directory. "
                "Recursively finds case.json files and evaluates each one. "
                "Returns aggregate pass/fail counts and per-case results."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to a directory containing case.json files."},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "create_nwchem_dft_input_from_request",
            "description": (
                "Create an NWChem DFT input from a high-level request, with automatic validation. "
                "Runs review_nwchem_input_request first to check readiness (basis availability, "
                "charge/multiplicity consistency, etc.). If ready, creates the full input with "
                "explicit basis blocks and geometry. Returns ready_to_create=false with guidance "
                "if requirements are missing. Simpler than create_nwchem_input — fewer required params."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "formula": {"type": "string", "description": "Molecular formula (e.g. 'C6H6'). Used to detect elements."},
                    "geometry_file": {"type": "string", "description": "Path to geometry file (.xyz)."},
                    "basis_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Element → basis name mapping.",
                    },
                    "ecp_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "default_basis": {"type": "string", "description": "Default basis for all elements."},
                    "default_ecp": {"type": "string"},
                    "xc_functional": {"type": "string", "description": "DFT functional (e.g. 'b3lyp', 'm06')."},
                    "task_operations": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Operations: ['optimize'], ['optimize', 'freq'], ['energy'].",
                    },
                    "charge": {"type": "integer"},
                    "multiplicity": {"type": "integer"},
                    "dft_settings": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Extra DFT block lines (e.g. ['grid fine', 'convergence energy 1e-8']).",
                    },
                    "extra_blocks": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Extra NWChem blocks to append (e.g. driver block). Do NOT use this for geometry options — use geometry_options instead.",
                    },
                    "geometry_options": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Geometry block options appended to the geometry header line (e.g. ['noautosym', 'noautoz']).",
                    },
                    "memory": {"type": "string", "description": "Memory directive (e.g. 'total 2000 mb')."},
                    "title": {"type": "string"},
                    "start_name": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "write_file": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
        },
        # --- Phase 5: Gap-fill tools ---
        {
            "name": "check_nwchem_spin_charge_state",
            "description": (
                "Validate the spin/charge state from a completed NWChem output. "
                "Checks <S²> vs expected, SOMO count, Mulliken spin density on metals, "
                "and flags spin contamination or wrong-state convergence. "
                "Essential after any open-shell SCF/DFT calculation."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the NWChem output file."},
                    "input_file": {"type": "string", "description": "Path to the NWChem input file."},
                    "expected_metals": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Metal element symbols to check spin density on (e.g. ['Fe', 'Ru']).",
                    },
                    "expected_somos": {
                        "type": "integer",
                        "description": "Expected number of singly-occupied MOs.",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_nwchem_multiplicity_scan",
            "description": (
                "Recommend a spin-multiplicity scan to verify the ground state of an "
                "open-shell metal system. A converged SCF does NOT confirm the spin state — "
                "NWChem converges cleanly to whatever multiplicity it is given, and the wrong "
                "spin state can sit tens of kcal/mol too high with no warning (the only reliable "
                "single-reference test is to run several multiplicities and compare energies). "
                "Returns whether a scan is warranted plus the parity-correct multiplicities to run "
                "(chemistry-aware for d-block via oxidation-state/ligand-field analysis; "
                "parity-window for f-block). Pass input_file to read elements/charge/multiplicity, "
                "or supply them explicitly."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the NWChem .nw input (read for elements/charge/multiplicity)."},
                    "elements": {
                        "type": "array", "items": {"type": "string"},
                        "description": "All element symbols (duplicates OK). Use instead of input_file.",
                    },
                    "charge": {"type": "integer", "description": "Total molecular charge (default 0)."},
                    "multiplicity": {"type": "integer", "description": "Current/requested multiplicity, if known."},
                    "metal_oxidation_states": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                        "description": "Optional metal->oxidation-state map, e.g. {'Fe': 3}, to sharpen d-block candidates.",
                    },
                    "output_dir": {"type": "string", "description": "Directory for the generated scan inputs (defaults to the input file's directory)."},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "inspect_nwchem_geometry",
            "description": (
                "Inspect the geometry from an NWChem input file. Returns atom count, "
                "elements, coordinate format (xyz/zmatrix), symmetry, bond distances, "
                "and detects potential issues (close contacts, missing atoms). "
                "Use before running a calculation to verify the geometry is reasonable."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the NWChem input file."},
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_nwchem_tasks",
            "description": (
                "Parse task boundaries and energies from NWChem output. "
                "Returns each task's module, operation, energy, status, and timing. "
                "Useful for multi-task outputs (e.g. opt followed by freq)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the NWChem output file."},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "parse_nwchem_trajectory",
            "description": (
                "Parse the optimization trajectory from NWChem output. "
                "Returns per-step energies, gradients, step sizes, and convergence criteria. "
                "Optionally includes atomic positions at each step. "
                "Use to understand optimization progress and convergence behavior."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the NWChem output file."},
                    "include_positions": {
                        "type": "boolean", "default": False,
                        "description": "Include atomic positions at each step (verbose).",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "review_nwchem_input_request",
            "description": (
                "Pre-flight review of input parameters before creating an NWChem input. "
                "Validates basis/element compatibility, checks charge/multiplicity, "
                "suggests corrections. Call this before create_nwchem_input to catch errors early."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "formula": {"type": "string", "description": "Molecular formula (e.g. 'C6H6')."},
                    "geometry_file": {"type": "string", "description": "Path to geometry file (.xyz)."},
                    "basis_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Element → basis name mapping.",
                    },
                    "ecp_assignments": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "default_basis": {"type": "string"},
                    "default_ecp": {"type": "string"},
                    "module": {"type": "string", "default": "dft", "description": "Calculation module (scf, dft, tce)."},
                    "task_operations": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Operations to perform (e.g. ['optimize', 'freq']).",
                    },
                    "functional": {"type": "string", "description": "DFT functional (e.g. 'b3lyp')."},
                    "charge": {"type": "integer"},
                    "multiplicity": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "review_nwchem_progress",
            "description": (
                "Check the progress of a running or recently completed NWChem job. "
                "Parses the output file, detects slow phases, reports convergence progress, "
                "and estimates remaining time. Works with both local and HPC jobs."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the NWChem output file."},
                    "input_file": {"type": "string"},
                    "error_file": {"type": "string", "description": "Path to stderr file (.err)."},
                    "process_id": {"type": "integer", "description": "PID for local jobs."},
                    "profile": {"type": "string", "description": "Runner profile name for HPC jobs."},
                    "job_id": {"type": "string", "description": "Scheduler job ID."},
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "summarize_nwchem_output",
            "description": (
                "Generate a compact summary of an NWChem output file. "
                "Returns key results: final energy, convergence status, "
                "spin state, geometry quality, and any warnings. "
                "Lighter than analyze_nwchem_case — good for quick checks."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string", "description": "Path to the NWChem output file."},
                    "input_file": {"type": "string"},
                    "expected_metals": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "expected_somos": {"type": "integer"},
                    "detail": {
                        "type": "string", "enum": ["summary", "full"], "default": "summary",
                        "description": "Level of detail.",
                    },
                },
                "required": ["output_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "summarize_nwchem_outputs",
            "description": (
                "Triage MANY NWChem outputs in one call. Give a directory, a glob, a single "
                "file, or a list; returns one compact row per file (method, stage, status, "
                "energy, failure_class, verdict, one-line headline) plus roll-up counts by "
                "verdict and failure_class. Use this instead of calling analyze_nwchem_case / "
                "summarize_nwchem_output once per file when assessing a batch — it is far "
                "cheaper and gives the whole picture at once. Drill into individual flagged "
                "files with analyze_nwchem_case afterward."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "A directory, a glob (e.g. 'runs/*.out'), or a single .out file."},
                    "paths": {"type": "array", "items": {"type": "string"}, "description": "Explicit list of paths (alternative to 'path')."},
                    "pattern": {"type": "string", "default": "*.out", "description": "Glob pattern when 'path' is a directory."},
                    "recursive": {"type": "boolean", "default": False, "description": "Recurse into subdirectories when 'path' is a directory."},
                    "limit": {"type": "integer", "description": "Cap the number of files processed (the response flags truncation)."},
                },
                "additionalProperties": False,
            },
        },
        # --- Phase 3: Campaign / scale management ---
        {
            "name": "register_nwchem_run",
            "description": (
                "Legacy: register a new NWChem run in the persistent run "
                "registry (pre-fills program='nwchem'). Equivalent to "
                "register_run(..., program='nwchem'). New code should use "
                "register_run directly so the registry stays program-aware."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string", "description": "Job name (e.g. 'mol_opt')."},
                    "input_file": {"type": "string"},
                    "output_file": {"type": "string"},
                    "profile": {"type": "string", "description": "Runner profile name."},
                    "method": {"type": "string", "description": "E.g. 'DFT', 'CCSD(T)'."},
                    "functional": {"type": "string"},
                    "basis": {"type": "string"},
                    "n_atoms": {"type": "integer"},
                    "elements": {"type": "array", "items": {"type": "string"}},
                    "charge": {"type": "integer"},
                    "multiplicity": {"type": "integer"},
                    "mpi_ranks": {"type": "integer"},
                    "campaign_id": {"type": "integer", "description": "Link to a campaign."},
                    "workflow_id": {"type": "integer", "description": "Link to a workflow."},
                    "workflow_step_id": {"type": "string", "description": "Step ID within a workflow."},
                    "parent_run_id": {"type": "integer", "description": "Previous run in a restart chain."},
                    "tags": {"type": "object", "description": "Arbitrary metadata."},
                },
                "required": ["job_name"],
                "additionalProperties": False,
            },
        },
        # --- Generic registry tools (program-agnostic, Phase 4b/c) ---
        {
            "name": "update_nwchem_run_status",
            "description": (
                "Legacy alias for `update_run_status`. New code should call "
                "`update_run_status` directly — the underlying behavior is "
                "program-agnostic (selects by run_id)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer", "description": "The run_id from register_nwchem_run."},
                    "status": {"type": "string", "enum": ["submitted", "running", "completed", "failed", "timelimited", "oom", "cancelled"]},
                    "energy_hartree": {"type": "number"},
                    "h_hartree": {"type": "number", "description": "Enthalpy H(T) in Hartree."},
                    "g_hartree": {"type": "number", "description": "Gibbs G(T) in Hartree."},
                    "imaginary_modes": {"type": "integer"},
                    "walltime_used_sec": {"type": "number"},
                    "sec_per_gradient": {"type": "number"},
                    "output_file": {"type": "string"},
                },
                "required": ["run_id", "status"],
                "additionalProperties": False,
            },
        },
        {
            "name": "list_nwchem_runs",
            "description": (
                "Legacy: list registered runs. Auto-filters to program='nwchem' "
                "when no explicit program is given (preserves the historical "
                "NWChem-only return)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "campaign_id": {"type": "integer"},
                    "workflow_id": {"type": "integer"},
                    "status": {"type": "string"},
                    "method": {"type": "string"},
                    "program": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "get_nwchem_run_summary",
            "description": "Legacy alias for `get_run_summary`.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer"},
                    "job_name": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "create_nwchem_campaign",
            "description": "Legacy alias for `create_campaign`. Campaigns are cross-program by design.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Unique campaign name."},
                    "description": {"type": "string"},
                    "tags": {"type": "object"},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_nwchem_campaign_status",
            "description": "Legacy alias for `get_campaign_status`.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "campaign_id": {"type": "integer"},
                    "name": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "get_nwchem_campaign_energies",
            "description": "Legacy alias for `get_campaign_energies`.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "campaign_id": {"type": "integer"},
                    "name": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "create_nwchem_workflow",
            "description": "Legacy alias for `create_workflow`.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Workflow name."},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "depends_on": {"type": "string"},
                                "input_file": {"type": "string"},
                                "profile": {"type": "string"},
                                "auto_input": {"type": "object"},
                            },
                            "required": ["id"],
                        },
                    },
                    "protocol": {"type": "string"},
                    "campaign_id": {"type": "integer"},
                },
                "required": ["name", "steps"],
                "additionalProperties": False,
            },
        },
        {
            "name": "advance_nwchem_workflow",
            "description": "Legacy alias for `advance_workflow`.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "integer", "description": "The workflow ID."},
                },
                "required": ["workflow_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "generate_nwchem_input_batch",
            "description": (
                "Generate multiple NWChem inputs by varying parameters from a template. "
                "Supports scanning over charge, multiplicity, task, memory, or any block.keyword "
                "(e.g. dft.xc for functionals). Generates all combinations (Cartesian product). "
                "Optionally registers all generated inputs in a campaign."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "template_input": {"type": "string", "description": "Path to the base .nw file."},
                    "vary": {
                        "type": "object",
                        "additionalProperties": {"type": "array"},
                        "description": "Parameters to vary. Keys are param names, values are lists. E.g. {'charge': [0,1,2], 'mult': [1,3,5]}.",
                    },
                    "output_dir": {"type": "string", "description": "Directory to write generated inputs."},
                    "naming_pattern": {"type": "string", "description": "Filename pattern. Placeholders: {stem}, {idx}, plus vary keys. Default: {stem}_{key}_{value}"},
                    "campaign_id": {"type": "integer", "description": "Register generated inputs in this campaign."},
                },
                "required": ["template_input", "vary", "output_dir"],
                "additionalProperties": False,
            },
        },
        {
            "name": "check_nwchem_memory_fit",
            "description": (
                "Check whether an NWChem input's memory directive fits the target node. "
                "Reads the 'memory total' line from the input file, multiplies by MPI ranks, "
                "and compares against the node's physical RAM (from the runner profile). "
                "Returns warnings if the job would crash with MA_init out-of-memory errors, "
                "and suggests a safe memory value. IMPORTANT: call this before launching, "
                "especially when switching profiles or changing node counts."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the .nw input file."},
                    "profile": {"type": "string", "description": "Runner profile name (reads node_memory_mb from it)."},
                    "profiles_path": {"type": "string", "description": "Custom profiles file path."},
                    "nodes": {"type": "integer", "description": "Override number of nodes."},
                    "mpi_ranks": {"type": "integer", "description": "Override total MPI ranks."},
                    "node_memory_mb": {"type": "integer", "description": "Override node RAM in MB (bypasses profile lookup)."},
                    "resource_overrides": {"type": "object", "description": "Resource overrides to merge into profile resources (e.g. from suggest_nwchem_resources)."},
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "estimate_nwchem_freq_walltime",
            "description": (
                "Estimate walltime needed for a numerical frequency calculation. "
                "NWChem numerical frequencies require 6*N_atoms gradient evaluations. "
                "CRITICAL: NWChem CANNOT checkpoint mid-frequency — if the job exceeds "
                "walltime, ALL progress is lost. This tool estimates total time and "
                "recommends multi-node scaling if the job won't fit in the walltime limit. "
                "Call this BEFORE launching any frequency job to avoid wasting compute time."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_atoms": {"type": "integer", "description": "Number of atoms in the molecule."},
                    "seconds_per_displacement": {"type": "number", "description": "Measured seconds per displacement (from a prior run). If omitted, uses heuristic estimate."},
                    "n_displacements": {"type": "integer", "description": "Total displacements (default: 6 * n_atoms for central differences)."},
                    "mpi_ranks": {"type": "integer", "description": "MPI ranks per node (default: 1)."},
                    "nodes": {"type": "integer", "description": "Number of nodes (default: 1)."},
                    "max_walltime_hours": {"type": "number", "description": "Maximum walltime in hours (default: 48)."},
                },
                "required": ["n_atoms"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_nwchem_resources",
            "description": (
                "Recommend HPC resources (nodes, ranks, walltime, memory) for a NWChem job. "
                "Returns resource_overrides ready for launch_nwchem_run. Call before launching."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the NWChem .nw input file."},
                    "profile": {"type": "string", "description": "Runner profile name (e.g. 'stampede3_skx')."},
                    "profiles_path": {"type": "string", "description": "Optional path to runner profiles YAML/JSON."},
                },
                "required": ["input_file", "profile"],
                "additionalProperties": False,
            },
        },
        {
            "name": "detect_nwchem_hpc_accounts",
            "description": (
                "Detect available HPC allocation accounts for a runner profile. "
                "Runs the profile's account_command (e.g. /usr/local/etc/taccinfo on TACC) "
                "to discover project names, available SUs, and expiration dates. "
                "Returns the recommended account (most SUs available) ready to use in "
                "resource_overrides. Automatically called by suggest_nwchem_resources "
                "when account is not set, but can also be called standalone to check "
                "allocation status."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "profile": {"type": "string", "description": "Runner profile name (e.g. 'stampede3_skx')."},
                    "profiles_path": {"type": "string", "description": "Optional path to runner profiles YAML/JSON."},
                },
                "required": ["profile"],
                "additionalProperties": False,
            },
        },
        {
            "name": "suggest_nwchem_partition",
            "description": (
                "Auto-select the best HPC partition by comparing all profiles on memory, "
                "walltime, SU cost, and queue availability. Returns resource_overrides for launch."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the NWChem .nw input file."},
                    "profiles_path": {"type": "string", "description": "Optional path to runner profiles YAML/JSON."},
                    "check_queue": {"type": "boolean", "default": True, "description": "If true, run sinfo to check partition availability and idle nodes."},
                },
                "required": ["input_file"],
                "additionalProperties": False,
            },
        },
        # ----- NWChem documentation tools (bundled docs) --------------------
        {
            "name": "list_nwchem_docs",
            "description": "List available bundled NWChem documentation files.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "search_nwchem_docs",
            "description": "Search the bundled NWChem documentation for syntax, keywords, or option details. Returns ranked excerpts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (keywords, directives, options)."},
                    "max_results": {"type": "integer", "default": 8, "description": "Maximum results to return."},
                    "context_lines": {"type": "integer", "default": 2, "description": "Lines of context around each match."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "lookup_nwchem_block_syntax",
            "description": "Look up NWChem input block syntax (e.g. scf, dft, mcscf, tce, vectors, geometry) from bundled docs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "block_name": {"type": "string", "description": "Block name to look up (e.g. 'scf', 'dft', 'tce')."},
                    "max_results": {"type": "integer", "default": 6},
                },
                "required": ["block_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "find_nwchem_examples",
            "description": "Search bundled NWChem example/tutorial documentation for a topic (e.g. fragment guess, mcscf, tce, dft).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to find examples for."},
                    "max_results": {"type": "integer", "default": 6},
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
        {
            "name": "read_nwchem_doc_excerpt",
            "description": "Read an excerpt from a bundled NWChem doc file by filename and line range, or around the first match for a query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "doc_name": {"type": "string", "description": "Doc filename (e.g. '11_QuantumMechanicalMethods.pdf.txt')."},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "query": {"type": "string", "description": "Find first occurrence of this text and show context around it."},
                    "context_lines": {"type": "integer", "default": 8},
                },
                "required": ["doc_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_nwchem_topic_guide",
            "description": "Get a curated documentation guide for a common NWChem topic: scf_open_shell, mcscf, fragment_guess, or tce.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic name (scf_open_shell, mcscf, fragment_guess, tce)."},
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
        # ----- NWChem community forum search --------------------------------
        {
            "name": "search_nwchem_forum",
            "description": (
                "Search the archived NWChem community forums for threads matching a query. "
                "Use this when encountering unusual NWChem errors, edge-case behavior, or "
                "issues that may have been discussed by the community. Fetches forum pages "
                "at runtime (requires internet). Returns matching thread titles, URLs, and "
                "optionally the thread content."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms (e.g. 'CCSD convergence', 'DFT grid error', 'segfault GA').",
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum threads to return (default 5).",
                    },
                    "fetch_content": {
                        "type": "boolean",
                        "default": True,
                        "description": "If true, fetch and include thread content (slower but more useful). If false, return titles and URLs only.",
                    },
                    "subforums": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict search to specific subforums. Options: 'Running NWChem', 'NWChem functionality', 'General Topics', 'Compiling NWChem', 'QM/MM'. Default: all.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    ]

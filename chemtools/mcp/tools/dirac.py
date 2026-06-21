"""DIRAC MCP tool definitions and handlers.

Importing this module registers every @_tool handler with the shared
`_TOOL_REGISTRY` in chemtools.mcp.decorator. The accompanying
`dirac_tool_definitions()` is appended into the multi-program tool list
exposed by `chemtools/mcp/tools/nwchem.py:tool_definitions()`.

Phase DC ships the read-only tool surface:

  Input / mol parsing:
    parse_dirac_input              parse a `.inp` job-control file into structured form
    parse_dirac_mol                parse a `.mol` geometry+basis file

  Output parsing:
    parse_dirac_output             single-pass parse of a DIRAC text output
    parse_dirac_scf_iterations     SCF iteration trace (energy / gradient / DIIS)
    parse_dirac_symmetry           per-irrep orbital counts + detected point group

  HDF5 checkpoint readers:
    read_dirac_h5_metadata         DIRAC version + n_atoms + symmetry + MO counts
    read_dirac_h5_geometry         atoms + nuclear charges from .h5 (bohr)
    read_dirac_orbitals            per-MO summary: index / fsym / irrep / E / occ / shell class
    read_dirac_mo_coefficients     MO coefficients (full or by index list)

  Open-shell + diagnostic:
    analyze_dirac_open_shell           cross-checks input AOC spec vs h5 orbital occupations
    analyze_dirac_open_shell_quality   deep open-shell verdict: character + energy clustering
                                       + chemistry-hint match (Phase DD)
    parse_dirac_vecpop                 per-MO j-character + AO populations from VECPOP block
    suggest_dirac_orbital_swaps        find spinor candidates with target character (Phase DD)
    summarize_dirac_run                high-level rollup of a DIRAC text+h5 pair

  MO reordering (Phase DE):
    draft_dirac_reorder_block          render a `.REORDER MO` block from per-ircop orders
    apply_dirac_reorder_to_input       insert/replace `.REORDER MO` in an existing .inp
    parse_dirac_reorder_block          extract any existing `.REORDER MO` spec

  Input drafting + atomic-start (Phase DF):
    draft_dirac_input                  render a `.inp` from a structured spec
    draft_dirac_mol                    render a `.mol` from atoms + per-element basis
    prepare_dirac_atomic_start         orchestrator: per-element atom jobs → molecule with --copy

  Launcher (Phase DG):
    prepare_dirac_launch               build the apptainer + pam-dirac command line

  Cm-class actinide workflow (Phase DI):
    prepare_dirac_cm_class_workflow    multi-step plan for hard actinides
                                       (Cm/Bk/Cf/Es/Fm/Md/No/Lr) where
                                       .KPSELE alone doesn't converge

  Core-ionization workflow (Phase DJ):
    prepare_dirac_core_ionization      ΔSCF plan for 1s ionization
                                       potentials (.REORDER + .OVLSEL +
                                       .NODYNSEL + .OPENFAC + --put)
    compute_dirac_core_ip              IP = E(ionized) - E(neutral) in
                                       Hartree + eV

  Documentation:
    list_dirac_docs                enumerate bundled DIRAC docs (180 files)
    search_dirac_docs              substring search across the doc corpus
    lookup_dirac_section           best doc match for a section / keyword name
    read_dirac_doc_excerpt         pull a slice of a bundled doc
    get_dirac_topic_guide          curated guides: aoc, cosci, reorder, atomic_start, ecp, checkpoint
"""

from __future__ import annotations

from typing import Any

from chemtools.mcp.decorator import _tool as _raw_tool


def _tool(name: str, *, needs: str = "none", program: str = "dirac"):
    """Program-scoped @_tool wrapper for DIRAC. Set program='generic' on the
    decorator call to register a cross-program tool here."""
    return _raw_tool(name, needs=needs, program=program)


# Make sure the DIRAC plugin is registered when this module is loaded.
import chemtools.programs.dirac  # noqa: F401,E402

from chemtools.programs.dirac.parse import (  # noqa: E402
    parse_output as _parse_output,
    parse_inp as _parse_inp,
    parse_mol as _parse_mol,
    parse_scf_iterations as _parse_scf_iters,
    parse_symmetry as _parse_symmetry,
)
from chemtools.programs.dirac.parse.output import (  # noqa: E402
    parse_spinor_spectrum as _parse_spinor_spectrum,
    parse_cosci_energies as _parse_cosci_energies,
)
from chemtools.programs.dirac.basis import (  # noqa: E402
    list_basis_sets as _list_basis_sets,
    suggest_basis as _suggest_basis,
)
from chemtools.programs.dirac.binary import (  # noqa: E402
    read_metadata as _h5_metadata,
    read_geometry as _h5_geometry,
    read_orbital_summary as _h5_orbitals,
    read_mo_coefficients as _h5_mo_coeffs,
    H5PY_AVAILABLE as _H5PY_AVAILABLE,
)
from chemtools.programs.dirac.docs import (  # noqa: E402
    list_docs as _list_docs,
    search_docs as _search_docs,
    lookup_section as _lookup_section,
    read_doc_excerpt as _read_doc_excerpt,
    get_topic_guide as _get_topic_guide,
)
from chemtools.programs.dirac.parse.vecpop import (  # noqa: E402
    parse_vecpop as _parse_vecpop,
)
from chemtools.programs.dirac.strategy.open_shell import (  # noqa: E402
    analyze_open_shell_quality as _analyze_open_shell_quality,
    suggest_orbital_swaps as _suggest_orbital_swaps,
)
from chemtools.programs.dirac.strategy.reorder import (  # noqa: E402
    draft_reorder_block as _draft_reorder_block,
    apply_reorder_to_input as _apply_reorder_to_input,
    parse_reorder_block as _parse_reorder_block,
)
from chemtools.programs.dirac.input.inp import draft_inp as _draft_inp  # noqa: E402
from chemtools.programs.dirac.input.mol import draft_mol as _draft_mol  # noqa: E402
from chemtools.programs.dirac.input.atomic_start import (  # noqa: E402
    prepare_atomic_start as _prepare_atomic_start,
    prepare_x2c_bootstrap as _prepare_x2c_bootstrap,
)
from chemtools.programs.dirac.input.cm_class import (  # noqa: E402
    prepare_cm_class_workflow as _prepare_cm_class_workflow,
    is_cm_class as _is_cm_class,
)
from chemtools.programs.dirac.input.core_ionization import (  # noqa: E402
    prepare_core_ionization as _prepare_core_ionization,
    compute_core_ip as _compute_core_ip,
)
from chemtools.programs.dirac.runtime import (  # noqa: E402
    prepare_launch as _prepare_launch,
)
from chemtools.programs.dirac.scheduler import (  # noqa: E402
    launch_dirac_run as _launch_dirac_run,
    get_dirac_run_status as _get_dirac_run_status,
    watch_dirac_run as _watch_dirac_run,
    terminate_dirac_run as _terminate_dirac_run,
)


def dirac_tool_definitions() -> list[dict[str, Any]]:
    return [
        # ----- Input / mol parsing -----
        {
            "name": "parse_dirac_input",
            "description": (
                "Parse a DIRAC ``.inp`` job-control file (**SECTION / .KEYWORD format) "
                "into a structured dict. Returns nested sections + boolean shortcuts "
                "(has_scf, has_dft, has_open_shell, has_closed_shell, has_mp2, has_cosci, "
                "has_reorder, has_ecp). Tolerant of section name variants like "
                "WAVE FUNCTION / WAVE FUNCTIONS / WAVE F."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"input_file": {"type": "string"}},
                "required": ["input_file"],
            },
        },
        {
            "name": "parse_dirac_mol",
            "description": (
                "Parse a DIRAC ``.mol`` geometry+basis file. Returns atomtype blocks "
                "(nuclear_charge, atoms, large_basis, small_basis), flat atoms list "
                "(coordinates in bohr or angstrom per the .mol header's `A` flag), "
                "symmetry generators, and basis_assignments by Z."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"mol_file": {"type": "string"}},
                "required": ["mol_file"],
            },
        },

        # ----- Output parsing -----
        {
            "name": "parse_dirac_output",
            "description": (
                "Single-pass parse of a DIRAC text output. Extracts: SCF iteration trace, "
                "total energy, detected tasks (scf/dft/mp2/cosci/krci/ccsd/response), "
                "symmetry detection + per-irrep orbital counts, AOC open-shell setup "
                "(.CLOSED SHELL + .OPEN SHELL blocks), per-symmetry HOMO/LUMO blocks "
                "from RESOLVE, Mulliken population (per-atom totals + per-spinor detail "
                "when .VECPOP was active), spinor eigenvalue spectrum (index, energy, "
                "occupation, j_label, m_j — electronic spinors only, positronic stripped), "
                "COSCI state energies in eV + cm-1 (when COSCI ran). "
                "Cheap enough to fit in agent context."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"output_file": {"type": "string"}},
                "required": ["output_file"],
            },
        },
        {
            "name": "parse_dirac_scf_iterations",
            "description": (
                "Extract just the SCF iteration trace. One dict per ``It. N`` line with "
                "iter, energy_hartree, delta_e, gradient_max, step_size, diis_n."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"output_file": {"type": "string"}},
                "required": ["output_file"],
            },
        },
        {
            "name": "parse_dirac_symmetry",
            "description": (
                "Extract the detected point-group elements + per-irrep orbital counts "
                "(total, large-component, small-component) from a DIRAC output. "
                "D2h gives 8 irreps; the point_group_elements list is the generators "
                "DIRAC printed (subset of X, Y, Z, inversion)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"output_file": {"type": "string"}},
                "required": ["output_file"],
            },
        },

        # ----- HDF5 checkpoint -----
        {
            "name": "read_dirac_h5_metadata",
            "description": (
                "Read top-level DIRAC HDF5 metadata: program version, n_atoms, "
                "fermion-symmetry counts (n_fsym), per-fsym MO counts, "
                "per-fsym basis dim, per-fsym positive-energy orbital counts, "
                "inversion symmetry, quaternion factor (nz). Requires the h5py "
                "package (install with `pip install chemtools[dirac]`)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"h5_file": {"type": "string"}},
                "required": ["h5_file"],
            },
        },
        {
            "name": "read_dirac_h5_geometry",
            "description": (
                "Read molecule geometry from a DIRAC .h5 checkpoint. Coordinates "
                "are in bohr per DIRAC convention; element symbols looked up from "
                "the nuclear charge table. Requires h5py."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"h5_file": {"type": "string"}},
                "required": ["h5_file"],
            },
        },
        {
            "name": "read_dirac_orbitals",
            "description": (
                "Read per-MO data from a DIRAC .h5 checkpoint: global_index, "
                "fermion_symmetry, positive_energy_index (matches RESOLVE output), "
                "irrep, energy_hartree, occupation, shell_class (closed / open / "
                "virtual / negative_energy). Filters: include_negative_energy "
                "(default False — drop positronic states), only_occupied "
                "(drop virtuals), fractional_only (return only AOC fractional-"
                "occupation orbitals — the open-shell electrons)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "h5_file": {"type": "string"},
                    "include_negative_energy": {"type": "boolean", "default": False},
                    "only_occupied": {"type": "boolean", "default": False},
                    "fractional_only": {"type": "boolean", "default": False},
                },
                "required": ["h5_file"],
            },
        },
        {
            "name": "read_dirac_mo_coefficients",
            "description": (
                "Read MO coefficient arrays from a DIRAC .h5 checkpoint. Returns "
                "shape (n_mo, n_basis, nz). With ``mo_indices`` (list of global "
                "MO indices), slices just those rows. Large arrays — call with "
                "indices when possible. Requires h5py."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "h5_file": {"type": "string"},
                    "mo_indices": {
                        "type": "array", "items": {"type": "integer"},
                        "description": "Global MO indices to extract; omit for full array.",
                    },
                },
                "required": ["h5_file"],
            },
        },

        # ----- Strategy + diagnostic -----
        {
            "name": "analyze_dirac_open_shell",
            "description": (
                "Cross-check a DIRAC open-shell setup: parse the .inp's .OPEN SHELL "
                "spec, read the converged .h5 orbital occupations, verify which "
                "spinors actually carry the fractional occupation, flag any "
                "mismatch between requested and observed open shells. Returns "
                "the AOC config, the observed fractionally-occupied orbitals, "
                "and a verdict (consistent / mismatch / converged_to_closed_shell)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "h5_file":    {"type": "string"},
                },
                "required": ["input_file", "h5_file"],
            },
        },
        {
            "name": "summarize_dirac_outputs",
            "description": (
                "Triage MANY DIRAC outputs in one call. Give a directory, a glob, a single "
                "file, or a list; returns one compact row per file (tasks, total energy, "
                "SCF convergence, symmetry, verdict, headline) plus roll-up counts by verdict. "
                "Use this instead of calling summarize_dirac_run once per file when assessing "
                "a batch; drill into flagged runs with summarize_dirac_run afterward."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "A directory, a glob (e.g. 'runs/*.out'), or a single output file."},
                    "paths": {"type": "array", "items": {"type": "string"}, "description": "Explicit list of paths (alternative to 'path')."},
                    "pattern": {"type": "string", "default": "*.out", "description": "Glob pattern when 'path' is a directory."},
                    "recursive": {"type": "boolean", "default": False, "description": "Recurse into subdirectories."},
                    "limit": {"type": "integer", "description": "Cap files processed (response flags truncation)."},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "summarize_dirac_run",
            "description": (
                "High-level rollup of a DIRAC run: parses the .out + (if available) "
                "the .h5, returns method/task list, total energy, SCF convergence, "
                "symmetry, AOC open-shell summary, and a one-line diagnosis. "
                "The agent's recommended first call when reviewing a DIRAC run."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "h5_file":     {"type": "string"},
                },
                "required": ["output_file"],
            },
        },

        # ----- Phase DD: VECPOP + deep open-shell + swap suggester -----
        {
            "name": "parse_dirac_vecpop",
            "description": (
                "Parse the DIRAC VECPOP / Mulliken-per-MO block. Returns one entry "
                "per electronic eigenvalue inside each fermion ircop, with energy, "
                "occupation, j-character (s 1/2, p 3/2, d 3/2, d 5/2, f 5/2, f 7/2, "
                "etc.), m_j quantum number, and per-AO gross populations. Requires "
                "**ANALYZE / .MULPOP plus *MULPOP / .VECPOP in the input."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"output_file": {"type": "string"}},
                "required": ["output_file"],
            },
        },
        {
            "name": "parse_dirac_spinor_spectrum",
            "description": (
                "Extract the electronic spinor eigenvalue spectrum from a DIRAC "
                "output that used .MULPOP + .VECPOP. Returns one entry per "
                "electronic spinor (positronic spinors > 37500 Ha stripped) with: "
                "index, energy_hartree, occupation (0 or 1), j_label (s 1/2, "
                "p 1/2, p 3/2, d 3/2, d 5/2, f 5/2, f 7/2, ...), mj, "
                "angular_momentum (s/p/d/f/g). "
                "Useful for identifying valence vs core spinors, checking "
                "actinide 5f orbital energies, or building correlation windows "
                "for COSCI / KRCI. "
                "Use parse_dirac_output for the full parse including this field; "
                "this tool is for when you only need the spectrum and want to "
                "avoid loading the full output dict."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "occupied_only": {
                        "type": "boolean", "default": False,
                        "description": "If True, return only spinors with occupation > 0.5.",
                    },
                    "energy_range": {
                        "type": "array", "items": {"type": "number"},
                        "description": "[e_min, e_max] in Hartree. Filter spinors to this range.",
                    },
                },
                "required": ["output_file"],
            },
        },
        {
            "name": "parse_dirac_cosci_energies",
            "description": (
                "Extract COSCI state energies from a DIRAC output. "
                "COSCI (Complete Open-Shell CI) prints a table:\n\n"
                "    Obtained COSCI states are as follows:\n"
                "    1    0.000    0.000    1 1 1 1 0 0 ...\n"
                "    2    0.097  781.4    1 1 1 1 0 0 ...\n\n"
                "Returns: n_states, states (state index, energy_ev relative to "
                "ground, energy_cm1 relative to ground, spinor_occupations list), "
                "ground_energy_hartree (SCF energy before COSCI). "
                "Returns None / empty if no COSCI output found. "
                "Typical use: spin-orbit splitting of open-shell actinides and "
                "p-block radicals."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"output_file": {"type": "string"}},
                "required": ["output_file"],
            },
        },
        {
            "name": "list_dirac_basis_sets",
            "description": (
                "List Dyall relativistic basis families available in DIRAC 25, "
                "with optional filtering and a recommendation.\n\n"
                "Families: dyall.2zp / 3zp / 4zp (DFT, most compact), "
                "dyall.v2z / v3z / v4z / v5z (valence, correlated), "
                "dyall.cv2z-cv5z (core-valence, NMR/EFG/core-IP), "
                "dyall.ae2z-ae5z (all-electron, full-core correlation), "
                "dyall.av*/acv*/aae* (augmented with diffuse — NOT available "
                "for lanthanides/actinides).\n\n"
                "Key rules:\n"
                "- Actinides (Ac-Lr, Z=89-103): dyall.2zp for AOC/SCF/DFT; "
                "  dyall.v2z/v3z for correlated (COSCI/KRCI/KR-CCSD); "
                "  NO diffuse families available.\n"
                "- Lanthanides (La-Lu): same coverage as actinides.\n"
                "- d-block: all families including augmented.\n"
                "- s/p-block: all families including augmented.\n"
                "Returns families list + recommendation for the given context."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "element": {
                        "type": "string",
                        "description": "Element symbol (e.g. 'Cm', 'U', 'Fe'). "
                                       "If given, marks availability per element and "
                                       "adds caveats for f-block.",
                    },
                    "family_type": {
                        "type": "string",
                        "description": "Filter by type: 'valence', 'dft', "
                                       "'core-valence', 'all-electron'. Prefix match.",
                    },
                    "zeta": {
                        "type": "integer",
                        "description": "Filter by zeta level (2, 3, 4, 5).",
                    },
                    "calc_type": {
                        "type": "string",
                        "description": "Filter to families suitable for this "
                                       "purpose: 'scf', 'dft', 'aoc', 'correlated', "
                                       "'cc', 'ci', 'core_ip', 'nmr', 'efg', "
                                       "'anion', 'benchmark'.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "suggest_dirac_basis",
            "description": (
                "Return a ranked Dyall basis recommendation for an element + "
                "calculation type. One best pick + alternatives + rationale + "
                "caveats.\n\n"
                "Examples:\n"
                "  element='Cm', calc_type='aoc'         → dyall.2zp\n"
                "  element='Cm', calc_type='correlated'  → dyall.v2z\n"
                "  element='U',  calc_type='nmr'         → dyall.cv2z (default zeta)\n"
                "  element='Fe', calc_type='anion'       → dyall.av2z\n"
                "  element='La', calc_type='anion'       → dyall.v3z "
                "(no diffuse for f-block; larger valence instead)\n"
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "element": {"type": "string"},
                    "calc_type": {
                        "type": "string",
                        "description": "Purpose: 'scf', 'dft', 'aoc', 'correlated', "
                                       "'cc', 'ci', 'cosci', 'krci', 'core_ip', "
                                       "'nmr', 'efg', 'anion', 'benchmark'.",
                    },
                    "zeta": {
                        "type": "integer",
                        "description": "Preferred zeta level. If omitted, defaults to 2.",
                    },
                },
                "required": ["element"],
            },
        },
        {
            "name": "analyze_dirac_open_shell_quality",
            "description": (
                "Deep open-shell quality verdict combining VECPOP j-character "
                "with energy-clustering checks. Returns verdict (healthy / "
                "caution / problematic) + issues list. Optionally pass "
                "``expected_character`` as a chemistry hint (``actinide_5f``, "
                "``valence_d``, ``valence_p``, etc.) or as an explicit list of "
                "j-character strings (e.g. ``['f 5/2', 'f 7/2']``) — mismatches "
                "are flagged as problematic findings with hints to run "
                "suggest_dirac_orbital_swaps. Checks open-shell energy lies "
                "between highest closed and lowest virtual in each ircop."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "expected_character": {
                        "description": (
                            "Chemistry hint or explicit list. Hints: "
                            "actinide_5f, lanthanide_4f, valence_d, valence_f, "
                            "transition_metal_d, valence_p, single_unpaired_s. "
                            "Or a list like ['f 5/2', 'f 7/2']."
                        ),
                    },
                },
                "required": ["output_file"],
            },
        },
        {
            "name": "suggest_dirac_orbital_swaps",
            "description": (
                "Find DIRAC MO swap candidates when the open shell has wrong "
                "character. Walks VECPOP, identifies current open-shell MOs "
                "whose character is NOT in target_character, then finds "
                "virtual / closed MOs with the target character IN THE SAME "
                "fermion ircop (.REORDER MO only reorders within an ircop). "
                "Returns one of four verdicts:\n"
                "  - no_action_needed: open shell already has target character\n"
                "  - swaps_available: actionable intra-ircop swaps; "
                "``per_ircop_orders`` carries the pre-rendered .REORDER spec "
                "ready to feed straight into apply_dirac_reorder_to_input\n"
                "  - parity_incompatible: wrong-character open is in one "
                "parity (e.g. ungerade/E1u) but target character is in the "
                "other (gerade/E1g). .REORDER cannot bridge parities — agent "
                "must redraft .OPEN SHELL spec via draft_dirac_input instead\n"
                "  - no_candidates_found: target character not present "
                "anywhere in the electronic spectrum (basis too small, or "
                "chemistry hint is wrong)\n"
                "Each ircop entry also reports parity (gerade/ungerade), "
                "the suggested (wrong, candidate) eigenvalue-index pairings, "
                "and the rendered reorder_spec string."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {"type": "string"},
                    "target_character": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of j-character strings the open shell SHOULD have, e.g. ['f 5/2', 'f 7/2'].",
                    },
                    "n_candidates": {"type": "integer", "default": 6},
                },
                "required": ["output_file", "target_character"],
            },
        },

        # ----- Phase DE: MO reordering -----
        {
            "name": "draft_dirac_reorder_block",
            "description": (
                "Render a ``.REORDER MO`` block from per-ircop order strings. "
                "Pass ``per_ircop_orders`` as a list of strings — one per "
                "fermion ircop. Each string uses DIRAC's range syntax: "
                "comma-separated indices with ``a..b`` ranges and ``..oo`` for "
                "the remainder. Example: ``['1..8,10,9', '1..oo']`` swaps MOs "
                "9 and 10 in ircop 1, leaves ircop 2 unchanged."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "per_ircop_orders": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["per_ircop_orders"],
            },
        },
        {
            "name": "apply_dirac_reorder_to_input",
            "description": (
                "Insert (or replace) a ``.REORDER MO`` block in a DIRAC ``.inp`` "
                "text. Locates **WAVE FUNCTION (any variant). Prefers placement "
                "under *SCF when that subsection exists, else inserts at the "
                "end of **WAVE FUNCTION. If a ``.REORDER`` already exists, set "
                "``replace=True`` to overwrite it cleanly; otherwise returns "
                "verdict=already_present."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file":       {"type": "string"},
                    "per_ircop_orders": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "replace":          {"type": "boolean", "default": False},
                    "output_path":      {
                        "type": "string",
                        "description": "Optional — if provided, write the patched text here. Otherwise just return patched_text in the payload.",
                    },
                },
                "required": ["input_file", "per_ircop_orders"],
            },
        },
        {
            "name": "parse_dirac_reorder_block",
            "description": (
                "Extract any existing ``.REORDER MO`` block from a DIRAC ``.inp``. "
                "Returns ``{ircop_orders: [...], line: int}`` or null."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"input_file": {"type": "string"}},
                "required": ["input_file"],
            },
        },

        # ----- Phase DF: input + mol drafters + atomic-start orchestrator -----
        {
            "name": "draft_dirac_input",
            "description": (
                "Render a DIRAC ``.inp`` file from a structured spec. Supported "
                "wave_function values: scf, dft, mp2, ccsd, cosci. Spec keys: "
                "title, wave_function, analyze (list, e.g. ['mulpop']), "
                "properties (bool), hamiltonian (e.g. {'x2c': True, 'spinfree': "
                "True, 'dft_functional': 'B3LYP', 'ecp': True, 'amfi': True}), "
                "integrals (e.g. {'uncontract': True}), scf "
                "({closed_shell: [n_fsym1, n_fsym2], open_shell: [{n_electrons, "
                "spinors}], reorder: ['1..oo', '1..oo'], max_iter, resolve}), "
                "extra_sections (list of [name, body] for power users). "
                "Returns the text — caller writes to disk."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec":        {"type": "object"},
                    "output_path": {"type": "string", "description": "Optional — write the text here."},
                },
                "required": ["spec"],
            },
        },
        {
            "name": "draft_dirac_mol",
            "description": (
                "Render a DIRAC ``.mol`` file from a structured spec. Required: "
                "atoms (list of {label, x, y, z, [nuclear_charge | element]}). "
                "Optional: basis ({element_symbol_or_Z: basis_name} mapping), "
                "default_basis, units ('bohr'|'angstrom'), title, symmetry "
                "('auto'|'C1'|list of generators). Returns the text — caller "
                "writes to disk."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "atoms":         {"type": "array", "items": {"type": "object"}},
                    "basis":         {"type": "object"},
                    "default_basis": {"type": "string"},
                    "units":         {"type": "string", "default": "bohr"},
                    "title":         {"type": "string"},
                    "symmetry":      {"description": "'auto', 'C1', or list of generators."},
                    "output_path":   {"type": "string"},
                },
                "required": ["atoms"],
            },
        },
        {
            "name": "prepare_dirac_atomic_start",
            "description": (
                "**Thick orchestrator** for the atomic-start workflow: for each "
                "unique element in ``molecule_atoms``, build a per-atom .inp + "
                ".mol using the element's ground-state AOC config (table covers "
                "H..Zn + key lanthanides + actinides through Cm); also build "
                "the molecular .inp + .mol; return a launch plan listing each "
                "atomic job in order followed by the molecule with the right "
                "``--copy=`` files. The agent runs each atomic job, copies its "
                ".h5 to ``<Element>.h5`` in the molecule's run directory, then "
                "launches the molecule with prepare_dirac_launch. Used to seed "
                "difficult heavy-element / open-shell molecular SCFs."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "molecule_atoms": {"type": "array", "items": {"type": "object"}},
                    "basis":          {"type": "object"},
                    "default_basis":  {"type": "string"},
                    "hamiltonian":    {"type": "object"},
                    "integrals":      {"type": "object"},
                    "use_x2c":        {"type": "boolean", "default": False,
                                       "description": "Use X2C approximation. Default False (full 4c Dirac-Coulomb). Set True only when explicitly needed; X2C has convergence issues for Z≥96 in DIRAC 25."},
                    "output_dir":     {"type": "string"},
                    "molecule_name":  {"type": "string", "default": "molecule"},
                    "molecule_scf":   {"type": "object"},
                    "molecule_units": {"type": "string", "default": "bohr"},
                    "write_files":    {
                        "type": "boolean", "default": False,
                        "description": "If True, write each .inp/.mol in the plan to disk under output_dir.",
                    },
                },
                "required": ["molecule_atoms"],
            },
        },

        # ----- 4c → X2C bootstrap workflow -----
        {
            "name": "prepare_dirac_x2c_bootstrap",
            "description": (
                "Two-step plan for bootstrapping X2C convergence from 4-component "
                "Dirac-Coulomb orbitals. Useful when X2C alone oscillates at a "
                "wrong fixed-point (e.g. Cm, Bk, Cf in DIRAC 25 + dyall.2zp).\n\n"
                "Step 1: Converge a full 4c atomic SCF, save orbitals via --outcmo.\n"
                "Step 2: Run X2C atomic SCF using --incmo from the 4c checkpoint.\n\n"
                "Hypothesis: the 4c orbitals are a good enough starting guess for "
                "X2C that the SCF escapes the wrong fixed-point. If successful, "
                "X2C energy will be close to the 4c energy (within ~1 Ha; different "
                "Hamiltonians so not exact). This would enable X2C for production "
                "molecular work on heavy actinides without the cost of full 4c.\n\n"
                "Returns plan (2 steps) + next_actions with the correct --outcmo / "
                "--incmo pam flags to thread between steps."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "element":       {"type": "string", "description": "Element symbol (e.g. 'Cm')."},
                    "basis":         {"type": "object"},
                    "default_basis": {"type": "string"},
                    "hamiltonian":   {"type": "object"},
                    "integrals":     {"type": "object"},
                    "output_dir":    {"type": "string"},
                },
                "required": ["element"],
            },
        },

        # ----- Phase DJ: ΔSCF core-ionization workflow -----
        {
            "name": "prepare_dirac_core_ionization",
            "description": (
                "Build the ΔSCF launch plan for 1s core ionization "
                "potentials (X-ray photoemission spectroscopy). Drafts a "
                "neutral SCF + one core-ionized SCF per target atom. The "
                "ionized inputs use .REORDER to move the target 1s out "
                "of the closed-shell range, .OPEN SHELL with 1 electron "
                "in 2 spinors, .OPENFAC 1.0, .OVLSEL + .NODYNSEL to "
                "lock the core hole to the right spinor (overlap-based "
                "selection prevents collapse to ground state).\n\n"
                "Per the DIRAC tutorial (release-26/tutorials/x_ray/"
                "CO_N2_IP1s): produces CO C1s = 297.3 eV (vs experiment "
                "295.9), CO O1s = 542.0 eV (vs 542.1). Works out of the "
                "box for HETERONUCLEAR systems. For HOMONUCLEAR diatomics "
                "(N2, O2), the symmetric ΔSCF overestimates by ~10 eV — "
                "the orchestrator detects this case, raises a warning, "
                "and refers to the Pipek-Mezey localization workflow.\n\n"
                "Orbital indexing: atoms are sorted by Z descending; "
                "1s MO index follows that order. For CO (O then C): "
                "MO 1 = O 1s, MO 2 = C 1s.\n\n"
                "Returns plan + ip_pairs ready to chain into "
                "prepare_dirac_launch and then compute_dirac_core_ip."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "atoms":             {"type": "array", "items": {"type": "object"}},
                    "target_atom_indices": {"type": "array", "items": {"type": "integer"},
                                            "description": "0-based atom indices to ionize."},
                    "n_total_electrons": {"type": "integer"},
                    "basis":             {"type": "object"},
                    "default_basis":     {"type": "string"},
                    "use_x2c":           {"type": "boolean", "default": False,
                                         "description": "Use X2C approximation. Default False (full 4c). Set True only when explicitly needed."},
                    "output_dir":        {"type": "string"},
                    "molecule_name":     {"type": "string", "default": "molecule"},
                    "molecule_units":    {"type": "string", "default": "bohr"},
                    "closed_shell_per_ircop": {"type": "array", "items": {"type": "integer"}},
                    "write_files":       {"type": "boolean", "default": False},
                },
                "required": ["atoms", "target_atom_indices", "n_total_electrons"],
            },
        },
        {
            "name": "compute_dirac_core_ip",
            "description": (
                "Compute one core ionization potential from a pair of "
                "DIRAC .out files (neutral + core-ionized). Returns "
                "{ip_hartree, ip_ev, neutral_total_energy_hartree, "
                "ionized_total_energy_hartree}. For multi-atom IPs, "
                "call once per (neutral, ionized) pair using the "
                "ip_pairs list returned by prepare_dirac_core_ionization."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "neutral_out": {"type": "string"},
                    "ionized_out": {"type": "string"},
                },
                "required": ["neutral_out", "ionized_out"],
            },
        },

        # ----- Phase DI: Cm-class multi-step convergence workflow -----
        {
            "name": "prepare_dirac_cm_class_workflow",
            "description": (
                "Build the 3-step convergence plan for heavy actinides "
                "(Cm, Bk, Cf, Es, Fm, Md, No, Lr) where the simple "
                "atomic-start + .KPSELE workflow stalls because the "
                "5f^7+ orbitals lie BELOW the outer 6d/7s shells. Per "
                "Mochizuki JCP 2003 / DIRAC CmF.md: (1) compute a "
                "LIGHTER reference atom (Ce default) with KPSELE; "
                "(2) molecular SCF as closed-shell with imported 5f^N "
                "frozen at chosen orbital positions; (3) closed shells "
                "frozen, 5f^N relaxes.\n\n"
                "Step 1's input is fully auto-drafted and runnable. Step 2 "
                "and Step 3 inputs are SCAFFOLDED with explanatory "
                "comments — they need a chemist to fill in the .FROZEN / "
                "orbital-position-remap blocks (the exact syntax lives "
                "in DIRAC's test/tutorial fixtures, not in the bundled "
                "docs). Returns the launch-command hints with the right "
                "``--put``/``--get`` plumbing for the cf.<elem> orbital-"
                "file passing between steps.\n\n"
                "Call get_dirac_topic_guide('cm_class_workflow') for the "
                "full strategy narrative."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "central_element":   {"type": "string"},
                    "molecule_atoms":    {"type": "array", "items": {"type": "object"}},
                    "basis":             {"type": "object"},
                    "default_basis":     {"type": "string", "default": "dyall.2zp"},
                    "reference_element": {"type": "string",
                                          "description": "Surrogate reference atom. Default: Pu for Cm/Bk/Cf/Es/Fm/Md/No/Lr (chemically closer + converges with KPSELE in DIRAC 25); Ce for legacy CmF.md compatibility."},
                    "output_dir":        {"type": "string"},
                    "molecule_name":     {"type": "string", "default": "molecule"},
                    "molecule_units":    {"type": "string", "default": "bohr"},
                    "n_5f_electrons":    {"type": "integer", "default": 7},
                    "write_files":       {"type": "boolean", "default": False,
                                          "description": "Write each step's .inp/.mol to disk."},
                },
                "required": ["central_element", "molecule_atoms"],
            },
        },

        # ----- Phase DG: pam-dirac launcher -----
        {
            "name": "prepare_dirac_launch",
            "description": (
                "Build the pam-dirac command an agent should execute. Does NOT "
                "execute it. Supports apptainer / singularity containers via "
                "``container_sif``. Flags: ``--mpi=N`` for ranks, "
                "``--mw=N --nw=N`` for master/node memory in MB, "
                "``--copy=\"a.h5 b.h5\"`` for atomic-start checkpoint chains, "
                "``--outcmo`` to keep MO coefficients, ``--get=NAME`` for "
                "Fortran-binary artifact retrieval (e.g. DFACMO, DFCOEF, "
                "DFPCMO). Returns the command list + shell-quoted string + "
                "expected output file paths."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file":     {"type": "string"},
                    "mol_file":       {"type": "string"},
                    "mpi":            {"type": "integer"},
                    "mw":             {"type": "integer"},
                    "nw":             {"type": "integer"},
                    "copy_files":     {"type": "array", "items": {"type": "string"}},
                    "outcmo":         {"type": "boolean", "default": False},
                    "get_files":      {"type": "array", "items": {"type": "string"}},
                    "container_sif":  {"type": "string"},
                    "pam_dirac_binary": {"type": "string", "default": "pam-dirac"},
                    "apptainer_binary": {"type": "string", "default": "apptainer"},
                    "work_dir":       {"type": "string"},
                    "extra_args":     {"type": "array", "items": {"type": "string"}},
                },
                "required": ["input_file", "mol_file"],
            },
        },

        # ----- Documentation -----
        {
            "name": "list_dirac_docs",
            "description": (
                "List the bundled DIRAC documentation files (180 .md files, "
                "covering basis, HAMILTONIAN, WAVE FUNCTION, AOC, COSCI, KRCI, "
                "RECP/ECP, atomic start, checkpoints, HDF5 schema, AMFI, "
                "complex response, TDA/TDDFT, BSSE, and more)."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "search_dirac_docs",
            "description": (
                "Substring search across all bundled DIRAC docs. Returns up to "
                "``max_hits`` matches with surrounding context. Use this when "
                "the agent needs to look up a keyword or DIRAC concept."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_hits": {"type": "integer", "default": 8},
                    "context_lines": {"type": "integer", "default": 2},
                },
                "required": ["query"],
            },
        },
        {
            "name": "lookup_dirac_section",
            "description": (
                "Find the bundled doc page that best documents a DIRAC section "
                "or keyword (e.g. 'WAVE FUNCTION', '.AOC', 'REORDER', 'COSCI', "
                "'atomic_start'). Ranked by filename match + title match + body "
                "frequency. Returns top match's preview text + filename."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "section": {"type": "string"},
                    "max_results": {"type": "integer", "default": 1},
                },
                "required": ["section"],
            },
        },
        {
            "name": "read_dirac_doc_excerpt",
            "description": (
                "Read a slice of a bundled DIRAC doc by filename (e.g. 'aoc.md', "
                "'reorder.md'). Defaults to the first 200 lines; pass start_line "
                "+ end_line for a specific range."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "max_lines": {"type": "integer", "default": 200},
                },
                "required": ["name"],
            },
        },
        {
            "name": "get_dirac_topic_guide",
            "description": (
                "Curated agent guidance for high-value DIRAC topics. Recognized "
                "topics: aoc (average-of-configurations open-shell HF — DIRAC "
                "has no ROHF), cosci (complete open-shell CI on AOC orbitals), "
                "reorder (.REORDER block under *SCF for fixing wrong starting "
                "orbitals), atomic_start (per-element atomic .h5 → molecule "
                "via --copy=), checkpoint (.h5 schema + DFCOEF / DFPCMO "
                "Fortran binaries), ecp (ECP/RECP in .mol files + .ECP "
                "directive). Each guide returns a summary, key doc files, and "
                "an agent_pattern showing how to chain calls."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
        # ----- Scheduler runner tools (HPC / local) -----
        {
            "name": "launch_dirac_run",
            "description": (
                "Submit a DIRAC job to the scheduler defined by a runner profile. "
                "Renders the submit script (which calls pam-dirac with --inp and --mol "
                "via the profile's script_template), calls sbatch / qsub, parses the "
                "job ID, and writes {job_name}.jobid. Set dry_run=true to preview."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the DIRAC .inp file."},
                    "mol_file": {"type": "string", "description": "Path to the matching .mol file."},
                    "profile": {"type": "string"},
                    "profiles_path": {"type": "string"},
                    "job_name": {"type": "string"},
                    "resource_overrides": {"type": "object"},
                    "env_overrides": {"type": "object"},
                    "write_script": {"type": "boolean", "default": True},
                    "dry_run": {"type": "boolean", "default": False},
                },
                "required": ["input_file", "mol_file", "profile"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_dirac_run_status",
            "description": (
                "Check the status of a DIRAC run. For HPC jobs the scheduler job ID "
                "is auto-detected from {job_name}.jobid. Returns scheduler state "
                "(queued/running/completed/failed/cancelled) and an overall_status."
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
            "name": "watch_dirac_run",
            "description": (
                "Poll DIRAC status until terminal state or timeout. For HPC jobs, "
                "omit timeout_seconds to block until scheduler completion."
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
                    "timeout_seconds": {"type": ["number", "null"], "default": 3600.0},
                    "max_polls": {"type": "integer"},
                    "history_limit": {"type": "integer", "default": 8},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "terminate_dirac_run",
            "description": (
                "Cancel a running DIRAC scheduler job. Provide job_id + profile "
                "(profile resolves the scancel/qdel/bkill command)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "profile": {"type": "string"},
                    "profiles_path": {"type": "string"},
                },
                "required": ["job_id", "profile"],
                "additionalProperties": False,
            },
        },
    ]


# =====================================================================
# Handlers
# =====================================================================

@_tool("parse_dirac_input")
def _handle_parse_dirac_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return _parse_inp(arguments["input_file"])


@_tool("parse_dirac_mol")
def _handle_parse_dirac_mol(arguments: dict[str, Any]) -> dict[str, Any]:
    return _parse_mol(arguments["mol_file"])


@_tool("parse_dirac_output")
def _handle_parse_dirac_output(arguments: dict[str, Any]) -> dict[str, Any]:
    return _parse_output(arguments["output_file"])


@_tool("parse_dirac_scf_iterations")
def _handle_parse_dirac_scf_iterations(arguments: dict[str, Any]) -> dict[str, Any]:
    with open(arguments["output_file"], encoding="utf-8", errors="replace") as f:
        contents = f.read()
    iters = _parse_scf_iters(contents)
    return {
        "n_iterations": len(iters),
        "iterations": iters,
        "final_energy_hartree": iters[-1]["energy_hartree"] if iters else None,
    }


@_tool("parse_dirac_symmetry")
def _handle_parse_dirac_symmetry(arguments: dict[str, Any]) -> dict[str, Any]:
    with open(arguments["output_file"], encoding="utf-8", errors="replace") as f:
        contents = f.read()
    return _parse_symmetry(contents)


@_tool("read_dirac_h5_metadata")
def _handle_read_dirac_h5_metadata(arguments: dict[str, Any]) -> dict[str, Any]:
    return _h5_metadata(arguments["h5_file"])


@_tool("read_dirac_h5_geometry")
def _handle_read_dirac_h5_geometry(arguments: dict[str, Any]) -> dict[str, Any]:
    return _h5_geometry(arguments["h5_file"])


@_tool("read_dirac_orbitals")
def _handle_read_dirac_orbitals(arguments: dict[str, Any]) -> dict[str, Any]:
    orbs = _h5_orbitals(
        arguments["h5_file"],
        include_negative_energy=bool(arguments.get("include_negative_energy", False)),
        only_occupied=bool(arguments.get("only_occupied", False)),
        fractional_only=bool(arguments.get("fractional_only", False)),
    )
    # Roll-up summary alongside the orbital list
    by_class: dict[str, int] = {}
    for o in orbs:
        by_class[o["shell_class"]] = by_class.get(o["shell_class"], 0) + 1
    return {
        "h5_file": arguments["h5_file"],
        "n_orbitals": len(orbs),
        "shell_class_counts": by_class,
        "orbitals": orbs,
    }


@_tool("read_dirac_mo_coefficients")
def _handle_read_dirac_mo_coefficients(arguments: dict[str, Any]) -> dict[str, Any]:
    indices = arguments.get("mo_indices")
    out = _h5_mo_coeffs(arguments["h5_file"], mo_indices=indices)
    # numpy → list so it serializes through JSON
    if hasattr(out.get("coefficients"), "tolist"):
        out["coefficients"] = out["coefficients"].tolist()
    return out


@_tool("analyze_dirac_open_shell")
def _handle_analyze_dirac_open_shell(arguments: dict[str, Any]) -> dict[str, Any]:
    inp = _parse_inp(arguments["input_file"])
    if not _H5PY_AVAILABLE:
        return {
            "verdict": "h5py_missing",
            "message": (
                "h5py is required to cross-check open-shell occupations. "
                "Install via `pip install chemtools[dirac]`."
            ),
            "input_summary": {
                "has_open_shell": inp.get("has_open_shell"),
                "has_closed_shell": inp.get("has_closed_shell"),
            },
        }
    orbs = _h5_orbitals(arguments["h5_file"])
    open_obs = [o for o in orbs if o["shell_class"] == "open"]

    has_open_inp = inp.get("has_open_shell", False)
    has_open_h5 = bool(open_obs)
    if has_open_inp and has_open_h5:
        verdict = "consistent"
    elif has_open_inp and not has_open_h5:
        verdict = "open_shell_requested_but_converged_to_closed"
    elif not has_open_inp and has_open_h5:
        verdict = "unexpected_fractional_occupation"
    else:
        verdict = "no_open_shell"

    # Per-fsym + per-irrep breakdown of the observed open shell
    by_fsym: dict[int, list[dict[str, Any]]] = {}
    for o in open_obs:
        by_fsym.setdefault(o["fermion_symmetry"], []).append({
            "irrep": o["irrep"],
            "positive_energy_index": o["positive_energy_index"],
            "energy_hartree": o["energy_hartree"],
            "occupation": o["occupation"],
        })

    total_open_occ = sum(o["occupation"] for o in open_obs)
    return {
        "verdict": verdict,
        "input_has_open_shell": has_open_inp,
        "h5_has_fractional_occupation": has_open_h5,
        "open_shell_n_orbitals": len(open_obs),
        "open_shell_total_occupation_kramers": total_open_occ,
        "open_shell_by_fermion_symmetry": by_fsym,
    }


@_tool("summarize_dirac_outputs")
def _handle_summarize_dirac_outputs(arguments: dict[str, Any]) -> dict[str, Any]:
    from chemtools.programs.dirac.strategy.triage import summarize_dirac_outputs
    target = arguments.get("paths") or arguments.get("path")
    if not target:
        return {"error": "Provide 'path' (a directory, glob, or file) or 'paths' (a list)."}
    return summarize_dirac_outputs(
        paths=target,
        pattern=arguments.get("pattern", "*.out"),
        recursive=arguments.get("recursive", False),
        limit=arguments.get("limit"),
    )


@_tool("summarize_dirac_run")
def _handle_summarize_dirac_run(arguments: dict[str, Any]) -> dict[str, Any]:
    out_path = arguments["output_file"]
    h5_path = arguments.get("h5_file")

    text_parse = _parse_output(out_path)
    summary: dict[str, Any] = {
        "program": "dirac",
        "output_file": out_path,
        "program_version": text_parse.get("program_version"),
        "tasks_detected": text_parse.get("tasks_detected"),
        "total_energy_hartree": text_parse.get("total_energy_hartree"),
        "scf_converged": text_parse.get("scf_converged"),
        "scf_n_iterations": text_parse.get("scf_n_iterations"),
        "symmetry": text_parse.get("symmetry"),
        "open_shell_setup": text_parse.get("open_shell_setup"),
        "homo_lumo_blocks_count": len(text_parse.get("homo_lumo_per_symmetry") or []),
    }

    excitations = text_parse.get("excitations") or {}
    if excitations.get("available"):
        summary["excited_states"] = {
            "n_excitations": excitations["n_excitations"],
            "lowest_excitation_ev": excitations.get("lowest_excitation_ev"),
            "sum_oscillator_strength": excitations.get("sum_oscillator_strength"),
            "excitations": excitations["excitations"],
        }

    relccsd = text_parse.get("relccsd") or {}
    if relccsd.get("available"):
        summary["correlation"] = {
            "mp2_total_hartree": relccsd.get("mp2_total_hartree"),
            "ccsd_total_hartree": relccsd.get("ccsd_total_hartree"),
            "ccsd_t_total_hartree": relccsd.get("ccsd_t_total_hartree"),
            "mp2_correlation_hartree": relccsd.get("mp2_correlation_hartree"),
            "ccsd_correlation_hartree": relccsd.get("ccsd_correlation_hartree"),
        }

    if h5_path:
        if not _H5PY_AVAILABLE:
            summary["h5_status"] = "h5py_missing"
        else:
            try:
                meta = _h5_metadata(h5_path)
                summary["h5_status"] = "loaded"
                summary["h5_version"] = meta.get("version")
                summary["h5_scf_energy_hartree"] = meta.get("scf_energy_hartree")
                summary["n_fermion_symmetries"] = meta.get("n_fermion_symmetries")
                summary["n_mo_per_fsym"] = meta.get("n_mo_per_fsym")
                summary["n_pos_energy_per_fsym"] = meta.get("n_pos_energy_per_fsym")
                # Cheap occupation-class rollup
                orbs = _h5_orbitals(h5_path)
                by_class: dict[str, int] = {}
                for o in orbs:
                    by_class[o["shell_class"]] = by_class.get(o["shell_class"], 0) + 1
                summary["shell_class_counts"] = by_class
                # Cross-check energy between text and h5
                if (summary["total_energy_hartree"] is not None
                    and summary.get("h5_scf_energy_hartree") is not None):
                    diff = abs(
                        summary["total_energy_hartree"]
                        - summary["h5_scf_energy_hartree"]
                    )
                    summary["text_vs_h5_energy_consistent"] = diff < 1e-6
            except Exception as e:
                summary["h5_status"] = f"error: {e}"

    # Verdict line
    if summary.get("scf_converged"):
        verdict = "scf_converged"
    elif summary.get("scf_n_iterations"):
        verdict = "scf_did_not_converge"
    else:
        verdict = "no_scf_detected"
    summary["verdict"] = verdict

    return summary


@_tool("parse_dirac_vecpop")
def _handle_parse_dirac_vecpop(arguments: dict[str, Any]) -> dict[str, Any]:
    with open(arguments["output_file"], encoding="utf-8", errors="replace") as f:
        text = f.read()
    return _parse_vecpop(text)


@_tool("parse_dirac_spinor_spectrum")
def _handle_parse_dirac_spinor_spectrum(arguments: dict[str, Any]) -> dict[str, Any]:
    with open(arguments["output_file"], encoding="utf-8", errors="replace") as f:
        text = f.read()
    spectrum = _parse_spinor_spectrum(text)
    if arguments.get("occupied_only"):
        spectrum = [s for s in spectrum if (s.get("occupation") or 0) > 0.5]
    erange = arguments.get("energy_range")
    if erange and len(erange) == 2:
        lo, hi = float(erange[0]), float(erange[1])
        spectrum = [s for s in spectrum if s.get("energy_hartree") is not None
                    and lo <= s["energy_hartree"] <= hi]
    return {"spinor_spectrum": spectrum, "n_spinors": len(spectrum)}


@_tool("parse_dirac_cosci_energies")
def _handle_parse_dirac_cosci_energies(arguments: dict[str, Any]) -> dict[str, Any]:
    with open(arguments["output_file"], encoding="utf-8", errors="replace") as f:
        text = f.read()
    result = _parse_cosci_energies(text)
    if result is None:
        return {"found": False, "message": "No COSCI state table found in this output."}
    result["found"] = True
    return result


@_tool("list_dirac_basis_sets")
def _handle_list_dirac_basis_sets(arguments: dict[str, Any]) -> dict[str, Any]:
    return _list_basis_sets(
        element=arguments.get("element"),
        family_type=arguments.get("family_type"),
        zeta=int(arguments["zeta"]) if arguments.get("zeta") is not None else None,
        calc_type=arguments.get("calc_type"),
    )


@_tool("suggest_dirac_basis")
def _handle_suggest_dirac_basis(arguments: dict[str, Any]) -> dict[str, Any]:
    return _suggest_basis(
        element=arguments["element"],
        calc_type=arguments.get("calc_type", "scf"),
        zeta=int(arguments["zeta"]) if arguments.get("zeta") is not None else None,
    )


@_tool("analyze_dirac_open_shell_quality")
def _handle_analyze_dirac_open_shell_quality(arguments: dict[str, Any]) -> dict[str, Any]:
    return _analyze_open_shell_quality(
        arguments["output_file"],
        expected_character=arguments.get("expected_character"),
    )


@_tool("suggest_dirac_orbital_swaps")
def _handle_suggest_dirac_orbital_swaps(arguments: dict[str, Any]) -> dict[str, Any]:
    target = arguments["target_character"]
    if isinstance(target, str):
        target = [target]
    return _suggest_orbital_swaps(
        arguments["output_file"],
        target_character=list(target),
        n_candidates=int(arguments.get("n_candidates", 6)),
    )


@_tool("draft_dirac_reorder_block")
def _handle_draft_dirac_reorder_block(arguments: dict[str, Any]) -> dict[str, Any]:
    block = _draft_reorder_block(list(arguments["per_ircop_orders"]))
    return {"block_text": block, "n_ircops": len(arguments["per_ircop_orders"])}


@_tool("apply_dirac_reorder_to_input")
def _handle_apply_dirac_reorder_to_input(arguments: dict[str, Any]) -> dict[str, Any]:
    with open(arguments["input_file"], encoding="utf-8", errors="replace") as f:
        text = f.read()
    result = _apply_reorder_to_input(
        text,
        list(arguments["per_ircop_orders"]),
        replace=bool(arguments.get("replace", False)),
    )
    out_path = arguments.get("output_path")
    if out_path and result["action"] != "already_present":
        from pathlib import Path
        Path(out_path).write_text(result["patched_text"], encoding="utf-8")
        result["output_file"] = out_path
    return result


@_tool("parse_dirac_reorder_block")
def _handle_parse_dirac_reorder_block(arguments: dict[str, Any]) -> dict[str, Any]:
    with open(arguments["input_file"], encoding="utf-8", errors="replace") as f:
        text = f.read()
    r = _parse_reorder_block(text)
    if r is None:
        return {"present": False, "input_file": arguments["input_file"]}
    return {"present": True, **r}


@_tool("draft_dirac_input")
def _handle_draft_dirac_input(arguments: dict[str, Any]) -> dict[str, Any]:
    text = _draft_inp(arguments["spec"])
    out = {"input_text": text, "n_lines": len(text.splitlines())}
    if arguments.get("output_path"):
        from pathlib import Path
        Path(arguments["output_path"]).write_text(text, encoding="utf-8")
        out["output_file"] = arguments["output_path"]
    return out


@_tool("draft_dirac_mol")
def _handle_draft_dirac_mol(arguments: dict[str, Any]) -> dict[str, Any]:
    # Normalize basis keys (JSON delivers ints as strings sometimes).
    raw_basis = arguments.get("basis")
    basis: dict[Any, str] | None = None
    if raw_basis:
        basis = {}
        for k, v in raw_basis.items():
            try:
                basis[int(k)] = v
            except (TypeError, ValueError):
                basis[k] = v
    text = _draft_mol(
        atoms=arguments["atoms"],
        basis=basis,
        default_basis=arguments.get("default_basis"),
        units=arguments.get("units", "bohr"),
        title=arguments.get("title", "DIRAC mol file generated by chemtools"),
        symmetry=arguments.get("symmetry", "auto"),
    )
    out = {"mol_text": text, "n_lines": len(text.splitlines())}
    if arguments.get("output_path"):
        from pathlib import Path
        Path(arguments["output_path"]).write_text(text, encoding="utf-8")
        out["output_file"] = arguments["output_path"]
    return out


@_tool("prepare_dirac_atomic_start")
def _handle_prepare_dirac_atomic_start(arguments: dict[str, Any]) -> dict[str, Any]:
    raw_basis = arguments.get("basis")
    basis: dict[Any, str] | None = None
    if raw_basis:
        basis = {}
        for k, v in raw_basis.items():
            try:
                basis[int(k)] = v
            except (TypeError, ValueError):
                basis[k] = v
    result = _prepare_atomic_start(
        molecule_atoms=arguments["molecule_atoms"],
        basis=basis,
        default_basis=arguments.get("default_basis"),
        hamiltonian=arguments.get("hamiltonian"),
        integrals=arguments.get("integrals"),
        use_x2c=bool(arguments.get("use_x2c", False)),
        output_dir=arguments.get("output_dir"),
        molecule_name=arguments.get("molecule_name", "molecule"),
        molecule_scf=arguments.get("molecule_scf"),
        molecule_units=arguments.get("molecule_units", "bohr"),
    )
    if arguments.get("write_files"):
        from pathlib import Path
        for p in result["plan"]:
            Path(p["inp_path"]).parent.mkdir(parents=True, exist_ok=True)
            Path(p["inp_path"]).write_text(p["inp_text"], encoding="utf-8")
            Path(p["mol_path"]).write_text(p["mol_text"], encoding="utf-8")
        result["files_written"] = True
    return result


@_tool("prepare_dirac_x2c_bootstrap")
def _handle_prepare_dirac_x2c_bootstrap(arguments: dict[str, Any]) -> dict[str, Any]:
    raw_basis = arguments.get("basis")
    basis: dict[str, str] | None = None
    if raw_basis:
        basis = {str(k): v for k, v in raw_basis.items()}
    return _prepare_x2c_bootstrap(
        element=arguments["element"],
        basis=basis,
        default_basis=arguments.get("default_basis"),
        hamiltonian=arguments.get("hamiltonian"),
        integrals=arguments.get("integrals"),
        output_dir=arguments.get("output_dir"),
    )


@_tool("prepare_dirac_core_ionization")
def _handle_prepare_dirac_core_ionization(arguments: dict[str, Any]) -> dict[str, Any]:
    raw_basis = arguments.get("basis")
    basis: dict[str, str] | None = None
    if raw_basis:
        basis = {str(k): v for k, v in raw_basis.items()}
    return _prepare_core_ionization(
        atoms=arguments["atoms"],
        target_atom_indices=arguments["target_atom_indices"],
        n_total_electrons=int(arguments["n_total_electrons"]),
        basis=basis,
        default_basis=arguments.get("default_basis"),
        use_x2c=bool(arguments.get("use_x2c", False)),
        output_dir=arguments.get("output_dir"),
        molecule_name=arguments.get("molecule_name", "molecule"),
        molecule_units=arguments.get("molecule_units", "bohr"),
        closed_shell_per_ircop=arguments.get("closed_shell_per_ircop"),
        write_files=bool(arguments.get("write_files", False)),
    )


@_tool("compute_dirac_core_ip")
def _handle_compute_dirac_core_ip(arguments: dict[str, Any]) -> dict[str, Any]:
    return _compute_core_ip(
        neutral_out=arguments["neutral_out"],
        ionized_out=arguments["ionized_out"],
    )


@_tool("prepare_dirac_cm_class_workflow")
def _handle_prepare_dirac_cm_class_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    raw_basis = arguments.get("basis")
    basis: dict[str, str] | None = None
    if raw_basis:
        basis = {str(k): v for k, v in raw_basis.items()}
    result = _prepare_cm_class_workflow(
        central_element=arguments["central_element"],
        molecule_atoms=arguments["molecule_atoms"],
        basis=basis,
        default_basis=arguments.get("default_basis", "dyall.2zp"),
        reference_element=arguments.get("reference_element"),  # None → auto-pick
        output_dir=arguments.get("output_dir"),
        molecule_name=arguments.get("molecule_name", "molecule"),
        molecule_units=arguments.get("molecule_units", "bohr"),
        n_5f_electrons=int(arguments.get("n_5f_electrons", 7)),
    )
    if arguments.get("write_files"):
        from pathlib import Path
        for p in result["plan"]:
            Path(p["inp_path"]).parent.mkdir(parents=True, exist_ok=True)
            Path(p["inp_path"]).write_text(p["inp_text"], encoding="utf-8")
            Path(p["mol_path"]).write_text(p["mol_text"], encoding="utf-8")
        result["files_written"] = True
    return result


@_tool("prepare_dirac_launch")
def _handle_prepare_dirac_launch(arguments: dict[str, Any]) -> dict[str, Any]:
    return _prepare_launch(
        input_file=arguments["input_file"],
        mol_file=arguments["mol_file"],
        mpi=arguments.get("mpi"),
        mw=arguments.get("mw"),
        nw=arguments.get("nw"),
        copy_files=arguments.get("copy_files"),
        outcmo=bool(arguments.get("outcmo", False)),
        get_files=arguments.get("get_files"),
        container_sif=arguments.get("container_sif"),
        pam_dirac_binary=arguments.get("pam_dirac_binary", "pam-dirac"),
        apptainer_binary=arguments.get("apptainer_binary", "apptainer"),
        work_dir=arguments.get("work_dir"),
        extra_args=arguments.get("extra_args"),
    )


@_tool("list_dirac_docs")
def _handle_list_dirac_docs(arguments: dict[str, Any]) -> dict[str, Any]:
    docs = _list_docs()
    return {"n_docs": len(docs), "docs": docs}


@_tool("search_dirac_docs")
def _handle_search_dirac_docs(arguments: dict[str, Any]) -> dict[str, Any]:
    return _search_docs(
        arguments["query"],
        max_hits=int(arguments.get("max_hits", 8)),
        context_lines=int(arguments.get("context_lines", 2)),
    )


@_tool("lookup_dirac_section")
def _handle_lookup_dirac_section(arguments: dict[str, Any]) -> dict[str, Any]:
    return _lookup_section(
        arguments["section"],
        max_results=int(arguments.get("max_results", 1)),
    )


@_tool("read_dirac_doc_excerpt")
def _handle_read_dirac_doc_excerpt(arguments: dict[str, Any]) -> dict[str, Any]:
    return _read_doc_excerpt(
        arguments["name"],
        start_line=arguments.get("start_line"),
        end_line=arguments.get("end_line"),
        max_lines=int(arguments.get("max_lines", 200)),
    )


@_tool("get_dirac_topic_guide")
def _handle_get_dirac_topic_guide(arguments: dict[str, Any]) -> dict[str, Any]:
    return _get_topic_guide(arguments["topic"])


# ----- Scheduler runner handlers -----------------------------------------------

@_tool("launch_dirac_run", needs="executable")
def _handle_launch_dirac_run(arguments: dict[str, Any]) -> dict[str, Any]:
    return _launch_dirac_run(
        input_path=arguments["input_file"],
        mol_file=arguments["mol_file"],
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path"),
        job_name=arguments.get("job_name"),
        resource_overrides=arguments.get("resource_overrides"),
        env_overrides=arguments.get("env_overrides"),
        write_script=arguments.get("write_script", True),
        dry_run=arguments.get("dry_run", False),
    )


@_tool("get_dirac_run_status", needs="executable")
def _handle_get_dirac_run_status(arguments: dict[str, Any]) -> dict[str, Any]:
    return _get_dirac_run_status(
        output_path=arguments.get("output_file"),
        input_path=arguments.get("input_file"),
        error_path=arguments.get("error_file"),
        process_id=arguments.get("process_id"),
        profile=arguments.get("profile"),
        job_id=arguments.get("job_id"),
        profiles_path=arguments.get("profiles_path"),
    )


@_tool("watch_dirac_run", needs="executable")
def _handle_watch_dirac_run(arguments: dict[str, Any]) -> dict[str, Any]:
    return _watch_dirac_run(
        output_path=arguments.get("output_file"),
        input_path=arguments.get("input_file"),
        error_path=arguments.get("error_file"),
        process_id=arguments.get("process_id"),
        profile=arguments.get("profile"),
        job_id=arguments.get("job_id"),
        profiles_path=arguments.get("profiles_path"),
        poll_interval_seconds=arguments.get("poll_interval_seconds", 10.0),
        adaptive_polling=arguments.get("adaptive_polling", True),
        max_poll_interval_seconds=arguments.get("max_poll_interval_seconds", 60.0),
        timeout_seconds=arguments.get("timeout_seconds", 3600.0),
        max_polls=arguments.get("max_polls"),
        history_limit=arguments.get("history_limit", 8),
    )


@_tool("terminate_dirac_run", needs="executable")
def _handle_terminate_dirac_run(arguments: dict[str, Any]) -> dict[str, Any]:
    import os
    profiles_path = arguments.get("profiles_path") or os.environ.get("CHEMTOOLS_RUNNER_PROFILES")
    return _terminate_dirac_run(
        job_id=arguments["job_id"],
        profile=arguments["profile"],
        profiles_path=profiles_path,
    )

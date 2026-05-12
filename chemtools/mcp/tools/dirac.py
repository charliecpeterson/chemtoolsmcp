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
    analyze_dirac_open_shell       cross-checks input AOC spec vs h5 orbital occupations
    summarize_dirac_run            high-level rollup of a DIRAC text+h5 pair

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
                "Single-pass parse of a DIRAC text output. Extracts SCF iteration trace, "
                "total energy, detected tasks (scf/dft/mp2/cosci/krci/ccsd/response), "
                "symmetry detection + per-irrep orbital counts, AOC open-shell setup "
                "(.CLOSED SHELL + .OPEN SHELL blocks), per-symmetry HOMO/LUMO blocks "
                "from RESOLVE, Mulliken population. Cheap enough to fit in agent context."
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

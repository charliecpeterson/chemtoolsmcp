"""NWChem MCP handlers for text and binary output parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.programs.nwchem.strategy.legacy_next_actions import (
    build_legacy_next_actions as _build_next_actions,
)
from chemtools.mcp.tools._nwchem_registration import _tool
from chemtools.programs.nwchem.binary.movecs import parse_nwchem_movecs
from chemtools.programs.nwchem.input.basis import inspect_nwchem_geometry
from chemtools.programs.nwchem.input.geometry import extract_nwchem_geometry
from chemtools.programs.nwchem.output import (
    parse_freq_progress,
    parse_mcscf_output,
    parse_mos,
    parse_nwchem_thermochem,
    parse_output,
    parse_population_analysis,
    parse_tasks,
    parse_tce_output,
    parse_trajectory,
    summarize_output,
)
from chemtools.programs.nwchem.parse.tce import parse_tce_amplitudes
from chemtools.programs.nwchem.strategy.diagnose import (
    parse_scf,
    summarize_electronic_structure,
    summarize_nwchem_outputs,
)


@_tool("parse_nwchem_output")
def _handle_parse_nwchem_output(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_output(
        arguments["output_file"],
        sections=arguments.get("sections"),
        top_n=arguments.get("top_n", 5),
        include_coefficients=arguments.get("include_coefficients", False),
        include_displacements=arguments.get("include_displacements", False),
        include_positions=arguments.get("include_positions", False),
    )


@_tool("parse_nwchem_thermochem")
def _handle_parse_nwchem_thermochem(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_nwchem_thermochem(
        path=arguments["output_file"],
        T=arguments.get("T", 298.15),
        P=arguments.get("P", 1.0),
    )


@_tool("summarize_nwchem_electronic_structure")
def _handle_summarize_electronic_structure(arguments: dict[str, Any]) -> dict[str, Any]:
    result = summarize_electronic_structure(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
    )
    result["next_actions"] = _build_next_actions(
        "electronic_structure", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("extract_nwchem_geometry")
def _handle_extract_nwchem_geometry(arguments: dict[str, Any]) -> dict[str, Any]:
    frame_arg = arguments.get("frame", "best")
    try:
        frame_arg = int(frame_arg)
    except (TypeError, ValueError):
        pass
    result = extract_nwchem_geometry(
        output_path=arguments["output_file"],
        frame=frame_arg,
        input_path=arguments.get("input_file"),
    )
    return result


@_tool("parse_nwchem_scf")
def _handle_parse_nwchem_scf(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_scf(arguments["file_path"])


@_tool("parse_nwchem_mos")
def _handle_parse_nwchem_mos(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_mos(
        arguments["file_path"],
        top_n=arguments.get("top_n", 5),
        include_coefficients=arguments.get("include_coefficients", False),
        include_all_orbitals=arguments.get("include_all_orbitals", False),
    )


@_tool("parse_nwchem_mcscf_output")
def _handle_parse_nwchem_mcscf_output(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_mcscf_output(arguments["file_path"])


@_tool("parse_nwchem_population_analysis")
def _handle_parse_nwchem_population_analysis(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_population_analysis(arguments["file_path"])


# ---------------------------------------------------------------------------
# Handlers — input inspection and linting
# ---------------------------------------------------------------------------


@_tool("parse_nwchem_freq_progress")
def _handle_parse_nwchem_freq_progress(arguments: dict[str, Any]) -> dict[str, Any]:
    result = parse_freq_progress(arguments["output_file"])
    result["next_actions"] = _build_next_actions(
        "freq_progress", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("parse_nwchem_tce_output")
def _handle_parse_nwchem_tce_output(arguments: dict[str, Any]) -> dict[str, Any]:
    output_file = arguments["output_file"]
    result = parse_tce_output(output_file)
    # Auto-include T1/D1 multireference diagnostics when amplitude files exist
    try:
        amp = parse_tce_amplitudes(arguments["output_file"])
        if amp.get("available"):
            result["multireference_diagnostics"] = {
                "t1_diagnostic": amp.get("t1_diagnostic"),
                "d1_diagnostic": amp.get("d1_diagnostic"),
                "t2_frobenius_norm": amp.get("t2_frobenius_norm"),
                "mr_assessment": amp.get("mr_assessment"),
                "mr_flags": amp.get("mr_flags", []),
                "top_t2_amplitudes": amp.get("top_t2_amplitudes", []),
                "amplitude_files": amp.get("amplitude_files", []),
                "note": (
                    "T1 > 0.02: moderate MR character; > 0.05: strong MR — CCSD unreliable. "
                    "D1 > 0.05: significant orbital relaxation."
                ),
            }
        else:
            result["multireference_diagnostics"] = {
                "available": False,
                "reason": amp.get("reason", "amplitude files not found"),
                "note": "Rerun with 'set tce:save_t T T' to enable T1/D1 diagnostics.",
            }
    except Exception as exc:
        result["multireference_diagnostics"] = {"available": False, "error": str(exc)}
    result["next_actions"] = _build_next_actions(
        "tce_output", result,
        output_file=output_file,
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("parse_nwchem_tce_amplitudes")
def _handle_parse_nwchem_tce_amplitudes(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_tce_amplitudes(arguments["output_file"])


@_tool("parse_nwchem_movecs")
def _handle_parse_nwchem_movecs(arguments: dict[str, Any]) -> dict[str, Any]:
    movecs_file = arguments["movecs_file"]
    result = parse_nwchem_movecs(movecs_file)
    # The natural sibling .out file usually shares the same stem, e.g.
    # water.movecs ↔ water.out — emit that as the next-action target.
    output_file = str(Path(movecs_file).with_suffix(".out"))
    result["next_actions"] = _build_next_actions(
        "movecs", result,
        output_file=output_file,
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("parse_nwchem_hessian")
def _handle_parse_nwchem_hessian(arguments: dict[str, Any]) -> dict[str, Any]:
    from chemtools.programs.nwchem.binary.hessian import parse_nwchem_hessian
    return parse_nwchem_hessian(
        arguments["hessian_file"],
        return_matrix=arguments.get("return_matrix", True),
    )


@_tool("inspect_nwchem_geometry")
def _handle_inspect_nwchem_geometry(arguments: dict[str, Any]) -> dict[str, Any]:
    return inspect_nwchem_geometry(
        input_path=arguments["input_file"],
    )


@_tool("parse_nwchem_tasks")
def _handle_parse_nwchem_tasks(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_tasks(arguments["output_file"])


@_tool("parse_nwchem_trajectory")
def _handle_parse_nwchem_trajectory(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_trajectory(
        path=arguments["output_file"],
        include_positions=arguments.get("include_positions", False),
    )


@_tool("summarize_nwchem_output")
def _handle_summarize_nwchem_output(arguments: dict[str, Any]) -> dict[str, Any]:
    result = summarize_output(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
        detail_level=arguments.get("detail", "summary"),
    )
    result["next_actions"] = _build_next_actions(
        "summarize_output", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("summarize_nwchem_outputs")
def _handle_summarize_nwchem_outputs(arguments: dict[str, Any]) -> dict[str, Any]:
    target = arguments.get("paths") or arguments.get("path")
    if not target:
        return {"error": "Provide 'path' (a directory, glob, or file) or 'paths' (a list)."}
    return summarize_nwchem_outputs(
        paths=target,
        pattern=arguments.get("pattern", "*.out"),
        recursive=arguments.get("recursive", False),
        limit=arguments.get("limit"),
    )


# ---------------------------------------------------------------------------
# Handlers — run registry, campaigns, workflows, batch generation
# ---------------------------------------------------------------------------

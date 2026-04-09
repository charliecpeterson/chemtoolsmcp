from __future__ import annotations

from typing import Any

from .common import detect_program, make_metadata, read_text
from .cube import parse_cube_file, summarize_cube_file
from .diagnostics import (
    analyze_frontier_orbitals as analyze_nwchem_frontier_orbitals,
    diagnose_nwchem_output,
    parse_scf,
    suggest_vectors_swaps as suggest_nwchem_vectors_swaps,
    summarize_nwchem_output,
)
from .nwchem_input import inspect_nwchem_input
from .nwchem_tce import parse_tce_output as _parse_tce_output
from . import molcas, molpro, nwchem


def _dispatch_parse_mos(
    path: str, contents: str, top_n: int = 5, include_coefficients: bool = False
) -> dict[str, Any]:
    """Dispatch parse_mos to the correct program module."""
    program = detect_program(contents)
    if program == "molpro":
        return molpro.parse_mos(path, contents, top_n=top_n, include_coefficients=include_coefficients)
    return nwchem.parse_mos(path, contents, top_n=top_n, include_coefficients=include_coefficients)


def parse_tasks(path: str) -> dict[str, Any]:
    contents = read_text(path)
    program = detect_program(contents)
    if program == "nwchem":
        return nwchem.parse_tasks(path, contents)
    if program == "molpro":
        return molpro.parse_tasks(path, contents)
    if program == "molcas":
        return molcas.parse_tasks(path, contents)
    return nwchem.parse_tasks(path, contents)


def parse_mos(path: str, top_n: int = 5, include_coefficients: bool = False) -> dict[str, Any]:
    contents = read_text(path)
    return _dispatch_parse_mos(path, contents, top_n=top_n, include_coefficients=include_coefficients)


def parse_population_analysis(path: str) -> dict[str, Any]:
    contents = read_text(path)
    return nwchem.parse_population_analysis(path, contents)


def parse_mcscf_output(path: str) -> dict[str, Any]:
    contents = read_text(path)
    return nwchem.parse_mcscf_output(path, contents)


def parse_freq(path: str, include_displacements: bool = False) -> dict[str, Any]:
    contents = read_text(path)
    return nwchem.parse_freq(path, contents, include_displacements=include_displacements)


def parse_trajectory(path: str, include_positions: bool = False) -> dict[str, Any]:
    contents = read_text(path)
    return nwchem.parse_trajectory(path, contents, include_positions=include_positions)


def analyze_frontier_orbitals(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
) -> dict[str, Any]:
    contents = read_text(output_path)
    program = detect_program(contents)
    if program != "nwchem":
        raise ValueError(f"frontier orbital analysis is not implemented for {program or 'unknown'}")

    mos = nwchem.parse_mos(output_path, contents, top_n=8)
    population = nwchem.parse_population_analysis(output_path, contents)
    input_summary = inspect_nwchem_input(input_path) if input_path else None
    metals = expected_metal_elements or (input_summary["transition_metals"] if input_summary else [])
    somo_target = expected_somo_count
    if somo_target is None and input_summary and input_summary["multiplicity"] and input_summary["multiplicity"] > 1:
        somo_target = input_summary["multiplicity"] - 1

    analysis = analyze_nwchem_frontier_orbitals(
        mos,
        population_payload=population,
        expected_metal_elements=metals,
        expected_somo_count=somo_target,
    )
    return {
        "metadata": make_metadata(output_path, contents, "nwchem"),
        "input_summary": input_summary,
        "expected_metal_elements": metals,
        "expected_somo_count": somo_target,
        "analysis": analysis,
    }


def suggest_vectors_swaps(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
) -> dict[str, Any]:
    contents = read_text(output_path)
    program = detect_program(contents)
    if program != "nwchem":
        raise ValueError(f"vectors swap suggestions are not implemented for {program or 'unknown'}")

    mos = nwchem.parse_mos(output_path, contents, top_n=8)
    input_summary = inspect_nwchem_input(input_path) if input_path else None
    metals = expected_metal_elements or (input_summary["transition_metals"] if input_summary else [])
    somo_target = expected_somo_count
    if somo_target is None and input_summary and input_summary["multiplicity"] and input_summary["multiplicity"] > 1:
        somo_target = input_summary["multiplicity"] - 1

    payload = suggest_nwchem_vectors_swaps(
        mos,
        expected_metal_elements=metals,
        expected_somo_count=somo_target,
        vectors_input=vectors_input,
        vectors_output=vectors_output,
    )
    return {
        "metadata": make_metadata(output_path, contents, "nwchem"),
        "input_summary": input_summary,
        "expected_metal_elements": metals,
        "expected_somo_count": somo_target,
        "suggestion": payload,
    }


def parse_cube(path: str, include_values: bool = False) -> dict[str, Any]:
    return parse_cube_file(path, include_values=include_values)


def summarize_cube(path: str, top_atoms: int = 5) -> dict[str, Any]:
    return summarize_cube_file(path, top_atoms=top_atoms)


def parse_output(
    path: str,
    sections: list[str] | None = None,
    top_n: int = 5,
    include_coefficients: bool = False,
    include_displacements: bool = False,
    include_positions: bool = False,
) -> dict[str, Any]:
    contents = read_text(path)
    metadata = make_metadata(path, contents)
    selected = sections or ["tasks"]

    output = {
        "metadata": metadata,
        "tasks": None,
        "mos": None,
        "population": None,
        "mcscf": None,
        "frequency": None,
        "trajectory": None,
        "errors": [],
    }

    for section in selected:
        try:
            if section == "tasks":
                output["tasks"] = parse_tasks(path)
            elif section == "mos":
                output["mos"] = parse_mos(path, top_n=top_n, include_coefficients=include_coefficients)
            elif section == "population":
                output["population"] = parse_population_analysis(path)
            elif section == "mcscf":
                output["mcscf"] = parse_mcscf_output(path)
            elif section == "freq":
                output["frequency"] = parse_freq(path, include_displacements=include_displacements)
            elif section == "trajectory":
                output["trajectory"] = parse_trajectory(path, include_positions=include_positions)
            else:
                output["errors"].append(f"unknown section: {section}")
        except Exception as exc:  # pragma: no cover
            output["errors"].append(f"{section}: {exc}")

    return output


def diagnose_output(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
) -> dict[str, Any]:
    return diagnose_nwchem_output(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
    )


def summarize_output(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    detail_level: str = "summary",
) -> dict[str, Any]:
    return summarize_nwchem_output(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
        detail_level=detail_level,
    )


def parse_scf_output(path: str) -> dict[str, Any]:
    """Alias for parse_scf (diagnostics module)."""
    return parse_scf(path)


def parse_tce_output(path: str) -> dict[str, Any]:
    """Parse NWChem TCE output: energies, frozen orbital counts, convergence."""
    contents = read_text(path)
    return _parse_tce_output(path, contents)

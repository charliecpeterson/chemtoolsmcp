from __future__ import annotations

from typing import Any

from . import (
    extract_basis_blocks,
    extract_nwchem_geometry_elements,
    parse_output,
    render_nwchem_basis_block,
    render_nwchem_basis_block_from_geometry,
    resolve_basis_set,
)


def parse_nwchem_output(file_path: str) -> dict[str, Any]:
    return parse_output(file_path, sections=["tasks", "mos", "freq", "trajectory"])


def resolve_nwchem_basis(basis_name: str, elements: list[str], library_path: str) -> dict[str, Any]:
    return resolve_basis_set(basis_name, elements, library_path)


def render_nwchem_basis(
    basis_name: str,
    library_path: str,
    elements: list[str] | None = None,
    input_file: str | None = None,
    block_name: str = "ao basis",
) -> dict[str, Any]:
    if input_file:
        return render_nwchem_basis_block_from_geometry(
            basis_name=basis_name,
            input_path=input_file,
            library_path=library_path,
            block_name=block_name,
        )
    if not elements:
        raise ValueError("provide either elements or input_file")
    return render_nwchem_basis_block(
        basis_name=basis_name,
        elements=elements,
        library_path=library_path,
        block_name=block_name,
    )


def extract_nwchem_basis_source_blocks(basis_name: str, elements: list[str], library_path: str) -> dict[str, Any]:
    return extract_basis_blocks(basis_name, elements, library_path)


def inspect_nwchem_input_geometry(input_file: str) -> dict[str, Any]:
    return extract_nwchem_geometry_elements(input_file)

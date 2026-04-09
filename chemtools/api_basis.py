from __future__ import annotations

from typing import Any

from .basis import (
    extract_basis_blocks,
    extract_nwchem_geometry_elements,
    list_basis_sets,
    render_mixed_nwchem_basis_block,
    render_mixed_nwchem_ecp_block,
    render_nwchem_ecp_block,
    render_nwchem_basis_block,
    render_nwchem_basis_block_from_geometry,
    resolve_ecp_set,
    resolve_mixed_basis_assignments,
    resolve_mixed_ecp_assignments,
    resolve_basis_set,
)
from .nwchem_input import load_geometry_source


def basis_library_summary(library_path: str) -> dict[str, Any]:
    return list_basis_sets(library_path)


def resolve_basis(basis_name: str, elements: list[str], library_path: str) -> dict[str, Any]:
    return resolve_basis_set(basis_name, elements, library_path)


def resolve_ecp(ecp_name: str, elements: list[str], library_path: str) -> dict[str, Any]:
    return resolve_ecp_set(ecp_name, elements, library_path)


def get_basis_blocks(basis_name: str, elements: list[str], library_path: str) -> dict[str, Any]:
    return extract_basis_blocks(basis_name, elements, library_path)


def inspect_nwchem_geometry(input_path: str) -> dict[str, Any]:
    return extract_nwchem_geometry_elements(input_path)


def resolve_basis_setup(
    geometry_path: str,
    library_path: str,
    basis_assignments: dict[str, str],
    ecp_assignments: dict[str, str] | None = None,
    default_basis: str | None = None,
    default_ecp: str | None = None,
    geometry_block_index: int = 0,
) -> dict[str, Any]:
    geometry = load_geometry_source(geometry_path, block_index=geometry_block_index)
    elements = list(dict.fromkeys(atom["element"] for atom in geometry["atoms"]))
    return {
        "geometry_source": geometry["file"],
        "geometry_source_kind": geometry.get("source_kind"),
        "elements": elements,
        "basis": resolve_mixed_basis_assignments(
            assignments=basis_assignments,
            elements=elements,
            library_path=library_path,
            default_basis=default_basis,
        ),
        "ecp": resolve_mixed_ecp_assignments(
            assignments=ecp_assignments or {},
            elements=elements,
            library_path=library_path,
            default_ecp=default_ecp,
        ),
    }


def render_basis_block(
    basis_name: str,
    elements: list[str],
    library_path: str,
    block_name: str = "ao basis",
    mode: str | None = None,
    inline_blocks: bool = True,
) -> dict[str, Any]:
    return render_nwchem_basis_block(
        basis_name=basis_name,
        elements=elements,
        library_path=library_path,
        block_name=block_name,
        mode=mode,
        inline_blocks=inline_blocks,
    )


def render_ecp_block(
    ecp_name: str,
    elements: list[str],
    library_path: str,
    inline_blocks: bool = True,
) -> dict[str, Any]:
    return render_nwchem_ecp_block(
        ecp_name=ecp_name,
        elements=elements,
        library_path=library_path,
        inline_blocks=inline_blocks,
    )


def render_basis_block_from_geometry(
    basis_name: str,
    input_path: str,
    library_path: str,
    block_name: str = "ao basis",
    mode: str | None = None,
    inline_blocks: bool = True,
) -> dict[str, Any]:
    return render_nwchem_basis_block_from_geometry(
        basis_name=basis_name,
        input_path=input_path,
        library_path=library_path,
        block_name=block_name,
        mode=mode,
        inline_blocks=inline_blocks,
    )


def render_nwchem_basis_setup(
    geometry_path: str,
    library_path: str,
    basis_assignments: dict[str, str],
    ecp_assignments: dict[str, str] | None = None,
    default_basis: str | None = None,
    default_ecp: str | None = None,
    basis_block_name: str = "ao basis",
    basis_mode: str | None = None,
    geometry_block_index: int = 0,
    inline_blocks: bool = True,
) -> dict[str, Any]:
    geometry = load_geometry_source(geometry_path, block_index=geometry_block_index)
    elements = list(dict.fromkeys(atom["element"] for atom in geometry["atoms"]))
    basis_block = render_mixed_nwchem_basis_block(
        assignments=basis_assignments,
        elements=elements,
        library_path=library_path,
        block_name=basis_block_name,
        default_basis=default_basis,
        mode=basis_mode,
        inline_blocks=inline_blocks,
    )
    ecp_block = render_mixed_nwchem_ecp_block(
        assignments=ecp_assignments or {},
        elements=elements,
        library_path=library_path,
        default_ecp=default_ecp,
        inline_blocks=inline_blocks,
    )
    return {
        "geometry_source": geometry["file"],
        "geometry_source_kind": geometry.get("source_kind"),
        "elements": elements,
        "basis_block": basis_block,
        "ecp_block": ecp_block,
    }

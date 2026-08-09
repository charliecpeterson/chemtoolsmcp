"""NWChem DFT workflow input drafters.

Two entry points:

  * create_nwchem_dft_workflow_input    Build a complete DFT workflow
                                        input (opt -> freq) given a
                                        geometry, basis, charge,
                                        multiplicity. Wraps the
                                        create_nwchem_input primitive
                                        with DFT-specific defaults
                                        (functional, integrals,
                                        convergence).

  * create_nwchem_dft_input_from_request  Higher-level: takes a free-form
                                          calculation request (atoms,
                                          method tag, basis, task), runs
                                          it through review_nwchem_input_request
                                          for sanity-checking, then routes
                                          to the workflow drafter.

These are the "easy path" drafters for DFT — they handle the most
common workflow patterns without requiring the agent to compose lower-
level helpers.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

from chemtools.programs.nwchem.parse.input import (
    inspect_nwchem_input,
    extract_nwchem_geometry_block,
    render_nwchem_geometry_block,
    render_nwchem_module_block,
    load_geometry_source,
)
from chemtools.programs.nwchem.input.basis import render_nwchem_basis_setup
from chemtools.programs.nwchem.input._utils import (
    _apply_default_dft_settings,
    _ensure_driver_block,
    _build_simple_input_file_plan,
    _normalize_nwchem_task_operation,
    _TRANSITION_METALS,
    _write_text_file,
)


def create_nwchem_dft_workflow_input(
    geometry_path: str,
    library_path: str,
    basis_assignments: dict[str, str],
    xc_functional: str,
    task_operations: list[str],
    *,
    ecp_assignments: dict[str, str] | None = None,
    default_basis: str | None = None,
    default_ecp: str | None = None,
    basis_block_name: str = "ao basis",
    basis_mode: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    dft_settings: list[str] | None = None,
    extra_blocks: list[str] | None = None,
    geometry_options: list[str] | None = None,
    memory: str | None = None,
    title: str | None = None,
    start_name: str | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    geometry_block_index: int = 0,
    output_dir: str | None = None,
    write_file: bool = False,
    inline_blocks: bool = True,
) -> dict[str, Any]:
    normalized_tasks = [_normalize_nwchem_task_operation(task) for task in task_operations]
    if not normalized_tasks:
        raise ValueError("at least one task operation is required")
    if not xc_functional.strip():
        raise ValueError("xc_functional is required")

    geometry = load_geometry_source(geometry_path, block_index=geometry_block_index)
    header_line = geometry["header_line"]
    # Append geometry options (e.g. noautosym, noautoz) to the header line
    if geometry_options:
        for opt in geometry_options:
            opt_stripped = opt.strip()
            if opt_stripped and opt_stripped.lower() not in header_line.lower():
                header_line = header_line.rstrip() + " " + opt_stripped
    geometry_block = render_nwchem_geometry_block(
        header_line,
        geometry["atoms"],
        directives=geometry["directives"],
    )
    basis_setup = render_nwchem_basis_setup(
        geometry_path=geometry_path,
        library_path=library_path,
        basis_assignments=basis_assignments,
        ecp_assignments=ecp_assignments,
        default_basis=default_basis,
        default_ecp=default_ecp,
        basis_block_name=basis_block_name,
        basis_mode=basis_mode,
        geometry_block_index=geometry_block_index,
        inline_blocks=inline_blocks,
    )

    resolved_start_name = start_name or Path(geometry_path).stem or "nwchem_dft_job"
    resolved_title = title or f"{resolved_start_name}: dft {'+'.join(normalized_tasks)}"
    resolved_vectors_output = vectors_output or f"{resolved_start_name}.movecs"

    rendered_dft_settings = [
        line.rstrip() if line[:1].isspace() else f"  {line.rstrip()}"
        for line in (dft_settings or [])
        if str(line).strip()
    ]
    rendered_dft_settings = _apply_default_dft_settings(
        rendered_dft_settings,
        xc_functional=xc_functional,
        multiplicity=multiplicity,
        vectors_input=vectors_input,
        vectors_output=resolved_vectors_output,
    )

    module_block = render_nwchem_module_block("dft", rendered_dft_settings)
    task_lines = [f"task dft {task}" for task in normalized_tasks]

    sections: list[str] = [f"start {resolved_start_name}", f'title "{resolved_title}"', "echo"]
    if memory:
        sections.append(f"memory {memory}")
    sections.append(geometry_block)
    sections.append(basis_setup["basis_block"]["text"])
    if basis_setup["ecp_block"]:
        sections.append(basis_setup["ecp_block"]["text"])
    if charge is not None:
        sections.append(f"charge {charge}")
    rendered_extra_blocks = [block.strip("\n") for block in (extra_blocks or []) if str(block).strip()]
    if "optimize" in normalized_tasks:
        # Default to Cartesian (xyz) optimizer for TM complexes with ≥4 ligands:
        # autoz (Z-matrix) can produce degenerate coordinates for symmetric metal cages.
        _geom_elements = {a.get("element", "") for a in geometry.get("atoms", [])}
        _has_tm = bool(_geom_elements & _TRANSITION_METALS)
        _n_heavy = sum(1 for e in _geom_elements if e not in {"H", "D"})
        _use_xyz = _has_tm and _n_heavy >= 4
        _ensure_driver_block(rendered_extra_blocks, use_xyz=_use_xyz)
    sections.extend(rendered_extra_blocks)
    sections.append(module_block)
    sections.extend(task_lines)

    input_text = "\n\n".join(sections).rstrip() + "\n"
    file_plan = _build_simple_input_file_plan(
        input_path=geometry_path,
        output_dir=output_dir,
        base_name=resolved_start_name,
    )
    written_file: str | None = None
    if write_file:
        written_file = _write_text_file(input_text, file_plan["input_file"])

    return {
        "geometry_source": geometry["file"],
        "geometry_source_kind": geometry.get("source_kind"),
        "module": "dft",
        "xc_functional": xc_functional,
        "task_operations": normalized_tasks,
        "charge": charge,
        "multiplicity": multiplicity,
        "basis_setup": (
            # When write_file=True the basis is already on disk — omit full text to save tokens
            {k: v for k, v in basis_setup.items() if k not in {"basis_block", "ecp_block"}}
            if write_file else basis_setup
        ),
        "dft_settings": [line.strip() for line in rendered_dft_settings],
        "vectors_input": vectors_input,
        "vectors_output": resolved_vectors_output,
        "input_text": None if write_file else input_text,  # omit full text when file written
        "written_file": written_file,
        "file_plan": file_plan,
        "inline_blocks": inline_blocks,
    }


def create_nwchem_dft_input_from_request(
    *,
    formula: str | None = None,
    geometry_path: str | None = None,
    library_path: str | None = None,
    basis_assignments: dict[str, str] | None = None,
    ecp_assignments: dict[str, str] | None = None,
    default_basis: str | None = None,
    default_ecp: str | None = None,
    xc_functional: str | None = None,
    task_operations: list[str] | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    dft_settings: list[str] | None = None,
    extra_blocks: list[str] | None = None,
    geometry_options: list[str] | None = None,
    memory: str | None = None,
    title: str | None = None,
    start_name: str | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    geometry_block_index: int = 0,
    output_dir: str | None = None,
    write_file: bool = False,
    inline_blocks: bool = True,
) -> dict[str, Any]:
    from chemtools.programs.nwchem.input.general import (
        review_nwchem_input_request,
    )
    review = review_nwchem_input_request(
        formula=formula,
        geometry_path=geometry_path,
        library_path=library_path,
        basis_assignments=basis_assignments,
        ecp_assignments=ecp_assignments,
        default_basis=default_basis,
        default_ecp=default_ecp,
        module="dft",
        task_operations=task_operations,
        functional=xc_functional,
        charge=charge,
        multiplicity=multiplicity,
    )

    if not review["ready_to_create"]:
        return {
            "ready_to_create": False,
            "created": False,
            "review": review,
            "next_action": "provide_missing_requirements",
            "input_text": None,
            "written_file": None,
        }

    if not geometry_path:
        raise ValueError("geometry_path is required when the request is ready to create")
    if not library_path:
        raise ValueError("library_path is required when the request is ready to create")
    if not xc_functional or not xc_functional.strip():
        raise ValueError("xc_functional is required when the request is ready to create")

    created = create_nwchem_dft_workflow_input(
        geometry_path=geometry_path,
        library_path=library_path,
        basis_assignments=basis_assignments or {},
        ecp_assignments=ecp_assignments,
        default_basis=default_basis,
        default_ecp=default_ecp,
        xc_functional=xc_functional,
        task_operations=task_operations or ["energy"],
        charge=review["charge"],
        multiplicity=multiplicity,
        dft_settings=dft_settings,
        extra_blocks=extra_blocks,
        geometry_options=geometry_options,
        memory=memory,
        title=title,
        start_name=start_name,
        vectors_input=vectors_input,
        vectors_output=vectors_output,
        geometry_block_index=geometry_block_index,
        output_dir=output_dir,
        write_file=write_file,
        inline_blocks=inline_blocks,
    )
    # When the file was written, strip the large raw text fields from the response
    # (input_text, basis_block.text, ecp_block.text) — the file on disk is the source
    # of truth and returning those texts wastes tokens.
    if write_file and created.get("written_file"):
        created = dict(created)
        created.pop("input_text", None)
        bs = created.get("basis_setup")
        if isinstance(bs, dict):
            bs = dict(bs)
            if isinstance(bs.get("basis_block"), dict):
                bb = dict(bs["basis_block"])
                bb.pop("text", None)
                bs["basis_block"] = bb
            if isinstance(bs.get("ecp_block"), dict):
                eb = dict(bs["ecp_block"])
                eb.pop("text", None)
                bs["ecp_block"] = eb
            created["basis_setup"] = bs

    result: dict[str, Any] = {
        "ready_to_create": True,
        "created": True,
        "next_action": "input_created",
        **created,
    }
    # Include warnings from the review at the top level if present, but omit the full review dict
    if review.get("warnings"):
        result["warnings"] = review["warnings"]
    return result



__all__ = [
    "create_nwchem_dft_workflow_input",
    "create_nwchem_dft_input_from_request",
]

"""NWChem cube-file input drafters.

Renders `dplot` blocks for orbital and density cube generation. Two entry
points:

  * `draft_nwchem_cube_input`         — explicit vectors + density modes
  * `draft_nwchem_frontier_cube_input` — auto-pick HOMO/LUMO/SOMOs from
                                          an existing output

Heavy text munging lives in `chemtools.programs.nwchem.input._utils`; this
module just composes the orchestration logic.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

from chemtools.core.common import read_text, make_metadata
from chemtools.programs.nwchem.parse.input import inspect_nwchem_input
from chemtools.programs.nwchem.input._utils import (
    _append_named_blocks_before_tasks,
    _build_cube_file_plan,
    _remove_keyword_blocks,
    _render_dplot_density_block,
    _render_dplot_orbital_block,
    _replace_or_insert_keyword_line,
    _replace_tasks_in_text,
    _write_text_file,
)


def draft_nwchem_cube_input(
    input_path: str,
    vectors_input: str,
    orbital_vectors: list[int] | None = None,
    density_modes: list[str] | None = None,
    orbital_spin: str = "total",
    orbital_requests: list[dict[str, Any]] | None = None,
    extent_angstrom: float = 6.0,
    grid_points: int = 120,
    gaussian: bool = True,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    resolved_orbitals = orbital_vectors or []
    resolved_orbital_requests = orbital_requests or []
    resolved_density_modes = [mode.lower() for mode in (density_modes or [])]
    if not resolved_orbitals and not resolved_density_modes and not resolved_orbital_requests:
        raise ValueError("provide at least one orbital vector or density mode for cube drafting")

    input_summary = inspect_nwchem_input(input_path)
    input_stem = Path(input_path).stem
    resolved_base_name = base_name or f"{input_stem}_cubes"

    dplot_blocks: list[str] = []
    cube_outputs: list[str] = []

    for mode in resolved_density_modes:
        if mode not in {"total", "spindens"}:
            raise ValueError("density_modes entries must be 'total' or 'spindens'")
        output_name = f"{resolved_base_name}_{mode}.cube"
        cube_outputs.append(output_name)
        dplot_blocks.append(
            _render_dplot_density_block(
                vectors_input=vectors_input,
                output_name=output_name,
                density_mode=mode,
                extent_angstrom=extent_angstrom,
                grid_points=grid_points,
                gaussian=gaussian,
            )
        )

    for vector_number in resolved_orbitals:
        output_name = f"{resolved_base_name}_mo_{vector_number:03d}.cube"
        cube_outputs.append(output_name)
        dplot_blocks.append(
            _render_dplot_orbital_block(
                vectors_input=vectors_input,
                output_name=output_name,
                vector_number=vector_number,
                spin=orbital_spin,
                title=f"Orbital {vector_number}",
                extent_angstrom=extent_angstrom,
                grid_points=grid_points,
                gaussian=gaussian,
            )
        )

    for request in resolved_orbital_requests:
        vector_number = int(request["vector_number"])
        spin = str(request.get("spin") or orbital_spin)
        output_name = request.get("output_name") or f"{resolved_base_name}_{spin}_mo_{vector_number:03d}.cube"
        cube_outputs.append(output_name)
        dplot_blocks.append(
            _render_dplot_orbital_block(
                vectors_input=vectors_input,
                output_name=output_name,
                vector_number=vector_number,
                spin=spin,
                title=request.get("title") or f"{spin.capitalize()} orbital {vector_number}",
                extent_angstrom=extent_angstrom,
                grid_points=grid_points,
                gaussian=gaussian,
            )
        )

    contents = read_text(input_path)
    cleaned = _remove_keyword_blocks(contents, {"dplot", "property", "driver"})
    cleaned = _append_named_blocks_before_tasks(cleaned, dplot_blocks)
    replaced_tasks = _replace_tasks_in_text(input_path, cleaned, ["task dplot"])
    final_text = replaced_tasks["text"]
    resolved_title = title or f'{resolved_base_name}: cube generation from chosen vectors'
    final_text = _replace_or_insert_keyword_line(final_text, "start", f"start {resolved_base_name}")
    final_text = _replace_or_insert_keyword_line(final_text, "title", f'title "{resolved_title}"', insert_after="start")

    file_plan = _build_cube_file_plan(
        input_path=input_path,
        output_dir=output_dir,
        base_name=resolved_base_name,
        cube_outputs=cube_outputs,
    )
    written_file: str | None = None
    if write_file:
        written_file = _write_text_file(final_text, file_plan["input_file"])

    return {
        "input_file": input_path,
        "input_summary": input_summary,
        "vectors_input": vectors_input,
        "orbital_vectors": resolved_orbitals,
        "orbital_requests": resolved_orbital_requests,
        "density_modes": resolved_density_modes,
        "orbital_spin": orbital_spin,
        "extent_angstrom": extent_angstrom,
        "grid_points": grid_points,
        "dplot_block_count": len(dplot_blocks),
        "dplot_blocks": dplot_blocks,
        "input_text": final_text,
        "file_plan": file_plan,
        "written_file": written_file,
    }


def draft_nwchem_frontier_cube_input(
    output_path: str,
    input_path: str,
    vectors_input: str | None = None,
    include_somos: bool = True,
    include_homo: bool = True,
    include_lumo: bool = True,
    include_density_modes: list[str] | None = None,
    extent_angstrom: float = 6.0,
    grid_points: int = 120,
    gaussian: bool = True,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    # parse_mos lives in programs/nwchem/output.py (the 1-arg wrapper).
    from chemtools.programs.nwchem.output import parse_mos
    mos = parse_mos(output_path, top_n=12)
    resolved_base_name = base_name or f"{Path(input_path).stem}_frontier_cubes"
    resolved_vectors_input = vectors_input or f"{Path(output_path).stem}.movecs"
    density_modes = include_density_modes or []

    orbital_requests: list[dict[str, Any]] = []
    seen_vectors: set[tuple[str, int]] = set()
    for spin, channel in mos.get("spin_channels", {}).items():
        if include_somos:
            for index, orbital in enumerate(channel.get("somos", []), start=1):
                vector_number = orbital["vector_number"]
                key = (spin, vector_number)
                if key in seen_vectors:
                    continue
                seen_vectors.add(key)
                orbital_requests.append(
                    {
                        "spin": spin,
                        "vector_number": vector_number,
                        "title": f"{spin.capitalize()} SOMO {index} (vector {vector_number})",
                        "output_name": f"{resolved_base_name}_{spin}_somo_{index}_v{vector_number:03d}.cube",
                    }
                )
        if include_homo and (orbital := channel.get("homo")) is not None:
            vector_number = orbital["vector_number"]
            key = (spin, vector_number)
            if key not in seen_vectors:
                seen_vectors.add(key)
                orbital_requests.append(
                    {
                        "spin": spin,
                        "vector_number": vector_number,
                        "title": f"{spin.capitalize()} HOMO (vector {vector_number})",
                        "output_name": f"{resolved_base_name}_{spin}_homo_v{vector_number:03d}.cube",
                    }
                )
        if include_lumo and (orbital := channel.get("lumo")) is not None:
            vector_number = orbital["vector_number"]
            key = (spin, vector_number)
            if key not in seen_vectors:
                seen_vectors.add(key)
                orbital_requests.append(
                    {
                        "spin": spin,
                        "vector_number": vector_number,
                        "title": f"{spin.capitalize()} LUMO (vector {vector_number})",
                        "output_name": f"{resolved_base_name}_{spin}_lumo_v{vector_number:03d}.cube",
                    }
                )

    drafted = draft_nwchem_cube_input(
        input_path=input_path,
        vectors_input=resolved_vectors_input,
        orbital_requests=orbital_requests,
        density_modes=density_modes,
        extent_angstrom=extent_angstrom,
        grid_points=grid_points,
        gaussian=gaussian,
        output_dir=output_dir,
        base_name=resolved_base_name,
        title=title,
        write_file=write_file,
    )
    drafted.update(
        {
            "output_file": output_path,
            "frontier_requests": orbital_requests,
            "metadata": make_metadata(output_path, read_text(output_path), "nwchem"),
        }
    )
    return drafted


__all__ = ["draft_nwchem_cube_input", "draft_nwchem_frontier_cube_input"]

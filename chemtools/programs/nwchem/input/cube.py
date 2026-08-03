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
import re
from typing import Any

from chemtools.core.common import read_text, make_metadata
from chemtools.core.units import ANGSTROM_PER_BOHR
from chemtools.programs.nwchem.parse.input import (
    extract_nwchem_geometry_block,
    inspect_nwchem_input,
)
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


_GEOMETRY_UNITS_RE = re.compile(
    r"\bunits\s+(angstroms?|au|a\.u\.?|bohrs?)\b",
    re.IGNORECASE,
)
_PYSCF_CUBE_MARGIN_BOHR = 3.0


def draft_nwchem_cube_input(
    input_path: str,
    vectors_input: str,
    orbital_vectors: list[int] | None = None,
    density_modes: list[str] | None = None,
    orbital_spin: str = "total",
    orbital_requests: list[dict[str, Any]] | None = None,
    extent_angstrom: float = 6.0,
    grid_points: int = 120,
    pyscf_compatible_grid_points: int | None = None,
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
    cube_grid = _pyscf_compatible_cube_grid(
        input_path,
        input_summary,
        pyscf_compatible_grid_points,
    )
    limitxyz_lines = _pyscf_limitxyz_lines(cube_grid)

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
                limitxyz_lines=limitxyz_lines,
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
                limitxyz_lines=limitxyz_lines,
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
                limitxyz_lines=limitxyz_lines,
            )
        )

    contents = read_text(input_path)
    if cube_grid is not None:
        contents = _preserve_input_cartesian_frame(contents, input_path)
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
        "cube_grid": cube_grid,
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
    pyscf_compatible_grid_points: int | None = None,
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
        pyscf_compatible_grid_points=pyscf_compatible_grid_points,
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


def _pyscf_compatible_cube_grid(
    input_path: str,
    input_summary: dict[str, Any],
    grid_points: int | None,
) -> dict[str, Any] | None:
    if grid_points is None:
        return None
    if isinstance(grid_points, bool) or not isinstance(grid_points, int) or not 20 <= grid_points <= 120:
        raise ValueError("pyscf_compatible_grid_points must be an integer between 20 and 120")
    if input_summary["geometry_block_count"] != 1:
        raise ValueError(
            "pyscf-compatible CUBE drafting requires exactly one Cartesian geometry block"
        )
    geometry = extract_nwchem_geometry_block(input_path)
    units = _geometry_units(geometry["header_line"])
    if units is None:
        raise ValueError(
            "pyscf-compatible CUBE drafting requires explicit geometry units"
        )
    coordinate_factor = 1.0 if units == "bohr" else 1.0 / ANGSTROM_PER_BOHR
    coordinates_bohr = [
        [atom[axis] * coordinate_factor for axis in ("x", "y", "z")]
        for atom in geometry["atoms"]
    ]
    lower_bounds = [min(axis) - _PYSCF_CUBE_MARGIN_BOHR for axis in zip(*coordinates_bohr)]
    upper_bounds = [max(axis) + _PYSCF_CUBE_MARGIN_BOHR for axis in zip(*coordinates_bohr)]
    return {
        "kind": "pyscf_compatible",
        "source_geometry_units": units,
        "coordinate_unit": "bohr",
        "lower_bounds_bohr": lower_bounds,
        "upper_bounds_bohr": upper_bounds,
        "grid_points": [grid_points, grid_points, grid_points],
        "nwchem_spacings": [grid_points - 1, grid_points - 1, grid_points - 1],
        "pyscf_margin_bohr": _PYSCF_CUBE_MARGIN_BOHR,
        "preserve_input_cartesian_frame": True,
    }


def _pyscf_limitxyz_lines(cube_grid: dict[str, Any] | None) -> list[str] | None:
    if cube_grid is None:
        return None
    return [
        "  limitxyz units bohr",
        *[
            f"   {lower:.12f}  {upper:.12f}  {spacings}"
            for lower, upper, spacings in zip(
                cube_grid["lower_bounds_bohr"],
                cube_grid["upper_bounds_bohr"],
                cube_grid["nwchem_spacings"],
            )
        ],
    ]


def _preserve_input_cartesian_frame(contents: str, input_path: str) -> str:
    geometry = extract_nwchem_geometry_block(input_path)
    header = geometry["header_line"].rstrip()
    header_words = header.lower().split()
    for control in ("nocenter", "noautosym", "noautoz"):
        if control not in header_words:
            header += f" {control}"
    lines = contents.splitlines()
    lines[geometry["start_line"]] = header
    return "\n".join(lines) + ("\n" if contents.endswith("\n") else "")


def _geometry_units(header_line: str) -> str | None:
    match = _GEOMETRY_UNITS_RE.search(header_line)
    if match is None:
        return None
    return "angstrom" if match.group(1).lower().startswith("angstrom") else "bohr"


__all__ = ["draft_nwchem_cube_input", "draft_nwchem_frontier_cube_input"]

"""NWChem imaginary-mode handling drafters.

Three closely-related functions for the TS / imaginary-frequency workflow:

  * analyze_imaginary_modes        Detect imaginary modes in a frequency
                                   output and characterize the affected
                                   atoms.
  * displace_geometry_along_mode   Generate +/- displaced geometries
                                   along a chosen imaginary mode.
  * draft_nwchem_imaginary_mode_inputs
                                   Build paired optimize / opt-then-freq
                                   inputs from the displaced geometries.

All three are thin MCP wrappers that auto-detect the program. Heavy
lifting (parsing modes, building file plans, writing displaced inputs)
is in parse/freq.py and input/_utils.py.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

from chemtools.core.common import read_text, detect_program, make_metadata
from chemtools.programs.nwchem.parse.freq import (
    analyze_imaginary_modes as _analyze_imaginary_modes_raw,
    displace_geometry_along_mode as _displace_geometry_along_mode_raw,
)
from chemtools.programs.nwchem.parse.input import (
    inspect_nwchem_input,
    extract_nwchem_geometry_block,
    render_nwchem_geometry_block,
    replace_nwchem_geometry_block,
)
from chemtools.programs.nwchem.input._utils import (
    _build_imaginary_follow_up_plan,
    _build_imaginary_output_file_plan,
    _ensure_module_vectors_output_in_text,
    _replace_tasks_in_text,
    _write_imaginary_input_files,
)


def analyze_imaginary_modes(
    path: str,
    significant_threshold_cm1: float = 20.0,
    top_atoms: int = 4,
    detail: str = "compact",
) -> dict[str, Any]:
    contents = read_text(path)
    program = detect_program(contents)
    if program == "nwchem":
        return _analyze_imaginary_modes_raw(
            path,
            contents,
            significant_threshold_cm1=significant_threshold_cm1,
            top_atoms=top_atoms,
            detail=detail,
        )
    raise ValueError(f"imaginary mode analysis is not implemented for {program or 'unknown'}")


def displace_geometry_along_mode(
    path: str,
    mode_number: int | None = None,
    amplitude_angstrom: float = 0.15,
    significant_threshold_cm1: float = 20.0,
) -> dict[str, Any]:
    contents = read_text(path)
    program = detect_program(contents)
    if program == "nwchem":
        return _displace_geometry_along_mode_raw(
            path,
            contents,
            mode_number=mode_number,
            amplitude_angstrom=amplitude_angstrom,
            significant_threshold_cm1=significant_threshold_cm1,
        )
    raise ValueError(f"mode displacement is not implemented for {program or 'unknown'}")


def draft_nwchem_imaginary_mode_inputs(
    output_path: str,
    input_path: str,
    mode_number: int | None = None,
    amplitude_angstrom: float = 0.15,
    significant_threshold_cm1: float = 20.0,
    add_noautosym: bool = True,
    enforce_symmetry_c1: bool = True,
    block_index: int = 0,
    task_strategy: str = "auto",
    output_dir: str | None = None,
    base_name: str | None = None,
    write_files: bool = False,
) -> dict[str, Any]:
    if task_strategy not in {"auto", "optimize_only", "optimize_then_freq"}:
        raise ValueError("task_strategy must be one of: auto, optimize_only, optimize_then_freq")

    displaced = displace_geometry_along_mode(
        output_path,
        mode_number=mode_number,
        amplitude_angstrom=amplitude_angstrom,
        significant_threshold_cm1=significant_threshold_cm1,
    )
    geometry = extract_nwchem_geometry_block(input_path, block_index=block_index)
    input_summary = inspect_nwchem_input(input_path)
    header_line = geometry["header_line"]
    if add_noautosym and "noautosym" not in header_line.lower():
        header_line = header_line.rstrip() + " noautosym"

    directives = list(geometry["directives"])
    if enforce_symmetry_c1:
        directives = [directive for directive in directives if not directive.lower().startswith("symmetry ")]
        directives.insert(0, "symmetry c1")

    plus_block = render_nwchem_geometry_block(header_line, displaced["plus_geometry"]["atoms"], directives=directives)
    minus_block = render_nwchem_geometry_block(
        header_line,
        displaced["minus_geometry"]["atoms"],
        directives=directives,
    )

    plus_geometry_replaced = replace_nwchem_geometry_block(input_path, plus_block, block_index=block_index)
    minus_geometry_replaced = replace_nwchem_geometry_block(input_path, minus_block, block_index=block_index)

    follow_up_plan = _build_imaginary_follow_up_plan(
        input_summary=input_summary,
        stability_assessment=displaced.get("stability_assessment"),
        selected_mode=displaced["selected_mode"],
        task_strategy=task_strategy,
    )
    plus_input = _replace_tasks_in_text(input_path, plus_geometry_replaced["text"], follow_up_plan["task_lines"])
    minus_input = _replace_tasks_in_text(input_path, minus_geometry_replaced["text"], follow_up_plan["task_lines"])

    file_plan = _build_imaginary_output_file_plan(
        input_path=input_path,
        selected_mode=displaced["selected_mode"],
        output_dir=output_dir,
        base_name=base_name,
    )
    plus_vectors_output = f"{Path(file_plan['plus_file']).stem}.movecs"
    minus_vectors_output = f"{Path(file_plan['minus_file']).stem}.movecs"
    plus_text, _ = _ensure_module_vectors_output_in_text(
        plus_input["text"],
        module=follow_up_plan["module"],
        vectors_output=plus_vectors_output,
    )
    minus_text, _ = _ensure_module_vectors_output_in_text(
        minus_input["text"],
        module=follow_up_plan["module"],
        vectors_output=minus_vectors_output,
    )
    written_files: dict[str, str] | None = None
    if write_files:
        written_files = _write_imaginary_input_files(
            plus_text=plus_text,
            minus_text=minus_text,
            plus_path=file_plan["plus_file"],
            minus_path=file_plan["minus_file"],
        )

    return {
        "metadata": make_metadata(output_path, read_text(output_path), "nwchem"),
        "input_file": input_path,
        "selected_mode": displaced["selected_mode"],
        "amplitude_angstrom": displaced["amplitude_angstrom"],
        "stability_assessment": displaced.get("stability_assessment"),
        "follow_up_plan": follow_up_plan,
        "file_plan": file_plan,
        "plus_vectors_output": plus_vectors_output,
        "minus_vectors_output": minus_vectors_output,
        "plus_geometry_block": plus_block,
        "minus_geometry_block": minus_block,
        "plus_input_text": plus_text,
        "minus_input_text": minus_text,
        "written_files": written_files,
    }


__all__ = [
    "analyze_imaginary_modes",
    "displace_geometry_along_mode",
    "draft_nwchem_imaginary_mode_inputs",
]

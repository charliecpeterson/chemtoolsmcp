"""NWChem SCF/state-recovery and property-check drafters.

Three closely-related drafters that all rewrite a single module body
(scf / dft / mcscf) and route the rebuilt input back through NWChem with
different intents:

  * draft_nwchem_vectors_swap_input     — apply suggested SOMO swaps and
                                          restart from a corrected guess
  * draft_nwchem_property_check_input   — request property data (mulliken,
                                          spin density, ...) from chosen
                                          vectors; auto-downgrades to an
                                          energy run if the reference state
                                          looks unstable
  * draft_nwchem_scf_stabilization_input — relax convergence (damping,
                                           smearing, iter count) to push a
                                           stuck SCF back to convergence

All three share the same machinery in _utils.py (module-body rewriters,
keyword-block manipulators, file-plan builder, _select_*_strategy helpers).
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

from chemtools.core.common import read_text, detect_program, make_metadata
from chemtools.programs.nwchem.parse.input import (
    inspect_nwchem_input,
    inspect_nwchem_module_vectors,
    extract_nwchem_module_block,
    replace_nwchem_module_block,
    render_nwchem_module_block,
)
from chemtools.programs.nwchem.output import suggest_vectors_swaps
from chemtools.programs.nwchem.strategy.diagnose import diagnose_nwchem_output
from chemtools.programs.nwchem.input._utils import (
    _select_primary_task_module,
    _select_scf_stabilization_strategy,
    _rewrite_module_body_for_vectors_swap,
    _rewrite_module_body_for_property_check,
    _rewrite_module_body_for_scf_stabilization,
    _extract_vectors_io_from_lines,
    _replace_tasks_in_text,
    _remove_keyword_blocks,
    _replace_or_insert_keyword_line,
    _replace_or_insert_named_block,
    _render_named_block,
    _build_vectors_swap_file_plan,
    _write_text_file,
)


def draft_nwchem_vectors_swap_input(
    output_path: str,
    input_path: str,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    module: str | None = None,
    block_index: int = -1,
    task_operation: str = "energy",
    iterations: int | None = 500,
    smear: float | None = 0.001,
    convergence_damp: int | None = 30,
    convergence_ncydp: int | None = 30,
    population_print: str | None = "mulliken",
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    contents = read_text(output_path)
    program = detect_program(contents)
    if program != "nwchem":
        raise ValueError(f"vectors swap drafting is not implemented for {program or 'unknown'}")

    input_summary = inspect_nwchem_input(input_path)
    module_name = module or _select_primary_task_module(input_summary)
    resolved_base_name = base_name or f"{Path(input_path).stem}_swap"
    resolved_vectors_input = vectors_input or f"{Path(output_path).stem}.movecs"
    resolved_vectors_output = vectors_output or f"{resolved_base_name}.movecs"

    suggestion_payload = suggest_vectors_swaps(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
        vectors_input=resolved_vectors_input,
        vectors_output=resolved_vectors_output,
    )
    suggestion = suggestion_payload["suggestion"]
    if not suggestion.get("available"):
        raise ValueError("no actionable vectors swap suggestion was identified for this output")

    module_block = extract_nwchem_module_block(input_path, module=module_name, block_index=block_index)
    rewritten_body_lines = _rewrite_module_body_for_vectors_swap(
        module_block["body_lines"],
        suggestion["vectors_block"],
        iterations=iterations,
        smear=smear,
        convergence_damp=convergence_damp,
        convergence_ncydp=convergence_ncydp,
        population_print=population_print,
    )
    rewritten_module_block = render_nwchem_module_block(module_block["header_line"], rewritten_body_lines)
    replaced_module = replace_nwchem_module_block(
        input_path,
        rewritten_module_block,
        module=module_name,
        block_index=block_index,
    )

    task_lines = [f"task {module_name} {task_operation}"]
    replaced_tasks = _replace_tasks_in_text(input_path, replaced_module["text"], task_lines)

    final_text = replaced_tasks["text"]
    if task_operation in {"energy", "property", "freq"}:
        final_text = _remove_keyword_blocks(final_text, {"driver"})
    if task_operation != "property":
        final_text = _remove_keyword_blocks(final_text, {"property"})
    resolved_title = title or f'{resolved_base_name}: push metal-centered orbitals into SOMO positions'
    final_text = _replace_or_insert_keyword_line(final_text, "start", f"start {resolved_base_name}")
    final_text = _replace_or_insert_keyword_line(final_text, "title", f'title "{resolved_title}"', insert_after="start")

    file_plan = _build_vectors_swap_file_plan(
        input_path=input_path,
        output_dir=output_dir,
        base_name=resolved_base_name,
        vectors_output=resolved_vectors_output,
    )
    written_file: str | None = None
    if write_file:
        written_file = _write_text_file(final_text, file_plan["input_file"])

    return {
        "metadata": make_metadata(output_path, contents, "nwchem"),
        "input_file": input_path,
        "input_summary": input_summary,
        "module": module_name,
        "module_block_index": block_index,
        "task_lines": task_lines,
        "suggestion": suggestion,
        "vectors_input": resolved_vectors_input,
        "vectors_output": resolved_vectors_output,
        "module_block_text": rewritten_module_block,
        "input_text": final_text,
        "file_plan": file_plan,
        "written_file": written_file,
    }


def draft_nwchem_property_check_input(
    input_path: str,
    reference_output_path: str | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    module: str | None = None,
    block_index: int = -1,
    property_keywords: list[str] | None = None,
    task_strategy: str = "auto",
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    iterations: int | None = 1,
    convergence_energy: str | None = "1e3",
    smear: float | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    from chemtools.programs.nwchem.strategy.case_review import (
        check_spin_charge_state,
    )

    input_summary = inspect_nwchem_input(input_path)
    module_name = module or _select_primary_task_module(input_summary)
    input_stem = Path(input_path).stem
    resolved_base_name = base_name or f"{input_stem}_prop"
    resolved_vectors_input = vectors_input or f"{input_stem}.movecs"
    resolved_vectors_output = vectors_output or f"{resolved_base_name}.movecs"
    resolved_property_keywords = property_keywords or ["mulliken"]
    if task_strategy not in {"auto", "property", "energy"}:
        raise ValueError("task_strategy must be one of: auto, property, energy")

    selected_task_operation = "property"
    selected_iterations = iterations
    selected_convergence_energy = convergence_energy
    selected_smear = smear
    strategy_notes: list[str] = []
    reference_state_review: dict[str, Any] | None = None
    reference_diagnosis: dict[str, Any] | None = None

    if task_strategy == "energy":
        selected_task_operation = "energy"
        strategy_notes.append("explicit_energy_strategy_requested")
    elif task_strategy == "auto" and reference_output_path:
        reference_state_review = check_spin_charge_state(
            output_path=reference_output_path,
            input_path=input_path,
            expected_metal_elements=expected_metal_elements,
            expected_somo_count=expected_somo_count,
        )
        reference_diagnosis = diagnose_nwchem_output(
            output_path=reference_output_path,
            input_path=input_path,
            expected_metal_elements=expected_metal_elements,
            expected_somo_count=expected_somo_count,
        )
        if (
            reference_state_review.get("assessment") == "suspicious"
            or reference_diagnosis.get("failure_class") in {"wrong_state_convergence", "scf_nonconvergence"}
        ):
            selected_task_operation = "energy"
            strategy_notes.append("auto_strategy_downgraded_to_energy_due_to_unstable_or_suspicious_state")

    if selected_task_operation == "energy":
        if selected_iterations in {None, 1}:
            selected_iterations = 80
        if selected_convergence_energy == "1e3":
            selected_convergence_energy = None

    module_block = extract_nwchem_module_block(input_path, module=module_name, block_index=block_index)
    rewritten_body_lines = _rewrite_module_body_for_property_check(
        module_block["body_lines"],
        vectors_input=resolved_vectors_input,
        vectors_output=resolved_vectors_output,
        iterations=selected_iterations,
        convergence_energy=selected_convergence_energy,
        smear=selected_smear,
        include_mulliken_in_module=selected_task_operation == "energy"
        and "mulliken" in {keyword.strip().lower() for keyword in resolved_property_keywords},
    )
    rewritten_module_block = render_nwchem_module_block(module_block["header_line"], rewritten_body_lines)
    replaced_module = replace_nwchem_module_block(
        input_path,
        rewritten_module_block,
        module=module_name,
        block_index=block_index,
    )

    property_block = _render_named_block("property", [f"  {keyword}" for keyword in resolved_property_keywords])
    if selected_task_operation == "property":
        with_property = _replace_or_insert_named_block(
            replaced_module["text"],
            "property",
            property_block,
            insert_before_task=True,
        )
        with_property = _remove_keyword_blocks(with_property, {"driver"})
        task_lines = [f"task {module_name} property"]
    else:
        with_property = _remove_keyword_blocks(replaced_module["text"], {"property", "driver"})
        task_lines = [f"task {module_name} energy"]
    replaced_tasks = _replace_tasks_in_text(input_path, with_property, task_lines)

    final_text = replaced_tasks["text"]
    resolved_title = title or (
        f'{resolved_base_name}: property check from chosen vectors'
        if selected_task_operation == "property"
        else f'{resolved_base_name}: state-check energy run from chosen vectors'
    )
    final_text = _replace_or_insert_keyword_line(final_text, "start", f"start {resolved_base_name}")
    final_text = _replace_or_insert_keyword_line(final_text, "title", f'title "{resolved_title}"', insert_after="start")

    file_plan = _build_vectors_swap_file_plan(
        input_path=input_path,
        output_dir=output_dir,
        base_name=resolved_base_name,
        vectors_output=resolved_vectors_output,
    )
    written_file: str | None = None
    if write_file:
        written_file = _write_text_file(final_text, file_plan["input_file"])

    return {
        "input_file": input_path,
        "input_summary": input_summary,
        "module": module_name,
        "module_block_index": block_index,
        "property_keywords": resolved_property_keywords,
        "task_strategy": task_strategy,
        "selected_task_operation": selected_task_operation,
        "strategy_notes": strategy_notes,
        "reference_output_file": reference_output_path,
        "reference_state_review": reference_state_review,
        "reference_diagnosis": reference_diagnosis,
        "task_lines": task_lines,
        "vectors_input": resolved_vectors_input,
        "vectors_output": resolved_vectors_output,
        "module_block_text": rewritten_module_block,
        "property_block_text": property_block,
        "input_text": final_text,
        "file_plan": file_plan,
        "written_file": written_file,
    }


def draft_nwchem_scf_stabilization_input(
    input_path: str,
    reference_output_path: str | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    module: str | None = None,
    block_index: int = -1,
    task_operation: str = "energy",
    iterations: int | None = None,
    smear: float | None = None,
    convergence_damp: int | None = None,
    convergence_ncydp: int | None = None,
    population_print: str | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    input_summary = inspect_nwchem_input(input_path)
    module_name = module or _select_primary_task_module(input_summary)
    input_stem = Path(input_path).stem
    resolved_base_name = base_name or f"{input_stem}_stabilize"

    module_vectors = inspect_nwchem_module_vectors(input_path, module=module_name, block_index=block_index)
    existing_vectors_input, existing_vectors_output = _extract_vectors_io_from_lines(module_vectors.get("vectors_lines") or [])
    resolved_vectors_input = vectors_input or existing_vectors_output or existing_vectors_input or f"{input_stem}.movecs"
    resolved_vectors_output = vectors_output or f"{resolved_base_name}.movecs"
    reference_diagnosis = None
    if reference_output_path:
        try:
            reference_diagnosis = diagnose_nwchem_output(output_path=reference_output_path, input_path=input_path)
        except Exception:  # reference output may be incomplete or from a failed run
            reference_diagnosis = None

    stabilization_strategy = _select_scf_stabilization_strategy(
        reference_diagnosis=reference_diagnosis,
        iterations=iterations,
        smear=smear,
        convergence_damp=convergence_damp,
        convergence_ncydp=convergence_ncydp,
        population_print=population_print,
    )

    module_block = extract_nwchem_module_block(input_path, module=module_name, block_index=block_index)
    rewritten_body_lines = _rewrite_module_body_for_scf_stabilization(
        module_block["body_lines"],
        vectors_input=resolved_vectors_input,
        vectors_output=resolved_vectors_output,
        iterations=stabilization_strategy["iterations"],
        smear=stabilization_strategy["smear"],
        convergence_damp=stabilization_strategy["convergence_damp"],
        convergence_ncydp=stabilization_strategy["convergence_ncydp"],
        population_print=stabilization_strategy["population_print"],
    )
    rewritten_module_block = render_nwchem_module_block(module_block["header_line"], rewritten_body_lines)
    replaced_module = replace_nwchem_module_block(
        input_path,
        rewritten_module_block,
        module=module_name,
        block_index=block_index,
    )

    final_text = _remove_keyword_blocks(replaced_module["text"], {"driver", "property"})
    task_lines = [f"task {module_name} {task_operation}"]
    replaced_tasks = _replace_tasks_in_text(input_path, final_text, task_lines)
    final_text = replaced_tasks["text"]
    resolved_title = title or f'{resolved_base_name}: stabilize SCF/state from previous vectors'
    final_text = _replace_or_insert_keyword_line(final_text, "start", f"start {resolved_base_name}")
    final_text = _replace_or_insert_keyword_line(final_text, "title", f'title "{resolved_title}"', insert_after="start")

    file_plan = _build_vectors_swap_file_plan(
        input_path=input_path,
        output_dir=output_dir,
        base_name=resolved_base_name,
        vectors_output=resolved_vectors_output,
    )
    written_file: str | None = None
    if write_file:
        written_file = _write_text_file(final_text, file_plan["input_file"])

    return {
        "input_file": input_path,
        "reference_output_file": reference_output_path,
        "reference_diagnosis": reference_diagnosis,
        "input_summary": input_summary,
        "module": module_name,
        "module_block_index": block_index,
        "task_operation": task_operation,
        "stabilization_strategy": stabilization_strategy["strategy"],
        "strategy_notes": stabilization_strategy["notes"],
        "iterations": stabilization_strategy["iterations"],
        "smear": stabilization_strategy["smear"],
        "convergence_damp": stabilization_strategy["convergence_damp"],
        "convergence_ncydp": stabilization_strategy["convergence_ncydp"],
        "population_print": stabilization_strategy["population_print"],
        "vectors_input": resolved_vectors_input,
        "vectors_output": resolved_vectors_output,
        "module_block_text": rewritten_module_block,
        "task_lines": task_lines,
        "input_text": final_text,
        "file_plan": file_plan,
        "written_file": written_file,
    }


__all__ = [
    "draft_nwchem_vectors_swap_input",
    "draft_nwchem_property_check_input",
    "draft_nwchem_scf_stabilization_input",
]

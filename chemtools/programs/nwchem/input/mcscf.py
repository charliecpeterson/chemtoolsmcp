"""NWChem MCSCF / CASSCF input drafters.

Two entry points:

  * draft_nwchem_mcscf_input        Render an MCSCF input from a recommended
                                    CAS(M,N) active space. Pairs with
                                    `prepare_nwchem_mcscf_setup` upstream.

  * draft_nwchem_mcscf_retry_input  Diagnose a previous MCSCF failure and
                                    re-render with adjusted convergence /
                                    active-space strategy.

Both draft functions delegate the messy details (active-space selection,
diagnosis review, module-body rewrites, file-plan building) to helpers
already moved into _utils / strategy / output modules.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

from chemtools.core.common import read_text
from chemtools.programs.nwchem.parse.input import (
    inspect_nwchem_input,
    inspect_nwchem_module_vectors,
)
from chemtools.programs.nwchem.output import parse_mcscf_output
from chemtools.programs.nwchem.input._utils import (
    _coerce_api_int,
    _coerce_api_float,
    _select_primary_task_module,
    _extract_vectors_io_from_lines,
    _build_mcscf_reorder_plan,
    _render_mcscf_block,
    _remove_keyword_blocks,
    _replace_or_insert_named_block,
    _replace_tasks_in_text,
    _replace_or_insert_keyword_line,
    _build_vectors_swap_file_plan,
    _write_text_file,
)


def draft_nwchem_mcscf_input(
    input_path: str,
    reference_output_path: str,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    active_space_mode: str = "minimal",
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    state_label: str | None = None,
    symmetry: int | None = None,
    hessian: str = "exact",
    maxiter: int = 80,
    thresh: float | None = 1.0e-5,
    level: float | None = 0.6,
    lock_vectors: bool = True,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    # Lazy imports — api_strategy is still flat; clean up after Phase 17 split.
    from chemtools.api_strategy import suggest_nwchem_mcscf_active_space

    if active_space_mode not in {"minimal", "expanded"}:
        raise ValueError("active_space_mode must be 'minimal' or 'expanded'")
    if hessian not in {"exact", "onel"}:
        raise ValueError("hessian must be 'exact' or 'onel'")

    input_summary = inspect_nwchem_input(input_path)
    active_space_payload = suggest_nwchem_mcscf_active_space(
        output_path=reference_output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
    )
    active_space = active_space_payload[f"{active_space_mode}_active_space"]
    if not active_space.get("active_orbitals"):
        parsed_mcscf = parse_mcscf_output(reference_output_path)
        parsed_settings = parsed_mcscf.get("settings") or {}
        fallback_active_orbitals = _coerce_api_int(parsed_settings.get("active_orbitals"))
        fallback_active_electrons = _coerce_api_int(parsed_settings.get("active_electrons"))
        fallback_inactive_shells = _coerce_api_int(parsed_settings.get("inactive_shells")) or 0
        if fallback_active_orbitals and fallback_active_electrons is not None:
            active_space = {
                "active_electrons": fallback_active_electrons,
                "active_orbitals": fallback_active_orbitals,
                "occupied_like_count": 0,
                "virtual_like_count": 0,
                "closed_shell_count": fallback_inactive_shells,
                "vector_numbers": [],
                "orbitals": [],
            }

    resolved_base_name = base_name or f"{Path(input_path).stem}_mcscf"
    primary_module = _select_primary_task_module(input_summary)
    try:
        module_vectors = inspect_nwchem_module_vectors(input_path, module=primary_module, block_index=-1)
        existing_vectors_input, existing_vectors_output = _extract_vectors_io_from_lines(module_vectors.get("vectors_lines") or [])
    except Exception:  # module block may be absent (e.g. no dft block in a scf-only input)
        existing_vectors_input, existing_vectors_output = (None, None)

    resolved_vectors_input = (
        vectors_input
        or existing_vectors_output
        or existing_vectors_input
        or f"{Path(reference_output_path).stem}.movecs"
    )
    resolved_vectors_output = vectors_output or f"{resolved_base_name}.movecs"

    reorder_plan = _build_mcscf_reorder_plan(active_space)
    mcscf_block = _render_mcscf_block(
        active_space=active_space,
        multiplicity=input_summary.get("multiplicity"),
        vectors_input=resolved_vectors_input,
        vectors_output=resolved_vectors_output,
        state_label=state_label,
        symmetry=symmetry,
        hessian=hessian,
        maxiter=maxiter,
        thresh=thresh,
        level=level,
        lock_vectors=lock_vectors,
        swap_pairs=reorder_plan["swap_pairs"],
    )

    contents = read_text(input_path)
    cleaned = _remove_keyword_blocks(contents, {"dft", "scf", "property", "driver", "mcscf"})
    with_mcscf = _replace_or_insert_named_block(cleaned, "mcscf", mcscf_block, insert_before_task=True)
    replaced_tasks = _replace_tasks_in_text(input_path, with_mcscf, ["task mcscf"])
    final_text = replaced_tasks["text"]

    resolved_title = title or f'{resolved_base_name}: mcscf from recommended active space'
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
        "input_summary": input_summary,
        "active_space_mode": active_space_mode,
        "active_space": active_space,
        "reorder_plan": reorder_plan,
        "vectors_input": resolved_vectors_input,
        "vectors_output": resolved_vectors_output,
        "state_label": state_label,
        "symmetry": symmetry,
        "hessian": hessian,
        "maxiter": maxiter,
        "thresh": thresh,
        "level": level,
        "lock_vectors": lock_vectors,
        "module_block_text": mcscf_block,
        "task_lines": ["task mcscf"],
        "input_text": final_text,
        "file_plan": file_plan,
        "written_file": written_file,
    }


def draft_nwchem_mcscf_retry_input(
    output_path: str,
    input_path: str,
    expected_metal_elements: list[str] | None = None,
    active_space_mode: str = "auto",
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    state_label: str | None = None,
    symmetry: int | None = None,
    hessian: str | None = None,
    maxiter: int | None = None,
    thresh: float | None = None,
    level: float | None = None,
    lock_vectors: bool = True,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    # Lazy import — review_nwchem_mcscf_case lives in api_strategy (still flat).
    from chemtools.api_strategy import review_nwchem_mcscf_case

    if active_space_mode not in {"auto", "minimal", "expanded"}:
        raise ValueError("active_space_mode must be 'auto', 'minimal', or 'expanded'")

    review = review_nwchem_mcscf_case(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
    )
    parsed = review["raw_mcscf"]
    settings = parsed.get("settings") or {}
    convergence_assessment = (review.get("convergence_review") or {}).get("assessment")
    occupation_assessment = (review.get("occupation_review") or {}).get("assessment")

    if active_space_mode == "auto":
        resolved_active_space_mode = "expanded" if occupation_assessment == "overly_pinned_active_space" else "minimal"
    else:
        resolved_active_space_mode = active_space_mode

    strategy_notes: list[str] = []
    if parsed["failure_mode"] == "input_parse_error":
        retry_strategy = "syntax_cleanup_retry"
        resolved_hessian = hessian or "exact"
        resolved_maxiter = maxiter or max(_coerce_api_int(settings.get("maxiter")) or 0, 80)
        resolved_thresh = thresh if thresh is not None else (_coerce_api_float(settings.get("thresh")) or 1.0e-5)
        resolved_level = level if level is not None else max(_coerce_api_float(settings.get("initial_levelshift")) or 0.0, 0.6)
        strategy_notes.append("removed problematic state syntax and rebuilt the mcscf block from parsed defaults")
    elif convergence_assessment in {"input_or_convergence_failure", "incomplete_mcscf_convergence"}:
        retry_strategy = "stronger_convergence_retry"
        resolved_hessian = hessian or "exact"
        resolved_maxiter = maxiter or max(_coerce_api_int(settings.get("maxiter")) or 0, 120)
        resolved_thresh = thresh if thresh is not None else (_coerce_api_float(settings.get("thresh")) or 1.0e-5)
        resolved_level = level if level is not None else max(_coerce_api_float(settings.get("initial_levelshift")) or 0.0, 0.6)
        strategy_notes.append("increased macroiteration budget for a previously incomplete or fragile mcscf run")
    elif convergence_assessment == "converged_with_stiff_orbital_optimization":
        retry_strategy = "stiff_but_converged_refinement"
        resolved_hessian = hessian or "exact"
        resolved_maxiter = maxiter or max(_coerce_api_int(settings.get("maxiter")) or 0, 120)
        resolved_thresh = thresh if thresh is not None else (_coerce_api_float(settings.get("thresh")) or 1.0e-5)
        resolved_level = level if level is not None else max(_coerce_api_float(settings.get("initial_levelshift")) or 0.0, 0.6)
        strategy_notes.append("kept exact hessian and a higher macroiteration budget because the previous mcscf converged stiffly")
    else:
        retry_strategy = "active_space_refinement_retry"
        resolved_hessian = hessian or (settings.get("hessian") or "exact")
        resolved_maxiter = maxiter or max(_coerce_api_int(settings.get("maxiter")) or 0, 80)
        resolved_thresh = thresh if thresh is not None else (_coerce_api_float(settings.get("thresh")) or 1.0e-5)
        resolved_level = level if level is not None else (_coerce_api_float(settings.get("initial_levelshift")) or 0.6)
        strategy_notes.append("reused the stable mcscf settings and focused the retry on active-space refinement")

    if resolved_active_space_mode == "expanded":
        strategy_notes.append("using expanded active space because the current active window looks too pinned or needs more flexibility")

    resolved_base_name = base_name or f"{Path(input_path).stem}_mcscf_retry"
    resolved_vectors_input = vectors_input or settings.get("vectors_output") or settings.get("vectors_input")
    resolved_vectors_output = vectors_output or f"{resolved_base_name}.movecs"
    resolved_title = title or f"{resolved_base_name}: {retry_strategy.replace('_', ' ')}"

    drafted = draft_nwchem_mcscf_input(
        input_path=input_path,
        reference_output_path=output_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=(review.get("input_summary") or {}).get("multiplicity", 1) - 1 if (review.get("input_summary") or {}).get("multiplicity") else None,
        active_space_mode=resolved_active_space_mode,
        vectors_input=resolved_vectors_input,
        vectors_output=resolved_vectors_output,
        state_label=state_label,
        symmetry=symmetry,
        hessian=resolved_hessian,
        maxiter=resolved_maxiter,
        thresh=resolved_thresh,
        level=resolved_level,
        lock_vectors=lock_vectors,
        output_dir=output_dir,
        base_name=resolved_base_name,
        title=resolved_title,
        write_file=write_file,
    )

    return {
        "output_file": output_path,
        "input_file": input_path,
        "retry_strategy": retry_strategy,
        "strategy_notes": strategy_notes,
        "mcscf_review": {
            "status": review["status"],
            "failure_mode": review["failure_mode"],
            "recommended_next_action": review["recommended_next_action"],
            "convergence_assessment": convergence_assessment,
            "occupation_assessment": occupation_assessment,
        },
        "resolved_settings": {
            "active_space_mode": resolved_active_space_mode,
            "vectors_input": resolved_vectors_input,
            "vectors_output": resolved_vectors_output,
            "hessian": resolved_hessian,
            "maxiter": resolved_maxiter,
            "thresh": resolved_thresh,
            "level": resolved_level,
            "lock_vectors": lock_vectors,
        },
        "drafted_input": drafted,
        "input_text": drafted["input_text"],
        "file_plan": drafted["file_plan"],
        "written_file": drafted["written_file"],
    }


__all__ = ["draft_nwchem_mcscf_input", "draft_nwchem_mcscf_retry_input"]

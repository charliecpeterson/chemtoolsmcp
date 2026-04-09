from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import nwchem
from .nwchem_input import render_nwchem_module_block

# Alias so private helpers in this module and in api_runner.py can use _COVALENT_RADII
_COVALENT_RADII = nwchem.COVALENT_RADII


def _coerce_api_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    parsed = nwchem._coerce_int(value)
    return parsed


def _coerce_api_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    return nwchem._coerce_float(value)


def _strategy_entry(
    *,
    name: str,
    priority: int,
    rationale: str,
    tool: str,
    docs_topics: list[str],
    when_to_use: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "priority": priority,
        "rationale": rationale,
        "tool": tool,
        "docs_topics": docs_topics,
        "when_to_use": when_to_use,
    }


def _summarize_prepared_artifact(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"artifact": name}

    if input_file := payload.get("written_file") or (payload.get("file_plan") or {}).get("input_file"):
        summary["input_file"] = input_file

    file_plan = payload.get("file_plan") or {}
    if plus_file := file_plan.get("plus_file"):
        summary["plus_file"] = plus_file
    if minus_file := file_plan.get("minus_file"):
        summary["minus_file"] = minus_file

    if follow_up_plan := payload.get("follow_up_plan"):
        if strategy := follow_up_plan.get("strategy"):
            summary["strategy"] = strategy
        if task_lines := follow_up_plan.get("task_lines"):
            summary["task_lines"] = task_lines

    if vectors_input := payload.get("vectors_input"):
        summary["vectors_input"] = vectors_input
    if vectors_output := payload.get("vectors_output"):
        summary["vectors_output"] = vectors_output
    if plus_vectors_output := payload.get("plus_vectors_output"):
        summary["plus_vectors_output"] = plus_vectors_output
    if minus_vectors_output := payload.get("minus_vectors_output"):
        summary["minus_vectors_output"] = minus_vectors_output

    if cube_outputs := file_plan.get("cube_outputs"):
        summary["cube_outputs"] = cube_outputs

    if swap_pairs := (((payload.get("suggestion") or {}).get("swap_pairs")) or None):
        summary["swap_pair_count"] = len(swap_pairs)

    if property_keywords := payload.get("property_keywords"):
        summary["property_keywords"] = property_keywords
    if selected_task_operation := payload.get("selected_task_operation"):
        summary["selected_task_operation"] = selected_task_operation
    if task_strategy := payload.get("task_strategy"):
        summary["task_strategy"] = task_strategy
    if stabilization_strategy := payload.get("stabilization_strategy"):
        summary["stabilization_strategy"] = stabilization_strategy

    return summary


KEYWORD_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\b", re.IGNORECASE)
CONVERGENCE_DAMP_RE = re.compile(r"^\s*convergence\s+damp\b", re.IGNORECASE)
CONVERGENCE_NCYDP_RE = re.compile(r"^\s*convergence\s+ncydp\b", re.IGNORECASE)
ITERATIONS_RE = re.compile(r"^\s*iterations\b", re.IGNORECASE)
SMEAR_RE = re.compile(r"^\s*smear\b", re.IGNORECASE)
PRINT_RE = re.compile(r"^\s*print\b", re.IGNORECASE)
CONVERGENCE_ENERGY_RE = re.compile(r"^\s*convergence\s+energy\b", re.IGNORECASE)
VECTORS_RE = re.compile(r"^\s*vectors\b", re.IGNORECASE)
VECTORS_INPUT_TOKEN_RE = re.compile(r"\binput\s+([^\s\\]+)", re.IGNORECASE)
VECTORS_OUTPUT_TOKEN_RE = re.compile(r"\boutput\s+([^\s\\]+)", re.IGNORECASE)


def _select_primary_task_module(input_summary: dict[str, Any]) -> str:
    task_modules = [task["module"] for task in input_summary.get("tasks", []) if task.get("module")]
    return task_modules[0] if task_modules else "dft"


def _select_scf_stabilization_strategy(
    *,
    reference_diagnosis: dict[str, Any] | None,
    iterations: int | None,
    smear: float | None,
    convergence_damp: int | None,
    convergence_ncydp: int | None,
    population_print: str | None,
) -> dict[str, Any]:
    strategy = "generic_restart"
    notes: list[str] = []

    selected_iterations = iterations if iterations is not None else 200
    selected_smear = smear if smear is not None else 0.001
    selected_convergence_damp = convergence_damp if convergence_damp is not None else 30
    selected_convergence_ncydp = convergence_ncydp if convergence_ncydp is not None else 30
    selected_population_print = population_print if population_print is not None else "mulliken"

    if reference_diagnosis:
        scf = reference_diagnosis.get("scf") or {}
        last_run = scf.get("last_run") or {}
        trend = (last_run.get("trend") or {}).get("pattern") or (scf.get("trend") or {}).get("pattern")
        iteration_count = last_run.get("iteration_count") or scf.get("iteration_count") or 0
        hit_max = bool(last_run.get("hit_max_iterations") or scf.get("hit_max_iterations"))
        failure_class = reference_diagnosis.get("failure_class")

        if failure_class == "scf_nonconvergence" and iteration_count <= 2 and trend in {"insufficient_data", "no_iterations_found"}:
            strategy = "state_check_recovery"
            notes.append("reference_run_failed_before_meaningful_scf_progress")
            if iterations is None:
                selected_iterations = 120
            if smear is None:
                selected_smear = 0.001
            if convergence_damp is None:
                selected_convergence_damp = 40
            if convergence_ncydp is None:
                selected_convergence_ncydp = 80
        elif trend == "oscillatory":
            strategy = "oscillation_control"
            notes.append("reference_scf_showed_energy_oscillation")
            if iterations is None:
                selected_iterations = 250
            if smear is None:
                selected_smear = 0.005
            if convergence_damp is None:
                selected_convergence_damp = 80
            if convergence_ncydp is None:
                selected_convergence_ncydp = 120
        elif trend in {"slow_improving", "nearly_converged"} and hit_max:
            strategy = "gentle_iteration_extension"
            notes.append("reference_scf_was_still_improving_near_max_iterations")
            if iterations is None:
                selected_iterations = 350
            if smear is None:
                selected_smear = None
            if convergence_damp is None:
                selected_convergence_damp = None
            if convergence_ncydp is None:
                selected_convergence_ncydp = None
        elif trend == "stalled":
            strategy = "stalled_density_restart"
            notes.append("reference_scf_stalled_without_clear_oscillation")
            if iterations is None:
                selected_iterations = 220
            if smear is None:
                selected_smear = 0.002
            if convergence_damp is None:
                selected_convergence_damp = 60
            if convergence_ncydp is None:
                selected_convergence_ncydp = 80
        elif failure_class == "wrong_state_convergence":
            strategy = "state_sensitive_restart"
            notes.append("reference_run_converged_to_suspicious_state")
            if iterations is None:
                selected_iterations = 220
            if smear is None:
                selected_smear = 0.001
            if convergence_damp is None:
                selected_convergence_damp = 40
            if convergence_ncydp is None:
                selected_convergence_ncydp = 60
        else:
            notes.append("using_generic_scf_restart_defaults")
    else:
        notes.append("no_reference_output_supplied_using_generic_restart_defaults")

    return {
        "strategy": strategy,
        "notes": notes,
        "iterations": selected_iterations,
        "smear": selected_smear,
        "convergence_damp": selected_convergence_damp,
        "convergence_ncydp": selected_convergence_ncydp,
        "population_print": selected_population_print,
    }


def _select_optimization_follow_up_strategy(
    *,
    task_strategy: str,
    trajectory: dict[str, Any],
    diagnosis: dict[str, Any],
) -> str:
    if task_strategy != "auto":
        return task_strategy
    if trajectory.get("restart_recommended"):
        return "optimize_only"
    if diagnosis.get("stage") == "frequency" and diagnosis.get("task_outcome") != "success":
        return "freq_only"
    if trajectory.get("optimization_status") == "converged":
        return "freq_only"
    return "optimize_only"


def _build_optimization_follow_up_plan(
    *,
    input_summary: dict[str, Any],
    trajectory: dict[str, Any],
    diagnosis: dict[str, Any],
    task_strategy: str,
) -> dict[str, Any]:
    module_name = _select_primary_task_module(input_summary)
    if task_strategy == "optimize_only":
        task_lines = [f"task {module_name} optimize"]
        rationale = "continue optimization from the last available geometry"
    elif task_strategy == "freq_only":
        task_lines = [f"task {module_name} freq"]
        rationale = "optimization geometry appears converged, so continue with frequency only"
    else:
        task_lines = [f"task {module_name} optimize", f"task {module_name} freq"]
        rationale = "continue optimization from the last geometry and follow with frequency"

    return {
        "strategy": task_strategy,
        "module": module_name,
        "task_lines": task_lines,
        "rationale": rationale,
        "optimization_status": trajectory.get("optimization_status"),
        "diagnosis_stage": diagnosis.get("stage"),
        "diagnosis_outcome": diagnosis.get("task_outcome"),
    }


def _rewrite_module_body_for_vectors_swap(
    body_lines: list[str],
    vectors_block: str,
    *,
    iterations: int | None,
    smear: float | None,
    convergence_damp: int | None,
    convergence_ncydp: int | None,
    population_print: str | None,
) -> list[str]:
    cleaned_lines: list[str] = []
    skip_vectors = False

    for line in body_lines:
        stripped = line.strip()
        lower = stripped.lower()

        if skip_vectors:
            if not line.rstrip().endswith("\\"):
                skip_vectors = False
            continue

        if lower.startswith("#") and ("swap" in lower or "swapped movecs" in lower):
            continue
        if lower.startswith("vectors "):
            skip_vectors = line.rstrip().endswith("\\")
            continue
        if lower.startswith("#") and ("swap" in lower or "swapped movecs" in lower):
            continue
        if ITERATIONS_RE.match(line):
            continue
        if SMEAR_RE.match(line):
            continue
        if CONVERGENCE_DAMP_RE.match(line):
            continue
        if CONVERGENCE_NCYDP_RE.match(line):
            continue
        if population_print and PRINT_RE.match(line):
            continue
        cleaned_lines.append(line.rstrip())

    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    additions: list[str] = []
    if iterations is not None:
        additions.append(f"  iterations {iterations}")
    if smear is not None:
        additions.append(f"  smear {smear:g}")
    if convergence_damp is not None:
        additions.append(f"  convergence damp {convergence_damp}")
    if convergence_ncydp is not None:
        additions.append(f"  convergence ncydp {convergence_ncydp}")
    additions.append("  # Restart from vectors with explicit orbital swaps")
    additions.extend(_indent_vectors_block_lines(vectors_block))
    if population_print:
        additions.append(f'  print "{population_print}"')

    if cleaned_lines:
        cleaned_lines.append("")
    cleaned_lines.extend(additions)
    return cleaned_lines


def _rewrite_module_body_for_property_check(
    body_lines: list[str],
    *,
    vectors_input: str,
    vectors_output: str,
    iterations: int | None,
    convergence_energy: str | None,
    smear: float | None,
    include_mulliken_in_module: bool = False,
) -> list[str]:
    cleaned_lines: list[str] = []
    skip_vectors = False

    for line in body_lines:
        stripped = line.strip()
        lower = stripped.lower()

        if skip_vectors:
            if not line.rstrip().endswith("\\"):
                skip_vectors = False
            continue

        if lower.startswith("vectors "):
            skip_vectors = line.rstrip().endswith("\\")
            continue
        if ITERATIONS_RE.match(line):
            continue
        if SMEAR_RE.match(line):
            continue
        if CONVERGENCE_DAMP_RE.match(line):
            continue
        if CONVERGENCE_NCYDP_RE.match(line):
            continue
        if CONVERGENCE_ENERGY_RE.match(line):
            continue
        if PRINT_RE.match(line):
            continue
        if lower.startswith("mulliken"):
            continue
        cleaned_lines.append(line.rstrip())

    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    additions: list[str] = []
    if smear is not None:
        additions.append(f"  smear {smear:g}")
    if iterations is not None:
        additions.append(f"  iterations {iterations}")
    if convergence_energy is not None:
        additions.append(f"  convergence energy {convergence_energy}")
    if include_mulliken_in_module:
        additions.append("  mulliken")
    additions.append(f"  vectors input {vectors_input} output {vectors_output}")

    if cleaned_lines:
        cleaned_lines.append("")
    cleaned_lines.extend(additions)
    return cleaned_lines


def _rewrite_module_body_for_scf_stabilization(
    body_lines: list[str],
    *,
    vectors_input: str,
    vectors_output: str,
    iterations: int | None,
    smear: float | None,
    convergence_damp: int | None,
    convergence_ncydp: int | None,
    population_print: str | None,
) -> list[str]:
    cleaned_lines: list[str] = []
    skip_vectors = False

    for line in body_lines:
        stripped = line.strip()
        lower = stripped.lower()

        if skip_vectors:
            if not line.rstrip().endswith("\\"):
                skip_vectors = False
            continue

        if lower.startswith("vectors "):
            skip_vectors = line.rstrip().endswith("\\")
            continue
        if ITERATIONS_RE.match(line):
            continue
        if SMEAR_RE.match(line):
            continue
        if CONVERGENCE_DAMP_RE.match(line):
            continue
        if CONVERGENCE_NCYDP_RE.match(line):
            continue
        if population_print and PRINT_RE.match(line):
            continue
        if lower.startswith("mulliken"):
            continue
        cleaned_lines.append(line.rstrip())

    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    additions: list[str] = []
    if iterations is not None:
        additions.append(f"  iterations {iterations}")
    if smear is not None:
        additions.append(f"  smear {smear:g}")
    if convergence_damp is not None:
        additions.append(f"  convergence damp {convergence_damp}")
    if convergence_ncydp is not None:
        additions.append(f"  convergence ncydp {convergence_ncydp}")
    if population_print:
        additions.append(f'  print "{population_print}"')
    additions.append(f"  vectors input {vectors_input} output {vectors_output}")

    if cleaned_lines:
        cleaned_lines.append("")
    cleaned_lines.extend(additions)
    return cleaned_lines


def _extract_vectors_io_from_lines(vectors_lines: list[str]) -> tuple[str | None, str | None]:
    if not vectors_lines:
        return None, None
    flattened = " ".join(raw.replace("\\", " ").strip() for raw in vectors_lines)
    input_name = None
    output_name = None
    if match := VECTORS_INPUT_TOKEN_RE.search(flattened):
        input_name = match.group(1)
    if match := VECTORS_OUTPUT_TOKEN_RE.search(flattened):
        output_name = match.group(1)
    return input_name, output_name


def _rewrite_module_body_for_vectors_output(
    body_lines: list[str],
    *,
    vectors_output: str,
    vectors_input: str | None = None,
) -> list[str]:
    cleaned_lines: list[str] = []
    existing_vectors_lines: list[str] = []
    current_vectors_lines: list[str] = []
    skip_vectors = False

    for line in body_lines:
        lower = line.strip().lower()

        if skip_vectors:
            current_vectors_lines.append(line.rstrip())
            if not line.rstrip().endswith("\\"):
                existing_vectors_lines = current_vectors_lines[:]
                current_vectors_lines = []
                skip_vectors = False
            continue

        if VECTORS_RE.match(line):
            current_vectors_lines = [line.rstrip()]
            if line.rstrip().endswith("\\"):
                skip_vectors = True
            else:
                existing_vectors_lines = current_vectors_lines[:]
                current_vectors_lines = []
            continue

        cleaned_lines.append(line.rstrip())

    if current_vectors_lines:
        existing_vectors_lines = current_vectors_lines[:]

    resolved_vectors_input = vectors_input
    if resolved_vectors_input is None and existing_vectors_lines:
        flattened = " ".join(raw.replace("\\", " ").strip() for raw in existing_vectors_lines)
        if match := VECTORS_INPUT_TOKEN_RE.search(flattened):
            resolved_vectors_input = match.group(1)

    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    if cleaned_lines:
        cleaned_lines.append("")
    if resolved_vectors_input:
        cleaned_lines.append(f"  vectors input {resolved_vectors_input} output {vectors_output}")
    else:
        cleaned_lines.append(f"  vectors output {vectors_output}")
    return cleaned_lines


def _indent_vectors_block_lines(vectors_block: str) -> list[str]:
    output: list[str] = []
    for index, raw_line in enumerate(vectors_block.splitlines()):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if index == 0:
            output.append(f"  {stripped}")
        else:
            output.append(f"          {stripped}")
    return output


def _replace_module_block_in_text(
    text: str,
    *,
    module: str,
    module_block_text: str,
    block_index: int = -1,
) -> str:
    lines = text.splitlines()
    blocks: list[tuple[int, int]] = []
    target = module.lower()
    idx = 0

    while idx < len(lines):
        stripped = lines[idx].strip()
        match = KEYWORD_LINE_RE.match(stripped)
        if match and match.group(1).lower() == target:
            start_line = idx
            idx += 1
            while idx < len(lines):
                if lines[idx].strip().lower() == "end":
                    blocks.append((start_line, idx))
                    break
                idx += 1
        idx += 1

    if not blocks:
        raise ValueError(f"could not find {module} block in text")

    start_line, end_line = blocks[block_index]
    new_lines = lines[:start_line] + module_block_text.splitlines() + lines[end_line + 1 :]
    return "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")


def _ensure_module_vectors_output_in_text(
    text: str,
    *,
    module: str,
    vectors_output: str | None,
    vectors_input: str | None = None,
    block_index: int = -1,
) -> tuple[str, str | None]:
    if module.lower() not in {"scf", "dft"} or not vectors_output:
        return text, None

    lines = text.splitlines()
    blocks: list[tuple[int, int, list[str], str]] = []
    target = module.lower()
    idx = 0

    while idx < len(lines):
        stripped = lines[idx].strip()
        match = KEYWORD_LINE_RE.match(stripped)
        if match and match.group(1).lower() == target:
            start_line = idx
            header_line = lines[idx]
            body_lines: list[str] = []
            idx += 1
            while idx < len(lines):
                if lines[idx].strip().lower() == "end":
                    blocks.append((start_line, idx, body_lines, header_line))
                    break
                body_lines.append(lines[idx])
                idx += 1
        idx += 1

    if not blocks:
        return text, None

    _, _, body_lines, header_line = blocks[block_index]
    rewritten_body = _rewrite_module_body_for_vectors_output(
        body_lines,
        vectors_output=vectors_output,
        vectors_input=vectors_input,
    )
    module_block_text = render_nwchem_module_block(header_line, rewritten_body)
    updated_text = _replace_module_block_in_text(
        text,
        module=module,
        module_block_text=module_block_text,
        block_index=block_index,
    )
    return updated_text, vectors_output


def _default_optimization_follow_up_base_name(input_path: str, strategy: str) -> str:
    stem = Path(input_path).stem
    if strategy == "optimize_only":
        suffix = "opt_restart"
    elif strategy == "freq_only":
        suffix = "freq_followup"
    else:
        suffix = "optfreq_followup"
    return f"{stem}_{suffix}"


def _default_optimization_follow_up_title(strategy: str) -> str:
    if strategy == "optimize_only":
        return "Continue optimization from the last available geometry"
    if strategy == "freq_only":
        return "Frequency follow-up from the converged optimization geometry"
    return "Optimization restart from last geometry followed by frequency"


def _build_simple_input_file_plan(
    *,
    input_path: str,
    output_dir: str | None,
    base_name: str,
) -> dict[str, str]:
    target_dir = Path(output_dir) if output_dir else Path(input_path).resolve().parent
    return {
        "output_dir": str(target_dir),
        "input_file": str(target_dir / f"{base_name}.nw"),
    }


def _apply_default_dft_settings(
    rendered_settings: list[str],
    *,
    xc_functional: str | None,
    multiplicity: int | None,
    vectors_input: str | None,
    vectors_output: str | None,
) -> list[str]:
    settings = list(rendered_settings)
    stripped_lower = [line.strip().lower() for line in settings]

    insertion_index = 0
    if xc_functional and not any(line.startswith("xc ") for line in stripped_lower):
        settings.insert(insertion_index, f"  xc {xc_functional}")
        insertion_index += 1
        stripped_lower = [line.strip().lower() for line in settings]
    else:
        for idx, line in enumerate(stripped_lower):
            if line.startswith("xc "):
                insertion_index = idx + 1
                break

    if multiplicity not in (None, 1) and not any(line.startswith("mult ") for line in stripped_lower):
        settings.insert(insertion_index, f"  mult {multiplicity}")
        insertion_index += 1
        stripped_lower = [line.strip().lower() for line in settings]
    else:
        for idx, line in enumerate(stripped_lower):
            if line.startswith("mult "):
                insertion_index = idx + 1
                break

    default_lines = [
        ("iterations ", "  iterations 300"),
        ("mulliken", "  mulliken"),
        ("direct", "  direct"),
        ("noio", "  noio"),
        ("grid nodisk", "  grid nodisk"),
    ]
    for keyword, default_line in default_lines:
        if not any(line == keyword or line.startswith(f"{keyword} ") for line in stripped_lower):
            settings.insert(insertion_index, default_line)
            insertion_index += 1
            stripped_lower = [line.strip().lower() for line in settings]

    if vectors_output and not any(line.startswith("vectors ") for line in stripped_lower):
        if vectors_input:
            settings.append(f"  vectors input {vectors_input} output {vectors_output}")
        else:
            settings.append(f"  vectors output {vectors_output}")
    return settings


def _ensure_driver_block(blocks: list[str]) -> None:
    for block in blocks:
        if block.lstrip().lower().startswith("driver"):
            return
    blocks.append("driver\n  maxiter 300\nend")


_TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}


def _parse_formula_elements(formula: str) -> list[str]:
    seen: list[str] = []
    for symbol in re.findall(r"([A-Z][a-z]?)", formula):
        normalized = symbol[0].upper() + symbol[1:].lower()
        if normalized not in seen:
            seen.append(normalized)
    return seen


def _normalize_nwchem_task_operation(task: str) -> str:
    lowered = task.strip().lower()
    mapping = {
        "opt": "optimize",
        "optimize": "optimize",
        "freq": "freq",
        "frequency": "freq",
        "sp": "energy",
        "energy": "energy",
        "property": "property",
    }
    if lowered not in mapping:
        raise ValueError(f"unsupported NWChem task operation: {task}")
    return mapping[lowered]


def _replace_or_insert_keyword_line(
    text: str,
    keyword: str,
    new_line: str,
    *,
    insert_after: str | None = None,
) -> str:
    lines = text.splitlines()
    keyword_lower = keyword.lower()
    for index, line in enumerate(lines):
        match = KEYWORD_LINE_RE.match(line.strip())
        if match and match.group(1).lower() == keyword_lower:
            lines[index] = new_line
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    insertion_index = 0
    if insert_after is not None:
        insert_after_lower = insert_after.lower()
        for index, line in enumerate(lines):
            match = KEYWORD_LINE_RE.match(line.strip())
            if match and match.group(1).lower() == insert_after_lower:
                insertion_index = index + 1
                break

    lines.insert(insertion_index, new_line)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _remove_keyword_blocks(text: str, keywords: set[str]) -> str:
    lines = text.splitlines()
    output: list[str] = []
    skip_block = False
    target_keywords = {keyword.lower() for keyword in keywords}

    for line in lines:
        stripped = line.strip()
        match = KEYWORD_LINE_RE.match(stripped)
        if skip_block:
            if stripped.lower() == "end":
                skip_block = False
            continue
        if match and match.group(1).lower() in target_keywords:
            skip_block = True
            continue
        output.append(line)

    return "\n".join(output) + ("\n" if text.endswith("\n") else "")


def _render_named_block(keyword: str, body_lines: list[str]) -> str:
    lines = [keyword]
    lines.extend(line.rstrip() for line in body_lines if line.strip())
    lines.append("end")
    return "\n".join(lines)


def _replace_or_insert_named_block(
    text: str,
    keyword: str,
    block_text: str,
    *,
    insert_before_task: bool = False,
) -> str:
    without_block = _remove_keyword_blocks(text, {keyword})
    lines = without_block.splitlines()

    insert_index = len(lines)
    if insert_before_task:
        for index, line in enumerate(lines):
            if line.lstrip().lower().startswith("task "):
                insert_index = index
                break

    block_lines = block_text.splitlines()
    new_lines = lines[:insert_index]
    if new_lines and new_lines[-1].strip():
        new_lines.append("")
    new_lines.extend(block_lines)
    if insert_index < len(lines):
        if block_lines and lines[insert_index].strip():
            new_lines.append("")
        new_lines.extend(lines[insert_index:])

    return "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")


def _append_named_blocks_before_tasks(text: str, block_texts: list[str]) -> str:
    lines = text.splitlines()
    insert_index = len(lines)
    for index, line in enumerate(lines):
        if line.lstrip().lower().startswith("task "):
            insert_index = index
            break

    blocks: list[str] = []
    for block_text in block_texts:
        if blocks:
            blocks.append("")
        blocks.extend(block_text.splitlines())

    new_lines = lines[:insert_index]
    if new_lines and new_lines[-1].strip():
        new_lines.append("")
    new_lines.extend(blocks)
    if insert_index < len(lines):
        if blocks and lines[insert_index].strip():
            new_lines.append("")
        new_lines.extend(lines[insert_index:])
    return "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")


def _render_limitxyz_lines(extent_angstrom: float, grid_points: int) -> list[str]:
    return [
        "  limitxyz",
        f"   {-extent_angstrom:.1f}  {extent_angstrom:.1f}  {grid_points}",
        f"   {-extent_angstrom:.1f}  {extent_angstrom:.1f}  {grid_points}",
        f"   {-extent_angstrom:.1f}  {extent_angstrom:.1f}  {grid_points}",
    ]


def _render_dplot_density_block(
    *,
    vectors_input: str,
    output_name: str,
    density_mode: str,
    extent_angstrom: float,
    grid_points: int,
    gaussian: bool,
) -> str:
    title = "Total density" if density_mode == "total" else "Spin density"
    body_lines = [
        f'  title "{title}"',
        f"  vectors {vectors_input}",
    ]
    if gaussian:
        body_lines.append("  gaussian")
    body_lines.append(f"  spin {density_mode}")
    body_lines.extend(_render_limitxyz_lines(extent_angstrom, grid_points))
    body_lines.append(f"  output {output_name}")
    return _render_named_block("dplot", body_lines)


def _render_dplot_orbital_block(
    *,
    vectors_input: str,
    output_name: str,
    vector_number: int,
    spin: str,
    title: str,
    extent_angstrom: float,
    grid_points: int,
    gaussian: bool,
) -> str:
    body_lines = [
        f'  title "{title}"',
        f"  vectors {vectors_input}",
    ]
    if gaussian:
        body_lines.append("  gaussian")
    body_lines.extend(_render_limitxyz_lines(extent_angstrom, grid_points))
    body_lines.append(f"  spin {spin}")
    body_lines.append(f"  orbitals view; 1; {vector_number}; output {output_name}")
    return _render_named_block("dplot", body_lines)


def _build_vectors_swap_file_plan(
    input_path: str,
    output_dir: str | None,
    base_name: str,
    vectors_output: str,
) -> dict[str, str]:
    input_path_obj = Path(input_path)
    target_dir = Path(output_dir) if output_dir else input_path_obj.parent
    return {
        "output_dir": str(target_dir),
        "input_file": str(target_dir / f"{base_name}.nw"),
        "movecs_file": str(target_dir / vectors_output),
    }


def _build_mcscf_reorder_plan(active_space: dict[str, Any]) -> dict[str, Any]:
    vector_numbers = sorted(active_space.get("vector_numbers") or [])
    active_orbitals = active_space.get("active_orbitals") or len(vector_numbers)
    closed_shell_count = active_space.get("closed_shell_count") or 0
    target_positions = list(range(closed_shell_count + 1, closed_shell_count + active_orbitals + 1))
    arrangement = {index: index for index in range(1, max(max(vector_numbers, default=0), max(target_positions, default=0)) + 1)}
    inverse = arrangement.copy()
    swap_pairs: list[dict[str, int]] = []

    for target_position, desired_vector in zip(target_positions, vector_numbers):
        current_vector = arrangement[target_position]
        if current_vector == desired_vector:
            continue
        desired_position = inverse[desired_vector]
        swap_pairs.append({"from_vector": desired_vector, "to_vector": current_vector})
        arrangement[target_position], arrangement[desired_position] = arrangement[desired_position], arrangement[target_position]
        inverse[current_vector], inverse[desired_vector] = desired_position, target_position

    return {
        "target_positions": target_positions,
        "desired_active_vectors": vector_numbers,
        "swap_pairs": swap_pairs,
    }


def _render_mcscf_block(
    *,
    active_space: dict[str, Any],
    multiplicity: int | None,
    vectors_input: str,
    vectors_output: str,
    state_label: str | None,
    symmetry: int | None,
    hessian: str,
    maxiter: int,
    thresh: float | None,
    level: float | None,
    lock_vectors: bool,
    swap_pairs: list[dict[str, int]],
) -> str:
    body_lines = []
    if state_label:
        body_lines.append(f"  state {state_label}")
    body_lines.append(f"  active {active_space['active_orbitals']}")
    body_lines.append(f"  actelec {active_space['active_electrons']}")
    if not state_label and multiplicity is not None:
        body_lines.append(f"  multiplicity {multiplicity}")
    if symmetry is not None:
        body_lines.append(f"  symmetry {symmetry}")

    vectors_line = f"  vectors input {vectors_input}"
    for pair in swap_pairs:
        vectors_line += f" swap {pair['from_vector']} {pair['to_vector']}"
    vectors_line += f" output {vectors_output}"
    if lock_vectors:
        vectors_line += " lock"
    body_lines.append(vectors_line)
    body_lines.append(f"  hessian {hessian}")
    body_lines.append(f"  maxiter {maxiter}")
    if thresh is not None:
        body_lines.append(f"  thresh {thresh:.1e}")
    if level is not None:
        body_lines.append(f"  level {level}")
    return _render_named_block("mcscf", body_lines)


def _build_cube_file_plan(
    input_path: str,
    output_dir: str | None,
    base_name: str,
    cube_outputs: list[str],
) -> dict[str, Any]:
    input_path_obj = Path(input_path)
    target_dir = Path(output_dir) if output_dir else input_path_obj.parent
    return {
        "output_dir": str(target_dir),
        "input_file": str(target_dir / f"{base_name}.nw"),
        "cube_files": [str(target_dir / name) for name in cube_outputs],
    }


def _write_text_file(text: str, path: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return str(target.resolve())


def _build_imaginary_follow_up_plan(
    input_summary: dict[str, Any],
    stability_assessment: dict[str, Any] | None,
    selected_mode: dict[str, Any],
    task_strategy: str,
) -> dict[str, Any]:
    module = "dft"
    task_modules = [task["module"] for task in input_summary.get("tasks", []) if task.get("module")]
    if task_modules:
        module = task_modules[0]

    if task_strategy == "auto":
        strategy = _auto_task_strategy(selected_mode, stability_assessment)
    else:
        strategy = task_strategy

    task_lines = [f"task {module} optimize"]
    if strategy == "optimize_then_freq":
        task_lines.append(f"task {module} freq")

    return {
        "strategy": strategy,
        "module": module,
        "task_lines": task_lines,
        "recommended_action": selected_mode.get("recommended_action"),
        "stability_classification": (stability_assessment or {}).get("classification"),
    }


def _auto_task_strategy(
    selected_mode: dict[str, Any],
    stability_assessment: dict[str, Any] | None,
) -> str:
    motion_type = selected_mode.get("motion_type")
    stability_classification = (stability_assessment or {}).get("classification")
    if stability_classification == "likely_projection_or_symmetry_zero_modes":
        return "optimize_only"
    if motion_type == "metal_ligand_distortion":
        return "optimize_only"
    if motion_type in {"torsion", "pyramidal_inversion", "bend", "stretch"}:
        return "optimize_then_freq"
    return "optimize_only"


def _replace_tasks_in_text(path: str, text: str, task_lines: list[str]) -> dict[str, Any]:
    lines = text.splitlines()
    task_indices = [index for index, line in enumerate(lines) if line.lstrip().lower().startswith("task ")]
    replacement = [task.rstrip() for task in task_lines]

    if task_indices:
        first = task_indices[0]
        new_lines = [line for index, line in enumerate(lines) if index not in set(task_indices)]
        new_lines = new_lines[:first] + replacement + new_lines[first:]
    else:
        new_lines = lines + ([""] if lines and lines[-1].strip() else []) + replacement

    return {
        "text": "\n".join(new_lines) + ("\n" if text.endswith("\n") else ""),
        "task_lines": replacement,
    }


def _build_imaginary_output_file_plan(
    input_path: str,
    selected_mode: dict[str, Any],
    output_dir: str | None,
    base_name: str | None,
) -> dict[str, str]:
    input_path_obj = Path(input_path)
    target_dir = Path(output_dir) if output_dir else input_path_obj.parent
    stem = base_name or input_path_obj.stem
    mode_tag = f"mode{selected_mode.get('mode_number') or 'auto'}"
    plus_name = f"{stem}_{mode_tag}_plus.nw"
    minus_name = f"{stem}_{mode_tag}_minus.nw"
    return {
        "output_dir": str(target_dir),
        "plus_file": str(target_dir / plus_name),
        "minus_file": str(target_dir / minus_name),
    }


def _write_imaginary_input_files(
    plus_text: str,
    minus_text: str,
    plus_path: str,
    minus_path: str,
) -> dict[str, str]:
    plus_target = Path(plus_path)
    minus_target = Path(minus_path)
    plus_target.parent.mkdir(parents=True, exist_ok=True)
    minus_target.parent.mkdir(parents=True, exist_ok=True)
    plus_target.write_text(plus_text, encoding="utf-8")
    minus_target.write_text(minus_text, encoding="utf-8")
    return {
        "plus_file": str(plus_target.resolve()),
        "minus_file": str(minus_target.resolve()),
    }

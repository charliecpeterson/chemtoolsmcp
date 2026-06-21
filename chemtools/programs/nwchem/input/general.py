"""NWChem general-purpose input drafter + variants + request review.

Three closely-related entry points that share a lot of the same plumbing
(geometry loading, basis setup, module-block rendering):

  * create_nwchem_input             The primary low-level drafter that
                                    every other module-specific drafter
                                    ultimately resolves through. Takes
                                    method/basis/geometry/charge/mult
                                    and returns a rendered .nw file.

  * review_nwchem_input_request     Pre-flight check on a calculation
                                    request — validates the
                                    geometry/basis/method combination
                                    before drafting and returns warnings
                                    about likely issues (wrong spin
                                    state, missing ECP, etc.).

  * create_nwchem_input_variant     Versioning helper — copies an
                                    existing input with structured
                                    parameter changes (charge, mult,
                                    functional, ...) preserving the
                                    rest. Used by recovery workflows.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Any

from chemtools.core.common import read_text
from chemtools.programs.nwchem.parse.input import (
    inspect_nwchem_input,
    extract_nwchem_geometry_block,
    render_nwchem_geometry_block,
    render_nwchem_module_block,
    load_geometry_source,
)
from chemtools.programs.nwchem.input.basis import (
    render_nwchem_basis_setup,
    resolve_basis_setup,
)
from chemtools.programs.nwchem.input._utils import (
    _apply_default_dft_settings,
    _build_simple_input_file_plan,
    _ensure_driver_block,
    _normalize_nwchem_task_operation,
    _parse_formula_elements,
    _TRANSITION_METALS,
    _write_text_file,
)


def create_nwchem_input(
    geometry_path: str,
    library_path: str,
    basis_assignments: dict[str, str],
    module: str,
    task_operation: str | None = None,
    ecp_assignments: dict[str, str] | None = None,
    default_basis: str | None = None,
    default_ecp: str | None = None,
    basis_block_name: str = "ao basis",
    basis_mode: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    module_settings: list[str] | None = None,
    extra_blocks: list[str] | None = None,
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
    resolved_module = module.strip().lower()
    if not resolved_module:
        raise ValueError("module is required")

    geometry = load_geometry_source(geometry_path, block_index=geometry_block_index)
    geometry_block = render_nwchem_geometry_block(
        geometry["header_line"],
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

    resolved_start_name = start_name or Path(geometry_path).stem or "nwchem_job"
    resolved_title = title or f"{resolved_start_name}: {resolved_module} {task_operation or 'run'}"
    resolved_vectors_output = vectors_output or (
        f"{resolved_start_name}.movecs" if resolved_module in {"scf", "dft"} else None
    )

    rendered_module_settings = [
        line.rstrip() if line[:1].isspace() else f"  {line.rstrip()}"
        for line in (module_settings or [])
        if str(line).strip()
    ]
    stripped_lower = [line.strip().lower() for line in rendered_module_settings]

    if resolved_module == "dft":
        rendered_module_settings = _apply_default_dft_settings(
            rendered_module_settings,
            xc_functional=None,
            multiplicity=multiplicity,
            vectors_input=vectors_input,
            vectors_output=resolved_vectors_output,
        )
        stripped_lower = [line.strip().lower() for line in rendered_module_settings]
    elif resolved_module in {"scf"} and resolved_vectors_output:
        if not any(line.startswith("vectors ") for line in stripped_lower):
            if vectors_input:
                rendered_module_settings.append(f"  vectors input {vectors_input} output {resolved_vectors_output}")
            else:
                rendered_module_settings.append(f"  vectors output {resolved_vectors_output}")

    module_block = render_nwchem_module_block(resolved_module, rendered_module_settings)

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
    if resolved_module == "dft" and task_operation == "optimize":
        _ensure_driver_block(rendered_extra_blocks)
    sections.extend(rendered_extra_blocks)
    sections.append(module_block)

    task_line = f"task {resolved_module}"
    if task_operation:
        task_line = f"{task_line} {task_operation}"
    sections.append(task_line)

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
        "module": resolved_module,
        "task_operation": task_operation,
        "charge": charge,
        "multiplicity": multiplicity,
        "basis_setup": basis_setup,
        "module_settings": [line.strip() for line in rendered_module_settings],
        "vectors_input": vectors_input,
        "vectors_output": resolved_vectors_output,
        "input_text": input_text,
        "file_plan": file_plan,
        "written_file": written_file,
        "inline_blocks": inline_blocks,
    }


def review_nwchem_input_request(
    *,
    formula: str | None = None,
    geometry_path: str | None = None,
    library_path: str | None = None,
    basis_assignments: dict[str, str] | None = None,
    ecp_assignments: dict[str, str] | None = None,
    default_basis: str | None = None,
    default_ecp: str | None = None,
    module: str = "dft",
    task_operations: list[str] | None = None,
    functional: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
) -> dict[str, Any]:
    normalized_module = module.strip().lower() or "dft"
    normalized_tasks = [_normalize_nwchem_task_operation(task) for task in (task_operations or ["energy"])]
    formula_elements = _parse_formula_elements(formula) if formula else []
    geometry_summary = inspect_nwchem_input(geometry_path) if geometry_path and Path(geometry_path).suffix.lower() != ".xyz" else None
    geometry = load_geometry_source(geometry_path) if geometry_path else None
    geometry_elements = list(dict.fromkeys(atom["element"] for atom in geometry["atoms"])) if geometry else []
    elements = geometry_elements or formula_elements
    transition_metals = [element for element in elements if element in _TRANSITION_METALS]

    inferred_charge = charge
    assumptions: list[str] = []
    if inferred_charge is None and formula and elements:
        inferred_charge = 0
        assumptions.append("assumed_neutral_formula_charge")

    missing_requirements: list[dict[str, str]] = []
    warnings: list[str] = []

    if not geometry_path:
        missing_requirements.append(
            {
                "field": "geometry_source",
                "reason": "An NWChem input creator needs explicit coordinates from an .xyz or existing .nw file. Do not invent geometry silently.",
            }
        )
    if not elements:
        missing_requirements.append(
            {
                "field": "composition",
                "reason": "No geometry or parsable formula was provided, so element assignments cannot be validated.",
            }
        )
    if not basis_assignments and not default_basis:
        missing_requirements.append(
            {
                "field": "basis_assignment_policy",
                "reason": "At least one explicit basis assignment or a default basis is required.",
            }
        )
    if normalized_module in {"dft", "scf"} and transition_metals and multiplicity is None:
        missing_requirements.append(
            {
                "field": "multiplicity",
                "reason": "Transition-metal/open-shell systems should not have multiplicity guessed automatically.",
            }
        )
    if formula_elements and geometry_elements and formula_elements != geometry_elements:
        warnings.append("formula_elements_do_not_match_geometry_elements")

    basis_preview = None
    if geometry_path and library_path and (basis_assignments or default_basis):
        try:
            basis_preview = resolve_basis_setup(
                geometry_path=geometry_path,
                library_path=library_path,
                basis_assignments=basis_assignments or {},
                ecp_assignments=ecp_assignments,
                default_basis=default_basis,
                default_ecp=default_ecp,
            )
            if not basis_preview["basis"]["all_elements_covered"]:
                missing_requirements.append(
                    {
                        "field": "basis_assignment_policy",
                        "reason": "Current basis assignments do not cover all elements in the geometry.",
                    }
                )
        except Exception as exc:
            warnings.append(f"basis_preview_failed: {exc}")

    ready_to_create = not missing_requirements and geometry_path is not None
    recommended_tool = None
    if ready_to_create:
        if normalized_module == "dft":
            recommended_tool = "create_nwchem_dft_workflow_input"
        else:
            recommended_tool = "create_nwchem_input"

    next_questions = [item["field"] for item in missing_requirements]
    return {
        "formula": formula,
        "geometry_file": geometry_path,
        "elements": elements,
        "transition_metals": transition_metals,
        "module": normalized_module,
        "task_operations": normalized_tasks,
        "functional": functional,
        "charge": inferred_charge,
        "multiplicity": multiplicity,
        "assumptions": assumptions,
        "ready_to_create": ready_to_create,
        "recommended_tool": recommended_tool,
        "missing_requirements": missing_requirements,
        "next_questions": next_questions,
        "warnings": warnings,
        "basis_preview": basis_preview,
        "input_summary": geometry_summary,
    }


def create_nwchem_input_variant(
    source_input: str,
    changes: dict[str, str],
    reason: str = "",
    output_path: str | None = None,
    write_file: bool = True,
) -> dict[str, Any]:
    """Create a versioned copy of an NWChem input with specified keyword changes.

    ``changes`` maps directive keys to new values.  Supported keys:

    * ``"memory"`` – e.g. ``"800 mb"``  (replaces the ``memory`` line)
    * ``"charge"`` – e.g. ``"2"``
    * ``"mult"`` – e.g. ``"6"``
    * ``"dft.iterations"`` – e.g. ``"200"``
    * ``"dft.xc"`` – e.g. ``"pbe0"``
    * ``"dft.convergence energy"`` – e.g. ``"1e-7"``
    * ``"scf.maxiter"`` – e.g. ``"200"``
    * ``"task"`` – e.g. ``"dft optimize"``  (replaces the last task line)
    * any other ``"block.keyword"`` pair – best-effort replacement inside
      ``block ... end``

    If *output_path* is ``None``, ``next_versioned_path`` is called on the
    source to generate ``_v2.nw``, ``_v3.nw``, etc.  The original file is
    never overwritten.
    """
    from chemtools.programs.nwchem.runner import next_versioned_path as _next_versioned_path

    src = Path(source_input)
    if not src.exists():
        raise FileNotFoundError(f"Source input not found: {source_input}")

    text = src.read_text(encoding="utf-8")
    diff_summary: list[dict[str, str | None]] = []

    for key, new_value in changes.items():
        old_value: str | None = None

        if key == "memory":
            m = re.search(r"^(\s*memory\s+)(.+)$", text, re.MULTILINE | re.IGNORECASE)
            if m:
                old_value = m.group(2).strip()
                text = text[: m.start()] + m.group(1) + new_value + text[m.end() :]
            else:
                old_value = None
                text = f"memory {new_value}\n" + text

        elif key == "charge":
            m = re.search(r"^(\s*charge\s+)(\S+)", text, re.MULTILINE | re.IGNORECASE)
            if m:
                old_value = m.group(2)
                text = text[: m.start()] + m.group(1) + new_value + text[m.end() :]
            else:
                old_value = None
                # insert before first geometry block
                gm = re.search(r"^\s*geometry\b", text, re.MULTILINE | re.IGNORECASE)
                pos = gm.start() if gm else 0
                text = text[:pos] + f"charge {new_value}\n" + text[pos:]

        elif key == "mult":
            # NWChem: inside geometry block or as standalone keyword
            m = re.search(
                r"^(\s*(?:geometry\b[^\n]*\n(?:.*\n)*?\s*))?(mult(?:iplicity)?\s+)(\d+)",
                text, re.MULTILINE | re.IGNORECASE,
            )
            if not m:
                # Try as standalone
                m2 = re.search(r"^(\s*mult(?:iplicity)?\s+)(\d+)", text, re.MULTILINE | re.IGNORECASE)
                if m2:
                    old_value = m2.group(2)
                    text = text[: m2.start()] + m2.group(1) + new_value + text[m2.end() :]
                else:
                    old_value = None
                    gm = re.search(r"^\s*geometry\b", text, re.MULTILINE | re.IGNORECASE)
                    pos = gm.start() if gm else 0
                    text = text[:pos] + f"mult {new_value}\n" + text[pos:]
            else:
                old_value = m.group(3)
                text = text[: m.start(2)] + f"mult {new_value}" + text[m.end(3) :]

        elif key == "task":
            # Replace the last task line
            task_matches = list(re.finditer(r"^\s*task\s+.*$", text, re.MULTILINE | re.IGNORECASE))
            if task_matches:
                last = task_matches[-1]
                old_value = last.group(0).strip()
                text = text[: last.start()] + f"task {new_value}" + text[last.end() :]
            else:
                old_value = None
                text = text.rstrip() + f"\ntask {new_value}\n"

        elif "." in key:
            # block.keyword pattern, e.g. "dft.iterations" or "dft.xc"
            block_name, kw = key.split(".", 1)
            block_pat = re.compile(
                rf"^(\s*{re.escape(block_name)}\b[^\n]*\n)(.*?)(^\s*end\b)",
                re.MULTILINE | re.DOTALL | re.IGNORECASE,
            )
            bm = block_pat.search(text)
            if bm:
                block_body = bm.group(2)
                kw_pat = re.compile(
                    rf"^(\s*{re.escape(kw)}\s+)(.+)$",
                    re.MULTILINE | re.IGNORECASE,
                )
                km = kw_pat.search(block_body)
                if km:
                    old_value = km.group(2).strip()
                    new_body = block_body[: km.start()] + km.group(1) + new_value + block_body[km.end() :]
                else:
                    old_value = None
                    new_body = block_body + f"  {kw} {new_value}\n"
                text = text[: bm.start(2)] + new_body + text[bm.start(3) :]

        diff_summary.append({
            "key": key,
            "old": old_value,
            "new": new_value,
        })

    # Determine output path
    if output_path is None:
        output_path = _next_versioned_path(source_input)

    written_file: str | None = None
    if write_file:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text, encoding="utf-8")
        written_file = output_path

    return {
        "output_file": output_path,
        "written_file": written_file,
        "source_input": source_input,
        "diff_summary": diff_summary,
        "reason": reason,
        "input_text": text,
    }



__all__ = [
    "create_nwchem_input",
    "review_nwchem_input_request",
    "create_nwchem_input_variant",
]

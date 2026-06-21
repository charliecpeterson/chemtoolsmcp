"""NWChem TCE (Tensor Contraction Engine) input drafters.

Four entry points + a couple of helpers:

  * draft_nwchem_tce_input          Render a TCE input (CCSD / CCSD(T) /
                                    MP2 etc.) from a converged SCF/DFT
                                    reference, with explicit `freeze N`
                                    and ordering-aware vectors.
  * validate_nwchem_tce_setup       Cross-check a drafted TCE input
                                    against the SCF reference (freeze
                                    count, orbital ordering, vectors path).
  * draft_nwchem_atom_input         Atom-only single-point used to derive
                                    ECP core counts for the freeze-count
                                    calculation.
  * draft_nwchem_tce_restart_input  Re-render a TCE input from an
                                    interrupted run, reusing the partial
                                    movecs.

TCE setup is the most error-prone NWChem workflow for an LLM agent —
the project rule "never use `freeze atomic`" means each input needs an
explicit count, and the orbital ordering must be inspected before
trusting that count. The companion thick tool
`prepare_nwchem_tce_setup` (in strategy/active_space.py) orchestrates
parse_movecs + ordering check + swap suggestion + draft routing into
one call.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Any

from chemtools.core.common import read_text, make_metadata
from chemtools.programs.nwchem.parse.input import (
    inspect_nwchem_input,
    inspect_nwchem_ecp_block,
    extract_nwchem_geometry_block,
    render_nwchem_geometry_block,
)
from chemtools.programs.nwchem.parse.mos import parse_mos as _parse_mos_raw
from chemtools.programs.nwchem.parse.tce import (
    suggest_tce_freeze_count,
    analyze_tce_orbital_ordering,
    parse_tce_output,
)
from chemtools.programs.nwchem.input._utils import (
    _coerce_api_int,
    _coerce_api_float,
)


def _normalize_stem_for_match(stem: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def _stem_tokens(stem: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", stem.lower()) if token]


# ---------------------------------------------------------------------------
# TCE input drafting
# ---------------------------------------------------------------------------

def _extract_ecp_nelec_from_input(
    input_path: str,
    basis_library_path: str | None = None,
) -> dict[str, int]:
    """Return {element: nelec} for all ECP-covered elements in an NWChem input.

    Tries inline ``nelec`` lines first (explicit ECP blocks), then falls back
    to the basis library for library-assigned ECPs.  Returns an empty dict if
    no ECP block exists.
    """
    from chemtools.programs.nwchem.parse.input import inspect_nwchem_ecp_block
    try:
        ecp_info = inspect_nwchem_ecp_block(input_path)
    except (ValueError, FileNotFoundError):
        # ValueError: no ECP block in file (expected for most inputs)
        # FileNotFoundError: input file doesn't exist
        return {}

    if not ecp_info.get("body_lines"):
        return {}

    # Inline nelec parsed directly from the ECP block body
    result: dict[str, int] = dict(ecp_info.get("nelec_by_element") or {})

    # Library-assigned elements: look up nelec from the basis library
    library_assignments = ecp_info.get("library_assignments") or {}
    if library_assignments and basis_library_path:
        from chemtools.programs.nwchem.input.basis import resolve_ecp
        for elem, ecp_name in library_assignments.items():
            if elem in result:
                continue
            try:
                resolved = resolve_ecp(ecp_name, [elem], basis_library_path)
                nelec = (resolved.get("nelec_by_element") or {}).get(elem)
                if nelec is not None:
                    result[elem] = nelec
            except Exception:  # ECP not in library for this element/name combination
                pass

    return result


def draft_nwchem_tce_input(
    scf_output_file: str,
    input_file: str,
    method: str = "mp2",
    freeze_count: int | None = None,
    swap_pairs: list[tuple[int, int]] | None = None,
    movecs_file: str | None = None,
    ecp_core_electrons: dict[str, int] | None = None,
    basis_library: str | None = None,
    start_name: str | None = None,
    title: str | None = None,
    memory: str | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    """Design a NWChem TCE input, inspecting SCF orbitals to determine freeze count.

    The agent MUST call this AFTER a completed SCF calculation so that the orbital
    ordering can be verified.  This function:

    1. Reads the SCF output and parses molecular orbitals.
    2. Computes a chemically-informed freeze count from the element list + ECP info.
    3. Checks the actual orbital ordering against the expected core pattern.
    4. Warns if swaps are needed (the agent should call swap_nwchem_movecs first).
    5. Returns a ready-to-use TCE input block with an explicit ``freeze N`` directive.

    Never uses ``freeze atomic`` — always emits an explicit integer.

    Parameters
    ----------
    scf_output_file:
        Path to the completed SCF/DFT output that contains the MO analysis.
    input_file:
        Path to the SCF input file (for geometry/basis/ECP metadata).
    method:
        TCE method: ``mp2``, ``ccsd``, or ``ccsd(t)``.  Default ``mp2``.
    freeze_count:
        Override the suggested freeze count.  If None, computed from chemistry.
    swap_pairs:
        List of (i, j) MO index pairs that have already been applied via
        swap_nwchem_movecs. If provided, the input notes that swaps were done.
    movecs_file:
        Path to the movecs file.  If None, inferred from the SCF output.
    ecp_core_electrons:
        ECP nelec values per element, e.g. ``{"Zn": 10, "I": 28}``.
    start_name, title, memory:
        NWChem input header directives.
    output_dir, base_name:
        Where to write the file (if write_file=True).
    write_file:
        If True, write the generated input to disk.
    """
    from chemtools.programs.nwchem.parse.tce import suggest_tce_freeze_count, analyze_tce_orbital_ordering
    from chemtools.core.common import read_text

    method_norm = method.strip().lower()
    valid_methods = {"mp2", "ccsd", "ccsd(t)", "ccsdt"}
    if method_norm not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}")
    tce_method_keyword = {
        "mp2": "mp2",
        "ccsd": "ccsd",
        "ccsd(t)": "ccsd(t)",
        "ccsdt": "ccsd(t)",
    }[method_norm]

    # --- Read SCF output and parse orbitals ---
    scf_contents = read_text(scf_output_file)
    mos_result = _parse_mos_raw(scf_output_file, scf_contents)
    orbitals = mos_result.get("orbitals", [])

    # --- Infer elements from the input file ---
    input_summary = inspect_nwchem_input(input_file)
    elements = input_summary.get("elements", [])

    # --- Auto-detect ECP nelec from input file (if not provided by caller) ---
    ecp_auto_detected: bool = False
    if ecp_core_electrons is None:
        detected = _extract_ecp_nelec_from_input(input_file, basis_library_path=basis_library)
        if detected:
            ecp_core_electrons = detected
            ecp_auto_detected = True

    # --- Suggest freeze count ---
    freeze_suggestion = suggest_tce_freeze_count(
        elements,
        ecp_core_electrons=ecp_core_electrons,
        charge=input_summary.get("charge") or 0,
        multiplicity=input_summary.get("multiplicity") or 1,
    )
    suggested_freeze = freeze_suggestion["freeze_count"]
    effective_freeze = freeze_count if freeze_count is not None else suggested_freeze

    # --- Analyse orbital ordering ---
    ordering_analysis: dict[str, Any] = {}
    if orbitals and effective_freeze > 0:
        ordering_analysis = analyze_tce_orbital_ordering(orbitals, effective_freeze)

    # --- Determine start_name and movecs ---
    scf_stem = Path(scf_output_file).stem
    resolved_start = start_name or base_name or scf_stem

    # Try to infer movecs from the SCF output text
    resolved_movecs = movecs_file
    if resolved_movecs is None:
        for line in scf_contents.splitlines():
            if "output vectors" in line.lower() and "=" in line:
                candidate = line.split("=", 1)[-1].strip().strip("./")
                candidate_path = Path(scf_output_file).parent / candidate
                if candidate_path.exists():
                    resolved_movecs = str(candidate_path.resolve())
                break

    has_ordering_warnings = bool(ordering_analysis.get("warnings"))
    pending_swaps = ordering_analysis.get("swap_suggestions", [])
    swaps_applied = swap_pairs or []

    # --- Build tce block ---
    tce_lines: list[str] = [f"  {tce_method_keyword}"]
    tce_lines.append(f"  freeze {effective_freeze}")
    tce_block = "tce\n" + "\n".join(tce_lines) + "\nend"

    # Always save T1 and T2 amplitude files so parse_nwchem_tce_amplitudes can
    # compute T1/D1/T2 diagnostics after the run.
    save_t_directive = "set tce:save_t T T"

    # --- Build geometry block (always include with symmetry c1 for TCE) ---
    from chemtools.programs.nwchem.parse.input import extract_nwchem_geometry_block, render_nwchem_geometry_block
    geo_section: str | None = None
    charge_line: str | None = None
    try:
        geo = extract_nwchem_geometry_block(input_file)
        directives = [d for d in (geo.get("directives") or []) if not d.lower().startswith("symmetry")]
        directives.insert(0, "symmetry c1")
        geo_section = render_nwchem_geometry_block(geo["header_line"], geo["atoms"], directives=directives)
    except Exception:
        pass
    charge = input_summary.get("charge")
    if charge is not None:
        charge_line = f"charge {charge}"

    # --- Build scf block (reference type from multiplicity) ---
    mult = input_summary.get("multiplicity") or 1
    nopen = mult - 1
    if nopen > 0:
        scf_ref = "rohf"
        scf_lines = ["scf", f"  {scf_ref}", f"  nopen {nopen}", "  thresh 1e-8", "  maxiter 200"]
    else:
        scf_ref = "rhf"
        scf_lines = ["scf", f"  {scf_ref}", "  thresh 1e-8", "  maxiter 200"]
    if resolved_movecs:
        movecs_basename = Path(resolved_movecs).name
        tce_movecs_out = f"{resolved_start}.movecs"
        scf_lines.append(f"  vectors input {movecs_basename} output {tce_movecs_out}")
    scf_block = "\n".join(scf_lines) + "\nend"

    # --- Assemble explanatory comment ---
    freeze_comment_lines = [
        f"# TCE {tce_method_keyword.upper()} — freeze analysis",
        f"# Effective freeze count: {effective_freeze} orbitals",
    ]
    if freeze_suggestion["per_element"]:
        freeze_comment_lines.append("# Per-element core orbital counts:")
        for pe in freeze_suggestion["per_element"]:
            if pe.get("freeze_orbitals") is not None:
                n_atoms = pe.get("n_atoms", 1)
                ecp_note = (
                    f" (ECP removes {pe['ecp_orbitals_removed_per_atom']} orb/atom)"
                    if pe.get("ecp_electrons", 0) > 0
                    else ""
                )
                atom_str = f"{n_atoms}×" if n_atoms > 1 else ""
                freeze_comment_lines.append(
                    f"#   {pe['element']} ({n_atoms} atoms): "
                    f"{pe.get('all_electron_core_orbitals_per_atom', '?')} all-e core/atom"
                    f"{ecp_note} → {atom_str}{pe['freeze_orbitals_per_atom']}={pe['freeze_orbitals']} orbitals"
                )

    if ordering_analysis.get("proposed_freeze_orbitals"):
        freeze_comment_lines.append("# Proposed frozen MOs (from SCF output):")
        for orb_info in ordering_analysis["proposed_freeze_orbitals"]:
            char = orb_info.get("dominant_character") or "?"
            freeze_comment_lines.append(
                f"#   MO {orb_info['mo']:3d}: E={orb_info['energy_hartree']:10.4f} h  {char}"
            )

    if ordering_analysis.get("warnings"):
        freeze_comment_lines.append("#")
        freeze_comment_lines.append("# *** ORBITAL ORDERING WARNINGS ***")
        for w in ordering_analysis["warnings"]:
            freeze_comment_lines.append(f"# {w}")

    if pending_swaps and not swaps_applied:
        freeze_comment_lines.append("#")
        freeze_comment_lines.append(
            "# ACTION REQUIRED: run swap_nwchem_movecs for each pair BEFORE this input:"
        )
        for sw in pending_swaps:
            freeze_comment_lines.append(
                f"#   swap MO {sw['from_mo']} <-> MO {sw['to_mo']}: {sw['reason']}"
            )

    if swaps_applied:
        freeze_comment_lines.append("#")
        freeze_comment_lines.append("# Swaps already applied to movecs:")
        for s_i, s_j in swaps_applied:
            freeze_comment_lines.append(f"#   MO {s_i} <-> MO {s_j}")

    comment_block = "\n".join(freeze_comment_lines)

    # --- Extract basis/ECP blocks from SCF input file ---
    basis_section: str | None = None
    ecp_section: str | None = None
    try:
        from chemtools.programs.nwchem.parse.input import inspect_all_nwchem_basis_blocks
        basis_blocks = inspect_all_nwchem_basis_blocks(input_file)
        if basis_blocks:
            raw = basis_blocks[-1]
            basis_section = raw["header_line"] + "\n" + "\n".join(raw["body_lines"]) + "\nend"
    except Exception:
        pass
    try:
        ecp_info = inspect_nwchem_ecp_block(input_file)
        if ecp_info.get("body_lines"):
            ecp_section = ecp_info["header_line"] + "\n" + "\n".join(ecp_info["body_lines"]) + "\nend"
    except Exception:
        pass

    # --- Assemble input sections ---
    sections: list[str] = [
        f"start {resolved_start}",
        "echo",
    ]
    if memory:
        sections.append(f"memory {memory}")
    if geo_section:
        sections.append(geo_section)
    if charge_line:
        sections.append(charge_line)
    if basis_section:
        sections.append(basis_section)
    if ecp_section:
        sections.append(ecp_section)
    sections.append(comment_block)
    sections.append(scf_block)
    sections.append("task scf energy")
    sections.append(tce_block)
    sections.append(save_t_directive)
    sections.append("task tce energy")

    input_text = "\n\n".join(sections).rstrip() + "\n"

    # --- File plan ---
    method_tag = tce_method_keyword.replace("(", "").replace(")", "")
    out_stem = base_name or f"{resolved_start}_tce_{method_tag}"
    out_dir = Path(output_dir) if output_dir else Path(scf_output_file).parent
    out_path = out_dir / f"{out_stem}.nw"

    written_file: str | None = None
    if write_file:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(input_text, encoding="utf-8")
        written_file = str(out_path.resolve())

    return {
        "input_text": input_text,
        "written_file": written_file,
        "planned_output_file": str(out_path),
        "method": tce_method_keyword,
        "effective_freeze_count": effective_freeze,
        "suggested_freeze_count": suggested_freeze,
        "n_electrons": freeze_suggestion.get("n_electrons"),
        "n_correlated": freeze_suggestion.get("n_correlated"),
        "ecp_core_electrons": ecp_core_electrons or {},
        "ecp_auto_detected": ecp_auto_detected,
        "freeze_suggestion": freeze_suggestion,
        "orbital_ordering_analysis": ordering_analysis,
        "needs_orbital_swap": has_ordering_warnings,
        "pending_swap_suggestions": pending_swaps,
        "movecs_file": resolved_movecs,
        "elements": elements,
        "n_orbitals_parsed": len(orbitals),
        "warnings": freeze_suggestion.get("warnings", []) + ordering_analysis.get("warnings", []),
    }


def validate_nwchem_tce_setup(
    tce_input_path: str,
    scf_output_path: str | None = None,
) -> dict[str, Any]:
    """Cross-check a NWChem TCE input file for common setup errors.

    Catches issues before submitting the job:

    * Missing ``symmetry c1`` in the geometry block
    * ``freeze atomic`` (forbidden — always use explicit count)
    * No freeze directive (all electrons correlated — expensive and wrong)
    * Freeze count far outside the element-derived suggestion
    * Missing or unreachable vectors file
    * ROHF reference missing for open-shell system
    * No TCE method keyword found

    Parameters
    ----------
    tce_input_path:
        Path to the NWChem TCE input file to validate.
    scf_output_path:
        Optional path to the SCF output file.  If provided and the file does
        not yet exist, a warning is emitted.

    Returns
    -------
    dict with ``status`` ("ok" | "warnings" | "errors"), ``issues`` list,
    ``detected`` parsed fields, and ``summary`` text.
    """
    import re as _re
    from pathlib import Path
    from chemtools.programs.nwchem.parse.tce import suggest_tce_freeze_count as _suggest_freeze
    from chemtools.programs.nwchem.parse.input import inspect_nwchem_input

    issues: list[dict[str, Any]] = []

    def _err(code: str, message: str) -> None:
        issues.append({"level": "error", "code": code, "message": message})

    def _warn(code: str, message: str) -> None:
        issues.append({"level": "warning", "code": code, "message": message})

    # --- Read file content once ---
    contents = read_text(tce_input_path)
    contents_lower = contents.lower()

    # --- Parse high-level input summary ---
    tce_summary = inspect_nwchem_input(tce_input_path)
    elements: list[str] = tce_summary.get("elements") or []
    charge: int = tce_summary.get("charge") or 0
    multiplicity: int | None = tce_summary.get("multiplicity")
    open_shell = multiplicity is not None and multiplicity > 1

    # --- Extract raw tce block via regex ---
    tce_block_match = _re.search(r"\btce\b(.*?)\bend\b", contents, _re.IGNORECASE | _re.DOTALL)
    tce_block = tce_block_match.group(1) if tce_block_match else ""
    tce_block_lower = tce_block.lower()

    # --- Extract geometry block via regex ---
    geo_block_match = _re.search(r"\bgeometry\b(.*?)\bend\b", contents, _re.IGNORECASE | _re.DOTALL)
    geo_block = geo_block_match.group(1) if geo_block_match else ""

    # --- Extract SCF block via regex ---
    scf_block_match = _re.search(r"\bscf\b(.*?)\bend\b", contents, _re.IGNORECASE | _re.DOTALL)
    scf_block = scf_block_match.group(1) if scf_block_match else ""
    scf_block_lower = scf_block.lower()

    # --- Method check ---
    method_found: str | None = None
    for m in ("ccsd(t)", "ccsd", "mp2", "cisd", "mbpt2", "mbpt3"):
        if m in tce_block_lower:
            method_found = m
            break
    if not tce_block:
        _err("tce_block_missing", "No 'tce...end' block found in the input file.")
    elif not method_found:
        _err("tce_no_method",
             "No recognized TCE method keyword found in the tce block. Expected: ccsd, mp2, ccsd(t), etc.")

    # --- Symmetry check ---
    _TCE_ABELIAN = {"c1", "ci", "cs", "c2", "c2v", "c2h", "d2", "d2h"}
    _sym_m = _re.search(r"\bsymmetry\s+(\S+)\b", geo_block, _re.IGNORECASE)
    _sym_group = _sym_m.group(1).lower() if _sym_m else None
    if _sym_group not in _TCE_ABELIAN:
        if _sym_group is None:
            _err("tce_missing_symmetry_c1",
                 "The geometry block is missing a symmetry directive. "
                 "NWChem TCE requires Abelian symmetry — add 'symmetry c1' (or d2h, c2v, etc.) "
                 "inside the geometry...end block.")
        else:
            _err("tce_missing_symmetry_c1",
                 f"Symmetry '{_sym_group}' is non-Abelian and not supported by NWChem TCE. "
                 f"Use one of: {', '.join(sorted(_TCE_ABELIAN))}.")

    # --- SCF reference check ---
    scf_ref = None
    for ref in ("rohf", "rhf", "uhf"):
        if _re.search(rf"\b{ref}\b", scf_block_lower):
            scf_ref = ref
            break
    if open_shell and scf_ref and scf_ref != "rohf":
        _err("tce_wrong_scf_reference",
             f"Open-shell system (mult={multiplicity}) requires ROHF reference, "
             f"but '{scf_ref}' was found in the SCF block. "
             "Use scf_type='rohf' and nopen=multiplicity-1.")
    elif open_shell and not scf_ref:
        _warn("tce_scf_reference_undetected",
              f"Open-shell system (mult={multiplicity}) but no SCF reference keyword (rohf/rhf/uhf) "
              "was found. Ensure 'rohf' is present in the scf block.")
    if scf_ref == "uhf" and not open_shell:
        _warn("tce_uhf_closed_shell",
              "UHF reference for closed-shell system — RHF is preferred as the TCE reference.")

    # --- Vectors file check ---
    vectors_match = _re.search(r"\bvectors\s+input\s+(\S+)", tce_block, _re.IGNORECASE)
    vectors_path_str: str | None = None
    if vectors_match:
        vectors_path_str = vectors_match.group(1)
        candidate = Path(vectors_path_str)
        if not candidate.is_absolute():
            candidate = Path(tce_input_path).parent / vectors_path_str
        if not candidate.exists():
            _err("tce_vectors_file_missing",
                 f"Vectors file '{vectors_path_str}' does not exist at '{candidate}'. "
                 "Run the SCF job first and verify the path.")
    else:
        _warn("tce_no_vectors_input",
              "No 'vectors input ...' directive found in the tce block. NWChem will use default vectors — "
              "results may be wrong for open-shell or orbital-reordered systems.")

    # --- Freeze count check ---
    freeze_match = _re.search(r"\bfreeze\s+(\d+)\b", tce_block, _re.IGNORECASE)
    actual_freeze: int | None = None
    if freeze_match:
        actual_freeze = int(freeze_match.group(1))
        if elements:
            try:
                suggestion = _suggest_freeze(elements, charge=charge, multiplicity=multiplicity or 1)
                suggested = suggestion.get("freeze_count", 0)
                n_elec = suggestion.get("n_electrons") or 0
                min_correlated = 2
                max_freeze = max(0, n_elec // 2 - min_correlated)
                if max_freeze > 0 and actual_freeze > max_freeze:
                    _err("tce_freeze_too_large",
                         f"Freeze count {actual_freeze} leaves fewer than {min_correlated} correlated electrons "
                         f"({n_elec} total electrons). This is likely wrong.")
                elif actual_freeze < suggested - 2:
                    _warn("tce_freeze_too_small",
                          f"Freeze count {actual_freeze} is less than suggested {suggested}. "
                          "Under-freezing wastes compute and may affect energetics.")
                elif actual_freeze > suggested + 2:
                    _warn("tce_freeze_larger_than_suggested",
                          f"Freeze count {actual_freeze} exceeds suggested {suggested}. "
                          "Verify this is intentional and not over-freezing valence electrons.")
            except Exception:
                pass
    elif "freeze atomic" in tce_block_lower:
        _err("tce_freeze_atomic_forbidden",
             "'freeze atomic' is forbidden in TCE inputs — always specify an explicit count "
             "(e.g. 'freeze 10'). Use suggest_nwchem_tce_freeze to compute the correct value.")
    else:
        _warn("tce_no_freeze",
              "No 'freeze N' directive found. All electrons will be correlated — this is very expensive "
              "and almost always wrong. Use suggest_nwchem_tce_freeze to compute the correct freeze count.")

    # --- SCF output existence check ---
    if scf_output_path and not Path(scf_output_path).exists():
        _warn("tce_scf_output_missing",
              f"SCF output file '{scf_output_path}' does not exist yet. Run the SCF job first.")

    # --- Build summary ---
    n_errors = sum(1 for i in issues if i["level"] == "error")
    n_warnings = sum(1 for i in issues if i["level"] == "warning")
    status = "errors" if n_errors else ("warnings" if n_warnings else "ok")

    summary_lines = [f"TCE setup validation: {status}"]
    if n_errors:
        summary_lines.append(f"  {n_errors} error(s) must be fixed before submitting.")
    if n_warnings:
        summary_lines.append(f"  {n_warnings} warning(s) worth reviewing.")
    if status == "ok":
        summary_lines.append("  No issues found. Input looks correct for TCE.")
    for iss in issues:
        summary_lines.append(f"  [{iss['level'].upper()}] {iss['code']}: {iss['message']}")

    return {
        "tce_input_file": tce_input_path,
        "scf_output_file": scf_output_path,
        "status": status,
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "issues": issues,
        "detected": {
            "method": method_found,
            "elements": elements,
            "charge": charge,
            "multiplicity": multiplicity,
            "scf_reference": scf_ref,
            "has_freeze_directive": actual_freeze is not None,
            "freeze_count": actual_freeze,
            "has_vectors_input": bool(vectors_match),
            "vectors_file": vectors_path_str,
        },
        "summary": "\n".join(summary_lines),
    }


# ---------------------------------------------------------------------------
# Atomic ground-state multiplicities (neutral, lowest term)
# ---------------------------------------------------------------------------

_ATOM_GROUND_MULT: dict[str, int] = {
    "H": 2, "He": 1, "Li": 2, "Be": 1, "B": 2, "C": 3, "N": 4,
    "O": 3, "F": 2, "Ne": 1, "Na": 2, "Mg": 1, "Al": 2, "Si": 3,
    "P": 4, "S": 3, "Cl": 2, "Ar": 1, "K": 2, "Ca": 1,
    # 3d transition metals
    "Sc": 2, "Ti": 3, "V": 4, "Cr": 7, "Mn": 6, "Fe": 5,
    "Co": 4, "Ni": 3, "Cu": 2, "Zn": 1,
    # p-block period 4
    "Ga": 2, "Ge": 3, "As": 4, "Se": 3, "Br": 2, "Kr": 1,
    "Rb": 2, "Sr": 1,
    # 4d transition metals
    "Y": 2, "Zr": 3, "Nb": 6, "Mo": 7, "Tc": 6, "Ru": 5,
    "Rh": 4, "Pd": 1, "Ag": 2, "Cd": 1,
    # p-block period 5
    "In": 2, "Sn": 3, "Sb": 4, "Te": 3, "I": 2, "Xe": 1,
    "Cs": 2, "Ba": 1,
    # 5d transition metals (simplified; use ROHF + MCSCF for production)
    "La": 2, "Hf": 3, "Ta": 4, "W": 5, "Re": 6, "Os": 5,
    "Ir": 4, "Pt": 3, "Au": 2, "Hg": 1,
    # p-block period 6
    "Tl": 2, "Pb": 3, "Bi": 4, "Po": 3, "At": 2, "Rn": 1,
}


def draft_nwchem_atom_input(
    element: str,
    basis: str,
    method: str = "scf",
    charge: int = 0,
    multiplicity: int | None = None,
    xc_functional: str = "m06",
    basis_assignments: dict[str, str] | None = None,
    ecp_assignments: dict[str, str] | None = None,
    memory: str | None = None,
    start_name: str | None = None,
    output_dir: str | None = None,
    write_file: bool = False,
    basis_library: str | None = None,
) -> dict[str, Any]:
    """Generate a NWChem input for a single atom (for atomization energies, IPs, etc.).

    Automatically looks up the neutral ground-state multiplicity for common elements.
    For ions, provide ``multiplicity`` explicitly or it will be estimated from electron
    count parity (with a warning).

    Parameters
    ----------
    element:
        Element symbol, e.g. ``"Fe"``, ``"O"``.
    basis:
        Basis set name (resolved from the local library), e.g. ``"6-31gs"``.
    method:
        NWChem module: ``"scf"``, ``"dft"``, or ``"mp2"``.
    charge:
        Total charge.  0 for neutral atom.
    multiplicity:
        Spin multiplicity (2S+1).  If None, looked up from the ground-state table;
        for ions, estimated from electron-count parity.
    xc_functional:
        XC functional used when ``method="dft"``.
    basis_assignments / ecp_assignments:
        Override basis/ECP per element (passed to render_nwchem_basis_setup).
    memory:
        NWChem memory directive string, e.g. ``"total 2000 mb"``.
    start_name:
        NWChem start name.  Defaults to ``"{element}_atom"``.
    output_dir, write_file:
        Where to write the file if ``write_file=True``.
    basis_library:
        Path to the basis library.  Auto-detected if None.
    """
    from pathlib import Path as _Path

    sym = element[0].upper() + element[1:].lower()

    # --- Determine multiplicity ---
    mult_source = "provided"
    if multiplicity is None:
        neutral_mult = _ATOM_GROUND_MULT.get(sym)
        if neutral_mult is None:
            raise ValueError(
                f"Unknown element '{sym}' or no ground-state table entry; "
                "provide multiplicity explicitly."
            )
        if charge == 0:
            multiplicity = neutral_mult
            mult_source = "ground_state_table"
        else:
            from chemtools.programs.nwchem.input._utils import ELEMENT_TO_Z
            z = ELEMENT_TO_Z.get(sym, 0)
            n_electrons = z - charge
            if n_electrons <= 0:
                raise ValueError(
                    f"Element {sym} (Z={z}) with charge {charge:+d} has {n_electrons} electrons."
                )
            # Parity rule: even electrons → even nopen, odd → odd nopen
            ion_nopen = n_electrons % 2
            multiplicity = ion_nopen + 1
            mult_source = "estimated_from_parity"

    nopen = multiplicity - 1

    # --- Resolve basis library path ---
    if basis_library is not None:
        lib_path = basis_library
    else:
        try:
            from importlib.resources import files as _pkg_files
            lib_path = str(_pkg_files("chemtools").joinpath("data/nwchem/basis_library"))
        except Exception:
            lib_path = None

    # Build basis block
    try:
        from chemtools.programs.nwchem.input.basis_library import render_nwchem_basis_block as _render_basis
        basis_info = _render_basis(
            basis_name=basis,
            elements=[sym],
            library_path=lib_path,
        )
        basis_block = basis_info["text"]
        ecp_block = basis_info.get("ecp_text") or None
    except Exception as exc:
        raise ValueError(f"Could not render basis '{basis}' for element '{sym}': {exc}") from exc

    # --- Build SCF/DFT block ---
    resolved_start = start_name or f"{sym.lower()}_atom"
    method_norm = method.strip().lower()

    if method_norm == "dft":
        if nopen > 0:
            scf_block = f"dft\n  odft\n  mult {multiplicity}\n  xc {xc_functional}\n  thresh 1e-8\n  maxiter 200\nend"
        else:
            scf_block = f"dft\n  xc {xc_functional}\n  thresh 1e-8\n  maxiter 200\nend"
        task_line = "task dft energy"
    elif method_norm in ("scf", "rohf", "rhf", "uhf"):
        if nopen > 0:
            scf_block = f"scf\n  rohf\n  nopen {nopen}\n  thresh 1e-8\n  maxiter 200\nend"
        else:
            scf_block = "scf\n  rhf\n  thresh 1e-8\n  maxiter 200\nend"
        task_line = "task scf energy"
    elif method_norm == "mp2":
        if nopen > 0:
            scf_block = f"scf\n  rohf\n  nopen {nopen}\n  thresh 1e-8\n  maxiter 200\nend"
        else:
            scf_block = "scf\n  rhf\n  thresh 1e-8\n  maxiter 200\nend"
        task_line = "task mp2 energy"
    else:
        raise ValueError(f"Unsupported method '{method}' for draft_nwchem_atom_input. Use scf, dft, or mp2.")

    # --- Geometry block (single atom at origin, always c1) ---
    geo_block = f"geometry units angstroms\n  symmetry c1\n  {sym}  0.00000  0.00000  0.00000\nend"

    # --- Assemble sections ---
    sections: list[str] = [f"start {resolved_start}", "echo"]
    if memory:
        sections.append(f"memory {memory}")
    sections.append(geo_block)
    if charge != 0:
        sections.append(f"charge {charge}")
    sections.append(basis_block)
    if ecp_block:
        sections.append(ecp_block)
    sections.append(scf_block)
    sections.append(task_line)

    input_text = "\n\n".join(sections).rstrip() + "\n"

    # --- Write file ---
    out_dir = _Path(output_dir) if output_dir else _Path(".")
    out_path = out_dir / f"{resolved_start}.nw"
    written_file: str | None = None
    if write_file:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(input_text, encoding="utf-8")
        written_file = str(out_path.resolve())

    warnings: list[str] = []
    if mult_source == "estimated_from_parity":
        warnings.append(
            f"Multiplicity {multiplicity} for {sym}{charge:+d} is estimated from electron-count parity only. "
            "Verify the correct ground state for this ion before running."
        )

    return {
        "input_text": input_text,
        "written_file": written_file,
        "planned_output_file": str(out_path),
        "element": sym,
        "charge": charge,
        "multiplicity": multiplicity,
        "nopen": nopen,
        "multiplicity_source": mult_source,
        "method": method_norm,
        "basis": basis,
        "start_name": resolved_start,
        "warnings": warnings,
    }


def draft_nwchem_tce_restart_input(
    tce_output_file: str,
    tce_input_file: str | None = None,
    max_iterations: int = 200,
    thresh: float = 1e-5,
    copy_amplitudes: bool = True,
    output_dir: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    """Generate a NWChem TCE restart input from a stalled or incomplete CCSD/MP2 run.

    This function:

    1. Parses the previous TCE output to determine method, freeze count, and
       convergence state (last iteration + residual).
    2. Locates saved amplitude files (``{stem}.t1amp.*`` or ``{stem}.t1_copy.*``).
    3. Optionally copies them to ``{start_name}.t1`` / ``{start_name}.t2`` so
       NWChem can read them via ``set tce:read_ta .true.``.
    4. Builds a ``restart`` input with ``set tce:read_ta .true.``,
       ``set tce:save_t T T``, the requested ``maxiter``, and ``thresh``.

    Parameters
    ----------
    tce_output_file:
        Path to the incomplete TCE output (``.out``) file.
    tce_input_file:
        Path to the previous TCE input (``.nw``) file.  Used to extract
        geometry, basis, ECP, and SCF blocks.  Auto-inferred from output stem
        if None.
    max_iterations:
        Maximum CCSD iterations for the restart run (default 200).
    thresh:
        CCSD residual convergence threshold (default 1e-5, 10× looser than
        the 1e-6 NWChem default — suitable for slowly-converging systems).
    copy_amplitudes:
        If True, copy the found amplitude files to the correct restart names
        (``{start_name}.t1`` / ``{start_name}.t2``).
    output_dir, write_file:
        Where to write the restart input file.
    """
    import glob as _glob
    import shutil as _shutil
    from pathlib import Path as _Path
    from chemtools.core.common import read_text as _read_text
    from chemtools.programs.nwchem.parse.tce import parse_tce_output as _parse_tce

    out_path = _Path(tce_output_file).resolve()
    out_dir = out_path.parent
    stem = out_path.stem

    # --- Parse previous TCE output ---
    contents = _read_text(tce_output_file)
    tce_result = _parse_tce(tce_output_file, contents)

    method = tce_result.get("method") or "CCSD"
    method_kw = method.lower().replace("(", "").replace(")", "")
    tce_method_kw = {"mp2": "mp2", "ccsd": "ccsd", "ccsdt": "ccsd(t)"}.get(method_kw, "ccsd")

    # Extract last iteration count and residual from output
    import re as _re
    iter_re = _re.compile(
        r"^\s*\d+\s+[-\d.Ee+]+\s+([\d.Ee+\-]+)\s*$", _re.IGNORECASE
    )
    last_iter: int | None = None
    last_residual: float | None = None
    iter_count = 0
    for line in contents.splitlines():
        m = iter_re.match(line)
        if m:
            iter_count += 1
            try:
                last_residual = float(m.group(1))
                last_iter = iter_count
            except ValueError:
                pass

    # Extract frozen core count from TCE section
    freeze_count: int | None = None
    for sec in tce_result.get("tce_sections", []):
        if sec.get("frozen_cores") is not None:
            freeze_count = sec["frozen_cores"]

    # --- Determine start_name from output stem ---
    # Output file might be named {start_name}.out or {start_name}_restart.out etc.
    start_name = stem
    # Try to read from the input file if available
    inferred_input = tce_input_file
    if inferred_input is None:
        for candidate in [
            out_dir / f"{stem}.nw",
            out_dir / f"{stem.replace('_restart', '')}.nw",
            out_dir / f"{stem.replace('_ccsd', '_ccsd_tce_ccsd')}.nw",
        ]:
            if candidate.exists():
                inferred_input = str(candidate)
                break

    if inferred_input and _Path(inferred_input).exists():
        try:
            from chemtools.programs.nwchem.parse.input import inspect_nwchem_input as _inspect
            summary = _inspect(inferred_input)
            blocks = summary.get("start_blocks", [])
            if blocks and blocks[0].get("start_name"):
                start_name = blocks[0]["start_name"]
            elif summary.get("start_present"):
                # Read the start line directly
                for line in _read_text(inferred_input).splitlines():
                    m2 = _re.match(r"^\s*(?:start|restart)\s+(\S+)", line, _re.IGNORECASE)
                    if m2:
                        start_name = m2.group(1)
                        break
        except Exception:
            pass

    # --- Locate amplitude files ---
    amp_patterns = [
        (f"{start_name}.t1amp.*", f"{start_name}.t2amp.*"),
        (f"{stem}.t1amp.*", f"{stem}.t2amp.*"),
        (f"{start_name}.t1_copy.*", f"{start_name}.t2_copy.*"),
        (f"{stem}.t1_copy.*", f"{stem}.t2_copy.*"),
    ]
    t1_src: str | None = None
    t2_src: str | None = None
    for t1_pat, t2_pat in amp_patterns:
        t1_candidates = sorted(_glob.glob(str(out_dir / t1_pat)))
        t2_candidates = sorted(_glob.glob(str(out_dir / t2_pat)))
        if t1_candidates:
            t1_src = t1_candidates[-1]  # take latest
        if t2_candidates:
            t2_src = t2_candidates[-1]
        if t1_src:
            break

    # --- Copy amplitude files if requested ---
    copied_files: list[str] = []
    t1_dest = str(out_dir / f"{start_name}.t1")
    t2_dest = str(out_dir / f"{start_name}.t2")
    copy_errors: list[str] = []

    if copy_amplitudes:
        if t1_src:
            try:
                _shutil.copy2(t1_src, t1_dest)
                copied_files.append(f"{t1_src} → {t1_dest}")
            except Exception as exc:
                copy_errors.append(f"Could not copy T1 file: {exc}")
        else:
            copy_errors.append(
                f"No T1 amplitude file found (tried patterns: "
                f"{', '.join(p[0] for p in amp_patterns)})."
            )
        if t2_src:
            try:
                _shutil.copy2(t2_src, t2_dest)
                copied_files.append(f"{t2_src} → {t2_dest}")
            except Exception as exc:
                copy_errors.append(f"Could not copy T2 file: {exc}")
        else:
            copy_errors.append(
                f"No T2 amplitude file found."
            )

    can_read_amplitudes = bool(
        _Path(t1_dest).exists() and _Path(t2_dest).exists()
    )

    # --- Extract geometry, basis, ECP, SCF blocks from previous input ---
    geo_section: str | None = None
    basis_section: str | None = None
    ecp_section: str | None = None
    scf_section: str | None = None
    charge_line: str | None = None
    movecs_line: str | None = None

    if inferred_input and _Path(inferred_input).exists():
        try:
            from chemtools.programs.nwchem.parse.input import (
                extract_nwchem_geometry_block,
                render_nwchem_geometry_block,
                inspect_all_nwchem_basis_blocks,
                inspect_nwchem_ecp_block,
            )
            geo = extract_nwchem_geometry_block(inferred_input)
            # Keep existing symmetry (already set correctly)
            geo_section = render_nwchem_geometry_block(
                geo["header_line"], geo["atoms"], directives=geo.get("directives", [])
            )
        except Exception:
            pass

        try:
            basis_blocks = inspect_all_nwchem_basis_blocks(inferred_input)
            if basis_blocks:
                raw = basis_blocks[-1]
                basis_section = raw["header_line"] + "\n" + "\n".join(raw["body_lines"]) + "\nend"
        except Exception:
            pass

        try:
            ecp_info = inspect_nwchem_ecp_block(inferred_input)
            if ecp_info.get("body_lines"):
                ecp_section = ecp_info["header_line"] + "\n" + "\n".join(ecp_info["body_lines"]) + "\nend"
        except Exception:
            pass

        try:
            from chemtools.programs.nwchem.parse.input import extract_nwchem_module_block
            # Try to find the last SCF block before the TCE block
            scf_blk = extract_nwchem_module_block(inferred_input, module="scf", block_index=-1)
            # Strip existing vectors lines; we'll use the restart movecs
            scf_lines = [l for l in scf_blk["body_lines"] if "vectors" not in l.lower()]
            movecs_file = f"{start_name}.movecs"
            restart_movecs = str(out_dir / movecs_file)
            if _Path(restart_movecs).exists():
                scf_lines.append(f"  vectors input {movecs_file} output {movecs_file}")
                movecs_line = movecs_file
            scf_section = scf_blk["header_line"] + "\n" + "\n".join(scf_lines) + "\nend"
        except Exception:
            pass

        try:
            from chemtools.programs.nwchem.parse.input import inspect_nwchem_input as _inspect
            summary = _inspect(inferred_input)
            charge = summary.get("charge")
            if charge:
                charge_line = f"charge {charge}"
        except Exception:
            pass

    # --- Build TCE block ---
    tce_lines = [f"  {tce_method_kw}"]
    if freeze_count is not None:
        tce_lines.append(f"  freeze {freeze_count}")
    tce_lines.append(f"  maxiter {max_iterations}")
    tce_lines.append(f"  thresh {thresh:.2e}")
    tce_block = "tce\n" + "\n".join(tce_lines) + "\nend"

    read_ta_line = "set tce:read_ta .true." if can_read_amplitudes else "# set tce:read_ta .true.  # (enable once .t1/.t2 files are in place)"
    save_t_line = "set tce:save_t T T"

    # --- Assemble input ---
    sections_list: list[str] = [f"restart {start_name}", "echo"]
    if geo_section:
        sections_list.append(geo_section)
    if charge_line:
        sections_list.append(charge_line)
    if basis_section:
        sections_list.append(basis_section)
    if ecp_section:
        sections_list.append(ecp_section)
    if scf_section:
        sections_list.append(scf_section)
        sections_list.append("task scf energy")
    sections_list.append(read_ta_line)
    sections_list.append(save_t_line)
    sections_list.append(tce_block)
    sections_list.append("task tce energy")

    input_text = "\n\n".join(sections_list).rstrip() + "\n"

    # --- Write file ---
    resolved_outdir = _Path(output_dir) if output_dir else out_dir
    out_nw = resolved_outdir / f"{start_name}_restart.nw"
    written_file: str | None = None
    if write_file:
        resolved_outdir.mkdir(parents=True, exist_ok=True)
        out_nw.write_text(input_text, encoding="utf-8")
        written_file = str(out_nw.resolve())

    return {
        "input_text": input_text,
        "written_file": written_file,
        "planned_output_file": str(out_nw),
        "restart_start_name": start_name,
        "method": tce_method_kw,
        "freeze_count": freeze_count,
        "max_iterations": max_iterations,
        "thresh": thresh,
        "previous_tce_last_iter": last_iter,
        "previous_tce_last_residual": last_residual,
        "amplitude_files_found": {
            "t1": t1_src,
            "t2": t2_src,
        },
        "amplitude_files_copied": copied_files,
        "can_read_amplitudes": can_read_amplitudes,
        "copy_errors": copy_errors,
        "movecs_file": movecs_line,
        "warnings": copy_errors,
    }


# ---------------------------------------------------------------------------
# Input variant with diff tracking
# ---------------------------------------------------------------------------


__all__ = [
    "draft_nwchem_tce_input",
    "validate_nwchem_tce_setup",
    "draft_nwchem_atom_input",
    "draft_nwchem_tce_restart_input",
]

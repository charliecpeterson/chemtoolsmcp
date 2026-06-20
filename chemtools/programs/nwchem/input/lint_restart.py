"""NWChem input lint + restart asset discovery.

Combines two closely-related responsibilities into one module:

  * inspect_input            Light wrapper over inspect_nwchem_input
                              that smooths the MCP surface.
  * lint_nwchem_input        Comprehensive linter — checks task lines,
                              basis blocks, charge/multiplicity, ECP
                              references, fragment guess sanity.
  * find_restart_assets      Scan a directory for .out / .err / .movecs /
                              .db / .hess / .nw files matching a stem
                              pattern; used by recovery and restart
                              workflows to locate prior-run artifacts.
  * _lint_fragment_guess     Internal helper for lint_nwchem_input
                              (fragment-state consistency check).
  * _normalize_stem_for_match, _stem_tokens
                              Stem-matching helpers used by find_restart_assets.

Both lint and restart-asset discovery touch the same set of underlying
parsers (basis, ECP, geometry, module blocks) which is why they share
a module here.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Any


def _normalize_stem_for_match(stem: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def _stem_tokens(stem: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", stem.lower()) if token]

from chemtools.programs.nwchem.parse.input import (
    inspect_nwchem_input,
    inspect_all_nwchem_basis_blocks,
    inspect_nwchem_ecp_block,
    inspect_nwchem_module_vectors,
    extract_nwchem_geometry_block,
    parse_start_blocks,
)
from chemtools.programs.nwchem.input._utils import _TRANSITION_METALS
from chemtools.programs.nwchem.input.basis_library import (
    resolve_mixed_basis_assignments,
    resolve_mixed_ecp_assignments,
)


def inspect_input(input_path: str) -> dict[str, Any]:
    return inspect_nwchem_input(input_path)



def _lint_fragment_guess(
    path: str,
    add_issue: Any,
) -> None:
    """Check that fragment Nα/Nβ sums match the molecular Nα/Nβ for every
    'vectors input fragment' block found in the file."""
    blocks = parse_start_blocks(path)

    if not any(b["fragment_inputs"] for b in blocks):
        return  # no fragment guess in this file

    # Build lookup: vectors_output filename → block (the producing start block).
    output_map: dict[str, dict[str, Any]] = {
        b["vectors_output"]: b for b in blocks if b["vectors_output"]
    }
    # Every output filename written anywhere in the file, including fragments
    # built inline within a single start block (vectors input atomic output X).
    all_outputs: set[str] = {
        out for b in blocks for out in (b.get("vectors_outputs") or [])
    }

    for mol in blocks:
        if not mol["fragment_inputs"]:
            continue

        if mol["multiplicity"] is None:
            add_issue(
                "warning",
                "fragment_mult_unknown",
                "A 'vectors input fragment' block is present but the molecular DFT "
                "multiplicity (mult N) is not set; cannot validate Nα/Nβ balance.",
            )
            continue

        missing_sources = [f for f in mol["fragment_inputs"] if f not in output_map]
        if missing_sources:
            # Fragments produced inline within a single start block (one `start`,
            # several `set geometry` + dft sections each with `vectors input
            # atomic output X`) exist but can't be mapped to a per-fragment block
            # for Nα/Nβ validation. That's a valid style, not an error — flag it
            # as info rather than warn that the sources are missing.
            inline = [f for f in missing_sources if f in all_outputs]
            if inline and len(inline) == len(missing_sources):
                add_issue(
                    "info",
                    "fragment_sources_inline",
                    "Fragment movecs files are produced inline within a single start "
                    "block; per-fragment Nα/Nβ balance is not validated for this style. "
                    "Use a separate start block per fragment to enable that check.",
                    {"inline": inline},
                )
                continue
            add_issue(
                "warning",
                "fragment_source_not_found",
                "Some fragment movecs files are not produced by a 'vectors output' "
                "in any start block in this file; Nα/Nβ balance cannot be checked.",
                {"missing": missing_sources},
            )
            continue

        mol_electrons = (
            sum(ELEMENT_TO_Z.get(e.capitalize(), 0) for e in mol["elements"])
            - mol["charge"]
        )
        mol_mult = mol["multiplicity"]
        mol_nalpha = (mol_electrons + (mol_mult - 1)) // 2
        mol_nbeta = mol_electrons - mol_nalpha

        frag_nalpha_sum = 0
        frag_nbeta_sum = 0
        incomplete = False
        for frag_file in mol["fragment_inputs"]:
            fb = output_map[frag_file]
            if fb["multiplicity"] is None:
                add_issue(
                    "warning",
                    "fragment_mult_unknown",
                    f"Fragment block producing '{frag_file}' has no multiplicity set; "
                    "cannot validate Nα/Nβ balance.",
                )
                incomplete = True
                break
            frag_electrons = (
                sum(ELEMENT_TO_Z.get(e.capitalize(), 0) for e in fb["elements"])
                - fb["charge"]
            )
            frag_mult = fb["multiplicity"]
            frag_nalpha = (frag_electrons + (frag_mult - 1)) // 2
            frag_nbeta = frag_electrons - frag_nalpha
            frag_nalpha_sum += frag_nalpha
            frag_nbeta_sum += frag_nbeta

        if incomplete:
            continue

        if frag_nalpha_sum == mol_nalpha and frag_nbeta_sum == mol_nbeta:
            add_issue(
                "info",
                "fragment_electron_balance_ok",
                f"Fragment Nα/Nβ sums ({frag_nalpha_sum}/{frag_nbeta_sum}) match "
                f"the molecular Nα/Nβ ({mol_nalpha}/{mol_nbeta}). "
                "Fragment guess electron counts are consistent.",
            )
        else:
            add_issue(
                "error",
                "fragment_electron_mismatch",
                f"Fragment Nα/Nβ sums ({frag_nalpha_sum}/{frag_nbeta_sum}) do not "
                f"match the molecular Nα/Nβ ({mol_nalpha}/{mol_nbeta}). "
                "NWChem will abort with 'movecs_fragment: open shell mismatch'. "
                "Adjust fragment multiplicities so their Nα and Nβ sum exactly to "
                "the molecular values.",
                {
                    "molecular": {
                        "nalpha": mol_nalpha,
                        "nbeta": mol_nbeta,
                        "mult": mol_mult,
                        "electrons": mol_electrons,
                    },
                    "fragments": {
                        "nalpha_sum": frag_nalpha_sum,
                        "nbeta_sum": frag_nbeta_sum,
                        "files": mol["fragment_inputs"],
                    },
                },
            )


def lint_nwchem_input(
    input_path: str,
    library_path: str | None = None,
) -> dict[str, Any]:
    input_summary = inspect_nwchem_input(input_path)
    issues: list[dict[str, Any]] = []

    def add_issue(level: str, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        payload = {"level": level, "code": code, "message": message}
        if details:
            payload["details"] = details
        issues.append(payload)

    if not input_summary["tasks"]:
        add_issue("error", "missing_tasks", "No task lines were found in the input.")
    if not input_summary["start_present"]:
        add_issue("warning", "missing_start", "No explicit start line was found in the input.")
    if input_summary["charge"] is None:
        add_issue("info", "charge_not_set", "Charge is not explicitly set; NWChem will assume the default.")
    if input_summary["multiplicity"] is None:
        add_issue("info", "multiplicity_not_set", "Multiplicity is not explicitly set in the input.")

    all_basis_blocks = inspect_all_nwchem_basis_blocks(input_path)
    basis_block = all_basis_blocks[0] if all_basis_blocks else None

    if not all_basis_blocks:
        add_issue("warning", "missing_basis_block", "No explicit basis block was found in the input.")
    else:
        for blk in all_basis_blocks:
            block_idx = blk["block_index"]
            blk_details_base: dict[str, Any] = {"block_index": block_idx}
            if blk["has_manual_content"] and not blk["has_library_lines"]:
                add_issue(
                    "info",
                    "manual_basis_content",
                    "Basis block contains manual basis data; library validation was skipped.",
                    {**blk_details_base, "elements": blk["explicit_elements"]},
                )
            elif blk["has_library_lines"]:
                if library_path:
                    resolved_basis = resolve_mixed_basis_assignments(
                        assignments=blk["library_assignments"],
                        elements=input_summary["elements"],
                        library_path=library_path,
                        default_basis=blk["default_library"],
                    )
                    if resolved_basis["missing_assignments"]:
                        add_issue(
                            "error",
                            "basis_assignment_missing",
                            "Some geometry elements do not have basis assignments.",
                            {**blk_details_base, "elements": resolved_basis["missing_assignments"]},
                        )
                    if resolved_basis["missing_coverage"]:
                        add_issue(
                            "error",
                            "basis_library_missing_coverage",
                            "The chosen basis library entries do not cover all assigned elements.",
                            {**blk_details_base, "elements": resolved_basis["missing_coverage"]},
                        )
                    if resolved_basis["all_elements_covered"]:
                        add_issue(
                            "info",
                            "basis_validated",
                            "Basis assignments were validated against the local basis library.",
                            blk_details_base,
                        )
                else:
                    add_issue(
                        "info",
                        "basis_library_not_checked",
                        "Basis block uses library entries, but no library path was provided for validation.",
                        blk_details_base,
                    )

    try:
        ecp_block = inspect_nwchem_ecp_block(input_path)
    except ValueError:
        ecp_block = None
    else:
        if ecp_block["has_manual_content"] and not ecp_block["has_library_lines"]:
            add_issue(
                "info",
                "manual_ecp_content",
                "ECP block contains manual ECP data; library validation was skipped.",
                {"elements": ecp_block["explicit_elements"]},
            )
        elif ecp_block["has_library_lines"]:
            if library_path:
                resolved_ecp = resolve_mixed_ecp_assignments(
                    assignments=ecp_block["library_assignments"],
                    elements=input_summary["elements"],
                    library_path=library_path,
                    default_ecp=ecp_block["default_library"],
                )
                if resolved_ecp["missing_coverage"]:
                    add_issue(
                        "error",
                        "ecp_library_missing_coverage",
                        "The chosen ECP library entries do not cover all assigned elements.",
                        {"elements": resolved_ecp["missing_coverage"]},
                    )
                if resolved_ecp["elements_with_ecp"]:
                    add_issue(
                        "info",
                        "ecp_validated",
                        "ECP assignments were validated against the local basis library.",
                        {"elements": resolved_ecp["elements_with_ecp"]},
                    )
            else:
                add_issue(
                    "info",
                    "ecp_library_not_checked",
                    "ECP block uses library entries, but no library path was provided for validation.",
                )

    if basis_block and basis_block["has_library_lines"] and not ecp_block:
        assigned_families = set(basis_block["library_assignments"].values())
        if basis_block["default_library"]:
            assigned_families.add(basis_block["default_library"])
        if any(("ecp" in family.lower()) or family.lower().endswith("-pp") for family in assigned_families):
            add_issue(
                "warning",
                "possible_missing_ecp_block",
                "Basis assignments look pseudopotential-based, but no explicit ECP block was found.",
            )

    task_modules = []
    seen_modules: set[str] = set()
    for task in input_summary["tasks"]:
        module_name = (task.get("module") or "").lower()
        operation_name = (task.get("operation") or "").lower()

        if module_name in {"optimize", "frequency", "freq", "energy", "property", "gradient", "hessian", "raman"} and not operation_name:
            suggested_module = "dft" if any(
                block_name in {module_name, "dft"}
                for block_name in [module_name, "dft"]
            ) else "dft"
            suggested_operation = "freq" if module_name in {"frequency", "freq"} else module_name
            add_issue(
                "error",
                "invalid_task_syntax",
                f"Task line 'task {module_name}' is not valid NWChem syntax for this workflow.",
                {
                    "task_module": module_name,
                    "suggested_task_line": f"task {suggested_module} {suggested_operation}",
                },
            )
            continue

        if module_name and module_name not in seen_modules:
            seen_modules.add(module_name)
            task_modules.append(module_name)

    for module_name in task_modules:
        try:
            module_vectors = inspect_nwchem_module_vectors(input_path, module=module_name)
        except ValueError:
            add_issue(
                "error",
                "missing_module_block",
                f"Task module '{module_name}' is referenced, but no matching module block was found.",
                {"module": module_name},
            )
            continue

        if module_name in {"scf", "dft"} and not module_vectors["has_vectors_output"]:
            add_issue(
                "warning",
                "missing_vectors_output",
                f"Module '{module_name}' does not explicitly write a movecs file.",
                {"module": module_name},
            )

    _lint_fragment_guess(input_path, add_issue)

    # --- Relativistic + ECP conflict check, and relativistic + SP-shell incompatibility ---
    try:
        import re as _re2
        from chemtools.core.common import read_text as _rt2
        _rc = _rt2(input_path)
        _has_rel = bool(_re2.search(r"^\s*relativistic\b", _rc, _re2.IGNORECASE | _re2.MULTILINE))
        _has_ecp = bool(_re2.search(r"^\s*ecp\b", _rc, _re2.IGNORECASE | _re2.MULTILINE))
        if _has_rel and _has_ecp:
            add_issue(
                "error",
                "relativistic_ecp_conflict",
                "Both a 'relativistic' block and an 'ecp' block are present. "
                "X2C and DKH are all-electron methods — they are incompatible with ECPs. "
                "Choose one: (a) all-electron basis + relativistic block, OR "
                "(b) ECP basis (no relativistic block needed — ECP implicitly encodes scalar relativistic effects).",
            )
        if _has_rel:
            # SP-contracted shells (Pople style) are incompatible with X2C/DKH.
            # NWChem builds an uncontracted auxiliary basis for the relativistic
            # one-electron operator; SP shells cause a dimension mismatch and
            # crash with "dimensions not the same" / MPI_Abort.
            _sp_elements = sorted({
                m.group(1)
                for m in _re2.finditer(
                    r"^\s*([A-Za-z][a-z]?)\s+SP\s*$", _rc, _re2.MULTILINE
                )
            })
            if _sp_elements:
                add_issue(
                    "error",
                    "relativistic_sp_shell_incompatibility",
                    f"SP-contracted basis shells detected for element(s) {_sp_elements} "
                    "while a relativistic block (X2C or DKH) is present. "
                    "NWChem X2C/DKH builds an uncontracted auxiliary basis internally; "
                    "Pople-style SP shells (6-31G*, 6-311G**, etc.) cause a 'dimensions not the same' "
                    "crash during this step. "
                    "Fix: replace the Pople basis with a Dunning basis (cc-pVDZ, cc-pVTZ, etc.) "
                    "or a def2 basis (def2-SVP, def2-TZVP) — both use separate S and P contractions "
                    "and are fully compatible with X2C/DKH.",
                )
    except Exception:
        pass

    # --- TCE-specific checks ---
    tce_tasks = [t for t in input_summary["tasks"] if (t.get("module") or "").lower() == "tce"]
    if tce_tasks:
        from chemtools.programs.nwchem.parse.input import extract_nwchem_geometry_block
        from chemtools.core.common import read_text as _read_text
        import re as _re

        # NWChem TCE accepts any of these Abelian point groups
        _TCE_ABELIAN = {"c1", "ci", "cs", "c2", "c2v", "c2h", "d2", "d2h"}

        # Check that geometry specifies an Abelian symmetry group
        try:
            geo = extract_nwchem_geometry_block(input_path)
            directives = [d.strip().lower() for d in (geo.get("directives") or [])]
            sym_group = None
            for d in directives:
                parts = d.split()
                if parts and parts[0] == "symmetry" and len(parts) > 1:
                    sym_group = parts[1]
                    break
            if sym_group not in _TCE_ABELIAN:
                if sym_group is None:
                    msg = (
                        "TCE requires Abelian symmetry. Add 'symmetry c1' (or d2h, c2v, etc.) "
                        "as a line inside the geometry block. "
                        "NWChem will abort with 'non-Abelian symmetry not permitted' otherwise."
                    )
                else:
                    msg = (
                        f"TCE requires Abelian symmetry; '{sym_group}' is non-Abelian. "
                        f"Use one of: {', '.join(sorted(_TCE_ABELIAN))}."
                    )
                add_issue("error", "tce_missing_symmetry_c1", msg)
        except Exception:
            pass

        # Check for symmetry placed on the geometry header line (wrong syntax)
        _lint_contents = _read_text(input_path)
        for _line in _lint_contents.splitlines():
            if _re.match(r"^\s*geometry\b.*\bsymmetry\b", _line, _re.IGNORECASE):
                add_issue(
                    "error",
                    "symmetry_on_geometry_header",
                    "'symmetry' must appear as its own line inside the geometry block, not on the "
                    "'geometry ...' header line. Correct form:\n"
                    "  geometry units angstrom\n"
                    "    symmetry c1\n"
                    "    ...\n"
                    "  end",
                )
                break

    # --- autoz + symmetric TM complex warning ---
    _tm_elements_in_input = {e for e in (input_summary.get("elements") or []) if e in _TRANSITION_METALS}
    if _tm_elements_in_input:
        _has_optimize_task = any(
            (t.get("operation") or "").lower() == "optimize" for t in (input_summary.get("tasks") or [])
        )
        if _has_optimize_task:
            _rc_full = open(input_path, encoding="utf-8", errors="replace").read()
            _has_driver = bool(_re2.search(r"^\s*driver\b", _rc_full, _re2.IGNORECASE | _re2.MULTILINE))
            _has_xyz = bool(_re2.search(r"^\s*xyz\b", _rc_full, _re2.IGNORECASE | _re2.MULTILINE))
            if not _has_driver or not _has_xyz:
                _n_heavy_in_input = sum(
                    1 for e in (input_summary.get("elements") or []) if e not in {"H", "D"}
                )
                if _n_heavy_in_input >= 4:
                    add_issue(
                        "warning",
                        "autoz_symmetric_tm_complex",
                        "Optimization of a TM complex without explicit 'driver; xyz; end' may produce "
                        "degenerate Z-matrix coordinates for symmetric geometries (e.g. octahedral, "
                        "tetrahedral), causing the optimizer to walk uphill. "
                        "Add 'driver\\n  xyz\\n  maxiter 300\\nend' before the DFT/SCF block.",
                    )

    # --- Memory directive consistency check ---
    try:
        _rc2 = open(input_path, encoding="utf-8", errors="replace").read()
        _mem_m = _re2.search(
            r"^\s*memory\s+total\s+(\d+)\s*mb\b.*stack\s+(\d+)\s*mb\b.*heap\s+(\d+)\s*mb\b.*global\s+(\d+)\s*mb\b",
            _rc2, _re2.IGNORECASE | _re2.MULTILINE,
        )
        if _mem_m:
            _total, _stack, _heap, _glob = (int(_mem_m.group(i)) for i in range(1, 5))
            if _stack + _heap + _glob > _total:
                add_issue(
                    "error",
                    "memory_subcomponents_exceed_total",
                    f"Memory sub-components (stack {_stack} + heap {_heap} + global {_glob} = "
                    f"{_stack + _heap + _glob} MB) exceed the declared total of {_total} MB. "
                    "NWChem will abort on startup with 'Memory_Defaults: Inconsistent memory specification'. "
                    f"Set total to at least {_stack + _heap + _glob} MB.",
                )
    except OSError:
        pass

    # --- In-core SCF deadlock risk: noio + grid nodisk ---
    # Forcing everything in core (`noio` keeps integrals/Fock in RAM, `grid nodisk`
    # keeps the XC grid in RAM) deadlocks mid-SCF when the per-rank allocation is
    # too small to hold them — the run pins 100% CPU and never advances. Seen on
    # lanthanide/actinide complexes whose inputs were written for larger nodes.
    try:
        _rc3 = open(input_path, encoding="utf-8", errors="replace").read()
        _has_noio = bool(_re2.search(r"^\s*noio\b", _rc3, _re2.IGNORECASE | _re2.MULTILINE))
        _has_grid_nodisk = bool(
            _re2.search(r"^\s*grid\b[^\n]*\bnodisk\b", _rc3, _re2.IGNORECASE | _re2.MULTILINE)
        )
        if _has_noio and _has_grid_nodisk:
            _natoms = input_summary.get("atom_count") or len(input_summary.get("elements") or [])
            add_issue(
                "warning",
                "incore_scf_deadlock_risk",
                f"Both 'noio' and 'grid nodisk' force the whole SCF in core ({_natoms} atoms). "
                "If the per-rank memory is too small to hold the integrals and XC grid, NWChem "
                "deadlocks mid-SCF (100% CPU, no progress) rather than spilling to disk. "
                "For large systems on a memory-tight host, drop 'noio' and 'grid nodisk' so it "
                "uses disk scratch, and raise the 'memory' directive.",
                {"atom_count": _natoms},
            )
    except OSError:
        pass

    severity_order = {"error": 3, "warning": 2, "info": 1}
    highest = max((severity_order[item["level"]] for item in issues), default=0)
    status = "ok"
    if highest >= 3:
        status = "error"
    elif highest == 2:
        status = "warning"

    return {
        "input_file": input_path,
        "library_path": library_path,
        "status": status,
        "issue_count": len(issues),
        "issues": issues,
        "counts": {
            "error": sum(1 for item in issues if item["level"] == "error"),
            "warning": sum(1 for item in issues if item["level"] == "warning"),
            "info": sum(1 for item in issues if item["level"] == "info"),
        },
        "input_summary": input_summary,
        "basis_block": basis_block,
        "ecp_block": ecp_block,
    }



def find_restart_assets(path: str) -> dict[str, Any]:
    target = Path(path).resolve()
    job_dir = target if target.is_dir() else target.parent
    focus_stem = None if target.is_dir() else target.stem

    relevant_suffixes = {
        ".nw": "inputs",
        ".out": "outputs",
        ".err": "errors",
        ".movecs": "movecs",
        ".db": "databases",
        ".xyz": "xyz",
        ".zmat": "zmat",
        ".cube": "cubes",
        ".nmode": "nmodes",
        ".normal": "normal_modes",
        ".hess": "hessians",
    }
    collections: dict[str, list[str]] = {label: [] for label in relevant_suffixes.values()}

    for child in sorted(job_dir.iterdir()):
        if not child.is_file():
            continue
        suffix = child.suffix.lower()
        label = relevant_suffixes.get(suffix)
        if label:
            collections[label].append(str(child.resolve()))

    related_files = sorted(
        str(child.resolve())
        for child in job_dir.iterdir()
        if child.is_file() and (focus_stem is None or child.name.startswith(focus_stem))
    )

    def choose_exact(suffix: str) -> str | None:
        if focus_stem is None:
            return None
        candidate = job_dir / f"{focus_stem}{suffix}"
        return str(candidate.resolve()) if candidate.exists() else None

    def newest(label: str) -> str | None:
        files = [Path(item) for item in collections[label]]
        if not files:
            return None
        return str(max(files, key=lambda candidate: candidate.stat().st_mtime).resolve())

    def best_related(label: str) -> str | None:
        files = [Path(item) for item in collections[label]]
        if not files:
            return None
        if focus_stem is None:
            return str(max(files, key=lambda candidate: candidate.stat().st_mtime).resolve())

        exact = choose_exact(files[0].suffix.lower())
        if exact:
            return exact

        normalized_focus = _normalize_stem_for_match(focus_stem)
        focus_tokens = set(_stem_tokens(focus_stem))
        scored: list[tuple[tuple[int, int, int, float], Path]] = []
        for candidate in files:
            stem = candidate.stem
            normalized = _normalize_stem_for_match(stem)
            candidate_tokens = set(_stem_tokens(stem))
            score = (
                1 if normalized == normalized_focus else 0,
                len(focus_tokens & candidate_tokens),
                1 if focus_tokens and (focus_tokens <= candidate_tokens or candidate_tokens <= focus_tokens) else 0,
                candidate.stat().st_mtime,
            )
            scored.append((score, candidate))

        best_score, best_path = max(scored, key=lambda item: item[0])
        if best_score[1] > 0 or best_score[0] > 0:
            return str(best_path.resolve())
        return str(max(files, key=lambda candidate: candidate.stat().st_mtime).resolve())

    preferred = {
        "input_file": choose_exact(".nw") or best_related("inputs"),
        "output_file": choose_exact(".out") or best_related("outputs"),
        "error_file": choose_exact(".err") or best_related("errors"),
        "vectors_file": choose_exact(".movecs") or best_related("movecs"),
        "database_file": choose_exact(".db") or best_related("databases"),
        "xyz_file": choose_exact(".xyz") or best_related("xyz"),
        "zmat_file": choose_exact(".zmat") or best_related("zmat"),
    }

    restart_candidates: list[dict[str, Any]] = []
    for key, label in (
        ("vectors_file", "movecs"),
        ("database_file", "database"),
        ("xyz_file", "xyz"),
        ("input_file", "input"),
    ):
        if preferred[key]:
            restart_candidates.append({"kind": label, "path": preferred[key]})

    return {
        "query_path": str(target),
        "job_dir": str(job_dir),
        "focus_stem": focus_stem,
        "preferred": preferred,
        "collections": collections,
        "related_files": related_files,
        "restart_candidates": restart_candidates,
    }




# TCE drafters moved to programs/nwchem/input/tce.py.
from chemtools.programs.nwchem.input.tce import (  # noqa: F401
    draft_nwchem_tce_input,
    validate_nwchem_tce_setup,
    draft_nwchem_atom_input,
    draft_nwchem_tce_restart_input,
)



__all__ = [
    "inspect_input",
    "lint_nwchem_input",
    "find_restart_assets",
]

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chemtools.api import diagnose_output, prepare_nwchem_next_step


def discover_case_files(path: str) -> list[str]:
    target = Path(path).resolve()
    if target.is_file():
        if not (target.name == "case.json" or target.name.endswith(".case.json")):
            raise ValueError("eval input file must be a case.json or *.case.json file")
        return [str(target)]
    if not target.is_dir():
        raise ValueError(f"case path does not exist: {path}")
    case_files = list(target.rglob("case.json")) + list(target.rglob("*.case.json"))
    unique = sorted({str(case_file.resolve()) for case_file in case_files})
    return unique


def load_case(path: str) -> dict[str, Any]:
    case_path = Path(path).resolve()
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    payload["__case_path__"] = str(case_path)
    payload["__case_dir__"] = str(case_path.parent)
    return payload


def evaluate_case(path: str) -> dict[str, Any]:
    """Evaluate a single case file, dispatching by the 'program' field."""
    case = load_case(path)
    program = case.get("program", "nwchem").lower()
    if program == "molcas":
        return _evaluate_molcas_case(case)
    if program == "dirac":
        return _evaluate_dirac_case(case)
    if program == "grasp":
        return _evaluate_grasp_case(case)
    # Default: NWChem evaluator
    return _evaluate_nwchem_case(case)


def evaluate_cases(path: str) -> dict[str, Any]:
    case_files = discover_case_files(path)
    results = [evaluate_case(case_file) for case_file in case_files]
    return {
        "root": str(Path(path).resolve()),
        "case_count": len(results),
        "passed_case_count": sum(1 for result in results if result["passed"]),
        "failed_case_count": sum(1 for result in results if not result["passed"]),
        "results": results,
    }


# ---------------------------------------------------------------------------
# NWChem evaluator (original)
# ---------------------------------------------------------------------------

def _evaluate_nwchem_case(case: dict[str, Any]) -> dict[str, Any]:
    case_dir = Path(case["__case_dir__"])
    files = case["files"]
    input_path = _resolve_case_file(case_dir, files.get("primary_input"), required=False)
    output_path = _resolve_case_file(case_dir, files["primary_output"], required=True)

    diagnosis = diagnose_output(output_path=output_path, input_path=input_path)
    workflow = prepare_nwchem_next_step(output_path=output_path, input_path=input_path)
    expectations = case.get("eval_expectations") or {}

    checks = [
        _make_check("diagnosis_failure_class",
                    expectations.get("diagnosis_failure_class"), diagnosis["failure_class"]),
        _make_check("diagnosis_stage",
                    expectations.get("diagnosis_stage"), diagnosis["stage"]),
        _make_check("recommended_next_action",
                    expectations.get("recommended_next_action"), diagnosis["recommended_next_action"]),
        _make_check("workflow",
                    expectations.get("workflow"), workflow["selected_workflow"]),
        _make_check("can_auto_prepare",
                    expectations.get("can_auto_prepare"), workflow["can_auto_prepare"]),
    ]

    active_checks = [c for c in checks if c["expected"] is not None]
    passed_checks = [c for c in active_checks if c["passed"]]
    failed_checks = [c for c in active_checks if not c["passed"]]

    return {
        "case_id": case["case_id"],
        "case_path": case["__case_path__"],
        "program": case["program"],
        "summary": case["summary"],
        "input_file": input_path,
        "output_file": output_path,
        "check_count": len(active_checks),
        "pass_count": len(passed_checks),
        "fail_count": len(failed_checks),
        "passed": not failed_checks,
        "checks": active_checks,
        "diagnosis": {
            "failure_class": diagnosis["failure_class"],
            "stage": diagnosis["stage"],
            "recommended_next_action": diagnosis["recommended_next_action"],
            "task_outcome": diagnosis["task_outcome"],
        },
        "workflow": {
            "selected_workflow": workflow["selected_workflow"],
            "can_auto_prepare": workflow["can_auto_prepare"],
            "notes": workflow["notes"],
        },
    }


# ---------------------------------------------------------------------------
# Molcas evaluator
# ---------------------------------------------------------------------------
# Molcas eval_expectations fields:
#   primary_energy_au        — float: check abs(actual - expected) < tolerance
#   primary_energy_tolerance — float: tolerance in Ha (default 1e-4)
#   modules_run              — list[str]: modules that must appear in tasks_overview
#   verdict                  — str: 'healthy'|'caution'|'problematic' from analyze_molcas_case
#   converged                — bool: whether the primary run converged

def _evaluate_molcas_case(case: dict[str, Any]) -> dict[str, Any]:
    from chemtools.programs.molcas.parse.output import parse_output_full

    case_dir = Path(case["__case_dir__"])
    files = case["files"]
    output_path = _resolve_case_file(case_dir, files["primary_output"], required=True)
    expectations = case.get("eval_expectations") or {}

    contents = Path(output_path).read_text(encoding="utf-8", errors="replace")
    parsed = parse_output_full(output_path, contents)
    energy_summary = parsed.get("energy_summary") or {}
    tasks_overview = parsed.get("tasks_overview") or []
    actual_modules = [t.get("extra", {}).get("module", "") for t in tasks_overview]
    primary_energy = energy_summary.get("primary_energy_hartree")

    checks: list[dict[str, Any]] = []

    # Energy check (approximate)
    if expectations.get("primary_energy_au") is not None:
        tol = expectations.get("primary_energy_tolerance", 1e-4)
        exp_e = float(expectations["primary_energy_au"])
        actual_e = primary_energy
        if actual_e is None:
            passed = False
        else:
            passed = abs(actual_e - exp_e) < tol
        checks.append({
            "name": "primary_energy_au",
            "expected": exp_e,
            "actual": actual_e,
            "passed": passed,
            "tolerance": tol,
        })

    # Modules check
    if expectations.get("modules_run"):
        for mod in expectations["modules_run"]:
            present = any(mod.upper() in m.upper() for m in actual_modules)
            checks.append(_make_check(f"module_{mod}", True, present))

    # Converged check (primary energy is not None as proxy)
    if expectations.get("converged") is not None:
        actual_converged = primary_energy is not None
        checks.append(_make_check("converged", expectations["converged"], actual_converged))

    # Verdict check (requires analyze_molcas_case)
    if expectations.get("verdict"):
        try:
            from chemtools.programs.molcas.strategy.orchestrators import analyze_molcas_case
            analysis = analyze_molcas_case(output_file=output_path)
            actual_verdict = analysis.get("verdict")
        except Exception:
            actual_verdict = None
        checks.append(_make_check("verdict", expectations["verdict"], actual_verdict))

    active_checks = [c for c in checks if c.get("expected") is not None]
    passed_checks = [c for c in active_checks if c["passed"]]
    failed_checks = [c for c in active_checks if not c["passed"]]

    return {
        "case_id": case["case_id"],
        "case_path": case["__case_path__"],
        "program": "molcas",
        "summary": case["summary"],
        "input_file": None,
        "output_file": output_path,
        "check_count": len(active_checks),
        "pass_count": len(passed_checks),
        "fail_count": len(failed_checks),
        "passed": not failed_checks,
        "checks": active_checks,
        "parsed": {
            "primary_energy_hartree": primary_energy,
            "primary_label": energy_summary.get("primary_label"),
            "modules_run": actual_modules,
        },
    }


# ---------------------------------------------------------------------------
# DIRAC evaluator
# ---------------------------------------------------------------------------
# DIRAC eval_expectations fields:
#   scf_energy_au            — float: expected SCF total energy (abs match within tolerance)
#   scf_energy_tolerance     — float: tolerance in Ha (default 1e-4)
#   converged                — bool: whether SCF converged
#   n_occupied_spinors       — int: expected occupied spinor count
#   n_cosci_states           — int: expected number of COSCI states

def _evaluate_dirac_case(case: dict[str, Any]) -> dict[str, Any]:
    from chemtools.programs.dirac.parse.output import parse_output

    case_dir = Path(case["__case_dir__"])
    files = case["files"]
    output_path = _resolve_case_file(case_dir, files["primary_output"], required=True)
    expectations = case.get("eval_expectations") or {}

    contents = Path(output_path).read_text(encoding="utf-8", errors="replace")
    parsed = parse_output(output_path, contents)
    actual_energy = parsed.get("total_energy_hartree")
    actual_n_occ = parsed.get("n_occupied_spinors")
    cosci = parsed.get("cosci")

    checks: list[dict[str, Any]] = []

    # SCF energy check
    if expectations.get("scf_energy_au") is not None:
        tol = expectations.get("scf_energy_tolerance", 1e-4)
        exp_e = float(expectations["scf_energy_au"])
        passed = (actual_energy is not None and abs(actual_energy - exp_e) < tol)
        checks.append({
            "name": "scf_energy_au",
            "expected": exp_e,
            "actual": actual_energy,
            "passed": passed,
            "tolerance": tol,
        })

    # Converged check (energy present as proxy)
    if expectations.get("converged") is not None:
        checks.append(_make_check("converged", expectations["converged"], actual_energy is not None))

    # Occupied spinor count
    if expectations.get("n_occupied_spinors") is not None:
        checks.append(_make_check("n_occupied_spinors",
                                  expectations["n_occupied_spinors"], actual_n_occ))

    # COSCI state count
    if expectations.get("n_cosci_states") is not None:
        actual_n_cosci = cosci["n_states"] if cosci else 0
        checks.append(_make_check("n_cosci_states",
                                  expectations["n_cosci_states"], actual_n_cosci))

    active_checks = [c for c in checks if c.get("expected") is not None]
    passed_checks = [c for c in active_checks if c["passed"]]
    failed_checks = [c for c in active_checks if not c["passed"]]

    return {
        "case_id": case["case_id"],
        "case_path": case["__case_path__"],
        "program": "dirac",
        "summary": case["summary"],
        "input_file": None,
        "output_file": output_path,
        "check_count": len(active_checks),
        "pass_count": len(passed_checks),
        "fail_count": len(failed_checks),
        "passed": not failed_checks,
        "checks": active_checks,
        "parsed": {
            "scf_energy_hartree": actual_energy,
            "n_occupied_spinors": actual_n_occ,
            "n_cosci_states": cosci["n_states"] if cosci else None,
        },
    }


# ---------------------------------------------------------------------------
# GRASP evaluator
# ---------------------------------------------------------------------------
# GRASP eval_expectations fields:
#   ground_energy_au         — float: expected ground-state energy (abs match)
#   ground_energy_tolerance  — float: tolerance in Ha (default 1e-4)
#   speed_of_light_au        — float: expected c (137.036 default, 2000+ = non-rel)
#   speed_of_light_tolerance — float: tolerance (default 0.1)
#   is_nonrel_limit          — bool: whether this is a non-rel-limit run
#   n_subshells              — int: number of relativistic subshells
#   n_levels                 — int: levels in rlevels output (if file_kind=rlevels)
#   atomic_number            — float: Z
#   file_kind                — str: 'rmcdhf_summary' | 'rlevels' | 'lsj_label'

def _evaluate_grasp_case(case: dict[str, Any]) -> dict[str, Any]:
    from chemtools.programs.grasp._plugin_parser import GRASP_PARSER

    case_dir = Path(case["__case_dir__"])
    files = case["files"]
    output_path = _resolve_case_file(case_dir, files["primary_output"], required=True)
    expectations = case.get("eval_expectations") or {}

    parsed_run = GRASP_PARSER.parse_output(output_path)
    derived = parsed_run.get("derived", {})
    file_kind = derived.get("grasp:file_kind")

    checks: list[dict[str, Any]] = []

    # Ground-state energy check (works for rmcdhf_summary and rlevels kinds)
    if expectations.get("ground_energy_au") is not None:
        tol = expectations.get("ground_energy_tolerance", 1e-4)
        exp_e = float(expectations["ground_energy_au"])
        actual_e = derived.get("final_energy_hartree")
        passed = (actual_e is not None and abs(actual_e - exp_e) < tol)
        checks.append({
            "name": "ground_energy_au",
            "expected": exp_e,
            "actual": actual_e,
            "passed": passed,
            "tolerance": tol,
        })

    # Speed-of-light check (catches non-rel-limit runs)
    if expectations.get("speed_of_light_au") is not None:
        tol = expectations.get("speed_of_light_tolerance", 0.1)
        exp_c = float(expectations["speed_of_light_au"])
        actual_c = derived.get("grasp:speed_of_light_au")
        passed = (actual_c is not None and abs(actual_c - exp_c) < tol)
        checks.append({
            "name": "speed_of_light_au",
            "expected": exp_c,
            "actual": actual_c,
            "passed": passed,
            "tolerance": tol,
        })

    if expectations.get("is_nonrel_limit") is not None:
        checks.append(_make_check("is_nonrel_limit",
                                  expectations["is_nonrel_limit"],
                                  derived.get("grasp:is_nonrel_limit")))

    if expectations.get("n_subshells") is not None:
        checks.append(_make_check("n_subshells",
                                  expectations["n_subshells"],
                                  derived.get("grasp:n_subshells")))

    if expectations.get("n_levels") is not None:
        checks.append(_make_check("n_levels",
                                  expectations["n_levels"],
                                  derived.get("grasp:n_levels")))

    if expectations.get("atomic_number") is not None:
        checks.append(_make_check("atomic_number",
                                  float(expectations["atomic_number"]),
                                  derived.get("grasp:atomic_number")))

    if expectations.get("file_kind") is not None:
        checks.append(_make_check("file_kind",
                                  expectations["file_kind"], file_kind))

    active_checks = [c for c in checks if c.get("expected") is not None]
    passed_checks = [c for c in active_checks if c["passed"]]
    failed_checks = [c for c in active_checks if not c["passed"]]

    return {
        "case_id": case["case_id"],
        "case_path": case["__case_path__"],
        "program": "grasp",
        "summary": case["summary"],
        "input_file": None,
        "output_file": output_path,
        "check_count": len(active_checks),
        "pass_count": len(passed_checks),
        "fail_count": len(failed_checks),
        "passed": not failed_checks,
        "checks": active_checks,
        "parsed": {
            "file_kind": file_kind,
            "ground_energy_hartree": derived.get("final_energy_hartree"),
            "speed_of_light_au": derived.get("grasp:speed_of_light_au"),
            "is_nonrel_limit": derived.get("grasp:is_nonrel_limit"),
            "n_subshells": derived.get("grasp:n_subshells"),
            "n_levels": derived.get("grasp:n_levels"),
            "atomic_number": derived.get("grasp:atomic_number"),
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_case_file(case_dir: Path, relative_path: str | None, required: bool) -> str | None:
    if not relative_path:
        return None
    resolved = (case_dir / relative_path).resolve()
    if resolved.exists():
        return str(resolved)
    if required:
        raise FileNotFoundError(f"case file does not exist: {resolved}")
    return None


def _make_check(name: str, expected: Any, actual: Any) -> dict[str, Any]:
    return {
        "name": name,
        "expected": expected,
        "actual": actual,
        "passed": expected == actual,
    }

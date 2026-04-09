from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .api import diagnose_output, prepare_nwchem_next_step


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
    case = load_case(path)
    case_dir = Path(case["__case_dir__"])
    files = case["files"]
    input_path = _resolve_case_file(case_dir, files.get("primary_input"), required=False)
    output_path = _resolve_case_file(case_dir, files["primary_output"], required=True)

    diagnosis = diagnose_output(output_path=output_path, input_path=input_path)
    workflow = prepare_nwchem_next_step(output_path=output_path, input_path=input_path)
    expectations = case.get("eval_expectations") or {}

    checks = [
        _make_check(
            "diagnosis_failure_class",
            expectations.get("diagnosis_failure_class"),
            diagnosis["failure_class"],
        ),
        _make_check(
            "diagnosis_stage",
            expectations.get("diagnosis_stage"),
            diagnosis["stage"],
        ),
        _make_check(
            "recommended_next_action",
            expectations.get("recommended_next_action"),
            diagnosis["recommended_next_action"],
        ),
        _make_check(
            "workflow",
            expectations.get("workflow"),
            workflow["selected_workflow"],
        ),
        _make_check(
            "can_auto_prepare",
            expectations.get("can_auto_prepare"),
            workflow["can_auto_prepare"],
        ),
    ]

    active_checks = [check for check in checks if check["expected"] is not None]
    passed_checks = [check for check in active_checks if check["passed"]]
    failed_checks = [check for check in active_checks if not check["passed"]]

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

"""Check the curated Open Babel conversion corpus through the fixed runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chemtools.integrations.science_runtime import (
    ScienceRuntimeClient,
    ScienceRuntimeCommandError,
    ScienceRuntimeUnavailableError,
)
from chemtools.science_runner import (
    OPENBABEL_CONVERSION_REQUEST_SCHEMA,
    OPENBABEL_CONVERSION_RESULT_SCHEMA,
)


CORPUS_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "openbabel"
    / "conversion_cases.json"
)
CORPUS_SCHEMA = "chemtools.openbabel-conversion-fixtures/1"
_EVIDENCE_FIELDS = {
    "canonical_smiles",
    "formula",
    "atom_count",
    "heavy_atom_count",
    "bond_count",
    "formal_charge",
    "radical_electrons",
    "fragment_count",
    "aromatic_atom_count",
    "stereocenter_count",
    "stereo_bond_count",
}


def load_openbabel_fixture_corpus(
    path: Path = CORPUS_PATH,
) -> dict[str, Any]:
    """Load and validate the committed conversion expectations."""
    corpus = json.loads(path.read_text(encoding="utf-8"))
    _validate_corpus(corpus)
    return corpus


def compare_openbabel_fixture_case(
    case: dict[str, Any],
    response: dict[str, Any],
    recorded_with: dict[str, str],
) -> list[str]:
    """Return exact semantic differences between one fixture and one response."""
    expected = case["expected"]
    differences = []
    if response.get("schema_version") != OPENBABEL_CONVERSION_RESULT_SCHEMA:
        differences.append("result schema differs")
    if response.get("status") != expected["status"]:
        differences.append("status differs")
        return differences

    converted = response.get("converted")
    comparison = response.get("comparison")
    provenance = response.get("provenance")
    if not isinstance(converted, dict):
        return [*differences, "converted artifact is missing"]
    if not isinstance(comparison, dict):
        return [*differences, "comparison evidence is missing"]
    if not isinstance(provenance, dict):
        return [*differences, "provenance is missing"]
    if provenance.get("openbabel_version") != recorded_with["openbabel_version"]:
        differences.append("Open Babel version differs")
    if provenance.get("rdkit_version") != recorded_with["rdkit_version"]:
        differences.append("RDKit version differs")
    if comparison.get("status") != expected["comparison_status"]:
        differences.append("comparison status differs")
    if converted.get("coordinate_status") != expected["coordinate_status"]:
        differences.append("coordinate status differs")
    converted_text = converted.get("text")
    if not isinstance(converted_text, str):
        differences.append("converted text is missing")
    elif any(text not in converted_text for text in expected["converted_text_contains"]):
        differences.append("converted text markers differ")
    if comparison.get("source_rdkit") != expected["source_rdkit"]:
        differences.append("source RDKit evidence differs")
    if comparison.get("converted_rdkit") != expected["converted_rdkit"]:
        differences.append("converted RDKit evidence differs")
    if sorted(comparison.get("differences", {})) != expected["difference_fields"]:
        differences.append("reported difference fields differ")
    warning_codes = [warning.get("code") for warning in response.get("warnings", [])]
    if warning_codes != expected["warning_codes"]:
        differences.append("warning codes differ")
    return differences


def run_openbabel_fixture_corpus(
    client: ScienceRuntimeClient,
    corpus: dict[str, Any],
) -> dict[str, Any]:
    """Run each fixed conversion and classify it against committed evidence."""
    records = []
    for case in corpus["cases"]:
        request = {
            "schema_version": OPENBABEL_CONVERSION_REQUEST_SCHEMA,
            **case["request"],
        }
        response = client.openbabel_convert(request)
        differences = compare_openbabel_fixture_case(
            case,
            response,
            corpus["recorded_with"],
        )
        records.append({
            "id": case["id"],
            "outcome": "agree" if not differences else "disagree",
            "differences": differences,
        })
    return {
        "schema_version": "chemtools.openbabel-conversion-check/1",
        "cases": records,
        "summary": {
            "agree": sum(record["outcome"] == "agree" for record in records),
            "disagree": sum(record["outcome"] == "disagree" for record in records),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the curated Open Babel conversion fixture corpus."
    )
    parser.add_argument(
        "--python",
        help="Explicit chemtools-science Python path; defaults to CHEMTOOLS_SCIENCE_PYTHON.",
    )
    arguments = parser.parse_args(argv)
    corpus = load_openbabel_fixture_corpus()
    try:
        report = run_openbabel_fixture_corpus(
            ScienceRuntimeClient(arguments.python),
            corpus,
        )
    except (ScienceRuntimeUnavailableError, ScienceRuntimeCommandError) as error:
        print(json.dumps({
            "schema_version": "chemtools.openbabel-conversion-check/1",
            "status": "tool_refused",
            "message": str(error),
        }, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0 if report["summary"]["disagree"] == 0 else 1


def _validate_corpus(corpus: Any) -> None:
    if not isinstance(corpus, dict) or set(corpus) != {
        "schema_version",
        "recorded_with",
        "cases",
    }:
        raise ValueError("Open Babel fixture corpus has an invalid top-level shape")
    if corpus["schema_version"] != CORPUS_SCHEMA:
        raise ValueError("Open Babel fixture corpus has an unsupported schema")
    recorded_with = corpus["recorded_with"]
    if not isinstance(recorded_with, dict) or set(recorded_with) != {
        "openbabel_version",
        "rdkit_version",
        "runner_result_schema",
    }:
        raise ValueError("Open Babel fixture corpus has invalid provenance")
    if recorded_with["runner_result_schema"] != OPENBABEL_CONVERSION_RESULT_SCHEMA:
        raise ValueError("Open Babel fixture corpus has an invalid runner schema")
    cases = corpus["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("Open Babel fixture corpus must contain cases")
    for case in cases:
        _validate_case(case)


def _validate_case(case: Any) -> None:
    if not isinstance(case, dict) or set(case) != {"id", "purpose", "request", "expected"}:
        raise ValueError("Open Babel fixture case has an invalid shape")
    if not isinstance(case["id"], str) or not case["id"]:
        raise ValueError("Open Babel fixture case id must be non-empty text")
    if not isinstance(case["purpose"], str) or not case["purpose"]:
        raise ValueError("Open Babel fixture case purpose must be non-empty text")
    request = case["request"]
    if not isinstance(request, dict) or set(request) != {
        "format",
        "source",
        "output_format",
    }:
        raise ValueError("Open Babel fixture request has an invalid shape")
    if request["format"] not in {"smiles", "molblock"}:
        raise ValueError("Open Babel fixture format is unsupported")
    if request["output_format"] not in {"smiles", "molblock"}:
        raise ValueError("Open Babel fixture output format is unsupported")
    if not isinstance(request["source"], str) or not request["source"].strip():
        raise ValueError("Open Babel fixture source must be non-empty text")
    expected = case["expected"]
    required_expected = {
        "status",
        "comparison_status",
        "coordinate_status",
        "converted_text_contains",
        "source_rdkit",
        "converted_rdkit",
        "difference_fields",
        "warning_codes",
    }
    if not isinstance(expected, dict) or set(expected) != required_expected:
        raise ValueError("Open Babel fixture expectation has an invalid shape")
    if expected["status"] != "completed":
        raise ValueError("Open Babel fixtures must record completed conversions")
    if expected["comparison_status"] not in {"matched", "different"}:
        raise ValueError("Open Babel fixture comparison status is invalid")
    if expected["coordinate_status"] not in {"not_generated", "not_applicable"}:
        raise ValueError("Open Babel fixture coordinate status is invalid")
    if not all(isinstance(item, str) and item for item in expected["converted_text_contains"]):
        raise ValueError("Open Babel fixture converted text markers are invalid")
    if not all(isinstance(item, str) and item for item in expected["difference_fields"]):
        raise ValueError("Open Babel fixture difference fields are invalid")
    if not all(isinstance(item, str) and item for item in expected["warning_codes"]):
        raise ValueError("Open Babel fixture warning codes are invalid")
    for evidence_name in ("source_rdkit", "converted_rdkit"):
        evidence = expected[evidence_name]
        if not isinstance(evidence, dict) or set(evidence) != _EVIDENCE_FIELDS:
            raise ValueError(f"Open Babel fixture {evidence_name} is invalid")

"""Application-level contracts for conservative run comparison."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import pytest

from chemtools.application.run_comparison import (
    compare_run_inspections,
    compare_runs,
)
from chemtools.core import registry
from chemtools.core.program import ProgramCapability
from chemtools.mcp.catalog import (
    BUILTIN_BACKENDS,
    load_backend,
    register_builtin_backends,
)
from chemtools.mcp.tools import guided
from chemtools.reference.external_corpus import (
    load_reference_manifest,
    verify_reference_case,
)


NWCHEM_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "chemtools"
    / "data"
    / "reference_cases"
    / "nwchem_behavior_cases.json"
)
MCP_FIXTURES = Path(__file__).parent / "golden" / "mcp" / "fixtures"


class _EnergyParser:
    def __init__(self, records):
        self.records = records

    def parse_output(self, path: str) -> dict:
        return self.records[Path(path).name]


def _parsed_run(energy: float, method: str = "DFT") -> dict:
    return {
        "program": "nwchem",
        "tasks": [{
            "index": 0,
            "kind": "energy",
            "name": "Single Point",
            "method": method,
            "basis": None,
            "energy_hartree": energy,
            "line_range": (1, 2),
            "outcome": "success",
            "has_usable_data": True,
            "selection_priority": 1,
        }],
        "derived": {"primary_energy_hartree": energy},
        "diagnostics": [],
    }


def _backend(records):
    backend = load_backend(BUILTIN_BACKENDS[0])
    return replace(
        backend,
        capabilities=backend.capabilities - {ProgramCapability.DIAGNOSIS_RUN},
        parser=_EnergyParser(records),
        diagnostics=None,
        consistency=None,
    )


def test_compare_runs_reports_conditional_lower_energy_without_inputs(tmp_path):
    reference = tmp_path / "reference.out"
    candidate = tmp_path / "candidate.out"
    reference.write_text("reference\n", encoding="utf-8")
    candidate.write_text("candidate\n", encoding="utf-8")

    compared = compare_runs(
        _backend({
            reference.name: _parsed_run(-10.0),
            candidate.name: _parsed_run(-10.1),
        }),
        reference,
        candidate,
    )

    assert compared["schema_version"] == "chemtools.compare-runs/1"
    assert compared["assessment"] == {
        "verdict": {
            "label": "candidate_lower_energy",
            "confidence": 0.6,
            "reasons": [
                "The candidate run has the lower parsed energy.",
                (
                    "The ordering is conditional because these fields were "
                    "not checked: charge, composition, xc_functional, basis, "
                    "geometry."
                ),
            ],
        },
        "comparability": {
            "status": "partially_checked",
            "blocking_fields": [],
            "unchecked_fields": [
                "charge",
                "composition",
                "xc_functional",
                "basis",
                "geometry",
            ],
        },
    }
    assert compared["evidence"]["energy"]["candidate_minus_reference"] == (
        pytest.approx(-0.1)
    )
    assert compared["evidence"]["energy"]["lower_energy_run"] == "candidate"
    assert compared["uncertainty"][-1]["code"] == (
        "run_comparability_partially_checked"
    )


def test_compare_runs_refuses_scientific_ordering_for_different_methods(tmp_path):
    reference = tmp_path / "reference.out"
    candidate = tmp_path / "candidate.out"
    reference.write_text("reference\n", encoding="utf-8")
    candidate.write_text("candidate\n", encoding="utf-8")

    compared = compare_runs(
        _backend({
            reference.name: _parsed_run(-10.0, "DFT"),
            candidate.name: _parsed_run(-11.0, "CCSD(T)"),
        }),
        reference,
        candidate,
    )

    assert compared["assessment"]["verdict"] == {
        "label": "comparison_not_supported",
        "confidence": 0.2,
        "reasons": [
            "Energy arithmetic is available, but required settings differ: method."
        ],
    }
    assert compared["assessment"]["comparability"]["blocking_fields"] == [
        "method"
    ]
    assert compared["evidence"]["energy"]["candidate_minus_reference"] == -1.0
    assert compared["next_actions"] == [{
        "action": "align_calculation_settings",
        "reason": "Re-run with matched required settings before interpreting energies.",
        "priority": 1,
    }]


def test_compare_runs_does_not_order_incomplete_energy_tasks(tmp_path):
    reference = tmp_path / "reference.out"
    candidate = tmp_path / "candidate.out"
    reference.write_text("reference\n", encoding="utf-8")
    candidate.write_text("candidate\n", encoding="utf-8")
    candidate_run = _parsed_run(-10.1)
    candidate_run["tasks"][0]["outcome"] = "incomplete"

    compared = compare_runs(
        _backend({
            reference.name: _parsed_run(-10.0),
            candidate.name: candidate_run,
        }),
        reference,
        candidate,
    )

    assert compared["assessment"]["verdict"] == {
        "label": "comparison_not_supported",
        "confidence": 0.2,
        "reasons": [
            "Energy arithmetic is available, but required settings differ: "
            "task_completion."
        ],
    }
    assert compared["assessment"]["comparability"]["blocking_fields"] == [
        "task_completion"
    ]


def test_compare_run_inspections_uses_state_for_primary_multitask_energy():
    reference = {
        "program": {"name": "nwchem"},
        "source": {"path": "reference.out"},
        "assessment": {"verdict": {"label": "success"}},
        "evidence": {
            "tasks": [_parsed_run(-10.0)["tasks"][0]],
            "input_output_consistency": {
                "checks": [{
                    "field": "multiplicity",
                    "status": "match",
                    "input": 6,
                    "output": 6,
                }],
            },
        },
        "uncertainty": [],
    }
    candidate_tasks = [
        _parsed_run(-2.0)["tasks"][0],
        _parsed_run(-3.0)["tasks"][0],
        _parsed_run(-10.5)["tasks"][0],
    ]
    for index, task in enumerate(candidate_tasks):
        task["index"] = index
    candidate = {
        "program": {"name": "nwchem"},
        "source": {"path": "candidate.out"},
        "assessment": {"verdict": {"label": "success"}},
        "evidence": {
            "tasks": candidate_tasks,
            "input_output_consistency": {
                "checks": [{
                    "field": "task_states",
                    "status": "match",
                    "tasks": [{
                        "task_index": 2,
                        "comparisons": {
                            "multiplicity": {
                                "status": "match",
                                "input": 2,
                                "output": 2,
                            },
                        },
                    }],
                }],
            },
        },
        "uncertainty": [],
    }

    compared = compare_run_inspections(reference, candidate)

    multiplicity = next(
        check
        for check in compared["evidence"]["comparability_checks"]
        if check["field"] == "multiplicity"
    )
    assert multiplicity == {
        "field": "multiplicity",
        "status": "different",
        "reference": 6,
        "candidate": 2,
        "comparison_axis": True,
    }
    assert compared["evidence"]["candidate"]["multiplicity"] == 2


def test_compare_runs_guided_handler_uses_application_contract():
    if not registry.has("nwchem"):
        register_builtin_backends()
    output = MCP_FIXTURES / "nwchem_scf.out"
    input_file = MCP_FIXTURES / "nwchem_h2.nw"

    compared = guided._handle_compare_runs({
        "reference_output_file": str(output),
        "candidate_output_file": str(output),
        "reference_input_file": str(input_file),
        "candidate_input_file": str(input_file),
        "program": "nwchem",
    })

    assert compared["schema_version"] == "chemtools.compare-runs/1"
    assert compared["program"] == "nwchem"
    assert compared["assessment"]["verdict"]["label"] == (
        "energies_equal_within_tolerance"
    )


def test_compare_runs_guided_handler_rejects_mixed_programs():
    if not registry.has("nwchem"):
        register_builtin_backends()

    compared = guided._handle_compare_runs({
        "reference_output_file": str(MCP_FIXTURES / "nwchem_scf.out"),
        "candidate_output_file": str(MCP_FIXTURES / "molcas_scf.out"),
    })

    assert compared == {
        "error": "program_mismatch",
        "message": (
            "run comparison requires outputs from the same program; detected "
            "'nwchem' and 'molcas'"
        ),
        "reference_program": "nwchem",
        "candidate_program": "molcas",
    }


@pytest.mark.skipif(
    not os.environ.get("CHEMTOOLS_REFERENCE_CORPUS"),
    reason="CHEMTOOLS_REFERENCE_CORPUS is not configured",
)
def test_feo_comparison_pins_quintet_energy_ordering():
    manifest = load_reference_manifest(NWCHEM_MANIFEST)
    verified = verify_reference_case(
        manifest,
        "nwchem.feo_spin_comparison",
        os.environ["CHEMTOOLS_REFERENCE_CORPUS"],
    )
    assert verified["outcome"] == "verified"
    artifacts = {
        artifact["id"]: artifact["path"]
        for artifact in verified["artifacts"]
    }

    compared = compare_runs(
        load_backend(BUILTIN_BACKENDS[0]),
        artifacts["triplet_output"],
        artifacts["quintet_output"],
        reference_input_file=artifacts["triplet_input"],
        candidate_input_file=artifacts["quintet_input"],
    )

    assert compared["assessment"]["verdict"]["label"] == (
        "candidate_lower_energy"
    )
    assert compared["assessment"]["comparability"] == {
        "status": "partially_checked",
        "blocking_fields": [],
        "unchecked_fields": ["geometry"],
    }
    assert compared["evidence"]["energy"] == {
        "status": "checked",
        "unit": "hartree",
        "reference": -1338.848567852637,
        "candidate": -1338.950662699587,
        "candidate_minus_reference": pytest.approx(-0.10209484695),
        "candidate_minus_reference_kcal_per_mol": pytest.approx(
            -64.0654837141
        ),
        "equality_tolerance_hartree": 1.0e-8,
        "lower_energy_run": "candidate",
    }
    multiplicity = next(
        check
        for check in compared["evidence"]["comparability_checks"]
        if check["field"] == "multiplicity"
    )
    assert multiplicity == {
        "field": "multiplicity",
        "status": "different",
        "reference": 3,
        "candidate": 5,
        "comparison_axis": True,
    }
    assert compared["next_actions"] == [
        {
            "action": "review_state_character",
            "run": "candidate",
            "reason": (
                "Confirm orbital occupations and state character before "
                "accepting the lower-energy solution."
            ),
            "priority": 1,
        },
        {
            "action": "extend_multiplicity_comparison",
            "reason": (
                "The runs compare different multiplicities; include other "
                "chemically plausible states when needed."
            ),
            "priority": 2,
        },
    ]

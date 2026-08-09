"""The guided calculation planner settles strategy before input syntax."""

from dataclasses import replace

import pytest

from chemtools.application.calculation_planning import (
    CALCULATION_PLAN_SCHEMA,
    CalculationPlanError,
    plan_calculation,
)
from chemtools.core.program import (
    InvalidProgramBackend,
    ProgramCapability,
    validate_backend,
)
from chemtools.mcp.catalog import BUILTIN_BACKENDS, load_backend
from chemtools.mcp.tools import guided


UO2_REQUEST = {
    "system": "UO2",
    "elements": ["U", "O"],
    "charge": 0,
    "multiplicity": 3,
    "stages": ["optimize", "frequency"],
}


def test_nwchem_plan_exposes_stages_decisions_and_assumptions():
    planned = plan_calculation(
        load_backend(BUILTIN_BACKENDS[0]),
        UO2_REQUEST,
    )

    assert planned["schema_version"] == CALCULATION_PLAN_SCHEMA
    assert planned["program"] == {"name": "nwchem"}
    assert planned["assessment"]["verdict"]["label"] == (
        "needs_scientific_decisions"
    )
    assert planned["evidence"]["protocol"]["name"] == "thermochem_dft"
    assert [
        (stage["kind"], stage["depends_on"])
        for stage in planned["evidence"]["stages"]
    ] == [("optimize", None), ("frequency", "opt")]
    assert [
        decision["field"]
        for decision in planned["evidence"]["required_decisions"]
    ] == [
        "geometry_source",
        "method",
        "functional",
        "basis",
        "relativistic",
        "state_strategy",
    ]
    assert [item["code"] for item in planned["uncertainty"]] == [
        "gas_phase_assumed",
        "harmonic_frequency_assumed",
    ]
    assert planned["next_actions"] == [{
        "action": "resolve_scientific_decisions",
        "fields": [
            "geometry_source",
            "method",
            "functional",
            "basis",
            "relativistic",
            "state_strategy",
        ],
        "reason": (
            "Set these choices before asking Chemtools to draft input syntax."
        ),
        "priority": 1,
    }]


def test_complete_nwchem_plan_is_ready_for_drafting():
    planned = plan_calculation(
        load_backend(BUILTIN_BACKENDS[0]),
        {
            **UO2_REQUEST,
            "method": "dft",
            "functional": "PBE0",
            "basis": {"U": "basis-u", "O": "def2-TZVP"},
            "ecp": {"U": "ecp-u"},
            "geometry_source": "reviewed XYZ coordinates in angstrom",
            "solvent": "gas phase",
            "state_strategy": "independent triplet starts and orbital review",
        },
    )

    assert planned["assessment"]["verdict"]["label"] == (
        "ready_for_input_drafting"
    )
    assert planned["evidence"]["required_decisions"] == []
    assert [item["code"] for item in planned["uncertainty"]] == [
        "harmonic_frequency_assumed"
    ]
    assert planned["next_actions"][0]["tool"] == "draft_input"


def test_plan_calculation_refuses_unimplemented_program_capability():
    molcas = load_backend(BUILTIN_BACKENDS[1])

    with pytest.raises(CalculationPlanError) as caught:
        plan_calculation(molcas, UO2_REQUEST)

    assert caught.value.as_dict() == {
        "error": "unsupported_capability",
        "message": "'molcas' does not support calculation planning",
        "program": "molcas",
    }


def test_plan_calculation_rejects_recipe_method_mismatch():
    with pytest.raises(CalculationPlanError) as caught:
        plan_calculation(
            load_backend(BUILTIN_BACKENDS[0]),
            {**UO2_REQUEST, "method": "hf"},
        )

    assert caught.value.code == "invalid_calculation_request"
    assert "support method='dft'" in str(caught.value)


def test_plan_calculation_rejects_unimplemented_stage_sequence():
    with pytest.raises(CalculationPlanError) as caught:
        plan_calculation(
            load_backend(BUILTIN_BACKENDS[0]),
            {**UO2_REQUEST, "stages": ["optimize", "energy"]},
        )

    assert caught.value.code == "invalid_calculation_request"
    assert "unsupported NWChem stage sequence" in str(caught.value)


def test_plan_calculation_does_not_need_or_create_files(tmp_path):
    before = tuple(tmp_path.iterdir())

    plan_calculation(load_backend(BUILTIN_BACKENDS[0]), UO2_REQUEST)

    assert tuple(tmp_path.iterdir()) == before == ()


def test_guided_plan_calculation_uses_application_contract():
    response = guided._handle_plan_calculation({
        "program": "nwchem",
        **UO2_REQUEST,
    })

    assert response["schema_version"] == CALCULATION_PLAN_SCHEMA
    assert response["request"]["system"] == "UO2"


def test_provider_capability_cannot_be_declared_without_provider():
    nwchem = load_backend(BUILTIN_BACKENDS[0])
    broken = replace(nwchem, planning=None)

    assert ProgramCapability.CALCULATION_PLAN in broken.capabilities
    with pytest.raises(
        InvalidProgramBackend,
        match=(
            "declares 'calculation.plan' but "
            "planning.plan_calculation is unavailable"
        ),
    ):
        validate_backend(broken)

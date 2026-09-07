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
from chemtools.core import registry
from chemtools.mcp.catalog import (
    BUILTIN_BACKENDS,
    load_backend,
    register_builtin_backends,
)
from chemtools.mcp.tools import guided


UO2_REQUEST = {
    "system": "UO2",
    "elements": ["U", "O"],
    "charge": 0,
    "multiplicity": 3,
    "stages": ["optimize", "frequency"],
}

BE_GRASP_REQUEST = {
    "system": "Be-9",
    "elements": ["Be"],
    "charge": 0,
    "multiplicity": 1,
    "stages": ["energy"],
    "method": "MCDHF + RCI",
}

BE_GRASP_OPTIONS = {
    "isotope_mass_number": 9,
    "nuclear_model": {
        "type": "fermi",
        "mass_treatment": "static",
        "bohr_radius_fm": 52917.7210544,
        "fermi_a_fm": 0.523387555310,
        "fermi_c_fm": 2.067117402628,
    },
    "reference_configurations": [
        "1s(2,i)2s(2,*)",
        "1s(2,i)2p(2,*)",
    ],
    "inactive_core_subshells": ["1s"],
    "inactive_core_electrons": 2,
    "active_subshells": ["2s", "2p-", "2p"],
    "active_electrons": 2,
    "target": {
        "two_j": 0,
        "parity": "+",
        "asf_selections": ["1"],
    },
    "correlation": {
        "substitutions": "SD",
        "active_layers": [
            ["3s", "3p-", "3p", "3d-", "3d"],
            ["4s", "4p-", "4p", "4d-", "4d", "4f-", "4f"],
        ],
    },
    "orbital_optimization": {
        "minimal_cas": "all",
        "correlation_layers": "new_only",
    },
    "hamiltonian_variants": [
        "dirac_coulomb",
        "dirac_coulomb_plus_zero_frequency_breit",
    ],
    "numerical_controls": {
        "radial_grid_h": 0.05,
        "radial_grid_points": 590,
        "scf_cycles": 100,
    },
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


def test_grasp_plan_preserves_atomic_model_space_and_accuracy_ladder():
    planned = plan_calculation(
        load_backend(BUILTIN_BACKENDS[3]),
        {**BE_GRASP_REQUEST, "program_options": BE_GRASP_OPTIONS},
    )

    assert planned["assessment"]["verdict"]["label"] == (
        "ready_for_grasp_workflow_planning"
    )
    assert planned["evidence"]["required_decisions"] == []
    model_space = planned["evidence"]["protocol"]["model_space"]
    assert model_space["electrons"] == 4
    assert model_space["inactive_core_electrons"] == 2
    assert model_space["active_electrons"] == 2
    assert model_space["active_spinors"] == 8
    assert model_space["raw_active_determinants"] == 28
    assert model_space["grasp_csf_count"] == "verify_after_rcsfgenerate"
    assert [stage["id"] for stage in planned["evidence"]["stages"]] == [
        "nuclear_setup",
        "dhf_reference",
        "minimal_cas_mcscf",
        "correlation_layer_1",
        "correlation_layer_2",
        "fixed_orbital_rci_variants",
    ]
    assert planned["next_actions"][0]["action"] == ("prepare_grasp_workflow_stages")


def test_grasp_plan_exposes_missing_atomic_decisions():
    planned = plan_calculation(
        load_backend(BUILTIN_BACKENDS[3]),
        BE_GRASP_REQUEST,
    )

    assert planned["assessment"]["verdict"]["label"] == ("needs_scientific_decisions")
    fields = {
        decision["field"] for decision in planned["evidence"]["required_decisions"]
    }
    assert "program_options.nuclear_model" in fields
    assert "program_options.inactive_core_subshells" in fields
    assert "program_options.correlation" in fields
    assert "program_options.hamiltonian_variants" in fields


def test_grasp_plan_rejects_inconsistent_core_active_electron_count():
    with pytest.raises(CalculationPlanError) as caught:
        plan_calculation(
            load_backend(BUILTIN_BACKENDS[3]),
            {
                **BE_GRASP_REQUEST,
                "program_options": {
                    **BE_GRASP_OPTIONS,
                    "active_electrons": 3,
                },
            },
        )

    assert caught.value.code == "invalid_calculation_request"
    assert "must equal Z - charge (4)" in str(caught.value)


def test_grasp_plan_rejects_nonphysical_nuclear_parameters():
    with pytest.raises(CalculationPlanError) as caught:
        plan_calculation(
            load_backend(BUILTIN_BACKENDS[3]),
            {
                **BE_GRASP_REQUEST,
                "program_options": {
                    **BE_GRASP_OPTIONS,
                    "nuclear_model": {
                        **BE_GRASP_OPTIONS["nuclear_model"],
                        "fermi_c_fm": -2.0,
                    },
                },
            },
        )

    assert caught.value.code == "invalid_calculation_request"
    assert "fermi_c_fm must be a positive finite number" in str(caught.value)


def test_grasp_plan_rejects_overlapping_core_and_active_subshells():
    with pytest.raises(CalculationPlanError) as caught:
        plan_calculation(
            load_backend(BUILTIN_BACKENDS[3]),
            {
                **BE_GRASP_REQUEST,
                "program_options": {
                    **BE_GRASP_OPTIONS,
                    "active_subshells": ["1s", "2s", "2p-", "2p"],
                },
            },
        )

    assert caught.value.code == "invalid_calculation_request"
    assert "inactive and active subshells overlap: 1s" in str(caught.value)


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
    if not registry.has("nwchem"):
        register_builtin_backends()
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

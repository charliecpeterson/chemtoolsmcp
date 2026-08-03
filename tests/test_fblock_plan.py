"""Scientific and MCP contracts for exact f-block atomic planning."""

from __future__ import annotations

import json

import pytest

from chemtools.mcp.decorator import _TOOL_CAPABILITIES, _TOOL_PROGRAMS
from chemtools.mcp.dispatch import dispatch_tool, tool_definitions
from chemtools.reference import (
    bundled_fblock_directory,
    load_atsp_element_recipes,
    load_fblock_catalog,
    plan_fblock_atomic_state,
)


@pytest.mark.parametrize(
    ("element", "state"),
    (
        ("Ce", "ion3_4f1"),
        ("Th", "ion0_6d27s2"),
        ("U", "ion4_5f16d1"),
        ("Tb", "ion0_4f85d16s2"),
        ("Am", "ion0_5f77s2"),
    ),
)
def test_planner_reproduces_every_shipped_grasp_stdin_example(element, state):
    inputs = plan_fblock_atomic_state(element, state).to_dict()["grasp2018"][
        "inputs"
    ]
    example = (
        bundled_fblock_directory()
        / "grasp"
        / "stdin-examples"
        / f"{element}-{state}"
    )

    for key in (
        "rnucleus",
        "rcsfgenerate",
        "rangular",
        "rwfnestimate",
        "rmcdhf",
        "rci",
    ):
        expected = (example / f"{key}.stdin").read_text(
            encoding="utf-8"
        ).splitlines()
        assert inputs[key] == expected


def test_atsp_recipes_match_all_633_catalog_states_exactly():
    catalog = load_fblock_catalog()
    recipes = [load_atsp_element_recipes(element) for element in catalog.elements]

    assert len(recipes) == 31
    assert sum(len(element.states) for element in recipes) == 633
    thorium = next(element for element in recipes if element.symbol == "Th")
    neutral = thorium.state("ion0_6d27s2")
    assert neutral.closed_shells == "  5s  5p  5d  6s  6p  7s"
    assert neutral.open_configuration == "6d(2)"
    uranium = next(element for element in recipes if element.symbol == "U")
    assert dict(uranium.donor_pins)["ion3_5f3"] == "ion4_5f2"
    assert dict(uranium.donor_pins)["ion5_7s1"] == "ion3_7s1"


def test_thorium_plan_reproduces_recorded_false_vacuum_inputs():
    payload = plan_fblock_atomic_state("Th", "ion0_6d27s2").to_dict()

    assert payload["schema_version"] == "chemtools.fblock-atomic-plan/1"
    assert payload["plan_status"] == "complete"
    assert payload["automation"]["status"] == "manual_steps_required"
    assert payload["automation"]["requirements"][0]["kind"] == (
        "atsp2k_seed_conversion"
    )
    assert payload["atsp2k"]["required_for_grasp_seed"] is True
    assert payload["atsp2k"]["ecp_card_included"] is False
    assert payload["atsp2k"]["stdin_lines"] == [
        "Th,AV,90.",
        "  5s  5p  5d  6s  6p  7s",
        "6d(2)",
        "all",
        "y",
        "n",
        "y",
        "y",
        "n",
        "99 2",
        "y",
        "n",
        "n",
    ]
    inputs = payload["grasp2018"]["inputs"]
    assert inputs == {
        "rnucleus": ["90", "232", "n", "0", "0.5", "1", "1"],
        "rcsfgenerate": [
            "*",
            "5",
            "4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)7s(2,i)",
            "",
            "7s,6p,6d,4f",
            "0,12",
            "0",
            "n",
        ],
        "rangular": ["y"],
        "rwfnestimate": ["y", "1", "prev.w", "*", "3", "*"],
        "rmcdhf": ["y", "1-2", "1", "1-3", "1", "1-2", "5", "*", "*", "100"],
        "rci": [
            "y", "ref", "y", "y", "1.d-6", "n", "n", "n", "n",
            "1-2", "1", "1-3", "1", "1-2",
        ],
    }
    assert payload["grasp2018"]["expected"]["energies_au"][
        "dirac_coulomb_breit"
    ] == -26475.800396508253
    assert "/home/" not in json.dumps(payload)


def test_uranium_plan_orders_prerequisites_and_preserves_both_donors():
    payload = plan_fblock_atomic_state("U", "ion4_5f16d1").to_dict()

    assert payload["plan_status"] == "complete"
    assert payload["dependencies"]["direct_donors"] == [
        {"identifier": "ion4_6d2", "kind": "catalog_state"},
        {"identifier": "ion5_5f1", "kind": "catalog_state"},
    ]
    assert [
        item["state"]
        for item in payload["dependencies"]["ordered_prerequisites"]
    ] == ["ion4_6d2", "ion6_closed", "ion5_5f1"]
    assert payload["dependencies"]["unresolved_donor_aliases"] == []
    assert payload["automation"]["requirements"][0] == {
        "kind": "multi_donor_orbital_merge",
        "donors": ["ion4_6d2", "ion5_5f1"],
        "detail": (
            "Merge donor radial-wavefunction records by orbital identity; "
            "the first donor wins duplicate orbitals."
        ),
    }
    assert payload["automation"]["requirements"][1]["kind"] == (
        "prerequisite_seed_preparation"
    )
    assert [
        item["state"]
        for item in payload["automation"]["requirements"][1]["states"]
    ] == ["ion4_6d2", "ion6_closed", "ion5_5f1"]
    assert payload["grasp2018"]["inputs"]["rci"][-7:] == [
        "1", "1-3", "1-4", "1-4", "1-4", "1-3", "1",
    ]


def test_external_donor_alias_is_retained_as_a_plan_blocker():
    payload = plan_fblock_atomic_state("La", "ion2_4f1").to_dict()

    assert payload["plan_status"] == "needs_donor_mapping"
    assert payload["dependencies"]["direct_donors"] == [
        {"identifier": "donor_Cef1", "kind": "external_alias"}
    ]
    assert payload["dependencies"]["ordered_prerequisites"] == []
    assert payload["dependencies"]["unresolved_donor_aliases"] == [
        {
            "element": "La",
            "consumer_state": "ion2_4f1",
            "alias": "donor_Cef1",
            "status": "unresolved",
            "reason": (
                "The committed catalog records the alias but does not "
                "identify an exact donor state or bundle the referenced "
                "radial-wavefunction artifact."
            ),
        }
    ]
    assert payload["dependencies"]["donor_alias_manifest"]["record_count"] == 132
    assert payload["dependencies"]["donor_alias_manifest"]["status"] == (
        "scientific_review_required"
    )
    assert payload["automation"]["requirements"][-1] == {
        "kind": "external_donor_mapping",
        "aliases": ["donor_Cef1"],
    }


def test_cold_state_plan_is_input_ready_and_omits_rmcdhf_weight_for_one_level():
    payload = plan_fblock_atomic_state("Ce", "ion4_closed").to_dict()

    assert payload["target"]["state"]["seed"]["class"] == "cold"
    assert payload["automation"] == {
        "status": "input_ready",
        "requirements": [],
    }
    assert payload["grasp2018"]["inputs"]["rwfnestimate"] == ["y", "2", "*"]
    assert payload["grasp2018"]["inputs"]["rmcdhf"] == [
        "y", "1", "*", "*", "100",
    ]


def test_incomplete_y_reference_refuses_to_infer_grasp_inputs():
    payload = plan_fblock_atomic_state("Y", "ion2_4f1").to_dict()

    assert payload["plan_status"] == "incomplete_reference_input"
    assert payload["automation"]["status"] == "unavailable"
    assert payload["grasp2018"]["availability"] == (
        "incomplete_reference_input"
    )
    assert payload["grasp2018"]["missing_fields"] == [
        "confline",
        "active_set",
        "jrange",
        "core_menu",
    ]
    assert payload["grasp2018"]["inputs"] is None
    assert payload["atsp2k"]["stdin_lines"][0] == "Y,AV,39."


def test_exactly_seventeen_y_states_lack_grasp_prompt_fields():
    yttrium = load_fblock_catalog().element("Y")
    incomplete = [
        state.slug
        for state in yttrium.states
        if not state.confline
        or not state.active_set
        or not state.jrange
        or state.core_menu is None
    ]

    assert incomplete == [
        "ion0_4d15s2",
        "ion0_4d25s1",
        "ion0_5s25p1",
        "ion0_4d15s15p1",
        "ion0_4d3",
        "ion1_5s2",
        "ion1_4d15s1",
        "ion1_4d2",
        "ion1_5s15p1",
        "ion2_4d1",
        "ion2_5s1",
        "ion2_5p1",
        "ion3_closed",
        "ion2_4f1",
        "ion2_6s1",
        "ion1_4d15p1",
        "ion1_5p2",
    ]


def test_mcp_plan_contract_is_analysis_safe_and_rejects_partial_queries():
    payload = dispatch_tool(
        "plan_fblock_atomic_state",
        {"element": "Ce", "state": "ion3_4f1"},
    )

    assert payload["plan_status"] == "complete"
    assert _TOOL_PROGRAMS["plan_fblock_atomic_state"] == "grasp"
    assert _TOOL_CAPABILITIES["plan_fblock_atomic_state"] == "none"

    with pytest.raises(
        ValueError,
        match=r"missing f-block plan arguments: \['state'\]",
    ):
        dispatch_tool("plan_fblock_atomic_state", {"element": "Ce"})
    with pytest.raises(
        ValueError,
        match=r"unknown f-block plan arguments: \['charge'\]",
    ):
        dispatch_tool(
            "plan_fblock_atomic_state",
            {"element": "Ce", "state": "ion3_4f1", "charge": 3},
        )


def test_mcp_plan_schema_requires_exact_catalog_identifiers():
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "plan_fblock_atomic_state"
    )

    assert definition["inputSchema"] == {
        "type": "object",
        "properties": {
            "element": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2,
                "pattern": "^[A-Za-z]{1,2}$",
                "description": (
                    "Exact element symbol in the catalog, such as Th, U, or "
                    "Y. Matching is case-insensitive."
                ),
            },
            "state": {
                "type": "string",
                "minLength": 8,
                "maxLength": 32,
                "pattern": "^ion[0-9]+_[a-z0-9]+$",
                "description": "Exact catalog state slug, such as ion0_6d27s2.",
            },
        },
        "required": ["element", "state"],
        "additionalProperties": False,
    }

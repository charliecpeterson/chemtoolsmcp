"""Public and MCP contracts for exact f-block reference lookup."""

from __future__ import annotations

import json

import pytest

from chemtools.mcp.decorator import _TOOL_CAPABILITIES, _TOOL_PROGRAMS
from chemtools.mcp.dispatch import dispatch_tool, tool_definitions
from chemtools.reference import lookup_grasp_fblock_state


def test_element_lookup_returns_bounded_state_index_and_provenance():
    payload = lookup_grasp_fblock_state("th").to_dict()

    assert payload["schema_version"] == "chemtools.fblock-reference-lookup/2"
    assert payload["query"] == {"element": "Th", "state": None}
    assert payload["reference"]["status"] == "validated_reference"
    assert payload["reference"]["recommendation_eligible"] is True
    assert payload["reference"]["recommendation_scope"] == (
        "grasp2018_reference_only"
    )
    assert payload["reference"]["cross_program_transfer_eligible"] is False
    assert payload["reference"]["dataset"] == {
        "id": "fblock.atomic_seeds",
        "version": "3",
        "rebuild_date": "2026-08-05",
        "payload_schema": "fblock.element-map/legacy-v2",
        "catalog_sha256": (
            "ba3397c7ff0634c489cc0dded2a857ef0b8151897618c7b2daa4d90502aadbec"
        ),
    }
    assert payload["reference"]["component"] == {
        "id": "grasp_v2_catalog",
        "purposes": ["scientific_regression", "workflow_recipe"],
    }
    assert payload["element"]["atomic_number"] == 90
    assert payload["state_index"]["total_count"] == 22
    assert payload["state_index"]["returned_count"] == 22
    assert payload["state_index"]["truncated"] is False
    assert payload["state_index"]["states"][0] == {
        "slug": "ion0_6d27s2",
        "ion": 0,
        "config": "6d(2)7s(2)",
        "role": "fit",
        "seed_class": "atsp_hf",
        "staged_birth": False,
        "energy_relative_au": -2.215842729452561,
        "unconstrained_scf_risk": "not_assessed",
        "cross_program_transfer_eligible": False,
    }


def test_exact_thorium_lookup_keeps_false_vacuum_method_context():
    payload = lookup_grasp_fblock_state("Th", "ion0_6d27s2").to_dict()
    semantics = payload["state"].pop("semantics")

    assert "state_index" not in payload
    assert payload["state"] == {
        "slug": "ion0_6d27s2",
        "ion": 0,
        "config": "6d(2)7s(2)",
        "core": "Xe",
        "confline": "4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)7s(2,i)",
        "role": "fit",
        "note": "neutral ground config (Th grounds d2s2, unlike Ce)",
        "seed": {
            "class": "atsp_hf",
            "instruction": (
                "ATSP-hf seed (non-relativistic orbitals, converted)"
            ),
            "hf_seed": None,
            "estimate_from": [],
            "vary_first": None,
        },
        "grasp": {
            "core_menu": 5,
            "active_set": "7s,6p,6d,4f",
            "jrange": "0,12",
            "j_blocks": ["0", "1", "2", "3", "4"],
            "ncsf": [2, 1, 3, 1, 2],
        },
        "energies_au": {
            "dirac_coulomb": -26510.488591596226,
            "dirac_coulomb_breit": -26475.800396508253,
            "relative_to_anchor": -2.215842729452561,
        },
    }
    assert "no QED or mass shifts" in (
        payload["reference"]["method_scope"]["grasp_hamiltonian"]
    )
    assert "method-scoped configuration averages" in (
        payload["reference"]["known_limitations"][0]
    )
    assert "/home/" not in json.dumps(payload)
    assert semantics["occupancy"]["electron_count"] == 90
    assert semantics["occupancy"]["complete"] is True
    assert semantics["cross_program_transfer"]["eligible"] is False


def test_closed_anchor_warns_before_cross_program_scf_transfer():
    semantics = lookup_grasp_fblock_state(
        "Tm",
        "ion15_closed",
    ).to_dict()["state"]["semantics"]

    assert semantics["state_class"] == "closed_anchor"
    assert semantics["unconstrained_scf_risk"] == "constraint_required"
    assert semantics["occupancy"]["electron_count"] == 54
    assert semantics["occupancy"]["represented_angular_electrons"] == {
        "s": 2,
        "p": 6,
        "d": 10,
        "f": 0,
        "g": 0,
        "h": 0,
    }
    assert semantics["occupancy"]["omitted_core_electrons"] == 36


def test_one_electron_lookup_reports_signed_f_d_separation_and_d2h_limit():
    semantics = lookup_grasp_fblock_state(
        "Ce",
        "ion3_4f1",
    ).to_dict()["state"]["semantics"]

    assert semantics["unconstrained_scf_risk"] == "constraint_required"
    assert semantics["cross_program_transfer"]["d2h_f_p_separation"] == (
        "insufficient"
    )
    assert semantics["f_d_separation"] == {
        "definition": "E_d_minus_E_f",
        "f_state": "ion3_4f1",
        "d_state": "ion3_5d1",
        "delta_au": 0.20213299809802265,
        "delta_ev": 5.500319084304852,
        "energy_field": "dirac_coulomb_breit",
    }


def test_exact_uranium_lookup_preserves_multi_donor_lineage():
    state = lookup_grasp_fblock_state("U", "ion4_5f16d1").to_dict()["state"]

    assert state["seed"] == {
        "class": "multi_donor",
        "instruction": "multi-donor merge: ion4_6d2 + ion5_5f1",
        "hf_seed": False,
        "estimate_from": ["ion4_6d2", "ion5_5f1"],
        "vary_first": None,
    }
    assert state["grasp"]["j_blocks"] == ["0", "1", "2", "3", "4", "5", "6"]
    assert state["grasp"]["ncsf"] == [1, 3, 4, 4, 4, 3, 1]


@pytest.mark.parametrize(
    ("element", "state", "error", "message"),
    (
        ("Fe", None, ValueError, "no f-block reference for element 'Fe'"),
        ("Thorium", None, ValueError, "invalid element symbol"),
        ("Th", 4, TypeError, "state must be a string"),
        ("Th", "ION0_6D27S2", ValueError, "exact catalog slug"),
        ("Th", "ion0_5f99", ValueError, "no f-block reference for state"),
    ),
)
def test_lookup_rejects_nonexact_or_unknown_identifiers(
    element,
    state,
    error,
    message,
):
    with pytest.raises(error, match=message):
        lookup_grasp_fblock_state(element, state)


def test_mcp_lookup_contract_is_grasp_scoped_and_rejects_unknown_arguments():
    payload = dispatch_tool(
        "lookup_grasp_fblock_state",
        {"element": "Y", "state": "ion0_4d15s2"},
    )

    assert payload["query"] == {"element": "Y", "state": "ion0_4d15s2"}
    assert payload["element"]["atomic_number"] == 39
    assert _TOOL_PROGRAMS["lookup_grasp_fblock_state"] == "grasp"
    assert _TOOL_CAPABILITIES["lookup_grasp_fblock_state"] == "none"

    with pytest.raises(
        ValueError,
        match=r"unknown f-block lookup arguments: \['path'\]",
    ):
        dispatch_tool(
            "lookup_grasp_fblock_state",
            {"element": "Y", "path": "/somewhere"},
        )


def test_mcp_lookup_schema_requires_only_bounded_identifiers():
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "lookup_grasp_fblock_state"
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
                "description": (
                    "Optional exact catalog state slug, such as "
                    "ion0_6d27s2. Omit it to list the element's available "
                    "states."
                ),
            },
        },
        "required": ["element"],
        "additionalProperties": False,
    }

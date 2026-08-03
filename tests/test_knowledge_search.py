"""Status, scope, and MCP contracts for curated knowledge search."""

from __future__ import annotations

import pytest

from chemtools.knowledge.search import search_knowledge_cards
from chemtools.mcp.dispatch import dispatch_tool, tool_definitions


def test_search_defaults_to_accepted_cards_only():
    result = search_knowledge_cards()

    assert [card.id for card in result.cards] == [
        "cross_program.optimizer_failure_sentinel_must_lose",
        "cross_program.same_producer_is_correlated",
        "cross_program.same_starting_guess_class_is_one_measurement",
        "grasp.rmcdhf.zero_exit_requires_convergence",
        "pyscf.electron_spin_consistency_is_runtime_required",
        "pyscf.scf_convergence_is_separate_from_execution",
        "qmcpack.determinant_only_vmc_offsets",
        "qmcpack.fblock_dmc_reference_protocol",
        "qmcpack.jastrow_vmc_energy_gate",
        "qmcpack.variational_parameter_sidecar",
    ]
    assert result.total_matches == 10
    assert all(card.status == "accepted" for card in result.cards)


def test_nonaccepted_cards_require_explicit_status():
    result = search_knowledge_cards(status="draft")

    assert [card.id for card in result.cards] == [
        "cross_program.cheap_invariants_find_wrong_basins",
        "cross_program.silent_success",
    ]
    payload = result.to_dict()
    assert payload["filters"]["status"] == "draft"
    assert [
        card["recommendation_eligible"] for card in payload["cards"]
    ] == [False, False]


def test_search_combines_text_and_scope_filters():
    result = search_knowledge_cards(
        query="positive convergence",
        program="GRASP",
        workflow="scf",
        kind="validation",
        confidence="high",
    )

    assert result.program == "grasp"
    assert result.total_matches == 1
    assert result.cards[0].id == (
        "grasp.rmcdhf.zero_exit_requires_convergence"
    )


def test_named_program_filter_includes_cross_program_cards():
    result = search_knowledge_cards(
        query="numeric sentinel",
        program="qmcpack",
    )

    assert [card.id for card in result.cards] == [
        "cross_program.optimizer_failure_sentinel_must_lose"
    ]


def test_qmcpack_cards_are_scoped_to_their_workflows():
    result = search_knowledge_cards(
        program="qmcpack",
        workflow="jastrow_optimization",
    )

    assert [card.id for card in result.cards] == [
        "qmcpack.jastrow_vmc_energy_gate",
        "qmcpack.variational_parameter_sidecar",
    ]
    assert all(card.status == "accepted" for card in result.cards)


def test_pyscf_cards_are_scoped_to_the_bounded_single_point_runner():
    result = search_knowledge_cards(
        program="pyscf",
        workflow="single_point_scf",
    )

    assert [card.id for card in result.cards] == [
        "pyscf.electron_spin_consistency_is_runtime_required",
        "pyscf.scf_convergence_is_separate_from_execution",
    ]
    assert all(card.status == "accepted" for card in result.cards)


def test_search_limit_reports_truncation_exactly():
    payload = search_knowledge_cards(limit=2).to_dict()

    assert payload["total_matches"] == 10
    assert payload["returned_count"] == 2
    assert payload["truncated"] is True
    assert payload["cards"][0]["recommendation_eligible"] is True
    assert payload["cards"][0]["sources"]
    assert payload["cards"][0]["tests"]


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    (
        ({"query": "   "}, ValueError, "query must be a non-empty string"),
        ({"query": "..."}, ValueError, "at least one letter or number"),
        ({"program": "quantum espresso"}, ValueError, "lowercase identifier"),
        ({"kind": "advice"}, ValueError, "kind must be one of"),
        ({"status": "any"}, ValueError, "status must be one of"),
        ({"status": None}, ValueError, "status must be one of"),
        ({"limit": True}, TypeError, "limit must be an integer"),
        ({"limit": 51}, ValueError, "limit must be between 1 and 50"),
    ),
)
def test_search_rejects_invalid_filters(arguments, error, message):
    with pytest.raises(error, match=message):
        search_knowledge_cards(**arguments)


def test_mcp_tool_defaults_to_accepted_and_returns_traceability():
    payload = dispatch_tool(
        "search_knowledge_cards",
        {"query": "failure sentinel"},
    )

    assert payload["schema_version"] == "chemtools.knowledge-search/1"
    assert payload["filters"]["status"] == "accepted"
    assert payload["returned_count"] == 1
    assert payload["cards"][0]["id"] == (
        "cross_program.optimizer_failure_sentinel_must_lose"
    )
    assert payload["cards"][0]["recommendation_eligible"] is True


def test_mcp_tool_rejects_unknown_arguments():
    with pytest.raises(
        ValueError,
        match=r"unknown search arguments: \['statsu'\]",
    ):
        dispatch_tool("search_knowledge_cards", {"statsu": "draft"})


def test_mcp_schema_pins_status_default_and_bounds():
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "search_knowledge_cards"
    )
    properties = definition["inputSchema"]["properties"]

    assert properties["status"] == {
        "type": "string",
        "enum": [
            "draft",
            "accepted",
            "exploratory",
            "shelved",
            "rejected",
        ],
        "default": "accepted",
        "description": (
            "One explicit curation state. Omit to search only accepted cards."
        ),
    }
    assert properties["limit"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 50,
        "default": 10,
    }

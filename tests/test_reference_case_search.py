"""Bounded application and MCP contracts for packaged reference-case search."""

from __future__ import annotations

import pytest

from chemtools.application.reference_case_search import (
    MAX_REFERENCE_RESULTS,
    REFERENCE_CASE_SEARCH_SCHEMA,
    ReferenceCaseSearchError,
    bundled_reference_manifest_paths,
    find_reference_cases,
)
from chemtools.mcp.tools import reference


def test_bundled_search_finds_ranked_nwchem_cases_with_required_pins():
    found = find_reference_cases(
        "open-shell fragment guess",
        program="nwchem",
        scientific_status="exploratory",
    )

    assert found["schema_version"] == REFERENCE_CASE_SEARCH_SCHEMA
    assert found["query"] == {
        "text": "open-shell fragment guess",
        "program": "nwchem",
        "scientific_status": "exploratory",
        "limit": 5,
    }
    assert [case["case_id"] for case in found["matches"]] == [
        "nwchem.fecn6_lowspin_fragment",
        "nwchem.hexaaquairon_swap_chain",
    ]
    first = found["matches"][0]
    assert first["scientific_status"] == "exploratory"
    assert first["pinning"]["required_artifact_count"] == 4
    assert [
        artifact["id"]
        for artifact in first["pinning"]["required_artifacts"]
    ] == [
        "failed_input",
        "failed_output",
        "solution_input",
        "solution_output",
    ]
    assert all(
        len(artifact["sha256"]) == 64
        and not artifact["relative_path"].startswith("/")
        for artifact in first["pinning"]["required_artifacts"]
    )
    assert [item["code"] for item in found["uncertainty"]] == [
        "artifact_availability_not_checked",
        "scientific_review_incomplete",
    ]


def test_validated_only_search_does_not_promote_exploratory_cases():
    found = find_reference_cases(
        "open-shell transition-metal SCF recovery",
        program="nwchem",
    )

    assert found["match_count"] == 0
    assert found["matches"] == []
    assert [item["code"] for item in found["uncertainty"]] == [
        "no_matching_reference_case",
    ]
    assert found["next_actions"] == []


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        ({"query": "   "}, "invalid_reference_query"),
        (
            {"query": "SCF", "scientific_status": "approved"},
            "invalid_reference_status",
        ),
        (
            {"query": "SCF", "limit": MAX_REFERENCE_RESULTS + 1},
            "invalid_reference_limit",
        ),
    ],
)
def test_search_rejects_invalid_boundary_values(arguments, code):
    with pytest.raises(ReferenceCaseSearchError) as caught:
        find_reference_cases(**arguments)

    assert caught.value.code == code


def test_packaged_manifests_are_the_only_default_search_sources():
    assert [path.name for path in bundled_reference_manifest_paths()] == [
        "non_nwchem_review_cases.json",
        "nwchem_behavior_cases.json",
        "orca_experimental_cases.json",
    ]


def test_mcp_handler_delegates_to_application_search(monkeypatch):
    captured = {}

    def fake_search(query, **arguments):
        captured.update(query=query, **arguments)
        return {"schema_version": REFERENCE_CASE_SEARCH_SCHEMA}

    monkeypatch.setattr(reference, "find_reference_cases", fake_search)

    response = reference._handle_find_reference_case({
        "query": "basis stepping",
        "program": "nwchem",
        "scientific_status": "exploratory",
        "limit": 2,
    })

    assert response == {"schema_version": REFERENCE_CASE_SEARCH_SCHEMA}
    assert captured == {
        "query": "basis stepping",
        "program": "nwchem",
        "scientific_status": "exploratory",
        "limit": 2,
    }

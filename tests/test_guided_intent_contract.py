"""The realistic prompt contract stays aligned with the guided MCP surface."""

from pathlib import Path

import yaml

from chemtools.mcp import modes
from chemtools.mcp.dispatch import tool_definitions


CONTRACT_PATH = (
    Path(__file__).parents[1] / "docs" / "guided-intent-contract.yaml"
)

EXPECTED_INTENTS = {
    "review_input": ("review_input", "review_input", "guided"),
    "inspect_run": ("inspect_run", "inspect_run", "guided"),
    "retrieve_knowledge": (
        "search_knowledge",
        "search_knowledge",
        "guided",
    ),
    "plan_calculation": (
        "plan_calculation",
        "plan_calculation",
        "guided",
    ),
    "draft_input": ("draft_input", "draft_input", "guided"),
    "plan_recovery": ("plan_recovery", "plan_recovery", "guided"),
    "compare_runs": ("compare_runs", "compare_runs", "guided"),
    "launch_run": ("launch_run", "launch_run", "guided"),
    "monitor_run": ("monitor_run", "monitor_run", "guided"),
    "visualize": ("visualize", "visualize", "guided"),
    "find_reference_case": (
        "find_reference_case",
        "find_reference_case",
        "guided",
    ),
}

EXPECTED_ANALYSIS_TOOLS = [
    "review_input",
    "inspect_run",
    "compare_runs",
    "plan_recovery",
]


def _contract() -> dict:
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_contract_covers_each_retained_intent_with_a_real_prompt():
    contract = _contract()
    intents = contract["intents"]

    assert contract["schema_version"] == (
        "chemtools.guided-intent-contract/1"
    )
    assert contract["maximum_public_tools"] == 12
    assert len(intents) == len(EXPECTED_INTENTS) == 11
    assert {intent["id"] for intent in intents} == set(EXPECTED_INTENTS)
    assert len({intent["prompt"] for intent in intents}) == len(intents)
    for intent in intents:
        assert 80 <= len(intent["prompt"]) <= 400
        assert intent["selection_rule"].strip()
        assert intent["target_tool"] not in intent["do_not_choose"]


def test_current_and_target_tool_choices_are_explicit():
    intents = {intent["id"]: intent for intent in _contract()["intents"]}

    for intent_id, expected in EXPECTED_INTENTS.items():
        intent = intents[intent_id]
        assert (
            intent["target_tool"],
            intent["current_tool"],
            intent["implementation"],
        ) == expected


def test_analysis_result_contract_names_shared_fields_and_finding_locations():
    result_contract = _contract()["analysis_result_contract"]

    assert result_contract["tools"] == EXPECTED_ANALYSIS_TOOLS
    assert set(result_contract["shared_fields"]) == {
        "assessment",
        "evidence",
        "uncertainty",
        "next_actions",
    }
    assert result_contract["finding_locations"] == {
        "review_input": ["evidence.lint.issues"],
        "inspect_run": [
            "evidence.diagnostics",
            "evidence.diagnosis_anchors",
        ],
        "compare_runs": [
            "evidence.comparability_checks",
            "evidence.energy",
        ],
        "plan_recovery": [
            "evidence.plan_kind",
            "evidence.prepared_artifacts",
        ],
    }


def test_implemented_choices_match_the_live_guided_preset():
    intents = _contract()["intents"]
    current_tools = {
        intent["current_tool"]
        for intent in intents
        if intent["current_tool"] is not None
    }
    all_public_tools = {
        definition["name"] for definition in tool_definitions()
    }

    assert current_tools == set(modes.TOOLSETS["guided"])
    assert current_tools <= all_public_tools
    for intent in intents:
        assert intent["implementation"] == "guided"
        assert intent["current_tool"] == intent["target_tool"]

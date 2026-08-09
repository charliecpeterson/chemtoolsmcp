"""The optional Codex plugin stays thin, portable, and guided."""

from collections import Counter
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
PLUGIN_ROOT = ROOT / "plugins" / "chemtools"
EXPECTED_SKILLS = {
    "chemtools-inspect-run",
    "chemtools-monitor-run",
    "chemtools-plan-calculation",
    "chemtools-review-input",
}
GUIDED_TOOLS = {
    "compare_runs",
    "draft_input",
    "find_reference_case",
    "inspect_run",
    "launch_run",
    "monitor_run",
    "plan_calculation",
    "plan_recovery",
    "review_input",
    "search_knowledge",
    "visualize",
}


def test_plugin_manifest_points_only_to_present_components():
    manifest = json.loads(
        (PLUGIN_ROOT / ".codex-plugin" / "plugin.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["name"] == "chemtools"
    assert manifest["version"] == "0.1.0"
    assert manifest["skills"] == "./skills/"
    assert manifest["mcpServers"] == "./.mcp.json"
    assert "apps" not in manifest
    assert "hooks" not in manifest
    assert manifest["interface"]["defaultPrompt"] == [
        "Review this chemistry input before I run it.",
        "Inspect this calculation output and explain any failure.",
        "Plan a calculation and identify unresolved choices.",
    ]


def test_plugin_starts_the_installed_guided_command():
    mcp_config = json.loads(
        (PLUGIN_ROOT / ".mcp.json").read_text(encoding="utf-8")
    )

    assert mcp_config == {
        "mcpServers": {
            "chemtools": {
                "command": "chemtools",
                "args": ["--toolset", "guided"],
            }
        }
    }


def test_plugin_skills_have_complete_metadata_and_no_embedded_code():
    skill_directories = {
        path.name for path in (PLUGIN_ROOT / "skills").iterdir() if path.is_dir()
    }

    assert skill_directories == EXPECTED_SKILLS
    for skill_name in EXPECTED_SKILLS:
        skill_root = PLUGIN_ROOT / "skills" / skill_name
        skill_text = (skill_root / "SKILL.md").read_text(encoding="utf-8")
        agent_metadata = yaml.safe_load(
            (skill_root / "agents" / "openai.yaml").read_text(encoding="utf-8")
        )

        assert skill_text.startswith(f"---\nname: {skill_name}\n")
        assert "[TODO:" not in skill_text
        assert not (skill_root / "scripts").exists()
        assert f"${skill_name}" in agent_metadata["interface"]["default_prompt"]


def test_prompt_contract_covers_routing_and_approval_boundaries():
    contract = yaml.safe_load(
        (PLUGIN_ROOT / "evals" / "prompt-routing.yaml").read_text(
            encoding="utf-8"
        )
    )
    cases = contract["cases"]

    assert contract["schema_version"] == "chemtools.plugin-prompt-routing/1"
    assert Counter(case["kind"] for case in cases) == {
        "direct": 4,
        "indirect": 4,
        "follow_up": 2,
        "unsupported": 3,
        "approval": 2,
    }
    assert all(
        case["expected_skill"] is None
        or case["expected_skill"] in EXPECTED_SKILLS
        for case in cases
    )
    assert all(
        case["expected_tool"] is None
        or case["expected_tool"] in GUIDED_TOOLS
        for case in cases
    )
    assert all(
        set(case["forbidden_tools"]) <= GUIDED_TOOLS for case in cases
    )

    approval_cases = {
        case["id"]: case for case in cases if case["kind"] == "approval"
    }
    assert approval_cases["approval_prepare"]["expected_behavior"] == (
        "omit_approval_token"
    )
    assert approval_cases["approval_submit"]["requires_prior_result"] == (
        "launch_preparation"
    )
    assert approval_cases["approval_submit"]["expected_behavior"] == (
        "reuse_exact_approval_token"
    )

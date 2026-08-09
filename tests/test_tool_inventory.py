"""Exact regression checks for generated MCP tool and capability counts."""

import json
from pathlib import Path

from chemtools.mcp.inventory import (
    build_inventory,
    check_documents,
    render_json,
    render_markdown,
)

ROOT = Path(__file__).parents[1]


def test_inventory_pins_live_registry_totals():
    summary = build_inventory()["summary"]

    assert summary["tool_count"] == 322
    assert summary["alias_count"] == 15
    assert summary["canonical_tool_count"] == 313
    assert summary["advertised_legacy_tool_count"] == 9
    assert summary["hidden_alias_count"] == 15
    assert summary["total_callable_name_count"] == 337
    assert summary["entrypoint_alias_count"] == 2
    assert summary["python_import_shim_count"] == 6
    assert summary["by_program"] == {
        "generic": 66,
        "nwchem": 101,
        "molcas": 41,
        "dirac": 35,
        "grasp": 49,
        "qe": 18,
        "qmcpack": 12,
        "orca": 0,
    }
    assert summary["by_capability"] == {
        "none": 266,
        "registry": 18,
        "runner_profile": 2,
        "executable_or_scheduler": 5,
        "executable": 28,
        "scheduler": 3,
    }
    assert summary["by_mode"] == {
        "analysis": 284,
        "local": 319,
        "hpc": 322,
    }
    assert summary["by_program_filter"] == {
        "nwchem": {"analysis": 149, "local": 164, "hpc": 167},
        "molcas": {"analysis": 101, "local": 106, "hpc": 107},
        "dirac": {"analysis": 96, "local": 100, "hpc": 101},
        "grasp": {"analysis": 91, "local": 114, "hpc": 115},
        "qe": {"analysis": 79, "local": 83, "hpc": 84},
        "qmcpack": {"analysis": 73, "local": 77, "hpc": 78},
        "orca": {"analysis": 61, "local": 65, "hpc": 66},
    }


def test_inventory_records_schema_and_owner_for_every_tool():
    inventory = build_inventory()

    assert inventory["schema"] == "chemtools.mcp-tool-inventory/3"
    assert inventory["server"] == {
        "name": "chemtools",
        "version": "0.2.0.dev0",
        "protocol_version": "2025-11-25",
        "supported_protocol_versions": [
            "2024-11-05",
            "2025-03-26",
            "2025-06-18",
            "2025-11-25",
            "2026-07-28",
        ],
    }
    assert len(inventory["tools"]) == 322
    assert all(tool["program"] for tool in inventory["tools"])
    assert all(tool["capability"] for tool in inventory["tools"])
    assert all(isinstance(tool["input_schema"], dict) for tool in inventory["tools"])
    guided = {
        "review_input",
        "inspect_run",
        "compare_runs",
        "plan_recovery",
        "plan_calculation",
        "launch_run",
        "monitor_run",
        "draft_input",
        "visualize",
        "search_knowledge",
        "find_reference_case",
    }
    assert {
        tool["name"]
        for tool in inventory["tools"]
        if tool["output_schema"] is not None
    } == guided
    assert all(tool["visible_modes"] for tool in inventory["tools"])
    assert {
        tool["lifecycle"] for tool in inventory["tools"]
    } == {"canonical", "advertised_legacy"}


def test_inventory_separates_compatibility_surfaces():
    inventory = build_inventory()

    assert [item["name"] for item in inventory["advertised_legacy_tools"]] == [
        "register_nwchem_run",
        "update_nwchem_run_status",
        "list_nwchem_runs",
        "get_nwchem_run_summary",
        "create_nwchem_campaign",
        "get_nwchem_campaign_status",
        "get_nwchem_campaign_energies",
        "create_nwchem_workflow",
        "advance_nwchem_workflow",
    ]
    assert [item["name"] for item in inventory["entrypoint_aliases"]] == [
        "chemtools-nwchem",
        "chemtools-nwchem-docs",
    ]
    assert [item["name"] for item in inventory["python_import_shims"]] == [
        "chemtools",
        "chemtools.api",
        "chemtools.api_input",
        "chemtools.api_strategy",
        "chemtools.mcp.nwchem",
        "chemtools.execution.executors",
    ]
    assert {
        item["state"] for item in inventory["python_import_shims"]
    } == {"compatibility_deprecated"}
    compatibility_items = (
        inventory["aliases"]
        + inventory["advertised_legacy_tools"]
        + inventory["entrypoint_aliases"]
        + inventory["python_import_shims"]
    )
    assert {
        item["deprecated_since"] for item in compatibility_items
    } == {"0.1.0"}
    assert all(item["remove_after"] is None for item in compatibility_items)
    assert all(
        item["contract_status"] == "unverified"
        for item in inventory["aliases"]
    )
    assert all(item["input_schema"] is None for item in inventory["aliases"])
    assert all(item["effects"] is None for item in inventory["aliases"])
    assert all(item["availability"] for item in inventory["aliases"])
    assert all(item["argument_adapter"] for item in inventory["aliases"])
    assert all(item["reason"] for item in inventory["aliases"])


def test_inventory_rendering_is_deterministic():
    first = build_inventory()
    second = build_inventory()

    assert render_json(first) == render_json(second)
    assert render_markdown(first) == render_markdown(second)
    assert (
        "Counts include the 66 generic tools"
        in render_markdown(first)
    )


def test_committed_inventory_documents_are_current():
    assert check_documents(build_inventory()) == []


def test_readme_counts_match_the_live_inventory():
    summary = build_inventory()["summary"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert (
        f"Plus {summary['by_program']['generic']} program-generic tools"
        in readme
    )
    assert f"Total: {summary['tool_count']} MCP tool definitions." in readme
    expected_mode_rows = {
        "analysis": (
            "| `analysis` (default if no `CHEMTOOLS_RUNNER_PROFILES`) "
            f"| {summary['by_mode']['analysis']} |"
        ),
        "local": f"| `local` | {summary['by_mode']['local']} |",
        "hpc": f"| `hpc` | {summary['by_mode']['hpc']} |",
    }
    for row in expected_mode_rows.values():
        assert row in readme
    assert "The minimum config exposes the eleven guided tools." in readme


def test_maintained_setup_examples_are_portable_and_guided():
    setup_paths = [
        ROOT / "README.md",
        ROOT / "docs" / "getting-started.md",
        ROOT / "docs" / "execution-smoke-tests.md",
        ROOT / "examples" / "local_workstation" / "CLAUDE.md",
        ROOT / "examples" / "local_workstation" / "runner_profiles.yaml",
        ROOT / "examples" / "tacc_stampede3" / ".mcp.json",
        ROOT / "examples" / "tacc_stampede3" / "CLAUDE.md",
        ROOT / "examples" / "tacc_stampede3" / "runner_profiles.yaml",
    ]
    setup_text = "\n".join(
        path.read_text(encoding="utf-8") for path in setup_paths
    )

    for personal_path in (
        "/home/charlie/",
        "/home1/01775/charlesp/",
        "/Users/charlie/",
    ):
        assert personal_path not in setup_text

    mcp_config = json.loads(
        (ROOT / "examples" / "tacc_stampede3" / ".mcp.json").read_text(
            encoding="utf-8"
        )
    )
    assert mcp_config == {
        "mcpServers": {
            "chemtools": {
                "command": "chemtools",
                "env": {
                    "CHEMTOOLS_RUNNER_PROFILES": "/path/to/runner_profiles.yaml"
                },
            }
        }
    }

    for project_file in (
        ROOT / "examples" / "local_workstation" / "CLAUDE.md",
        ROOT / "examples" / "tacc_stampede3" / "CLAUDE.md",
    ):
        project_text = project_file.read_text(encoding="utf-8")
        assert "review_input" in project_text
        assert "launch_run" in project_text
        assert "monitor_run" in project_text

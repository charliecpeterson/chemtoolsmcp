"""Exact regression checks for generated MCP tool and capability counts."""

from chemtools.mcp.inventory import (
    build_inventory,
    check_documents,
    render_json,
    render_markdown,
)


def test_inventory_pins_live_registry_totals():
    summary = build_inventory()["summary"]

    assert summary["tool_count"] == 326
    assert summary["alias_count"] == 13
    assert summary["by_program"] == {
        "generic": 56,
        "nwchem": 101,
        "molcas": 45,
        "dirac": 39,
        "grasp": 51,
        "qe": 20,
        "qmcpack": 14,
    }
    assert summary["by_capability"] == {
        "none": 254,
        "registry": 18,
        "runner_profile": 4,
        "executable_or_scheduler": 5,
        "executable": 42,
        "scheduler": 3,
    }
    assert summary["by_mode"] == {
        "analysis": 272,
        "local": 323,
        "hpc": 326,
    }
    assert summary["by_program_filter"] == {
        "nwchem": {"analysis": 139, "local": 154, "hpc": 157},
        "molcas": {"analysis": 91, "local": 100, "hpc": 101},
        "dirac": {"analysis": 86, "local": 94, "hpc": 95},
        "grasp": {"analysis": 79, "local": 106, "hpc": 107},
        "qe": {"analysis": 69, "local": 75, "hpc": 76},
        "qmcpack": {"analysis": 63, "local": 69, "hpc": 70},
    }


def test_inventory_records_schema_and_owner_for_every_tool():
    inventory = build_inventory()

    assert inventory["server"] == {
        "name": "chemtools-nwchem",
        "version": "0.1.0",
        "protocol_version": "2024-11-05",
        "supported_protocol_versions": ["2024-11-05"],
    }
    assert len(inventory["tools"]) == 326
    assert all(tool["program"] for tool in inventory["tools"])
    assert all(tool["capability"] for tool in inventory["tools"])
    assert all(isinstance(tool["input_schema"], dict) for tool in inventory["tools"])
    assert all(tool["visible_modes"] for tool in inventory["tools"])


def test_inventory_rendering_is_deterministic():
    first = build_inventory()
    second = build_inventory()

    assert render_json(first) == render_json(second)
    assert render_markdown(first) == render_markdown(second)
    assert (
        "Counts include the 56 generic tools"
        in render_markdown(first)
    )


def test_committed_inventory_documents_are_current():
    assert check_documents(build_inventory()) == []

"""Exact regression checks for generated MCP tool and capability counts."""

from chemtools.mcp.inventory import (
    build_inventory,
    check_documents,
    render_json,
    render_markdown,
)


def test_inventory_pins_live_registry_totals():
    summary = build_inventory()["summary"]

    assert summary["tool_count"] == 331
    assert summary["alias_count"] == 13
    assert summary["by_program"] == {
        "generic": 59,
        "nwchem": 101,
        "molcas": 45,
        "dirac": 39,
        "grasp": 53,
        "qe": 20,
        "qmcpack": 14,
    }
    assert summary["by_capability"] == {
        "none": 259,
        "registry": 18,
        "runner_profile": 4,
        "executable_or_scheduler": 5,
        "executable": 42,
        "scheduler": 3,
    }
    assert summary["by_mode"] == {
        "analysis": 277,
        "local": 328,
        "hpc": 331,
    }
    assert summary["by_program_filter"] == {
        "nwchem": {"analysis": 142, "local": 157, "hpc": 160},
        "molcas": {"analysis": 94, "local": 103, "hpc": 104},
        "dirac": {"analysis": 89, "local": 97, "hpc": 98},
        "grasp": {"analysis": 84, "local": 111, "hpc": 112},
        "qe": {"analysis": 72, "local": 78, "hpc": 79},
        "qmcpack": {"analysis": 66, "local": 72, "hpc": 73},
    }


def test_inventory_records_schema_and_owner_for_every_tool():
    inventory = build_inventory()

    assert inventory["server"] == {
        "name": "chemtools-nwchem",
        "version": "0.1.0",
        "protocol_version": "2024-11-05",
        "supported_protocol_versions": ["2024-11-05"],
    }
    assert len(inventory["tools"]) == 331
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
        "Counts include the 59 generic tools"
        in render_markdown(first)
    )


def test_committed_inventory_documents_are_current():
    assert check_documents(build_inventory()) == []

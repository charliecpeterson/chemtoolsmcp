"""Generate deterministic MCP tool metadata and human-readable count tables."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from chemtools.mcp.decorator import (
    SERVER_NAME,
    SERVER_VERSION,
    _TOOL_CAPABILITIES,
    _TOOL_PROGRAMS,
)
from chemtools.mcp.catalog import builtin_program_names
from chemtools.mcp.dispatch import _TOOL_ALIASES, tool_definitions
from chemtools.mcp.modes import MODE_CAPABILITIES, VALID_MODES, filter_tools
from chemtools.mcp.server import (
    DEFAULT_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
)

INVENTORY_SCHEMA = "chemtools.mcp-tool-inventory/1"
PROGRAM_ORDER = ("generic", *builtin_program_names())
CAPABILITY_ORDER = (
    "none",
    "registry",
    "runner_profile",
    "executable_or_scheduler",
    "executable",
    "scheduler",
)


def build_inventory() -> dict[str, Any]:
    definitions = sorted(tool_definitions(), key=lambda definition: definition["name"])
    tools = []
    for definition in definitions:
        name = definition["name"]
        capability = _TOOL_CAPABILITIES.get(name, "none")
        program = _TOOL_PROGRAMS.get(name, "generic")
        tools.append(
            {
                "name": name,
                "program": program,
                "capability": capability,
                "visible_modes": [
                    mode
                    for mode in VALID_MODES
                    if capability in MODE_CAPABILITIES[mode]
                ],
                "description": definition["description"],
                "input_schema": definition["inputSchema"],
            }
        )

    program_counts = _program_counts(tool["program"] for tool in tools)
    programs = _ordered_values(program_counts, PROGRAM_ORDER)
    capabilities = _ordered_values(
        Counter(tool["capability"] for tool in tools),
        CAPABILITY_ORDER,
    )
    modes = {
        mode: len(
            filter_tools(
                definitions,
                _TOOL_CAPABILITIES,
                mode,
                program_tags=_TOOL_PROGRAMS,
            )
        )
        for mode in VALID_MODES
    }
    mode_programs = {
        mode: _ordered_values(
            _program_counts(
                tool["program"]
                for tool in tools
                if mode in tool["visible_modes"]
            ),
            PROGRAM_ORDER,
        )
        for mode in VALID_MODES
    }
    program_filters = {
        program: {
            mode: len(
                filter_tools(
                    definitions,
                    _TOOL_CAPABILITIES,
                    mode,
                    programs={program},
                    program_tags=_TOOL_PROGRAMS,
                )
            )
            for mode in VALID_MODES
        }
        for program in programs
        if program != "generic"
    }
    aliases = [
        {"alias": alias, "target": target}
        for alias, (target, _translator) in sorted(_TOOL_ALIASES.items())
    ]
    return {
        "schema": INVENTORY_SCHEMA,
        "server": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
            "protocol_version": DEFAULT_PROTOCOL_VERSION,
            "supported_protocol_versions": list(SUPPORTED_PROTOCOL_VERSIONS),
        },
        "summary": {
            "tool_count": len(tools),
            "alias_count": len(aliases),
            "by_program": programs,
            "by_capability": capabilities,
            "by_mode": modes,
            "by_mode_and_program": mode_programs,
            "by_program_filter": program_filters,
        },
        "aliases": aliases,
        "tools": tools,
    }


def render_json(inventory: dict[str, Any]) -> str:
    return json.dumps(inventory, indent=2, ensure_ascii=False) + "\n"


def render_markdown(inventory: dict[str, Any]) -> str:
    summary = inventory["summary"]
    lines = [
        "# MCP tool inventory",
        "",
        "Generated from the live decorator registry and tool definitions.",
        "Do not edit counts by hand. Regenerate with:",
        "",
        "```bash",
        ".venv/bin/python scripts/generate_tool_inventory.py --write-docs",
        "```",
        "",
        "The JSON companion contains every tool description and input schema.",
        "",
        "## Summary",
        "",
        f"- Default protocol version: `{inventory['server']['protocol_version']}`",
        "- Supported protocol versions: "
        + ", ".join(
            f"`{version}`"
            for version in inventory["server"]["supported_protocol_versions"]
        ),
        f"- Public tool definitions: {summary['tool_count']}",
        f"- Compatibility aliases: {summary['alias_count']}",
        "",
        "### Programs",
        "",
        "| Program | Tools |",
        "| --- | ---: |",
    ]
    for program, count in summary["by_program"].items():
        lines.append(f"| `{program}` | {count} |")

    lines.extend(
        [
            "",
            "### Capabilities",
            "",
            "| Capability | Tools |",
            "| --- | ---: |",
        ]
    )
    for capability, count in summary["by_capability"].items():
        lines.append(f"| `{capability}` | {count} |")

    lines.extend(
        [
            "",
            "### Modes",
            "",
            "| Mode | All programs |",
            "| --- | ---: |",
        ]
    )
    for mode, count in summary["by_mode"].items():
        lines.append(f"| `{mode}` | {count} |")

    lines.extend(
        [
            "",
            "### Program filters",
            "",
            (
                f"Counts include the {summary['by_program']['generic']} generic "
                "tools where the active mode permits them."
            ),
            "",
            "| Program filter | Analysis | Local | HPC |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for program, counts in summary["by_program_filter"].items():
        lines.append(
            f"| `{program}` | {counts['analysis']} | "
            f"{counts['local']} | {counts['hpc']} |"
        )

    lines.extend(
        [
            "",
            "## Compatibility aliases",
            "",
            "Aliases remain callable but are omitted from `tools/list`.",
            "",
            "| Alias | Canonical tool |",
            "| --- | --- |",
        ]
    )
    for alias in inventory["aliases"]:
        lines.append(f"| `{alias['alias']}` | `{alias['target']}` |")

    lines.extend(
        [
            "",
            "## Tools",
            "",
            "| Tool | Program | Capability | Visible modes |",
            "| --- | --- | --- | --- |",
        ]
    )
    for tool in inventory["tools"]:
        modes = ", ".join(f"`{mode}`" for mode in tool["visible_modes"])
        lines.append(
            f"| `{tool['name']}` | `{tool['program']}` | "
            f"`{tool['capability']}` | {modes} |"
        )
    return "\n".join(lines) + "\n"


def default_document_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    return root / "docs" / "tool-inventory.json", root / "docs" / "tool-inventory.md"


def write_documents(inventory: dict[str, Any]) -> tuple[Path, Path]:
    json_path, markdown_path = default_document_paths()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(render_json(inventory), encoding="utf-8")
    markdown_path.write_text(render_markdown(inventory), encoding="utf-8")
    return json_path, markdown_path


def check_documents(inventory: dict[str, Any]) -> list[Path]:
    json_path, markdown_path = default_document_paths()
    expected = {
        json_path: render_json(inventory),
        markdown_path: render_markdown(inventory),
    }
    return [
        path
        for path, contents in expected.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != contents
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the live Chemtools MCP tool inventory."
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="stdout format when neither --write-docs nor --check is used",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--write-docs",
        action="store_true",
        help="write docs/tool-inventory.json and docs/tool-inventory.md",
    )
    actions.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed inventory documents are stale",
    )
    arguments = parser.parse_args(argv)
    inventory = build_inventory()

    if arguments.write_docs:
        for path in write_documents(inventory):
            print(path)
        return 0
    if arguments.check:
        stale = check_documents(inventory)
        if stale:
            for path in stale:
                print(f"stale: {path}")
            return 1
        print("tool inventory documents are current")
        return 0

    rendered = render_json(inventory) if arguments.format == "json" else render_markdown(inventory)
    print(rendered, end="")
    return 0


def _ordered_values(counts: Counter[str], preferred: tuple[str, ...]) -> dict[str, int]:
    ordered = [value for value in preferred if value in counts]
    ordered.extend(sorted(value for value in counts if value not in preferred))
    return {value: counts[value] for value in ordered}


def _program_counts(values: Iterable[str]) -> Counter[str]:
    counts = Counter({program: 0 for program in PROGRAM_ORDER})
    counts.update(values)
    return counts

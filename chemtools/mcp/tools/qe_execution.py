"""Quantum ESPRESSO pw.x launch tools."""

from __future__ import annotations

from typing import Any

from chemtools.application.qe_execution import (
    launch_qe_with_service,
    render_qe_launch,
)
from chemtools.mcp.decorator import _tool, get_execution_service


@_tool("render_qe_launch", needs="runner_profile", program="qe")
def _handle_render_qe_launch(arguments: dict[str, Any]) -> dict[str, Any]:
    preview, _ = render_qe_launch(
        input_path=arguments["qe_input"],
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path"),
        job_name=arguments.get("job_name"),
        resource_overrides=arguments.get("resource_overrides"),
        env_overrides=arguments.get("env_overrides"),
    )
    return preview


@_tool("launch_qe_run", needs="executable", program="qe")
def _handle_launch_qe_run(arguments: dict[str, Any]) -> dict[str, Any]:
    return launch_qe_with_service(
        get_execution_service(),
        input_path=arguments["qe_input"],
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path"),
        job_name=arguments.get("job_name"),
        resource_overrides=arguments.get("resource_overrides"),
        env_overrides=arguments.get("env_overrides"),
        dry_run=arguments.get("dry_run", False),
    )


def qe_execution_tool_definitions() -> list[dict[str, Any]]:
    properties = {
        "qe_input": {
            "type": "string",
            "description": "Path to a Quantum ESPRESSO pw.x input file.",
        },
        "profile": {
            "type": "string",
            "description": "Named Chemtools runner profile.",
        },
        "profiles_path": {
            "type": "string",
            "description": "Optional runner-profile YAML or JSON path.",
        },
        "job_name": {
            "type": "string",
            "description": "Optional output-file stem.",
        },
        "resource_overrides": {
            "type": "object",
            "description": "Optional profile resource overrides.",
        },
        "env_overrides": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Optional environment overrides for this launch.",
        },
    }
    return [{
        "name": "render_qe_launch",
        "description": (
            "Render the configured QE pw.x local or scheduler command without "
            "starting it. Use this to verify the selected runner profile, "
            "resources, environment, and expected output paths."
        ),
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": ["qe_input", "profile"],
            "additionalProperties": False,
        },
    }, {
        "name": "launch_qe_run",
        "description": (
            "Launch a reviewed QE pw.x input through a configured runner "
            "profile. A dry run renders the command only; an enabled execution "
            "service is required for a live launch."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                **properties,
                "dry_run": {
                    "type": "boolean",
                    "default": False,
                    "description": "Render only; do not start QE.",
                },
            },
            "required": ["qe_input", "profile"],
            "additionalProperties": False,
        },
    }]


__all__ = [
    "_handle_launch_qe_run",
    "_handle_render_qe_launch",
    "qe_execution_tool_definitions",
]

"""Command-line behavior that must also work from an installed wheel."""

import json
import sys

import yaml

from chemtools.mcp import cli
from chemtools.mcp.cli import _build_arg_parser, _profile_example_text


def test_cli_prints_the_portable_local_profile():
    profile = json.loads(_profile_example_text("local"))

    assert profile["profiles"]["workstation_serial"]["programs"]["nwchem"] == {
        "executable_argv": ["/path/to/nwchem"],
    }


def test_cli_prints_the_portable_slurm_profile():
    profile = yaml.safe_load(_profile_example_text("slurm"))

    assert profile["profiles"]["slurm"]["programs"]["nwchem"] == {
        "launcher_argv": ["srun"],
        "executable_argv": ["nwchem"],
    }


def test_profile_example_option_is_part_of_the_shared_cli():
    arguments = _build_arg_parser().parse_args([
        "--print-profile-example",
        "slurm",
    ])

    assert arguments.print_profile_example == "slurm"


def test_cli_passes_one_resolved_state_to_the_server(monkeypatch):
    captured = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chemtools",
            "--mode",
            "local",
            "--programs",
            "nwchem",
            "--toolset",
            "guided",
        ],
    )
    monkeypatch.setattr(cli, "serve", captured.append)

    cli.main()

    assert len(captured) == 1
    state = captured[0]
    assert state.mode == "local"
    assert state.programs == frozenset({"nwchem"})
    assert state.toolset == frozenset({
        "review_input",
        "inspect_run",
        "compare_runs",
        "plan_recovery",
        "plan_calculation",
        "launch_run",
        "monitor_run",
        "draft_input",
        "search_knowledge",
        "find_reference_case",
        "visualize",
    })
    assert state.execution_service.enable_execution is True


def test_cli_defaults_to_the_guided_toolset(monkeypatch):
    captured = []
    monkeypatch.setattr(sys, "argv", ["chemtools", "--mode", "analysis"])
    monkeypatch.setattr(cli, "serve", captured.append)

    cli.main()

    assert len(captured) == 1
    assert captured[0].toolset == cli._modes.TOOLSETS["guided"]


def test_cli_developer_toolset_keeps_the_complete_surface(monkeypatch):
    captured = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["chemtools", "--mode", "analysis", "--toolset", "developer"],
    )
    monkeypatch.setattr(cli, "serve", captured.append)

    cli.main()

    assert len(captured) == 1
    assert captured[0].toolset is None


def test_legacy_command_names_its_replacement(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "main", lambda *, prog: calls.append(prog))

    cli.main_legacy_nwchem()

    assert calls == ["chemtools-nwchem"]
    assert capsys.readouterr().err == (
        "chemtools-nwchem: deprecated compatibility command; use 'chemtools'. "
        "Update your MCP configs before the compatibility command is removed.\n"
    )

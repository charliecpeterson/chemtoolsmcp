"""Install a built wheel behind an isolated user boundary and smoke-test it.

The check runs outside the repository in a virtual environment or clean user
site, then verifies MCP transport, package data, and one NWChem inspection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "nwchem_pyscf"
    / "h2o_rhf_sto3g.out"
)
GUIDED_TOOLS = (
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
)

_INSTALLED_CHECK = """
import importlib.util
from importlib.metadata import version as package_version
import json
from pathlib import Path
import sys

import chemtools
import numpy
import yaml
from chemtools import api_input, api_strategy
from chemtools.application.reference_case_search import find_reference_cases
from chemtools.core.artifact_registry import (
    record_run_artifacts as compatible_record_run_artifacts,
)
from chemtools.core.registry_db import connect_registry as compatible_connect_registry
from chemtools.core.run_records import register_run as compatible_register_run
from chemtools.execution.launch_registry import (
    create_launch_record as compatible_create_launch_record,
)
from chemtools.execution.legacy_archive import (
    archive_paths,
    archive_previous_outputs,
)
from chemtools.execution.legacy_runner import (
    archive_paths as compatible_archive_paths,
    archive_previous_outputs as compatible_archive_previous_outputs,
    get_local_resource_budget as compatible_get_local_resource_budget,
    query_partition_specs as compatible_query_partition_specs,
)
from chemtools.execution.legacy_status import inspect_run_status
from chemtools.execution.profiles import (
    DEFAULT_RUNNER_PROFILES,
    load_runner_profiles,
)
from chemtools.execution.resource_inspection import (
    get_local_resource_budget,
    query_partition_specs,
)
from chemtools.knowledge.cards import load_knowledge_cards
from chemtools.mcp.dispatch import dispatch_tool
from chemtools.mcp.decorator import SERVER_VERSION
from chemtools.persistence.artifacts import record_run_artifacts
from chemtools.persistence.launches import create_launch_record
from chemtools.persistence.runs import register_run
from chemtools.persistence.sqlite import connect_registry
from chemtools.programs.molcas.docs import list_docs as list_molcas_docs
from chemtools.programs.molcas.input.basis_library import list_basis_sets
from chemtools.programs.nwchem._plugin_examples import NWCHEM_EXAMPLES
from chemtools.programs.nwchem.docs import list_docs as list_nwchem_docs
from chemtools.programs.nwchem.input.general import create_nwchem_input
from chemtools.programs.nwchem.input.lint_restart import lint_nwchem_input
from chemtools.programs.nwchem.legacy_status import inspect_nwchem_run_status
from chemtools.programs.nwchem.input.basis_library import (
    bundled_basis_library_path,
)
from chemtools.programs.nwchem.strategy.case_review import (
    check_spin_charge_state,
)
from chemtools.programs.nwchem.strategy.hpc_resources import (
    suggest_hpc_resources,
)

fixture = Path(sys.argv[1]).resolve()
repository = Path(sys.argv[2]).resolve()
extras = json.loads(sys.argv[3])
expected_install_root = Path(sys.argv[4]).resolve()
install_layout = sys.argv[5]
package_path = Path(chemtools.__file__).resolve()
slurm_profile_path = package_path.parent / "runner_profiles.slurm.example.yaml"
try:
    package_path.relative_to(repository)
except ValueError:
    pass
else:
    raise AssertionError(f"import resolved to the checkout: {package_path}")
for installed_path in (
    package_path,
    Path(numpy.__file__).resolve(),
    Path(yaml.__file__).resolve(),
):
    try:
        installed_path.relative_to(expected_install_root)
    except ValueError as error:
        raise AssertionError(
            f"dependency escaped isolated install: {installed_path}"
        ) from error

profiles = load_runner_profiles()
slurm_profiles = load_runner_profiles(str(slurm_profile_path))
cards = load_knowledge_cards()
molcas_basis_sets = list_basis_sets()
molcas_documents = list_molcas_docs()
nwchem_documents = list_nwchem_docs()
examples = NWCHEM_EXAMPLES.list_examples()
inspection = dispatch_tool(
    "inspect_run",
    {"output_file": str(fixture), "program": "nwchem"},
)
generic_legacy_status = inspect_run_status(output_path=str(fixture))
nwchem_legacy_status = inspect_nwchem_run_status(output_path=str(fixture))
reference_cases = find_reference_cases(
    "open-shell fragment guess",
    program="nwchem",
    scientific_status="exploratory",
)

assert DEFAULT_RUNNER_PROFILES.is_file()
assert profiles["__source__"] == str(DEFAULT_RUNNER_PROFILES.resolve())
assert slurm_profiles["profiles"]["slurm"]["programs"]["nwchem"] == {
    "launcher_argv": ["srun"],
    "executable_argv": ["nwchem"],
}
assert cards
assert len(molcas_basis_sets) == 71
assert len(molcas_documents) == 133
assert len(nwchem_documents) == 29
assert bundled_basis_library_path().is_dir()
assert api_input.create_nwchem_input is create_nwchem_input
assert api_input.lint_nwchem_input is lint_nwchem_input
assert api_strategy.check_spin_charge_state is check_spin_charge_state
assert api_strategy.suggest_hpc_resources is suggest_hpc_resources
assert chemtools.create_nwchem_input is create_nwchem_input
assert chemtools.suggest_hpc_resources is suggest_hpc_resources
assert compatible_connect_registry is connect_registry
assert compatible_register_run is register_run
assert compatible_record_run_artifacts is record_run_artifacts
assert compatible_create_launch_record is create_launch_record
assert compatible_archive_paths is archive_paths
assert compatible_archive_previous_outputs is archive_previous_outputs
assert compatible_get_local_resource_budget is get_local_resource_budget
assert compatible_query_partition_specs is query_partition_specs
assert package_version("chemtools-mcp") == SERVER_VERSION
assert importlib.util.find_spec("chemtools.mcp.tools._nwchem_base") is None
assert importlib.util.find_spec("chemtools.mcp.tools.nwchem") is None
assert importlib.util.find_spec("chemtools.execution.legacy_profiles") is None
assert "chemtools.mcp.tools._nwchem_base" not in sys.modules
assert "chemtools.mcp.tools.nwchem" not in sys.modules
assert "chemtools.execution.legacy_profiles" not in sys.modules
assert examples
assert NWCHEM_EXAMPLES.read_example(examples[0]["name"]).strip()
assert inspection.get("error") is None
assert inspection["schema_version"] == "chemtools.inspect-run/1"
assert inspection["program"]["name"] == "nwchem"
assert inspection["assessment"]["verdict"]["label"] == "success"
assert len(inspection["evidence"]["tasks"]) == 1
assert generic_legacy_status["overall_status"] == "output_present_unknown"
assert generic_legacy_status["progress_summary"] is None
assert nwchem_legacy_status["overall_status"] == "completed_success"
assert nwchem_legacy_status["output_summary"]["task_count"] == 1
assert inspection["next_actions"][0]["action"] == (
    inspection["next_actions"][0]["tool"]
)
assert [case["case_id"] for case in reference_cases["matches"]] == [
    "nwchem.fecn6_lowspin_fragment",
    "nwchem.hexaaquairon_swap_chain",
]

h5py_version = None
if "dirac" in extras:
    import h5py

    from chemtools.programs.dirac.binary.h5 import read_metadata

    checkpoint = Path.cwd() / "dirac-smoke.h5"
    with h5py.File(checkpoint, "w") as document:
        document.attrs["DIRAC_VERSION"] = "25.0"
        molecule = document.create_group("input/molecule")
        molecule.create_dataset("n_atoms", data=[1])
    assert read_metadata(str(checkpoint)) == {
        "path": str(checkpoint),
        "version": "25.0",
        "n_atoms": 1,
        "scf_energy_hartree": None,
    }
    h5py_version = h5py.__version__
    assert Path(h5py.__file__).resolve().is_relative_to(expected_install_root)
else:
    assert importlib.util.find_spec("h5py") is None

print(json.dumps({
    "package_path": str(package_path),
    "package_version": SERVER_VERSION,
    "compatibility_aliases": True,
    "persistence_owners": True,
    "focused_nwchem_provider": True,
    "default_profile_count": len(profiles["profiles"]),
    "portable_slurm_profile": slurm_profiles["profiles"]["slurm"]["description"],
    "knowledge_card_count": len(cards),
    "molcas_basis_set_count": len(molcas_basis_sets),
    "molcas_document_count": len(molcas_documents),
    "nwchem_document_count": len(nwchem_documents),
    "nwchem_example_count": len(examples),
    "inspection_verdict": inspection["assessment"]["verdict"]["label"],
    "legacy_status_boundary": True,
    "inspection_next_action": inspection["next_actions"][0]["action"],
    "reference_case_count": reference_cases["match_count"],
    "installed_extras": extras,
    "install_layout": install_layout,
    "numpy_path": str(Path(numpy.__file__).resolve()),
    "yaml_path": str(Path(yaml.__file__).resolve()),
    "h5py_version": h5py_version,
}, sort_keys=True))
"""


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        input=input_text,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        rendered = " ".join(command[:4])
        raise RuntimeError(
            f"command failed ({completed.returncode}): {rendered}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


_MCP_EXCHANGE_CHECK = """
import anyio
import json
import os
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


chemtools = sys.argv[1]
fixture = Path(sys.argv[2]).resolve()
launch_input = Path(sys.argv[3]).resolve()


def payload(result):
    assert result.is_error is False
    assert result.content
    assert result.content[0].type == "text"
    parsed = json.loads(result.content[0].text)
    assert result.structured_content == parsed
    return parsed


async def check():
    parameters = StdioServerParameters(
        command=chemtools,
        args=["--mode", "analysis"],
        env=dict(os.environ),
        cwd=Path.cwd(),
    )
    async with stdio_client(parameters) as streams:
        async with ClientSession(*streams) as client:
            initialized = await client.initialize()
            listed = await client.list_tools()
            tool_names = tuple(tool.name for tool in listed.tools)
            assert tool_names == (
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
            )

            review = payload(await client.call_tool(
                "review_input",
                {
                    "input_file": str(launch_input),
                    "program": "nwchem",
                },
            ))
            inspection = payload(await client.call_tool(
                "inspect_run",
                {
                    "output_file": str(fixture),
                    "program": "nwchem",
                },
            ))
            comparison = payload(await client.call_tool(
                "compare_runs",
                {
                    "reference_output_file": str(fixture),
                    "candidate_output_file": str(fixture),
                    "program": "nwchem",
                },
            ))
            recovery = payload(await client.call_tool(
                "plan_recovery",
                {
                    "output_file": str(fixture),
                    "program": "nwchem",
                },
            ))
            planning = payload(await client.call_tool(
                "plan_calculation",
                {
                    "program": "nwchem",
                    "system": "UO2",
                    "elements": ["U", "O"],
                    "charge": 0,
                    "multiplicity": 3,
                    "stages": ["optimize", "frequency"],
                },
            ))
            draft = payload(await client.call_tool(
                "draft_input",
                {
                    "program": "nwchem",
                    "atoms": [
                        {"element": "H", "x": 0.0, "y": 0.0, "z": 0.0},
                        {"element": "H", "x": 0.0, "y": 0.0, "z": 0.74},
                    ],
                    "charge": 0,
                    "multiplicity": 1,
                    "method": "scf",
                    "basis": "sto-3g",
                    "task": "energy",
                },
            ))
            knowledge = payload(await client.call_tool(
                "search_knowledge",
                {"query": "failure sentinel"},
            ))
            reference_cases = payload(await client.call_tool(
                "find_reference_case",
                {
                    "query": "open-shell fragment guess",
                    "program": "nwchem",
                    "scientific_status": "exploratory",
                },
            ))
            launch_preparation = payload(await client.call_tool(
                "launch_run",
                {
                    "program": "nwchem",
                    "input_file": str(launch_input),
                    "profile": "local_mpirun",
                    "resources": {"mpi_ranks": 2},
                },
            ))
            unknown_monitor = payload(await client.call_tool(
                "monitor_run",
                {
                    "launch_id": (
                        "7c9a2d1e-0000-4000-8000-000000000000"
                    ),
                },
            ))
            visualize = payload(await client.call_tool(
                "visualize",
                {"path": str(fixture)},
            ))

    assert initialized.protocol_version == "2025-11-25"
    assert initialized.server_info.name == "chemtools"
    assert review["schema_version"] == "chemtools.review-input/1"
    assert inspection["program"]["name"] == "nwchem"
    assert inspection["source"]["path"] == str(fixture)
    assert inspection["assessment"]["verdict"]["label"] == "success"
    assert inspection["next_actions"][0]["action"] == (
        inspection["next_actions"][0]["tool"]
    )
    assert comparison["schema_version"] == "chemtools.compare-runs/1"
    assert recovery["schema_version"] == "chemtools.plan-recovery/1"
    assert planning["program"] == {"name": "nwchem"}
    assert planning["assessment"]["verdict"]["label"] == (
        "needs_scientific_decisions"
    )
    assert [
        stage["kind"] for stage in planning["evidence"]["stages"]
    ] == ["optimize", "frequency"]
    assert draft["schema_version"] == "chemtools.draft-input/1"
    assert knowledge["returned_count"] == 1
    assert knowledge["cards"][0]["id"] == (
        "cross_program.optimizer_failure_sentinel_must_lose"
    )
    assert [case["case_id"] for case in reference_cases["matches"]] == [
        "nwchem.fecn6_lowspin_fragment",
        "nwchem.hexaaquairon_swap_chain",
    ]
    assert launch_preparation["status"] == "awaiting_approval"
    assert launch_preparation["evidence"]["plan"]["argv"] == [
        "mpirun",
        "-np",
        "2",
        "nwchem",
        launch_input.name,
    ]
    assert launch_preparation["approval"]["token"].startswith("sha256:")
    assert not launch_input.with_suffix(".out").exists()
    assert not launch_input.with_suffix(".err").exists()
    assert unknown_monitor["error"] == "launch_not_owned"
    assert visualize["status"] == "unavailable"
    assert visualize["error"] == "orbitron_unavailable"
    return {
        "protocol_version": initialized.protocol_version,
        "server_name": initialized.server_info.name,
        "tool_count": len(tool_names),
        "inspection_verdict": inspection["assessment"]["verdict"]["label"],
        "inspection_next_action": inspection["next_actions"][0]["action"],
        "planning_verdict": planning["assessment"]["verdict"]["label"],
        "knowledge_card_id": knowledge["cards"][0]["id"],
        "reference_case_count": reference_cases["match_count"],
        "launch_preparation_status": launch_preparation["status"],
        "unknown_monitor_error": unknown_monitor["error"],
        "visualize_status": visualize["status"],
    }


print(json.dumps(anyio.run(check), sort_keys=True))
"""


def _check_mcp_exchange(
    python: Path,
    chemtools: Path,
    *,
    fixture: Path,
    launch_input: Path,
    workspace: Path,
    environment: dict[str, str],
) -> dict:
    completed = _run(
        [
            str(python),
            "-c",
            _MCP_EXCHANGE_CHECK,
            str(chemtools),
            str(fixture),
            str(launch_input),
        ],
        cwd=workspace,
        environment=environment,
    )
    return json.loads(completed.stdout)



def check_wheel(
    wheel: Path,
    *,
    fixture: Path,
    wheelhouse: Path | None,
    work_root: Path | None,
    extras: tuple[str, ...],
    install_layout: str,
) -> dict:
    with tempfile.TemporaryDirectory(
        prefix="chemtools-wheel-smoke-",
        dir=work_root,
    ) as temporary_directory:
        workspace = Path(temporary_directory)
        home = workspace / "home"
        userbase = workspace / "userbase"
        config_home = home / ".config"
        home.mkdir()
        userbase.mkdir()
        config_home.mkdir()
        clean_environment = dict(os.environ)
        for name in tuple(clean_environment):
            if name.startswith("CHEMTOOLS_"):
                clean_environment.pop(name)
        for name in (
            "PYTHONHOME",
            "PYTHONNOUSERSITE",
            "PYTHONPATH",
            "PYTHONUSERBASE",
            "VIRTUAL_ENV",
        ):
            clean_environment.pop(name, None)
        clean_environment.update({
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(config_home),
            "CHEMTOOLS_ORBITRON_CLI": str(workspace / "missing-orbitron"),
            "PIP_CACHE_DIR": str(workspace / "pip-cache"),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
        })

        if install_layout == "venv":
            environment = workspace / "venv"
            _run(
                [sys.executable, "-m", "venv", str(environment)],
                cwd=workspace,
                environment=clean_environment,
            )
            python = environment / "bin" / "python"
            install_root = environment
            chemtools = environment / "bin" / "chemtools"
            install_command = [str(python), "-m", "pip", "install"]
            check_command = [str(python), "-m", "pip", "check"]
        else:
            python = Path(
                getattr(sys, "_base_executable", sys.executable)
            ).resolve()
            install_root = userbase
            chemtools = userbase / "bin" / "chemtools"
            clean_environment["PYTHONUSERBASE"] = str(userbase)
            pip_for_base = [
                sys.executable,
                "-m",
                "pip",
                "--python",
                str(python),
            ]
            install_command = [*pip_for_base, "install"]
            check_command = [*pip_for_base, "check"]

        if install_layout == "user":
            install_command.extend(["--user", "--ignore-installed"])
        if wheelhouse is not None:
            install_command.extend([
                "--no-index",
                "--find-links",
                str(wheelhouse),
            ])
        install_target = str(wheel)
        if extras:
            install_target += f"[{','.join(extras)}]"
        install_command.append(install_target)
        _run(
            install_command,
            cwd=workspace,
            environment=clean_environment,
        )
        _run(
            check_command,
            cwd=workspace,
            environment=clean_environment,
        )

        listed = _run(
            [
                str(chemtools),
                "--mode",
                "analysis",
                "--list-tools",
            ],
            cwd=workspace,
            environment=clean_environment,
        )
        tool_names = tuple(
            line.strip()
            for line in listed.stdout.splitlines()
            if line.strip()
        )
        if tool_names != GUIDED_TOOLS:
            raise AssertionError(
                f"guided tool surface changed: {tool_names!r}"
            )

        printed_slurm = _run(
            [str(chemtools), "--print-profile-example", "slurm"],
            cwd=workspace,
            environment=clean_environment,
        )
        if "/home/" in printed_slurm.stdout or "/Users/" in printed_slurm.stdout:
            raise AssertionError("installed Slurm example contains a user path")

        launch_input = workspace / "water.nw"
        launch_input.write_text(
            "start water_review\n"
            "geometry units angstroms\n"
            "  O 0.0 0.0 0.0\n"
            "  H 0.0 0.0 1.0\n"
            "  H 0.0 1.0 0.0\n"
            "end\n"
            "basis\n"
            "  * library sto-3g\n"
            "end\n"
            "scf\n"
            "  singlet\n"
            "  thresh 1.0e-8\n"
            "end\n"
            "task scf energy\n",
            encoding="utf-8",
        )
        mcp_exchange = _check_mcp_exchange(
            python,
            chemtools,
            fixture=fixture,
            launch_input=launch_input,
            workspace=workspace,
            environment=clean_environment,
        )

        installed = _run(
            [
                str(python),
                "-c",
                _INSTALLED_CHECK,
                str(fixture),
                str(ROOT),
                json.dumps(extras),
                str(install_root),
                install_layout,
            ],
            cwd=workspace,
            environment=clean_environment,
        )
        report = json.loads(installed.stdout)
        report["guided_tools"] = list(tool_names)
        report["mcp_exchange"] = mcp_exchange
        report["wheel"] = str(wheel)
        report["isolated_home"] = str(home)
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test a chemtools wheel outside the checkout and current "
            "user configuration."
        )
    )
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        help="Install dependencies offline from this directory.",
    )
    parser.add_argument(
        "--install-layout",
        choices=("venv", "user"),
        default="venv",
        help="Install into a fresh virtual environment or isolated user site.",
    )
    parser.add_argument(
        "--extra",
        action="append",
        choices=("dirac",),
        default=[],
        dest="extras",
        help="Install and verify this optional package feature.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help="Representative NWChem output to inspect.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        help="Parent directory for the temporary installation workspace.",
    )
    arguments = parser.parse_args(argv)

    wheel = arguments.wheel.expanduser().resolve()
    fixture = arguments.fixture.expanduser().resolve()
    wheelhouse = (
        arguments.wheelhouse.expanduser().resolve()
        if arguments.wheelhouse is not None
        else None
    )
    work_root = (
        arguments.work_root.expanduser().resolve()
        if arguments.work_root is not None
        else None
    )
    if not wheel.is_file() or wheel.suffix != ".whl":
        parser.error(f"wheel is not a readable .whl file: {wheel}")
    if not fixture.is_file():
        parser.error(f"fixture is not a readable file: {fixture}")
    if wheelhouse is not None and not wheelhouse.is_dir():
        parser.error(f"wheelhouse is not a directory: {wheelhouse}")
    if work_root is not None:
        work_root.mkdir(parents=True, exist_ok=True)

    report = check_wheel(
        wheel,
        fixture=fixture,
        wheelhouse=wheelhouse,
        work_root=work_root,
        extras=tuple(dict.fromkeys(arguments.extras)),
        install_layout=arguments.install_layout,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

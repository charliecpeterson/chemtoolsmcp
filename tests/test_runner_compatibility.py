"""Program-neutral runner names and NWChem compatibility imports."""

import ast
import importlib
from pathlib import Path

import pytest

import chemtools.core.runner as core_runner
import chemtools.execution.external_status as external_status
import chemtools.execution.legacy_archive as legacy_archive
import chemtools.execution.legacy_runner as runner
import chemtools.execution.profiles as profiles
import chemtools.execution.resource_inspection as resource_inspection
import chemtools.programs.nwchem.external_status as nwchem_status


SCHEDULER_MODULES = (
    "grasp",
)
NEUTRAL_RUNNER_IMPORTS = {
    "render_calculation_run",
    "run_calculation",
}


def test_nwchem_render_names_are_direct_compatibility_aliases():
    assert runner.run_nwchem is runner.run_calculation
    assert runner.render_nwchem_run is runner.render_calculation_run
    assert runner.run_nwchem.__name__ == "run_calculation"
    assert runner.render_nwchem_run.__name__ == "render_calculation_run"


def test_core_nwchem_status_names_use_the_program_adapter():
    assert (
        core_runner.inspect_nwchem_run_status
        is nwchem_status.inspect_nwchem_run_status
    )
    assert core_runner.watch_nwchem_run is nwchem_status.watch_nwchem_run_status


def test_core_runner_reexports_split_legacy_modules_directly():
    assert runner.archive_paths is legacy_archive.archive_paths
    assert (
        runner.archive_previous_outputs
        is legacy_archive.archive_previous_outputs
    )
    assert core_runner.run_calculation is runner.run_calculation
    assert core_runner.render_calculation_run is runner.render_calculation_run
    assert (
        runner.query_partition_specs
        is resource_inspection.query_partition_specs
    )
    assert (
        runner.get_local_resource_budget
        is resource_inspection.get_local_resource_budget
    )
    assert (
        runner._detect_local_cpu_arch
        is resource_inspection._detect_local_cpu_arch
    )
    assert core_runner.query_partition_specs is runner.query_partition_specs
    assert (
        core_runner.get_local_resource_budget
        is runner.get_local_resource_budget
    )
    assert runner.load_runner_profiles is profiles.load_runner_profiles
    assert runner.resolve_runner_profile is profiles.resolve_runner_profile
    assert runner._resolve_profile is profiles._resolve_profile
    assert core_runner.inspect_run_status is external_status.inspect_run_status
    assert core_runner.watch_run is external_status.watch_run
    assert core_runner.tail_text_file is external_status.tail_text_file


@pytest.mark.parametrize("program", SCHEDULER_MODULES)
def test_non_nwchem_scheduler_imports_only_neutral_runner_names(program):
    scheduler_path = (
        Path(__file__).parents[1]
        / "chemtools"
        / "programs"
        / program
        / "scheduler.py"
    )
    tree = ast.parse(scheduler_path.read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "chemtools.execution.legacy_runner"
        for alias in node.names
    }

    assert imported_names == NEUTRAL_RUNNER_IMPORTS


def test_direct_cancellation_wrappers_are_removed():
    assert not hasattr(runner, "cancel_scheduler_job")
    assert not hasattr(
        importlib.import_module("chemtools.programs.nwchem.runner"),
        "terminate_nwchem_run",
    )
    for program in SCHEDULER_MODULES:
        assert not hasattr(
            importlib.import_module(f"chemtools.programs.{program}.scheduler"),
            f"terminate_{program}_run",
        )

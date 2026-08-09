"""Program-neutral runner names and NWChem compatibility imports."""

import ast
from pathlib import Path

import pytest

import chemtools.core.runner as core_runner
import chemtools.execution.legacy_archive as legacy_archive
import chemtools.execution.legacy_runner as runner
import chemtools.execution.legacy_profiles as legacy_profiles
import chemtools.execution.legacy_status as legacy_status
import chemtools.execution.profiles as profiles
import chemtools.execution.resource_inspection as resource_inspection
import chemtools.programs.nwchem.legacy_status as nwchem_status


SCHEDULER_MODULES = (
    "molcas",
    "dirac",
    "grasp",
)
NEUTRAL_RUNNER_IMPORTS = {
    "cancel_scheduler_job",
    "inspect_run_status",
    "render_calculation_run",
    "run_calculation",
    "watch_run",
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
    assert runner.load_runner_profiles is (
        legacy_profiles.load_runner_profiles
    )
    assert runner.resolve_runner_profile is (
        legacy_profiles.resolve_runner_profile
    )
    assert runner._resolve_profile is legacy_profiles._resolve_profile
    assert runner.inspect_run_status is legacy_status.inspect_run_status
    assert runner.watch_run is legacy_status.watch_run
    assert runner.tail_text_file is legacy_status.tail_text_file
    assert runner.cancel_scheduler_job is (
        legacy_status.cancel_scheduler_job
    )


def test_legacy_profile_module_is_an_exact_compatibility_facade():
    assert legacy_profiles.__all__ == profiles.__all__
    for name in profiles.__all__:
        assert getattr(legacy_profiles, name) is getattr(profiles, name)
    assert legacy_profiles._format_template is profiles._format_template
    assert legacy_profiles._resolve_profile is profiles._resolve_profile


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

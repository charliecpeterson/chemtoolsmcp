"""Program-neutral runner names and NWChem compatibility imports."""

import ast
from pathlib import Path

import pytest

import chemtools.core.runner as runner
import chemtools.execution.legacy_profiles as legacy_profiles
import chemtools.execution.legacy_status as legacy_status


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


def test_nwchem_runner_names_are_direct_compatibility_aliases():
    assert runner.run_nwchem is runner.run_calculation
    assert runner.render_nwchem_run is runner.render_calculation_run
    assert runner.inspect_nwchem_run_status is runner.inspect_run_status
    assert runner.watch_nwchem_run is runner.watch_run
    assert runner.run_nwchem.__name__ == "run_calculation"
    assert runner.render_nwchem_run.__name__ == "render_calculation_run"
    assert runner.inspect_nwchem_run_status.__name__ == "inspect_run_status"
    assert runner.watch_nwchem_run.__name__ == "watch_run"


def test_core_runner_reexports_split_legacy_modules_directly():
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
        and node.module == "chemtools.core.runner"
        for alias in node.names
    }

    assert imported_names == NEUTRAL_RUNNER_IMPORTS

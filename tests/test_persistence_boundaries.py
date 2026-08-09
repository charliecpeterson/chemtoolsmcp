"""Persistence owners and their temporary import paths stay object-identical."""

import chemtools.application.legacy_artifacts as legacy_artifacts
import chemtools.application.run_registry as run_registry
import chemtools.core.artifact_registry as old_artifacts
import chemtools.core.legacy_artifacts as old_legacy_artifacts
import chemtools.core.registry_db as old_sqlite
import chemtools.core.run_records as old_runs
import chemtools.core.run_registry as old_run_registry
import chemtools.execution.launch_registry as old_launches
import chemtools.persistence.artifacts as artifacts
import chemtools.persistence.launches as launches
import chemtools.persistence.runs as runs
import chemtools.persistence.sqlite as sqlite


def test_core_sqlite_path_is_an_exact_compatibility_import():
    assert old_sqlite.connect_registry is sqlite.connect_registry
    assert old_sqlite.ensure_registry_schema is sqlite.ensure_registry_schema


def test_core_run_record_path_is_an_exact_compatibility_import():
    assert old_runs.register_run is runs.register_run
    assert old_runs.update_run_status is runs.update_run_status
    assert old_runs.list_runs is runs.list_runs
    assert old_runs.get_run_summary is runs.get_run_summary


def test_core_artifact_store_path_is_an_exact_compatibility_import():
    assert old_artifacts.record_run_artifacts is artifacts.record_run_artifacts
    assert old_artifacts.load_run_artifacts is artifacts.load_run_artifacts
    assert old_artifacts.UnknownRunUidError is artifacts.UnknownRunUidError


def test_execution_launch_store_path_is_an_exact_compatibility_import():
    assert old_launches.create_launch_record is launches.create_launch_record
    assert old_launches.load_launch_record is launches.load_launch_record
    assert old_launches.update_launch_record is launches.update_launch_record


def test_combined_service_paths_are_exact_compatibility_imports():
    assert old_run_registry.create_campaign is run_registry.create_campaign
    assert old_run_registry.register_run is runs.register_run
    assert (
        old_legacy_artifacts.project_registered_run_artifacts
        is legacy_artifacts.project_registered_run_artifacts
    )

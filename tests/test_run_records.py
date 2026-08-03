"""Public import compatibility for extracted run-record persistence."""

import chemtools
import chemtools.core.run_records as run_records
import chemtools.core.run_registry as run_registry


def test_run_registry_reexports_run_record_functions_without_wrappers():
    assert run_registry.register_run is run_records.register_run
    assert run_registry.update_run_status is run_records.update_run_status
    assert run_registry.list_runs is run_records.list_runs
    assert run_registry.get_run_summary is run_records.get_run_summary
    assert chemtools.register_run is run_records.register_run
    assert chemtools.update_run_status is run_records.update_run_status


def test_run_registry_public_surface_keeps_legacy_services():
    assert callable(run_registry.advance_workflow)
    assert callable(run_registry.create_campaign)
    assert callable(run_registry.create_workflow)
    assert callable(run_registry.generate_input_batch)
    assert callable(run_registry.get_campaign_energies)
    assert callable(run_registry.get_campaign_status)

"""Compatibility imports for legacy execution and NWChem status."""

from chemtools.execution.legacy_runner import (
    _declared_profile_installation,
    _detect_local_cpu_arch,
    _render_environment,
    _render_hook_block,
    _render_module_block,
    _render_submit_command,
    archive_paths,
    archive_previous_outputs,
    get_local_resource_budget,
    query_partition_specs,
    render_calculation_run,
    render_nwchem_run,
    run_calculation,
    run_nwchem,
)
from chemtools.execution.external_status import (
    inspect_run_status,
    tail_text_file,
    watch_run,
)
from chemtools.execution.profiles import (
    DEFAULT_RUNNER_PROFILES,
    RUNNER_PROFILES_ENV,
    _format_template,
    _resolve_profile,
    declared_program_installation,
    load_runner_profiles,
    resolve_runner_profile,
)
from chemtools.programs.nwchem.external_status import (
    inspect_nwchem_run_status,
    watch_nwchem_run_status as watch_nwchem_run,
)

__all__ = [
    "DEFAULT_RUNNER_PROFILES",
    "RUNNER_PROFILES_ENV",
    "archive_paths",
    "archive_previous_outputs",
    "declared_program_installation",
    "get_local_resource_budget",
    "inspect_nwchem_run_status",
    "inspect_run_status",
    "load_runner_profiles",
    "query_partition_specs",
    "render_calculation_run",
    "render_nwchem_run",
    "resolve_runner_profile",
    "run_calculation",
    "run_nwchem",
    "tail_text_file",
    "watch_nwchem_run",
    "watch_run",
]

"""Compatibility imports for the canonical runner-profile owner."""

from chemtools.execution.profiles import (
    DEFAULT_RUNNER_PROFILES,
    RUNNER_PROFILES_ENV,
    _format_template,
    _resolve_profile,
    declared_program_installation,
    direct_installation,
    environment_values,
    expanded_profile_path,
    hardware_description,
    load_runner_profiles,
    module_lines,
    program_settings,
    require_version_1,
    resolve_runner_profile,
    resource_request,
    scheduler_type,
    slurm_scheduler_defaults,
)

__all__ = [
    "DEFAULT_RUNNER_PROFILES",
    "RUNNER_PROFILES_ENV",
    "declared_program_installation",
    "direct_installation",
    "environment_values",
    "expanded_profile_path",
    "hardware_description",
    "load_runner_profiles",
    "module_lines",
    "program_settings",
    "require_version_1",
    "resolve_runner_profile",
    "resource_request",
    "scheduler_type",
    "slurm_scheduler_defaults",
]

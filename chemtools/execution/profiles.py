"""Load runner profiles and convert them into typed execution models.

The current profile schema is version 1. Program adapters supply executable
details while this module owns shared resources, hardware, modules,
installations, direct commands, and Slurm fields.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
from typing import Any, Mapping

from chemtools.core.execution import (
    HardwareDescription,
    ProgramInstallation,
    ResourceRequest,
    SchedulerDefaults,
)

DEFAULT_RUNNER_PROFILES = (
    Path(__file__).resolve().parent.parent
    / "runner_profiles.example.json"
)
RUNNER_PROFILES_ENV = "CHEMTOOLS_RUNNER_PROFILES"


def load_runner_profiles(path: str | None = None) -> dict[str, Any]:
    configured_path = path or os.environ.get(RUNNER_PROFILES_ENV)
    source = (
        Path(configured_path).resolve()
        if configured_path
        else DEFAULT_RUNNER_PROFILES.resolve()
    )
    if not source.is_file():
        raise ValueError(
            f"runner profiles file does not exist: {source}"
        )
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json" or text.lstrip().startswith("{"):
        payload = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            json_sidecar = source.with_suffix(".json")
            if json_sidecar.is_file():
                return load_runner_profiles(str(json_sidecar))
            raise ValueError(
                "YAML runner profiles require PyYAML, or use JSON "
                f"instead: {source}"
            ) from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(
            f"runner profiles file must contain a mapping: {source}"
        )
    payload["__source__"] = str(source)
    return payload


def _resolve_profile(
    profiles: dict[str, Any],
    profile_name: str,
) -> dict[str, Any]:
    defaults = deepcopy(profiles.get("defaults", {}))
    profile = deepcopy((profiles.get("profiles") or {}).get(profile_name))
    if not profile:
        raise ValueError(f"unknown runner profile: {profile_name}")
    return _deep_merge(defaults, profile)


def resolve_runner_profile(
    profiles: dict[str, Any],
    profile_name: str,
) -> dict[str, Any]:
    """Resolve version 1 defaults without exposing its merge implementation."""
    return _resolve_profile(profiles, profile_name)


def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _format_template(
    template: str | None,
    context: dict[str, Any],
) -> str:
    if template is None:
        return ""
    safe_context = {
        key: "" if value is None else value
        for key, value in context.items()
    }
    return template.format_map(safe_context)


def require_version_1(profiles: Mapping[str, Any]) -> None:
    schema_version = str(profiles.get("schema_version") or "")
    if schema_version != "1.0":
        raise ValueError(
            f"unsupported legacy runner profile schema {schema_version!r}"
        )


def resource_request(values: Mapping[str, Any]) -> ResourceRequest:
    return ResourceRequest(
        nodes=values.get("nodes") or 1,
        mpi_ranks=values.get("mpi_ranks") or 1,
        omp_threads=values.get("omp_threads") or 1,
        memory_mb_per_node=values.get("memory_mb_per_node"),
        walltime=values.get("walltime"),
        partition=values.get("partition"),
        account=values.get("account"),
    )


def hardware_description(
    values: Mapping[str, Any],
) -> HardwareDescription:
    return HardwareDescription(
        cores_per_node=values.get("cores_per_node"),
        memory_mb_per_node=values.get("node_memory_mb"),
        cpu_arch=values.get("cpu_arch"),
    )


def environment_values(
    *mappings: Mapping[str, Any],
) -> dict[str, str]:
    environment = {}
    for values in mappings:
        environment.update({
            str(key): str(value)
            for key, value in values.items()
            if value is not None
        })
    return environment


def expanded_profile_path(value: str, *, field_name: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError(
            f"{field_name} contains an unresolved variable: {value}"
        )
    return expanded


def module_lines(values: Mapping[str, Any]) -> tuple[str, ...]:
    lines = []
    if values.get("purge_first"):
        lines.append("module purge")
    lines.extend(
        f"module load {module}"
        for module in values.get("load") or ()
    )
    return tuple(lines)


def program_settings(
    profile: Mapping[str, Any],
    program: str,
) -> Mapping[str, Any]:
    programs = profile.get("programs") or {}
    if not isinstance(programs, Mapping):
        raise ValueError("runner profile programs must be a mapping")
    settings = programs.get(program) or {}
    if not isinstance(settings, Mapping):
        raise ValueError(
            f"runner profile programs.{program} must be a mapping"
        )
    return settings


def _program_argv(
    settings: Mapping[str, Any],
    field_name: str,
    *,
    required: bool,
) -> tuple[str, ...]:
    values = settings.get(field_name)
    if values is None:
        if required:
            raise ValueError(
                f"runner profile {field_name} must be a non-empty array"
            )
        return ()
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, (list, tuple))
    ):
        raise ValueError(
            f"runner profile {field_name} must be an array of strings"
        )
    argv = tuple(
        expanded_profile_path(
            value,
            field_name=f"runner profile {field_name}[{index}]",
        )
        if isinstance(value, str)
        else value
        for index, value in enumerate(values)
    )
    if required and not argv:
        raise ValueError(
            f"runner profile {field_name} must be a non-empty array"
        )
    return argv


def declared_program_installation(
    profile: Mapping[str, Any],
    program: str,
    *,
    environment: Mapping[str, str] | None = None,
    setup_lines: tuple[str, ...] = (),
    pre_run_lines: tuple[str, ...] = (),
    entrypoints: Mapping[str, tuple[str, ...]] | None = None,
) -> ProgramInstallation | None:
    settings = program_settings(profile, program)
    if not settings:
        return None
    return ProgramInstallation(
        launcher_argv=_program_argv(
            settings,
            "launcher_argv",
            required=False,
        ),
        executable_argv=_program_argv(
            settings,
            "executable_argv",
            required=True,
        ),
        environment=environment or {},
        setup_lines=setup_lines,
        pre_run_lines=pre_run_lines,
        entrypoints=entrypoints or {},
    )


def direct_installation(
    profile: Mapping[str, Any],
    *,
    default_command: str,
    environment: Mapping[str, str] | None = None,
) -> ProgramInstallation:
    launcher = profile.get("launcher") or {}
    tokens = tuple(
        shlex.split(launcher.get("command") or default_command)
    )
    if not tokens:
        raise ValueError("legacy direct launcher command is empty")
    return ProgramInstallation(
        launcher_argv=tokens[:-1],
        executable_argv=(tokens[-1],),
        environment=environment or {},
    )


def scheduler_type(profile: Mapping[str, Any]) -> str:
    launcher = profile.get("launcher") or {}
    scheduler = profile.get("scheduler") or {}
    return str(
        scheduler.get("system")
        or launcher.get("scheduler_type")
        or "slurm"
    ).lower()


def slurm_scheduler_defaults(
    profile: Mapping[str, Any],
) -> SchedulerDefaults:
    resolved_scheduler = scheduler_type(profile)
    if resolved_scheduler != "slurm":
        raise ValueError(
            f"legacy scheduler {resolved_scheduler!r} is not canonical"
        )
    launcher = profile.get("launcher") or {}
    scheduler = profile.get("scheduler") or {}
    submit = tuple(
        shlex.split(launcher.get("submit_command") or "sbatch")
    )
    status = tuple(
        shlex.split(
            launcher.get("status_command")
            or "squeue -j {job_id} -h -o %T"
        )
    )
    accounting = tuple(
        shlex.split(
            launcher.get("accounting_command")
            or (
                "sacct -n -X -j {job_id} "
                "-o State%30,ExitCode,ElapsedRaw -P"
            )
        )
    )
    cancel = tuple(
        shlex.split(
            launcher.get("cancel_command") or "scancel {job_id}"
        )
    )
    script_name = scheduler.get("submit_script_name") or "{job_name}.job"
    if script_name != "{job_name}.job":
        raise ValueError(
            "legacy target requires '{job_name}.job' script naming"
        )
    return SchedulerDefaults(
        submit_argv=(*submit, "{script_file}"),
        status_argv=status,
        cancel_argv=cancel,
        accounting_argv=accounting,
        job_id_regex=(
            launcher.get("job_id_regex")
            or scheduler.get(
                "job_id_regex",
                r"Submitted batch job (\d+)",
            )
        ),
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

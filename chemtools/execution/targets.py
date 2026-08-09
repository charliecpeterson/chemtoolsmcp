"""Load schema-2 machine targets into immutable execution models.

Target files contain trusted host and scheduler configuration. Chemistry
requests select a target by name and cannot replace its commands or roots.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from chemtools.core.execution import (
    ExecutionTarget,
    HardwareDescription,
    ProgramInstallation,
    ResourceRequest,
    SchedulerDefaults,
)


TARGET_CONFIG_ENV = "CHEMTOOLS_TARGETS"
TARGET_CONFIG_SCHEMA = "2.0"


@dataclass(frozen=True)
class TargetCatalog:
    targets: Mapping[str, ExecutionTarget]
    default_target: str | None
    enable_execution: bool
    source: Path

    def __post_init__(self) -> None:
        targets = dict(self.targets)
        if not targets:
            raise ValueError(
                "target configuration must define at least one target"
            )
        for name, target in targets.items():
            if name != target.name:
                raise ValueError(
                    f"target key {name!r} does not match target name "
                    f"{target.name!r}"
                )
        object.__setattr__(self, "targets", MappingProxyType(targets))
        if (
            self.default_target is not None
            and self.default_target not in targets
        ):
            raise ValueError(
                f"unknown default target: {self.default_target!r}"
            )
        if not isinstance(self.enable_execution, bool):
            raise TypeError("enable_execution must be a boolean")
        object.__setattr__(self, "source", Path(self.source).resolve())

    def resolve(
        self,
        name: str | None = None,
        *,
        program: str | None = None,
    ) -> ExecutionTarget:
        selected = name or self.default_target
        if selected is None:
            raise ValueError(
                "target name is required; no default_target is configured"
            )
        try:
            target = self.targets[selected]
        except KeyError as exc:
            raise ValueError(f"unknown execution target: {selected!r}") from exc
        if program is not None and program not in target.programs:
            raise ValueError(
                f"target {selected!r} has no {program!r} installation"
            )
        return target


def load_target_catalog(path: str | Path | None = None) -> TargetCatalog:
    configured = path or os.environ.get(TARGET_CONFIG_ENV)
    if configured is None:
        raise ValueError(
            f"target configuration path is required or set {TARGET_CONFIG_ENV}"
        )
    source = Path(configured).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"target configuration file does not exist: {source}")
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json" or text.lstrip().startswith("{"):
        payload = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ValueError(
                "YAML target configuration requires PyYAML"
            ) from exc
        payload = yaml.safe_load(text)
    return parse_target_catalog(payload, source=source)


def parse_target_catalog(
    payload: Any,
    *,
    source: str | Path,
) -> TargetCatalog:
    root = _mapping(payload, "target configuration")
    _reject_unknown(
        root,
        {"schema_version", "chemtools", "targets"},
        "target configuration",
    )
    if str(root.get("schema_version") or "") != TARGET_CONFIG_SCHEMA:
        raise ValueError(
            f"unsupported target configuration schema {root.get('schema_version')!r}"
        )
    settings = _mapping(root.get("chemtools") or {}, "chemtools")
    _reject_unknown(
        settings,
        {"default_target", "enable_execution"},
        "chemtools",
    )
    enabled = settings.get("enable_execution", False)
    if not isinstance(enabled, bool):
        raise ValueError("chemtools.enable_execution must be a boolean")
    default_target = settings.get("default_target")
    if default_target is not None and (
        not isinstance(default_target, str) or not default_target
    ):
        raise ValueError("chemtools.default_target must be a non-empty string")

    target_values = _mapping(root.get("targets"), "targets")
    targets = {
        name: _target(name, value)
        for name, value in target_values.items()
    }
    return TargetCatalog(
        targets=targets,
        default_target=default_target,
        enable_execution=enabled,
        source=Path(source),
    )


def _target(name: Any, value: Any) -> ExecutionTarget:
    if not isinstance(name, str) or not name:
        raise ValueError("target names must be non-empty strings")
    config = _mapping(value, f"targets.{name}")
    _reject_unknown(
        config,
        {
            "executor",
            "allowed_work_roots",
            "hardware",
            "resources",
            "scheduler",
            "programs",
        },
        f"targets.{name}",
    )
    roots = tuple(
        _absolute_path(item, f"targets.{name}.allowed_work_roots")
        for item in _sequence(
            config.get("allowed_work_roots"),
            f"targets.{name}.allowed_work_roots",
        )
    )
    programs = _mapping(config.get("programs"), f"targets.{name}.programs")
    scheduler_value = config.get("scheduler")
    return ExecutionTarget(
        name=name,
        executor=config.get("executor"),
        allowed_work_roots=roots,
        hardware=_hardware(
            config.get("hardware") or {},
            f"targets.{name}.hardware",
        ),
        programs={
            program: _installation(
                settings,
                f"targets.{name}.programs.{program}",
            )
            for program, settings in programs.items()
        },
        scheduler=(
            _scheduler(scheduler_value, f"targets.{name}.scheduler")
            if scheduler_value is not None
            else None
        ),
        default_resources=_resources(
            config.get("resources") or {},
            f"targets.{name}.resources",
        ),
    )


def _hardware(value: Any, field: str) -> HardwareDescription:
    config = _mapping(value, field)
    _reject_unknown(
        config,
        {"cores_per_node", "memory_mb_per_node", "cpu_arch"},
        field,
    )
    return HardwareDescription(**config)


def _resources(value: Any, field: str) -> ResourceRequest:
    config = _mapping(value, field)
    _reject_unknown(
        config,
        {
            "nodes",
            "mpi_ranks",
            "omp_threads",
            "memory_mb_per_node",
            "walltime",
            "partition",
            "account",
        },
        field,
    )
    return ResourceRequest(**config)


def _scheduler(value: Any, field: str) -> SchedulerDefaults:
    config = _mapping(value, field)
    _reject_unknown(
        config,
        {
            "submit_argv",
            "status_argv",
            "cancel_argv",
            "accounting_argv",
            "job_id_regex",
            "script_suffix",
        },
        field,
    )
    required = {
        name: _argv(config.get(name), f"{field}.{name}", required=True)
        for name in ("submit_argv", "status_argv", "cancel_argv")
    }
    optional: dict[str, Any] = {}
    if "accounting_argv" in config:
        optional["accounting_argv"] = _argv(
            config["accounting_argv"],
            f"{field}.accounting_argv",
            required=False,
        )
    for name in ("job_id_regex", "script_suffix"):
        if name in config:
            optional[name] = config[name]
    return SchedulerDefaults(**required, **optional)


def _installation(value: Any, field: str) -> ProgramInstallation:
    config = _mapping(value, field)
    _reject_unknown(
        config,
        {
            "executable_argv",
            "launcher_argv",
            "environment",
            "setup_lines",
            "pre_run_lines",
            "entrypoints",
        },
        field,
    )
    entrypoints = _mapping(
        config.get("entrypoints") or {},
        f"{field}.entrypoints",
    )
    environment = _mapping(
        config.get("environment") or {},
        f"{field}.environment",
    )
    return ProgramInstallation(
        executable_argv=_expanded_argv(
            config.get("executable_argv"),
            f"{field}.executable_argv",
            required=True,
        ),
        launcher_argv=_expanded_argv(
            config.get("launcher_argv") or (),
            f"{field}.launcher_argv",
            required=False,
        ),
        environment={
            _string(key, f"{field}.environment key"): _expanded_value(
                item,
                f"{field}.environment.{key}",
            )
            for key, item in environment.items()
        },
        setup_lines=_argv(
            config.get("setup_lines") or (),
            f"{field}.setup_lines",
            required=False,
        ),
        pre_run_lines=_argv(
            config.get("pre_run_lines") or (),
            f"{field}.pre_run_lines",
            required=False,
        ),
        entrypoints={
            _string(name, f"{field}.entrypoints key"): _expanded_argv(
                argv,
                f"{field}.entrypoints.{name}",
                required=True,
            )
            for name, argv in entrypoints.items()
        },
    )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _sequence(value: Any, field: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array")
    if not value:
        raise ValueError(f"{field} must not be empty")
    return tuple(value)


def _argv(value: Any, field: str, *, required: bool) -> tuple[str, ...]:
    if value is None:
        if required:
            raise ValueError(f"{field} must be a non-empty array")
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array")
    values = tuple(value)
    if required and not values:
        raise ValueError(f"{field} must be a non-empty array")
    return tuple(
        _string(item, f"{field}[{index}]")
        for index, item in enumerate(values)
    )


def _expanded_argv(value: Any, field: str, *, required: bool) -> tuple[str, ...]:
    return tuple(
        _expanded_value(item, f"{field}[{index}]")
        for index, item in enumerate(_argv(value, field, required=required))
    )


def _expanded_value(value: Any, field: str) -> str:
    text = _string(value, field)
    expanded = os.path.expandvars(os.path.expanduser(text))
    if "$" in expanded:
        raise ValueError(
            f"{field} contains an unresolved environment variable"
        )
    return expanded


def _absolute_path(value: Any, field: str) -> Path:
    expanded = Path(_expanded_value(value, field))
    if not expanded.is_absolute():
        raise ValueError(f"{field} entries must be absolute paths")
    return expanded.resolve()


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _reject_unknown(
    values: Mapping[str, Any],
    allowed: set[str],
    field: str,
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{field} has unknown fields: {', '.join(unknown)}")


__all__ = [
    "TARGET_CONFIG_ENV",
    "TARGET_CONFIG_SCHEMA",
    "TargetCatalog",
    "load_target_catalog",
    "parse_target_catalog",
]

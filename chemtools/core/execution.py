"""Immutable execution targets, launch plans, and executor result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping
from uuid import UUID

from chemtools.core.artifacts import ExpectedArtifact
from chemtools.core.slurm import (
    RecordedSlurmStatus,
    SlurmStatusResult,
    SlurmStatusState,
)


ExecutorKind = Literal["local", "slurm"]
StagingMode = Literal["copy", "symlink"]
LaunchStatus = Literal[
    "pending",
    "started",
    "completed",
    "failed",
    "timed_out",
    "submitted",
    "submitted_untracked",
    "submit_failed",
    "launch_failed",
    "cancelled",
    "cancel_failed",
]
_LAUNCH_STATUSES = frozenset({
    "pending",
    "started",
    "completed",
    "failed",
    "timed_out",
    "submitted",
    "submitted_untracked",
    "submit_failed",
    "launch_failed",
    "cancelled",
    "cancel_failed",
})
_TARGET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PROGRAM_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_WALLTIME_RE = re.compile(r"^\d+:[0-5]\d:[0-5]\d$")


def _text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{field_name} contains a forbidden control character")
    return value


def _argv(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    normalized = tuple(values)
    for index, value in enumerate(normalized):
        _text(value, f"{field_name}[{index}]")
    return normalized


def _environment(
    values: Mapping[str, str],
    field_name: str,
) -> Mapping[str, str]:
    normalized: dict[str, str] = {}
    for key, value in values.items():
        _text(key, f"{field_name} key")
        if not isinstance(value, str):
            raise TypeError(f"{field_name}.{key} must be a string")
        if any(character in value for character in ("\x00", "\n", "\r")):
            raise ValueError(
                f"{field_name}.{key} contains a forbidden control character"
            )
        normalized[key] = value
    return MappingProxyType(normalized)


def _canonical_uuid(value: str, field_name: str) -> None:
    _text(value, field_name)
    try:
        normalized = str(UUID(value))
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be a canonical UUID string"
        ) from exc
    if normalized != value:
        raise ValueError(f"{field_name} must be a canonical UUID string")


def _aware_datetime(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a UTC offset")


@dataclass(frozen=True)
class ResourceRequest:
    nodes: int = 1
    mpi_ranks: int = 1
    omp_threads: int = 1
    memory_mb_per_node: int | None = None
    walltime: str | None = None
    partition: str | None = None
    account: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("nodes", "mpi_ranks", "omp_threads"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            self.memory_mb_per_node is not None
            and (
                isinstance(self.memory_mb_per_node, bool)
                or not isinstance(self.memory_mb_per_node, int)
                or self.memory_mb_per_node < 1
            )
        ):
            raise ValueError("memory_mb_per_node must be a positive integer")
        if (
            self.walltime is not None
            and not _WALLTIME_RE.fullmatch(self.walltime)
        ):
            raise ValueError("walltime must use H+:MM:SS")
        for field_name in ("partition", "account"):
            value = getattr(self, field_name)
            if value is not None:
                _text(value, field_name)


@dataclass(frozen=True)
class HardwareDescription:
    cores_per_node: int | None = None
    memory_mb_per_node: int | None = None
    cpu_arch: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("cores_per_node", "memory_mb_per_node"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(f"{field_name} must be a positive integer")
        if self.cpu_arch is not None:
            _text(self.cpu_arch, "cpu_arch")


@dataclass(frozen=True)
class StagedFile:
    source: Path
    destination: Path
    mode: StagingMode = "copy"
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source))
        object.__setattr__(self, "destination", Path(self.destination))
        if self.mode not in ("copy", "symlink"):
            raise ValueError("staging mode must be 'copy' or 'symlink'")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a boolean")


@dataclass(frozen=True)
class ProgramInstallation:
    executable_argv: tuple[str, ...]
    launcher_argv: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    setup_lines: tuple[str, ...] = ()
    pre_run_lines: tuple[str, ...] = ()
    entrypoints: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        executable = _argv(self.executable_argv, "executable_argv")
        if not executable:
            raise ValueError("executable_argv must not be empty")
        object.__setattr__(self, "executable_argv", executable)
        object.__setattr__(
            self,
            "launcher_argv",
            _argv(self.launcher_argv, "launcher_argv"),
        )
        object.__setattr__(
            self,
            "environment",
            _environment(self.environment, "environment"),
        )
        object.__setattr__(
            self,
            "setup_lines",
            _argv(self.setup_lines, "setup_lines"),
        )
        object.__setattr__(
            self,
            "pre_run_lines",
            _argv(self.pre_run_lines, "pre_run_lines"),
        )
        entrypoints: dict[str, tuple[str, ...]] = {}
        for name, values in self.entrypoints.items():
            _text(name, "entrypoint name")
            argv = _argv(tuple(values), f"entrypoints.{name}")
            if not argv:
                raise ValueError(
                    f"entrypoints.{name} must not be empty"
                )
            entrypoints[name] = argv
        object.__setattr__(
            self,
            "entrypoints",
            MappingProxyType(entrypoints),
        )


@dataclass(frozen=True)
class SchedulerDefaults:
    submit_argv: tuple[str, ...]
    status_argv: tuple[str, ...]
    cancel_argv: tuple[str, ...]
    accounting_argv: tuple[str, ...] = ()
    job_id_regex: str = r"Submitted batch job (\d+)"
    script_suffix: str = ".job"

    def __post_init__(self) -> None:
        for field_name in ("submit_argv", "status_argv", "cancel_argv"):
            values = _argv(getattr(self, field_name), field_name)
            if not values:
                raise ValueError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, values)
        object.__setattr__(
            self,
            "accounting_argv",
            _argv(self.accounting_argv, "accounting_argv"),
        )
        _text(self.job_id_regex, "job_id_regex")
        if re.compile(self.job_id_regex).groups < 1:
            raise ValueError("job_id_regex must contain a capture group")
        _text(self.script_suffix, "script_suffix")
        if not self.script_suffix.startswith("."):
            raise ValueError("script_suffix must start with '.'")


@dataclass(frozen=True)
class ExecutionTarget:
    name: str
    executor: ExecutorKind
    allowed_work_roots: tuple[Path, ...]
    hardware: HardwareDescription
    programs: Mapping[str, ProgramInstallation]
    scheduler: SchedulerDefaults | None = None

    def __post_init__(self) -> None:
        if not _TARGET_NAME_RE.fullmatch(self.name):
            raise ValueError("invalid execution target name")
        if self.executor not in ("local", "slurm"):
            raise ValueError("executor must be 'local' or 'slurm'")
        roots = tuple(Path(root) for root in self.allowed_work_roots)
        if not roots:
            raise ValueError("allowed_work_roots must not be empty")
        if any(not root.is_absolute() for root in roots):
            raise ValueError("allowed_work_roots must be absolute")
        object.__setattr__(self, "allowed_work_roots", roots)

        programs = dict(self.programs)
        if not programs:
            raise ValueError("programs must not be empty")
        for name, installation in programs.items():
            if not _PROGRAM_NAME_RE.fullmatch(name):
                raise ValueError(f"invalid program name: {name!r}")
            if not isinstance(installation, ProgramInstallation):
                raise TypeError(
                    f"program {name!r} must use ProgramInstallation"
                )
        object.__setattr__(self, "programs", MappingProxyType(programs))

        if self.executor == "local" and self.scheduler is not None:
            raise ValueError("local targets cannot define scheduler defaults")
        if self.executor == "slurm" and self.scheduler is None:
            raise ValueError("slurm targets require scheduler defaults")


@dataclass(frozen=True)
class LaunchPlan:
    job_name: str
    program: str
    program_arguments: tuple[str, ...]
    environment: Mapping[str, str]
    working_directory: Path
    staged_files: tuple[StagedFile, ...]
    expected_artifacts: tuple[ExpectedArtifact, ...]
    resources: ResourceRequest
    entrypoint: str | None = None
    stdin_text: str | None = None
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not _TARGET_NAME_RE.fullmatch(self.job_name):
            raise ValueError("invalid job_name")
        if not _PROGRAM_NAME_RE.fullmatch(self.program):
            raise ValueError("invalid program name")
        object.__setattr__(
            self,
            "program_arguments",
            _argv(self.program_arguments, "program_arguments"),
        )
        object.__setattr__(
            self,
            "environment",
            _environment(self.environment, "environment"),
        )
        working_directory = Path(self.working_directory)
        if not working_directory.is_absolute():
            raise ValueError("working_directory must be absolute")
        object.__setattr__(self, "working_directory", working_directory)
        object.__setattr__(self, "staged_files", tuple(self.staged_files))
        object.__setattr__(
            self,
            "expected_artifacts",
            tuple(self.expected_artifacts),
        )
        if self.entrypoint is not None:
            _text(self.entrypoint, "entrypoint")
        if self.stdin_text is not None:
            if not isinstance(self.stdin_text, str):
                raise TypeError("stdin_text must be a string")
            if "\x00" in self.stdin_text:
                raise ValueError("stdin_text contains a null byte")
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive number")


@dataclass(frozen=True)
class PreparedLaunch:
    plan: LaunchPlan
    target: ExecutionTarget
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.plan, LaunchPlan):
            raise TypeError("plan must use LaunchPlan")
        if not isinstance(self.target, ExecutionTarget):
            raise TypeError("target must use ExecutionTarget")
        if self.plan.program not in self.target.programs:
            raise ValueError(
                f"target {self.target.name!r} does not configure "
                f"program {self.plan.program!r}"
            )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True)
class RenderedCommand:
    target: str
    program: str
    executor: ExecutorKind
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    working_directory: Path
    stdout_path: Path | None
    stderr_path: Path | None
    staged_files: tuple[StagedFile, ...] = ()
    stdin_text: str | None = None
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(
            self,
            "environment",
            _environment(self.environment, "environment"),
        )
        object.__setattr__(
            self,
            "working_directory",
            Path(self.working_directory),
        )
        if self.stdout_path is not None:
            object.__setattr__(self, "stdout_path", Path(self.stdout_path))
        if self.stderr_path is not None:
            object.__setattr__(self, "stderr_path", Path(self.stderr_path))
        staged_files = tuple(self.staged_files)
        destinations: set[Path] = set()
        for staged_file in staged_files:
            if not isinstance(staged_file, StagedFile):
                raise TypeError("staged_files must use StagedFile")
            if (
                not staged_file.source.is_absolute()
                or not staged_file.destination.is_absolute()
            ):
                raise ValueError(
                    "rendered staged file paths must be absolute"
                )
            if staged_file.destination in destinations:
                raise ValueError(
                    "rendered staged destinations must be unique"
                )
            destinations.add(staged_file.destination)
        object.__setattr__(self, "staged_files", staged_files)


@dataclass(frozen=True)
class RenderedSlurmScript:
    command: RenderedCommand
    script_path: Path
    script_text: str
    submit_argv: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "script_path", Path(self.script_path))
        object.__setattr__(self, "submit_argv", tuple(self.submit_argv))


@dataclass(frozen=True)
class LocalLaunchResult:
    command: RenderedCommand
    process_id: int
    status: Literal["started"]
    started_at: datetime


@dataclass(frozen=True)
class LocalStatusResult:
    process_id: int
    status: Literal["running", "completed", "failed"]
    return_code: int | None
    checked_at: datetime

    def __post_init__(self) -> None:
        if (
            isinstance(self.process_id, bool)
            or not isinstance(self.process_id, int)
            or self.process_id < 1
        ):
            raise ValueError("process_id must be a positive integer")
        if self.status == "running" and self.return_code is not None:
            raise ValueError("running process status cannot have a return code")
        if self.status != "running" and self.return_code is None:
            raise ValueError("terminal process status requires a return code")
        _aware_datetime(self.checked_at, "checked_at")


@dataclass(frozen=True)
class LocalSynchronousResult:
    command: RenderedCommand
    status: Literal["completed", "failed", "timed_out"]
    return_code: int
    stdout: str
    stderr: str
    started_at: datetime
    completed_at: datetime
    elapsed_seconds: float


@dataclass(frozen=True)
class SlurmSubmissionResult:
    script: RenderedSlurmScript
    status: Literal[
        "submitted",
        "submitted_untracked",
        "submit_failed",
    ]
    return_code: int
    stdout: str
    stderr: str
    job_id: str | None
    submitted_at: datetime


@dataclass(frozen=True)
class LocalCancellationResult:
    process_id: int
    status: Literal["cancelled", "cancel_failed"]
    signal: Literal["SIGTERM", "SIGKILL"]
    error: str | None
    cancelled_at: datetime


@dataclass(frozen=True)
class SlurmCancellationResult:
    job_id: str
    argv: tuple[str, ...]
    status: Literal["cancelled", "cancel_failed"]
    return_code: int
    stdout: str
    stderr: str
    cancelled_at: datetime


@dataclass(frozen=True)
class ExecutionLaunchRecord:
    launch_id: str
    instance_id: str
    target: str
    executor: ExecutorKind
    program: str
    working_directory: Path
    argv: tuple[str, ...]
    environment_keys: tuple[str, ...]
    resources: ResourceRequest
    status: LaunchStatus
    created_at: datetime
    updated_at: datetime
    staged_files: tuple[StagedFile, ...] = ()
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    script_path: Path | None = None
    process_id: int | None = None
    job_id: str | None = None
    stdin_sha256: str | None = None
    stdin_size_bytes: int | None = None
    return_code: int | None = None
    elapsed_seconds: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        _canonical_uuid(self.launch_id, "launch_id")
        _canonical_uuid(self.instance_id, "instance_id")
        _text(self.target, "target")
        if self.executor not in ("local", "slurm"):
            raise ValueError("executor must be 'local' or 'slurm'")
        if self.status not in _LAUNCH_STATUSES:
            raise ValueError("invalid launch record status")
        if not _PROGRAM_NAME_RE.fullmatch(self.program):
            raise ValueError("invalid program name")
        working_directory = Path(self.working_directory)
        if not working_directory.is_absolute():
            raise ValueError("working_directory must be absolute")
        object.__setattr__(
            self,
            "working_directory",
            working_directory,
        )
        object.__setattr__(self, "argv", _argv(self.argv, "argv"))
        environment_keys = tuple(self.environment_keys)
        for index, key in enumerate(environment_keys):
            _text(key, f"environment_keys[{index}]")
        if len(environment_keys) != len(set(environment_keys)):
            raise ValueError("environment_keys must not contain duplicates")
        object.__setattr__(
            self,
            "environment_keys",
            environment_keys,
        )
        if not isinstance(self.resources, ResourceRequest):
            raise TypeError("resources must use ResourceRequest")
        staged_files = tuple(self.staged_files)
        destinations: set[Path] = set()
        for staged_file in staged_files:
            if not isinstance(staged_file, StagedFile):
                raise TypeError("staged_files must use StagedFile")
            if (
                not staged_file.source.is_absolute()
                or not staged_file.destination.is_absolute()
            ):
                raise ValueError(
                    "launch record staged file paths must be absolute"
                )
            if staged_file.destination in destinations:
                raise ValueError(
                    "launch record staged destinations must be unique"
                )
            destinations.add(staged_file.destination)
        object.__setattr__(self, "staged_files", staged_files)
        for field_name in ("stdout_path", "stderr_path", "script_path"):
            path = getattr(self, field_name)
            if path is None:
                continue
            normalized_path = Path(path)
            if not normalized_path.is_absolute():
                raise ValueError(f"{field_name} must be absolute")
            object.__setattr__(self, field_name, normalized_path)
        _aware_datetime(self.created_at, "created_at")
        _aware_datetime(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        if self.process_id is not None and (
            isinstance(self.process_id, bool)
            or not isinstance(self.process_id, int)
            or self.process_id < 1
        ):
            raise ValueError("process_id must be a positive integer")
        if self.job_id is not None:
            _text(self.job_id, "job_id")
        if self.stdin_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}",
            self.stdin_sha256,
        ):
            raise ValueError(
                "stdin_sha256 must be a lowercase SHA-256 digest"
            )
        if self.stdin_size_bytes is not None and (
            isinstance(self.stdin_size_bytes, bool)
            or not isinstance(self.stdin_size_bytes, int)
            or self.stdin_size_bytes < 0
        ):
            raise ValueError(
                "stdin_size_bytes must be a non-negative integer"
            )
        if (self.stdin_sha256 is None) != (
            self.stdin_size_bytes is None
        ):
            raise ValueError(
                "stdin digest and size must be recorded together"
            )
        if self.return_code is not None and (
            isinstance(self.return_code, bool)
            or not isinstance(self.return_code, int)
        ):
            raise TypeError("return_code must be an integer")
        if self.elapsed_seconds is not None and (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or self.elapsed_seconds < 0
        ):
            raise ValueError(
                "elapsed_seconds must be a non-negative number"
            )
        if (
            self.executor == "local"
            and self.status in {"completed", "failed", "timed_out"}
        ):
            if self.return_code is None:
                raise ValueError(
                    "terminal synchronous records require return_code"
                )
            if self.elapsed_seconds is None:
                raise ValueError(
                    "terminal synchronous records require elapsed_seconds"
                )
        if self.executor == "local" and self.job_id is not None:
            raise ValueError("local launch records cannot contain job_id")
        if self.executor == "slurm" and self.process_id is not None:
            raise ValueError(
                "slurm launch records cannot contain process_id"
            )
        if self.status == "started" and self.process_id is None:
            raise ValueError("started launch records require process_id")
        if self.status == "submitted" and self.job_id is None:
            raise ValueError("submitted launch records require job_id")


@dataclass(frozen=True)
class ExecutionRunLink:
    launch_id: str
    run_uid: str
    linked_at: datetime

    def __post_init__(self) -> None:
        _canonical_uuid(self.launch_id, "launch_id")
        _canonical_uuid(self.run_uid, "run_uid")
        _aware_datetime(self.linked_at, "linked_at")


@dataclass(frozen=True)
class RecordedLaunch:
    record: ExecutionLaunchRecord
    result: LocalLaunchResult | SlurmSubmissionResult


@dataclass(frozen=True)
class RecordedLocalStatus:
    record: ExecutionLaunchRecord
    result: LocalStatusResult


@dataclass(frozen=True)
class RecordedSynchronousRun:
    record: ExecutionLaunchRecord
    result: LocalSynchronousResult


@dataclass(frozen=True)
class RecordedCancellation:
    record: ExecutionLaunchRecord
    result: LocalCancellationResult | SlurmCancellationResult


__all__ = [
    "ExecutionLaunchRecord",
    "ExecutionRunLink",
    "ExecutionTarget",
    "ExecutorKind",
    "HardwareDescription",
    "LaunchPlan",
    "LaunchStatus",
    "LocalCancellationResult",
    "LocalLaunchResult",
    "LocalStatusResult",
    "LocalSynchronousResult",
    "PreparedLaunch",
    "ProgramInstallation",
    "RecordedCancellation",
    "RecordedLaunch",
    "RecordedLocalStatus",
    "RecordedSlurmStatus",
    "RecordedSynchronousRun",
    "RenderedCommand",
    "RenderedSlurmScript",
    "ResourceRequest",
    "SchedulerDefaults",
    "SlurmCancellationResult",
    "SlurmStatusResult",
    "SlurmStatusState",
    "SlurmSubmissionResult",
    "StagedFile",
    "StagingMode",
]

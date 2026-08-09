"""QMCPACK launch plans and runner-profile target adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from chemtools.core.artifacts import (
    ArtifactLocation,
    ArtifactRole,
    ExpectedArtifact,
)
from chemtools.core.execution import ExecutionTarget, LaunchPlan, ResourceRequest
from chemtools.execution.profiles import (
    declared_program_installation,
    direct_installation,
    environment_values,
    hardware_description,
    resource_request,
    require_version_1,
    resolve_runner_profile,
    slurm_scheduler_defaults,
)


@dataclass(frozen=True)
class LegacyQmcpackTarget:
    target: ExecutionTarget
    output_template: str
    error_template: str

    @property
    def default_resources(self) -> ResourceRequest:
        return self.target.default_resources


def adapt_legacy_qmcpack_profile(
    profiles: dict[str, Any],
    profile_name: str,
    *,
    allowed_work_roots: tuple[Path, ...],
) -> LegacyQmcpackTarget:
    require_version_1(profiles)
    profile = resolve_runner_profile(profiles, profile_name)
    launcher = profile.get("launcher") or {}
    kind = launcher.get("kind", "direct")
    environment = environment_values(
        profile.get("env") or {},
        (profile.get("execution") or {}).get("env") or {},
    )
    installation = declared_program_installation(
        profile,
        "qmcpack",
        environment=environment,
    )
    if installation is None:
        installation = direct_installation(
            profile,
            default_command="qmcpack",
            environment=environment,
        )
    if kind == "direct":
        executor = "local"
        scheduler = None
    elif kind == "scheduler":
        executor = "slurm"
        scheduler = slurm_scheduler_defaults(profile)
    else:
        raise ValueError(f"unsupported legacy launcher kind: {kind!r}")
    file_rules = profile.get("file_rules") or {}
    return LegacyQmcpackTarget(
        target=ExecutionTarget(
            name=profile_name,
            executor=executor,
            allowed_work_roots=allowed_work_roots,
            hardware=hardware_description(profile.get("resources") or {}),
            programs={"qmcpack": installation},
            scheduler=scheduler,
            default_resources=resource_request(
                profile.get("resources") or {}
            ),
        ),
        output_template=file_rules.get("output_file", "{job_name}.out"),
        error_template=file_rules.get("error_file", "{job_name}.err"),
    )


def build_qmcpack_launch_plan(
    input_path: str | Path,
    resources: ResourceRequest,
    *,
    job_name: str | None = None,
    output_template: str = "{job_name}.out",
    error_template: str = "{job_name}.err",
    environment: Mapping[str, str] | None = None,
    qmcpack_dry_run: bool = False,
) -> LaunchPlan:
    input_file = Path(input_path).resolve()
    if not input_file.is_file():
        raise ValueError(f"input file does not exist: {input_path}")
    effective_job_name = job_name or input_file.stem
    context = {"job_name": effective_job_name}
    output_path = input_file.parent / output_template.format_map(context)
    error_path = input_file.parent / error_template.format_map(context)
    return LaunchPlan(
        job_name=effective_job_name,
        program="qmcpack",
        program_arguments=(
            (input_file.name, "--dryrun")
            if qmcpack_dry_run
            else (input_file.name,)
        ),
        environment=environment or {},
        working_directory=input_file.parent,
        staged_files=(),
        expected_artifacts=(
            ExpectedArtifact(
                expectation_id=f"{effective_job_name}:stdout",
                roles=frozenset({
                    ArtifactRole.PRIMARY_OUTPUT,
                    ArtifactRole.STDOUT,
                }),
                kind="qmcpack.output",
                location=ArtifactLocation(path=output_path, entry_type="file"),
                required=True,
            ),
            ExpectedArtifact(
                expectation_id=f"{effective_job_name}:stderr",
                roles=frozenset({ArtifactRole.STDERR}),
                kind="qmcpack.error",
                location=ArtifactLocation(path=error_path, entry_type="file"),
                required=False,
            ),
        ),
        resources=resources,
    )


__all__ = [
    "LegacyQmcpackTarget",
    "adapt_legacy_qmcpack_profile",
    "build_qmcpack_launch_plan",
]

"""Quantum ESPRESSO pw.x launch plans and version 1 profile adaptation."""

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
from chemtools.core.runner import resolve_runner_profile
from chemtools.execution.legacy_profiles import (
    declared_program_installation,
    direct_installation,
    environment_values,
    hardware_description,
    require_version_1,
    resource_request,
    slurm_scheduler_defaults,
)


@dataclass(frozen=True)
class LegacyQeTarget:
    target: ExecutionTarget
    default_resources: ResourceRequest
    output_template: str
    error_template: str


def adapt_legacy_qe_profile(
    profiles: dict[str, Any],
    profile_name: str,
    *,
    allowed_work_roots: tuple[Path, ...],
) -> LegacyQeTarget:
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
        "qe",
        environment=environment,
    )
    if installation is None:
        installation = direct_installation(
            profile,
            default_command="pw.x",
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
    return LegacyQeTarget(
        target=ExecutionTarget(
            name=profile_name,
            executor=executor,
            allowed_work_roots=allowed_work_roots,
            hardware=hardware_description(profile.get("resources") or {}),
            programs={"qe": installation},
            scheduler=scheduler,
        ),
        default_resources=resource_request(profile.get("resources") or {}),
        output_template=file_rules.get("output_file", "{job_name}.out"),
        error_template=file_rules.get("error_file", "{job_name}.err"),
    )


def build_qe_launch_plan(
    input_path: str | Path,
    resources: ResourceRequest,
    *,
    job_name: str | None = None,
    output_template: str = "{job_name}.out",
    error_template: str = "{job_name}.err",
    environment: Mapping[str, str] | None = None,
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
        program="qe",
        program_arguments=("-in", input_file.name),
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
                kind="qe.output",
                location=ArtifactLocation(path=output_path, entry_type="file"),
                required=True,
            ),
            ExpectedArtifact(
                expectation_id=f"{effective_job_name}:stderr",
                roles=frozenset({ArtifactRole.STDERR}),
                kind="qe.error",
                location=ArtifactLocation(path=error_path, entry_type="file"),
                required=False,
            ),
        ),
        resources=resources,
    )


__all__ = [
    "LegacyQeTarget",
    "adapt_legacy_qe_profile",
    "build_qe_launch_plan",
]

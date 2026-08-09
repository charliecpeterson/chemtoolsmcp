"""NWChem launch plans and the version 1 runner-profile target adapter.

The adapter reads trusted legacy profile commands. New launch plans keep
NWChem filenames and arguments separate from target-owned command prefixes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Any, Mapping

from chemtools.core.artifacts import (
    ArtifactLocation,
    ArtifactRole,
    ExpectedArtifact,
)
from chemtools.core.execution import (
    ExecutionTarget,
    LaunchPlan,
    ProgramInstallation,
    ResourceRequest,
)
from chemtools.execution.profiles import (
    declared_program_installation,
    direct_installation,
    environment_values,
    hardware_description,
    module_lines,
    require_version_1,
    resolve_runner_profile,
    resource_request,
    slurm_scheduler_defaults,
)


@dataclass(frozen=True)
class LegacyNwchemTarget:
    target: ExecutionTarget
    default_resources: ResourceRequest
    output_template: str
    error_template: str


def _direct_installation(
    profile: Mapping[str, Any],
) -> ProgramInstallation:
    declared = declared_program_installation(
        profile,
        "nwchem",
        environment=environment_values(profile.get("env") or {}),
    )
    if declared is not None:
        return declared
    return direct_installation(
        profile,
        default_command="nwchem",
        environment=environment_values(profile.get("env") or {}),
    )


def _slurm_installation(
    profile: Mapping[str, Any],
) -> ProgramInstallation:
    hooks = profile.get("hooks") or {}
    declared = declared_program_installation(
        profile,
        "nwchem",
        environment=environment_values(profile.get("env") or {}),
        setup_lines=module_lines(profile.get("modules") or {}),
        pre_run_lines=tuple(hooks.get("pre_run") or ()),
    )
    if declared is not None:
        return declared
    execution = profile.get("execution") or {}
    executable = tuple(
        shlex.split(execution.get("nwchem_executable") or "nwchem")
    )
    launcher = tuple(
        shlex.split(execution.get("mpi_launch") or "")
    )
    return ProgramInstallation(
        launcher_argv=launcher,
        executable_argv=executable,
        environment=environment_values(profile.get("env") or {}),
        setup_lines=module_lines(profile.get("modules") or {}),
        pre_run_lines=tuple(hooks.get("pre_run") or ()),
    )


def adapt_legacy_nwchem_profile(
    profiles: dict[str, Any],
    profile_name: str,
    *,
    allowed_work_roots: tuple[Path, ...],
) -> LegacyNwchemTarget:
    require_version_1(profiles)
    profile = resolve_runner_profile(profiles, profile_name)
    launcher = profile.get("launcher") or {}
    kind = launcher.get("kind", "direct")
    resources = profile.get("resources") or {}
    if kind == "direct":
        executor = "local"
        installation = _direct_installation(profile)
        scheduler = None
    elif kind == "scheduler":
        executor = "slurm"
        installation = _slurm_installation(profile)
        scheduler = slurm_scheduler_defaults(profile)
    else:
        raise ValueError(f"unsupported legacy launcher kind: {kind!r}")

    file_rules = profile.get("file_rules") or {}
    return LegacyNwchemTarget(
        target=ExecutionTarget(
            name=profile_name,
            executor=executor,
            allowed_work_roots=allowed_work_roots,
            hardware=hardware_description(resources),
            programs={"nwchem": installation},
            scheduler=scheduler,
        ),
        default_resources=resource_request(resources),
        output_template=file_rules.get(
            "output_file",
            "{job_name}.out",
        ),
        error_template=file_rules.get(
            "error_file",
            "{job_name}.err",
        ),
    )


def build_nwchem_launch_plan(
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
        program="nwchem",
        program_arguments=(input_file.name,),
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
                kind="nwchem.output",
                location=ArtifactLocation(
                    path=output_path,
                    entry_type="file",
                ),
                required=True,
            ),
            ExpectedArtifact(
                expectation_id=f"{effective_job_name}:stderr",
                roles=frozenset({ArtifactRole.STDERR}),
                kind="nwchem.error",
                location=ArtifactLocation(
                    path=error_path,
                    entry_type="file",
                ),
                required=False,
            ),
        ),
        resources=resources,
    )


__all__ = [
    "LegacyNwchemTarget",
    "adapt_legacy_nwchem_profile",
    "build_nwchem_launch_plan",
]

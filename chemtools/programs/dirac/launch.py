"""DIRAC launch plans and the version 1 runner-profile target adapter.

DIRAC owns pam-dirac arguments and paired input naming. Targets own
containers, modules, environment values, and scheduler commands.
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
    expanded_profile_path,
    hardware_description,
    module_lines,
    program_settings,
    require_version_1,
    resolve_runner_profile,
    resource_request,
    slurm_scheduler_defaults,
)
from chemtools.programs.dirac.runtime import pam_dirac_arguments


@dataclass(frozen=True)
class LegacyDiracTarget:
    target: ExecutionTarget
    default_resources: ResourceRequest
    output_template: str
    error_template: str
    master_memory_mb: int | None
    node_memory_mb: int | None


def _profile_environment(
    profile: Mapping[str, Any],
) -> dict[str, str]:
    execution = profile.get("execution") or {}
    return environment_values(
        profile.get("env") or {},
        execution.get("env") or {},
    )


def _configured_installation(
    profile: Mapping[str, Any],
) -> ProgramInstallation:
    hooks = profile.get("hooks") or {}
    declared = declared_program_installation(
        profile,
        "dirac",
        environment=_profile_environment(profile),
        setup_lines=module_lines(profile.get("modules") or {}),
        pre_run_lines=tuple(hooks.get("pre_run") or ()),
    )
    if declared is not None:
        return declared
    binary = tuple(
        shlex.split(profile.get("pam_dirac_binary") or "pam-dirac")
    )
    if not binary:
        raise ValueError("DIRAC pam_dirac_binary is empty")
    container = profile.get("container_sif")
    if container:
        launcher = (
            str(profile.get("apptainer_binary") or "apptainer"),
            "exec",
            expanded_profile_path(
                str(container),
                field_name="DIRAC container path",
            ),
        )
    else:
        launcher = ()
    return ProgramInstallation(
        launcher_argv=launcher,
        executable_argv=binary,
        environment=_profile_environment(profile),
        setup_lines=module_lines(profile.get("modules") or {}),
        pre_run_lines=tuple(hooks.get("pre_run") or ()),
    )


def _direct_installation(
    profile: Mapping[str, Any],
) -> ProgramInstallation:
    declared = declared_program_installation(
        profile,
        "dirac",
        environment=_profile_environment(profile),
    )
    if declared is not None:
        return declared
    if (
        profile.get("container_sif")
        or profile.get("pam_dirac_binary")
    ):
        return _configured_installation(profile)
    return direct_installation(
        profile,
        default_command="pam-dirac",
        environment=_profile_environment(profile),
    )


def adapt_legacy_dirac_profile(
    profiles: dict[str, Any],
    profile_name: str,
    *,
    allowed_work_roots: tuple[Path, ...],
) -> LegacyDiracTarget:
    require_version_1(profiles)
    profile = resolve_runner_profile(profiles, profile_name)
    launcher = profile.get("launcher") or {}
    dirac = program_settings(profile, "dirac")
    kind = launcher.get("kind", "direct")
    resources = dict(profile.get("resources") or {})
    if (
        not resources.get("mpi_ranks")
        and dirac.get(
            "default_mpi",
            profile.get("default_mpi"),
        ) is not None
    ):
        resources["mpi_ranks"] = dirac.get(
            "default_mpi",
            profile.get("default_mpi"),
        )
    if kind == "direct":
        executor = "local"
        installation = _direct_installation(profile)
        scheduler = None
    elif kind == "scheduler":
        executor = "slurm"
        installation = _configured_installation(profile)
        scheduler = slurm_scheduler_defaults(profile)
    else:
        raise ValueError(f"unsupported legacy launcher kind: {kind!r}")

    file_rules = profile.get("file_rules") or {}
    return LegacyDiracTarget(
        target=ExecutionTarget(
            name=profile_name,
            executor=executor,
            allowed_work_roots=allowed_work_roots,
            hardware=hardware_description(resources),
            programs={"dirac": installation},
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
        master_memory_mb=dirac.get(
            "default_mw",
            profile.get("default_mw"),
        ),
        node_memory_mb=dirac.get(
            "default_nw",
            profile.get("default_nw"),
        ),
    )


def build_dirac_launch_plan(
    input_path: str | Path,
    mol_path: str | Path,
    resources: ResourceRequest,
    *,
    master_memory_mb: int | None = None,
    node_memory_mb: int | None = None,
    job_name: str | None = None,
    output_template: str = "{job_name}.out",
    error_template: str = "{job_name}.err",
    environment: Mapping[str, str] | None = None,
) -> LaunchPlan:
    input_file = Path(input_path).resolve()
    molecule_file = Path(mol_path).resolve()
    if not input_file.is_file():
        raise ValueError(f"input file does not exist: {input_path}")
    if not molecule_file.is_file():
        raise ValueError(f"molecule file does not exist: {mol_path}")
    if molecule_file.parent != input_file.parent:
        raise ValueError(
            "typed DIRAC execution currently requires the .inp and .mol "
            "files to use the same working directory"
        )

    effective_job_name = job_name or input_file.stem
    context = {"job_name": effective_job_name}
    output_path = (
        input_file.parent / output_template.format_map(context)
    )
    error_path = (
        input_file.parent / error_template.format_map(context)
    )
    return LaunchPlan(
        job_name=effective_job_name,
        program="dirac",
        program_arguments=tuple(pam_dirac_arguments(
            input_file.name,
            molecule_file.name,
            mpi=resources.mpi_ranks,
            mw=master_memory_mb,
            nw=node_memory_mb,
        )),
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
                kind="dirac.output",
                location=ArtifactLocation(
                    path=output_path,
                    entry_type="file",
                ),
                required=True,
            ),
            ExpectedArtifact(
                expectation_id=f"{effective_job_name}:stderr",
                roles=frozenset({ArtifactRole.STDERR}),
                kind="dirac.error",
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
    "LegacyDiracTarget",
    "adapt_legacy_dirac_profile",
    "build_dirac_launch_plan",
]

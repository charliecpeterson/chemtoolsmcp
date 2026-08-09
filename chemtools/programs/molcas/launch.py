"""OpenMolcas launch plans and the version 1 profile adapter.

Molcas owns CASPT2 parallelism checks, pymolcas arguments, scratch identity,
and expected files. Targets own containers, modules, and scheduler commands.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from chemtools.programs.molcas.runtime import (
    _resolve_parallelism,
    detect_caspt2,
)


@dataclass(frozen=True)
class LegacyMolcasTarget:
    target: ExecutionTarget
    default_resources: ResourceRequest
    output_template: str
    error_template: str
    parallel_caspt2_supported: bool


@dataclass(frozen=True)
class PreparedMolcasLaunch:
    plan: LaunchPlan
    requested_mpi_ranks: int
    effective_mpi_ranks: int
    has_caspt2: bool
    parallel_caspt2_supported: bool
    warnings: tuple[str, ...]


def _profile_environment(
    profile: Mapping[str, Any],
) -> dict[str, str]:
    execution = profile.get("execution") or {}
    return environment_values(
        profile.get("env") or {},
        execution.get("env") or {},
    )


def _slurm_installation(
    profile: Mapping[str, Any],
) -> ProgramInstallation:
    hooks = profile.get("hooks") or {}
    pre_run_lines = (
        "export MOLCAS_PROJECT={job_name}_$SLURM_JOB_ID",
        *tuple(hooks.get("pre_run") or ()),
    )
    declared = declared_program_installation(
        profile,
        "molcas",
        environment=_profile_environment(profile),
        setup_lines=module_lines(profile.get("modules") or {}),
        pre_run_lines=pre_run_lines,
    )
    if declared is not None:
        return declared
    execution = profile.get("execution") or {}
    pymolcas = tuple(
        shlex.split(execution.get("pymolcas_command") or "pymolcas")
    )
    if not pymolcas:
        raise ValueError("Molcas pymolcas_command is empty")
    sif = (
        execution.get("apptainer_sif")
        or profile.get("apptainer_sif")
    )
    if sif:
        apptainer = str(
            execution.get("apptainer_binary") or "apptainer"
        )
        launcher = (
            apptainer,
            "exec",
            expanded_profile_path(
                str(sif),
                field_name="Molcas execution path",
            ),
        )
    else:
        launcher = ()
    return ProgramInstallation(
        launcher_argv=launcher,
        executable_argv=pymolcas,
        environment=_profile_environment(profile),
        setup_lines=module_lines(profile.get("modules") or {}),
        pre_run_lines=pre_run_lines,
    )


def adapt_legacy_molcas_profile(
    profiles: dict[str, Any],
    profile_name: str,
    *,
    allowed_work_roots: tuple[Path, ...],
) -> LegacyMolcasTarget:
    require_version_1(profiles)
    profile = resolve_runner_profile(profiles, profile_name)
    launcher = profile.get("launcher") or {}
    kind = launcher.get("kind", "direct")
    resources = profile.get("resources") or {}
    if kind == "direct":
        target_executor = "local"
        installation = declared_program_installation(
            profile,
            "molcas",
            environment=_profile_environment(profile),
        )
        if installation is None:
            installation = direct_installation(
                profile,
                default_command="pymolcas",
                environment=_profile_environment(profile),
            )
        scheduler = None
    elif kind == "scheduler":
        target_executor = "slurm"
        installation = _slurm_installation(profile)
        scheduler = slurm_scheduler_defaults(profile)
    else:
        raise ValueError(f"unsupported legacy launcher kind: {kind!r}")

    execution = profile.get("execution") or {}
    molcas = program_settings(profile, "molcas")
    file_rules = profile.get("file_rules") or {}
    return LegacyMolcasTarget(
        target=ExecutionTarget(
            name=profile_name,
            executor=target_executor,
            allowed_work_roots=allowed_work_roots,
            hardware=hardware_description(resources),
            programs={"molcas": installation},
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
        parallel_caspt2_supported=bool(
            molcas.get(
                "parallel_caspt2_supported",
                execution.get("parallel_caspt2_supported", True),
            )
        ),
    )


def build_molcas_launch_plan(
    input_path: str | Path,
    resources: ResourceRequest,
    *,
    parallel_caspt2_supported: bool,
    job_name: str | None = None,
    output_template: str = "{job_name}.out",
    error_template: str = "{job_name}.err",
    environment: Mapping[str, str] | None = None,
) -> PreparedMolcasLaunch:
    input_file = Path(input_path).resolve()
    if not input_file.is_file():
        raise ValueError(f"input file does not exist: {input_path}")
    input_text = input_file.read_text(
        encoding="utf-8",
        errors="replace",
    )
    has_caspt2 = detect_caspt2(input_text)
    effective_ranks, warnings = _resolve_parallelism(
        has_caspt2=has_caspt2,
        parallel_caspt2_supported=parallel_caspt2_supported,
        requested_np=resources.mpi_ranks,
    )
    effective_resources = resources
    if effective_ranks != resources.mpi_ranks:
        effective_resources = replace(
            resources,
            nodes=1,
            mpi_ranks=effective_ranks,
        )
    effective_job_name = job_name or input_file.stem
    context = {"job_name": effective_job_name}
    output_path = (
        input_file.parent / output_template.format_map(context)
    )
    error_path = (
        input_file.parent / error_template.format_map(context)
    )
    launch_environment = {
        **(environment or {}),
        "MOLCAS_PROJECT": effective_job_name,
        "MOLCAS_NPROCS": str(effective_ranks),
    }
    plan = LaunchPlan(
        job_name=effective_job_name,
        program="molcas",
        program_arguments=(
            "-np",
            str(effective_ranks),
            input_file.name,
        ),
        environment=launch_environment,
        working_directory=input_file.parent,
        staged_files=(),
        expected_artifacts=(
            ExpectedArtifact(
                expectation_id=f"{effective_job_name}:stdout",
                roles=frozenset({
                    ArtifactRole.PRIMARY_OUTPUT,
                    ArtifactRole.STDOUT,
                }),
                kind="molcas.output",
                location=ArtifactLocation(
                    path=output_path,
                    entry_type="file",
                ),
                required=True,
            ),
            ExpectedArtifact(
                expectation_id=f"{effective_job_name}:stderr",
                roles=frozenset({ArtifactRole.STDERR}),
                kind="molcas.error",
                location=ArtifactLocation(
                    path=error_path,
                    entry_type="file",
                ),
                required=False,
            ),
        ),
        resources=effective_resources,
    )
    return PreparedMolcasLaunch(
        plan=plan,
        requested_mpi_ranks=resources.mpi_ranks,
        effective_mpi_ranks=effective_ranks,
        has_caspt2=has_caspt2,
        parallel_caspt2_supported=parallel_caspt2_supported,
        warnings=warnings,
    )


__all__ = [
    "LegacyMolcasTarget",
    "PreparedMolcasLaunch",
    "adapt_legacy_molcas_profile",
    "build_molcas_launch_plan",
]

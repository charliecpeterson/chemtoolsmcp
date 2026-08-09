"""GRASP workflow launch plans and the version 1 target adapter.

The calculation input is a shell workflow containing ordered GRASP steps.
Targets own the container, shell executable, modules, and scheduler commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from chemtools.core.artifacts import (
    ArtifactLocation,
    ArtifactRole,
    ExpectedArtifact,
)
from chemtools.core.execution import (
    ExecutionTarget,
    HardwareDescription,
    LaunchPlan,
    ProgramInstallation,
    ResourceRequest,
)
from chemtools.execution.profiles import (
    declared_program_installation,
    environment_values,
    expanded_profile_path,
    hardware_description,
    module_lines,
    require_version_1,
    resolve_runner_profile,
    resource_request,
    slurm_scheduler_defaults,
)
from chemtools.programs.grasp.runtime import resolve_container


GRASP_EXECUTABLES = frozenset({
    "hf",
    "jj2lsj",
    "rangular",
    "rangular_mpi",
    "rbiotransform",
    "rbiotransform_mpi",
    "rci",
    "rci_mpi",
    "rcsfgenerate",
    "rlevels",
    "rhfs",
    "rhfs_lsj",
    "ris4",
    "rmcdhf",
    "rmcdhf_mpi",
    "rnucleus",
    "rsave",
    "rtransition",
    "rtransition_mpi",
    "rwfnestimate",
    "rwfnmchfmcdf",
})


@dataclass(frozen=True)
class LegacyGraspTarget:
    target: ExecutionTarget
    output_template: str
    error_template: str

    @property
    def default_resources(self) -> ResourceRequest:
        return self.target.default_resources


def _installation(
    profile: Mapping[str, Any],
) -> ProgramInstallation:
    hooks = profile.get("hooks") or {}
    environment = environment_values(
        profile.get("env") or {},
        (profile.get("execution") or {}).get("env") or {},
    )
    setup_lines = module_lines(profile.get("modules") or {})
    pre_run_lines = tuple(hooks.get("pre_run") or ())
    entrypoints = {
        executable: (executable,)
        for executable in sorted(GRASP_EXECUTABLES)
    }
    declared = declared_program_installation(
        profile,
        "grasp",
        environment=environment,
        setup_lines=setup_lines,
        pre_run_lines=pre_run_lines,
        entrypoints=entrypoints,
    )
    if declared is not None:
        return declared
    execution = profile.get("execution") or {}
    container = (
        profile.get("apptainer_sif")
        or execution.get("apptainer_sif")
    )
    if not container:
        raise ValueError(
            "typed GRASP workflow execution requires apptainer_sif "
            "in the runner profile"
        )
    return ProgramInstallation(
        launcher_argv=(
            str(
                profile.get("apptainer_binary")
                or execution.get("apptainer_binary")
                or "apptainer"
            ),
            "exec",
            expanded_profile_path(
                str(container),
                field_name="GRASP container path",
            ),
        ),
        executable_argv=("bash",),
        environment=environment,
        setup_lines=setup_lines,
        pre_run_lines=pre_run_lines,
        entrypoints=entrypoints,
    )


def build_grasp_interactive_target(
    working_directory: str | Path,
    *,
    container: str | None = None,
    apptainer_binary: str = "apptainer",
) -> ExecutionTarget:
    work = Path(working_directory).resolve()
    container_path = container or resolve_container()
    return ExecutionTarget(
        name="grasp_interactive_local",
        executor="local",
        allowed_work_roots=(work,),
        hardware=HardwareDescription(),
        programs={
            "grasp": ProgramInstallation(
                launcher_argv=(
                    apptainer_binary,
                    "exec",
                    expanded_profile_path(
                        container_path,
                        field_name="GRASP container path",
                    ),
                ),
                executable_argv=("bash",),
                entrypoints={
                    executable: (executable,)
                    for executable in sorted(GRASP_EXECUTABLES)
                },
            ),
        },
    )


def adapt_legacy_grasp_profile(
    profiles: dict[str, Any],
    profile_name: str,
    *,
    allowed_work_roots: tuple[Path, ...],
) -> LegacyGraspTarget:
    require_version_1(profiles)
    profile = resolve_runner_profile(profiles, profile_name)
    launcher = profile.get("launcher") or {}
    kind = launcher.get("kind", "direct")
    if kind == "direct":
        executor = "local"
        scheduler = None
    elif kind == "scheduler":
        executor = "slurm"
        scheduler = slurm_scheduler_defaults(profile)
    else:
        raise ValueError(f"unsupported legacy launcher kind: {kind!r}")

    resources = profile.get("resources") or {}
    file_rules = profile.get("file_rules") or {}
    return LegacyGraspTarget(
        target=ExecutionTarget(
            name=profile_name,
            executor=executor,
            allowed_work_roots=allowed_work_roots,
            hardware=hardware_description(resources),
            programs={"grasp": _installation(profile)},
            scheduler=scheduler,
            default_resources=resource_request(resources),
        ),
        output_template=file_rules.get(
            "output_file",
            "{job_name}.out",
        ),
        error_template=file_rules.get(
            "error_file",
            "{job_name}.err",
        ),
    )


def build_grasp_workflow_launch_plan(
    workflow_script_path: str | Path,
    resources: ResourceRequest,
    *,
    job_name: str | None = None,
    output_template: str = "{job_name}.out",
    error_template: str = "{job_name}.err",
    environment: Mapping[str, str] | None = None,
) -> LaunchPlan:
    workflow_script = Path(workflow_script_path).resolve()
    if not workflow_script.is_file():
        raise ValueError(
            f"GRASP workflow script does not exist: "
            f"{workflow_script_path}"
        )
    effective_job_name = job_name or workflow_script.stem
    context = {"job_name": effective_job_name}
    output_path = (
        workflow_script.parent / output_template.format_map(context)
    )
    error_path = (
        workflow_script.parent / error_template.format_map(context)
    )
    return LaunchPlan(
        job_name=effective_job_name,
        program="grasp",
        program_arguments=(workflow_script.name,),
        environment=environment or {},
        working_directory=workflow_script.parent,
        staged_files=(),
        expected_artifacts=(
            ExpectedArtifact(
                expectation_id=f"{effective_job_name}:stdout",
                roles=frozenset({
                    ArtifactRole.PRIMARY_OUTPUT,
                    ArtifactRole.STDOUT,
                }),
                kind="grasp.output",
                location=ArtifactLocation(
                    path=output_path,
                    entry_type="file",
                ),
                required=True,
            ),
            ExpectedArtifact(
                expectation_id=f"{effective_job_name}:stderr",
                roles=frozenset({ArtifactRole.STDERR}),
                kind="grasp.error",
                location=ArtifactLocation(
                    path=error_path,
                    entry_type="file",
                ),
                required=False,
            ),
        ),
        resources=resources,
    )


def build_grasp_interactive_launch_plan(
    exe: str,
    *,
    working_directory: str | Path,
    stdin_lines: list[str] | str,
    args: list[str] | None = None,
    timeout_seconds: float = 600.0,
) -> LaunchPlan:
    if exe not in GRASP_EXECUTABLES:
        raise ValueError(
            f"unsupported GRASP executable {exe!r}; expected one of "
            f"{sorted(GRASP_EXECUTABLES)}"
        )
    if isinstance(stdin_lines, list):
        if not all(isinstance(line, str) for line in stdin_lines):
            raise TypeError("GRASP stdin lines must be strings")
        stdin_text = "\n".join(stdin_lines) + "\n"
    elif isinstance(stdin_lines, str):
        stdin_text = (
            stdin_lines
            if stdin_lines.endswith("\n")
            else stdin_lines + "\n"
        )
    else:
        raise TypeError("GRASP stdin must be a string or list of strings")
    arguments = tuple(args or ())
    if not all(isinstance(value, str) for value in arguments):
        raise TypeError("GRASP executable arguments must be strings")
    return LaunchPlan(
        job_name=exe,
        program="grasp",
        entrypoint=exe,
        program_arguments=arguments,
        environment={},
        working_directory=Path(working_directory).resolve(),
        staged_files=(),
        expected_artifacts=(),
        resources=ResourceRequest(),
        stdin_text=stdin_text,
        timeout_seconds=timeout_seconds,
    )


__all__ = [
    "GRASP_EXECUTABLES",
    "LegacyGraspTarget",
    "adapt_legacy_grasp_profile",
    "build_grasp_interactive_launch_plan",
    "build_grasp_interactive_target",
    "build_grasp_workflow_launch_plan",
]

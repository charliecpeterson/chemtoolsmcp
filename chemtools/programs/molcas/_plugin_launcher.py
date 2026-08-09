"""Prepare guided Molcas plans from named targets or version 1 profiles."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from chemtools.core.execution import ExecutionTarget, PreparedLaunch
from chemtools.execution.profiles import (
    environment_values,
    load_runner_profiles,
    merge_profile_resources,
    render_profile_value,
    resolve_runner_profile,
    resource_request,
)
from chemtools.programs.molcas.launch import (
    PreparedMolcasLaunch,
    adapt_legacy_molcas_profile,
    build_molcas_launch_plan,
)


def _guided_launch(
    prepared: PreparedMolcasLaunch,
    target: ExecutionTarget,
    metadata: Mapping[str, Any],
) -> PreparedLaunch:
    adjustments = tuple(
        {
            "code": "molcas_parallel_caspt2_serialized",
            "requested_mpi_ranks": prepared.requested_mpi_ranks,
            "effective_mpi_ranks": prepared.effective_mpi_ranks,
            "reason": warning,
        }
        for warning in prepared.warnings
    )
    return PreparedLaunch(
        plan=prepared.plan,
        target=target,
        metadata={
            **metadata,
            "adjustments": adjustments,
        },
    )


class _MolcasLaunchPlanner:
    def prepare_launch(
        self,
        request: Mapping[str, Any],
    ) -> PreparedLaunch:
        configured_target = request.get("execution_target")
        if configured_target is not None:
            if not isinstance(configured_target, ExecutionTarget):
                raise TypeError("execution_target must use ExecutionTarget")
            input_file = Path(request["input_file"]).resolve()
            if not input_file.is_file():
                raise ValueError(f"input file does not exist: {input_file}")
            values = asdict(configured_target.default_resources)
            values.update(request.get("resources") or {})
            job_name = str(request.get("job_name") or input_file.stem)
            prepared = build_molcas_launch_plan(
                input_file,
                resource_request(values),
                parallel_caspt2_supported=False,
                job_name=job_name,
            )
            return _guided_launch(
                prepared,
                configured_target,
                {
                    "target": configured_target.name,
                    "target_source": "configured",
                },
            )

        profiles_path = request.get("profiles_path")
        profiles = load_runner_profiles(
            str(profiles_path) if profiles_path is not None else None
        )
        profile_name = str(request["profile"])
        profile = resolve_runner_profile(profiles, profile_name)
        input_file = Path(request["input_file"]).resolve()
        if not input_file.is_file():
            raise ValueError(f"input file does not exist: {input_file}")
        job_name = str(request.get("job_name") or input_file.stem)
        resources = merge_profile_resources(
            profile,
            request.get("resources") or {},
        )
        context = {
            "job_name": job_name,
            "job_dir": str(input_file.parent),
            "input_file": input_file.name,
            "input_file_abs": str(input_file),
            **resources,
        }
        file_rules = profile.get("file_rules") or {}
        output_file = render_profile_value(
            str(file_rules.get("output_file", "{job_name}.out")),
            context,
        )
        error_file = render_profile_value(
            str(file_rules.get("error_file", "{job_name}.err")),
            context,
        )
        context.update({
            "output_file": output_file,
            "error_file": error_file,
            "restart_prefix": render_profile_value(
                str(file_rules.get("restart_prefix", "{job_name}")),
                context,
            ),
        })
        execution = profile.get("execution") or {}
        working_directory = Path(render_profile_value(
            str(execution.get("working_directory", "{job_dir}")),
            context,
        )).resolve()
        if working_directory != input_file.parent:
            raise ValueError(
                "guided Molcas execution requires the input directory as the "
                "working directory"
            )

        adapted = adapt_legacy_molcas_profile(
            profiles,
            profile_name,
            allowed_work_roots=(working_directory,),
        )
        environment = {
            key: render_profile_value(value, context)
            for key, value in environment_values(
                profile.get("env") or {},
                execution.get("env") or {},
            ).items()
        }
        prepared = build_molcas_launch_plan(
            input_file,
            resource_request(resources),
            parallel_caspt2_supported=(
                adapted.parallel_caspt2_supported
            ),
            job_name=job_name,
            output_template=output_file,
            error_template=error_file,
            environment=environment,
        )
        return _guided_launch(
            prepared,
            adapted.target,
            {
                "profile": profile_name,
                "profiles_path": profiles["__source__"],
                "launcher_kind": (
                    profile.get("launcher") or {}
                ).get("kind", "direct"),
            },
        )


MOLCAS_LAUNCH_PLANNER = _MolcasLaunchPlanner()


__all__ = ["MOLCAS_LAUNCH_PLANNER"]

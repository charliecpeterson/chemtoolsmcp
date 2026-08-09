"""Prepare guided GRASP workflow plans from targets or migration profiles."""

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
from chemtools.programs.grasp.launch import (
    adapt_legacy_grasp_profile,
    build_grasp_workflow_launch_plan,
)


class _GraspLaunchPlanner:
    def prepare_launch(
        self,
        request: Mapping[str, Any],
    ) -> PreparedLaunch:
        workflow_script = Path(request["input_file"]).resolve()
        if not workflow_script.is_file():
            raise ValueError(
                f"GRASP workflow script does not exist: {workflow_script}"
            )

        configured_target = request.get("execution_target")
        if configured_target is not None:
            if not isinstance(configured_target, ExecutionTarget):
                raise TypeError("execution_target must use ExecutionTarget")
            values = asdict(configured_target.default_resources)
            values.update(request.get("resources") or {})
            job_name = str(
                request.get("job_name") or workflow_script.stem
            )
            return PreparedLaunch(
                plan=build_grasp_workflow_launch_plan(
                    workflow_script,
                    resource_request(values),
                    job_name=job_name,
                ),
                target=configured_target,
                metadata={
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
        job_name = str(
            request.get("job_name") or workflow_script.stem
        )
        resources = merge_profile_resources(
            profile,
            request.get("resources") or {},
        )
        context = {
            "job_name": job_name,
            "job_dir": str(workflow_script.parent),
            "input_file": workflow_script.name,
            "input_file_abs": str(workflow_script),
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
        if working_directory != workflow_script.parent:
            raise ValueError(
                "guided GRASP execution requires the workflow script "
                "directory as the working directory"
            )

        adapted = adapt_legacy_grasp_profile(
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
        plan = build_grasp_workflow_launch_plan(
            workflow_script,
            resource_request(resources),
            job_name=job_name,
            output_template=output_file,
            error_template=error_file,
            environment=environment,
        )
        return PreparedLaunch(
            plan=plan,
            target=adapted.target,
            metadata={
                "profile": profile_name,
                "profiles_path": profiles["__source__"],
                "launcher_kind": (
                    profile.get("launcher") or {}
                ).get("kind", "direct"),
            },
        )


GRASP_LAUNCH_PLANNER = _GraspLaunchPlanner()


__all__ = ["GRASP_LAUNCH_PLANNER"]

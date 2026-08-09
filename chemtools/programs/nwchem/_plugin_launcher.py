"""NWChem adapter from runner profiles to an immutable launch plan."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from chemtools.core.execution import ExecutionTarget, PreparedLaunch
from chemtools.execution.profiles import (
    load_runner_profiles,
    merge_profile_resources,
    render_profile_value,
    resolve_runner_profile,
    resource_request,
)
from chemtools.programs.nwchem.launch import (
    adapt_legacy_nwchem_profile,
    build_nwchem_launch_plan,
)


class _NwchemLaunchPlanner:
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
            return PreparedLaunch(
                plan=build_nwchem_launch_plan(
                    input_file,
                    resource_request(values),
                    job_name=job_name,
                    environment={
                        str(key): str(value)
                        for key, value in (
                            request.get("env_overrides") or {}
                        ).items()
                    },
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
                "guided NWChem execution requires the input directory as the "
                "working directory"
            )

        adapted = adapt_legacy_nwchem_profile(
            profiles,
            profile_name,
            allowed_work_roots=(working_directory,),
        )
        environment = {
            str(key): render_profile_value(str(value), context)
            for key, value in (profile.get("env") or {}).items()
            if value is not None
        }
        environment.update({
            str(key): str(value)
            for key, value in (request.get("env_overrides") or {}).items()
        })
        plan = build_nwchem_launch_plan(
            input_file,
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
                "compatibility": {
                    "resources": resources,
                    "restart_prefix": context["restart_prefix"],
                    "shell": (profile.get("execution") or {}).get(
                        "shell",
                        "/bin/bash",
                    ),
                },
            },
        )


NWCHEM_LAUNCH_PLANNER = _NwchemLaunchPlanner()


__all__ = ["NWCHEM_LAUNCH_PLANNER"]

"""Prepare guided DIRAC plans from named targets or version 1 profiles."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from chemtools.core.execution import ExecutionTarget, PreparedLaunch
from chemtools.execution.profiles import (
    environment_values,
    load_runner_profiles,
    merge_profile_resources,
    program_settings,
    render_profile_value,
    resolve_runner_profile,
    resource_request,
)
from chemtools.programs.dirac.launch import (
    adapt_legacy_dirac_profile,
    build_dirac_launch_plan,
)


class _DiracLaunchPlanner:
    def prepare_launch(
        self,
        request: Mapping[str, Any],
    ) -> PreparedLaunch:
        input_file = Path(request["input_file"]).resolve()
        molecule_file = Path(request["molecule_file"]).resolve()
        if not input_file.is_file():
            raise ValueError(f"input file does not exist: {input_file}")
        if not molecule_file.is_file():
            raise ValueError(
                f"molecule file does not exist: {molecule_file}"
            )

        configured_target = request.get("execution_target")
        if configured_target is not None:
            if not isinstance(configured_target, ExecutionTarget):
                raise TypeError("execution_target must use ExecutionTarget")
            values = asdict(configured_target.default_resources)
            values.update(request.get("resources") or {})
            job_name = str(request.get("job_name") or input_file.stem)
            return PreparedLaunch(
                plan=build_dirac_launch_plan(
                    input_file,
                    molecule_file,
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
        job_name = str(request.get("job_name") or input_file.stem)
        resources = merge_profile_resources(
            profile,
            request.get("resources") or {},
        )
        dirac = program_settings(profile, "dirac")
        default_mpi = dirac.get(
            "default_mpi",
            profile.get("default_mpi"),
        )
        if not resources.get("mpi_ranks") and default_mpi is not None:
            resources["mpi_ranks"] = default_mpi
        context = {
            "job_name": job_name,
            "job_dir": str(input_file.parent),
            "input_file": input_file.name,
            "input_file_abs": str(input_file),
            "mol_file": molecule_file.name,
            "mol_file_abs": str(molecule_file),
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
                "guided DIRAC execution requires the input directory as the "
                "working directory"
            )

        adapted = adapt_legacy_dirac_profile(
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
        plan = build_dirac_launch_plan(
            input_file,
            molecule_file,
            resource_request(resources),
            master_memory_mb=adapted.master_memory_mb,
            node_memory_mb=adapted.node_memory_mb,
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


DIRAC_LAUNCH_PLANNER = _DiracLaunchPlanner()


__all__ = ["DIRAC_LAUNCH_PLANNER"]

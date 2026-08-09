"""NWChem adapter from runner profiles to an immutable launch plan."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from chemtools.core.execution import PreparedLaunch
from chemtools.execution.profiles import (
    load_runner_profiles,
    resolve_runner_profile,
    resource_request,
)
from chemtools.programs.nwchem.launch import (
    adapt_legacy_nwchem_profile,
    build_nwchem_launch_plan,
)


def _resource_values(
    profile: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    values = dict(profile.get("resources") or {})
    values.update(overrides)
    nodes = values.get("nodes") or 1
    mpi_ranks = values.get("mpi_ranks") or 1
    cores_per_node = values.get("cores_per_node") or mpi_ranks
    if nodes > 1 and mpi_ranks <= cores_per_node:
        values["mpi_ranks"] = cores_per_node * nodes
    return values


def _format_profile_value(
    value: str,
    context: Mapping[str, Any],
) -> str:
    return value.format_map({
        key: "" if item is None else item
        for key, item in context.items()
    })


class _NwchemLaunchPlanner:
    def prepare_launch(
        self,
        request: Mapping[str, Any],
    ) -> PreparedLaunch:
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
        resources = _resource_values(
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
        output_file = _format_profile_value(
            str(file_rules.get("output_file", "{job_name}.out")),
            context,
        )
        error_file = _format_profile_value(
            str(file_rules.get("error_file", "{job_name}.err")),
            context,
        )
        context.update({
            "output_file": output_file,
            "error_file": error_file,
            "restart_prefix": _format_profile_value(
                str(file_rules.get("restart_prefix", "{job_name}")),
                context,
            ),
        })
        execution = profile.get("execution") or {}
        working_directory = Path(_format_profile_value(
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
            str(key): _format_profile_value(str(value), context)
            for key, value in (profile.get("env") or {}).items()
            if value is not None
        }
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
            },
        )


NWCHEM_LAUNCH_PLANNER = _NwchemLaunchPlanner()


__all__ = ["NWCHEM_LAUNCH_PLANNER"]

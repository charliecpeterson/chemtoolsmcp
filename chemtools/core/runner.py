"""Legacy profile rendering and compatibility imports.

This module retains resource inspection plus version 1 render and launch
behavior. Profile loading lives in ``execution.legacy_profiles``; process,
scheduler, file, and optional NWChem progress status live in
``execution.legacy_status``. Their public names are re-exported here for
existing Python callers.

Program-neutral entry points:

  run_calculation
  render_calculation_run
  inspect_run_status      (accepts progress_summary_fn=)
  watch_run               (accepts progress_summary_fn=)

The NWChem-named aliases are also kept for compatibility.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from chemtools.execution.legacy_profiles import (
    DEFAULT_RUNNER_PROFILES,
    RUNNER_PROFILES_ENV,
    _format_template,
    _resolve_profile,
    declared_program_installation,
    load_runner_profiles,
    resolve_runner_profile,
)
from chemtools.execution.legacy_status import (
    cancel_scheduler_job,
    inspect_run_status,
    tail_text_file,
    watch_run,
)

# Session-level cache for partition specs (avoids repeated sinfo calls)
_PARTITION_SPECS_CACHE: dict[str, dict[str, Any]] = {}


def _declared_profile_installation(
    profile: dict[str, Any],
):
    programs = profile.get("programs") or {}
    if not isinstance(programs, dict) or len(programs) != 1:
        return None, None
    program = next(iter(programs))
    return program, declared_program_installation(profile, program)


def query_partition_specs(
    partition: str,
    scheduler_type: str,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Query the scheduler for real node specs on a partition.

    Returns node_memory_mb, cpus_per_node, cpu_arch, and raw features.
    Falls back to None values if the query fails.
    """
    effective_cache = cache if cache is not None else _PARTITION_SPECS_CACHE
    if partition in effective_cache:
        return effective_cache[partition]

    result: dict[str, Any] = {
        "node_memory_mb": None,
        "cpus_per_node": None,
        "cpu_arch": "generic",
        "features": [],
    }

    if scheduler_type == "slurm":
        import shutil
        if not shutil.which("sinfo"):
            return result
        try:
            proc = subprocess.run(
                ["sinfo", "-p", partition, "-o", "%m %c %f", "--noheader"],
                capture_output=True, text=True, timeout=10,
            )
            lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
            if not lines:
                return result
            rows = [l.split(None, 2) for l in lines]
            # Use minimum memory (conservative for heterogeneous partitions)
            min_mem = min(int(r[0]) for r in rows if r[0].isdigit())
            min_cpu = min(int(r[1]) for r in rows if len(r) > 1 and r[1].isdigit())
            all_features = set()
            for r in rows:
                if len(r) > 2:
                    all_features.update(r[2].split(","))
            arch = (
                "spr" if "spr" in all_features else
                "skx" if "skx" in all_features else
                "knl" if "knl" in all_features else
                "generic"
            )
            result = {
                "node_memory_mb": min_mem,
                "cpus_per_node": min_cpu,
                "cpu_arch": arch,
                "features": sorted(all_features),
            }
        except Exception:
            pass

    elif scheduler_type == "pbs":
        import shutil
        if not shutil.which("pbsnodes"):
            return result
        try:
            proc = subprocess.run(
                ["pbsnodes", "-a"],
                capture_output=True, text=True, timeout=15,
            )
            # Parse total memory and ncpus from first node block
            mem_match = re.search(r"resources_available\.mem\s*=\s*(\d+)kb", proc.stdout, re.IGNORECASE)
            cpu_match = re.search(r"resources_available\.ncpus\s*=\s*(\d+)", proc.stdout, re.IGNORECASE)
            if mem_match:
                result["node_memory_mb"] = int(mem_match.group(1)) // 1024
            if cpu_match:
                result["cpus_per_node"] = int(cpu_match.group(1))
        except Exception:
            pass

    effective_cache[partition] = result
    return result


def get_local_resource_budget() -> dict[str, Any]:
    """Return available CPU cores and memory on the local machine."""
    try:
        import psutil
        phys_cores = psutil.cpu_count(logical=False) or 1
        load_1min = psutil.getloadavg()[0]
        cores_in_use = min(int(load_1min + 0.5), phys_cores - 1)
        available_cores = max(1, phys_cores - cores_in_use)

        mem = psutil.virtual_memory()
        available_mem_mb = int(mem.available / 1_000_000 * 0.85)
        total_mem_mb = int(mem.total / 1_000_000)

        return {
            "physical_cores": phys_cores,
            "available_cores": available_cores,
            "current_load_1min": load_1min,
            "total_mem_mb": total_mem_mb,
            "available_mem_mb": available_mem_mb,
            "cpu_arch": _detect_local_cpu_arch(),
        }
    except ImportError:
        cores = os.cpu_count() or 1
        return {
            "physical_cores": cores,
            "available_cores": max(1, cores - 1),
            "current_load_1min": None,
            "total_mem_mb": None,
            "available_mem_mb": None,
            "cpu_arch": "generic",
        }


def _detect_local_cpu_arch() -> str:
    """Detect AVX-512 / AVX2 / generic from /proc/cpuinfo or platform."""
    try:
        with open("/proc/cpuinfo") as f:
            flags = f.read()
        if "avx512f" in flags:
            return "avx512"
        if "avx2" in flags:
            return "avx2"
    except OSError:
        pass
    import platform
    machine = platform.machine().lower()
    return "arm" if "arm" in machine or "aarch" in machine else "generic"


def archive_paths(paths: list[Path]) -> list[str]:
    """Rename non-empty files with one timestamp, without overwriting."""
    archived: list[str] = []
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M")
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            destination = path.with_name(f"{path.name}.{ts}")
            # Avoid overwriting an existing archive
            if destination.exists():
                counter = 2
                while destination.exists():
                    destination = path.with_name(
                        f"{path.name}.{ts}.{counter}"
                    )
                    counter += 1
            path.rename(destination)
            archived.append(str(destination))
    return archived


def archive_previous_outputs(job_dir: str, job_name: str) -> list[str]:
    """Archive the legacy output, error, and scheduler script paths."""
    return archive_paths([
        Path(job_dir) / f"{job_name}{extension}"
        for extension in (".out", ".err", ".job")
    ])


def run_calculation(
    input_path: str,
    profile: str,
    *,
    profiles_path: str | None = None,
    job_name: str | None = None,
    resource_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, str] | None = None,
    execute: bool = False,
    write_script: bool = True,
    archive_outputs: bool = True,
    context_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profiles = load_runner_profiles(profiles_path)
    rendered = render_calculation_run(
        input_path=input_path,
        profile=profile,
        profiles=profiles,
        job_name=job_name,
        resource_overrides=resource_overrides,
        env_overrides=env_overrides,
        context_overrides=context_overrides,
    )
    # Pop environment now: it is only needed for subprocess calls, not the response payload.
    env = rendered.pop("environment")

    # Archive previous output files before overwriting
    if execute and archive_outputs:
        archived = archive_previous_outputs(
            rendered["working_directory"],
            rendered["job_name"],
        )
        if archived:
            rendered["archived_previous_outputs"] = archived

    if not execute:
        rendered["executed"] = False
        return rendered

    if rendered["launcher_kind"] == "direct":
        shell = rendered["shell"]
        command = f"cd {shlex.quote(rendered['working_directory'])} && {rendered['command']}"
        process_id = os.spawnve(
            os.P_NOWAIT,
            shell,
            [shell, "-lc", command],
            env,
        )
        rendered["executed"] = True
        rendered["process_id"] = process_id
        rendered["status"] = "started"
        return rendered

    if rendered["launcher_kind"] == "scheduler":
        script_path = rendered["submit_script_path"]
        if write_script:
            Path(script_path).parent.mkdir(parents=True, exist_ok=True)
            Path(script_path).write_text(rendered["submit_script_text"], encoding="utf-8")
        submit_command = rendered["submit_command"]
        try:
            completed = subprocess.run(
                submit_command,
                cwd=rendered["working_directory"],
                env=env,
                capture_output=True,
                text=True,
                shell=isinstance(submit_command, str),
                executable=rendered["shell"] if isinstance(submit_command, str) else None,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            rendered["executed"] = True
            rendered["status"] = "submit_failed"
            rendered["return_code"] = -1
            rendered["stdout"] = ""
            rendered["stderr"] = "sbatch/qsub timed out after 60 seconds"
            return rendered
        rendered["executed"] = True
        rendered["status"] = "submitted" if completed.returncode == 0 else "submit_failed"
        rendered["return_code"] = completed.returncode
        rendered["stdout"] = completed.stdout
        rendered["stderr"] = completed.stderr
        # Parse job ID from scheduler submit output and persist to .jobid file
        job_id: str | None = None
        if completed.returncode == 0:
            job_id_regex = rendered.get("job_id_regex", r"Submitted batch job (\d+)")
            m = re.search(job_id_regex, completed.stdout)
            if m:
                job_id = m.group(1)
        rendered["job_id"] = job_id
        if job_id:
            jobid_path = Path(rendered["working_directory"]) / f"{rendered['job_name']}.jobid"
            try:
                jobid_path.write_text(job_id, encoding="utf-8")
                rendered["jobid_file"] = str(jobid_path)
            except Exception as exc:
                rendered["jobid_file_error"] = str(exc)
        return rendered

    raise ValueError(f"unsupported launcher kind: {rendered['launcher_kind']}")


def render_calculation_run(
    input_path: str,
    profile: str,
    *,
    profiles: dict[str, Any] | None = None,
    profiles_path: str | None = None,
    job_name: str | None = None,
    resource_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, str] | None = None,
    context_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loaded = profiles or load_runner_profiles(profiles_path)
    profile_payload = _resolve_profile(loaded, profile)

    input_obj = Path(input_path).resolve()
    if not input_obj.is_file():
        raise ValueError(f"input file does not exist: {input_path}")

    effective_job_name = job_name or input_obj.stem
    job_dir = str(input_obj.parent)
    resources = deepcopy(profile_payload.get("resources", {}))
    for key, value in (resource_overrides or {}).items():
        resources[key] = value

    # Sanity check: if nodes changed but mpi_ranks wasn't scaled, warn
    nodes = resources.get("nodes") or 1
    cores_per_node = resources.get("cores_per_node") or resources.get("mpi_ranks") or 1
    mpi_ranks = resources.get("mpi_ranks") or 1
    if nodes > 1 and mpi_ranks <= cores_per_node:
        # Auto-scale mpi_ranks = cores_per_node * nodes when only nodes was overridden
        resources["mpi_ranks"] = cores_per_node * nodes

    context: dict[str, Any] = {
        "job_name": effective_job_name,
        "job_dir": job_dir,
        "input_file": input_obj.name,
        "input_file_abs": str(input_obj),
    }
    context.update(resources)

    file_rules = profile_payload.get("file_rules", {})
    output_file_name = _format_template(file_rules.get("output_file", "{job_name}.out"), context)
    error_file_name = _format_template(file_rules.get("error_file", "{job_name}.err"), context)
    restart_prefix = _format_template(file_rules.get("restart_prefix", "{job_name}"), context)
    context.update(
        {
            "output_file": output_file_name,
            "error_file": error_file_name,
            "restart_prefix": restart_prefix,
        }
    )

    shell = profile_payload.get("execution", {}).get("shell", "/bin/bash")
    environment = _render_environment(
        profile_payload.get("env", {}),
        context,
        env_overrides=env_overrides,
    )
    launcher = profile_payload.get("launcher", {})
    launcher_kind = launcher.get("kind", "direct")
    declared_program, declared_installation = (
        _declared_profile_installation(profile_payload)
    )
    if declared_installation is None:
        program_command = None
    else:
        program_command = shlex.join(
            _format_template(value, context)
            for value in (
                *declared_installation.launcher_argv,
                *declared_installation.executable_argv,
            )
        )

    rendered: dict[str, Any] = {
        "profile": profile,
        "profiles_path": loaded["__source__"],
        "launcher_kind": launcher_kind,
        "input_file": str(input_obj),
        "job_name": effective_job_name,
        "working_directory": _format_template(
            profile_payload.get("execution", {}).get("working_directory", "{job_dir}"),
            context,
        ),
        "shell": shell,
        "environment": environment,
        "output_file": str(Path(job_dir) / output_file_name),
        "error_file": str(Path(job_dir) / error_file_name),
        "restart_prefix": restart_prefix,
        "resources": resources,
        "executed": False,
    }

    if launcher_kind == "direct":
        launcher_command = _format_template(
            program_command or launcher.get("command", "nwchem"),
            context,
        )
        context["launcher"] = launcher_command
        command = _format_template(
            profile_payload.get("execution", {}).get(
                "command_template",
                "{launcher} {input_file} > {output_file} 2> {error_file}",
            ),
            context,
        )
        rendered["launcher_command"] = launcher_command
        rendered["command"] = command
        # Warn if the launcher command contains unexpanded shell variable placeholders.
        # These expand to empty string in a direct launch; {input_file} in the
        # command_template already passes the input file, making "$1" / "$@" etc. harmful.
        if re.search(r"\$\{?\w", launcher_command):
            rendered["launcher_warnings"] = [
                f"Launcher command contains unexpanded shell variable(s) "
                f"(detected in: {launcher_command!r}). In a direct launch these expand "
                f"to empty string, which causes NWChem to fail to open the input file. "
                f"Remove positional placeholders such as \"$1\" or \"$@\" from the "
                f"launcher command — the input file is already appended by {{input_file}} "
                f"in the command_template."
            ]
        return rendered

    if launcher_kind == "scheduler":
        submit_command = launcher.get("submit_command", "sbatch")
        scheduler = profile_payload.get("scheduler", {})
        modules = profile_payload.get("modules", {})
        hooks = profile_payload.get("hooks", {})
        execution = profile_payload.get("execution", {})
        module_block = _render_module_block(modules)
        pre_run_block = _render_hook_block(hooks.get("pre_run", []), context)
        scheduler_type = (scheduler.get("system") or launcher.get("scheduler_type", "slurm")).lower()
        # Extra context fields for scheduler templates
        nwchem_executable = (
            (
                shlex.join(declared_installation.executable_argv)
                if declared_program == "nwchem"
                and declared_installation is not None
                else None
            )
            or execution.get("nwchem_executable")
            or profile_payload.get("resources", {}).get("nwchem_executable")
            or "nwchem"
        )
        mpi_launch = (
            (
                shlex.join(declared_installation.launcher_argv)
                if declared_program == "nwchem"
                and declared_installation is not None
                else None
            )
            or execution.get("mpi_launch")
            or profile_payload.get("resources", {}).get("mpi_launch")
            or ""
        )
        # Multi-program container placeholders. Different programs put their
        # apptainer image path in different spots in the profile; resolve here
        # so script_template can reference {apptainer_sif} / {container_sif}.
        apptainer_sif = (
            execution.get("apptainer_sif")
            or profile_payload.get("apptainer_sif")
            or ""
        )
        container_sif = profile_payload.get("container_sif") or ""
        pymolcas_command = (
            (
                shlex.join(declared_installation.executable_argv)
                if declared_program == "molcas"
                and declared_installation is not None
                else None
            )
            or execution.get("pymolcas_command")
            or "pymolcas"
        )
        pam_dirac_binary = (
            (
                shlex.join(declared_installation.executable_argv)
                if declared_program == "dirac"
                and declared_installation is not None
                else None
            )
            or profile_payload.get("pam_dirac_binary")
            or "pam-dirac"
        )
        account = context.get("account")
        if account:
            if scheduler_type == "slurm":
                account_line = f"#SBATCH -A {account}"
            elif scheduler_type == "pbs":
                account_line = f"#PBS -A {account}"
            elif scheduler_type == "lsf":
                account_line = f"#BSUB -P {account}"
            else:
                account_line = ""
        else:
            account_line = ""
        scheduler_context = dict(context)
        scheduler_context.update({
            "module_block": module_block,
            "pre_run_block": pre_run_block,
            "nwchem_executable": nwchem_executable,
            "mpi_launch": mpi_launch,
            "account_line": account_line,
            "apptainer_sif": apptainer_sif,
            "container_sif": container_sif,
            "pymolcas_command": pymolcas_command,
            "pam_dirac_binary": pam_dirac_binary,
            "program_command": program_command or "",
            # Default mol_file to empty; DIRAC callers override via context_overrides.
            "mol_file": "",
        })
        # Caller-supplied overrides win (e.g. DIRAC passing mol_file).
        if context_overrides:
            scheduler_context.update(context_overrides)
        script_text = _format_template(scheduler.get("script_template", ""), scheduler_context)
        submit_script_name = _format_template(
            scheduler.get("submit_script_name", "{job_name}.submit"),
            scheduler_context,
        )
        submit_script_path = str(Path(job_dir) / submit_script_name)
        job_id_regex = launcher.get("job_id_regex") or scheduler.get("job_id_regex", r"Submitted batch job (\d+)")
        rendered["submit_script_name"] = submit_script_name
        rendered["submit_script_path"] = submit_script_path
        rendered["submit_script_text"] = script_text
        rendered["submit_command"] = _render_submit_command(submit_command, submit_script_path)
        rendered["job_id_regex"] = job_id_regex
        rendered["scheduler_type"] = scheduler_type
        return rendered

    raise ValueError(f"unsupported launcher kind: {launcher_kind}")


def _render_environment(
    env_template: dict[str, Any],
    context: dict[str, Any],
    *,
    env_overrides: dict[str, str] | None,
) -> dict[str, str]:
    environment = dict(os.environ)
    for key, value in env_template.items():
        if value is None:
            continue
        environment[key] = _format_template(str(value), context)
    for key, value in (env_overrides or {}).items():
        environment[key] = value
    return environment


def _render_module_block(modules: dict[str, Any]) -> str:
    lines: list[str] = []
    if modules.get("purge_first"):
        lines.append("module purge")
    for entry in modules.get("load", []) or []:
        lines.append(f"module load {entry}")
    return "\n".join(lines)


def _render_hook_block(commands: list[str], context: dict[str, Any]) -> str:
    return "\n".join(_format_template(command, context) for command in commands)


def _render_submit_command(submit_command: str, submit_script_path: str) -> list[str]:
    parts = shlex.split(submit_command)
    return parts + [submit_script_path]


# ---------------------------------------------------------------------------
# NWChem compatibility aliases
# ---------------------------------------------------------------------------
# Public NWChem imports predate the multi-program runner. Keep them object
# identical to the canonical functions during the compatibility window.
run_nwchem = run_calculation
render_nwchem_run = render_calculation_run
inspect_nwchem_run_status = inspect_run_status
watch_nwchem_run = watch_run

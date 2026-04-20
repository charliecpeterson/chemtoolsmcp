from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timezone
UTC = timezone.utc
from pathlib import Path
from typing import Any

from .common import detect_program, read_text
from . import nwchem
from .nwchem_input import inspect_nwchem_input


DEFAULT_RUNNER_PROFILES = Path(__file__).resolve().parent / "runner_profiles.example.json"
RUNNER_PROFILES_ENV = "CHEMTOOLS_RUNNER_PROFILES"

# Session-level cache for partition specs (avoids repeated sinfo calls)
_PARTITION_SPECS_CACHE: dict[str, dict[str, Any]] = {}


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


def archive_previous_outputs(job_dir: str, job_name: str) -> list[str]:
    """If {job_name}.out/.err/.job already exist, rename them with a timestamp suffix.

    Returns list of archived file paths.
    """
    archived: list[str] = []
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M")
    for ext in (".out", ".err", ".job"):
        p = Path(job_dir) / f"{job_name}{ext}"
        if p.exists() and p.stat().st_size > 0:
            dest = p.with_name(f"{job_name}{ext}.{ts}")
            # Avoid overwriting an existing archive
            if dest.exists():
                counter = 2
                while dest.exists():
                    dest = p.with_name(f"{job_name}{ext}.{ts}.{counter}")
                    counter += 1
            p.rename(dest)
            archived.append(str(dest))
    return archived


def run_nwchem(
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
) -> dict[str, Any]:
    profiles = load_runner_profiles(profiles_path)
    rendered = render_nwchem_run(
        input_path=input_path,
        profile=profile,
        profiles=profiles,
        job_name=job_name,
        resource_overrides=resource_overrides,
        env_overrides=env_overrides,
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


def render_nwchem_run(
    input_path: str,
    profile: str,
    *,
    profiles: dict[str, Any] | None = None,
    profiles_path: str | None = None,
    job_name: str | None = None,
    resource_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, str] | None = None,
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
        launcher_command = _format_template(launcher.get("command", "nwchem"), context)
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
            execution.get("nwchem_executable")
            or profile_payload.get("resources", {}).get("nwchem_executable")
            or "nwchem"
        )
        mpi_launch = execution.get("mpi_launch") or profile_payload.get("resources", {}).get("mpi_launch") or ""
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
        })
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


def load_runner_profiles(path: str | None = None) -> dict[str, Any]:
    configured_path = path or os.environ.get(RUNNER_PROFILES_ENV)
    source = Path(configured_path).resolve() if configured_path else DEFAULT_RUNNER_PROFILES.resolve()
    if not source.is_file():
        raise ValueError(f"runner profiles file does not exist: {source}")
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json" or text.lstrip().startswith("{"):
        payload = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            # Auto-try a .json sidecar before failing
            json_sidecar = source.with_suffix(".json")
            if json_sidecar.is_file():
                return load_runner_profiles(str(json_sidecar))
            raise ValueError(
                f"YAML runner profiles require PyYAML, or use JSON instead: {source}"
            ) from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"runner profiles file must contain a mapping: {source}")
    payload["__source__"] = str(source)
    return payload


def inspect_nwchem_run_status(
    *,
    output_path: str | None = None,
    input_path: str | None = None,
    error_path: str | None = None,
    process_id: int | None = None,
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    output_info = _file_info(output_path)
    error_info = _file_info(error_path)
    process_status = _process_status(process_id)
    # Auto-detect job_id from .jobid file if not provided
    jobid_stale_warning = None
    if job_id is None:
        _jf = _auto_jobid_path(output_path, input_path)
        if _jf is not None:
            try:
                job_id = _jf.read_text(encoding="utf-8").strip() or None
            except Exception:
                pass
    # Cross-check .jobid against .err file for stale job ID detection
    if job_id:
        _err_path = error_path
        if not _err_path:
            for p in (output_path, input_path):
                if p:
                    base = re.sub(r"\.(out|nw|log|nwout)$", "", str(Path(p).resolve()), flags=re.IGNORECASE)
                    candidate = Path(base + ".err")
                    if candidate.exists():
                        _err_path = str(candidate)
                        break
        err_job_id = _extract_job_id_from_err(_err_path)
        if err_job_id and err_job_id != job_id:
            jobid_stale_warning = (
                f"Stale .jobid detected: file says {job_id} but .err references job {err_job_id}. "
                f"Using {err_job_id} from .err (more recent)."
            )
            job_id = err_job_id
    scheduler_status = None
    if profile and job_id:
        scheduler_status = _scheduler_status(profile=profile, job_id=job_id, profiles_path=profiles_path)

    input_summary = None
    input_raw_text: str | None = None
    if input_path:
        try:
            input_raw_text = read_text(input_path)
            input_summary = inspect_nwchem_input(input_path)
            if input_summary is not None:
                input_summary = dict(input_summary)
                input_summary["raw_text"] = input_raw_text
        except Exception:  # pragma: no cover
            input_summary = None

    parsed_output = None
    compact_summary = None
    progress_summary = None
    task_preview = None
    if output_info["exists"]:
        try:
            contents = read_text(output_info["path"])
            if detect_program(contents) == "nwchem":
                parsed_output = nwchem.parse_tasks(output_info["path"], contents)
                progress_summary = _build_nwchem_progress_summary(
                    contents,
                    parsed_output,
                    input_summary=input_summary,
                )
                compact_summary = _compact_program_summary(
                    parsed_output,
                    progress_summary=progress_summary,
                )
                task_preview = parsed_output.get("generic_tasks", [])[:5]
        except Exception as exc:  # pragma: no cover
            parsed_output = {"error": str(exc), "incomplete": True}

    overall_status = "unknown"
    sched_status = (scheduler_status or {}).get("status")
    if sched_status == "queued":
        overall_status = "queued"
    elif sched_status == "running" or process_status == "running":
        overall_status = "running"
    elif parsed_output and parsed_output.get("program_summary", {}).get("outcome") == "success":
        overall_status = "completed_success"
    elif parsed_output and parsed_output.get("program_summary", {}).get("outcome") == "failed":
        overall_status = "completed_failed"
    elif parsed_output and parsed_output.get("program_summary", {}).get("outcome") == "incomplete":
        overall_status = "completed_incomplete"
    elif sched_status == "failed":
        overall_status = "completed_failed"
    elif sched_status == "cancelled":
        overall_status = "cancelled"
    elif error_info["exists"] and error_info["size_bytes"] > 0:
        overall_status = "error_only"
    elif output_info["exists"]:
        overall_status = "output_present_unknown"
    else:
        overall_status = "not_started"

    result = {
        "output_file": output_info,
        "input_file": {"path": str(Path(input_path).resolve()) if input_path else None, "exists": bool(input_summary)},
        "error_file": error_info,
        "process": {
            "process_id": process_id,
            "status": process_status,
        },
        "scheduler": scheduler_status,
        "output_summary": compact_summary,
        "progress_summary": progress_summary,
        "task_preview": task_preview,
        "parsed_tasks": parsed_output if not compact_summary else None,
        "overall_status": overall_status,
    }
    if jobid_stale_warning:
        result["jobid_stale_warning"] = jobid_stale_warning
    return result


def watch_nwchem_run(
    *,
    output_path: str | None = None,
    input_path: str | None = None,
    error_path: str | None = None,
    process_id: int | None = None,
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
    poll_interval_seconds: float = 10.0,
    adaptive_polling: bool = True,
    max_poll_interval_seconds: float | None = 60.0,
    timeout_seconds: float | None = 3600.0,
    max_polls: int | None = None,
    history_limit: int = 8,
) -> dict[str, Any]:
    if poll_interval_seconds < 0:
        raise ValueError("poll_interval_seconds must be non-negative")
    if max_poll_interval_seconds is not None and max_poll_interval_seconds < 0:
        raise ValueError("max_poll_interval_seconds must be non-negative when provided")
    if timeout_seconds is not None and timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative when provided")
    if max_polls is not None and max_polls <= 0:
        raise ValueError("max_polls must be positive when provided")
    if history_limit <= 0:
        raise ValueError("history_limit must be positive")

    started = time.monotonic()
    poll_count = 0
    snapshots: list[dict[str, Any]] = []
    final_status: dict[str, Any] | None = None
    stop_reason = "unknown"
    terminal = False
    previous_signature: tuple[Any, ...] | None = None
    stable_poll_count = 0
    last_sleep_seconds = 0.0

    while True:
        final_status = inspect_nwchem_run_status(
            output_path=output_path,
            input_path=input_path,
            error_path=error_path,
            process_id=process_id,
            profile=profile,
            job_id=job_id,
            profiles_path=profiles_path,
        )
        poll_count += 1
        elapsed_seconds = time.monotonic() - started
        snapshot = {
            "poll": poll_count,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "overall_status": final_status["overall_status"],
            "current_phase": (final_status.get("output_summary") or {}).get("current_phase"),
            "status_line": ((final_status.get("output_summary") or {}).get("status_line")
                            or (final_status.get("progress_summary") or {}).get("status_line")),
        }
        signature = (
            snapshot["overall_status"],
            snapshot["current_phase"],
            snapshot["status_line"],
            ((final_status.get("process") or {}).get("status")),
            (((final_status.get("output_file") or {}).get("size_bytes"))),
        )
        if not snapshots or snapshot != snapshots[-1]:
            snapshots.append(snapshot)
            if len(snapshots) > history_limit:
                snapshots = snapshots[-history_limit:]

        if previous_signature is None or signature != previous_signature:
            stable_poll_count = 0
        else:
            stable_poll_count += 1
        previous_signature = signature

        if _is_terminal_status(final_status):
            terminal = True
            stop_reason = "terminal_status"
            break
        if max_polls is not None and poll_count >= max_polls:
            stop_reason = "max_polls_reached"
            break
        if timeout_seconds is not None and elapsed_seconds >= timeout_seconds:
            stop_reason = "timeout_reached"
            break
        if poll_interval_seconds > 0:
            last_sleep_seconds = _compute_watch_sleep_seconds(
                base_interval_seconds=poll_interval_seconds,
                stable_poll_count=stable_poll_count,
                adaptive_polling=adaptive_polling,
                max_poll_interval_seconds=max_poll_interval_seconds,
            )
            time.sleep(last_sleep_seconds)

    assert final_status is not None
    return {
        "terminal": terminal,
        "stop_reason": stop_reason,
        "poll_count": poll_count,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "adaptive_polling": adaptive_polling,
        "max_poll_interval_seconds": max_poll_interval_seconds,
        "history_limit": history_limit,
        "last_sleep_seconds": round(last_sleep_seconds, 3),
        "final_status": final_status,
        "history": snapshots,
    }


def tail_text_file(path: str, lines: int = 30, max_characters: int = 4000) -> dict[str, Any]:
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise ValueError(f"file does not exist: {path}")
    contents = file_path.read_text(encoding="utf-8", errors="replace")
    all_lines = contents.splitlines()
    excerpt_lines = all_lines[-lines:] if lines > 0 else all_lines
    excerpt = "\n".join(excerpt_lines)
    if len(excerpt) > max_characters:
        excerpt = excerpt[-max_characters:]
    last_nonempty_line = next((line for line in reversed(excerpt_lines) if line.strip()), None)
    return {
        "path": str(file_path),
        "requested_lines": lines,
        "returned_line_count": len(excerpt_lines),
        "total_line_count": len(all_lines),
        "tail_text": excerpt,
        "last_nonempty_line": last_nonempty_line,
    }


def _resolve_profile(profiles: dict[str, Any], profile_name: str) -> dict[str, Any]:
    defaults = deepcopy(profiles.get("defaults", {}))
    profile = deepcopy((profiles.get("profiles") or {}).get(profile_name))
    if not profile:
        raise ValueError(f"unknown runner profile: {profile_name}")
    return _deep_merge(defaults, profile)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _format_template(template: str | None, context: dict[str, Any]) -> str:
    if template is None:
        return ""
    safe_context = {key: ("" if value is None else value) for key, value in context.items()}
    return template.format_map(safe_context)


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


def _auto_jobid_path(output_path: str | None, input_path: str | None) -> Path | None:
    """Derive the .jobid file location from the output or input path."""
    for p in (output_path, input_path):
        if not p:
            continue
        base = re.sub(r"\.(out|nw|log|nwout|err)$", "", str(Path(p).resolve()), flags=re.IGNORECASE)
        candidate = Path(base + ".jobid")
        if candidate.exists():
            return candidate
    return None


def _extract_job_id_from_err(error_path: str | None) -> str | None:
    """Extract the most recent SLURM job ID from a .err file.

    SLURM writes lines like:
        slurmstepd: error: *** JOB 3031012 ON c511-043 CANCELLED ...
    or general SLURM messages referencing the job ID.
    """
    if not error_path:
        return None
    try:
        p = Path(error_path)
        if not p.exists() or p.stat().st_size == 0:
            return None
        # Read last 4KB to find most recent job ID
        text = p.read_text(encoding="utf-8", errors="replace")[-4096:]
        matches = re.findall(r"JOB\s+(\d+)\s+ON", text)
        if matches:
            return matches[-1]  # most recent
    except Exception:
        pass
    return None


def _file_info(path: str | None) -> dict[str, Any]:
    if not path:
        return {
            "path": None,
            "exists": False,
            "size_bytes": None,
            "modified_utc": None,
        }
    file_path = Path(path).resolve()
    if not file_path.exists():
        return {
            "path": str(file_path),
            "exists": False,
            "size_bytes": None,
            "modified_utc": None,
        }
    stat = file_path.stat()
    return {
        "path": str(file_path),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
    }


def _process_status(process_id: int | None) -> str:
    if process_id is None:
        return "unknown"
    try:
        waited_pid, _ = os.waitpid(process_id, os.WNOHANG)
        if waited_pid == process_id:
            return "exited"
    except ChildProcessError:
        pass
    except OSError:  # pragma: no cover
        pass
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return "not_found"
    except PermissionError:
        return "permission_denied"
    try:
        completed = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(process_id)],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if completed.returncode == 0:
            state = completed.stdout.strip()
            if not state:
                return "not_found"
            if "Z" in state.upper():
                return "zombie"
    except Exception:  # pragma: no cover
        pass
    return "running"


# Scheduler state → normalized status mappings
_SLURM_STATE_MAP: dict[str, str] = {
    "PENDING": "queued", "CONFIGURING": "queued", "SUSPENDED": "queued",
    "RUNNING": "running", "COMPLETING": "running",
    "COMPLETED": "completed",
    "FAILED": "failed", "NODE_FAIL": "failed", "TIMEOUT": "failed",
    "OUT_OF_MEMORY": "failed", "PREEMPTED": "failed", "REVOKED": "failed",
    "DEADLINE": "failed", "BOOT_FAIL": "failed", "SPECIAL_EXIT": "failed",
    "CANCELLED": "cancelled",
}

_PBS_STATE_MAP: dict[str, str] = {
    "Q": "queued", "H": "queued", "W": "queued", "T": "queued", "S": "queued",
    "R": "running", "E": "running",
    "C": "completed", "F": "failed",
}

_LSF_STATE_MAP: dict[str, str] = {
    "PEND": "queued", "PSUSP": "queued", "USUSP": "queued", "SSUSP": "queued",
    "RUN": "running",
    "DONE": "completed",
    "EXIT": "failed", "ZOMBI": "failed",
    "UNKWN": "unknown",
}


def _scheduler_status(profile: str, job_id: str, profiles_path: str | None) -> dict[str, Any]:
    profiles = load_runner_profiles(profiles_path)
    profile_payload = _resolve_profile(profiles, profile)
    launcher = profile_payload.get("launcher", {})
    scheduler = profile_payload.get("scheduler", {})
    status_template = launcher.get("status_command")
    scheduler_type = (scheduler.get("system") or launcher.get("scheduler_type", "slurm")).lower()

    if not status_template:
        return {
            "job_id": job_id,
            "status": "unsupported",
            "scheduler_type": scheduler_type,
            "command": None,
            "return_code": None,
            "raw_state": None,
            "stdout": None,
            "stderr": None,
        }

    command = shlex.split(_format_template(status_template, {"job_id": job_id}))
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    except Exception as exc:
        return {
            "job_id": job_id,
            "status": "error",
            "scheduler_type": scheduler_type,
            "command": command,
            "return_code": None,
            "raw_state": None,
            "error": str(exc),
            "stdout": None,
            "stderr": None,
        }

    stdout = completed.stdout.strip()
    raw_state: str | None = None
    normalized: str

    if scheduler_type == "slurm":
        # Works with "squeue -j {job_id} -h -o %T" (just the state word)
        # or full squeue table — take last token of first non-header line
        for line in stdout.splitlines():
            line = line.strip()
            if line and not line.upper().startswith("JOBID") and not line.startswith("-"):
                raw_state = line.split()[-1].upper()
                break
        if raw_state:
            normalized = _SLURM_STATE_MAP.get(raw_state, "unknown")
        else:
            # Empty squeue output = job aged out of queue (completed long ago)
            normalized = "not_found"
    elif scheduler_type == "pbs":
        m = re.search(r"job_state\s*=\s*(\w+)", stdout, re.IGNORECASE)
        if m:
            raw_state = m.group(1).upper()
            normalized = _PBS_STATE_MAP.get(raw_state, "unknown")
        elif not stdout and completed.returncode != 0:
            normalized = "not_found"
        else:
            normalized = "unknown"
    elif scheduler_type == "lsf":
        lines = [l for l in stdout.splitlines() if l.strip()]
        if len(lines) >= 2:
            header = lines[0].split()
            data = lines[1].split()
            if "STAT" in header and len(data) > header.index("STAT"):
                raw_state = data[header.index("STAT")].upper()
                normalized = _LSF_STATE_MAP.get(raw_state, "unknown")
            else:
                normalized = "unknown"
        elif not stdout and completed.returncode != 0:
            normalized = "not_found"
        else:
            normalized = "unknown"
    else:
        normalized = "running" if (completed.returncode == 0 and stdout) else "not_found"

    return {
        "job_id": job_id,
        "status": normalized,
        "scheduler_type": scheduler_type,
        "raw_state": raw_state,
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def cancel_scheduler_job(profile: str, job_id: str, profiles_path: str | None = None) -> dict[str, Any]:
    """Send a cancel/delete command to the scheduler for the given job ID."""
    profiles = load_runner_profiles(profiles_path)
    profile_payload = _resolve_profile(profiles, profile)
    launcher = profile_payload.get("launcher", {})
    cancel_template = launcher.get("cancel_command")
    if not cancel_template:
        return {"job_id": job_id, "cancelled": False, "error": "no cancel_command configured in profile"}
    command = shlex.split(_format_template(cancel_template, {"job_id": job_id}))
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
        return {
            "job_id": job_id,
            "cancelled": completed.returncode == 0,
            "command": command,
            "return_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:
        return {"job_id": job_id, "cancelled": False, "command": command, "error": str(exc)}


def _is_terminal_status(status: dict[str, Any]) -> bool:
    overall_status = status.get("overall_status")
    if overall_status in {
        "completed_success",
        "completed_failed",
        "completed_incomplete",
        "cancelled",
        "error_only",
    }:
        return True

    scheduler = status.get("scheduler") or {}
    sched_status = scheduler.get("status")
    process = status.get("process") or {}
    output_file = status.get("output_file") or {}

    # HPC: scheduler reports a hard terminal state — job is done regardless of output
    if sched_status in {"failed", "cancelled"}:
        return True
    # HPC: scheduler says completed (or aged out of squeue) and output file exists
    if sched_status in {"completed", "not_found"} and output_file.get("exists"):
        return True

    if (
        overall_status == "output_present_unknown"
        and sched_status not in {"running", "queued"}
        and process.get("status") not in {"running"}
        and output_file.get("exists")
    ):
        return True
    return False


def _compute_watch_sleep_seconds(
    *,
    base_interval_seconds: float,
    stable_poll_count: int,
    adaptive_polling: bool,
    max_poll_interval_seconds: float | None,
) -> float:
    if base_interval_seconds <= 0:
        return 0.0
    if not adaptive_polling:
        return base_interval_seconds
    multiplier = min(2 ** stable_poll_count, 8)
    interval = base_interval_seconds * multiplier
    if max_poll_interval_seconds is not None:
        interval = min(interval, max_poll_interval_seconds)
    return interval


def _compact_program_summary(
    parsed_output: dict[str, Any],
    *,
    progress_summary: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    program_summary = parsed_output.get("program_summary")
    if not isinstance(program_summary, dict):
        return None
    payload = {
        "kind": program_summary.get("kind"),
        "outcome": program_summary.get("outcome"),
        "task_count": program_summary.get("task_count"),
        "diagnostics": program_summary.get("diagnostics"),
    }
    if progress_summary is not None:
        payload["current_task_kind"] = progress_summary.get("current_task_kind")
        payload["current_phase"] = progress_summary.get("current_phase")
        payload["status_line"] = progress_summary.get("status_line")
    return payload


def _detect_slow_phase(contents: str, input_summary: dict[str, Any] | None) -> dict[str, Any]:
    """Identify known output-silent phases so the watcher can report 'expected slow' vs 'hung'.

    Returns a dict with ``phase`` (str|None) and ``message`` (str).
    """
    tail = contents[-8000:] if len(contents) > 8000 else contents
    lower = tail.lower()

    # Check if input uses a relativistic Hamiltonian (X2C, DKH)
    has_relativistic = False
    if input_summary:
        raw_input = input_summary.get("raw_text") or ""
        has_relativistic = bool(
            "relativistic" in raw_input.lower()
            or "x2c" in raw_input.lower()
            or "dkh" in raw_input.lower()
            or "douglas" in raw_input.lower()
        )
    # Also check from output content itself
    if not has_relativistic:
        has_relativistic = any(
            kw in contents.lower()
            for kw in ("x2c hamiltonian", "dkh hamiltonian", "relativistic effects",
                       "scalar relativistic", "x2c-mf", "x2c transform")
        )

    # SAD / initial guess
    if "superposition of atomic density" in lower:
        scf_started = "general information" in lower and (
            "scf calculation" in lower or "dft calculation" in lower
        )
        if not scf_started:
            if has_relativistic:
                return {
                    "phase": "sad_x2c_guess",
                    "message": (
                        "SAD (Superposition of Atomic Density) guess with X2C relativistic Hamiltonian. "
                        "X2C requires solving a relativistic atomic SCF for each unique element — "
                        "for transition metals (e.g. Fe, Ru, W) this can take 30–120+ minutes with "
                        "no output. This is expected; do NOT intervene."
                    ),
                }
            return {
                "phase": "sad_guess",
                "message": (
                    "SAD (Superposition of Atomic Density) guess in progress. "
                    "Output is silent while NWChem builds initial densities — this is normal."
                ),
            }

    # DFT grid generation
    if any(kw in lower for kw in ("xc grid generation", "dft grid", "numerical integration")):
        grid_done = "grid construction" in lower and "done" in lower
        if not grid_done:
            return {
                "phase": "dft_grid_generation",
                "message": (
                    "DFT numerical integration grid generation in progress. "
                    "For large molecules or fine grids this can be slow with no output."
                ),
            }

    # Frequency / Hessian numerical differentiation
    if any(kw in lower for kw in ("nuclear hessian", "freq task", "p.frequency", "normal mode")):
        freq_done = "frequency analysis" in lower and ("done" in lower or "completed" in lower)
        if not freq_done:
            return {
                "phase": "frequency_numerical_hessian",
                "message": (
                    "Frequency/Hessian numerical differentiation in progress. "
                    "Each displacement requires a full energy+gradient calculation; "
                    "output may be sparse between displacements."
                ),
            }

    # TCE AO→MO integral transformation
    if any(kw in lower for kw in ("tce", "ao-to-mo", "transformation of integrals", "integral transformation")):
        tce_iter = "iterative solution" in lower or "ccsd iteration" in lower
        if not tce_iter:
            return {
                "phase": "tce_ao_mo_transform",
                "message": (
                    "TCE AO→MO integral transformation in progress. "
                    "For large basis sets this takes significant time and memory with no output."
                ),
            }

    # Geometry optimization between steps (writing/reading Hessian)
    if any(kw in lower for kw in ("driver: starting", "geometry optimization", "optimize:")):
        return {
            "phase": "geometry_optimization_step",
            "message": "Geometry optimization step in progress.",
        }

    return {"phase": None, "message": ""}


def _build_nwchem_progress_summary(
    contents: str,
    parsed_output: dict[str, Any],
    *,
    input_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_tasks = ((parsed_output.get("program_summary") or {}).get("raw") or {}).get("tasks") or []
    last_task = raw_tasks[-1] if raw_tasks else None
    trajectory = nwchem.parse_trajectory("<status>", contents)
    frequency_started = any(
        marker in contents
        for marker in (
            "NORMAL MODE EIGENVECTORS",
            "NWChem Nuclear Hessian and Frequency Analysis",
            "P.Frequency",
        )
    )
    frequency = nwchem.parse_freq("<status>", contents) if frequency_started else None

    summary: dict[str, Any] = {
        "current_task_kind": last_task.get("kind") if last_task else None,
        "current_task_label": last_task.get("label") if last_task else None,
        "current_phase": "unknown",
        "optimization_status": trajectory.get("optimization_status"),
        "optimization_step_count": trajectory.get("step_count"),
        "optimization_last_step": trajectory.get("last_step"),
        "optimization_final_energy_hartree": trajectory.get("final_energy_hartree"),
        "optimization_unmet_criteria": trajectory.get("unmet_criteria"),
        "frequency_started": frequency_started,
        "frequency_mode_count": frequency.get("mode_count") if frequency is not None else 0,
        "significant_imaginary_mode_count": (
            frequency.get("significant_imaginary_mode_count") if frequency is not None else 0
        ),
        "status_line": None,
    }
    if input_summary is not None:
        requested = _summarize_requested_task_progress(input_summary, raw_tasks)
        summary.update(requested)

    # --- Detect known output-silent phases before first task completes ---
    slow_phase = _detect_slow_phase(contents, input_summary)
    summary["slow_phase"] = slow_phase.get("phase")
    summary["slow_phase_message"] = slow_phase.get("message")

    if last_task is None:
        if slow_phase.get("phase"):
            summary["current_phase"] = "initialization_slow_phase"
            summary["status_line"] = slow_phase["message"]
        else:
            summary["status_line"] = "No NWChem task structure detected yet."
        return summary

    kind = last_task.get("kind")
    outcome = last_task.get("outcome")

    if kind == "optimization":
        if trajectory.get("optimization_status") == "converged":
            summary["current_phase"] = "optimization_completed"
            summary["status_line"] = (
                f"Optimization converged after {trajectory.get('step_count') or 0} steps."
            )
        else:
            summary["current_phase"] = "optimization_in_progress"
            status_bits = [f"Optimization {trajectory.get('optimization_status') or outcome}"]
            if trajectory.get("last_step") is not None:
                status_bits.append(f"last completed step {trajectory['last_step']}")
            if trajectory.get("final_energy_hartree") is not None:
                status_bits.append(f"energy {trajectory['final_energy_hartree']:.12f} Ha")
            if trajectory.get("unmet_criteria"):
                status_bits.append("unmet " + ", ".join(trajectory["unmet_criteria"]))
            summary["status_line"] = "; ".join(status_bits) + "."
        if not frequency_started:
            summary["frequency_status"] = "not_started"
        elif frequency is not None and frequency.get("mode_count", 0) > 0:
            summary["frequency_status"] = "started"
        else:
            summary["frequency_status"] = "not_detected"
        _annotate_status_with_requested_tasks(summary)
        return summary

    if kind in {"frequency", "raman"}:
        if outcome == "success" and frequency is not None and frequency.get("mode_count", 0) > 0:
            summary["current_phase"] = "frequency_completed"
            summary["status_line"] = (
                f"Frequency task completed with {frequency['mode_count']} modes."
            )
        else:
            summary["current_phase"] = "frequency_in_progress_or_interrupted"
            summary["status_line"] = "Frequency task has started but is not complete."
        summary["frequency_status"] = "started"
        _annotate_status_with_requested_tasks(summary)
        return summary

    if kind in {"single_point", "property"}:
        summary["current_phase"] = f"{kind}_task"
        summary["status_line"] = f"{last_task.get('label', 'Task')} is {outcome}."
        _annotate_status_with_requested_tasks(summary)
        return summary

    summary["current_phase"] = "other"
    summary["status_line"] = f"{last_task.get('label', 'Task')} is {outcome}."
    _annotate_status_with_requested_tasks(summary)
    return summary


def _summarize_requested_task_progress(
    input_summary: dict[str, Any],
    raw_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    requested_tasks = [_normalize_requested_task(task) for task in input_summary.get("tasks", [])]
    observed_tasks = [
        {
            "kind": task.get("kind"),
            "label": task.get("label"),
            "outcome": task.get("outcome"),
        }
        for task in raw_tasks
    ]

    completed_count = 0
    current_requested_task = None
    next_requested_task = None
    observed_current_task = None

    observed_idx = 0
    for requested in requested_tasks:
        matched = None
        while observed_idx < len(observed_tasks):
            candidate = observed_tasks[observed_idx]
            if candidate.get("kind") == requested.get("kind"):
                matched = candidate
                observed_idx += 1
                break
            observed_idx += 1
        if matched is None:
            current_requested_task = requested
            break
        if matched.get("outcome") == "success":
            completed_count += 1
            continue
        current_requested_task = requested
        observed_current_task = matched
        break

    if current_requested_task is None and completed_count < len(requested_tasks):
        current_requested_task = requested_tasks[completed_count]

    if current_requested_task is not None:
        current_index = requested_tasks.index(current_requested_task)
        if current_index + 1 < len(requested_tasks):
            next_requested_task = requested_tasks[current_index + 1]

    return {
        "requested_tasks": requested_tasks,
        "requested_task_count": len(requested_tasks),
        "observed_task_sequence": observed_tasks,
        "completed_requested_task_count": completed_count,
        "current_requested_task": current_requested_task,
        "next_requested_task": next_requested_task,
        "observed_current_task": observed_current_task,
    }


def _normalize_requested_task(task: dict[str, Any]) -> dict[str, Any]:
    module = (task.get("module") or "").lower()
    operation = (task.get("operation") or "").lower()

    if operation in {"opt", "optimize", "saddle"}:
        kind = "optimization"
    elif operation in {"freq", "frequency", "frequencies", "hessian", "raman"}:
        kind = "frequency"
    elif operation in {"property", "prop"}:
        kind = "property"
    elif operation in {"energy", "gradient", ""}:
        kind = "single_point"
    else:
        kind = operation or "other"

    label_parts = [part for part in (module.upper() if module else None, operation or None) if part]
    label = " ".join(label_parts) if label_parts else kind
    return {
        "module": module or None,
        "operation": operation or None,
        "kind": kind,
        "label": label,
    }


def _annotate_status_with_requested_tasks(summary: dict[str, Any]) -> None:
    current_requested = summary.get("current_requested_task")
    next_requested = summary.get("next_requested_task")
    if not current_requested:
        return
    suffix = f" Requested task: {current_requested['label']}."
    if next_requested:
        suffix += f" Next task not started: {next_requested['label']}."
    status_line = summary.get("status_line") or ""
    if suffix.strip() not in status_line:
        summary["status_line"] = f"{status_line}{suffix}"

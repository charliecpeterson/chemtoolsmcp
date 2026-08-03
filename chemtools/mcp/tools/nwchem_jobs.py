"""NWChem MCP handlers — jobs.

Split from mcp/tools/nwchem.py by category. Shared imports/helpers live in
_nwchem_base (pulled in below); nwchem.py imports this module so its @_tool
handlers register.
"""
from __future__ import annotations

from chemtools.mcp.tools._nwchem_base import *  # noqa: F401,F403
from chemtools.mcp.tools._nwchem_base import _tool, _build_next_actions  # noqa: F401
from chemtools.application.nwchem_execution import (
    launch_nwchem_with_service,
    register_nwchem_launch_with_service,
    terminate_nwchem_with_service,
)
from chemtools.application.nwchem_monitoring import (
    inspect_nwchem_status_with_service,
    watch_nwchem_status_with_service,
)
from chemtools.application.execution import LaunchStatusError
from chemtools.mcp.decorator import get_execution_service


@_tool("plan_nwchem_workflow")
def _handle_plan_nwchem_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    return plan_nwchem_workflow(
        goal=arguments["goal"],
        elements=arguments["elements"],
        charge=arguments["charge"],
        multiplicity=arguments["multiplicity"],
        basis=arguments.get("basis"),
        method=arguments.get("method", "ccsd"),
        xc_functional=arguments.get("xc_functional", "b3lyp"),
        has_geometry_file=arguments.get("has_geometry_file", False),
        has_dft_output=arguments.get("has_dft_output", False),
        has_scf_output=arguments.get("has_scf_output", False),
    )


@_tool("find_nwchem_restart_assets")
def _handle_find_nwchem_restart_assets(arguments: dict[str, Any]) -> dict[str, Any]:
    return find_restart_assets(arguments["path"])


# ---------------------------------------------------------------------------
# Handlers — runner / job management
# ---------------------------------------------------------------------------


@_tool("launch_nwchem_run", needs="executable")
def _handle_launch_nwchem_run(arguments: dict[str, Any]) -> dict[str, Any]:
    dry_run = arguments.get("dry_run", False)
    auto_watch = arguments.get("auto_watch", True)
    auto_register = arguments.get("auto_register", True)
    execution_service = get_execution_service()
    result = launch_nwchem_with_service(
        execution_service,
        input_path=arguments["input_file"],
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path"),
        job_name=arguments.get("job_name"),
        resource_overrides=arguments.get("resource_overrides"),
        env_overrides=arguments.get("env_overrides"),
        write_script=arguments.get("write_script", True),
        dry_run=dry_run,
    )
    # Auto-register in the run registry
    if not dry_run and auto_register:
        try:
            reg = register_nwchem_launch_with_service(
                execution_service,
                launch_id=result["launch_id"],
                job_name=result.get("job_name", arguments.get("job_name", "")),
                input_file=arguments["input_file"],
                profile=arguments["profile"],
                campaign_id=arguments.get("campaign_id"),
                workflow_id=arguments.get("workflow_id"),
                workflow_step_id=arguments.get("workflow_step_id"),
                parent_run_id=arguments.get("parent_run_id"),
            )
            result["registry"] = reg
        except Exception as exc:
            result["registry_error"] = str(exc)

    # For scheduler jobs: automatically watch until terminal unless opted out
    if (
        not dry_run
        and auto_watch
        and result.get("launcher_kind") == "scheduler"
        and result.get("job_id")
    ):
        out_file = result.get("output_file")
        in_file = arguments["input_file"]
        profiles_path = arguments.get("profiles_path")
        profile = arguments["profile"]
        watch_result = watch_nwchem_status_with_service(
            execution_service,
            job_id=result["job_id"],
            profile=profile,
            output_path=out_file,
            input_path=in_file,
            error_path=result.get("error_file"),
            profiles_path=profiles_path,
            poll_interval_seconds=30.0,
            adaptive_polling=True,
            max_poll_interval_seconds=120.0,
            timeout_seconds=None,
        )
        result["watch"] = watch_result
    return result


@_tool("get_nwchem_run_status", needs="executable")
def _handle_get_nwchem_run_status(arguments: dict[str, Any]) -> dict[str, Any]:
    process_id = arguments.get("process_id")
    job_id = arguments.get("job_id")
    profile = arguments.get("profile")
    status = inspect_nwchem_status_with_service(
        get_execution_service(),
        output_path=arguments.get("output_file"),
        input_path=arguments.get("input_file"),
        error_path=arguments.get("error_file"),
        process_id=process_id,
        profile=profile,
        job_id=job_id,
        profiles_path=arguments.get("profiles_path"),
    )
    inspected_process_id = (
        process_id
        if (status.get("process") or {}).get("status") == "running"
        else None
    )
    scheduler = status.get("scheduler") or {}
    typed_scheduler = scheduler.get("source") in {
        "queue",
        "accounting",
        "record",
    }
    inspected_profile = None if typed_scheduler else profile
    inspected_job_id = None if typed_scheduler else job_id
    # Add compact progress summary when output file is available
    if arguments.get("output_file"):
        try:
            progress = review_nwchem_progress(
                output_path=arguments["output_file"],
                input_path=arguments.get("input_file"),
                error_path=arguments.get("error_file"),
                process_id=inspected_process_id,
                profile=inspected_profile,
                job_id=inspected_job_id,
                profiles_path=arguments.get("profiles_path"),
            )
            status["progress"] = progress
        except Exception as exc:
            status["progress_error"] = str(exc)
    status["next_actions"] = _build_next_actions(
        "run_status", status,
        output_file=arguments.get("output_file", ""),
        input_file=arguments.get("input_file", ""),
        profile=arguments.get("profile", ""),
    )
    return status


@_tool("tail_nwchem_output", needs="executable")
def _handle_tail_nwchem_output(arguments: dict[str, Any]) -> dict[str, Any]:
    return tail_nwchem_output(
        arguments["output_file"],
        lines=arguments.get("lines", 30),
        max_characters=min(arguments.get("max_characters", 4000), 10000),
    )


@_tool("terminate_nwchem_run", needs="executable")
def _handle_terminate_nwchem_run(arguments: dict[str, Any]) -> dict[str, Any]:
    return terminate_nwchem_with_service(
        get_execution_service(),
        process_id=arguments.get("process_id"),
        signal_name=arguments.get("signal_name", "term"),
        job_id=arguments.get("job_id"),
        profile=arguments.get("profile"),
    )


@_tool("watch_nwchem_run", needs="executable")
def _handle_watch_nwchem_run(arguments: dict[str, Any]) -> dict[str, Any]:
    job_id = arguments.get("job_id")
    process_id = arguments.get("process_id")
    profile = arguments.get("profile")
    service = get_execution_service()

    watch_arguments = {
        "output_path": arguments.get("output_file"),
        "input_path": arguments.get("input_file"),
        "error_path": arguments.get("error_file"),
        "profiles_path": arguments.get("profiles_path"),
        "poll_interval_seconds": arguments.get(
            "poll_interval_seconds",
            10.0,
        ),
        "adaptive_polling": arguments.get("adaptive_polling", True),
        "max_poll_interval_seconds": arguments.get(
            "max_poll_interval_seconds",
            60.0,
        ),
        "timeout_seconds": arguments.get("timeout_seconds", 3600.0),
        "max_polls": arguments.get("max_polls"),
        "history_limit": arguments.get("history_limit", 8),
    }
    result = None
    if job_id is not None or process_id is not None:
        try:
            result = watch_nwchem_status_with_service(
                service,
                process_id=process_id if job_id is None else None,
                job_id=job_id,
                profile=profile,
                **watch_arguments,
            )
        except LaunchStatusError as exc:
            if exc.as_dict()["error"] != "launch_not_owned":
                raise
    if result is None:
        result = watch_nwchem_run(
            process_id=process_id,
            profile=profile,
            job_id=job_id,
            **watch_arguments,
        )
    result["next_actions"] = _build_next_actions(
        "watch_run", result,
        output_file=arguments.get("output_file", ""),
        input_file=arguments.get("input_file", ""),
        profile=arguments.get("profile", ""),
    )
    return result


# ---------------------------------------------------------------------------
# Handlers — run comparison and follow-up
# ---------------------------------------------------------------------------


@_tool("get_nwchem_workflow_state")
def _handle_get_nwchem_workflow_state(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_nwchem_workflow_state(
        input_file=arguments.get("input_file"),
        output_file=arguments["output_file"],
        profile=arguments.get("profile", ""),
        error_file=arguments.get("error_file"),
    )


@_tool("plan_nwchem_calculation")
def _handle_plan_nwchem_calculation(arguments: dict[str, Any]) -> dict[str, Any]:
    return plan_calculation(
        input_file=arguments["input_file"],
        protocol=arguments["protocol"],
        profile=arguments.get("profile", ""),
        output_dir=arguments.get("output_dir"),
        overrides=arguments.get("overrides"),
    )


@_tool("list_nwchem_protocols")
def _handle_list_nwchem_protocols(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"protocols": list_protocols()}


@_tool("register_nwchem_run", needs="registry")
def _handle_register_run(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy NWChem-tagged registry — pre-fills program='nwchem'.

    Equivalent to ``register_run(..., program='nwchem')``. Kept for one
    release so older agents/tests don't break.
    """
    return register_run(
        program="nwchem",
        job_name=arguments["job_name"],
        input_file=arguments.get("input_file"),
        output_file=arguments.get("output_file"),
        profile=arguments.get("profile"),
        method=arguments.get("method"),
        functional=arguments.get("functional"),
        basis=arguments.get("basis"),
        n_atoms=arguments.get("n_atoms"),
        elements=arguments.get("elements"),
        charge=arguments.get("charge"),
        multiplicity=arguments.get("multiplicity"),
        mpi_ranks=arguments.get("mpi_ranks"),
        campaign_id=arguments.get("campaign_id"),
        workflow_id=arguments.get("workflow_id"),
        workflow_step_id=arguments.get("workflow_step_id"),
        parent_run_id=arguments.get("parent_run_id"),
        tags=arguments.get("tags"),
    )


# --- Registry: per-run status + lookup ---


def _do_update_run_status(arguments: dict[str, Any]) -> dict[str, Any]:
    return update_run_status(
        run_id=arguments["run_id"],
        status=arguments["status"],
        energy_hartree=arguments.get("energy_hartree"),
        h_hartree=arguments.get("h_hartree"),
        g_hartree=arguments.get("g_hartree"),
        imaginary_modes=arguments.get("imaginary_modes"),
        walltime_used_sec=arguments.get("walltime_used_sec"),
        sec_per_gradient=arguments.get("sec_per_gradient"),
        output_file=arguments.get("output_file"),
    )


@_tool("update_nwchem_run_status", needs="registry")
def _handle_update_run_status(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy alias for update_run_status. Run-status updates aren't
    program-specific; the run_id selects the row to modify."""
    return _do_update_run_status(arguments)


def _do_list_runs(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"runs": list_runs(
        campaign_id=arguments.get("campaign_id"),
        workflow_id=arguments.get("workflow_id"),
        status=arguments.get("status"),
        method=arguments.get("method"),
        program=arguments.get("program"),
        limit=arguments.get("limit", 50),
    )}


@_tool("list_nwchem_runs", needs="registry")
def _handle_list_runs(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy: list runs. Pre-fills program='nwchem' if no program filter
    is given (preserves the historical NWChem-only return for callers
    that haven't migrated)."""
    args = dict(arguments)
    if args.get("program") is None:
        args["program"] = "nwchem"
    return _do_list_runs(args)


def _do_get_run_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_run_summary(
        run_id=arguments.get("run_id"),
        job_name=arguments.get("job_name"),
    )


@_tool("get_nwchem_run_summary", needs="registry")
def _handle_get_run_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy alias — fetches by run_id or job_name; program-agnostic."""
    return _do_get_run_summary(arguments)


# --- Registry: campaigns ---


def _do_create_campaign(arguments: dict[str, Any]) -> dict[str, Any]:
    return create_campaign(
        name=arguments["name"],
        description=arguments.get("description"),
        tags=arguments.get("tags"),
    )


@_tool("create_nwchem_campaign", needs="registry")
def _handle_create_campaign(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy alias. Campaigns are cross-program by design — no program tag
    on campaigns themselves; runs inside the campaign carry their own."""
    return _do_create_campaign(arguments)


def _do_get_campaign_status(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_campaign_status(
        campaign_id=arguments.get("campaign_id"),
        name=arguments.get("name"),
    )


@_tool("get_nwchem_campaign_status", needs="registry")
def _handle_get_campaign_status(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy alias."""
    return _do_get_campaign_status(arguments)


def _do_get_campaign_energies(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_campaign_energies(
        campaign_id=arguments.get("campaign_id"),
        name=arguments.get("name"),
    )


@_tool("get_nwchem_campaign_energies", needs="registry")
def _handle_get_campaign_energies(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy alias. Each returned row now includes the run's program tag."""
    return _do_get_campaign_energies(arguments)


# --- Registry: workflows ---


def _do_create_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    return create_workflow(
        name=arguments["name"],
        steps=arguments["steps"],
        protocol=arguments.get("protocol"),
        campaign_id=arguments.get("campaign_id"),
    )


@_tool("create_nwchem_workflow", needs="registry")
def _handle_create_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy alias. Workflows themselves are cross-program; the per-step
    `program` field inside the steps_json controls each run."""
    return _do_create_workflow(arguments)


def _do_advance_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    return advance_workflow(workflow_id=arguments["workflow_id"])


@_tool("advance_nwchem_workflow", needs="registry")
def _handle_advance_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    """Legacy alias."""
    return _do_advance_workflow(arguments)


@_tool("generate_nwchem_input_batch", needs="executable_or_scheduler")
def _handle_generate_input_batch(arguments: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        template_input=arguments["template_input"],
        vary=arguments["vary"],
        output_dir=arguments["output_dir"],
    )
    if arguments.get("naming_pattern"):
        kwargs["naming_pattern"] = arguments["naming_pattern"]
    if arguments.get("campaign_id") is not None:
        kwargs["campaign_id"] = arguments["campaign_id"]
    return generate_input_batch(**kwargs)


@_tool("check_nwchem_memory_fit", needs="executable_or_scheduler")
def _handle_check_memory_fit(arguments: dict[str, Any]) -> dict[str, Any]:
    profile_resources = None
    if arguments.get("profile"):
        from chemtools.core.runner import load_runner_profiles, _resolve_profile
        profiles_path = arguments.get("profiles_path")
        loaded = load_runner_profiles(profiles_path)
        resolved = _resolve_profile(loaded, arguments["profile"])
        profile_resources = resolved.get("resources", {})
        # Merge resource_overrides if present
        if arguments.get("resource_overrides"):
            profile_resources = {**profile_resources, **arguments["resource_overrides"]}
    kwargs: dict[str, Any] = {
        "input_file": arguments["input_file"],
        "profile_resources": profile_resources,
    }
    if "nodes" in arguments:
        kwargs["nodes"] = arguments["nodes"]
    if "mpi_ranks" in arguments:
        kwargs["mpi_ranks"] = arguments["mpi_ranks"]
    if "node_memory_mb" in arguments:
        kwargs["node_memory_mb"] = arguments["node_memory_mb"]
    return check_memory_fit(**kwargs)


@_tool("estimate_nwchem_freq_walltime", needs="executable_or_scheduler")
def _handle_estimate_freq_walltime(arguments: dict[str, Any]) -> dict[str, Any]:
    return estimate_freq_walltime(
        n_atoms=arguments["n_atoms"],
        seconds_per_displacement=arguments.get("seconds_per_displacement"),
        n_displacements=arguments.get("n_displacements"),
        mpi_ranks=arguments.get("mpi_ranks", 1),
        nodes=arguments.get("nodes", 1),
        max_walltime_hours=arguments.get("max_walltime_hours", 48.0),
    )


@_tool("suggest_nwchem_resources", needs="executable_or_scheduler")
def _handle_suggest_hpc_resources(arguments: dict[str, Any]) -> dict[str, Any]:
    return suggest_hpc_resources(
        input_file=arguments["input_file"],
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path"),
    )


@_tool("detect_nwchem_hpc_accounts", needs="scheduler")
def _handle_detect_hpc_accounts(arguments: dict[str, Any]) -> dict[str, Any]:
    return detect_hpc_accounts(
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path"),
    )


@_tool("suggest_nwchem_partition", needs="scheduler")
def _handle_suggest_partition(arguments: dict[str, Any]) -> dict[str, Any]:
    return suggest_partition(
        input_file=arguments["input_file"],
        profiles_path=arguments.get("profiles_path"),
        check_queue=arguments.get("check_queue", True),
    )


# ---------------------------------------------------------------------------
# Handlers — NWChem documentation (bundled)
# ---------------------------------------------------------------------------

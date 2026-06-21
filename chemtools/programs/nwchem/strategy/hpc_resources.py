"""NWChem HPC resource and queue-account advisors.

Three entry points for HPC job sizing and submission:

  * detect_hpc_accounts     Discover available SLURM/PBS allocations
                            for a runner profile.
  * suggest_hpc_resources   Recommend nodes / ranks / walltime / memory
                            for an NWChem input on a given runner profile.
                            Combines input analysis with profile hardware.
  * suggest_partition       Recommend the right partition / queue for a
                            given job size on a SLURM cluster.

All three pull from chemtools/core/runner.py for profile loading and
partition specs. _analyze_job_size lives in api_strategy.py (still flat
for now) and is lazy-imported below to avoid a cycle.
"""

from __future__ import annotations
import math
import re
from pathlib import Path
from typing import Any

from chemtools.core.runner import (
    load_runner_profiles,
    _resolve_profile,
)
from chemtools.programs.nwchem.strategy.resources import suggest_memory, _BF_PER_RANK_TARGET
from chemtools.programs.nwchem.strategy.workflow_state import _format_walltime, _parse_walltime_hours


def _job_size(input_file: str) -> dict[str, Any]:
    """Lazy proxy for api_strategy._analyze_job_size — avoids the cycle."""
    from chemtools.api_strategy import _analyze_job_size
    return _analyze_job_size(input_file)


def detect_hpc_accounts(
    profile: str,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    """Detect available HPC allocation accounts for a runner profile.

    Runs the profile's ``account_command`` (e.g. ``/usr/local/etc/taccinfo``)
    and parses the output to find project names and available SUs.

    Returns a dict with ``accounts`` list (each with name, avail_sus, expires)
    and ``recommended`` (the account with the most SUs remaining).
    """
    import subprocess
    from chemtools.core.runner import load_runner_profiles, _resolve_profile

    loaded = load_runner_profiles(profiles_path)
    profile_payload = _resolve_profile(loaded, profile)
    res = profile_payload.get("resources", {})
    account_cmd = res.get("account_command")

    if not account_cmd:
        static_account = res.get("account")
        if static_account:
            return {
                "accounts": [{"name": static_account, "avail_sus": None, "expires": None}],
                "recommended": static_account,
                "source": "profile_static",
            }
        return {
            "accounts": [],
            "recommended": None,
            "source": "none",
            "message": "No account_command or static account in profile. "
                       "Set resources.account or resources.account_command.",
        }

    try:
        result = subprocess.run(
            account_cmd, shell=True, capture_output=True, text=True, timeout=15,
        )
        output = result.stdout
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {
            "accounts": [],
            "recommended": res.get("account"),
            "source": "error",
            "message": f"Failed to run account_command: {exc}",
        }

    # Parse taccinfo-style output:
    #   | Name           Avail SUs     Expires |
    #   | TG-CHE250093         818  2026-06-08 |
    # Also handle generic formats: "account_name  SUs  date"
    import re
    accounts: list[dict[str, Any]] = []
    for line in output.splitlines():
        # Strip table borders
        stripped = line.strip().strip("|").strip()
        if not stripped or stripped.startswith("-") or "Name" in stripped:
            continue
        # Try to match: project_name  number  date
        m = re.match(
            r"(\S+)\s+(\d+)\s+(\d{4}-\d{2}-\d{2})", stripped
        )
        if m:
            accounts.append({
                "name": m.group(1),
                "avail_sus": int(m.group(2)),
                "expires": m.group(3),
            })

    # Pick the account with the most SUs as recommended
    recommended = None
    if accounts:
        best = max(accounts, key=lambda a: a["avail_sus"] or 0)
        recommended = best["name"]

    return {
        "accounts": accounts,
        "recommended": recommended,
        "source": "account_command",
        "command": account_cmd,
        "raw_output": output.strip(),
    }


def suggest_hpc_resources(
    input_file: str,
    profile: str,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    """Recommend optimal HPC resources for a NWChem job based on profile hardware specs.

    Analyzes the input file (atoms, basis, method, task type) and the profile's
    hardware description (cores_per_node, node_memory_mb, max_nodes, max_walltime,
    cpu_arch) to recommend:
    - nodes, mpi_ranks, walltime
    - NWChem memory directive
    - partition (if profile specifies one)
    - account (auto-detected from account_command if available)

    Returns a dict with recommended ``resource_overrides`` ready to pass to
    ``launch_nwchem_run``, plus rationale explaining the choices.

    Args:
        input_file: Path to the NWChem .nw input file.
        profile: Runner profile name (must have hardware fields populated).
        profiles_path: Optional path to profiles YAML/JSON.
    """
    from chemtools.core.runner import load_runner_profiles, _resolve_profile

    # --- Load profile hardware specs ---
    loaded = load_runner_profiles(profiles_path)
    profile_payload = _resolve_profile(loaded, profile)
    res = profile_payload.get("resources", {})

    cores_per_node = res.get("cores_per_node") or res.get("mpi_ranks") or 48
    node_memory_mb = res.get("node_memory_mb")
    max_nodes = res.get("max_nodes") or 1
    max_wt_str = res.get("max_walltime")
    max_wt_hours = _parse_walltime_hours(max_wt_str) or 48.0
    cpu_arch = res.get("cpu_arch") or "generic"
    partition = res.get("partition")

    # --- Analyze input file ---
    job = _job_size(input_file)
    n_atoms = job["n_atoms"]
    n_heavy = job["n_heavy"]
    n_bf = job["n_bf"]
    module = job["module"]
    operation = job["operation"]
    is_freq = job["is_freq"]
    is_opt = job["is_opt"]
    is_tce = job["is_tce"]
    basis_name = job["basis_name"]

    rationale: list[str] = []
    warnings: list[str] = []

    # --- Step 1: Determine optimal MPI ranks per node ---
    bf_per_rank = _BF_PER_RANK_TARGET.get(cpu_arch, 80)
    ranks_by_scaling = max(1, n_bf // bf_per_rank)

    # Memory constraint on ranks per node
    if node_memory_mb:
        # Reserve 15% for OS + MPI
        usable_mb = int(node_memory_mb * 0.85)
        # Need at least 400 MB/rank for NWChem to start
        max_ranks_by_memory = max(1, usable_mb // 400)
    else:
        max_ranks_by_memory = cores_per_node

    ranks_per_node = min(ranks_by_scaling, max_ranks_by_memory, cores_per_node)
    ranks_per_node = max(1, ranks_per_node)

    # For small molecules, don't use more ranks than useful
    if ranks_by_scaling < cores_per_node // 2:
        ranks_per_node = max(1, ranks_by_scaling)
        rationale.append(
            f"Small molecule ({n_atoms} atoms, ~{n_bf} BF): using {ranks_per_node} "
            f"ranks/node instead of full {cores_per_node} cores for efficiency"
        )
    else:
        ranks_per_node = min(cores_per_node, max_ranks_by_memory)
        rationale.append(
            f"{n_atoms} atoms, ~{n_bf} BF, {bf_per_rank} BF/rank target → "
            f"{ranks_per_node} ranks/node (of {cores_per_node} available)"
        )

    # --- Step 2: Determine number of nodes ---
    nodes = 1

    if is_freq:
        # Numerical freq: 6*N_atoms displacements, no checkpoint, need to finish in one go
        n_displacements = n_atoms * 6
        # Rough heuristic: seconds per displacement
        base_seconds = 300.0 * (n_atoms / 20.0) ** 1.5
        total_ranks_1node = ranks_per_node
        eff_1 = min(total_ranks_1node, 64) + max(0, total_ranks_1node - 64) * 0.3
        secs_per_disp_1 = base_seconds * (48.0 / max(1, eff_1))
        hours_1node = (n_displacements * secs_per_disp_1) / 3600.0

        if hours_1node > max_wt_hours * 0.85:
            # Need multi-node
            for n in range(2, max_nodes + 1):
                total_ranks_n = ranks_per_node * n
                eff_n = min(total_ranks_n, 64) + max(0, total_ranks_n - 64) * 0.3
                secs_n = base_seconds * (48.0 / max(1, eff_n))
                hours_n = (n_displacements * secs_n) / 3600.0
                if hours_n <= max_wt_hours * 0.85:
                    nodes = n
                    break
            else:
                nodes = min(max_nodes, 8)
                warnings.append(
                    f"Frequency job estimated at {hours_1node:.0f}h on 1 node. "
                    f"Even with {nodes} nodes it may exceed {max_wt_hours:.0f}h walltime. "
                    f"Consider analytical frequencies or a smaller basis."
                )
            rationale.append(
                f"Numerical freq: {n_displacements} displacements, estimated "
                f"{hours_1node:.0f}h on 1 node → {nodes} nodes to fit in "
                f"{max_wt_hours:.0f}h walltime"
            )
        else:
            rationale.append(
                f"Numerical freq: {n_displacements} displacements, estimated "
                f"{hours_1node:.1f}h — fits on 1 node"
            )
        warnings.append(
            "NWChem CANNOT checkpoint numerical frequencies. If the job "
            "exceeds walltime, ALL progress is lost."
        )

    elif is_tce:
        # TCE is memory-hungry — may need multi-node for memory
        mem_rec = suggest_memory(n_atoms=n_atoms, basis=basis_name or "6-31g*",
                                method="tce", n_heavy_atoms=n_heavy)
        total_mem_needed = mem_rec["recommended_total_mb"] * ranks_per_node
        if node_memory_mb and total_mem_needed > node_memory_mb * 0.80:
            nodes = min(max_nodes, max(2, math.ceil(
                total_mem_needed / (node_memory_mb * 0.80)
            )))
            rationale.append(
                f"TCE: estimated {total_mem_needed} MB total memory needed, "
                f"node has {node_memory_mb} MB → {nodes} nodes for memory"
            )
        else:
            rationale.append("TCE single-point: fits on 1 node")

    total_ranks = ranks_per_node * nodes

    # --- Step 3: Determine walltime ---
    if is_freq:
        n_displacements = n_atoms * 6
        base_seconds = 300.0 * (n_atoms / 20.0) ** 1.5
        eff = min(total_ranks, 64) + max(0, total_ranks - 64) * 0.3
        secs_per_disp = base_seconds * (48.0 / max(1, eff))
        est_hours = (n_displacements * secs_per_disp) / 3600.0
        # Add 20% safety margin
        walltime_hours = min(max_wt_hours, est_hours * 1.2)
        walltime_hours = max(2.0, walltime_hours)  # minimum 2h
        rationale.append(
            f"Freq walltime: {est_hours:.1f}h estimated + 20% margin → "
            f"{walltime_hours:.1f}h"
        )
    elif is_opt:
        # Optimization: moderate walltime, depends on molecule size
        if n_atoms <= 10:
            walltime_hours = min(max_wt_hours, 4.0)
        elif n_atoms <= 30:
            walltime_hours = min(max_wt_hours, 12.0)
        elif n_atoms <= 60:
            walltime_hours = min(max_wt_hours, 24.0)
        else:
            walltime_hours = min(max_wt_hours, 48.0)
        rationale.append(
            f"Optimization: {n_atoms} atoms → {walltime_hours:.0f}h walltime"
        )
    elif is_tce:
        # TCE single-points can be long
        if n_atoms <= 5:
            walltime_hours = min(max_wt_hours, 6.0)
        elif n_atoms <= 15:
            walltime_hours = min(max_wt_hours, 24.0)
        else:
            walltime_hours = min(max_wt_hours, 48.0)
        rationale.append(
            f"TCE single-point: {n_atoms} atoms → {walltime_hours:.0f}h walltime"
        )
    else:
        # Single-point energy: usually fast
        if n_atoms <= 5:
            walltime_hours = min(max_wt_hours, 1.0)
        elif n_atoms <= 20:
            walltime_hours = min(max_wt_hours, 4.0)
        elif n_atoms <= 50:
            walltime_hours = min(max_wt_hours, 8.0)
        else:
            walltime_hours = min(max_wt_hours, 24.0)
        rationale.append(
            f"Single-point energy: {n_atoms} atoms → {walltime_hours:.0f}h walltime"
        )

    walltime_str = _format_walltime(walltime_hours)

    # --- Step 4: Determine NWChem memory directive ---
    if node_memory_mb:
        usable_mb = int(node_memory_mb * 0.85)
        mem_per_rank = max(400, (usable_mb // ranks_per_node // 100) * 100)
    else:
        mem_rec = suggest_memory(
            n_atoms=n_atoms, basis=basis_name or "6-31g*",
            method=module, n_heavy_atoms=n_heavy,
        )
        mem_per_rank = mem_rec["recommended_total_mb"]

    mem_suggestion = suggest_memory(
        n_atoms=n_atoms, basis=basis_name or "6-31g*",
        method=module, n_heavy_atoms=n_heavy,
    )
    # Use the larger of: what the method needs, or what fits the node
    recommended_mem = max(mem_suggestion["recommended_total_mb"], 500)
    if node_memory_mb:
        usable_mb = int(node_memory_mb * 0.85)
        ceiling = max(400, (usable_mb // ranks_per_node // 100) * 100)
        if recommended_mem > ceiling:
            warnings.append(
                f"Recommended memory {recommended_mem} MB/rank exceeds safe "
                f"ceiling {ceiling} MB/rank for {ranks_per_node} ranks on "
                f"{node_memory_mb} MB node. Capping to {ceiling} MB."
            )
            recommended_mem = ceiling
    nwchem_mem = recommended_mem

    # --- Step 5: Detect account ---
    account = res.get("account")
    account_info: dict[str, Any] | None = None
    if not account and res.get("account_command"):
        acct_result = detect_hpc_accounts(profile, profiles_path)
        account_info = acct_result
        if acct_result.get("recommended"):
            account = acct_result["recommended"]
            # Find the recommended account's SU balance
            rec_sus = next(
                (a.get("avail_sus", "?") for a in acct_result["accounts"]
                 if a["name"] == account), "?"
            )
            rationale.append(
                f"Account auto-detected: {account} ({rec_sus} SUs available)"
            )

    # Build resource_overrides dict
    resource_overrides: dict[str, Any] = {
        "nodes": nodes,
        "mpi_ranks": total_ranks,
        "walltime": walltime_str,
    }
    if account:
        resource_overrides["account"] = account

    result: dict[str, Any] = {
        "profile": profile,
        "resource_overrides": resource_overrides,
        "recommended_memory_per_rank_mb": nwchem_mem,
        "nwchem_memory_directive": f"memory total {nwchem_mem} mb",
        "nodes": nodes,
        "ranks_per_node": ranks_per_node,
        "total_mpi_ranks": total_ranks,
        "walltime": walltime_str,
        "estimated_basis_functions": n_bf,
        "n_atoms": n_atoms,
        "n_heavy_atoms": n_heavy,
        "method": module,
        "task_type": operation,
        "partition": partition,
        "hardware": {
            "cores_per_node": cores_per_node,
            "node_memory_mb": node_memory_mb,
            "max_nodes": max_nodes,
            "max_walltime": max_wt_str,
            "cpu_arch": cpu_arch,
        },
        "rationale": rationale,
        "warnings": warnings,
    }
    if account:
        result["account"] = account
    if account_info:
        result["account_info"] = account_info
    return result


# ── Smart partition / queue selection ────────────────────────────────────


def suggest_partition(
    input_file: str,
    profiles_path: str | None = None,
    check_queue: bool = True,
) -> dict[str, Any]:
    """Suggest the best partition/queue for a job across all available profiles.

    Scans all scheduler-type profiles, evaluates job fit (memory, walltime),
    checks if dev queues are suitable for short jobs, and optionally queries
    ``sinfo`` for current queue availability.

    Args:
        input_file: Path to the NWChem .nw input file.
        profiles_path: Optional path to runner profiles YAML/JSON.
        check_queue: If True, run ``sinfo`` to check partition availability.

    Returns:
        Dict with ``recommended_profile``, ``recommended_partition``,
        ``resource_overrides``, comparison table, and rationale.
    """
    import subprocess
    from chemtools.core.runner import load_runner_profiles, _resolve_profile
    loaded = load_runner_profiles(profiles_path)
    all_profile_names = list((loaded.get("profiles") or {}).keys())

    # --- Analyze the input file once ---
    job = _job_size(input_file)
    n_atoms = job["n_atoms"]
    n_heavy = job["n_heavy"]
    n_bf = job["n_bf"]
    module = job["module"]
    operation = job["operation"]
    is_freq = job["is_freq"]
    is_opt = job["is_opt"]
    is_tce = job["is_tce"]
    basis_name = job["basis_name"]

    # --- Estimate walltime needed ---
    if is_freq:
        n_disp = n_atoms * 6
        base_seconds = 300.0 * (n_atoms / 20.0) ** 1.5
        est_hours = (n_disp * base_seconds) / 3600.0 / max(1, n_bf // 80)
    elif is_opt:
        if n_atoms <= 10:
            est_hours = 0.5
        elif n_atoms <= 30:
            est_hours = 4.0
        else:
            est_hours = 12.0
    elif is_tce:
        if n_atoms <= 5:
            est_hours = 2.0
        elif n_atoms <= 15:
            est_hours = 12.0
        else:
            est_hours = 24.0
    else:
        if n_atoms <= 5:
            est_hours = 0.25
        elif n_atoms <= 20:
            est_hours = 1.0
        else:
            est_hours = 4.0

    # Estimate memory per rank
    mem_rec = suggest_memory(
        n_atoms=n_atoms, basis=basis_name or "6-31g*",
        method=module, n_heavy_atoms=n_heavy,
    )
    mem_per_rank_needed = mem_rec["recommended_total_mb"]

    # --- Get queue status if requested ---
    queue_info: dict[str, dict[str, Any]] = {}
    if check_queue:
        try:
            proc = subprocess.run(
                ["sinfo", "-o", "%P %a %F %l", "--noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0:
                for line in proc.stdout.strip().splitlines():
                    parts = line.split()
                    if len(parts) >= 4:
                        pname = parts[0].rstrip("*")
                        avail = parts[1]
                        # Node counts: allocated/idle/other/total
                        node_counts = parts[2]
                        timelimit = parts[3]
                        idle = 0
                        total = 0
                        try:
                            nc = node_counts.split("/")
                            idle = int(nc[1]) if len(nc) > 1 else 0
                            total = int(nc[3]) if len(nc) > 3 else 0
                        except (ValueError, IndexError):
                            pass
                        queue_info[pname] = {
                            "available": avail == "up",
                            "idle_nodes": idle,
                            "total_nodes": total,
                            "timelimit": timelimit,
                        }
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

    # --- Evaluate each profile ---
    candidates: list[dict[str, Any]] = []
    for prof_name in all_profile_names:
        try:
            prof = _resolve_profile(loaded, prof_name)
        except ValueError:
            continue

        launcher = prof.get("launcher", {})
        if launcher.get("kind") != "scheduler":
            continue

        res = prof.get("resources", {})
        cores_per_node = res.get("cores_per_node") or res.get("mpi_ranks") or 48
        node_memory_mb = res.get("node_memory_mb")
        max_wt_str = res.get("max_walltime")
        max_wt_hours = _parse_walltime_hours(max_wt_str) or 48.0
        partition = res.get("partition", "")
        cpu_arch = res.get("cpu_arch") or "generic"
        max_nodes = res.get("max_nodes") or 1

        # Memory per core
        mem_per_core = (node_memory_mb / cores_per_node) if node_memory_mb else 4000

        # Does the job fit in walltime?
        fits_walltime = est_hours <= max_wt_hours * 0.90

        # Does the job fit in memory?
        fits_memory = mem_per_rank_needed <= mem_per_core * 0.85 if node_memory_mb else True

        # Is this a dev queue?
        is_dev = "dev" in prof_name.lower() or "dev" in partition.lower()

        # SU rate (default 1.0 if not specified)
        # Infer from common TACC naming
        su_rate = 1.0
        if "spr" in prof_name.lower():
            su_rate = 2.0
        elif "icx" in prof_name.lower():
            su_rate = 1.5

        # Queue status
        q_status = queue_info.get(partition, {})
        queue_available = q_status.get("available", True)  # assume available if no sinfo
        idle_nodes = q_status.get("idle_nodes", 0)

        # --- Scoring ---
        score = 0.0

        if not fits_walltime:
            score -= 100  # Disqualify
        if not fits_memory:
            score -= 50

        if not queue_available:
            score -= 200

        # Dev queue bonus for short jobs
        if is_dev and est_hours <= max_wt_hours * 0.75:
            score += 30  # Strong preference for dev queue when job fits

        # Cheaper SU rate is better
        score -= su_rate * 5

        # Memory headroom is good
        if node_memory_mb:
            headroom = mem_per_core / max(mem_per_rank_needed, 100)
            score += min(10, headroom * 2)

        # Idle nodes bonus
        if idle_nodes > 0:
            score += min(10, idle_nodes)

        candidates.append({
            "profile": prof_name,
            "partition": partition,
            "cores_per_node": cores_per_node,
            "node_memory_mb": node_memory_mb,
            "mem_per_core_mb": round(mem_per_core),
            "max_walltime": max_wt_str or "unknown",
            "max_walltime_hours": max_wt_hours,
            "cpu_arch": cpu_arch,
            "su_rate": su_rate,
            "is_dev": is_dev,
            "fits_walltime": fits_walltime,
            "fits_memory": fits_memory,
            "queue_available": queue_available,
            "idle_nodes": idle_nodes,
            "score": round(score, 1),
        })

    # Sort by score descending
    candidates.sort(key=lambda c: -c["score"])

    rationale: list[str] = []
    rationale.append(
        f"Job: {n_atoms} atoms, ~{n_bf} BF, {module}/{operation}, "
        f"estimated ~{est_hours:.1f}h, ~{mem_per_rank_needed} MB/rank"
    )

    recommended = candidates[0] if candidates else None
    if recommended:
        rec_name = recommended["profile"]
        rec_part = recommended["partition"]

        if recommended["is_dev"]:
            rationale.append(
                f"Recommended dev queue '{rec_part}' — job estimated at "
                f"{est_hours:.1f}h fits within {recommended['max_walltime']} "
                f"max walltime, faster queue turnaround"
            )
        else:
            rationale.append(
                f"Recommended '{rec_part}' — best fit for memory "
                f"({recommended['mem_per_core_mb']} MB/core) and cost "
                f"({recommended['su_rate']}x SU rate)"
            )

        if queue_info:
            idle = recommended.get("idle_nodes", 0)
            if idle > 0:
                rationale.append(f"Queue status: {idle} idle nodes on {rec_part}")
            else:
                rationale.append(f"Queue status: no idle nodes on {rec_part} (job will queue)")

        # Run full resource suggestion for the recommended profile
        full_suggestion = suggest_hpc_resources(input_file, rec_name, profiles_path)

        return {
            "recommended_profile": rec_name,
            "recommended_partition": rec_part,
            "resource_overrides": full_suggestion.get("resource_overrides", {}),
            "nwchem_memory_directive": full_suggestion.get("nwchem_memory_directive", ""),
            "estimated_walltime_hours": round(est_hours, 2),
            "job_summary": {
                "n_atoms": n_atoms,
                "n_heavy_atoms": n_heavy,
                "estimated_basis_functions": n_bf,
                "method": module,
                "task_type": operation,
                "mem_per_rank_needed_mb": mem_per_rank_needed,
            },
            "partition_comparison": candidates,
            "queue_status_available": bool(queue_info),
            "rationale": rationale + full_suggestion.get("rationale", []),
            "warnings": full_suggestion.get("warnings", []),
        }

    return {
        "recommended_profile": None,
        "recommended_partition": None,
        "error": "No suitable scheduler profiles found",
        "profiles_scanned": all_profile_names,
        "rationale": rationale,
    }
__all__ = [
    "detect_hpc_accounts",
    "suggest_hpc_resources",
    "suggest_partition",
]

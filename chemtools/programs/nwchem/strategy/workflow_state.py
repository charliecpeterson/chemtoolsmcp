"""NWChem workflow state inspection and pre-launch checks.

Three top-level entry points plus several workflow helpers:

  * prepare_freq_restart         Inspect an interrupted frequency run
                                  and prepare a restart input that picks
                                  up where it left off.
  * preflight_check              Pre-launch validation — verify the
                                  geometry / basis / spin combination is
                                  sane before submitting a job.
  * get_nwchem_workflow_state    Inspect a directory of NWChem files and
                                  return current workflow state (which
                                  steps done, which step is next).

Plus workflow helpers (_extract_input_echo, _find_fdrst,
_check_related_slurm_jobs, _is_job_running_for_output, _wf_result,
_parse_walltime_hours, _format_walltime).
"""

from __future__ import annotations
import re
import subprocess
from pathlib import Path
from typing import Any

from chemtools.core.common import read_text
from chemtools.programs.nwchem.parse.freq import parse_freq


def _extract_nwchem_geometry(*args, **kwargs):
    """Lazy proxy for chemtools.programs.nwchem.input.geometry.extract_nwchem_geometry."""
    from chemtools.programs.nwchem.input.geometry import extract_nwchem_geometry
    return extract_nwchem_geometry(*args, **kwargs)


def _create_nwchem_input(*args, **kwargs):
    """Lazy proxy for chemtools.programs.nwchem.input.general.create_nwchem_input."""
    from chemtools.programs.nwchem.input.general import create_nwchem_input
    return create_nwchem_input(*args, **kwargs)


def prepare_freq_restart(
    input_file: str,
    output_file: str,
    profile: str | None = None,
) -> dict[str, Any]:
    """Validate that a freq restart is ready and return a submit-ready report.

    Checks: restart keyword present, .fdrst exists, reports progress.
    Does NOT submit — caller decides whether to call launch_nwchem_run.
    """
    import re
    from pathlib import Path
    from chemtools.programs.nwchem.parse.freq import parse_freq_progress

    nw_text = Path(input_file).read_text(encoding="utf-8")
    issues: list[str] = []

    # Check restart keyword
    has_restart = bool(re.search(r"^\s*restart\b", nw_text, re.MULTILINE | re.IGNORECASE))
    if not has_restart:
        issues.append("Input is missing 'restart' keyword — NWChem will start from scratch")

    # Determine restart prefix name
    restart_match = re.search(r"^\s*restart\s+(\S+)", nw_text, re.MULTILINE | re.IGNORECASE)
    if restart_match:
        restart_prefix = restart_match.group(1)
    else:
        start_match = re.search(r"^\s*start\s+(\S+)", nw_text, re.MULTILINE | re.IGNORECASE)
        restart_prefix = start_match.group(1) if start_match else Path(input_file).stem

    # Check .fdrst and .db exist
    job_dir = Path(input_file).parent
    fdrst_path = job_dir / f"{restart_prefix}.fdrst"
    db_path = job_dir / f"{restart_prefix}.db"

    has_fdrst = fdrst_path.exists()
    has_db = db_path.exists()
    if not has_fdrst:
        issues.append(f"Checkpoint file {fdrst_path.name} not found — freq will start from atom 1")
    if not has_db:
        issues.append(f"Database file {restart_prefix}.db not found — restart may fail")

    fdrst_info: dict[str, Any] = {"path": str(fdrst_path), "exists": has_fdrst}
    if has_fdrst:
        from datetime import datetime, timezone as _tz
        stat = fdrst_path.stat()
        fdrst_info.update({
            "size_kb": round(stat.st_size / 1024, 1),
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=_tz.utc).isoformat(),
        })

    # Parse progress from previous output
    progress: dict[str, Any] = {}
    out_path = Path(output_file)
    if out_path.exists():
        try:
            out_text = read_text(str(out_path))
            progress = parse_freq_progress(str(out_path), out_text)
        except Exception:
            pass

    return {
        "ready_to_restart": len(issues) == 0,
        "issues": issues,
        "input_file": str(Path(input_file).resolve()),
        "restart_prefix": restart_prefix,
        "has_restart_keyword": has_restart,
        "fdrst": fdrst_info,
        "db_exists": has_db,
        "progress": progress,
        "suggested_profile": profile,
    }


# ---------------------------------------------------------------------------
# Preflight check
# ---------------------------------------------------------------------------

def preflight_check(
    input_file: str,
    profile: str,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    """Run all pre-submission checks and return a pass/fail report.

    Combines: lint, movecs file existence, memory vs node RAM ceiling.
    """
    import re
    from pathlib import Path
    from chemtools.api_input import lint_nwchem_input
    from chemtools.core.runner import load_runner_profiles, _resolve_profile

    checks: list[dict[str, Any]] = []

    # 1. Lint
    lint = lint_nwchem_input(input_file)
    lint_errors = [i for i in lint.get("issues", []) if i.get("level") == "error"]
    checks.append({
        "check": "lint",
        "passed": len(lint_errors) == 0,
        "issues": lint.get("issues", []),
    })

    # 2. movecs input files exist
    nw_text = Path(input_file).read_text(encoding="utf-8")
    job_dir = Path(input_file).parent
    for m in re.finditer(r"vectors\s+input\s+(\S+)", nw_text, re.IGNORECASE):
        movecs_name = m.group(1)
        movecs_path = job_dir / movecs_name
        exists = movecs_path.exists()
        checks.append({
            "check": f"movecs_exists:{movecs_name}",
            "passed": exists,
            "issues": [] if exists else [
                {"level": "error", "message": f"vectors input file not found: {movecs_path}"}
            ],
        })

    # 3. Memory vs. node RAM
    try:
        profiles = load_runner_profiles(profiles_path)
        profile_payload = _resolve_profile(profiles, profile)
        resources = profile_payload.get("resources", {})
        partition = resources.get("partition")
        launcher = profile_payload.get("launcher", {})
        scheduler_type = (
            profile_payload.get("scheduler", {}).get("system")
            or launcher.get("scheduler_type", "slurm")
        ).lower()

        # Query real partition specs if on a scheduler
        node_mem_mb = None
        cpus_per_node = None
        if launcher.get("kind") == "scheduler" and partition:
            from chemtools.core.runner import query_partition_specs
            hw = query_partition_specs(partition, scheduler_type)
            node_mem_mb = hw.get("node_memory_mb")
            cpus_per_node = hw.get("cpus_per_node")

        # Parse memory directive from input
        mem_match = re.search(r"memory\s+(?:total\s+)?(\d+)\s*(mb|mw|gb)", nw_text, re.IGNORECASE)
        if mem_match and node_mem_mb:
            mem_val = int(mem_match.group(1))
            mem_unit = mem_match.group(2).lower()
            if mem_unit == "gb":
                mem_val *= 1024
            elif mem_unit == "mw":
                mem_val *= 8  # 1 MW = 8 MB

            mpi_ranks = resources.get("mpi_ranks", cpus_per_node or 1)
            total_requested_mb = mem_val * mpi_ranks
            ceiling_mb = int(node_mem_mb * 0.90)
            ok = total_requested_mb <= ceiling_mb
            checks.append({
                "check": "memory_ceiling",
                "passed": ok,
                "details": {
                    "memory_per_rank_mb": mem_val,
                    "mpi_ranks": mpi_ranks,
                    "total_requested_mb": total_requested_mb,
                    "node_memory_mb": node_mem_mb,
                    "ceiling_90pct_mb": ceiling_mb,
                },
                "issues": [] if ok else [
                    {"level": "error",
                     "message": (
                         f"Memory request {total_requested_mb} MB ({mem_val} MB × {mpi_ranks} ranks) "
                         f"exceeds 90% of node RAM ({node_mem_mb} MB). "
                         f"Reduce memory directive or mpi_ranks."
                     )}
                ],
            })
    except Exception as exc:
        checks.append({
            "check": "memory_ceiling",
            "passed": True,
            "issues": [{"level": "info", "message": f"Could not check memory ceiling: {exc}"}],
        })

    all_passed = all(c["passed"] for c in checks)
    return {
        "ready_to_submit": all_passed,
        "checks": checks,
        "summary": (
            f"{'PASS' if all_passed else 'FAIL'}: "
            f"{sum(c['passed'] for c in checks)}/{len(checks)} checks passed"
        ),
    }


# ---------------------------------------------------------------------------
# Workflow state machine
# ---------------------------------------------------------------------------

WORKFLOW_STATES = [
    "pending",                # not yet submitted
    "queued",                 # submitted, waiting in scheduler queue
    "running_scf",            # SCF/DFT in progress
    "running_opt",            # geometry optimization in progress
    "running_freq",           # freq displacements in progress
    "freq_timelimited",       # freq hit walltime, fdrst exists, can restart
    "freq_complete",          # all displacements done
    "opt_converged",          # geometry optimized, ready for next step
    "opt_failed_convergence", # optimization not converging
    "scf_failed",             # SCF not converging
    "imaginary_modes",        # freq done but imaginary modes present
    "oom",                    # out of memory
    "completed",              # all tasks done, results valid
    "cancelled",              # job was cancelled
    "failed",                 # generic failure
    "needs_user_input",       # tool cannot decide; escalate to human
]


def get_nwchem_workflow_state(
    input_file: str | None = None,
    output_file: str = "",
    profile: str = "",
    error_file: str | None = None,
) -> dict[str, Any]:
    """Determine workflow state and return the exact next tool call to advance.

    Encodes domain logic so the LLM does not need to reason about NWChem
    internals — it just calls this tool, executes ``next_action``, and repeats.

    ``input_file`` is optional — when only the ``.out`` file is available the
    function parses the NWChem input echo from the output to determine the
    task type.  Missing companion files (``.nw``, ``.fdrst``, ``.err``, etc.)
    are reported in ``missing_files`` so the model can ask the user.

    If SLURM is available, ``squeue`` is checked for running jobs whose name
    matches the output stem.  Matches are reported in ``related_jobs`` so the
    model can ask the user whether the running job belongs to this output —
    it should **never assume** a match without confirmation.
    """
    import shutil
    from chemtools.programs.nwchem.parse.freq import parse_freq_progress as _parse_freq_progress, analyze_imaginary_modes as _analyze_imag

    out = Path(output_file)
    inp = Path(input_file) if input_file else None

    # Try to find the input file from the output path if not provided
    if inp is None or not inp.exists():
        candidate_nw = out.with_suffix(".nw")
        if candidate_nw.exists():
            inp = candidate_nw
            input_file = str(inp)

    # Derive error file if not given
    if error_file is None:
        candidate = out.with_suffix(".err")
        if candidate.exists():
            error_file = str(candidate)

    # Track missing companion files
    missing_files: list[str] = []
    if inp is None or not inp.exists():
        missing_files.append(f"{out.stem}.nw (input file)")
    if error_file is None or not Path(error_file).exists():
        missing_files.append(f"{out.stem}.err (error file)")

    # --- Check for related running SLURM jobs ---
    related_jobs = _check_related_slurm_jobs(out.stem)

    # --- 0. Output file must exist ---
    if not out.exists() or out.stat().st_size == 0:
        if inp and inp.exists():
            # Check if job is queued via .jobid
            jobid_file = out.with_suffix(".jobid")
            if not jobid_file.exists():
                jobid_file = inp.with_suffix(".jobid")
            if jobid_file.exists():
                r = _wf_result(
                    "queued", 0,
                    "Job submitted (jobid file exists) but no output yet.",
                    {"tool": "get_nwchem_run_status",
                     "params": {"output_file": output_file,
                                "input_file": input_file or "", "profile": profile}},
                    0.85,
                )
                r["related_jobs"] = related_jobs
                r["missing_files"] = missing_files
                return r
            r = _wf_result(
                "pending", 0,
                "Output file not found or empty — job has not started.",
                {"tool": "launch_nwchem_run",
                 "params": {"input_file": input_file or "", "profile": profile}},
                0.90,
            )
            r["related_jobs"] = related_jobs
            r["missing_files"] = missing_files
            return r
        # No input, no output
        r = _wf_result(
            "needs_user_input", 0,
            "Neither input nor output file found. Provide the correct file paths.",
            None, 0.3,
        )
        r["related_jobs"] = related_jobs
        r["missing_files"] = missing_files
        return r

    contents = out.read_text(encoding="utf-8", errors="replace")

    # Read input text — from file if available, otherwise parse from output echo
    input_text = ""
    if inp and inp.exists():
        input_text = inp.read_text(encoding="utf-8", errors="replace")
    else:
        input_text = _extract_input_echo(contents)

    err_text = ""
    if error_file and Path(error_file).exists():
        err_text = Path(error_file).read_text(encoding="utf-8", errors="replace")

    # --- Determine task type from input text ---
    is_freq = bool(re.search(r"task\s+\w+\s+freq", input_text, re.IGNORECASE))
    is_opt = bool(re.search(r"task\s+\w+\s+optim", input_text, re.IGNORECASE))

    # --- Find fdrst: check both input dir and output dir ---
    fdrst_path = _find_fdrst(inp, out, input_text)

    # --- Check for other restart assets ---
    for ext in (".db", ".movecs"):
        asset = out.with_suffix(ext)
        if not asset.exists() and inp:
            asset = inp.with_suffix(ext)
        if not asset.exists():
            missing_files.append(f"{out.stem}{ext}")

    if fdrst_path is None and is_freq:
        missing_files.append(f"{out.stem}.fdrst (freq checkpoint)")

    # Helper to attach context to every result
    def _enrich(result: dict[str, Any]) -> dict[str, Any]:
        result["related_jobs"] = related_jobs
        result["missing_files"] = missing_files
        if not input_file and inp and inp.exists():
            result["resolved_input_file"] = str(inp)
        if related_jobs:
            result["related_jobs_note"] = (
                "SLURM shows job(s) with a similar name. "
                "Confirm with the user whether any of these belong to this output "
                "before assuming the job is still running."
            )
        return result

    # --- 1. OOM? ---
    if re.search(r"MA_ERR|insufficient\s+memory|failed to allocate", contents, re.IGNORECASE) or \
       re.search(r"MA_ERR|MemoryError|Killed", err_text, re.IGNORECASE):
        return _enrich(_wf_result(
            "oom", 0,
            "Out of memory — reduce memory directive or mpi_ranks.",
            {"tool": "create_nwchem_input_variant",
             "params": {"source_input": input_file or "",
                        "changes": {"memory": "800 mb"},
                        "reason": "OOM failure — reducing memory"}} if input_file else None,
            0.85,
        ))

    # --- 2. Timelimit? ---
    if "DUE TO TIME LIMIT" in err_text or ("CANCELLED" in err_text and "TIME" in err_text):
        if is_freq and fdrst_path:
            try:
                progress = _parse_freq_progress(output_file, contents)
                pct = progress.get("pct_complete", 0) or 0
            except Exception:
                pct = 0
            return _enrich(_wf_result(
                "freq_timelimited", pct,
                f"Freq hit walltime at {pct:.0f}% complete. fdrst checkpoint valid — resubmit to continue.",
                {"tool": "launch_nwchem_run",
                 "params": {"input_file": input_file or "", "profile": profile,
                            "resource_overrides": {"walltime": "48:00:00"}}} if input_file else None,
                0.95,
            ))
        return _enrich(_wf_result(
            "cancelled", 0,
            "Job cancelled due to time limit.",
            {"tool": "launch_nwchem_run",
             "params": {"input_file": input_file or "", "profile": profile,
                        "resource_overrides": {"walltime": "48:00:00"}}} if input_file else None,
            0.70,
        ))

    # --- 3. SCF failed? ---
    if re.search(r"(convergence|scf)\s+(has\s+)?not\s+been?\s+(achieved|reached|converged)", contents, re.IGNORECASE):
        return _enrich(_wf_result(
            "scf_failed", 0,
            "SCF did not converge.",
            {"tool": "suggest_nwchem_recovery",
             "params": {"output_file": output_file, "input_file": input_file or "", "mode": "scf"}},
            0.90,
        ))

    # --- 4. Check if still running (no "Total times" line = incomplete) ---
    has_total_times = bool(re.search(r"Total\s+times\s+cpu:", contents, re.IGNORECASE))
    if not has_total_times:
        # Distinguish "incomplete and still running" from "incomplete and stopped"
        job_is_running = _is_job_running_for_output(out, related_jobs)

        if is_freq:
            try:
                progress = _parse_freq_progress(output_file, contents)
                pct = progress.get("pct_complete", 0) or 0
                n_done = progress.get("n_done_cumulative", 0)
                n_total = progress.get("n_total_displacements", 0)
            except Exception:
                pct, n_done, n_total = 0, 0, 0

            if job_is_running:
                return _enrich(_wf_result(
                    "running_freq", pct,
                    f"Frequency calculation in progress — {n_done}/{n_total} displacements ({pct:.0f}% done).",
                    {"tool": "watch_nwchem_run",
                     "params": {"output_file": output_file, "input_file": input_file or "",
                                "profile": profile}},
                    0.90,
                ))
            else:
                # Stopped mid-freq — needs restart
                if fdrst_path:
                    return _enrich(_wf_result(
                        "freq_timelimited", pct,
                        f"Freq stopped at {pct:.0f}% ({n_done}/{n_total} displacements). "
                        f"fdrst checkpoint exists — can restart.",
                        {"tool": "prepare_nwchem_freq_restart",
                         "params": {"input_file": input_file or "", "output_file": output_file,
                                    "profile": profile}} if input_file else None,
                        0.85,
                    ))
                else:
                    return _enrich(_wf_result(
                        "freq_timelimited", pct,
                        f"Freq stopped at {pct:.0f}% ({n_done}/{n_total} displacements). "
                        f"No .fdrst found — restart may repeat completed work. "
                        f"Check if .fdrst exists in the original job directory.",
                        {"tool": "analyze_nwchem_case",
                         "params": {"output_file": output_file, "input_file": input_file or ""}},
                        0.60,
                    ))
        elif is_opt:
            state = "running_opt" if job_is_running else "failed"
            summary = ("Geometry optimization in progress." if job_is_running
                       else "Optimization stopped before converging (walltime or error).")
            return _enrich(_wf_result(
                state, 0, summary,
                {"tool": "watch_nwchem_run" if job_is_running else "suggest_nwchem_recovery",
                 "params": {"output_file": output_file, "input_file": input_file or "",
                            "profile": profile} if job_is_running else
                           {"output_file": output_file, "input_file": input_file or "", "mode": "auto"}},
                0.85 if job_is_running else 0.70,
            ))
        else:
            state = "running_scf" if job_is_running else "failed"
            summary = ("SCF/DFT calculation in progress." if job_is_running
                       else "Calculation stopped before completing (walltime or error).")
            return _enrich(_wf_result(
                state, 0, summary,
                {"tool": "watch_nwchem_run" if job_is_running else "analyze_nwchem_case",
                 "params": {"output_file": output_file, "input_file": input_file or "",
                            "profile": profile} if job_is_running else
                           {"output_file": output_file, "input_file": input_file or ""}},
                0.80 if job_is_running else 0.65,
            ))

    # --- 5. Completed — determine what finished ---

    # 5a. Freq job: check for imaginary modes
    if re.search(r"P\.Frequency|Normal\s+Mode\s+Eigenvalue", contents, re.IGNORECASE):
        try:
            imag = _analyze_imag(output_file, contents)
            sig_count = imag.get("significant_imaginary_mode_count", 0)
        except Exception:
            sig_count = 0

        if sig_count > 0:
            return _enrich(_wf_result(
                "imaginary_modes", 100,
                f"Freq complete but {sig_count} significant imaginary mode(s) found.",
                {"tool": "draft_nwchem_imaginary_mode_inputs",
                 "params": {"output_file": output_file, "input_file": input_file or ""}},
                0.85,
            ))
        return _enrich(_wf_result(
            "freq_complete", 100,
            "Frequency calculation completed — no significant imaginary modes.",
            {"tool": "parse_nwchem_output",
             "params": {"output_file": output_file, "sections": ["freq", "tasks"]}},
            0.95,
        ))

    # 5b. Optimization: check convergence
    if re.search(r"Optimization\s+converged", contents, re.IGNORECASE):
        return _enrich(_wf_result(
            "opt_converged", 100,
            "Geometry optimization converged.",
            {"tool": "extract_nwchem_geometry",
             "params": {"output_file": output_file, "frame": "best"}},
            0.90,
        ))
    if is_opt:
        return _enrich(_wf_result(
            "opt_failed_convergence", 0,
            "Optimization did not converge.",
            {"tool": "suggest_nwchem_recovery",
             "params": {"output_file": output_file, "input_file": input_file or "", "mode": "auto"}},
            0.80,
        ))

    # 5c. General success
    return _enrich(_wf_result(
        "completed", 100,
        "Calculation completed.",
        {"tool": "analyze_nwchem_case",
         "params": {"output_file": output_file, "input_file": input_file or ""}},
        0.85,
    ))


def _extract_input_echo(contents: str) -> str:
    """Extract the NWChem input echo from the output file.

    NWChem echoes the full input between markers like::

        ============================== echo of input deck ==============================
        ...
        ================================================================================
    """
    m = re.search(
        r"={10,}\s*echo of input deck\s*={10,}\n(.*?)={10,}",
        contents, re.DOTALL | re.IGNORECASE,
    )
    return m.group(1) if m else ""


def _find_fdrst(
    inp: "Path | None",
    out: Path,
    input_text: str,
) -> "Path | None":
    """Search for .fdrst file in both input and output directories."""
    # Try stem from start/restart keyword
    stem_match = re.search(r"^\s*(?:start|restart)\s+(\S+)", input_text, re.MULTILINE | re.IGNORECASE)
    stems_to_try = [out.stem]
    if stem_match:
        stems_to_try.insert(0, stem_match.group(1))
    if inp and inp.stem not in stems_to_try:
        stems_to_try.append(inp.stem)

    dirs_to_check = [out.parent]
    if inp and inp.parent != out.parent:
        dirs_to_check.append(inp.parent)

    for d in dirs_to_check:
        for stem in stems_to_try:
            candidate = d / (stem + ".fdrst")
            if candidate.exists():
                return candidate
    return None


def _check_related_slurm_jobs(output_stem: str) -> list[dict[str, str]]:
    """Check squeue for running/pending jobs whose name matches the output stem.

    Returns a list of dicts with job_id, name, partition, state, time.
    Never assumes a match — the model must ask the user to confirm.
    """
    import shutil

    if not shutil.which("squeue"):
        return []

    try:
        proc = subprocess.run(
            ["squeue", "-u", str(subprocess.check_output(["whoami"]).decode().strip()),
             "-h", "-o", "%i %j %P %T %M"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return []
    except Exception:
        return []

    # Normalize the output stem for fuzzy matching
    # e.g. "am2pba3h2_s_reopt_v2_freq" should match job name "am2pba3h"
    stem_lower = output_stem.lower()
    # Extract a short prefix for matching (first 8 chars or first token)
    stem_prefix = re.split(r"[_\-.]", stem_lower)[0] if stem_lower else ""

    related = []
    for line in proc.stdout.strip().splitlines():
        parts = line.split(None, 4)
        if len(parts) < 4:
            continue
        job_id, name, partition, state = parts[0], parts[1], parts[2], parts[3]
        time_str = parts[4] if len(parts) > 4 else ""

        name_lower = name.lower()
        # Match if: stem starts with job name, job name starts with stem,
        # or they share a significant prefix (>=6 chars)
        is_related = (
            stem_lower.startswith(name_lower)
            or name_lower.startswith(stem_lower)
            or (stem_prefix and len(stem_prefix) >= 6 and name_lower.startswith(stem_prefix))
            or (len(name_lower) >= 6 and stem_lower.startswith(name_lower[:8]))
        )
        if is_related:
            related.append({
                "job_id": job_id,
                "name": name,
                "partition": partition,
                "state": state,
                "time": time_str,
            })

    return related


def _is_job_running_for_output(out: Path, related_jobs: list[dict[str, str]]) -> bool:
    """Determine if a SLURM job is currently running that writes to this output.

    Checks: (1) .jobid file exists and the job is in squeue, or
    (2) the output file was modified very recently (< 5 min ago).
    """
    import time

    # Check .jobid
    jobid_file = out.with_suffix(".jobid")
    if jobid_file.exists():
        try:
            jid = jobid_file.read_text().strip()
            if any(j["job_id"] == jid and j["state"] == "RUNNING" for j in related_jobs):
                return True
        except Exception:
            pass

    # Check if output was modified very recently
    try:
        mtime = out.stat().st_mtime
        if (time.time() - mtime) < 300:  # 5 minutes
            return True
    except Exception:
        pass

    return False


def _wf_result(
    state: str,
    progress_pct: float,
    summary: str,
    next_action: dict[str, Any] | None,
    confidence: float,
) -> dict[str, Any]:
    """Build a workflow state result dict."""
    result: dict[str, Any] = {
        "state": state,
        "progress_pct": round(progress_pct, 1),
        "human_summary": summary,
        "confidence": confidence,
    }
    if next_action is not None:
        result["next_action"] = next_action
    return result


# ---------------------------------------------------------------------------
# HPC resource advisor (profile-aware, multi-node, task-type-aware)
# ---------------------------------------------------------------------------


def _parse_walltime_hours(wt: str | None) -> float | None:
    """Parse HH:MM:SS walltime string to hours."""
    if not wt:
        return None
    parts = wt.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) + int(parts[1]) / 60 + int(parts[2]) / 3600
        if len(parts) == 2:
            return int(parts[0]) + int(parts[1]) / 60
        return float(parts[0])
    except (ValueError, IndexError):
        return None


def _format_walltime(hours: float) -> str:
    """Format hours as HH:MM:SS walltime string."""
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h}:{m:02d}:00"



__all__ = [
    "prepare_freq_restart",
    "preflight_check",
    "get_nwchem_workflow_state",
]

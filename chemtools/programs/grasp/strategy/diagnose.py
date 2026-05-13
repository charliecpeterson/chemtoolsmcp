"""GRASP convergence / failure diagnosis.

Inspect a working directory and classify what went wrong (or right). The
real pain point for GRASP workflows is the interactive debugging loop:
when rmcdhf fails, you have to read scattered output, decide which knob
to turn, and rerun. These tools collapse that loop into a single call.

Two entry points:

  analyze_grasp_case(working_dir) — survey the working directory: which
      steps ran, which produced their expected files, which errored. Returns
      a structured case report with verdict (healthy / partial / failed /
      not_started) + per-step status.

  suggest_grasp_recovery(working_dir or error_text) — given a failed run,
      classify the failure mode and suggest the next action (rerun with
      hf bootstrap, increase iterations, fix block-level selection, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.programs.grasp.parse.rmcdhf_log import parse_rmcdhf_log
from chemtools.programs.grasp.parse.sum_file import parse_sum


# Each step in the canonical DHF workflow writes specific artifacts.
# We use these to decide which steps "ran" by file presence.
_STEP_ARTIFACTS = {
    "rnucleus":      ["isodata"],
    "rcsfgenerate":  ["rcsf.out"],
    "rangular":      ["mcp.30", "mcp.31"],  # at minimum
    "rwfnestimate":  ["rwfn.inp"],
    "rmcdhf":        ["rwfn.out", "rmcdhf.sum", "rmcdhf.log"],
    "rsave":         [],  # rsave is named after the prefix, can't predict
    "jj2lsj":        [],  # .lsj.lbl is named after the prefix
}


def analyze_grasp_case(working_dir: str) -> dict[str, Any]:
    """Inspect a GRASP working directory and report status of each step.

    Returns
    -------
    dict with::

        {
          "working_dir": "...",
          "session_log_exists": bool,
          "verdict": "healthy" | "partial" | "failed" | "not_started",
          "steps": [
            {step, status: "missing"|"ran"|"errored", artifacts_found, ...}
          ],
          "scf_summary": {... from rmcdhf .sum if present ...} | None,
          "scf_iterations": {... from rmcdhf log if present ...} | None,
          "issues": [{severity, code, message}, ...],
          "next_actions": ["..."],
        }
    """
    work = Path(working_dir)
    if not work.exists():
        return {
            "working_dir": str(work),
            "verdict": "not_started",
            "error": "working_dir_not_found",
        }

    out: dict[str, Any] = {
        "working_dir": str(work.resolve()),
        "session_log_exists": (work / "grasp_session.md").exists(),
    }

    # Per-step file presence audit.
    steps: list[dict[str, Any]] = []
    for step, artifacts in _STEP_ARTIFACTS.items():
        present = [a for a in artifacts if (work / a).exists()]
        missing = [a for a in artifacts if not (work / a).exists()]
        if artifacts:
            status = "ran" if present else "missing"
        else:
            status = "ran"  # placeholder for rsave / jj2lsj
        steps.append({
            "step": step,
            "status": status,
            "artifacts_found": present,
            "artifacts_missing": missing,
        })
    out["steps"] = steps

    # Pull the SCF summary if the rmcdhf .sum was written.
    scf_summary = None
    scf_iters = None
    sum_path = work / "rmcdhf.sum"
    if sum_path.exists():
        try:
            scf_summary = parse_sum(sum_path.read_text(encoding="utf-8", errors="replace"))
        except Exception as e:
            scf_summary = {"error": str(e)}
    out["scf_summary"] = scf_summary

    # Look for an rmcdhf stdout capture (we typically name it rmcdhf.out
    # via capture_log_file). If absent, the .log file has the input echo.
    for candidate in ("rmcdhf.out", "rmcdhf_stdout.log", "rmcdhf.log"):
        p = work / candidate
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        parsed = parse_rmcdhf_log(text)
        if parsed.get("n_iterations", 0) > 0:
            scf_iters = parsed
            scf_iters["_source"] = candidate
            break
    out["scf_iterations"] = scf_iters

    # Verdict + issues.
    issues: list[dict[str, Any]] = []
    next_actions: list[str] = []
    ran_steps = {s["step"] for s in steps if s["status"] == "ran"}

    if not ran_steps:
        verdict = "not_started"
    elif "rmcdhf" not in ran_steps:
        verdict = "partial"
        last_run = _last_step_run(steps)
        next_actions.append(
            f"Last step that ran: `{last_run}`. The next step "
            f"`{_next_step(last_run)}` produced no artifacts — check the "
            f"session log for its stdout."
        )
    else:
        # rmcdhf produced files; check convergence
        if scf_iters and scf_iters.get("explicitly_not_converged"):
            verdict = "failed"
            issues.append({
                "severity": "error",
                "code": "scf_not_converged",
                "message": "rmcdhf reports SCF did not converge.",
            })
            next_actions.append(
                "Run `suggest_grasp_recovery` on this directory for "
                "specific remediation guidance (likely: hf-bootstrap "
                "for high-Z, or increase max_scf_cycles, or non-rel "
                "warm-up)."
            )
        elif scf_iters and not scf_iters.get("converged"):
            verdict = "partial"
            issues.append({
                "severity": "warning",
                "code": "scf_convergence_unknown",
                "message": (
                    f"rmcdhf ran {scf_iters['n_iterations']} iterations but "
                    "the explicit convergence marker wasn't found in the log. "
                    "Likely interrupted or output was truncated."
                ),
            })
        else:
            verdict = "healthy"

    if scf_summary and scf_summary.get("speed_of_light_au"):
        c = scf_summary["speed_of_light_au"]
        if scf_summary.get("is_nonrel_limit"):
            issues.append({
                "severity": "info",
                "code": "nonrel_limit_active",
                "message": (
                    f"This run used speed of light = {c:.1f} au "
                    "(non-relativistic limit). Energy levels won't match "
                    "physical/relativistic values."
                ),
            })

    out["verdict"] = verdict
    out["issues"] = issues
    out["next_actions"] = next_actions
    return out


# --- Recovery suggester ---------------------------------------------------

# Failure-mode classifier patterns.
# Each entry: (pattern_substring, classifier dict)
_FAILURE_PATTERNS: list[tuple[str, dict[str, Any]]] = [
    (
        "TFWAVE: Unable to compute radial",
        {
            "failure_class": "tfwave_divergence",
            "severity": "error",
            "root_cause": (
                "Thomas-Fermi initial estimate diverges for one or more "
                "inner subshells. Common for high-Z atoms (Z≥30) and "
                "especially actinides."
            ),
            "fix_recipe": (
                "Switch to the hf-bootstrap workflow: run the non-rel hf "
                "code first, convert via rwfnmchfmcdf, then DHF reads the "
                "converted orbitals from rwfn.inp."
            ),
            "next_actions": [
                "Call plan_grasp_hf_bootstrap_workflow with the same "
                "element/configuration, then run_grasp_workflow.",
                "Check get_grasp_topic_guide('hf_bootstrap') for details.",
            ],
        },
    ),
    (
        "serial numbers must be in the range",
        {
            "failure_class": "block_level_mismatch",
            "severity": "error",
            "root_cause": (
                "The block_level_selections passed to rmcdhf has more "
                "levels than a CSF block actually contains, or wrong "
                "block count."
            ),
            "fix_recipe": (
                "Read the 'block ncf nev 2j+1 parity' line from rmcdhf's "
                "stdout (or rerun rcsfgenerate and count blocks). Pass "
                "exactly N entries, each with selections within [1, ncf] "
                "of that block."
            ),
            "next_actions": [
                "Check the rmcdhf stdout banner that lists blocks.",
                "Re-call run_grasp_rmcdhf with corrected block_level_selections.",
            ],
        },
    ),
    (
        "End of file",
        {
            "failure_class": "premature_eof",
            "severity": "error",
            "root_cause": (
                "GRASP exe expected another prompt answer but stdin ended. "
                "Usually means the heredoc didn't provide all the required "
                "answers (most often a block_level_selections list is "
                "shorter than the block count)."
            ),
            "fix_recipe": (
                "Look at the LAST 'block N    ncf = ...' or 'Enter ...' "
                "prompt before the End-of-file error. That's the question "
                "your stdin didn't answer."
            ),
            "next_actions": [
                "Re-examine the heredoc builder call — make sure all "
                "per-block selections are provided.",
            ],
        },
    ),
    (
        "Could not open file",
        {
            "failure_class": "missing_input_file",
            "severity": "error",
            "root_cause": (
                "A required input file (isodata, rcsf.inp, rwfn.inp, "
                "mcp.30, etc.) wasn't found by the exe."
            ),
            "fix_recipe": (
                "Verify the previous step in the workflow ran and "
                "produced its output. For rangular this means rcsf.inp "
                "must exist (rcsfgenerate writes rcsf.out — needs cp "
                "to rcsf.inp). For rmcdhf this means rwfn.inp + isodata "
                "+ mcp.30..mcp.39 + rcsf.inp must all exist."
            ),
            "next_actions": [
                "Call analyze_grasp_case(working_dir) to see exactly which "
                "step's artifacts are missing.",
            ],
        },
    ),
    (
        "orbitals diverging",
        {
            "failure_class": "orbital_divergence",
            "severity": "error",
            "root_cause": (
                "SCF orbital amplitudes diverged during iteration — typical "
                "for very high-Z or strongly correlated systems where the "
                "starting guess is too far from the true wavefunction."
            ),
            "fix_recipe": (
                "Try the non-rel limit first (c=2000) to get reasonable "
                "orbitals, save them via rsave, then restart with full "
                "relativistic c=137.036 using the saved *.w as initial "
                "guess (plan_grasp_restart_from_workflow)."
            ),
            "next_actions": [
                "Run plan_grasp_nonrel_limit_workflow → save → restart "
                "via plan_grasp_restart_from_workflow.",
            ],
        },
    ),
    (
        "Convergence not reached",
        {
            "failure_class": "max_iter_exhausted",
            "severity": "warning",
            "root_cause": (
                "rmcdhf hit the SCF cycle cap without converging. The "
                "default cap is 100 but actual convergence may need more "
                "for hard cases (high-Z, near-degenerate states)."
            ),
            "fix_recipe": (
                "Bump max_scf_cycles to 200 or 300 and rerun. If the "
                "energy is still oscillating after that, try the non-rel "
                "limit + restart pattern instead."
            ),
            "next_actions": [
                "Re-call run_grasp_rmcdhf with max_scf_cycles=300.",
            ],
        },
    ),
]


def suggest_grasp_recovery(
    *,
    working_dir: str | None = None,
    error_text: str | None = None,
) -> dict[str, Any]:
    """Classify a GRASP failure and suggest recovery actions.

    Provide either ``working_dir`` (the function reads the session log +
    rmcdhf stdout) or ``error_text`` (a captured stderr/stdout chunk).
    """
    if working_dir is None and error_text is None:
        return {"error": "must_provide_working_dir_or_error_text"}

    text = error_text or ""
    sources: list[str] = []
    if working_dir is not None:
        work = Path(working_dir)
        # Look in the obvious places for failure markers.
        for candidate in ("grasp_session.md", "rmcdhf.out",
                          "rmcdhf_stdout.log", "rmcdhf.log"):
            p = work / candidate
            if p.exists():
                try:
                    text += "\n" + p.read_text(encoding="utf-8", errors="replace")
                    sources.append(candidate)
                except Exception:
                    pass

    # Classify by walking the failure-mode priority list (first match wins).
    for needle, info in _FAILURE_PATTERNS:
        if needle in text:
            payload = {
                "matched_pattern": needle,
                "sources_inspected": sources,
                **info,
            }
            return payload

    # Fallback: no known failure pattern matched.
    return {
        "failure_class": "unknown",
        "severity": "info",
        "matched_pattern": None,
        "sources_inspected": sources,
        "root_cause": "No known failure pattern matched.",
        "fix_recipe": (
            "Run analyze_grasp_case to see workflow status. Read the "
            "session log directly to identify which step errored."
        ),
        "next_actions": [
            "analyze_grasp_case(working_dir)",
            "read_grasp_session_log(working_dir)",
        ],
    }


# --- Internal helpers ----------------------------------------------------

def _last_step_run(steps: list[dict[str, Any]]) -> str:
    last = None
    for s in steps:
        if s["status"] == "ran":
            last = s["step"]
    return last or "(none)"


def _next_step(after: str) -> str:
    order = list(_STEP_ARTIFACTS.keys())
    try:
        i = order.index(after)
    except ValueError:
        return "(unknown)"
    return order[i + 1] if i + 1 < len(order) else "(workflow complete)"

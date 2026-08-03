"""Compatibility facade for run, campaign, workflow, and batch services.

Run-record persistence lives in ``run_records``. Existing callers can keep
importing those functions from this module while the remaining campaign,
workflow, and batch responsibilities are split along their own seams.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from chemtools.core.registry_db import connect_registry as _connect
from chemtools.core.run_records import (
    get_run_summary,
    list_runs,
    register_run,
    row_to_dict as _row_to_dict,
    update_run_status,
)


# ---------------------------------------------------------------------------
# Campaign management
# ---------------------------------------------------------------------------

def create_campaign(
    name: str,
    description: str | None = None,
    tags: dict[str, Any] | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Create a new campaign. Returns the campaign_id."""
    conn = _connect(db_path)
    now = datetime.now(timezone.utc).isoformat()
    try:
        cur = conn.execute(
            "INSERT INTO campaigns (name, description, created_at, tags) VALUES (?, ?, ?, ?)",
            (name, description, now, json.dumps(tags) if tags else None),
        )
        conn.commit()
        return {"campaign_id": cur.lastrowid, "name": name}
    finally:
        conn.close()


def get_campaign_status(
    campaign_id: int | None = None,
    name: str | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Get aggregate status for a campaign."""
    conn = _connect(db_path)
    try:
        # Resolve campaign
        if campaign_id is not None:
            camp = conn.execute("SELECT * FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
        elif name is not None:
            camp = conn.execute("SELECT * FROM campaigns WHERE name = ?", (name,)).fetchone()
        else:
            return {"error": "Provide campaign_id or name."}
        if camp is None:
            return {"error": f"Campaign not found."}

        cid = camp["id"]

        # Aggregate run statuses
        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM runs WHERE campaign_id = ? GROUP BY status",
            (cid,),
        ).fetchall()
        status_counts: dict[str, int] = {r["status"]: r["cnt"] for r in rows}
        total = sum(status_counts.values())

        completed = status_counts.get("completed", 0)
        running = status_counts.get("running", 0) + status_counts.get("submitted", 0)
        failed = status_counts.get("failed", 0) + status_counts.get("oom", 0)
        pending = status_counts.get("pending", 0)
        timelimited = status_counts.get("timelimited", 0)

        # Energies available
        energy_count = conn.execute(
            "SELECT COUNT(*) FROM runs WHERE campaign_id = ? AND energy_hartree IS NOT NULL",
            (cid,),
        ).fetchone()[0]

        # Estimate remaining time from completed runs
        avg_wall = conn.execute(
            "SELECT AVG(walltime_used_sec) FROM runs WHERE campaign_id = ? AND status = 'completed' AND walltime_used_sec IS NOT NULL",
            (cid,),
        ).fetchone()[0]
        remaining_runs = total - completed - failed
        est_remaining_hours = None
        if avg_wall and remaining_runs > 0:
            est_remaining_hours = round(remaining_runs * avg_wall / 3600, 1)

        return {
            "campaign_id": cid,
            "name": camp["name"],
            "description": camp["description"],
            "total_runs": total,
            "completed": completed,
            "running": running,
            "failed": failed,
            "timelimited": timelimited,
            "pending": pending,
            "completion_pct": round(100 * completed / total, 1) if total > 0 else 0,
            "energies_available": energy_count,
            "estimated_remaining_hours": est_remaining_hours,
            "status_breakdown": status_counts,
        }
    finally:
        conn.close()


def get_campaign_energies(
    campaign_id: int | None = None,
    name: str | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Get energy table for a campaign's completed runs."""
    conn = _connect(db_path)
    try:
        if campaign_id is None and name is not None:
            camp = conn.execute("SELECT id FROM campaigns WHERE name = ?", (name,)).fetchone()
            if camp:
                campaign_id = camp["id"]
        if campaign_id is None:
            return {"error": "Campaign not found."}

        rows = conn.execute(
            """SELECT id, program, job_name, method, basis, charge, multiplicity,
                      energy_hartree, h_hartree, g_hartree, imaginary_modes, status
               FROM runs WHERE campaign_id = ? AND energy_hartree IS NOT NULL
               ORDER BY energy_hartree""",
            (campaign_id,),
        ).fetchall()

        runs = [_row_to_dict(r) for r in rows]

        # Compute relative energies
        if runs:
            ref_e = runs[0]["energy_hartree"]  # lowest energy
            ha_to_kcal = 627.5094740631
            for r in runs:
                r["relative_energy_kcal_mol"] = round(
                    (r["energy_hartree"] - ref_e) * ha_to_kcal, 2
                )
                if r.get("g_hartree") is not None and runs[0].get("g_hartree") is not None:
                    r["relative_g_kcal_mol"] = round(
                        (r["g_hartree"] - runs[0]["g_hartree"]) * ha_to_kcal, 2
                    )

        return {
            "campaign_id": campaign_id,
            "n_runs": len(runs),
            "reference_energy_hartree": runs[0]["energy_hartree"] if runs else None,
            "runs": runs,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Workflow DAG
# ---------------------------------------------------------------------------

def create_workflow(
    name: str,
    steps: list[dict[str, Any]],
    protocol: str | None = None,
    campaign_id: int | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Create a workflow with step dependencies.

    Each step dict should have:
      - id: unique step identifier (e.g. "opt", "freq")
      - depends_on: step id this depends on (or None)
      - input_file: path to input (or None if generated from prior step)
      - profile: runner profile name
      - auto_input: optional dict describing how to generate input from prior step
    """
    conn = _connect(db_path)
    now = datetime.now(timezone.utc).isoformat()
    try:
        cur = conn.execute(
            "INSERT INTO workflows (name, protocol, campaign_id, state, steps_json, created_at) VALUES (?, ?, ?, 'pending', ?, ?)",
            (name, protocol, campaign_id, json.dumps(steps), now),
        )
        conn.commit()
        return {
            "workflow_id": cur.lastrowid,
            "name": name,
            "n_steps": len(steps),
            "steps": [s["id"] for s in steps],
        }
    finally:
        conn.close()


def advance_workflow(
    workflow_id: int,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Check workflow state and identify which steps are ready to launch.

    Does NOT actually launch jobs — returns the steps that are unblocked
    and ready to go, along with the current state of all steps.
    The caller (model or automation) decides whether to launch them.
    """
    conn = _connect(db_path)
    try:
        wf = conn.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,)).fetchone()
        if wf is None:
            return {"error": f"Workflow {workflow_id} not found."}

        steps = json.loads(wf["steps_json"])

        # Get all runs for this workflow
        runs = conn.execute(
            "SELECT * FROM runs WHERE workflow_id = ? ORDER BY id",
            (workflow_id,),
        ).fetchall()

        # Build step → run status mapping
        step_runs: dict[str, list[dict[str, Any]]] = {}
        for r in runs:
            sid = r["workflow_step_id"]
            if sid:
                step_runs.setdefault(sid, []).append(_row_to_dict(r))

        # Determine each step's status
        step_states: list[dict[str, Any]] = []
        completed_steps: set[str] = set()
        failed_steps: set[str] = set()
        running_steps: set[str] = set()

        for step in steps:
            sid = step["id"]
            step_run_list = step_runs.get(sid, [])
            if not step_run_list:
                step_status = "pending"
            else:
                latest = step_run_list[-1]
                step_status = latest["status"]
                if step_status == "completed":
                    completed_steps.add(sid)
                elif step_status in ("failed", "oom"):
                    failed_steps.add(sid)
                elif step_status in ("running", "submitted"):
                    running_steps.add(sid)

            step_states.append({
                "step_id": sid,
                "depends_on": step.get("depends_on"),
                "status": step_status,
                "run_count": len(step_run_list),
                "latest_run_id": step_run_list[-1]["id"] if step_run_list else None,
            })

        # Find steps that are ready to launch
        ready_to_launch: list[dict[str, Any]] = []
        for step in steps:
            sid = step["id"]
            if sid in completed_steps or sid in running_steps:
                continue
            dep = step.get("depends_on")
            if dep is None or dep in completed_steps:
                ready_to_launch.append({
                    "step_id": sid,
                    "input_file": step.get("input_file"),
                    "profile": step.get("profile"),
                    "auto_input": step.get("auto_input"),
                    "depends_on": dep,
                })

        # Overall workflow state
        all_done = len(completed_steps) == len(steps)
        has_failures = len(failed_steps) > 0
        if all_done:
            wf_state = "completed"
        elif has_failures and not running_steps and not ready_to_launch:
            wf_state = "failed"
        elif running_steps:
            wf_state = "running"
        elif ready_to_launch:
            wf_state = "ready"
        else:
            wf_state = "blocked"

        # Update workflow state
        conn.execute("UPDATE workflows SET state = ? WHERE id = ?", (wf_state, workflow_id))
        conn.commit()

        return {
            "workflow_id": workflow_id,
            "name": wf["name"],
            "state": wf_state,
            "steps": step_states,
            "completed": sorted(completed_steps),
            "running": sorted(running_steps),
            "failed": sorted(failed_steps),
            "ready_to_launch": ready_to_launch,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Batch input generation
# ---------------------------------------------------------------------------

def generate_input_batch(
    template_input: str,
    vary: dict[str, list[Any]],
    output_dir: str,
    naming_pattern: str = "{stem}_{key}_{value}",
    campaign_id: int | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Generate multiple NWChem inputs by varying parameters from a template.

    Parameters
    ----------
    template_input : str
        Path to the base .nw file.
    vary : dict
        Keys are parameter names, values are lists to iterate over.
        Supported keys: charge, mult, task, memory, and any block.keyword
        (e.g. "dft.xc" for the functional).
        All combinations are generated (Cartesian product).
    output_dir : str
        Directory to write generated inputs.
    naming_pattern : str
        Filename pattern. Available placeholders: {stem}, {key}, {value},
        {idx} (0-based index), plus any vary key names.
    campaign_id : int | None
        If set, register each generated input in this campaign.

    Returns
    -------
    dict with generated file list and count.
    """
    import itertools
    from pathlib import Path as _Path

    template_path = _Path(template_input)
    if not template_path.exists():
        return {"error": f"Template not found: {template_input}"}

    template_text = template_path.read_text()
    stem = template_path.stem
    out_dir = _Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build all combinations
    keys = list(vary.keys())
    value_lists = [vary[k] for k in keys]
    combos = list(itertools.product(*value_lists))

    generated: list[dict[str, Any]] = []
    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        # Apply changes via the registered program's Drafter.patch_input.
        # Default to nwchem since the existing _apply_change logic was
        # NWChem-specific; the dispatch lets a future Molpro/Molcas
        # plugin handle the same call without forking generate_input_batch.
        from chemtools.core import registry as _program_registry
        try:
            _plugin = _program_registry.resolve(program="nwchem")
        except Exception:
            _plugin = None
        change_map = {k: str(v) for k, v in params.items()}
        if _plugin is not None and getattr(_plugin, "drafter", None) is not None:
            text = _plugin.drafter.patch_input(template_text, change_map)
        else:
            # Fallback for environments where the plugin failed to load.
            text = template_text
            for key, value in params.items():
                text = _apply_change(text, key, str(value))

        # Build filename
        fmt_vars = {"stem": stem, "idx": idx, **{k: str(v) for k, v in params.items()}}
        # Build a simple name from all varied params
        parts = [f"{k}{v}" for k, v in params.items()]
        fmt_vars["key"] = "_".join(keys)
        fmt_vars["value"] = "_".join(str(v) for v in combo)
        try:
            filename = naming_pattern.format(**fmt_vars) + ".nw"
        except KeyError:
            filename = f"{stem}_{'_'.join(parts)}.nw"

        out_path = out_dir / filename
        out_path.write_text(text)

        entry: dict[str, Any] = {
            "file": str(out_path),
            "parameters": params,
        }

        # Register in campaign if requested
        if campaign_id is not None:
            reg = register_run(
                job_name=out_path.stem,
                input_file=str(out_path),
                campaign_id=campaign_id,
                charge=params.get("charge"),
                multiplicity=params.get("mult"),
                tags=params,
                db_path=db_path,
            )
            entry["run_id"] = reg["run_id"]

        generated.append(entry)

    return {
        "template": str(template_path),
        "output_dir": str(out_dir),
        "n_generated": len(generated),
        "varied_parameters": keys,
        "n_combinations": len(combos),
        "generated": generated,
    }


# Fallback path: kept here as a safety net for callers that import this
# module without the program plugins available (e.g. early bootstrapping
# or tests with a stripped install). The canonical patch logic now lives
# in chemtools.programs.nwchem.input.general.create_nwchem_input_variant
# and is reached via plugin.drafter.patch_input in generate_input_batch.
def _apply_change(text: str, key: str, value: str) -> str:
    """Apply a single parameter change to NWChem input text.

    Supports:
      - charge, mult/multiplicity: top-level directives
      - task: task line
      - memory: memory directive
      - block.keyword (e.g. dft.xc): keyword within a block
    """
    import re

    key_lower = key.lower()

    if key_lower == "charge":
        if re.search(r"^\s*charge\s+", text, re.MULTILINE):
            text = re.sub(r"^(\s*charge\s+).*$", rf"\g<1>{value}", text, flags=re.MULTILINE)
        else:
            # Insert before first block or task
            text = f"charge {value}\n" + text
        return text

    if key_lower in ("mult", "multiplicity"):
        # NWChem uses "scf; nopen N; end" for multiplicity
        nopen = int(value) - 1
        if re.search(r"^\s*nopen\s+", text, re.MULTILINE):
            text = re.sub(r"^(\s*nopen\s+).*$", rf"\g<1>{nopen}", text, flags=re.MULTILINE)
        else:
            # Try to insert inside scf/dft block
            for block in ("scf", "dft"):
                pattern = rf"^(\s*{block}\s*\n)"
                if re.search(pattern, text, re.MULTILINE):
                    text = re.sub(pattern, rf"\g<1>  nopen {nopen}\n", text, flags=re.MULTILINE)
                    break
        return text

    if key_lower == "task":
        text = re.sub(r"^(\s*task\s+).*$", rf"\g<1>{value}", text, flags=re.MULTILINE)
        return text

    if key_lower == "memory":
        text = re.sub(r"^(\s*memory\s+).*$", rf"\g<1>{value}", text, flags=re.MULTILINE)
        return text

    # block.keyword pattern (e.g. dft.xc → find "xc" inside "dft...end")
    if "." in key_lower:
        block, kw = key_lower.split(".", 1)
        pattern = re.compile(
            rf"^(\s*{re.escape(block)}\s*\n(?:.*\n)*?\s*){re.escape(kw)}\s+\S+",
            re.MULTILINE,
        )
        match = pattern.search(text)
        if match:
            text = pattern.sub(rf"\g<1>{kw} {value}", text, count=1)
        return text

    return text

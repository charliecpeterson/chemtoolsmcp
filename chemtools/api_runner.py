from __future__ import annotations

import math
import os
import signal
from typing import Any

from .common import detect_program, read_text
from .diagnostics import (
    diagnose_nwchem_output,
    parse_scf,
)
from .runner import (
    load_runner_profiles,
    render_nwchem_run,
    run_nwchem,
    inspect_nwchem_run_status,
    tail_text_file,
    watch_nwchem_run as watch_nwchem_run_payload,
)
from . import nwchem
from ._api_utils import _TRANSITION_METALS, _COVALENT_RADII

# Forward reference - review_nwchem_mcscf_case is in api_strategy.py
# We import lazily to avoid circular imports
def _get_review_nwchem_mcscf_case():
    from .api_strategy import review_nwchem_mcscf_case
    return review_nwchem_mcscf_case



def inspect_runner_profiles(profiles_path: str | None = None) -> dict[str, Any]:
    payload = load_runner_profiles(profiles_path)
    profiles = payload.get("profiles", {})
    return {
        "profiles_path": payload["__source__"],
        "profile_names": sorted(profiles.keys()),
        "profiles": {
            name: {
                "description": profile.get("description"),
                "launcher_kind": (profile.get("launcher") or {}).get("kind", "direct"),
            }
            for name, profile in profiles.items()
        },
    }


def launch_nwchem_run(
    input_path: str,
    profile: str,
    profiles_path: str | None = None,
    job_name: str | None = None,
    resource_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, str] | None = None,
    write_script: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        result = render_nwchem_run(
            input_path=input_path,
            profile=profile,
            profiles_path=profiles_path,
            job_name=job_name,
            resource_overrides=resource_overrides,
            env_overrides=env_overrides,
        )
        result.pop("environment", None)
        return result
    return run_nwchem(
        input_path=input_path,
        profile=profile,
        profiles_path=profiles_path,
        job_name=job_name,
        resource_overrides=resource_overrides,
        env_overrides=env_overrides,
        execute=True,
        write_script=write_script,
    )


def prepare_nwchem_run(
    input_path: str,
    profile: str,
    profiles_path: str | None = None,
    job_name: str | None = None,
    resource_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Deprecated: use launch_nwchem_run with dry_run=True. Kept for backward compatibility."""
    return launch_nwchem_run(
        input_path=input_path,
        profile=profile,
        profiles_path=profiles_path,
        job_name=job_name,
        resource_overrides=resource_overrides,
        env_overrides=env_overrides,
        dry_run=True,
    )


def check_nwchem_run_status(
    output_path: str | None = None,
    input_path: str | None = None,
    error_path: str | None = None,
    process_id: int | None = None,
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    return inspect_nwchem_run_status(
        output_path=output_path,
        input_path=input_path,
        error_path=error_path,
        process_id=process_id,
        profile=profile,
        job_id=job_id,
        profiles_path=profiles_path,
    )


def review_nwchem_progress(
    output_path: str,
    input_path: str | None = None,
    error_path: str | None = None,
    process_id: int | None = None,
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    status = check_nwchem_run_status(
        output_path=output_path,
        input_path=input_path,
        error_path=error_path,
        process_id=process_id,
        profile=profile,
        job_id=job_id,
        profiles_path=profiles_path,
    )
    progress = status.get("progress_summary") or {}
    output_summary = status.get("output_summary") or {}
    output_file = status.get("output_file") or {}
    error_file = status.get("error_file") or {}
    process = status.get("process") or {}
    headline = _build_nwchem_progress_headline(status["overall_status"], progress, output_summary)
    intervention = _assess_nwchem_progress_intervention(
        output_path=output_path,
        status=status,
    )

    bullets: list[str] = [f"Overall status: {status['overall_status']}"]
    if headline:
        bullets.append(headline)
    status_line = output_summary.get("status_line") or progress.get("status_line")
    if status_line:
        bullets.append(status_line)
    if output_file.get("exists"):
        size_bytes = output_file.get("size_bytes")
        if size_bytes is not None:
            bullets.append(f"Output size: {size_bytes} bytes")
    if process.get("status") and process.get("status") != "unknown":
        bullets.append(f"Process: {process['status']}")
    if error_file.get("exists") and (error_file.get("size_bytes") or 0) > 0:
        bullets.append(f"Error file has content: {error_file['size_bytes']} bytes")
    if intervention["assessment"] != "not_applicable":
        bullets.append(f"Intervention: {intervention['recommended_action']}")

    return {
        "output_file": output_file,
        "input_file": status.get("input_file"),
        "error_file": error_file,
        "overall_status": status["overall_status"],
        "progress_headline": headline,
        "current_phase": output_summary.get("current_phase") or progress.get("current_phase"),
        "current_task_kind": output_summary.get("current_task_kind") or progress.get("current_task_kind"),
        "requested_tasks": progress.get("requested_tasks"),
        "current_requested_task": progress.get("current_requested_task"),
        "next_requested_task": progress.get("next_requested_task"),
        "progress_summary": progress,
        "intervention": intervention,
        "summary_bullets": bullets,
        "summary_text": "\n".join(f"- {bullet}" for bullet in bullets),
    }


def terminate_nwchem_run(process_id: int, signal_name: str = "term") -> dict[str, Any]:
    normalized = signal_name.strip().lower()
    if normalized in {"term", "sigterm", "terminate"}:
        sig = signal.SIGTERM
        used = "SIGTERM"
    elif normalized in {"kill", "sigkill"}:
        sig = signal.SIGKILL
        used = "SIGKILL"
    else:
        raise ValueError("signal_name must be one of: term, kill")

    try:
        os.kill(process_id, sig)
        sent = True
        error = None
    except ProcessLookupError:
        sent = False
        error = "process_not_found"
    except PermissionError:
        sent = False
        error = "permission_denied"

    return {
        "process_id": process_id,
        "signal": used,
        "sent": sent,
        "error": error,
    }


def watch_nwchem_run(
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
    watched = watch_nwchem_run_payload(
        output_path=output_path,
        input_path=input_path,
        error_path=error_path,
        process_id=process_id,
        profile=profile,
        job_id=job_id,
        profiles_path=profiles_path,
        poll_interval_seconds=poll_interval_seconds,
        adaptive_polling=adaptive_polling,
        max_poll_interval_seconds=max_poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        max_polls=max_polls,
        history_limit=history_limit,
    )
    final_progress = (
        review_nwchem_progress(
            output_path=output_path,
            input_path=input_path,
            error_path=error_path,
            process_id=process_id,
            profile=profile,
            job_id=job_id,
            profiles_path=profiles_path,
        )
        if output_path
        else None
    )
    return {
        "terminal": watched["terminal"],
        "stop_reason": watched["stop_reason"],
        "poll_count": watched["poll_count"],
        "elapsed_seconds": watched["elapsed_seconds"],
        "adaptive_polling": watched["adaptive_polling"],
        "max_poll_interval_seconds": watched["max_poll_interval_seconds"],
        "history_limit": watched["history_limit"],
        "last_sleep_seconds": watched["last_sleep_seconds"],
        "history": watched["history"],
        "final_status": watched["final_status"],
        "final_progress": final_progress,
        "summary_text": (final_progress or {}).get("summary_text"),
    }


def _build_nwchem_progress_headline(
    overall_status: str,
    progress: dict[str, Any],
    output_summary: dict[str, Any],
) -> str | None:
    current_requested = progress.get("current_requested_task")
    next_requested = progress.get("next_requested_task")
    current_phase = output_summary.get("current_phase") or progress.get("current_phase")

    if current_requested is None:
        return None

    module = (current_requested.get("module") or "").upper()
    operation = current_requested.get("operation") or current_requested.get("kind") or "task"

    if current_phase == "optimization_in_progress":
        headline = f"{module} optimization is still running" if module else "Optimization is still running"
    elif current_phase == "optimization_completed":
        headline = f"{module} optimization has completed" if module else "Optimization has completed"
    elif current_phase == "frequency_in_progress_or_interrupted":
        headline = f"{module} frequency analysis has started but is not complete" if module else "Frequency analysis has started but is not complete"
    elif current_phase == "frequency_completed":
        headline = f"{module} frequency analysis has completed" if module else "Frequency analysis has completed"
    elif current_phase == "single_point_task":
        headline = f"{module} single-point task is {overall_status}" if module else f"Single-point task is {overall_status}"
    elif current_phase == "property_task":
        headline = f"{module} property task is {overall_status}" if module else f"Property task is {overall_status}"
    else:
        label = current_requested.get("label") or operation
        headline = f"{label} is {current_phase or overall_status}"

    if current_requested.get("kind") == "optimization":
        step = progress.get("optimization_last_step")
        if step is not None:
            headline += f" at step {step}"
    elif current_requested.get("kind") == "frequency":
        mode_count = progress.get("frequency_mode_count")
        if mode_count:
            headline += f" with {mode_count} modes parsed"

    if next_requested is not None:
        next_module = (next_requested.get("module") or "").upper()
        next_operation = next_requested.get("operation") or next_requested.get("kind") or "task"
        if next_requested.get("kind") == "frequency":
            next_text = f"{next_module} frequency has not started yet" if next_module else "Frequency has not started yet"
        elif next_requested.get("kind") == "optimization":
            next_text = f"{next_module} optimization has not started yet" if next_module else "Optimization has not started yet"
        else:
            next_text = f"next task has not started yet: {next_requested.get('label') or next_operation}"
        headline += f"; {next_text}"

    return headline + "."


def _assess_nwchem_progress_intervention(
    *,
    output_path: str,
    status: dict[str, Any],
) -> dict[str, Any]:
    overall_status = status.get("overall_status")
    if overall_status in {"completed_success", "completed_failed", "error_only"}:
        return {
            "assessment": "not_applicable",
            "should_terminate_process": False,
            "confidence": "low",
            "basis": "terminal_status",
            "recommended_action": "no_live_intervention_needed",
            "reasons": [],
            "geometry_alerts": [],
            "primary_geometry_alert": None,
        }

    output_summary = status.get("output_summary") or {}
    current_phase = output_summary.get("current_phase") or (status.get("progress_summary") or {}).get("current_phase")
    contents = read_text(output_path)
    if detect_program(contents) != "nwchem":
        return {
            "assessment": "not_applicable",
            "should_terminate_process": False,
            "confidence": "low",
            "basis": "unsupported_program",
            "recommended_action": "continue_monitoring",
            "reasons": [],
            "geometry_alerts": [],
            "primary_geometry_alert": None,
        }

    if current_phase == "optimization_in_progress":
        trajectory = nwchem.parse_trajectory(output_path, contents, include_positions=True)
        energies = [value for value in trajectory.get("energies_hartree", []) if value is not None]
        recent = energies[-4:]
        positive_steps = sum(1 for idx in range(1, len(recent)) if recent[idx] > recent[idx - 1] + 1e-4)
        net_rise = (recent[-1] - min(recent[:-1])) if len(recent) >= 2 else None
        gradient_trend = _assess_optimization_gradient_trend(trajectory.get("step_metrics") or [])
        max_atom_displacement = _compute_last_frame_max_atomic_displacement(trajectory.get("frames") or [])
        max_pair_distance_change = _compute_last_frame_pair_distance_change(trajectory.get("frames") or [])
        structure_drift = _assess_optimization_structure_drift(trajectory.get("frames") or [])
        severe_geometry_drift = bool(
            (max_atom_displacement is not None and max_atom_displacement > 1.5)
            or (max_pair_distance_change is not None and max_pair_distance_change > 1.0)
            or structure_drift["severe"]
        )
        if (
            severe_geometry_drift
            or (
                trajectory.get("step_count", 0) >= 6
                and len(recent) >= 4
                and positive_steps >= 3
                and net_rise is not None
                and net_rise > 0.05
                and gradient_trend["worsening"]
                and (
                    (max_atom_displacement is not None and max_atom_displacement > 0.75)
                    or (max_pair_distance_change is not None and max_pair_distance_change > 0.5)
                )
            )
        ):
            reasons: list[str] = []
            if positive_steps >= 3 and net_rise is not None and net_rise > 0.05:
                reasons.append(f"recent optimization energies rose by {net_rise:.6f} Ha")
            if gradient_trend["worsening"]:
                reasons.append(
                    f"gradients are worsening (gmax {gradient_trend['first_gmax']:.5f} -> {gradient_trend['last_gmax']:.5f})"
                )
            if max_atom_displacement is not None:
                reasons.append(f"largest last-step atomic displacement {max_atom_displacement:.3f} A")
            if max_pair_distance_change is not None:
                reasons.append(f"largest last-step pair-distance change {max_pair_distance_change:.3f} A")
            reasons.extend(structure_drift["reasons"])
            return {
                "assessment": "kill_recommended",
                "should_terminate_process": True,
                "confidence": "medium",
                "basis": "optimization",
                "recommended_action": "kill_and_restart_from_better_geometry_or_guess",
                "reasons": reasons,
                "geometry_alerts": structure_drift.get("alerts", []),
                "primary_geometry_alert": structure_drift.get("primary_alert"),
            }
        return {
            "assessment": "continue",
            "should_terminate_process": False,
            "confidence": "low",
            "basis": "optimization",
            "recommended_action": "continue_monitoring_optimization",
            "reasons": [],
            "geometry_alerts": structure_drift.get("alerts", []),
            "primary_geometry_alert": structure_drift.get("primary_alert"),
        }

    if current_phase in {"single_point_task", "property_task", "other", "unknown"}:
        scf = parse_scf(output_path)
        last_run = scf.get("last_run") or {}
        trend = ((last_run.get("trend") or {}).get("pattern")) or ((scf.get("trend") or {}).get("pattern"))
        iteration_count = last_run.get("iteration_count") or scf.get("iteration_count") or 0
        density_ratio = ((last_run.get("trend") or {}).get("density_ratio_recent"))
        if density_ratio is None:
            density_ratio = (scf.get("trend") or {}).get("density_ratio_recent")

        if trend in {"oscillatory", "oscillatory_but_converged"} and iteration_count >= 12:
            return {
                "assessment": "kill_recommended",
                "should_terminate_process": True,
                "confidence": "high",
                "basis": "scf",
                "recommended_action": "kill_and_change_scf_strategy",
                "reasons": [f"SCF is oscillatory after {iteration_count} iterations"],
                "geometry_alerts": [],
                "primary_geometry_alert": None,
            }
        if trend == "stalled" and iteration_count >= 20 and (density_ratio is None or density_ratio > 0.5):
            return {
                "assessment": "kill_recommended",
                "should_terminate_process": True,
                "confidence": "high",
                "basis": "scf",
                "recommended_action": "kill_and_restart_with_stabilization_changes",
                "reasons": [f"SCF appears stalled after {iteration_count} iterations"],
                "geometry_alerts": [],
                "primary_geometry_alert": None,
            }
        if trend in {"slow_improving", "nearly_converged"}:
            return {
                "assessment": "continue",
                "should_terminate_process": False,
                "confidence": "medium",
                "basis": "scf",
                "recommended_action": "continue_monitoring_scf",
                "reasons": [f"SCF trend is {trend}"],
                "geometry_alerts": [],
                "primary_geometry_alert": None,
            }
        return {
            "assessment": "continue",
            "should_terminate_process": False,
            "confidence": "low",
            "basis": "scf",
            "recommended_action": "continue_monitoring_scf",
            "reasons": [],
            "geometry_alerts": [],
            "primary_geometry_alert": None,
        }

    return {
        "assessment": "continue",
        "should_terminate_process": False,
        "confidence": "low",
        "basis": "general",
        "recommended_action": "continue_monitoring",
        "reasons": [],
        "geometry_alerts": [],
        "primary_geometry_alert": None,
    }


def _assess_optimization_gradient_trend(step_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    recent = [metric for metric in step_metrics[-4:] if metric.get("gmax") is not None and metric.get("grms") is not None]
    if len(recent) < 3:
        return {
            "worsening": False,
            "first_gmax": 0.0,
            "last_gmax": 0.0,
            "first_grms": 0.0,
            "last_grms": 0.0,
        }
    gmax_values = [float(metric["gmax"]) for metric in recent]
    grms_values = [float(metric["grms"]) for metric in recent]
    gmax_rises = sum(1 for idx in range(1, len(gmax_values)) if gmax_values[idx] > gmax_values[idx - 1] * 1.05)
    grms_rises = sum(1 for idx in range(1, len(grms_values)) if grms_values[idx] > grms_values[idx - 1] * 1.05)
    worsening = (
        (gmax_rises >= 2 and gmax_values[-1] > gmax_values[0] * 1.5)
        or (grms_rises >= 2 and grms_values[-1] > grms_values[0] * 1.5)
    )
    return {
        "worsening": worsening,
        "first_gmax": gmax_values[0],
        "last_gmax": gmax_values[-1],
        "first_grms": grms_values[0],
        "last_grms": grms_values[-1],
    }


def _compute_last_frame_max_atomic_displacement(frames: list[dict[str, Any]]) -> float | None:
    if len(frames) < 2:
        return None
    previous = frames[-2].get("positions_angstrom")
    current = frames[-1].get("positions_angstrom")
    if not previous or not current or len(previous) != len(current):
        return None
    return max(_distance(point_a, point_b) for point_a, point_b in zip(previous, current))


def _compute_last_frame_pair_distance_change(frames: list[dict[str, Any]]) -> float | None:
    if len(frames) < 2:
        return None
    previous = frames[-2].get("positions_angstrom")
    current = frames[-1].get("positions_angstrom")
    if not previous or not current or len(previous) != len(current):
        return None
    atom_count = len(current)
    if atom_count < 2 or atom_count > 40:
        return None
    max_change = 0.0
    for idx in range(atom_count):
        for jdx in range(idx + 1, atom_count):
            before = _distance(previous[idx], previous[jdx])
            after = _distance(current[idx], current[jdx])
            max_change = max(max_change, abs(after - before))
    return max_change


def _assess_optimization_structure_drift(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if len(frames) < 2:
        return {"severe": False, "reasons": [], "alerts": [], "primary_alert": None}
    initial_labels = frames[0].get("labels") or []
    final_labels = frames[-1].get("labels") or []
    initial_positions = frames[0].get("positions_angstrom") or []
    final_positions = frames[-1].get("positions_angstrom") or []
    if (
        not initial_labels
        or not final_labels
        or len(initial_labels) != len(final_labels)
        or len(initial_positions) != len(final_positions)
        or len(initial_labels) != len(initial_positions)
    ):
        return {"severe": False, "reasons": [], "alerts": [], "primary_alert": None}

    reasons: list[str] = []
    alerts: list[dict[str, Any]] = []
    severe = False
    nearest_neighbor_flags: list[tuple[str, float, float]] = []
    metal_ligand_flags: list[tuple[str, float, float]] = []
    pair_blowout = _largest_pair_distance_blowout(
        initial_labels=initial_labels,
        initial_positions=initial_positions,
        final_positions=final_positions,
    )
    metal_ligand_blowout = _largest_metal_ligand_distance_blowout(
        initial_labels=initial_labels,
        initial_positions=initial_positions,
        final_positions=final_positions,
    )

    for idx, label in enumerate(initial_labels):
        start_nearest = _nearest_neighbor_distance(initial_positions, idx)
        end_nearest = _nearest_neighbor_distance(final_positions, idx)
        if start_nearest is None or end_nearest is None or start_nearest <= 0:
            continue
        delta = end_nearest - start_nearest
        ratio = end_nearest / start_nearest
        if delta > 0.8 and ratio > 1.5:
            nearest_neighbor_flags.append((label, delta, ratio))
        if label in _TRANSITION_METALS and delta > 0.7 and ratio > 1.35:
            metal_ligand_flags.append((label, delta, ratio))

    if nearest_neighbor_flags:
        label, delta, ratio = max(nearest_neighbor_flags, key=lambda item: item[1])
        message = f"nearest-neighbor separation around {label} increased by {delta:.3f} A (x{ratio:.2f})"
        reasons.append(message)
        alerts.append(
            {
                "kind": "nearest_neighbor_separation",
                "atom_label": label,
                "delta_angstrom": delta,
                "ratio": ratio,
                "message": message,
                "severity_score": delta,
            }
        )
        severe = True
    if pair_blowout is not None:
        message = (
            f"contact {pair_blowout['pair_label']} increased by {pair_blowout['delta']:.3f} A "
            f"({pair_blowout['initial_distance']:.3f} -> {pair_blowout['final_distance']:.3f} A)"
        )
        reasons.append(message)
        alerts.append(
            {
                "kind": "contact_blowout",
                "pair_label": pair_blowout["pair_label"],
                "pair": list(pair_blowout["pair"]),
                "initial_distance_angstrom": pair_blowout["initial_distance"],
                "final_distance_angstrom": pair_blowout["final_distance"],
                "delta_angstrom": pair_blowout["delta"],
                "ratio": pair_blowout["ratio"],
                "message": message,
                "severity_score": pair_blowout["delta"],
            }
        )
        severe = True
    if metal_ligand_flags:
        label, delta, ratio = max(metal_ligand_flags, key=lambda item: item[1])
        message = (
            f"possible metal-ligand dissociation around {label}: nearest contact increased by {delta:.3f} A (x{ratio:.2f})"
        )
        reasons.append(message)
        alerts.append(
            {
                "kind": "metal_ligand_contact_expansion",
                "metal_label": label,
                "delta_angstrom": delta,
                "ratio": ratio,
                "message": message,
                "severity_score": delta,
            }
        )
        severe = True
    if metal_ligand_blowout is not None:
        message = (
            f"possible metal-ligand dissociation: ligand fragment {metal_ligand_blowout['fragment_label']} from "
            f"{metal_ligand_blowout['metal_label']} via {metal_ligand_blowout['pair_label']} increased by "
            f"{metal_ligand_blowout['delta']:.3f} A "
            f"({metal_ligand_blowout['initial_distance']:.3f} -> {metal_ligand_blowout['final_distance']:.3f} A)"
        )
        reasons.append(message)
        alerts.append(
            {
                "kind": "metal_ligand_dissociation",
                "metal_label": metal_ligand_blowout["metal_label"],
                "ligand_label": metal_ligand_blowout["ligand_label"],
                "fragment_label": metal_ligand_blowout["fragment_label"],
                "fragment_labels": metal_ligand_blowout["fragment_labels"],
                "pair_label": metal_ligand_blowout["pair_label"],
                "pair": list(metal_ligand_blowout["pair"]),
                "initial_distance_angstrom": metal_ligand_blowout["initial_distance"],
                "final_distance_angstrom": metal_ligand_blowout["final_distance"],
                "delta_angstrom": metal_ligand_blowout["delta"],
                "ratio": metal_ligand_blowout["ratio"],
                "message": message,
                "severity_score": metal_ligand_blowout["delta"] + 0.25,
            }
        )
        severe = True

    radial_change = _max_radius_change(initial_positions, final_positions)
    if radial_change is not None and radial_change > 2.0:
        message = f"maximum radial expansion from origin increased by {radial_change:.3f} A"
        reasons.append(message)
        alerts.append(
            {
                "kind": "radial_expansion",
                "delta_angstrom": radial_change,
                "message": message,
                "severity_score": radial_change,
            }
        )
        severe = True

    def _alert_priority(alert: dict[str, Any]) -> tuple[int, float]:
        kind = str(alert.get("kind") or "")
        if kind == "metal_ligand_dissociation":
            return (4, float(alert.get("severity_score", 0.0)))
        if kind == "contact_blowout":
            return (3, float(alert.get("severity_score", 0.0)))
        if kind in {"metal_ligand_contact_expansion", "nearest_neighbor_separation"}:
            return (2, float(alert.get("severity_score", 0.0)))
        return (1, float(alert.get("severity_score", 0.0)))

    primary_alert = max(alerts, key=_alert_priority, default=None)
    return {"severe": severe, "reasons": reasons, "alerts": alerts, "primary_alert": primary_alert}


def _nearest_neighbor_distance(positions: list[list[float]], index: int) -> float | None:
    if len(positions) < 2:
        return None
    distances = [
        _distance(positions[index], other)
        for jdx, other in enumerate(positions)
        if jdx != index
    ]
    return min(distances) if distances else None


def _largest_pair_distance_blowout(
    *,
    initial_labels: list[str],
    initial_positions: list[list[float]],
    final_positions: list[list[float]],
) -> dict[str, Any] | None:
    if len(initial_positions) != len(final_positions) or len(initial_labels) != len(initial_positions):
        return None
    best: dict[str, Any] | None = None
    atom_count = len(initial_positions)
    for idx in range(atom_count):
        for jdx in range(idx + 1, atom_count):
            initial_distance = _distance(initial_positions[idx], initial_positions[jdx])
            final_distance = _distance(final_positions[idx], final_positions[jdx])
            if initial_distance <= 0:
                continue
            delta = final_distance - initial_distance
            ratio = final_distance / initial_distance
            if delta <= 0.8 or ratio <= 1.5:
                continue
            candidate = {
                "pair": (idx, jdx),
                "pair_label": f"{initial_labels[idx]}{idx + 1}-{initial_labels[jdx]}{jdx + 1}",
                "initial_distance": initial_distance,
                "final_distance": final_distance,
                "delta": delta,
                "ratio": ratio,
            }
            if best is None or candidate["delta"] > best["delta"]:
                best = candidate
    return best


def _largest_metal_ligand_distance_blowout(
    *,
    initial_labels: list[str],
    initial_positions: list[list[float]],
    final_positions: list[list[float]],
) -> dict[str, Any] | None:
    if len(initial_positions) != len(final_positions) or len(initial_labels) != len(initial_positions):
        return None
    best: dict[str, Any] | None = None
    atom_count = len(initial_positions)
    for idx in range(atom_count):
        label_i = initial_labels[idx]
        if label_i not in _TRANSITION_METALS:
            continue
        for jdx in range(atom_count):
            if idx == jdx:
                continue
            label_j = initial_labels[jdx]
            if label_j in _TRANSITION_METALS:
                continue
            initial_distance = _distance(initial_positions[idx], initial_positions[jdx])
            final_distance = _distance(final_positions[idx], final_positions[jdx])
            if initial_distance <= 0:
                continue
            delta = final_distance - initial_distance
            ratio = final_distance / initial_distance
            if delta <= 0.7 or ratio <= 1.35:
                continue
            fragment_labels = _infer_nonmetal_fragment_labels(
                labels=initial_labels,
                positions=initial_positions,
                seed_index=jdx,
            )
            candidate = {
                "pair": (idx, jdx),
                "pair_label": f"{label_i}{idx + 1}-{label_j}{jdx + 1}",
                "metal_label": f"{label_i}{idx + 1}",
                "ligand_label": f"{label_j}{jdx + 1}",
                "fragment_labels": fragment_labels,
                "fragment_label": _format_fragment_label(fragment_labels),
                "initial_distance": initial_distance,
                "final_distance": final_distance,
                "delta": delta,
                "ratio": ratio,
            }
            if best is None or candidate["delta"] > best["delta"]:
                best = candidate
    return best


def _max_radius_change(initial_positions: list[list[float]], final_positions: list[list[float]]) -> float | None:
    if len(initial_positions) != len(final_positions):
        return None
    max_change = 0.0
    for initial, final in zip(initial_positions, final_positions):
        initial_radius = math.sqrt(sum(float(value) ** 2 for value in initial))
        final_radius = math.sqrt(sum(float(value) ** 2 for value in final))
        max_change = max(max_change, final_radius - initial_radius)
    return max_change


def _infer_nonmetal_fragment_labels(
    *,
    labels: list[str],
    positions: list[list[float]],
    seed_index: int,
) -> list[str]:
    if seed_index >= len(labels) or labels[seed_index] in _TRANSITION_METALS:
        return []
    graph: dict[int, list[int]] = {idx: [] for idx, label in enumerate(labels) if label not in _TRANSITION_METALS}
    for idx in list(graph):
        for jdx in list(graph):
            if jdx <= idx:
                continue
            cutoff = _covalent_cutoff(labels[idx], labels[jdx])
            if cutoff is None:
                continue
            distance = _distance(positions[idx], positions[jdx])
            if distance <= cutoff:
                graph[idx].append(jdx)
                graph[jdx].append(idx)
    seen: set[int] = set()
    stack = [seed_index]
    while stack:
        idx = stack.pop()
        if idx in seen or idx not in graph:
            continue
        seen.add(idx)
        stack.extend(graph[idx])
    return [f"{labels[idx]}{idx + 1}" for idx in sorted(seen)]


def _format_fragment_label(fragment_labels: list[str]) -> str:
    if not fragment_labels:
        return "unknown_fragment"
    if len(fragment_labels) <= 4:
        return ",".join(fragment_labels)
    return ",".join(fragment_labels[:4]) + ",..."


def _covalent_cutoff(label_a: str, label_b: str) -> float | None:
    radius_a = _COVALENT_RADII.get(label_a)
    radius_b = _COVALENT_RADII.get(label_b)
    if radius_a is None or radius_b is None:
        return None
    return radius_a + radius_b + 0.45


def _distance(point_a: list[float], point_b: list[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(point_a, point_b)))


def tail_nwchem_output(path: str, lines: int = 30, max_characters: int = 4000) -> dict[str, Any]:
    return tail_text_file(path, lines=lines, max_characters=max_characters)


def compare_nwchem_runs(
    reference_output_path: str,
    candidate_output_path: str,
    reference_input_path: str | None = None,
    candidate_input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
) -> dict[str, Any]:
    reference = diagnose_nwchem_output(
        output_path=reference_output_path,
        input_path=reference_input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
    )
    candidate = diagnose_nwchem_output(
        output_path=candidate_output_path,
        input_path=candidate_input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
    )

    ref_energy = reference["scf"]["total_energy_hartree"]
    cand_energy = candidate["scf"]["total_energy_hartree"]
    delta_hartree = (cand_energy - ref_energy) if ref_energy is not None and cand_energy is not None else None
    delta_kcal = (delta_hartree * 627.509474) if delta_hartree is not None else None

    changes: list[str] = []
    if reference["task_outcome"] != candidate["task_outcome"]:
        changes.append(f"task_outcome: {reference['task_outcome']} -> {candidate['task_outcome']}")
    if reference["failure_class"] != candidate["failure_class"]:
        changes.append(f"failure_class: {reference['failure_class']} -> {candidate['failure_class']}")
    if reference["state_check"]["assessment"] != candidate["state_check"]["assessment"]:
        changes.append(
            f"state_check: {reference['state_check']['assessment']} -> {candidate['state_check']['assessment']}"
        )
    if delta_hartree is not None:
        changes.append(f"energy_delta_hartree: {delta_hartree:.12f}")
    if reference["scf"]["status"] != candidate["scf"]["status"]:
        changes.append(f"scf_status: {reference['scf']['status']} -> {candidate['scf']['status']}")
    reference_scf_pattern = ((reference["scf"].get("last_run") or {}).get("trend") or {}).get("pattern") or (
        reference["scf"].get("trend") or {}
    ).get("pattern")
    candidate_scf_pattern = ((candidate["scf"].get("last_run") or {}).get("trend") or {}).get("pattern") or (
        candidate["scf"].get("trend") or {}
    ).get("pattern")
    if reference_scf_pattern != candidate_scf_pattern:
        changes.append(f"scf_pattern: {reference_scf_pattern} -> {candidate_scf_pattern}")

    improved_signals: list[str] = []
    regressed_signals: list[str] = []

    if reference["failure_class"] == "wrong_state_convergence" and candidate["failure_class"] != "wrong_state_convergence":
        improved_signals.append("candidate_is_no_longer_flagged_as_wrong_state")
    elif reference["failure_class"] != "wrong_state_convergence" and candidate["failure_class"] == "wrong_state_convergence":
        regressed_signals.append("candidate_is_now_flagged_as_wrong_state")

    if reference["scf"]["status"] == "failed" and candidate["scf"]["status"] != "failed":
        improved_signals.append("candidate_recovers_scf")
    elif reference["scf"]["status"] != "failed" and candidate["scf"]["status"] == "failed":
        regressed_signals.append("candidate_loses_scf_convergence")

    if reference_scf_pattern in {"oscillatory", "stalled", "insufficient_data"} and candidate_scf_pattern in {
        "well_converged",
        "converged",
        "slow_improving",
        "nearly_converged",
    }:
        improved_signals.append("candidate_improves_scf_behavior")
    elif reference_scf_pattern in {"well_converged", "converged"} and candidate_scf_pattern in {
        "oscillatory",
        "stalled",
        "insufficient_data",
    }:
        regressed_signals.append("candidate_worsens_scf_behavior")

    if delta_hartree is not None:
        if delta_hartree < -1e-6:
            improved_signals.append("candidate_is_lower_in_energy")
        elif delta_hartree > 1e-6:
            regressed_signals.append("candidate_is_higher_in_energy")

    ref_imag = ((reference.get("frequency") or {}).get("significant_imaginary_mode_count"))
    cand_imag = ((candidate.get("frequency") or {}).get("significant_imaginary_mode_count"))
    if ref_imag is not None and cand_imag is not None:
        if cand_imag < ref_imag:
            improved_signals.append("candidate_has_fewer_imaginary_modes")
        elif cand_imag > ref_imag:
            regressed_signals.append("candidate_has_more_imaginary_modes")

    if improved_signals and not regressed_signals:
        overall_assessment = "improved"
    elif regressed_signals and not improved_signals:
        overall_assessment = "regressed"
    elif improved_signals or regressed_signals:
        overall_assessment = "mixed"
    else:
        overall_assessment = "no_clear_change"

    return {
        "reference_output": reference_output_path,
        "candidate_output": candidate_output_path,
        "reference_summary": {
            "task_outcome": reference["task_outcome"],
            "failure_class": reference["failure_class"],
            "state_assessment": reference["state_check"]["assessment"],
            "energy_hartree": ref_energy,
            "scf_status": reference["scf"]["status"],
            "scf_pattern": reference_scf_pattern,
        },
        "candidate_summary": {
            "task_outcome": candidate["task_outcome"],
            "failure_class": candidate["failure_class"],
            "state_assessment": candidate["state_check"]["assessment"],
            "energy_hartree": cand_energy,
            "scf_status": candidate["scf"]["status"],
            "scf_pattern": candidate_scf_pattern,
        },
        "energy_delta_hartree": delta_hartree,
        "energy_delta_kcal_mol": delta_kcal,
        "changes": changes,
        "improved_signals": improved_signals,
        "regressed_signals": regressed_signals,
        "overall_assessment": overall_assessment,
    }


def review_nwchem_followup_outcome(
    reference_output_path: str,
    candidate_output_path: str,
    reference_input_path: str | None = None,
    candidate_input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
) -> dict[str, Any]:
    # Lazy import to break circular dependency with api_input
    from .api_input import prepare_nwchem_next_step
    comparison = compare_nwchem_runs(
        reference_output_path=reference_output_path,
        candidate_output_path=candidate_output_path,
        reference_input_path=reference_input_path,
        candidate_input_path=candidate_input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
    )

    candidate_failure = comparison["candidate_summary"]["failure_class"]
    candidate_state = comparison["candidate_summary"]["state_assessment"]
    overall = comparison["overall_assessment"]

    if overall == "improved":
        headline = "Candidate follow-up improved the run."
    elif overall == "regressed":
        headline = "Candidate follow-up regressed relative to the reference run."
    elif overall == "mixed":
        headline = "Candidate follow-up changed the run, but the outcome is mixed."
    else:
        headline = "Candidate follow-up shows no clear overall change."

    details: list[str] = []
    if comparison["candidate_summary"]["scf_status"] == "failed":
        details.append("SCF is still failing")
    elif "candidate_recovers_scf" in comparison["improved_signals"]:
        details.append("SCF now converges")

    if candidate_failure == "wrong_state_convergence":
        details.append("the electronic state still looks suspicious")
    elif candidate_state == "state_consistent_with_expected_metal_open_shell":
        details.append("the electronic state looks more consistent")

    if comparison["candidate_summary"]["scf_pattern"]:
        details.append(f"SCF pattern: {comparison['candidate_summary']['scf_pattern']}")

    headline = headline.rstrip(".")
    if details:
        headline += ": " + "; ".join(details)
    headline += "."

    candidate_next_step = None
    if candidate_failure != "no_clear_failure_detected":
        candidate_next_step = prepare_nwchem_next_step(
            output_path=candidate_output_path,
            input_path=candidate_input_path,
            expected_metal_elements=expected_metal_elements,
            expected_somo_count=expected_somo_count,
            output_dir=output_dir,
            base_name=base_name,
            write_files=False,
            include_property_check=True,
            include_frontier_cubes=False,
        )

    return {
        "reference_output_file": reference_output_path,
        "candidate_output_file": candidate_output_path,
        "comparison_headline": headline,
        "comparison": comparison,
        "candidate_next_step": None
        if candidate_next_step is None
        else {
            "selected_workflow": candidate_next_step["selected_workflow"],
            "can_auto_prepare": candidate_next_step["can_auto_prepare"],
            "artifact_order": candidate_next_step["artifact_order"],
            "prepared_artifact_summaries": candidate_next_step["prepared_artifact_summaries"],
            "notes": candidate_next_step["notes"],
        },
    }


def _mcscf_status_rank(status: str | None) -> int:
    order = {
        "failed": 0,
        "incomplete": 1,
        "converged": 2,
    }
    return order.get((status or "").lower(), -1)


def review_nwchem_mcscf_followup_outcome(
    reference_output_path: str,
    candidate_output_path: str,
    reference_input_path: str | None = None,
    candidate_input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
) -> dict[str, Any]:
    # Lazy import to break circular dependency with api_input
    from .api_input import draft_nwchem_mcscf_retry_input
    _review_nwchem_mcscf_case = _get_review_nwchem_mcscf_case()
    reference = _review_nwchem_mcscf_case(
        output_path=reference_output_path,
        input_path=reference_input_path,
        expected_metal_elements=expected_metal_elements,
    )
    candidate = _review_nwchem_mcscf_case(
        output_path=candidate_output_path,
        input_path=candidate_input_path,
        expected_metal_elements=expected_metal_elements,
    )

    improved_signals: list[str] = []
    regressed_signals: list[str] = []

    reference_status_rank = _mcscf_status_rank(reference["status"])
    candidate_status_rank = _mcscf_status_rank(candidate["status"])
    if candidate_status_rank > reference_status_rank:
        improved_signals.append("candidate_reaches_better_mcscf_status")
    elif candidate_status_rank < reference_status_rank:
        regressed_signals.append("candidate_mcscf_status_worsened")

    ref_energy = reference["raw_mcscf"].get("final_energy_hartree")
    cand_energy = candidate["raw_mcscf"].get("final_energy_hartree")
    energy_delta_hartree = None
    if ref_energy is not None and cand_energy is not None:
        energy_delta_hartree = cand_energy - ref_energy
        if energy_delta_hartree < -1.0e-6:
            improved_signals.append("candidate_has_lower_mcscf_energy")
        elif energy_delta_hartree > 1.0e-6:
            regressed_signals.append("candidate_has_higher_mcscf_energy")

    ref_conv = reference["convergence_review"]
    cand_conv = candidate["convergence_review"]
    if cand_conv["assessment"] == "converged_with_stiff_orbital_optimization" and reference["status"] != "converged":
        improved_signals.append("candidate_converged_with_usable_but_stiff_optimization")
    if reference_status_rank >= 1 and candidate_status_rank >= 1 and reference_status_rank == candidate_status_rank:
        if (
            cand_conv.get("precondition_warning_count", 0) + 10
            < ref_conv.get("precondition_warning_count", 0)
        ):
            improved_signals.append("candidate_has_fewer_precondition_failures")
        elif (
            ref_conv.get("precondition_warning_count", 0) + 10
            < cand_conv.get("precondition_warning_count", 0)
        ):
            regressed_signals.append("candidate_has_more_precondition_failures")

    ref_occ = reference["occupation_review"]["assessment"]
    cand_occ = candidate["occupation_review"]["assessment"]
    if cand_occ == "healthy_active_space" and ref_occ != "healthy_active_space":
        improved_signals.append("candidate_active_space_looks_healthier")
    elif ref_occ == "healthy_active_space" and cand_occ != "healthy_active_space":
        regressed_signals.append("candidate_active_space_looks_less_healthy")

    ref_action = reference["recommended_next_action"]
    cand_action = candidate["recommended_next_action"]
    if cand_action == "use_mcscf_as_reference_or_seed_for_follow_up" and ref_action != cand_action:
        improved_signals.append("candidate_is_now_usable_as_reference")
    elif ref_action == "use_mcscf_as_reference_or_seed_for_follow_up" and cand_action != ref_action:
        regressed_signals.append("candidate_is_no_longer_usable_as_reference")

    if improved_signals and regressed_signals:
        overall_assessment = "mixed"
    elif improved_signals:
        overall_assessment = "improved"
    elif regressed_signals:
        overall_assessment = "regressed"
    else:
        overall_assessment = "no_clear_change"

    if overall_assessment == "improved":
        headline = "Candidate MCSCF follow-up improved the run."
    elif overall_assessment == "regressed":
        headline = "Candidate MCSCF follow-up regressed relative to the reference run."
    elif overall_assessment == "mixed":
        headline = "Candidate MCSCF follow-up changed the run, but the outcome is mixed."
    else:
        headline = "Candidate MCSCF follow-up shows no clear overall change."

    details: list[str] = []
    if candidate["status"] == "converged":
        details.append("MCSCF now converges")
    elif candidate["status"] == "incomplete":
        details.append("MCSCF is still incomplete")
    elif candidate["status"] == "failed":
        details.append("MCSCF still fails")

    if cand_occ == "healthy_active_space":
        details.append("the active space looks healthy")
    elif cand_occ == "borderline_active_space":
        details.append("the active space still has edge orbitals worth reviewing")
    elif cand_occ == "overly_pinned_active_space":
        details.append("the active space still looks too pinned")

    if energy_delta_hartree is not None:
        details.append(f"energy delta: {energy_delta_hartree:.6f} Ha")

    headline = headline.rstrip(".")
    if details:
        headline += ": " + "; ".join(details)
    headline += "."

    candidate_next_step = None
    if candidate_input_path and cand_action != "use_mcscf_as_reference_or_seed_for_follow_up":
        candidate_next_step = draft_nwchem_mcscf_retry_input(
            output_path=candidate_output_path,
            input_path=candidate_input_path,
            expected_metal_elements=expected_metal_elements,
            output_dir=output_dir,
            base_name=base_name,
            write_file=False,
        )

    comparison = {
        "reference_summary": {
            "status": reference["status"],
            "failure_mode": reference["failure_mode"],
            "recommended_next_action": reference["recommended_next_action"],
            "convergence_assessment": reference["convergence_review"]["assessment"],
            "occupation_assessment": reference["occupation_review"]["assessment"],
        },
        "candidate_summary": {
            "status": candidate["status"],
            "failure_mode": candidate["failure_mode"],
            "recommended_next_action": candidate["recommended_next_action"],
            "convergence_assessment": candidate["convergence_review"]["assessment"],
            "occupation_assessment": candidate["occupation_review"]["assessment"],
        },
        "energy_delta_hartree": energy_delta_hartree,
        "improved_signals": improved_signals,
        "regressed_signals": regressed_signals,
        "overall_assessment": overall_assessment,
    }

    return {
        "reference_output_file": reference_output_path,
        "candidate_output_file": candidate_output_path,
        "comparison_headline": headline,
        "comparison": comparison,
        "candidate_next_step": None
        if candidate_next_step is None
        else {
            "retry_strategy": candidate_next_step["retry_strategy"],
            "strategy_notes": candidate_next_step["strategy_notes"],
            "resolved_settings": candidate_next_step["resolved_settings"],
            "file_plan": candidate_next_step["file_plan"],
        },
    }

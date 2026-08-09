"""NWChem-specific progress summary + slow-phase detection.

Used by the NWChem legacy-status adapter and compatibility runner. Originally
lived in ``core/runner.py``; the execution layer now accepts this logic as an
injected output reader.

Public surface:

- ``load_input_summary(input_path, raw_text=None)`` — wrap
  ``inspect_nwchem_input`` and tack ``raw_text`` onto the dict so the
  slow-phase detector can sniff for relativistic Hamiltonian flags.
- ``parse_progress_state(contents, output_path)`` — run ``parse_tasks``
  and return the parsed-output dict the runner stitches into status.
- ``build_progress_summary(contents, parsed_output, *, input_summary)``
  returns phase, status line, and per-task progress.
- ``compact_program_summary(parsed_output, *, progress_summary)`` —
  five-field roll-up used in the ``compact_summary`` slot.
"""

from __future__ import annotations

from typing import Any

from chemtools.programs.nwchem.parse.tasks import parse_tasks
from chemtools.programs.nwchem.parse.freq import parse_freq, parse_trajectory
from chemtools.programs.nwchem.parse.input import inspect_nwchem_input


def load_input_summary(
    input_path: str,
    raw_text: str | None = None,
) -> dict[str, Any] | None:
    """Return the NWChem input summary used by the progress builder.

    Reads the input via ``inspect_nwchem_input`` and attaches the raw
    text under ``raw_text`` so ``_detect_slow_phase`` can sniff for
    relativistic Hamiltonian flags.
    """
    try:
        summary = inspect_nwchem_input(input_path)
    except Exception:  # pragma: no cover
        return None
    if summary is None:
        return None
    summary = dict(summary)
    if raw_text is not None:
        summary["raw_text"] = raw_text
    return summary


def parse_progress_state(contents: str, output_path: str) -> dict[str, Any]:
    """Run ``parse_tasks`` and return the parsed-output dict."""
    return parse_tasks(output_path, contents)


def compact_program_summary(
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


def inspect_legacy_status_output(
    contents: str,
    output_path: str,
    *,
    input_path: str | None = None,
    input_raw_text: str | None = None,
    progress_summary_fn: Any = None,
) -> dict[str, Any]:
    input_summary = (
        load_input_summary(input_path, raw_text=input_raw_text)
        if input_path
        else None
    )
    parsed_output = parse_progress_state(contents, output_path)
    build_progress = progress_summary_fn or build_progress_summary
    progress_summary = build_progress(
        contents,
        parsed_output,
        input_summary=input_summary,
    )
    return {
        "input_summary": input_summary,
        "parsed_output": parsed_output,
        "progress_summary": progress_summary,
        "compact_summary": compact_program_summary(
            parsed_output,
            progress_summary=progress_summary,
        ),
        "task_preview": parsed_output.get("generic_tasks", [])[:5],
    }


def _detect_slow_phase(contents: str, input_summary: dict[str, Any] | None) -> dict[str, Any]:
    """Identify known output-silent phases so the watcher can report
    "expected slow" vs "hung"."""
    tail = contents[-8000:] if len(contents) > 8000 else contents
    lower = tail.lower()

    has_relativistic = False
    if input_summary:
        raw_input = input_summary.get("raw_text") or ""
        has_relativistic = bool(
            "relativistic" in raw_input.lower()
            or "x2c" in raw_input.lower()
            or "dkh" in raw_input.lower()
            or "douglas" in raw_input.lower()
        )
    if not has_relativistic:
        has_relativistic = any(
            kw in contents.lower()
            for kw in ("x2c hamiltonian", "dkh hamiltonian", "relativistic effects",
                       "scalar relativistic", "x2c-mf", "x2c transform")
        )

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

    if any(kw in lower for kw in ("driver: starting", "geometry optimization", "optimize:")):
        return {
            "phase": "geometry_optimization_step",
            "message": "Geometry optimization step in progress.",
        }

    return {"phase": None, "message": ""}


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


def build_progress_summary(
    contents: str,
    parsed_output: dict[str, Any],
    *,
    input_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_tasks = ((parsed_output.get("program_summary") or {}).get("raw") or {}).get("tasks") or []
    last_task = raw_tasks[-1] if raw_tasks else None
    trajectory = parse_trajectory("<status>", contents)
    frequency_started = any(
        marker in contents
        for marker in (
            "NORMAL MODE EIGENVECTORS",
            "NWChem Nuclear Hessian and Frequency Analysis",
            "P.Frequency",
        )
    )
    frequency = parse_freq("<status>", contents) if frequency_started else None

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

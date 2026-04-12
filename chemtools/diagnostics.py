from __future__ import annotations

import re
from typing import Any

from .common import make_metadata, parse_scientific_float, read_text
from . import nwchem
from .nwchem_input import inspect_nwchem_input


SCF_ITER_RE = re.compile(
    r"^\s*d=\s*\d+.*?diis\s+(\d+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.]+)\s*$"
)
PLAIN_SCF_ITER_RE = re.compile(
    r"^\s*(\d+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.]+)\s*$"
)
MAX_ITER_RE = re.compile(r"Maximum number of iterations:\s*([0-9]+)", re.IGNORECASE)
TOTAL_ENERGY_RE = re.compile(r"Total\s+(?:DFT|SCF)\s+energy\s*=\s*([-\d.DEde+]+)")


def parse_scf(path: str) -> dict[str, Any]:
    contents = read_text(path)
    iterations: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    max_iterations: int | None = None
    failure_messages: list[str] = []
    total_energy: float | None = None
    current_run: dict[str, Any] | None = None
    current_byte = 0

    for raw_line in contents.splitlines(keepends=True):
        line = raw_line.rstrip("\n")
        if match := MAX_ITER_RE.search(line):
            max_iterations = int(match.group(1))

        if "Calculation failed to converge" in line:
            failure_messages.append(line.strip())
            if current_run is not None:
                current_run["failure_messages"].append(line.strip())

        if total_energy is None and (match := TOTAL_ENERGY_RE.search(line)):
            total_energy = parse_scientific_float(match.group(1))

        if "Starting SCF solution at" in line or "convergence    iter" in line:
            new_table_type = "dft" if "convergence    iter" in line else "plain"
            if current_run is not None:
                if current_run["iterations"] or current_run["final_energy_hartree"] is not None:
                    _finalize_scf_run(current_run, max_iterations)
                    runs.append(current_run)
                    current_run = None
                else:
                    current_run["table_type"] = new_table_type
            if current_run is None:
                current_run = {
                    "start_byte": current_byte,
                    "table_type": new_table_type,
                    "iterations": [],
                    "failure_messages": [],
                    "final_energy_hartree": None,
                    "max_iterations": max_iterations,
                }

        parsed_iteration = _parse_scf_iteration_line(line)
        if parsed_iteration is not None:
            iterations.append(parsed_iteration)
            if current_run is None:
                current_run = {
                    "start_byte": current_byte,
                    "table_type": parsed_iteration["table_type"],
                    "iterations": [],
                    "failure_messages": [],
                    "final_energy_hartree": None,
                    "max_iterations": max_iterations,
                }
            current_run["iterations"].append(parsed_iteration)

        if current_run is not None and (match := TOTAL_ENERGY_RE.search(line)):
            current_run["final_energy_hartree"] = parse_scientific_float(match.group(1))
            _finalize_scf_run(current_run, max_iterations, end_byte=current_byte + len(raw_line))
            runs.append(current_run)
            current_run = None

        current_byte += len(raw_line)

    if current_run is not None:
        _finalize_scf_run(current_run, max_iterations, end_byte=current_byte)
        runs.append(current_run)

    status = "unknown"
    if failure_messages:
        status = "failed"
    elif runs:
        status = "converged"

    hit_max_iterations = bool(
        status == "failed"
        and max_iterations is not None
        and iterations
        and iterations[-1]["iteration"] >= max_iterations
    )

    recent = iterations[-8:]
    delta_sign_changes = _count_sign_changes(
        [
            entry["delta_e_hartree"]
            for entry in recent
            if entry["delta_e_hartree"] is not None and abs(entry["delta_e_hartree"]) > 1e-8
        ]
    )
    density_ratio = _density_ratio(recent)
    pattern = _classify_scf_pattern(status, recent, density_ratio, delta_sign_changes)

    iteration_increase_reasonable = bool(
        status == "failed" and hit_max_iterations and pattern in {"slow_improving", "nearly_converged"}
    )

    return {
        "metadata": make_metadata(path, contents, "nwchem"),
        "status": status,
        "max_iterations": max_iterations,
        "iteration_count": len(iterations),
        "run_count": len(runs),
        "runs": runs,
        "last_run": runs[-1] if runs else None,
        "hit_max_iterations": hit_max_iterations,
        "total_energy_hartree": total_energy,
        "failure_messages": failure_messages,
        "recent_iterations": recent,
        "trend": {
            "pattern": pattern,
            "density_ratio_recent": density_ratio,
            "delta_e_sign_changes_recent": delta_sign_changes,
            "iteration_increase_reasonable": iteration_increase_reasonable,
        },
    }


def _analyze_err_file(err_path: str) -> dict[str, Any]:
    """Parse a NWChem .err file and classify the crash type.

    Returns a dict with:
    - available: bool
    - err_type: str | None  (classification of the root cause)
    - err_message: str | None  (human-readable explanation)
    - signal: int | None  (dominant kill signal, if any)
    - killed_signals: list[int]
    - mpi_abort_count: int
    - raw_error_lines: list[str]  (first N non-noise lines from the file)
    """
    # Noise patterns that appear in err files but aren't NWChem errors
    _NOISE = [
        "gocryptfs", "singularity", "apptainer",
        "INFO:", "WARNING: setrlimit", "slurm_script",
    ]
    result: dict[str, Any] = {
        "err_file": err_path,
        "available": False,
        "err_type": None,
        "err_message": None,
        "signal": None,
        "killed_signals": [],
        "mpi_abort_count": 0,
        "raw_error_lines": [],
    }
    try:
        content = read_text(err_path)
    except Exception:
        return result

    result["available"] = True
    lower = content.lower()
    lines = content.splitlines()

    # Non-noise lines
    error_lines = [
        l.strip() for l in lines
        if l.strip() and not any(n.lower() in l for n in _NOISE)
    ]
    result["raw_error_lines"] = error_lines[:40]

    # Kill signals from BAD TERMINATION blocks in the output (also appear here sometimes)
    signals = [int(m) for m in re.findall(r"KILLED BY SIGNAL:\s*(\d+)", content)]
    result["killed_signals"] = sorted(set(signals))
    result["signal"] = signals[0] if signals else None
    result["mpi_abort_count"] = len(re.findall(r"application called MPI_Abort", content, re.IGNORECASE))

    # Classification — priority order
    if re.search(r"MA_alloc_failed|ma_alloc_noc_|ga_create.*[Ee]rror|Error.*ga_create", content):
        result["err_type"] = "memory_allocation_failed"
        result["err_message"] = (
            "NWChem MA/Global Arrays memory allocation failed. "
            "Increase the memory directive in the input (e.g. 'memory total 8 gb'), "
            "or request more RAM from the scheduler. "
            "Also check that memory * nproc doesn't exceed the node's physical RAM."
        )
    elif "out of memory" in lower or "cannot allocate memory" in lower:
        result["err_type"] = "out_of_memory"
        result["err_message"] = (
            "System out-of-memory error. Request more RAM, reduce basis size, or reduce parallelism."
        )
    elif 9 in signals and "dimensions not the same" not in lower:
        # SIGKILL with no dimension-mismatch explanation → OOM killer
        result["err_type"] = "oom_killed"
        result["err_message"] = (
            "Process killed by SIGKILL (signal 9). "
            "This is usually the OS OOM killer or a scheduler memory limit. "
            "Request more memory or reduce basis/parallelism."
        )
    elif 15 in signals:
        result["err_type"] = "walltime_killed"
        result["err_message"] = (
            "Process killed by SIGTERM (signal 15). "
            "This is typically a scheduler walltime limit. "
            "Request more walltime, or split the job (e.g. opt then freq separately)."
        )
    elif "dimensions not the same" in lower:
        result["err_type"] = "dimension_mismatch"
        result["err_message"] = (
            "'dimensions not the same' — internal NWChem array size error propagated through MPI. "
            "This is NOT a network/communication failure. "
            "Common causes: (1) SP-contracted Pople basis (6-31G*, 6-311G**) + X2C/DKH relativistic — "
            "incompatible; use cc-pVDZ/cc-pVTZ or def2-SVP/def2-TZVP instead. "
            "(2) Global Arrays memory mismatch when using unusual parallelism settings."
        )
    elif "segmentation fault" in lower or 11 in signals or "sigsegv" in lower:
        result["err_type"] = "segfault"
        result["err_message"] = (
            "Segmentation fault (SIGSEGV / signal 11). "
            "Possible causes: NWChem bug triggered by unusual input, corrupted restart files, "
            "or stack/memory overflow from a very large calculation."
        )
    elif 6 in signals or "sigabrt" in lower:
        result["err_type"] = "aborted"
        result["err_message"] = (
            "Process aborted (SIGABRT / signal 6). "
            "NWChem called abort() after detecting an internal error. "
            "Check the .out file for the NWChem-level error message printed just before the crash."
        )
    elif result["mpi_abort_count"] > 0:
        # MPI_Abort called but no clear root cause visible in err file alone
        first_hint = next(
            (l for l in error_lines
             if "received an error" not in l.lower()
             and "mpi_abort" not in l.lower()
             and "abort(" not in l.lower()),
            None,
        )
        result["err_type"] = "mpi_abort_unknown_cause"
        result["err_message"] = (
            "NWChem called MPI_Abort — all ranks terminated. "
            "This is a symptom; the root cause is in the .out file or the first rank's message. "
            + (f"First non-MPI error line: '{first_hint}'. " if first_hint else "")
            + "Check the .out file for a NWChem-level error printed before the crash."
        )

    return result


def _auto_err_path(output_path: str) -> str | None:
    """Derive the expected .err file path from a .out path."""
    import os
    base = re.sub(r"\.(out|log|nwout)$", "", output_path, flags=re.IGNORECASE)
    for ext in (".err", ".error", ".stderr"):
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate
    return None


def diagnose_nwchem_output(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    err_file: str | None = None,
    _contents: str | None = None,
) -> dict[str, Any]:
    contents = _contents if _contents is not None else read_text(output_path)
    tasks = nwchem.parse_tasks(output_path, contents)
    scf = parse_scf(output_path)
    mos = nwchem.parse_mos(output_path, contents, top_n=8)
    population = nwchem.parse_population_analysis(output_path, contents)
    freq = nwchem.parse_freq(output_path, contents) if _looks_like_frequency_run(tasks) else None
    trajectory = nwchem.parse_trajectory(output_path, contents) if _looks_like_optimization_run(tasks) else None

    # Analyze the .err file (auto-detect if not provided)
    _err_path = err_file or _auto_err_path(output_path)
    err_analysis = _analyze_err_file(_err_path) if _err_path else {"available": False}

    input_summary = inspect_nwchem_input(input_path) if input_path else None
    metals = expected_metal_elements or (input_summary["transition_metals"] if input_summary else [])
    somo_target = expected_somo_count
    if somo_target is None and input_summary and input_summary["multiplicity"] and input_summary["multiplicity"] > 1:
        somo_target = input_summary["multiplicity"] - 1

    state_check = analyze_frontier_orbitals(
        mos,
        population_payload=population,
        expected_metal_elements=metals,
        expected_somo_count=somo_target,
    )
    swap_suggestion = suggest_vectors_swaps(
        mos,
        expected_metal_elements=metals,
        expected_somo_count=somo_target,
    )

    stage = _last_task_kind(tasks)
    failure_class = "unknown"
    likely_cause = "insufficient_evidence"
    next_action = "inspect_raw_output"
    confidence = "low"

    # --- Early-exit: X2C/DKH + SP-shell basis crash ---
    # NWChem builds an uncontracted auxiliary basis for the relativistic one-electron
    # operator. Pople-style SP shells cause a dimension mismatch that kills the job
    # before SCF even starts. The stderr shows "dimensions not the same" / MPI_Abort,
    # which looks like a communication error but is actually a basis incompatibility.
    _x2c_sp_crash = False
    try:
        import re as _rd
        _lower = contents.lower()
        # X2C/DKH uncontraction phase was running when the job died
        _in_x2c_uncontract = (
            "x2c_1e_scalar" in _lower or "calc_x2c_1e_scalar" in _lower
            or "uncontracted auxiliary basis" in _lower
        )
        # Job terminated abnormally
        _bad_term = "bad termination" in _lower or "killed by signal" in _lower
        # Input has SP shells
        _has_sp = bool(_rd.search(r"^\s*[A-Za-z][a-z]?\s+SP\s*$", contents, _rd.MULTILINE))
        # Input has a relativistic block
        _has_rel_block = bool(_rd.search(r"^\s*relativistic\b", contents, _rd.IGNORECASE | _rd.MULTILINE))
        # Also check input file if provided
        if not _has_sp and input_path:
            from .common import read_text as _rt_d
            _ic = _rt_d(input_path)
            _has_sp = bool(_rd.search(r"^\s*[A-Za-z][a-z]?\s+SP\s*$", _ic, _rd.MULTILINE))
            _has_rel_block = _has_rel_block or bool(
                _rd.search(r"^\s*relativistic\b", _ic, _rd.IGNORECASE | _rd.MULTILINE)
            )
        if (_in_x2c_uncontract or _has_rel_block) and _bad_term and _has_sp:
            _x2c_sp_crash = True
    except Exception:
        pass

    if _x2c_sp_crash:
        failure_class = "basis_incompatibility"
        likely_cause = "sp_shells_incompatible_with_x2c_dkh"
        next_action = (
            "replace_pople_basis_with_dunning_or_def2: "
            "6-31G* / 6-311G** use SP-contracted shells (shared S+P exponents). "
            "NWChem X2C/DKH must uncontract the basis to build the relativistic one-electron "
            "operator; SP shells cause a dimension mismatch and abort (erroneously logged as "
            "'dimensions not the same' / MPI_Abort). "
            "Use cc-pVDZ/cc-pVTZ or def2-SVP/def2-TZVP — both use separate S and P shells "
            "and are fully compatible with X2C and DKH."
        )
        confidence = "high"
    elif scf["status"] == "failed":
        failure_class = "scf_nonconvergence"
        confidence = "high"
        if scf["trend"]["iteration_increase_reasonable"]:
            likely_cause = "slow_but_improving_scf"
            next_action = "modest_iteration_increase_or_restart_from_last_vectors"
        elif scf["trend"]["pattern"] in {"oscillatory", "stalled"}:
            likely_cause = "scf_strategy_problem"
            next_action = "change_guess_state_or_convergence_strategy"
        else:
            likely_cause = "scf_failed_before_convergence"
            next_action = "inspect_guess_state_and_recent_iterations"
    elif state_check["assessment"] == "metal_state_mismatch_suspected":
        failure_class = "wrong_state_convergence"
        likely_cause = "singly_occupied_orbitals_do_not_match_expected_metal_state"
        next_action = "inspect_somos_then_try_fragment_guess_or_vectors_swap"
        confidence = "medium"
    elif (
        stage == "frequency"
        and tasks["program_summary"]["outcome"] == "incomplete"
        and trajectory is not None
        and trajectory["optimization_status"] == "converged"
    ):
        failure_class = "post_optimization_frequency_interrupted"
        likely_cause = "frequency_stage_interrupted_after_completed_optimization"
        next_action = "restart_frequency_from_last_geometry_without_reoptimizing"
        confidence = "high"
    elif (
        stage == "frequency"
        and freq is not None
        and freq["significant_imaginary_mode_count"] > 0
    ):
        failure_class = "frequency_interpretation_required"
        likely_cause = "imaginary_modes_present"
        next_action = "inspect_imaginary_modes_and_displace_if_needed"
        confidence = "medium"
    elif stage == "optimization" and trajectory is not None and trajectory["frame_count"] > 0:
        failure_class = "optimization_review"
        if trajectory["optimization_status"] == "converged":
            likely_cause = "optimization_completed_without_obvious_electronic_failure"
            next_action = "inspect_final_geometry_and_follow_with_frequency_if_needed"
            confidence = "medium"
        else:
            likely_cause = "optimization_stopped_before_convergence"
            next_action = "restart_from_last_geometry_and_inspect_recent_steps"
            confidence = "medium"
    elif tasks["program_summary"]["outcome"] == "success":
        failure_class = "no_clear_failure_detected"
        likely_cause = "run_completed_normally"
        next_action = "verify_state_quality_before_accepting_result"
        confidence = "medium"

    # --- Err-file override: if output analysis is still "unknown", let err_analysis clarify ---
    _ERR_TYPE_TO_CLASS = {
        "memory_allocation_failed": ("out_of_memory", "nwchem_memory_allocation_failed"),
        "out_of_memory": ("out_of_memory", "system_out_of_memory"),
        "oom_killed": ("oom_killed", "scheduler_or_os_killed_process_sigkill"),
        "walltime_killed": ("walltime_exceeded", "scheduler_killed_process_sigterm"),
        "dimension_mismatch": ("internal_error", "dimension_mismatch_check_basis_and_parallelism"),
        "segfault": ("crash_segfault", "segmentation_fault_check_input_and_restart_files"),
        "aborted": ("crash_aborted", "internal_abort_check_output_for_nwchem_error"),
        "mpi_abort_unknown_cause": ("crash_mpi_abort", "mpi_abort_root_cause_unknown_check_output"),
    }
    err_type = err_analysis.get("err_type")
    if failure_class == "unknown" and err_type and err_type in _ERR_TYPE_TO_CLASS:
        failure_class, likely_cause = _ERR_TYPE_TO_CLASS[err_type]
        next_action = err_analysis.get("err_message") or next_action
        confidence = "medium"

    return {
        "metadata": make_metadata(output_path, contents, "nwchem"),
        "input_summary": input_summary,
        "tasks": tasks,
        "stage": stage,
        "task_outcome": tasks["program_summary"]["outcome"],
        "failure_class": failure_class,
        "likely_cause": likely_cause,
        "recommended_next_action": next_action,
        "confidence": confidence,
        "scf": scf,
        "state_check": state_check,
        "swap_suggestion": swap_suggestion,
        "population_analysis": population,
        "frequency": freq,
        "trajectory": trajectory,
        "err_analysis": err_analysis,
    }


def summarize_nwchem_output(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    detail_level: str = "summary",
    err_file: str | None = None,
    _contents: str | None = None,
) -> dict[str, Any]:
    diagnosis = diagnose_nwchem_output(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
        err_file=err_file,
        _contents=_contents,
    )

    scf = diagnosis["scf"]
    frequency = diagnosis["frequency"]
    state_check = diagnosis["state_check"]
    task_summaries = _build_task_summaries(
        diagnosis["tasks"],
        scf,
        frequency_payload=frequency,
        trajectory_payload=diagnosis["trajectory"],
    )

    bullets: list[str] = []
    bullets.append(f"Stage: {diagnosis['stage'] or 'unknown'}")
    bullets.append(f"Outcome: {diagnosis['task_outcome']}")
    bullets.append(f"Diagnosis: {diagnosis['failure_class']}")

    if scf["total_energy_hartree"] is not None:
        bullets.append(f"Energy: {scf['total_energy_hartree']:.12f} Ha")

    if scf["last_run"] is not None:
        last_run = scf["last_run"]
        if scf.get("run_count", 0) > 1:
            bullets.append(
                f"SCF runs: {scf['run_count']} total; last {last_run['status']} in {last_run['iteration_count']} iterations"
                + (f" ({last_run['trend']['pattern']})" if last_run["trend"]["pattern"] else "")
            )
        else:
            bullets.append(
                f"SCF: {last_run['status']} in {last_run['iteration_count']} iterations"
                + (f" ({last_run['trend']['pattern']})" if last_run["trend"]["pattern"] else "")
            )
    elif scf["iteration_count"] > 0:
        trend = scf["trend"]["pattern"]
        bullets.append(
            f"SCF: {scf['status']} after {scf['iteration_count']} iterations"
            + (f" ({trend})" if trend else "")
        )
    elif scf["status"] != "unknown":
        bullets.append(f"SCF: {scf['status']}")

    if frequency is not None:
        freq_text = (
            f"Frequencies: {frequency['mode_count']} {frequency['preferred_kind']} modes, "
            f"{frequency['significant_imaginary_mode_count']} significant imaginary, "
            f"{frequency['near_zero_mode_count']} near-zero"
        )
        if frequency["lowest_vibrational_frequencies_cm1"]:
            preview = ", ".join(
                f"{value:.2f}" for value in frequency["lowest_vibrational_frequencies_cm1"][:4]
            )
            freq_text += f", leading vibrational {preview} cm^-1"
        if frequency["significant_imaginary_frequencies_cm1"]:
            imag_preview = ", ".join(
                f"{value:.2f}" for value in frequency["significant_imaginary_frequencies_cm1"][:3]
            )
            freq_text += f", imaginary {imag_preview} cm^-1"
        bullets.append(freq_text)
        thermo = frequency.get("thermochemistry")
        if thermo is not None:
            thermo_text = f"Thermochemistry: {thermo['temperature_kelvin']:.2f} K"
            zero_point = thermo.get("zero_point_correction")
            if zero_point and zero_point.get("hartree") is not None:
                thermo_text += f", ZPE {zero_point['hartree']:.6f} Ha"
            bullets.append(thermo_text)

    trajectory = diagnosis.get("trajectory")
    if trajectory is not None and trajectory["frame_count"] > 0:
        opt_text = (
            f"Optimization: {trajectory['optimization_status']}, "
            f"{trajectory['step_count']} steps"
        )
        if trajectory["final_energy_hartree"] is not None:
            opt_text += f", final energy {trajectory['final_energy_hartree']:.12f} Ha"
        final_metrics = trajectory.get("final_metrics")
        criteria_met = trajectory.get("criteria_met")
        if final_metrics is not None:
            parts: list[str] = []
            if final_metrics.get("gmax") is not None and trajectory["thresholds"].get("gmax") is not None:
                parts.append(
                    f"gmax {final_metrics['gmax']:.2e}/{trajectory['thresholds']['gmax']:.2e}"
                )
            if final_metrics.get("grms") is not None and trajectory["thresholds"].get("grms") is not None:
                parts.append(
                    f"grms {final_metrics['grms']:.2e}/{trajectory['thresholds']['grms']:.2e}"
                )
            if parts:
                opt_text += ", " + ", ".join(parts)
        if criteria_met is not None:
            if criteria_met.get("all_met") is True:
                opt_text += ", all criteria met"
            elif trajectory.get("unmet_criteria"):
                opt_text += f", unmet {', '.join(trajectory['unmet_criteria'])}"
        bullets.append(opt_text)

    if state_check["available"]:
        bullets.append(
            f"SOMOs: {state_check['somo_count']} found, metal-like {state_check['metal_like_somo_count']}"
        )
    elif state_check["assessment"] != "unavailable":
        bullets.append(f"State check: {state_check['assessment']}")
    if state_check.get("spin_density_summary", {}).get("available"):
        spin_summary = state_check["spin_density_summary"]
        if spin_summary.get("dominant_site"):
            dominant = spin_summary["dominant_site"]
            bullets.append(
                f"Spin density: dominant on {dominant['atom_index']} {dominant['element']} ({dominant['population']:.2f})"
            )
    swap = diagnosis.get("swap_suggestion")
    if swap and swap.get("available") and swap.get("swap_pairs"):
        bullets.append(
            "Suggested swaps: "
            + ", ".join(
                f"{pair['spin']} {pair['from_vector']}->{pair['to_vector']}"
                for pair in swap["swap_pairs"][:4]
            )
        )

    bullets.extend(task_summaries)
    bullets.append(f"Next action: {diagnosis['recommended_next_action']}")

    summary_text = "\n".join(f"- {bullet}" for bullet in bullets)

    if detail_level == "full":
        return {
            "metadata": diagnosis["metadata"],
            "failure_class": diagnosis["failure_class"],
            "likely_cause": diagnosis["likely_cause"],
            "recommended_next_action": diagnosis["recommended_next_action"],
            "confidence": diagnosis["confidence"],
            "summary_bullets": bullets,
            "summary_text": summary_text,
            "diagnosis": diagnosis,
        }

    # summary level: return compact view, omit heavy per-task data
    scf_for_summary = {
        "status": scf.get("status"),
        "total_energy_hartree": scf.get("total_energy_hartree"),
        "iteration_count": scf.get("iteration_count"),
        "trend_pattern": (scf.get("last_run") or scf.get("trend") or {}).get("pattern"),
    }
    freq_for_summary = None
    if frequency is not None:
        freq_for_summary = {
            "mode_count": frequency.get("mode_count"),
            "significant_imaginary_mode_count": frequency.get("significant_imaginary_mode_count"),
            "near_zero_mode_count": frequency.get("near_zero_mode_count"),
            "lowest_vibrational_frequencies_cm1": (frequency.get("lowest_vibrational_frequencies_cm1") or [])[:4],
            "significant_imaginary_frequencies_cm1": frequency.get("significant_imaginary_frequencies_cm1"),
            "zpe_hartree": ((frequency.get("thermochemistry") or {}).get("zero_point_correction") or {}).get("hartree"),
        }
    spin_for_summary = {
        "somo_count": state_check.get("somo_count"),
        "metal_like_somo_count": state_check.get("metal_like_somo_count"),
        "assessment": state_check.get("assessment"),
        "dominant_spin_atom": (state_check.get("spin_density_summary") or {}).get("dominant_site"),
    }
    return {
        "metadata": diagnosis["metadata"],
        "outcome": diagnosis["task_outcome"],
        "failure_class": diagnosis["failure_class"],
        "likely_cause": diagnosis["likely_cause"],
        "recommended_next_action": diagnosis["recommended_next_action"],
        "confidence": diagnosis["confidence"],
        "energy_hartree": scf.get("total_energy_hartree"),
        "scf": scf_for_summary,
        "optimization_status": (diagnosis.get("trajectory") or {}).get("optimization_status"),
        "optimization_step_count": (diagnosis.get("trajectory") or {}).get("step_count"),
        "frequency": freq_for_summary,
        "spin": spin_for_summary,
        "summary_bullets": bullets,
        "summary_text": summary_text,
        "diagnosis": {
            "stage": diagnosis["stage"],
            "task_outcome": diagnosis["task_outcome"],
            "failure_class": diagnosis["failure_class"],
            "likely_cause": diagnosis["likely_cause"],
            "recommended_next_action": diagnosis["recommended_next_action"],
            "confidence": diagnosis["confidence"],
        },
        "err_analysis": diagnosis.get("err_analysis"),
    }


def _parse_scf_iteration_line(line: str) -> dict[str, Any] | None:
    if match := SCF_ITER_RE.match(line):
        return {
            "table_type": "dft",
            "iteration": int(match.group(1)),
            "energy_hartree": parse_scientific_float(match.group(2)),
            "delta_e_hartree": parse_scientific_float(match.group(3)),
            "rms_density": parse_scientific_float(match.group(4)),
            "diis_error": parse_scientific_float(match.group(5)),
            "gnorm": None,
            "gmax": None,
            "time_seconds": parse_scientific_float(match.group(6)),
        }
    if match := PLAIN_SCF_ITER_RE.match(line):
        return {
            "table_type": "plain",
            "iteration": int(match.group(1)),
            "energy_hartree": parse_scientific_float(match.group(2)),
            "delta_e_hartree": None,
            "rms_density": None,
            "diis_error": None,
            "gnorm": parse_scientific_float(match.group(3)),
            "gmax": parse_scientific_float(match.group(4)),
            "time_seconds": parse_scientific_float(match.group(5)),
        }
    return None


def _finalize_scf_run(run: dict[str, Any], max_iterations: int | None, end_byte: int | None = None) -> None:
    iterations = run["iterations"]
    recent = iterations[-8:]
    density_ratio = _density_ratio(recent)
    delta_sign_changes = _count_sign_changes(
        [
            entry["delta_e_hartree"]
            for entry in recent
            if entry.get("delta_e_hartree") is not None and abs(entry["delta_e_hartree"]) > 1e-8
        ]
    )

    if run["table_type"] == "plain":
        pattern = _classify_plain_scf_pattern(recent)
    else:
        pattern = _classify_scf_pattern(
            "failed" if run["failure_messages"] else "converged",
            recent,
            density_ratio,
            delta_sign_changes,
        )

    iteration_count = len(iterations)
    hit_max = bool(
        max_iterations is not None and iterations and iterations[-1]["iteration"] >= max_iterations
    )
    status = "failed" if run["failure_messages"] else ("converged" if iterations else "unknown")

    run["max_iterations"] = max_iterations
    run["end_byte"] = end_byte if end_byte is not None else run["start_byte"]
    run["iteration_count"] = iteration_count
    run["status"] = status
    run["hit_max_iterations"] = hit_max and status == "failed"
    run["recent_iterations"] = recent
    run["trend"] = {
        "pattern": pattern,
        "density_ratio_recent": density_ratio,
        "delta_e_sign_changes_recent": delta_sign_changes,
    }
    if run["final_energy_hartree"] is None and iterations:
        run["final_energy_hartree"] = iterations[-1]["energy_hartree"]


def _classify_plain_scf_pattern(recent: list[dict[str, Any]]) -> str:
    if not recent:
        return "no_iterations_found"
    last = recent[-1]
    gnorm = last.get("gnorm")
    gmax = last.get("gmax")
    if gnorm is None or gmax is None:
        return "converged"
    if gnorm < 1e-5 and gmax < 1e-5:
        return "well_converged"
    if gnorm < 1e-3 and gmax < 1e-3:
        return "converged"
    return "not_converged"


def _build_task_summaries(
    tasks_payload: dict[str, Any],
    scf_payload: dict[str, Any],
    frequency_payload: dict[str, Any] | None = None,
    trajectory_payload: dict[str, Any] | None = None,
) -> list[str]:
    raw_tasks = tasks_payload["program_summary"]["raw"]["tasks"]
    runs = scf_payload.get("runs", [])
    bullets: list[str] = []

    for task in raw_tasks:
        task_runs = [
            run
            for run in runs
            if task["boundary"]["start_byte"] <= run["start_byte"] <= task["boundary"]["end_byte"]
        ]
        label = task["label"]
        kind = task["kind"]

        if kind == "optimization":
            step_count = task["frame_count"] or 0
            status = task["outcome"]
            if trajectory_payload is not None and trajectory_payload["frame_count"] > 0:
                step_count = trajectory_payload["step_count"] or step_count
                status = trajectory_payload["optimization_status"]
            text = f"Task {label}: {status}, {step_count} optimization steps"
            final_energy = task["total_energy_hartree"]
            if trajectory_payload is not None and trajectory_payload["final_energy_hartree"] is not None:
                final_energy = trajectory_payload["final_energy_hartree"]
            if final_energy is not None:
                text += f", final energy {final_energy:.12f} Ha"
            if task_runs:
                text += f", last SCF {task_runs[-1]['status']} in {task_runs[-1]['iteration_count']} iterations"
            if trajectory_payload is not None and trajectory_payload.get("criteria_met") is not None:
                if trajectory_payload["criteria_met"].get("all_met") is True:
                    text += ", all optimization criteria met"
                elif trajectory_payload.get("unmet_criteria"):
                    text += f", unmet {', '.join(trajectory_payload['unmet_criteria'])}"
            bullets.append(text)
            continue

        if kind in {"frequency", "raman"}:
            modes = task.get("frequency_modes", []) or []
            if frequency_payload is not None and frequency_payload["modes"]:
                modes = frequency_payload["modes"]
                mode_count = frequency_payload["mode_count"]
                imaginary_count = frequency_payload["significant_imaginary_mode_count"]
                near_zero_count = frequency_payload["near_zero_mode_count"]
                nonzero = frequency_payload["lowest_vibrational_frequencies_cm1"]
                preferred_kind = frequency_payload["preferred_kind"]
                text = (
                    f"Task {label}: {mode_count} {preferred_kind} modes, "
                    f"{imaginary_count} significant imaginary, {near_zero_count} near-zero"
                )
            else:
                nonzero = [mode["frequency_cm1"] for mode in modes if abs(mode["frequency_cm1"]) > 1e-3]
                imaginary = [mode["frequency_cm1"] for mode in modes if mode["frequency_cm1"] < 0]
                text = (
                    f"Task {label}: {task.get('mode_count') or len(modes)} modes, {len(imaginary)} imaginary"
                )
            if task_runs:
                text += f", {len(task_runs)} SCF solves, last in {task_runs[-1]['iteration_count']} iterations"
            if nonzero:
                preview = ", ".join(f"{value:.2f}" for value in nonzero[:4])
                text += f", leading frequencies {preview} cm^-1"
            if frequency_payload is not None and frequency_payload.get("thermochemistry") is not None:
                thermo = frequency_payload["thermochemistry"]
                if thermo.get("temperature_kelvin") is not None:
                    text += f", thermo at {thermo['temperature_kelvin']:.2f} K"
            bullets.append(text)
            continue

        if kind in {"single_point", "property"}:
            text = f"Task {label}: {task['outcome']}"
            if task_runs:
                text += f", converged in {task_runs[-1]['iteration_count']} iterations"
            if task["total_energy_hartree"] is not None:
                text += f", energy {task['total_energy_hartree']:.12f} Ha"
            bullets.append(text)
            continue

        bullets.append(f"Task {label}: {task['outcome']}")

    return bullets


def analyze_frontier_orbitals(
    mos_payload: dict[str, Any],
    population_payload: dict[str, Any] | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
) -> dict[str, Any]:
    metal_set = {element.lower() for element in expected_metal_elements or []}
    spin_channels = mos_payload.get("spin_channels", {})
    frontier_channels = {
        channel: _summarize_frontier_channel(channel_payload, metal_set)
        for channel, channel_payload in spin_channels.items()
    }
    available = bool(frontier_channels)
    somos = []
    for channel_payload in frontier_channels.values():
        somos.extend(channel_payload["somos"])

    metal_like_count = sum(1 for somo in somos if somo["metal_like"])
    ligand_like_count = sum(1 for somo in somos if somo["ligand_like"])
    d_like_count = sum(1 for somo in somos if somo["d_like"])
    f_like_count = sum(1 for somo in somos if somo["f_like"])

    assessment = "not_flagged"
    if not available:
        assessment = "unavailable"
    elif expected_somo_count is not None and len(somos) != expected_somo_count:
        assessment = "somo_count_mismatch"
    elif metal_set and somos:
        target = expected_somo_count if expected_somo_count is not None else len(somos)
        minimum_expected_metal = max(1, target // 2)
        if metal_like_count < minimum_expected_metal:
            assessment = "metal_state_mismatch_suspected"
        else:
            assessment = "state_consistent_with_expected_metal_open_shell"

    spin_density_summary = _summarize_spin_density(population_payload, metal_set)
    if (
        assessment == "state_consistent_with_expected_metal_open_shell"
        and spin_density_summary["available"]
        and metal_set
        and spin_density_summary["metal_population_sum"] < 0.35 * max(1, len(somos))
    ):
        assessment = "metal_state_mismatch_suspected"

    return {
        "available": available,
        "assessment": assessment,
        "reason": None if available else "frontier_orbitals_not_found",
        "frontier_channels": frontier_channels,
        "somos": somos,
        "somo_count": len(somos),
        "expected_somo_count": expected_somo_count,
        "metal_like_somo_count": metal_like_count,
        "ligand_like_somo_count": ligand_like_count,
        "d_like_somo_count": d_like_count,
        "f_like_somo_count": f_like_count,
        "spin_density_summary": spin_density_summary,
    }


def analyze_somos(
    mos_payload: dict[str, Any],
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
) -> dict[str, Any]:
    return analyze_frontier_orbitals(
        mos_payload,
        population_payload=None,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
    )


def suggest_vectors_swaps(
    mos_payload: dict[str, Any],
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
) -> dict[str, Any]:
    metal_set = {element.lower() for element in expected_metal_elements or []}
    frontier = analyze_frontier_orbitals(
        mos_payload,
        population_payload=None,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
    )
    if not frontier["available"] or not frontier["frontier_channels"] or not metal_set:
        return {
            "available": False,
            "reason": "frontier_analysis_or_expected_metals_unavailable",
            "swap_pairs": [],
            "vectors_block": None,
        }

    primary_spin, primary_channel = max(
        frontier["frontier_channels"].items(),
        key=lambda item: item[1].get("somo_count", 0),
    )
    somos = primary_channel.get("somos", [])
    if not somos:
        return {
            "available": False,
            "reason": "no_somos_in_primary_spin_channel",
            "swap_pairs": [],
            "vectors_block": None,
        }

    wrong_slots = [
        orbital
        for orbital in somos
        if orbital["character_class"].startswith("ligand_centered") or not orbital["metal_like"]
    ]
    wrong_slots.sort(key=lambda orbital: orbital["vector_number"])
    if not wrong_slots:
        return {
            "available": False,
            "reason": "no_ligand_like_somo_slots_detected",
            "swap_pairs": [],
            "vectors_block": None,
        }

    min_somo_vector = min(orbital["vector_number"] for orbital in somos)
    occupied_orbitals = [
        orbital
        for orbital in mos_payload["orbitals"]
        if orbital.get("spin") == primary_spin
        and orbital["occupation_label"] == "occupied"
        and orbital["vector_number"] < min_somo_vector
    ]
    candidate_pool = []
    for orbital in occupied_orbitals:
        summarized = _summarize_frontier_orbital(orbital, metal_set)
        if summarized is None:
            continue
        if summarized["metal_like"] and (
            summarized["d_like"]
            or summarized["f_like"]
            or summarized["character_class"].startswith("metal_centered")
        ):
            candidate_pool.append(summarized)
    candidate_pool.sort(key=lambda orbital: orbital["energy_hartree"], reverse=True)

    swap_pairs = []
    used_vectors: set[int] = set()
    for wrong_slot, candidate in zip(wrong_slots, candidate_pool):
        if candidate["vector_number"] in used_vectors:
            continue
        used_vectors.add(candidate["vector_number"])
        swap_pairs.append(
            {
                "spin": primary_spin,
                "from_vector": candidate["vector_number"],
                "to_vector": wrong_slot["vector_number"],
                "reason": (
                    f"replace {wrong_slot['character_class']} SOMO with "
                    f"{candidate['character_class']} buried occupied orbital"
                ),
                "from_orbital": candidate,
                "to_orbital": wrong_slot,
            }
        )

        opposite_spin = "beta" if primary_spin == "alpha" else "alpha"
        beta_orbital = next(
            (
                orbital
                for orbital in mos_payload["orbitals"]
                if orbital.get("spin") == opposite_spin
                and orbital["vector_number"] == candidate["vector_number"]
                and orbital["occupancy"] > 0.1
            ),
            None,
        )
        if beta_orbital is not None:
            beta_summary = _summarize_frontier_orbital(beta_orbital, metal_set)
            if beta_summary is not None and beta_summary["metal_like"]:
                swap_pairs.append(
                    {
                        "spin": opposite_spin,
                        "from_vector": candidate["vector_number"],
                        "to_vector": wrong_slot["vector_number"],
                        "reason": "remove doubly occupied metal-like orbital from opposite-spin occupied space",
                        "from_orbital": beta_summary,
                        "to_orbital": None,
                    }
                )

    vectors_block = None
    if swap_pairs:
        input_name = vectors_input or "CURRENT.movecs"
        output_name = vectors_output or "swapped.movecs"
        lines = [f"vectors input {input_name} \\"]
        for pair in swap_pairs:
            lines.append(f"        swap {pair['spin']} {pair['from_vector']} {pair['to_vector']} \\")
        lines.append(f"        output {output_name}")
        vectors_block = "\n".join(lines)

    return {
        "available": bool(swap_pairs),
        "reason": None if swap_pairs else "no_swap_candidates_identified",
        "primary_spin": primary_spin,
        "wrong_somo_slots": wrong_slots,
        "buried_metal_candidates": candidate_pool[: max(6, len(wrong_slots))],
        "swap_pairs": swap_pairs,
        "vectors_block": vectors_block,
    }


def _summarize_frontier_channel(channel_payload: dict[str, Any], metal_set: set[str]) -> dict[str, Any]:
    return {
        "orbital_count": channel_payload.get("orbital_count", 0),
        "somo_count": channel_payload.get("somo_count", 0),
        "homo": _summarize_frontier_orbital(channel_payload.get("homo"), metal_set),
        "lumo": _summarize_frontier_orbital(channel_payload.get("lumo"), metal_set),
        "somos": [
            _summarize_frontier_orbital(orbital, metal_set)
            for orbital in channel_payload.get("somos", [])
        ],
        "homo_lumo_gap_hartree": channel_payload.get("homo_lumo_gap_hartree"),
    }


def _summarize_frontier_orbital(orbital: dict[str, Any] | None, metal_set: set[str]) -> dict[str, Any] | None:
    if orbital is None:
        return None
    top_atoms = orbital.get("top_atom_contributions") or []
    metal_fraction = sum(
        item.get("fraction_of_visible", 0.0)
        for item in top_atoms
        if item.get("element", "").lower() in metal_set
    )
    ligand_fraction = sum(
        item.get("fraction_of_visible", 0.0)
        for item in top_atoms
        if item.get("element", "").lower() not in metal_set
    )
    shell_contributions = {
        item["ao_shell"]: item.get("fraction_of_visible", 0.0)
        for item in (orbital.get("ao_shell_contributions") or [])
    }
    d_fraction = shell_contributions.get("d", 0.0)
    f_fraction = shell_contributions.get("f", 0.0)
    p_fraction = shell_contributions.get("p", 0.0)
    s_fraction = shell_contributions.get("s", 0.0)

    if metal_fraction >= 0.6 and d_fraction >= 0.35:
        character_class = "metal_centered_d"
    elif metal_fraction >= 0.6 and f_fraction >= 0.25:
        character_class = "metal_centered_f"
    elif metal_fraction >= 0.6:
        character_class = "metal_centered_mixed"
    elif metal_fraction >= 0.3 and ligand_fraction >= 0.3:
        character_class = "mixed_metal_ligand"
    elif p_fraction >= 0.45:
        character_class = "ligand_centered_pi"
    elif s_fraction >= 0.45:
        character_class = "ligand_centered_sigma"
    else:
        character_class = "ligand_centered_mixed"

    return {
        "spin": orbital.get("spin"),
        "vector_number": orbital["vector_number"],
        "energy_hartree": orbital["energy_hartree"],
        "occupancy": orbital["occupancy"],
        "occupation_label": orbital.get("occupation_label"),
        "symmetry": orbital.get("symmetry"),
        "dominant_character": orbital.get("dominant_character"),
        "top_atom_contributions": top_atoms,
        "metal_fraction": metal_fraction,
        "ligand_fraction": ligand_fraction,
        "d_fraction": d_fraction,
        "f_fraction": f_fraction,
        "p_fraction": p_fraction,
        "s_fraction": s_fraction,
        "metal_like": metal_fraction >= 0.5,
        "ligand_like": metal_fraction < 0.35,
        "d_like": d_fraction >= 0.35,
        "f_like": f_fraction >= 0.25,
        "character_class": character_class,
    }


def _summarize_spin_density(population_payload: dict[str, Any] | None, metal_set: set[str]) -> dict[str, Any]:
    if not population_payload:
        return {
            "available": False,
            "method": None,
            "dominant_site": None,
            "metal_population_sum": None,
            "largest_sites": [],
        }

    methods = population_payload.get("methods", {})
    preferred = None
    for method_name in ("lowdin", "mulliken"):
        method_payload = methods.get(method_name)
        if method_payload and method_payload.get("latest_spin") is not None:
            preferred = method_payload
            preferred["method_name"] = method_name
            break
    if preferred is None:
        return {
            "available": False,
            "method": None,
            "dominant_site": None,
            "metal_population_sum": None,
            "largest_sites": [],
        }

    spin_section = preferred["latest_spin"]
    atoms = spin_section.get("atoms", [])
    largest_sites = sorted(atoms, key=lambda item: abs(item["population"]), reverse=True)[:6]
    metal_population_sum = sum(
        abs(atom["population"]) for atom in atoms if atom.get("element", "").lower() in metal_set
    )
    return {
        "available": True,
        "method": preferred["method_name"],
        "dominant_site": largest_sites[0] if largest_sites else None,
        "metal_population_sum": metal_population_sum,
        "largest_sites": largest_sites,
    }
    return {
        "spin": spin,
        "vector_number": orbital["vector_number"],
        "energy_hartree": orbital["energy_hartree"],
        "metal_like": metal_like,
        "metal_score": metal_score,
        "metal_d_score": metal_d_score,
        "top_labels": [contributor["label"] for contributor in top_contributors[:3]],
    }


def _count_sign_changes(values: list[float]) -> int:
    signs: list[int] = []
    for value in values:
        if value > 0:
            signs.append(1)
        elif value < 0:
            signs.append(-1)
    return sum(1 for idx in range(1, len(signs)) if signs[idx] != signs[idx - 1])


def _density_ratio(iterations: list[dict[str, Any]]) -> float | None:
    if len(iterations) < 2:
        return None
    start = iterations[0]["rms_density"]
    end = iterations[-1]["rms_density"]
    if start is None or end is None or start == 0:
        return None
    return end / start


def _classify_scf_pattern(
    status: str,
    recent: list[dict[str, Any]],
    density_ratio: float | None,
    delta_sign_changes: int,
) -> str:
    if not recent:
        return "no_iterations_found"
    if density_ratio is None:
        return "insufficient_data"
    final_density = recent[-1]["rms_density"] or 0.0
    if status == "failed":
        if density_ratio < 0.05 and final_density < 1e-4:
            return "nearly_converged"
        if density_ratio < 0.15 and delta_sign_changes <= 1:
            return "slow_improving"
        if delta_sign_changes >= 2:
            return "oscillatory"
        return "stalled"
    if delta_sign_changes >= 3 and final_density > 1e-5:
        return "oscillatory_but_converged"
    if density_ratio < 0.05:
        return "well_converged"
    return "converged"


def _looks_like_frequency_run(tasks_payload: dict[str, Any]) -> bool:
    return any(task["kind"] == "frequency" for task in tasks_payload["generic_tasks"])


def _looks_like_optimization_run(tasks_payload: dict[str, Any]) -> bool:
    return any(task["kind"] == "optimization" for task in tasks_payload["generic_tasks"])


def _last_task_kind(tasks_payload: dict[str, Any]) -> str | None:
    generic_tasks = tasks_payload["generic_tasks"]
    if not generic_tasks:
        return None
    return generic_tasks[-1]["kind"]

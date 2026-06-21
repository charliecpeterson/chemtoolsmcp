"""next_actions builder for NWChem tool results.

Extracted from mcp/tools/nwchem.py — a ~900-line pure-data helper that maps
analysis results to a structured, model-executable next_actions list. Shared
by ~17 handlers; kept separate so the handler module stays navigable.
"""
from __future__ import annotations

from typing import Any


def _build_next_actions(
    context: str,
    result: dict[str, Any],
    output_file: str = "",
    input_file: str = "",
    profile: str = "",
) -> list[dict[str, Any]]:
    """Build a structured next_actions list from analysis results.

    Each action is a dict with: priority, tool, params, reason, confidence.
    The model can execute actions[0] without understanding NWChem internals.
    """
    actions: list[dict[str, Any]] = []

    if context == "analyze_case":
        diagnosis = result.get("diagnosis") or {}
        task_outcome = diagnosis.get("task_outcome", "")
        failure_class = diagnosis.get("failure_class", "")
        recommended_next_action = diagnosis.get("recommended_next_action", "")
        next_step = result.get("next_step") or {}
        selected_workflow = next_step.get("selected_workflow", "")

        if task_outcome == "success":
            wf = selected_workflow.lower()
            if "tce" in wf or "ccsd" in wf or "mp2" in wf:
                actions.append({
                    "priority": 1,
                    "tool": "parse_nwchem_tce_output",
                    "params": {"output_file": output_file},
                    "reason": "Correlated calculation completed — extract energies and T1/D1 diagnostics.",
                    "confidence": 0.95,
                })
            elif "freq" in wf:
                actions.append({
                    "priority": 1,
                    "tool": "check_nwchem_freq_plausibility",
                    "params": {"output_file": output_file, "input_file": input_file},
                    "reason": "Frequency calculation completed — verify plausibility before using results.",
                    "confidence": 0.95,
                })
            elif "opt" in wf or "geometry" in wf:
                actions.append({
                    "priority": 1,
                    "tool": "extract_nwchem_geometry",
                    "params": {"output_file": output_file, "frame": "best"},
                    "reason": "Optimization converged — extract geometry for next step.",
                    "confidence": 0.90,
                })
            elif failure_class == "wrong_state_convergence":
                actions.append({
                    "priority": 1,
                    "tool": "analyze_nwchem_frontier_orbitals",
                    "params": {"output_file": output_file, "input_file": input_file},
                    "reason": "Converged to a suspect spin state — check whether the unpaired "
                              "electrons sit on the metal or have leaked onto the ligands.",
                    "confidence": 0.85,
                })
                actions.append({
                    "priority": 2,
                    "tool": "suggest_nwchem_multiplicity_scan",
                    "params": {"input_file": input_file},
                    "reason": "Confirm the spin ground state: scan candidate multiplicities and take "
                              "the lowest energy — the converged multiplicity may be too high.",
                    "confidence": 0.85,
                })
            elif failure_class == "frequency_interpretation_required":
                actions.append({
                    "priority": 1,
                    "tool": "analyze_nwchem_imaginary_modes",
                    "params": {"output_file": output_file, "input_file": input_file},
                    "reason": "Imaginary mode(s) present — inspect them to tell a real transition "
                              "state from a numerical artifact before using the geometry.",
                    "confidence": 0.9,
                })
            elif recommended_next_action == "verify_state_quality_before_accepting_result":
                actions.append({
                    "priority": 1,
                    "tool": "analyze_nwchem_frontier_orbitals",
                    "params": {"output_file": output_file, "input_file": input_file},
                    "reason": "SCF converged, but the spin state isn't verified — first confirm "
                              "the unpaired electrons sit on the metal, not the ligands.",
                    "confidence": 0.85,
                })
                actions.append({
                    "priority": 2,
                    "tool": "suggest_nwchem_multiplicity_scan",
                    "params": {"input_file": input_file},
                    "reason": "Then confirm this is the lowest spin state — a clean SCF converges "
                              "to whatever multiplicity it's given; scan candidates and compare energies.",
                    "confidence": 0.8,
                })
            else:
                actions.append({
                    "priority": 1,
                    "tool": "parse_nwchem_output",
                    "params": {"output_file": output_file, "sections": ["tasks"]},
                    "reason": "Calculation completed — review results.",
                    "confidence": 0.80,
                })
        elif task_outcome in ("failed", "error", "scf_failed"):
            if failure_class == "scf_convergence":
                actions.append({
                    "priority": 1,
                    "tool": "suggest_nwchem_recovery",
                    "params": {"output_file": output_file, "input_file": input_file, "mode": "scf"},
                    "reason": "SCF convergence failure — get targeted recovery strategies.",
                    "confidence": 0.90,
                })
            elif failure_class in ("bad_state", "wrong_state", "state_mismatch"):
                actions.append({
                    "priority": 1,
                    "tool": "suggest_nwchem_recovery",
                    "params": {"output_file": output_file, "input_file": input_file, "mode": "state"},
                    "reason": "Spin/state error — recover with state correction strategies.",
                    "confidence": 0.85,
                })
            elif failure_class in ("memory", "oom", "ma_init"):
                actions.append({
                    "priority": 1,
                    "tool": "create_nwchem_input_variant",
                    "params": {
                        "source_input": input_file,
                        "changes": {"memory": "800 mb"},
                        "reason": f"OOM failure ({failure_class}) — reduce memory",
                    },
                    "reason": "Out of memory — reduce memory directive and resubmit.",
                    "confidence": 0.80,
                })
            else:
                actions.append({
                    "priority": 1,
                    "tool": "suggest_nwchem_recovery",
                    "params": {"output_file": output_file, "input_file": input_file, "mode": "auto"},
                    "reason": f"Calculation failed ({failure_class or 'unknown'}) — get recovery recommendations.",
                    "confidence": 0.75,
                })
        elif task_outcome == "timelimit":
            actions.append({
                "priority": 1,
                "tool": "prepare_nwchem_freq_restart",
                "params": {"input_file": input_file, "output_file": output_file, "profile": profile},
                "reason": "Hit walltime limit — check if freq restart is ready.",
                "confidence": 0.85,
            })

    elif context == "watch_run":
        overall = result.get("overall_status", "")

        if overall == "completed":
            actions.append({
                "priority": 1,
                "tool": "analyze_nwchem_case",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "Job completed — run full analysis to determine next steps.",
                "confidence": 0.95,
            })
        elif overall in ("failed", "error"):
            actions.append({
                "priority": 1,
                "tool": "analyze_nwchem_case",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "Job failed — diagnose the failure.",
                "confidence": 0.90,
            })
        elif overall == "running":
            actions.append({
                "priority": 1,
                "tool": "watch_nwchem_run",
                "params": {"output_file": output_file, "input_file": input_file, "profile": profile},
                "reason": "Job is still running — continue monitoring.",
                "confidence": 0.95,
            })

    elif context == "freq_plausibility":
        assessment = result.get("overall_assessment", "")
        imag_count = result.get("imaginary_mode_count", 0)
        if assessment == "suspicious" and imag_count and imag_count > 0:
            actions.append({
                "priority": 1,
                "tool": "analyze_nwchem_imaginary_modes",
                "params": {"output_file": output_file},
                "reason": f"Found {imag_count} imaginary mode(s) — analyze which atoms are involved.",
                "confidence": 0.90,
            })
            actions.append({
                "priority": 2,
                "tool": "draft_nwchem_imaginary_mode_inputs",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "Generate displaced geometries for re-optimization.",
                "confidence": 0.80,
            })
        elif assessment in ("ok", "plausible"):
            actions.append({
                "priority": 1,
                "tool": "parse_nwchem_output",
                "params": {"output_file": output_file, "sections": ["freq", "tasks"]},
                "reason": "Frequencies look reasonable — extract thermochemistry data.",
                "confidence": 0.90,
            })

    elif context == "run_status":
        status = result.get("status", "")
        if status in ("completed", "done"):
            actions.append({
                "priority": 1,
                "tool": "analyze_nwchem_case",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "Job completed — run full analysis.",
                "confidence": 0.95,
            })
        elif status in ("failed", "error"):
            actions.append({
                "priority": 1,
                "tool": "analyze_nwchem_case",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "Job failed — diagnose the failure.",
                "confidence": 0.90,
            })
        elif status in ("cancelled", "timelimit"):
            actions.append({
                "priority": 1,
                "tool": "prepare_nwchem_freq_restart",
                "params": {"input_file": input_file, "output_file": output_file, "profile": profile},
                "reason": "Job cancelled/timelimit — check if restart is possible.",
                "confidence": 0.80,
            })

    elif context == "imaginary_modes":
        sig_count = result.get("significant_imaginary_mode_count", 0)
        if sig_count > 0:
            actions.append({
                "priority": 1,
                "tool": "draft_nwchem_imaginary_mode_inputs",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": f"{sig_count} significant imaginary mode(s) — generate displaced inputs for re-optimization.",
                "confidence": 0.85,
            })
        else:
            actions.append({
                "priority": 1,
                "tool": "parse_nwchem_output",
                "params": {"output_file": output_file, "sections": ["freq", "tasks"]},
                "reason": "No significant imaginary modes — extract thermochemistry data.",
                "confidence": 0.90,
            })

    elif context == "spin_charge_state":
        # Handler dispatches on `assessment` ("plausible" / "suspicious") and
        # the state_check_assessment values from analyze_frontier_orbitals.
        assessment = (result.get("assessment") or "").lower()
        state_check = (result.get("state_check_assessment") or "").lower()
        observed = result.get("observed_somo_count")
        expected = result.get("expected_somo_count")
        rec = result.get("recommended_next_action") or ""

        # Mismatch by count is the cleanest "swap me" signal.
        if (
            assessment == "suspicious"
            and isinstance(observed, int) and isinstance(expected, int)
            and observed != expected
        ):
            actions.append({
                "priority": 1,
                "tool": "draft_nwchem_vectors_swap_input",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": (
                    f"Observed SOMO count ({observed}) does not match expected "
                    f"({expected}). Recover by swapping vectors so the metal-centered "
                    f"orbitals sit in the SOMO positions, then re-converge SCF."
                ),
                "confidence": 0.85,
            })
        elif state_check == "metal_state_mismatch_suspected":
            actions.append({
                "priority": 1,
                "tool": "analyze_nwchem_frontier_orbitals",
                "params": {
                    "output_file": output_file,
                    "input_file": input_file,
                },
                "reason": (
                    "State check flags metal-state mismatch — drill into the frontier "
                    "orbital characters to confirm which SOMOs are ligand-centered."
                ),
                "confidence": 0.8,
            })
            actions.append({
                "priority": 2,
                "tool": "draft_nwchem_vectors_swap_input",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "If the frontier analysis confirms ligand-centered SOMOs, swap to fix.",
                "confidence": 0.65,
            })
        elif assessment == "suspicious":
            # Catch-all suspicious: route to recovery strategy advisor.
            actions.append({
                "priority": 1,
                "tool": "suggest_nwchem_state_recovery_strategy",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": f"State looks suspicious: {rec or 'reason unclear'}. Get explicit recovery routes.",
                "confidence": 0.7,
            })
        else:
            # plausible / unavailable — usually still want to drill in once.
            actions.append({
                "priority": 1,
                "tool": "analyze_nwchem_frontier_orbitals",
                "params": {
                    "output_file": output_file,
                    "input_file": input_file,
                },
                "reason": "State looks plausible — confirm by inspecting the SOMO characters.",
                "confidence": 0.7,
            })

    elif context == "electronic_structure":
        # summarize_electronic_structure returns spin_state_consistent + somo_count.
        consistent = result.get("spin_state_consistent")
        somo_count = result.get("somo_count") or 0
        metal_centers = result.get("metal_centers") or []

        if consistent is False:
            actions.append({
                "priority": 1,
                "tool": "check_nwchem_spin_charge_state",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": (
                    f"Spin state looks inconsistent (somo_count={somo_count}; "
                    f"metal centers: {metal_centers or 'none'}) — drill in with the "
                    "state check tool to confirm and route recovery."
                ),
                "confidence": 0.8,
            })
        elif metal_centers and somo_count > 0:
            actions.append({
                "priority": 1,
                "tool": "analyze_nwchem_frontier_orbitals",
                "params": {
                    "output_file": output_file,
                    "input_file": input_file,
                    "expected_metals": metal_centers,
                },
                "reason": (
                    "Open-shell metal complex — confirm the SOMO characters are "
                    "metal-centered before trusting the result."
                ),
                "confidence": 0.8,
            })
        else:
            actions.append({
                "priority": 1,
                "tool": "parse_nwchem_thermochem",
                "params": {"output_file": output_file},
                "reason": "Electronic structure summary looks clean — extract thermochemistry.",
                "confidence": 0.8,
            })

    elif context == "summarize_output":
        # summarize_output returns outcome + failure_class + recommended_next_action
        outcome = (result.get("outcome") or "").lower()
        failure_class = (result.get("failure_class") or "").lower()
        rec = (result.get("recommended_next_action") or "").lower()

        if outcome in {"success", "completed"}:
            # Pick a follow-up based on what was computed.
            if result.get("frequency"):
                actions.append({
                    "priority": 1,
                    "tool": "check_nwchem_freq_plausibility",
                    "params": {"output_file": output_file, "input_file": input_file},
                    "reason": "Run completed with frequency data — validate before using thermochem.",
                    "confidence": 0.85,
                })
            elif result.get("optimization_status") == "converged":
                actions.append({
                    "priority": 1,
                    "tool": "check_nwchem_geometry_plausibility",
                    "params": {"output_file": output_file, "input_file": input_file},
                    "reason": "Optimization converged — validate the final geometry.",
                    "confidence": 0.85,
                })
            elif result.get("correlated_method"):
                actions.append({
                    "priority": 1,
                    "tool": "parse_nwchem_tce_output",
                    "params": {"output_file": output_file},
                    "reason": "Correlated calc completed — extract correlation energy + MR diagnostics.",
                    "confidence": 0.85,
                })
            else:
                actions.append({
                    "priority": 1,
                    "tool": "analyze_nwchem_frontier_orbitals",
                    "params": {"output_file": output_file, "input_file": input_file},
                    "reason": "SCF/DFT energy completed — confirm orbital ordering before trusting the result.",
                    "confidence": 0.8,
                })
        elif outcome in {"failed", "error", "crashed"}:
            mode = "scf" if failure_class == "scf_nonconvergence" else "auto"
            actions.append({
                "priority": 1,
                "tool": "suggest_nwchem_recovery",
                "params": {"output_file": output_file, "input_file": input_file, "mode": mode},
                "reason": (
                    f"Run did not complete ({failure_class or 'unknown failure class'}). "
                    "Get recovery strategies."
                ),
                "confidence": 0.85,
            })
        else:
            # incomplete or unknown — drill in with full case analysis
            actions.append({
                "priority": 1,
                "tool": "analyze_nwchem_case",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": f"Output is {outcome or 'unknown'} — run the full case analysis.",
                "confidence": 0.75,
            })

    elif context == "track_spin_state":
        # track_spin_state_across_optimization flags state flips along an opt
        # trajectory. recommendation is already populated; map to a tool call.
        flip_detected = result.get("flip_detected")
        flip_steps = result.get("flip_steps") or []
        warnings = result.get("warnings") or []
        recommendation_text = result.get("recommendation") or ""

        if flip_detected:
            actions.append({
                "priority": 1,
                "tool": "draft_nwchem_vectors_swap_input",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": (
                    f"State flip(s) detected at step(s) {flip_steps[:3]} — recover "
                    f"by swapping vectors and re-converging. {recommendation_text}"
                ),
                "confidence": 0.85,
            })
            actions.append({
                "priority": 2,
                "tool": "suggest_nwchem_state_recovery_strategy",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "If a simple swap doesn't stabilize the state, get more recovery strategies.",
                "confidence": 0.7,
            })
        elif warnings:
            actions.append({
                "priority": 1,
                "tool": "check_nwchem_spin_charge_state",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": (
                    f"{len(warnings)} warning(s) on the spin state trajectory but no flips — "
                    f"confirm the final state is what was requested."
                ),
                "confidence": 0.7,
            })
        else:
            actions.append({
                "priority": 1,
                "tool": "check_nwchem_freq_plausibility",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "Spin state stable across the trajectory — proceed with freq validation.",
                "confidence": 0.8,
            })

    elif context == "freq_progress":
        # parse_nwchem_freq_progress returns pct_complete + restart estimates.
        pct = result.get("pct_complete")
        runs_needed = result.get("runs_needed_at_48h_walltime")
        fdrst = result.get("fdrst") or {}
        fdrst_present = fdrst.get("exists")

        if pct is None:
            actions.append({
                "priority": 1,
                "tool": "parse_nwchem_output",
                "params": {"output_file": output_file, "sections": ["tasks"]},
                "reason": "Could not detect freq progress markers — fall back to task-level parsing.",
                "confidence": 0.55,
            })
        elif pct >= 99.0:
            actions.append({
                "priority": 1,
                "tool": "check_nwchem_freq_plausibility",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "Frequency calc is essentially complete — validate the results.",
                "confidence": 0.9,
            })
        elif fdrst_present and isinstance(runs_needed, int) and runs_needed > 1:
            actions.append({
                "priority": 1,
                "tool": "prepare_nwchem_freq_restart",
                "params": {"input_file": input_file, "output_file": output_file},
                "reason": (
                    f"Run is {pct:.0f}% through; the .fdrst restart file is present "
                    f"and an estimated {runs_needed} restart cycle(s) are needed at "
                    f"48h walltime — prepare a restart input."
                ),
                "confidence": 0.85,
            })
        else:
            actions.append({
                "priority": 1,
                "tool": "watch_nwchem_run",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": (
                    f"Freq calc is {pct:.0f}% complete; estimated remaining "
                    f"{result.get('estimated_remaining_hours', 0):.1f} h — keep watching."
                ),
                "confidence": 0.7,
            })

    elif context == "review_progress":
        # review_nwchem_progress returns overall_status + intervention block.
        overall = (result.get("overall_status") or "").lower()
        intervention = result.get("intervention") or {}
        rec = (intervention.get("recommended_action") or "").lower()
        should_terminate = intervention.get("should_terminate_process")

        if overall in {"completed_success", "completed", "finished"}:
            actions.append({
                "priority": 1,
                "tool": "analyze_nwchem_case",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "Job completed — run full case analysis to confirm and pick next step.",
                "confidence": 0.9,
            })
        elif should_terminate:
            actions.append({
                "priority": 1,
                "tool": "terminate_nwchem_run",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": (
                    f"Intervention recommends terminating the job: {rec or 'see reasons'}. "
                    "Cancel and re-plan."
                ),
                "confidence": 0.8,
            })
            actions.append({
                "priority": 2,
                "tool": "suggest_nwchem_recovery",
                "params": {"output_file": output_file, "input_file": input_file, "mode": "auto"},
                "reason": "After termination, get recovery strategy for the next attempt.",
                "confidence": 0.75,
            })
        elif overall in {"running", "in_progress", "active"}:
            actions.append({
                "priority": 1,
                "tool": "watch_nwchem_run",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "Job is still running and nothing flags intervention — keep watching.",
                "confidence": 0.75,
            })
        elif overall in {"failed", "error", "crashed"}:
            actions.append({
                "priority": 1,
                "tool": "analyze_nwchem_case",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "Job did not complete cleanly — run full case analysis for failure diagnosis.",
                "confidence": 0.85,
            })
        else:
            actions.append({
                "priority": 1,
                "tool": "watch_nwchem_run",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": f"Unknown status ({overall!r}) — continue watching.",
                "confidence": 0.5,
            })

    elif context == "tce_validation":
        # validate_nwchem_tce_setup returns status ("ok"/"warnings"/"errors"),
        # issues (with level/code), detected (parsed fields).
        status = (result.get("status") or "").lower()
        issues_list = result.get("issues") or []
        error_codes = {i.get("code") for i in issues_list if i.get("level") == "error"}

        if status == "errors":
            # Specific recovery paths for known error codes.
            if "freeze_atomic_forbidden" in error_codes or "missing_freeze" in error_codes:
                actions.append({
                    "priority": 1,
                    "tool": "prepare_nwchem_tce_setup",
                    "params": {"scf_output_path": output_file or "<scf.out>"},
                    "reason": (
                        "Freeze directive missing or uses `freeze atomic` — "
                        "run the TCE setup orchestrator to compute the explicit "
                        "freeze count and regenerate the input."
                    ),
                    "confidence": 0.85,
                })
            elif "missing_vectors" in error_codes or "vectors_unreachable" in error_codes:
                actions.append({
                    "priority": 1,
                    "tool": "find_nwchem_restart_assets",
                    "params": {"path": input_file or "<tce_input.nw>"},
                    "reason": "Vectors file path is missing or unreachable — locate the right movecs in the run directory.",
                    "confidence": 0.8,
                })
            else:
                actions.append({
                    "priority": 1,
                    "tool": "prepare_nwchem_tce_setup",
                    "params": {"scf_output_path": output_file or "<scf.out>"},
                    "reason": (
                        f"TCE input has {len(error_codes)} error(s): "
                        f"{', '.join(sorted(c for c in error_codes if c)) or 'unspecified'}. "
                        "Use the TCE setup orchestrator to regenerate cleanly."
                    ),
                    "confidence": 0.7,
                })
        elif status == "warnings":
            actions.append({
                "priority": 1,
                "tool": "launch_nwchem_run",
                "params": {"input_file": input_file or "<tce_input.nw>"},
                "reason": (
                    f"TCE input passes validation with {len(issues_list)} warning(s) — "
                    "OK to launch, but review the warnings first if any flag a "
                    "specific concern (e.g. unusual freeze count)."
                ),
                "confidence": 0.7,
            })
        else:
            # ok
            actions.append({
                "priority": 1,
                "tool": "launch_nwchem_run",
                "params": {"input_file": input_file or "<tce_input.nw>"},
                "reason": "TCE input passes all validation checks — ready to launch.",
                "confidence": 0.9,
            })

    elif context == "movecs":
        # parse_nwchem_movecs returns binary-only eigenvalues + occupancies.
        # Orbital character (metal-d vs ligand-π etc.) requires the matching
        # .out text — route the agent toward the right next call.
        n_mo = result.get("n_mo") or 0
        n_occ = result.get("n_occupied") or 0
        orbitals = result.get("orbitals") or []

        # Sanity-check the eigenvalues — a degenerate or inverted ordering hints
        # at the kind of pathology that motivates calling movecs in the first place.
        has_ordering_issue = False
        if orbitals:
            occ_energies = [o["energy_hartree"] for o in orbitals if o.get("occupied")]
            if occ_energies and any(
                occ_energies[i + 1] < occ_energies[i] - 1e-6
                for i in range(len(occ_energies) - 1)
            ):
                has_ordering_issue = True

        if has_ordering_issue:
            actions.append({
                "priority": 1,
                "tool": "parse_nwchem_mos",
                "params": {"output_file": output_file, "top_n": 20},
                "reason": (
                    "Occupied-orbital eigenvalues are not monotonically increasing — "
                    "fetch dominant_character from the .out to confirm whether a swap "
                    "is needed before TCE."
                ),
                "confidence": 0.8,
            })
            actions.append({
                "priority": 2,
                "tool": "prepare_nwchem_tce_setup",
                "params": {"scf_output_path": output_file},
                "reason": "After confirming ordering, run the TCE setup orchestrator to draft the input with correct freeze + swap_list.",
                "confidence": 0.65,
            })
        else:
            actions.append({
                "priority": 1,
                "tool": "prepare_nwchem_tce_setup",
                "params": {"scf_output_path": output_file},
                "reason": (
                    f"Movecs has {n_mo} MOs ({n_occ} occupied) with sane ordering — "
                    "feed into the TCE setup orchestrator for freeze count + draft."
                ),
                "confidence": 0.8,
            })

    elif context == "tce_output":
        # parse_nwchem_tce_output returns scf_total_energy_hartree + method +
        # correlation_energy_hartree + multireference_diagnostics (added by
        # the handler when amplitude files are present).
        mr = result.get("multireference_diagnostics") or {}
        mr_assessment = (mr.get("mr_assessment") or "").lower()
        method = (result.get("method") or "").lower()
        correlation_energy = result.get("correlation_energy_hartree")

        if mr_assessment == "strong":
            actions.append({
                "priority": 1,
                "tool": "prepare_nwchem_mcscf_setup",
                "params": {"scf_output_path": output_file, "input_path": input_file},
                "reason": (
                    "Strong multireference character (T1 > 0.05 or D1 > 0.05) — "
                    "single-reference CCSD is unreliable. Switch to a "
                    "multireference treatment (CASSCF) and re-run."
                ),
                "confidence": 0.85,
            })
        elif mr_assessment == "moderate":
            actions.append({
                "priority": 1,
                "tool": "draft_nwchem_tce_input",
                "params": {
                    "scf_output_path": output_file,
                    "method": "ccsd(t)",
                    "input_file": input_file,
                },
                "reason": (
                    "Moderate MR character (0.02 < T1 < 0.05). CCSD is borderline — "
                    "draft a CCSD(T) job to bracket the answer."
                ),
                "confidence": 0.75,
            })
        elif correlation_energy is None:
            actions.append({
                "priority": 1,
                "tool": "draft_nwchem_tce_restart_input",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": "TCE did not produce a correlation energy — likely incomplete. Draft a restart input.",
                "confidence": 0.7,
            })
        else:
            # Clean TCE run — proceed with the workflow (thermochem, comparison, etc.)
            actions.append({
                "priority": 1,
                "tool": "parse_nwchem_thermochem",
                "params": {"output_file": output_file},
                "reason": (
                    f"{method.upper() if method else 'TCE'} correlation energy "
                    f"converged; extract thermochemistry to finish the workflow."
                ),
                "confidence": 0.85,
            })

    elif context == "compare_runs":
        # compare_nwchem_runs returns overall_assessment + energy_delta_kcal_mol.
        assessment = (result.get("overall_assessment") or "").lower()
        regressions = result.get("regressed_signals") or []
        improvements = result.get("improved_signals") or []
        delta_kcal = result.get("energy_delta_kcal_mol")

        if assessment in {"candidate_better", "candidate_improved", "improved"}:
            actions.append({
                "priority": 1,
                "tool": "analyze_nwchem_case",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": (
                    f"Candidate run improved over the reference"
                    + (f" (Δ = {delta_kcal:+.2f} kcal/mol)" if isinstance(delta_kcal, (int, float)) else "")
                    + (f", {len(improvements)} signal(s) improved" if improvements else "")
                    + ". Run full case analysis on the candidate to confirm and pick the next step."
                ),
                "confidence": 0.85,
            })
        elif assessment in {"candidate_worse", "regressed", "candidate_regressed"}:
            actions.append({
                "priority": 1,
                "tool": "suggest_nwchem_recovery",
                "params": {"output_file": output_file, "input_file": input_file, "mode": "auto"},
                "reason": (
                    f"Candidate run regressed (Δ={delta_kcal}; {len(regressions)} signal(s) worse). "
                    "Get recovery suggestions before another attempt."
                ),
                "confidence": 0.75,
            })
        else:
            # no_clear_change / unknown — drill into freq + frontier orbitals
            actions.append({
                "priority": 1,
                "tool": "check_nwchem_freq_plausibility",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": (
                    "No clear improvement or regression — disambiguate by checking the "
                    "candidate's frequency plausibility (real minimum? imaginary modes?)."
                ),
                "confidence": 0.65,
            })

    elif context == "geometry_plausibility":
        # Geometry plausibility returns plausible (bool), red_flags (list),
        # warnings (list), and bond/coordination summaries.
        plausible = result.get("plausible")
        red_flags = result.get("red_flags") or []
        warnings = result.get("warnings") or []

        if plausible is False or red_flags:
            actions.append({
                "priority": 1,
                "tool": "draft_nwchem_optimization_followup_input",
                "params": {"output_path": output_file, "input_path": input_file},
                "reason": (
                    f"Geometry has {len(red_flags)} red flag(s) — "
                    "draft a follow-up optimization with adjusted strategy."
                ),
                "confidence": 0.8,
            })
            actions.append({
                "priority": 2,
                "tool": "extract_nwchem_geometry",
                "params": {"output_file": output_file, "frame": "best"},
                "reason": "If the follow-up strategy fails, fall back to the best frame extracted from the trajectory.",
                "confidence": 0.6,
            })
        elif warnings:
            actions.append({
                "priority": 1,
                "tool": "check_nwchem_freq_plausibility",
                "params": {"output_file": output_file, "input_file": input_file},
                "reason": (
                    f"{len(warnings)} geometry warning(s) — check the frequency "
                    f"calc to see whether the structure is a real minimum."
                ),
                "confidence": 0.75,
            })
        else:
            actions.append({
                "priority": 1,
                "tool": "parse_nwchem_output",
                "params": {"output_file": output_file, "sections": ["tasks"]},
                "reason": "Geometry is plausible — proceed with energy extraction or freq calc.",
                "confidence": 0.85,
            })

    elif context == "frontier_orbitals":
        analysis = result.get("analysis") or {}
        analysis_assessment = (analysis.get("assessment") or "").lower()
        somo_count = analysis.get("somo_count") or 0
        expected_somo = analysis.get("expected_somo_count")
        metal_like = analysis.get("metal_like_somo_count") or 0
        ligand_like = analysis.get("ligand_like_somo_count") or 0
        expected_metals = result.get("expected_metal_elements") or []

        if analysis_assessment == "metal_state_mismatch_suspected":
            actions.append({
                "priority": 1,
                "tool": "draft_nwchem_vectors_swap_input",
                "params": {
                    "output_file": output_file,
                    "input_file": input_file,
                    "expected_metal_elements": expected_metals,
                    "expected_somo_count": expected_somo,
                },
                "reason": (
                    f"Frontier orbitals flagged as metal-state mismatch "
                    f"({metal_like} metal-like vs {ligand_like} ligand-like of "
                    f"{somo_count} SOMOs). Apply vectors swap to push metal "
                    f"orbitals into the SOMO positions."
                ),
                "confidence": 0.85,
            })
            actions.append({
                "priority": 2,
                "tool": "suggest_nwchem_vectors_swaps",
                "params": {
                    "output_file": output_file,
                    "input_file": input_file,
                    "expected_metal_elements": expected_metals,
                    "expected_somo_count": expected_somo,
                },
                "reason": "If automated drafting needs different swap pairs, get explicit pair suggestions.",
                "confidence": 0.75,
            })
        elif expected_metals and metal_like == 0 and somo_count > 0:
            # Open-shell run with expected metals but ligand-centered SOMOs.
            actions.append({
                "priority": 1,
                "tool": "draft_nwchem_vectors_swap_input",
                "params": {
                    "output_file": output_file,
                    "input_file": input_file,
                    "expected_metal_elements": expected_metals,
                    "expected_somo_count": expected_somo,
                },
                "reason": (
                    f"Expected metal-centered open shell on {expected_metals} but all "
                    f"{somo_count} SOMOs are ligand-centered — classic case for "
                    f"vectors swap."
                ),
                "confidence": 0.85,
            })
        elif not analysis.get("available"):
            actions.append({
                "priority": 1,
                "tool": "parse_nwchem_mos",
                "params": {"output_file": output_file, "top_n": 12},
                "reason": "Frontier orbitals not parseable — inspect the raw MO list.",
                "confidence": 0.6,
            })
        else:
            # Clean frontier — agent can proceed to thermochem / energies.
            actions.append({
                "priority": 1,
                "tool": "parse_nwchem_output",
                "params": {"output_file": output_file, "sections": ["tasks"]},
                "reason": "Frontier orbitals look consistent with expectations — extract final energy / next step.",
                "confidence": 0.85,
            })

    return actions

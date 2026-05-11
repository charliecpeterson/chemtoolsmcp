from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from chemtools.core.common import detect_program, make_metadata, read_text, ELEMENT_TO_Z
from chemtools.programs.nwchem.strategy.diagnose import (
    analyze_frontier_orbitals as analyze_nwchem_frontier_orbitals,
    diagnose_nwchem_output,
    parse_scf,
    suggest_vectors_swaps as suggest_nwchem_vectors_swaps,
    summarize_nwchem_output,
)
from chemtools.programs.nwchem.parse.input import inspect_nwchem_input
from chemtools.programs.nwchem.parse.freq import parse_trajectory
from chemtools.programs.nwchem.parse.mos import parse_mos, parse_population_analysis
from chemtools.programs.nwchem.input._utils import _TRANSITION_METALS, _COVALENT_RADII, _strategy_entry, _coerce_api_int, _coerce_api_float
from chemtools.programs.nwchem.output import parse_mcscf_output


def check_spin_charge_state(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
) -> dict[str, Any]:
    diagnosis = diagnose_nwchem_output(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
    )
    input_summary = diagnosis["input_summary"]
    state_check = diagnosis["state_check"]
    multiplicity = input_summary["multiplicity"] if input_summary else None
    charge = input_summary["charge"] if input_summary else None
    inferred_expected_somos = expected_somo_count
    if inferred_expected_somos is None and multiplicity is not None and multiplicity > 1:
        inferred_expected_somos = multiplicity - 1

    reasons: list[str] = []
    assessment = "unavailable"
    confidence = diagnosis["confidence"]

    if not state_check["available"]:
        if state_check["assessment"] != "unavailable":
            assessment = "suspicious"
            reasons.append(state_check["assessment"])
        else:
            reasons.append("frontier_state_analysis_unavailable")
    else:
        if multiplicity == 1 and state_check["somo_count"] > 0:
            assessment = "suspicious"
            reasons.append("singlet_input_but_open_shell_somos_found")
        elif inferred_expected_somos is not None and state_check["somo_count"] != inferred_expected_somos:
            assessment = "suspicious"
            reasons.append("somo_count_does_not_match_multiplicity_or_expected_state")
        elif state_check["assessment"] in {"metal_state_mismatch_suspected", "somo_count_mismatch"}:
            assessment = "suspicious"
            reasons.append(state_check["assessment"])
        else:
            assessment = "plausible"
            reasons.append("frontier_and_spin_signals_are_consistent_with_requested_state")

    dominant_site = (state_check.get("spin_density_summary") or {}).get("dominant_site")
    if dominant_site is not None:
        reasons.append(
            f"dominant_spin_density_on_{dominant_site['element']}{dominant_site['atom_index']}"
        )

    return {
        "output_file": output_path,
        "input_file": input_path,
        "assessment": assessment,
        "confidence": confidence,
        "charge": charge,
        "multiplicity": multiplicity,
        "expected_somo_count": inferred_expected_somos,
        "observed_somo_count": state_check.get("somo_count"),
        "metal_like_somo_count": state_check.get("metal_like_somo_count"),
        "ligand_like_somo_count": state_check.get("ligand_like_somo_count"),
        "state_check_assessment": state_check.get("assessment"),
        "dominant_spin_site": dominant_site,
        "reasons": reasons,
        "recommended_next_action": diagnosis["recommended_next_action"],
        "input_summary": input_summary,
    }


def _try_parse_tce(output_path: str, _contents: str | None = None) -> "dict[str, Any] | None":
    """Return parse_tce_output result if the file contains a TCE section, else None."""
    try:
        from chemtools.core.common import read_text
        from chemtools.programs.nwchem.parse.tce import parse_tce_output as _parse_tce_output
        contents = _contents if _contents is not None else read_text(output_path)
        result = _parse_tce_output(output_path, contents)
        if result.get("tce_sections"):
            return result
    except Exception:
        pass
    return None


def _tce_summary_bullets(tce: "dict[str, Any]", amp: "dict[str, Any] | None" = None) -> "list[str]":
    """Build summary bullet strings from a parse_tce_output payload.

    Parameters
    ----------
    tce:
        Result of ``parse_tce_output``.
    amp:
        Optional result of ``parse_tce_amplitudes``.  When provided, T1/D1/T2
        diagnostics and the MR verdict are included in the bullets.
    """
    bullets: list[str] = []
    method = tce.get("method") or "TCE"
    converged = tce.get("converged", False)
    total_e = tce.get("total_energy_hartree")
    corr_e = tce.get("correlation_energy_hartree")

    # Pull frozen core count and n_orbitals from the first section
    section = tce["tce_sections"][0] if tce.get("tce_sections") else {}
    frozen = section.get("frozen_cores")
    n_orb = section.get("n_orbitals")
    n_elec = section.get("n_electrons")

    status_str = "converged" if converged else "NOT converged"
    bullets.append(f"TCE method: {method} — {status_str}")

    if total_e is not None:
        bullets.append(f"TCE total energy: {total_e:.12f} Ha")
    if corr_e is not None:
        bullets.append(f"Correlation energy: {corr_e:.12f} Ha")
    if frozen is not None:
        core_note = f"{frozen} frozen core orbital{'s' if frozen != 1 else ''}"
        if n_orb is not None and n_elec is not None:
            correlated_elec = n_elec - 2 * frozen
            bullets.append(f"Frozen core: {core_note}; {correlated_elec} correlated electrons, {n_orb} orbitals")
        else:
            bullets.append(f"Frozen core: {core_note}")

    # Amplitude diagnostics (only when amplitude files were saved and parsed)
    if amp and amp.get("available"):
        t1 = amp.get("t1_diagnostic")
        d1 = amp.get("d1_diagnostic")
        t2_max = amp.get("t2_max_amplitude")
        t2_norm = amp.get("t2_frobenius_norm")
        t2_dom = amp.get("t2_dominance_fraction")
        sw = amp.get("t1_t2_singles_weight")
        tf = amp.get("triples_fraction")
        mr = amp.get("mr_assessment", "unknown")
        flags = amp.get("mr_flags", [])
        n_above_10 = amp.get("t2_count_above_010", 0)

        t1_str = f"T1={t1:.4f}" if t1 is not None else "T1=n/a"
        d1_str = f"D1={d1:.4f}" if d1 is not None else "D1=nosym_required"
        bullets.append(f"MR diagnostics: {t1_str}, {d1_str}")

        if t2_norm is not None:
            top = amp.get("t2_top_amplitudes", [])
            top_str = f"max={t2_max:.4f}" if t2_max is not None else ""
            dom_str = f", top-10 captures {t2_dom*100:.0f}% of ||T2||²" if t2_dom is not None else ""
            large_str = f", {n_above_10} amplitudes >0.10" if n_above_10 else ""
            bullets.append(f"T2 amplitudes: {top_str}{large_str}{dom_str}")
            if top:
                bullets.append(f"T2 top-10 magnitudes: {[round(v,4) for v in top[:10]]}")

        if sw is not None:
            bullets.append(
                f"Singles/doubles balance: {sw*100:.1f}% singles weight "
                f"({'dominant doubles — check for bond-breaking/diradical character' if sw < 0.05 else 'normal'})"
            )
        if tf is not None:
            bullets.append(
                f"Triples fraction: {tf*100:.1f}% of CCSD correlation "
                f"({'large — CC hierarchy may not converge' if tf > 0.15 else 'acceptable' if tf > 0.05 else 'small'})"
            )

        verdict_map = {
            "single_reference_ok": "single-reference OK — CCSD(T) results reliable",
            "moderate_mr_character": "moderate MR character — verify with MCSCF/CASSCF",
            "strong_mr_character": "strong MR character — CCSD likely unreliable",
            "unreliable_ccsd": "CCSD unreliable — use MCSCF/CASSCF instead",
        }
        bullets.append(f"MR verdict: {verdict_map.get(mr, mr)}")
        if flags:
            bullets.append(f"MR flags triggered: {', '.join(flags)}")
    elif amp and not amp.get("available"):
        bullets.append(
            "MR diagnostics: amplitude files not found — re-run with 'set tce:save_t T T' to enable"
        )

    # Next-step suggestion
    if not converged:
        bullets.append("TCE next step: increase maxiter or restart from existing amplitudes")
    elif method == "CCSD":
        bullets.append("TCE next step: consider CCSD(T) for perturbative triples correction")
    elif method == "MP2":
        bullets.append("TCE next step: consider CCSD or CCSD(T) for higher accuracy")
    else:
        bullets.append("TCE next step: calculation complete — verify freeze count and state before accepting")

    return bullets


def _build_state_check(
    diagnosis: "dict[str, Any]",
    output_path: str,
    input_path: "str | None",
    expected_somo_count: "int | None",
) -> "dict[str, Any]":
    """Build the spin/state check result from an already-computed diagnosis dict.

    Replicates check_spin_charge_state logic without re-calling diagnose_nwchem_output.
    """
    input_summary = diagnosis["input_summary"]
    state_check = diagnosis["state_check"]
    multiplicity = input_summary["multiplicity"] if input_summary else None
    charge = input_summary["charge"] if input_summary else None
    inferred_expected_somos = expected_somo_count
    if inferred_expected_somos is None and multiplicity is not None and multiplicity > 1:
        inferred_expected_somos = multiplicity - 1

    reasons: list[str] = []
    assessment = "unavailable"
    confidence = diagnosis["confidence"]

    if not state_check["available"]:
        if state_check["assessment"] != "unavailable":
            assessment = "suspicious"
            reasons.append(state_check["assessment"])
        else:
            reasons.append("frontier_state_analysis_unavailable")
    else:
        if multiplicity == 1 and state_check["somo_count"] > 0:
            assessment = "suspicious"
            reasons.append("singlet_input_but_open_shell_somos_found")
        elif inferred_expected_somos is not None and state_check["somo_count"] != inferred_expected_somos:
            assessment = "suspicious"
            reasons.append("somo_count_does_not_match_multiplicity_or_expected_state")
        elif state_check["assessment"] in {"metal_state_mismatch_suspected", "somo_count_mismatch"}:
            assessment = "suspicious"
            reasons.append(state_check["assessment"])
        else:
            assessment = "plausible"
            reasons.append("frontier_and_spin_signals_are_consistent_with_requested_state")

    dominant_site = (state_check.get("spin_density_summary") or {}).get("dominant_site")
    if dominant_site is not None:
        reasons.append(
            f"dominant_spin_density_on_{dominant_site['element']}{dominant_site['atom_index']}"
        )

    return {
        "output_file": output_path,
        "input_file": input_path,
        "assessment": assessment,
        "confidence": confidence,
        "charge": charge,
        "multiplicity": multiplicity,
        "expected_somo_count": inferred_expected_somos,
        "observed_somo_count": state_check.get("somo_count"),
        "metal_like_somo_count": state_check.get("metal_like_somo_count"),
        "ligand_like_somo_count": state_check.get("ligand_like_somo_count"),
        "state_check_assessment": state_check.get("assessment"),
        "dominant_spin_site": dominant_site,
        "reasons": reasons,
        "recommended_next_action": diagnosis["recommended_next_action"],
        "input_summary": input_summary,
    }


def summarize_nwchem_case(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    library_path: str | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
    err_file: str | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    # Lazy import to break circular dependency with api_input
    from .api_input import prepare_nwchem_next_step, lint_nwchem_input, find_restart_assets
    from chemtools.core.common import read_text

    # Read the output file once — reused by all downstream parsers to avoid redundant I/O
    output_contents = read_text(output_path)

    # Detect TCE early — drives what we skip and what we add below
    tce = _try_parse_tce(output_path, _contents=output_contents)
    is_tce = tce is not None

    # Try amplitude diagnostics (only present if save_t was set)
    tce_amp: dict[str, Any] | None = None
    if is_tce:
        try:
            from chemtools.programs.nwchem.parse.tce import parse_tce_amplitudes as _parse_amp
            tce_amp = _parse_amp(output_path)
        except Exception:
            tce_amp = None

    # detail_level="full" embeds the full diagnosis — lets us build the state check
    # inline without a second diagnose_nwchem_output call on the same file.
    summary = summarize_nwchem_output(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
        detail_level="full",
        err_file=err_file,
        _contents=output_contents,
    )
    next_step = prepare_nwchem_next_step(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
        output_dir=output_dir,
        base_name=base_name,
        write_files=False,
        _precomputed_summary=summary,
    )
    # Strip full input_text from prepared_artifacts — reduces response by 2-10 KB.
    # Callers who need the draft input should call the drafter tools directly with write_file=True.
    for artifact in next_step.get("prepared_artifacts", {}).values():
        artifact.pop("input_text", None)
        artifact.pop("plus_input_text", None)
        artifact.pop("minus_input_text", None)
    lint = lint_nwchem_input(input_path, library_path=library_path) if input_path else None
    assets = find_restart_assets(input_path or output_path)
    # Spin/state check is only meaningful for SCF/DFT runs where frontier MOs are printed.
    # Skip it for TCE — the correlated wavefunction doesn't expose frontier MOs the same way.
    # Build from the already-computed diagnosis to avoid a second full file parse.
    state: dict[str, Any] | None
    if is_tce:
        state = None
    else:
        diagnosis = summary["diagnosis"]
        state = _build_state_check(
            diagnosis=diagnosis,
            output_path=output_path,
            input_path=input_path,
            expected_somo_count=expected_somo_count,
        )

    bullets = list(summary["summary_bullets"])

    # TCE-specific bullets inserted right after the base summary
    if is_tce:
        bullets.extend(_tce_summary_bullets(tce, amp=tce_amp))

    if lint is not None:
        bullets.append(
            f"Input lint: {lint['status']} ({lint['counts']['error']} errors, {lint['counts']['warning']} warnings)"
        )
    if state is not None:
        bullets.append(f"Spin/state plausibility: {state['assessment']}")
    elif is_tce:
        bullets.append("Spin/state check: skipped for TCE (state is from SCF reference; verify freeze count separately)")
    preferred = assets["preferred"]
    if preferred.get("vectors_file"):
        bullets.append(f"Preferred restart vectors: {Path(preferred['vectors_file']).name}")
    if preferred.get("database_file"):
        bullets.append(f"Preferred restart database: {Path(preferred['database_file']).name}")
    bullets.append(f"Prepared workflow: {next_step['selected_workflow']}")

    summary_text = "\n".join(f"- {item}" for item in bullets)
    payload = {
        "output_file": output_path,
        "input_file": input_path,
        "summary_bullets": bullets,
        "summary_text": summary_text,
        "diagnosis_summary": summary,
        "tce": tce,
        "tce_amplitudes": tce_amp,
        "lint": lint,
        "restart_assets": assets,
        "spin_charge_state": state,
        "next_step": next_step,
    }
    if compact:
        return _build_compact_case_summary(payload)
    payload["compact_summary"] = _build_compact_case_summary(payload)
    return payload


def review_nwchem_case(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
    expected_somo_count: int | None = None,
    library_path: str | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
) -> dict[str, Any]:
    return summarize_nwchem_case(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
        library_path=library_path,
        output_dir=output_dir,
        base_name=base_name,
        compact=True,
    )


def review_nwchem_mcscf_case(
    output_path: str,
    input_path: str | None = None,
    expected_metal_elements: list[str] | None = None,
) -> dict[str, Any]:
    parsed = parse_mcscf_output(output_path)
    input_summary = inspect_nwchem_input(input_path) if input_path else None
    metal_elements = expected_metal_elements or (input_summary["transition_metals"] if input_summary else [])
    occupation_review = _review_mcscf_occupations(parsed)
    active_space_density = _review_mcscf_active_space_density(parsed, metal_elements)
    convergence_review = _review_mcscf_convergence(parsed)

    status = parsed["status"]
    failure_mode = parsed["failure_mode"]
    if status == "failed" and failure_mode == "input_parse_error":
        recommended_action = "fix_mcscf_block_syntax_before_retry"
        rationale = "The current MCSCF input failed before the wavefunction optimization started."
    elif status != "converged":
        recommended_action = "adjust_seed_levelshift_or_active_space_before_retry"
        rationale = "The MCSCF run did not reach a stable final state."
    elif occupation_review["assessment"] == "healthy_active_space":
        recommended_action = "use_mcscf_as_reference_or_seed_for_follow_up"
        rationale = "The active-space occupations show meaningful partial occupancy and the MCSCF run converged."
    elif occupation_review["assessment"] == "borderline_active_space":
        recommended_action = "inspect_active_space_edges_before_large_follow_up"
        rationale = "The MCSCF run converged, but at least one active orbital is effectively pinned and worth reviewing."
    else:
        recommended_action = "refine_active_space_then_rerun_mcscf"
        rationale = "The current active window looks too pinned to be an optimal long-term active space."

    bullets = [
        f"MCSCF status: {status}",
        f"Active space: CAS({parsed['settings']['active_electrons']},{parsed['settings']['active_orbitals']}) with multiplicity {parsed['settings']['multiplicity']}",
    ]
    if parsed["final_energy_hartree"] is not None:
        bullets.append(f"Final MCSCF energy: {parsed['final_energy_hartree']:.12f} Ha")
    bullets.append(convergence_review["summary"])
    if occupation_review["summary"]:
        bullets.append(occupation_review["summary"])
    if active_space_density["summary"]:
        bullets.append(active_space_density["summary"])
    bullets.append(f"Next action: {recommended_action}")

    return {
        "output_file": output_path,
        "input_file": input_path,
        "input_summary": input_summary,
        "status": status,
        "failure_mode": failure_mode,
        "summary_bullets": bullets,
        "summary_text": "\n".join(f"- {item}" for item in bullets),
        "settings": parsed["settings"],
        "convergence_review": convergence_review,
        "occupation_review": occupation_review,
        "active_space_density_review": active_space_density,
        "recommended_next_action": recommended_action,
        "rationale": rationale,
        "raw_mcscf": parsed,
    }


def _compact_tce(tce: "dict[str, Any] | None") -> "dict[str, Any] | None":
    """Return a trimmed TCE summary for the compact case payload."""
    if tce is None:
        return None
    section = tce["tce_sections"][0] if tce.get("tce_sections") else {}
    return {
        "method": tce.get("method"),
        "converged": tce.get("converged", False),
        "total_energy_hartree": tce.get("total_energy_hartree"),
        "correlation_energy_hartree": tce.get("correlation_energy_hartree"),
        "frozen_cores": section.get("frozen_cores"),
        "n_electrons": section.get("n_electrons"),
        "n_orbitals": section.get("n_orbitals"),
    }


def _build_compact_case_summary(payload: dict[str, Any]) -> dict[str, Any]:
    lint = payload.get("lint")
    assets = payload["restart_assets"]
    state = payload["spin_charge_state"]
    next_step = payload["next_step"]
    diagnosis = payload["diagnosis_summary"]["diagnosis"]
    preferred = assets["preferred"]

    return {
        "output_file": payload["output_file"],
        "input_file": payload.get("input_file"),
        "summary_text": payload["summary_text"],
        "summary_bullets": payload["summary_bullets"],
        "diagnosis": {
            "stage": diagnosis["stage"],
            "task_outcome": diagnosis["task_outcome"],
            "failure_class": diagnosis["failure_class"],
            "likely_cause": diagnosis["likely_cause"],
            "recommended_next_action": diagnosis["recommended_next_action"],
            "confidence": diagnosis["confidence"],
        },
        "lint": None
        if lint is None
        else {
            "status": lint["status"],
            "counts": lint["counts"],
            "top_issues": lint["issues"][:5],
        },
        "spin_charge_state": None
        if state is None
        else {
            "assessment": state["assessment"],
            "confidence": state["confidence"],
            "charge": state["charge"],
            "multiplicity": state["multiplicity"],
            "expected_somo_count": state["expected_somo_count"],
            "observed_somo_count": state["observed_somo_count"],
            "state_check_assessment": state["state_check_assessment"],
            "reasons": state["reasons"][:5],
            "recommended_next_action": state["recommended_next_action"],
        },
        "tce": _compact_tce(payload.get("tce")),
        "restart_assets": {
            "job_dir": assets["job_dir"],
            "focus_stem": assets["focus_stem"],
            "preferred": {
                key: value for key, value in preferred.items() if value is not None
            },
            "restart_candidates": assets["restart_candidates"][:6],
        },
        "next_step": {
            "selected_workflow": next_step["selected_workflow"],
            "can_auto_prepare": next_step["can_auto_prepare"],
            "artifact_order": next_step["artifact_order"],
            "prepared_artifact_summaries": next_step["prepared_artifact_summaries"],
            "notes": next_step["notes"],
        },
    }


def _review_mcscf_convergence(parsed: dict[str, Any]) -> dict[str, Any]:
    status = parsed["status"]
    iterations = parsed.get("iteration_count") or 0
    warnings = parsed.get("precondition_warning_count") or 0
    initial_level = parsed.get("settings", {}).get("initial_levelshift")
    final_level = parsed.get("final_levelshift")
    negative_curvatures = parsed.get("negative_curvature_count") or 0

    if status == "failed":
        assessment = "input_or_convergence_failure"
    elif status != "converged":
        assessment = "incomplete_mcscf_convergence"
    elif warnings >= 20 or negative_curvatures >= 3:
        assessment = "converged_with_stiff_orbital_optimization"
    elif warnings > 0:
        assessment = "converged_with_minor_preconditioning_warnings"
    else:
        assessment = "clean_mcscf_convergence"

    parts = [f"MCSCF macroiterations: {iterations}"]
    if warnings:
        parts.append(f"{warnings} precondition warnings")
    if initial_level is not None and final_level is not None:
        parts.append(f"level shift {initial_level:.2f} -> {final_level:.2f}")
    if negative_curvatures:
        parts.append(f"{negative_curvatures} negative-curvature events")

    return {
        "assessment": assessment,
        "iteration_count": iterations,
        "precondition_warning_count": warnings,
        "negative_curvature_count": negative_curvatures,
        "initial_levelshift": initial_level,
        "final_levelshift": final_level,
        "summary": "; ".join(parts),
    }


def _review_mcscf_occupations(parsed: dict[str, Any]) -> dict[str, Any]:
    natural = parsed.get("natural_occupations") or []
    settings = parsed.get("settings") or {}
    inactive_shells = settings.get("inactive_shells")
    active_orbitals = settings.get("active_orbitals")

    active_window: list[dict[str, Any]] = []
    if natural and inactive_shells is not None and active_orbitals:
        start = inactive_shells + 1
        end = inactive_shells + active_orbitals
        active_window = [item for item in natural if start <= item["orbital_index"] <= end]
        if len(active_window) < active_orbitals:
            active_window = natural[-active_orbitals:]
    elif natural and active_orbitals:
        active_window = natural[-active_orbitals:]

    if not active_window:
        return {
            "assessment": "occupations_unavailable",
            "active_window": [],
            "strongly_occupied_count": 0,
            "partially_occupied_count": 0,
            "near_empty_count": 0,
            "edge_candidates": {"occupied_side": [], "virtual_side": []},
            "summary": "Natural occupations were not available for active-space review.",
        }

    strongly_occupied = [item for item in active_window if (item["occupation"] or 0.0) >= 1.98]
    near_empty = [item for item in active_window if (item["occupation"] or 0.0) <= 0.02]
    partially_occupied = [item for item in active_window if item not in strongly_occupied and item not in near_empty]

    by_index = {item["orbital_index"]: item["occupation"] for item in natural}
    occupied_edge: list[dict[str, Any]] = []
    virtual_edge: list[dict[str, Any]] = []
    if inactive_shells is not None and active_orbitals:
        start = inactive_shells + 1
        end = inactive_shells + active_orbitals
        for orbital_index in range(max(1, start - 3), start):
            occupation = by_index.get(orbital_index)
            if occupation is not None and occupation < 1.98:
                occupied_edge.append({"orbital_index": orbital_index, "occupation": occupation})
        for orbital_index in range(end + 1, end + 4):
            occupation = by_index.get(orbital_index)
            if occupation is not None and occupation > 0.02:
                virtual_edge.append({"orbital_index": orbital_index, "occupation": occupation})

    if len(partially_occupied) >= max(3, (active_orbitals or len(active_window)) // 2) and not occupied_edge and not virtual_edge:
        assessment = "healthy_active_space"
    elif len(partially_occupied) >= 2:
        assessment = "borderline_active_space"
    else:
        assessment = "overly_pinned_active_space"

    summary = (
        f"Active occupations: {len(strongly_occupied)} near-2, "
        f"{len(partially_occupied)} fractional, {len(near_empty)} near-0"
    )
    if occupied_edge or virtual_edge:
        edge_notes = []
        if occupied_edge:
            edge_notes.append("occupied edge orbitals just below the active space are not fully pinned")
        if virtual_edge:
            edge_notes.append("virtual edge orbitals just above the active space are not fully empty")
        summary += "; " + "; ".join(edge_notes)

    return {
        "assessment": assessment,
        "active_window": active_window,
        "strongly_occupied_count": len(strongly_occupied),
        "partially_occupied_count": len(partially_occupied),
        "near_empty_count": len(near_empty),
        "edge_candidates": {
            "occupied_side": occupied_edge,
            "virtual_side": virtual_edge,
        },
        "swap_out_candidates": [item["orbital_index"] for item in strongly_occupied + near_empty],
        "summary": summary,
    }


def _review_mcscf_active_space_density(parsed: dict[str, Any], metal_elements: list[str]) -> dict[str, Any]:
    active_density = parsed.get("active_space_mulliken") or {}
    atoms = active_density.get("atoms") or []
    if not atoms:
        return {
            "assessment": "active_space_density_unavailable",
            "metal_fraction": None,
            "dominant_atoms": [],
            "summary": None,
        }

    total_population = sum(item.get("charge") or 0.0 for item in atoms)
    dominant_atoms = sorted(atoms, key=lambda item: item.get("charge") or 0.0, reverse=True)
    metal_set = {element.lower() for element in metal_elements}
    metal_population = sum((item.get("charge") or 0.0) for item in atoms if item["element"].lower() in metal_set)
    metal_fraction = (metal_population / total_population) if total_population else None

    if metal_fraction is None:
        assessment = "active_space_density_unavailable"
    elif metal_fraction >= 0.45:
        assessment = "metal_participation_significant"
    elif metal_fraction >= 0.2:
        assessment = "mixed_metal_ligand_active_space"
    else:
        assessment = "ligand_dominated_active_space"

    top_labels = ", ".join(
        f"{item['element']}{item['atom_index']} {item['charge']:.2f}" for item in dominant_atoms[:3]
    )
    summary = (
        f"Active-space Mulliken density: top contributors {top_labels}"
        + (f"; metal fraction {metal_fraction:.2f}" if metal_fraction is not None else "")
    )
    return {
        "assessment": assessment,
        "metal_fraction": metal_fraction,
        "dominant_atoms": dominant_atoms[:6],
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Geometry plausibility checker
# ---------------------------------------------------------------------------

# Typical max coordination numbers per element (above this → red flag)
_MAX_COORD: dict[str, int] = {
    "H": 2, "He": 0,
    "Li": 4, "Be": 4, "B": 4, "C": 4, "N": 4, "O": 3, "F": 1, "Ne": 0,
    "Na": 6, "Mg": 6, "Al": 6, "Si": 6, "P": 6, "S": 6, "Cl": 1, "Ar": 0,
    "K": 8, "Ca": 8, "Ga": 6, "Ge": 4, "As": 6, "Se": 6, "Br": 1, "Kr": 0,
    "Rb": 8, "Sr": 8, "In": 6, "Sn": 6, "Sb": 6, "Te": 6, "I": 1, "Xe": 0,
    "Cs": 12, "Ba": 12, "Tl": 6, "Pb": 6, "Bi": 6,
}
# Transition metals: typical max CN = 9
for _tm in _TRANSITION_METALS:
    if _tm not in _MAX_COORD:
        _MAX_COORD[_tm] = 9

# Lanthanides and actinides: high-CN chemistry (up to 12–14)
_LANTHANIDES = {"La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"}
_ACTINIDES = {"Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"}
for _hm in _LANTHANIDES | _ACTINIDES:
    if _hm not in _MAX_COORD:
        _MAX_COORD[_hm] = 14

# All elements that behave like metals (used for coordination reporting)
_ALL_METALS = _TRANSITION_METALS | _LANTHANIDES | _ACTINIDES | {
    "Li","Na","K","Rb","Cs","Be","Mg","Ca","Sr","Ba",
    "Al","Ga","In","Tl","Sn","Pb","Bi",
}

# Typical CN ranges for main-group elements  {element: (min_ok, max_ok)}
_TYPICAL_COORD: dict[str, tuple[int, int]] = {
    "H": (1, 1), "He": (0, 0),
    "B": (2, 4), "C": (1, 4), "N": (1, 4), "O": (1, 2), "F": (1, 1),
    "Si": (2, 6), "P": (1, 6), "S": (1, 6), "Cl": (1, 1),
    "Ge": (2, 4), "As": (2, 6), "Se": (1, 6), "Br": (1, 1),
    "Sn": (2, 6), "Sb": (2, 6), "Te": (1, 6), "I": (1, 1),
    "Pb": (2, 6), "Bi": (2, 6),
}


def _d_count_for_ox(element: str, oxidation_state: int) -> int | None:
    tm = _TM_Z_CORE.get(element)
    if tm is None:
        return None
    z, core = tm
    d = z - core - oxidation_state
    return d if 0 <= d <= 10 else None


def suggest_spin_state(
    elements: list[str],
    charge: int = 0,
    metal_oxidation_states: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Suggest likely spin multiplicities for a molecule given elements and charge.

    For transition-metal systems this computes d-electron counts and returns
    high-spin (Hund) and low-spin (strong-field octahedral) multiplicity
    candidates with plain-language explanations.

    Args:
        elements: All element symbols in the molecule (duplicates OK, e.g. ['Fe', 'Cl', 'Cl']).
        charge: Total molecular charge.
        metal_oxidation_states: Optional dict mapping metal symbol to formal oxidation state,
            e.g. {'Fe': 2}.  When omitted, common oxidation states for each metal are enumerated.

    Returns dict with 'recommended_multiplicity', 'metal_analyses', and 'summary'.
    """
    norm = [e[0].upper() + e[1:].lower() for e in elements]
    unique_elements = list(dict.fromkeys(norm))
    metal_elements = [e for e in unique_elements if e in _TRANSITION_METALS]
    ligand_elements = [e for e in unique_elements if e not in _TRANSITION_METALS]

    has_strong_field = any(e in _STRONG_FIELD_ELEMENTS for e in ligand_elements)
    has_weak_field = any(e in _WEAK_FIELD_ELEMENTS for e in ligand_elements)

    metal_analyses: list[dict[str, Any]] = []

    for metal in metal_elements:
        if metal_oxidation_states and metal in metal_oxidation_states:
            ox_list = [metal_oxidation_states[metal]]
            ox_source = "provided"
        else:
            ox_list = _TM_COMMON_OX.get(metal, [2, 3])
            ox_source = "common_states"

        ox_analyses = []
        for ox in ox_list:
            d = _d_count_for_ox(metal, ox)
            if d is None:
                continue
            hs_u = _D_HS_UNPAIRED[d]
            ls_u = _D_LS_UNPAIRED[d]
            hs_mult = hs_u + 1
            ls_mult = ls_u + 1
            spin_states = [{"spin_state": "high-spin", "multiplicity": hs_mult, "unpaired": hs_u, "d_count": d}]
            if ls_mult != hs_mult:
                spin_states.append({"spin_state": "low-spin", "multiplicity": ls_mult, "unpaired": ls_u, "d_count": d})

            if len(spin_states) == 1:
                rec_idx, rec_reason = 0, "only one possible spin state for d%d" % d
            elif has_strong_field and not has_weak_field:
                rec_idx, rec_reason = 1, "strong-field ligands (C/N/P donors) favor low-spin"
            elif has_weak_field and not has_strong_field:
                rec_idx, rec_reason = 0, "weak-field ligands (halide/chalcogenide) favor high-spin"
            else:
                rec_idx, rec_reason = 0, "defaulting to high-spin (Hund's rule); verify ligand field"

            ox_analyses.append({
                "oxidation_state": ox,
                "oxidation_state_source": ox_source,
                "d_count": d,
                "spin_states": spin_states,
                "recommended_spin_state": spin_states[rec_idx]["spin_state"],
                "recommended_multiplicity": spin_states[rec_idx]["multiplicity"],
                "recommendation_reason": rec_reason,
            })

        metal_analyses.append({"element": metal, "oxidation_state_analyses": ox_analyses})

    # Non-TM case: infer from total electron count
    if not metal_elements:
        total_e = sum(ELEMENT_TO_Z.get(e, 0) for e in norm) - charge
        rec_mult = 2 if total_e % 2 == 1 else 1
        return {
            "elements": unique_elements,
            "charge": charge,
            "has_transition_metals": False,
            "total_electrons": total_e,
            "recommended_multiplicity": rec_mult,
            "metal_analyses": [],
            "summary": (
                f"No transition metals. Total electrons: {total_e}. "
                f"Recommended multiplicity: {rec_mult} "
                f"({'doublet' if rec_mult == 2 else 'singlet'})."
            ),
        }

    # Derive overall recommendation from first metal / first (most common) ox state
    rec_mult: int | None = None
    rec_spin: str | None = None
    if metal_analyses and metal_analyses[0]["oxidation_state_analyses"]:
        first_ox = metal_analyses[0]["oxidation_state_analyses"][0]
        rec_mult = first_ox["recommended_multiplicity"]
        rec_spin = first_ox["recommended_spin_state"]

    summary_lines = []
    for ma in metal_analyses:
        for oa in ma["oxidation_state_analyses"]:
            mults = ", ".join(f"{s['spin_state']}=mult{s['multiplicity']}" for s in oa["spin_states"])
            summary_lines.append(
                f"{ma['element']}({oa['oxidation_state']:+d}) d{oa['d_count']}: {mults}"
                f" → recommended {oa['recommended_spin_state']} (mult={oa['recommended_multiplicity']},"
                f" nopen={oa['recommended_multiplicity'] - 1})"
            )

    return {
        "elements": unique_elements,
        "charge": charge,
        "has_transition_metals": True,
        "metal_elements": metal_elements,
        "ligand_elements": ligand_elements,
        "ligand_field_hints": {"has_strong_field": has_strong_field, "has_weak_field": has_weak_field},
        "metal_analyses": metal_analyses,
        "recommended_multiplicity": rec_mult,
        "recommended_spin_state": rec_spin,
        "recommended_nopen": (rec_mult - 1) if rec_mult is not None else None,
        "summary": "\n".join(summary_lines) if summary_lines else "No analysis available.",
    }


# ---------------------------------------------------------------------------
# Basis set advisor
# ---------------------------------------------------------------------------

def suggest_basis_set(
    elements: list[str],
    purpose: str = "geometry",
    library_path: str | None = None,
) -> dict[str, Any]:
    """Suggest an appropriate basis set (and ECP when needed) for a molecule.

    Args:
        elements: Element symbols present in the molecule.
        purpose: One of "geometry" (fast opt), "single_point" (DFT energy),
                 "correlated" (MP2/CCSD), or "heavy_elements" (post-Kr metals).
        library_path: Optional path to basis library (used only for validation note).

    Returns dict with 'basis_assignments', 'ecp_assignments', 'recommended_basis',
    and 'notes' ready to pass to create_nwchem_input.
    """
    norm = list(dict.fromkeys(e[0].upper() + e[1:].lower() for e in elements))
    heavy = [e for e in norm if ELEMENT_TO_Z.get(e, 0) > 36]
    has_heavy = bool(heavy)
    has_tm = any(e in _TRANSITION_METALS for e in norm)
    has_lanthanides = any(57 <= ELEMENT_TO_Z.get(e, 0) <= 71 for e in norm)

    p = purpose.strip().lower()

    if p in ("geometry", "opt", "optimization"):
        basis = "def2-svp"
        ecp = "def2-ecp" if has_heavy else None
        explanation = (
            "def2-SVP for geometry optimization — balanced speed and accuracy. "
            + ("def2-ECP applied to heavy elements (Z>36). " if has_heavy else "")
        )
        alternatives = ["def2-tzvp", "6-31gs"]
    elif p in ("single_point", "sp", "energy", "dft"):
        basis = "def2-tzvp"
        ecp = "def2-ecp" if has_heavy else None
        explanation = (
            "def2-TZVP for production DFT single-point energies. "
            + ("def2-ECP for heavy elements (Z>36). " if has_heavy else "")
        )
        alternatives = ["def2-svp", "cc-pvtz"]
    elif p in ("correlated", "ccsd", "mp2", "post-hf", "wft"):
        if has_heavy or has_tm:
            basis = "def2-tzvp"
            ecp = "def2-ecp" if has_heavy else None
            explanation = (
                "def2-TZVP for correlated calculations with transition metals. "
                + ("def2-ECP for heavy elements. " if has_heavy else "")
                + "For pure main-group systems, cc-pVTZ is preferred."
            )
            alternatives = ["cc-pvtz", "def2-svp"]
        else:
            basis = "cc-pvtz"
            ecp = None
            explanation = (
                "cc-pVTZ for correlated methods (MP2, CCSD, CCSD(T)) on main-group elements. "
                "Designed for systematic basis-set convergence."
            )
            alternatives = ["cc-pvdz", "aug-cc-pvtz", "def2-tzvp"]
    elif p in ("heavy", "heavy_elements", "lanthanides", "actinides"):
        basis = "def2-tzvp"
        ecp = "def2-ecp"
        explanation = "def2-TZVP + Stuttgart def2-ECP for relativistic treatment of heavy elements."
        if has_lanthanides:
            explanation += " Note: lanthanides may need dedicated f-basis (e.g. ano-rcc or cc-pVTZ-PP)."
        alternatives = ["def2-svp", "crenbl"]
    else:
        basis = "def2-svp"
        ecp = "def2-ecp" if has_heavy else None
        explanation = f"Unknown purpose '{purpose}'; defaulting to def2-SVP."
        alternatives = ["def2-tzvp"]

    basis_assignments = {e: basis for e in norm}
    ecp_assignments: dict[str, str] | None = None
    if ecp:
        ecp_assignments = {e: ecp for e in heavy}
        if not ecp_assignments:
            ecp_assignments = None

    return {
        "elements": norm,
        "purpose": p,
        "has_heavy_elements": has_heavy,
        "has_transition_metals": has_tm,
        "recommended_basis": basis,
        "recommended_ecp": ecp,
        "explanation": explanation.strip(),
        "alternatives": alternatives,
        "basis_assignments": basis_assignments,
        "ecp_assignments": ecp_assignments,
        "usage_note": (
            "Pass basis_assignments (and ecp_assignments if not None) directly to "
            "create_nwchem_input or create_nwchem_dft_workflow_input."
        ),
    }


# ---------------------------------------------------------------------------
# Memory advisor
# ---------------------------------------------------------------------------

_BASIS_SCALE: dict[str, float] = {
    "sto-3g": 0.3, "sto": 0.3,
    "3-21g": 0.5, "3-21": 0.5,
    "6-31g": 1.0, "6-31gs": 1.0, "6-31gss": 1.2, "6-311g": 1.5,
    "svp": 1.0, "def2-svp": 1.0, "def2-svpp": 1.2,
    "tzvp": 2.5, "def2-tzvp": 2.5, "def2-tzvpp": 3.0,
    "qzvp": 6.0, "def2-qzvp": 6.0,
    # Dunning correlation-consistent families
    "pvdz": 1.0, "cc-pvdz": 1.0, "aug-cc-pvdz": 1.4,
    "pvtz": 2.5, "cc-pvtz": 2.5, "aug-cc-pvtz": 3.5,
    "pvqz": 6.0, "cc-pvqz": 6.0, "aug-cc-pvqz": 8.0,
    "pv5z": 12.0, "cc-pv5z": 12.0,
    # Douglas-Kroll Dunning (same size as base set)
    "pvdz-dk": 1.0, "cc-pvdz-dk": 1.0,
    "pvtz-dk": 2.5, "cc-pvtz-dk": 2.5,
    "pvqz-dk": 6.0, "cc-pvqz-dk": 6.0,
    # Segmented DK (Stuttgart)
    "dhf-svp": 1.0, "dhf-tzvp": 2.5, "dhf-tzvpp": 3.0,
    # ANO families
    "ano-rcc": 3.0, "ano-r": 2.5,
    # Pople diffuse
    "6-31+g": 1.2, "6-31++g": 1.4, "6-311+g": 1.8,
}


def suggest_relativistic_correction(
    elements: list[str],
    basis_assignments: dict[str, str] | None = None,
    ecp_assignments: dict[str, str] | None = None,
    purpose: str = "dft",
) -> dict[str, Any]:
    """Advise on relativistic corrections for a molecular calculation.

    Returns a recommendation (or "none needed") with the NWChem block to add,
    and compatibility warnings when ECPs are present.

    Parameters
    ----------
    elements:
        All element symbols in the system.
    basis_assignments:
        Dict mapping element → basis name.  Used to detect DK-type bases.
    ecp_assignments:
        Dict mapping element → ECP name.  If present, warns about X2C/DKH incompatibility.
    purpose:
        One of "dft", "scf", "ccsd", "property".  Affects recommendation.

    Returns dict with ``recommended_method``, ``nwchem_block``, ``reason``,
    ``warnings``, and ``per_element_z_scores``.
    """
    norm = [e[0].upper() + e[1:].lower() for e in elements]
    unique = list(dict.fromkeys(norm))

    has_ecp = bool(ecp_assignments)
    ecp_elements = list((ecp_assignments or {}).keys())

    # Z-based analysis
    per_element: list[dict[str, Any]] = []
    max_z = 0
    for el in unique:
        z = ELEMENT_TO_Z.get(el, 0)
        max_z = max(max_z, z)
        if z >= _REL_CRITICAL_Z:
            level = "critical"
        elif z >= _REL_IMPORTANT_Z:
            level = "important"
        elif z >= _REL_SIGNIFICANT_Z:
            level = "significant"
        elif z >= 18:
            level = "minor"
        else:
            level = "negligible"
        per_element.append({"element": el, "Z": z, "relativistic_importance": level})

    has_critical = any(p["relativistic_importance"] == "critical" for p in per_element)
    has_important = any(p["relativistic_importance"] == "important" for p in per_element)
    has_significant = any(p["relativistic_importance"] == "significant" for p in per_element)

    # Detect DK basis sets
    basis_lower = {k.lower(): v.lower() for k, v in (basis_assignments or {}).items()}
    has_dk_basis = any(
        any(b.startswith(dk) or b in _DK_BASIS_PATTERNS for dk in _DK_BASIS_PATTERNS)
        for b in basis_lower.values()
    ) if basis_lower else False

    warnings: list[str] = []
    incompatible_elements: list[str] = []

    if has_ecp:
        # X2C/DKH treat core relativistic effects via all-electron; ECP replaces the core.
        # Using both is either redundant (same element) or inconsistent.
        incompatible_elements = [
            el for el in ecp_elements
            if el in unique
        ]
        if incompatible_elements:
            warnings.append(
                f"INCOMPATIBILITY: Elements {incompatible_elements} use ECPs — "
                "X2C and DKH are all-electron methods that replace ECPs. "
                "You must choose ONE: (a) all-electron + relativistic block, OR "
                "(b) ECP (removes core electrons; no relativistic block needed). "
                "Using both for the same element is incorrect."
            )

    # Detect Pople-style basis sets — they use SP shells, incompatible with X2C/DKH
    _POPLE_PATTERNS = ("sto-", "3-21g", "6-21g", "4-31g", "6-31g", "6-311g", "6-31+g", "6-311+g")
    pople_elements: list[str] = []
    for el, bname in (basis_assignments or {}).items():
        bname_lower = bname.lower().replace(" ", "")
        if any(bname_lower.startswith(p) or p in bname_lower for p in _POPLE_PATTERNS):
            pople_elements.append(el)
    has_pople = bool(pople_elements)

    # Recommendation logic
    if not (has_critical or has_important or has_significant):
        recommended = "none"
        nwchem_block = None
        reason = (
            f"All elements have Z < {_REL_SIGNIFICANT_Z} — relativistic effects are negligible. "
            "No relativistic block needed."
        )
    elif has_ecp and incompatible_elements:
        # ECPs already implicitly encode relativistic effects for heavy atoms
        recommended = "ecp_implicit"
        nwchem_block = None
        reason = (
            "ECP is in use for heavy elements — the ECP implicitly accounts for scalar "
            "relativistic effects on the core. Do not add a relativistic block for ECP-covered elements. "
            "If you want explicit all-electron relativistic treatment, remove the ECP and use "
            "an all-electron basis with X2C or DKH2."
        )
    elif has_critical:
        recommended = "x2c"
        nwchem_block = _REL_METHODS["x2c"]["nwchem_block"]
        reason = (
            f"Heavy element(s) with Z ≥ {_REL_CRITICAL_Z} present (5d metals or heavy p-block). "
            "Relativistic effects are chemically critical — X2C is the recommended method. "
            "Pair with cc-pVTZ-DK, cc-pVDZ-DK, or x2c-TZVPall basis sets."
        )
        if has_pople:
            warnings.append(
                f"INCOMPATIBILITY: Pople-style basis detected for {pople_elements}. "
                "6-31G* / 6-311G** and similar Pople bases use SP-contracted shells, which are "
                "incompatible with X2C/DKH. NWChem will crash with 'dimensions not the same' "
                "during the relativistic uncontraction step. "
                "Replace with cc-pVDZ-DK, cc-pVTZ-DK, or def2-SVP / def2-TZVP."
            )
        elif not has_dk_basis:
            warnings.append(
                "BASIS WARNING: No DK-quality basis detected. "
                "X2C/DKH calculations require bases designed for relativistic calculations "
                "(cc-pVDZ-DK, cc-pVTZ-DK, x2c-SVPall, etc.). "
                "Standard def2 bases are acceptable; avoid Pople bases (SP-shell incompatibility)."
            )
    elif has_important:
        recommended = "x2c"
        nwchem_block = _REL_METHODS["x2c"]["nwchem_block"]
        reason = (
            f"Element(s) with Z ≥ {_REL_IMPORTANT_Z} (4d metals / heavy main-group) present. "
            "Scalar relativistic effects are important for accurate energetics. "
            "X2C with DK basis sets recommended."
        )
        if has_pople:
            warnings.append(
                f"INCOMPATIBILITY: Pople-style basis detected for {pople_elements}. "
                "6-31G* / 6-311G** and similar Pople bases use SP-contracted shells, which are "
                "incompatible with X2C/DKH. NWChem will crash with 'dimensions not the same'. "
                "Replace with cc-pVDZ-DK, cc-pVTZ-DK, or def2-SVP / def2-TZVP."
            )
        elif not has_dk_basis:
            warnings.append(
                "BASIS WARNING: Consider switching to cc-pVDZ-DK or cc-pVTZ-DK basis sets."
            )
    else:
        # Z >= _REL_SIGNIFICANT_Z (3d TMs): recommend X2C when DK basis present, optional otherwise
        if has_dk_basis:
            recommended = "x2c"
            nwchem_block = _REL_METHODS["x2c"]["nwchem_block"]
            reason = (
                f"DK-type basis set detected with element(s) in the 3d/4d transition metal range. "
                "DK-family bases are designed for use with relativistic Hamiltonians (X2C or DKH2). "
                "X2C is strongly recommended — using a DK basis without a relativistic block "
                "gives inconsistent results."
            )
        else:
            recommended = "x2c_optional"
            nwchem_block = _REL_METHODS["x2c"]["nwchem_block"]
            reason = (
                f"Element(s) with Z ≥ {_REL_SIGNIFICANT_Z} present — scalar relativistic effects "
                "are non-negligible but often acceptable without correction at this level. "
                "Add X2C with a DK-type basis if targeting high accuracy."
            )
        if has_pople and recommended in ("x2c", "x2c_optional"):
            warnings.append(
                f"INCOMPATIBILITY: Pople-style basis detected for {pople_elements}. "
                "6-31G* / 6-311G** and similar Pople bases use SP-contracted shells, which are "
                "incompatible with X2C/DKH. NWChem will crash with 'dimensions not the same'. "
                "Replace with cc-pVDZ, cc-pVTZ, def2-SVP, or def2-TZVP."
            )

    # Performance note for X2C + SAD
    sad_note: str | None = None
    if recommended in ("x2c", "x2c_optional") and nwchem_block:
        heavy_tms = [p["element"] for p in per_element if p["Z"] >= _REL_SIGNIFICANT_Z]
        if heavy_tms:
            sad_note = (
                f"PERFORMANCE NOTE: X2C requires solving relativistic atomic SCFs for {heavy_tms} "
                "during the SAD initial guess. This runs with no output for potentially 30–120+ minutes. "
                "This is expected behavior — do not terminate the job during this phase."
            )
            warnings.append(sad_note)

    return {
        "recommended_method": recommended,
        "nwchem_block": nwchem_block,
        "reason": reason,
        "per_element": per_element,
        "max_z": max_z,
        "has_dk_basis": has_dk_basis,
        "has_ecp": has_ecp,
        "ecp_incompatible_elements": incompatible_elements,
        "has_pople_basis": has_pople,
        "pople_basis_elements": pople_elements,
        "available_methods": {k: {
            "nwchem_block": v["nwchem_block"],
            "description": v["description"],
            "cost": v["cost"],
        } for k, v in _REL_METHODS.items()},
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Frequency restart helper
# ---------------------------------------------------------------------------

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
    from chemtools.core.common import read_text
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
    from .api_input import lint_nwchem_input
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
    import subprocess
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
    import subprocess

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


# HPC resource advisors moved to programs/nwchem/strategy/hpc_resources.py.
from chemtools.programs.nwchem.strategy.hpc_resources import (  # noqa: F401, E402
    detect_hpc_accounts,
    suggest_hpc_resources,
    suggest_partition,
)


# SCF / state recovery advisors moved to programs/nwchem/strategy/recovery.py.
from chemtools.programs.nwchem.strategy.recovery import (  # noqa: F401, E402
    suggest_nwchem_scf_fix_strategy,
    suggest_nwchem_state_recovery_strategy,
)


# MCSCF active-space advisor moved to programs/nwchem/strategy/mcscf_active_space.py.
from chemtools.programs.nwchem.strategy.mcscf_active_space import (  # noqa: F401, E402
    suggest_nwchem_mcscf_active_space,
)


# Resource sizing advisors moved to programs/nwchem/strategy/resources.py.
from chemtools.programs.nwchem.strategy.resources import (  # noqa: F401, E402
    suggest_resources,
    suggest_memory,
    check_memory_fit,
    estimate_freq_walltime,
)
# Helpers still referenced by chemtools/api_strategy and by
# chemtools/programs/nwchem/strategy/hpc_resources.py via lazy import:
from chemtools.programs.nwchem.strategy.resources import (  # noqa: F401, E402
    _analyze_job_size,
    _basis_scale,
)


# Plausibility checks moved to programs/nwchem/strategy/plausibility.py.
from chemtools.programs.nwchem.strategy.plausibility import (  # noqa: F401, E402
    check_nwchem_geometry_plausibility,
    check_nwchem_freq_plausibility,
)

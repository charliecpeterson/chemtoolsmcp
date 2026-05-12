"""NWChem case review and state inspection.

A connected family of analyzers that an agent strings together to
understand what an NWChem run actually produced:

  * check_spin_charge_state              Quick state-quality check —
                                          does the SCF state match the
                                          requested charge / multiplicity?
  * summarize_nwchem_case                Compact human-friendly summary
                                          of a run (status, energy,
                                          warnings).
  * review_nwchem_case                   Detailed review combining
                                          diagnose + state check + freq /
                                          population.
  * review_nwchem_mcscf_case             MCSCF-specific case review:
                                          parse the MCSCF output, check
                                          convergence + occupations +
                                          active-space density.

Plus several internal helpers (_try_parse_tce, _tce_summary_bullets,
_build_state_check, _compact_tce, _build_compact_case_summary, the
_review_mcscf_* trio).
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Any

from chemtools.core.common import read_text
from chemtools.programs.nwchem.parse.input import inspect_nwchem_input
from chemtools.programs.nwchem.parse.mos import parse_mcscf_output
from chemtools.programs.nwchem.parse.tce import (
    parse_tce_output,
    parse_tce_amplitudes,
)
from chemtools.programs.nwchem.strategy.diagnose import (
    diagnose_nwchem_output,
    summarize_nwchem_output,
)
from chemtools.programs.nwchem.input._utils import _TRANSITION_METALS


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
    from chemtools.api_input import prepare_nwchem_next_step, lint_nwchem_input, find_restart_assets
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


__all__ = [
    "check_spin_charge_state",
    "summarize_nwchem_case",
    "review_nwchem_case",
    "review_nwchem_mcscf_case",
]

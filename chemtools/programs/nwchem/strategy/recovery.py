"""NWChem SCF / state-recovery strategy advisors.

Two entry points that return thick recovery strategy payloads for
common SCF convergence failures:

  * suggest_nwchem_scf_fix_strategy        For SCF that did not
                                            converge (DIIS divergence,
                                            level-shift drift, etc.).
                                            Returns a strategy entry
                                            list with concrete next-tool
                                            recommendations (damping,
                                            smearing, vectors swap).

  * suggest_nwchem_state_recovery_strategy For SCF that converged but to
                                            a suspicious / wrong state
                                            (spin contamination, ligand-
                                            centered SOMOs in a metal
                                            complex, etc.). Routes
                                            through vectors swap or
                                            fragment guess depending on
                                            the failure pattern.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

from chemtools.programs.nwchem.parse.input import inspect_nwchem_input
from chemtools.programs.nwchem.strategy.diagnose import diagnose_nwchem_output
from chemtools.programs.nwchem.input._utils import _strategy_entry


def suggest_nwchem_scf_fix_strategy(
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
    scf = diagnosis.get("scf") or {}
    last_run = scf.get("last_run") or {}
    trend = ((last_run.get("trend") or {}).get("pattern")) or ((scf.get("trend") or {}).get("pattern")) or "unknown"
    iteration_count = last_run.get("iteration_count") or scf.get("iteration_count") or 0
    hit_max = bool(last_run.get("hit_max_iterations") or scf.get("hit_max_iterations"))
    failure_class = diagnosis.get("failure_class")
    state_check = diagnosis.get("state_check") or {}

    strategies: list[dict[str, Any]] = []
    notes: list[str] = []
    strategy_family = "review_only"

    if failure_class == "no_clear_failure_detected" and scf.get("status") == "success":
        strategy_family = "no_scf_fix_needed"
        strategies.append(
            _strategy_entry(
                name="no_scf_fix_needed",
                priority=1,
                rationale="The SCF portion is converged and no SCF-specific failure was detected.",
                tool="review_nwchem_case",
                docs_topics=["scf_open_shell"],
                when_to_use="Use this when the run completed and the remaining question is about state quality or chemistry, not SCF rescue.",
            )
        )
    elif failure_class == "wrong_state_convergence":
        strategy_family = "state_recovery_over_scf_tuning"
        notes.append("do_not_just_add_iterations_for_wrong_state_convergence")
        strategies.extend(
            [
                _strategy_entry(
                    name="vectors_swap_restart",
                    priority=1,
                    rationale="The SCF converged, but it converged to the wrong state; occupation steering is more appropriate than generic SCF damping.",
                    tool="prepare_nwchem_next_step",
                    docs_topics=["scf_open_shell"],
                    when_to_use="Use when SOMO count is numerically right but SOMO character is wrong.",
                ),
                _strategy_entry(
                    name="fragment_guess_seed",
                    priority=2,
                    rationale="A different initial guess can move the calculation into a different electronic basin when swap restarts do not redirect the state.",
                    tool="suggest_nwchem_state_recovery_strategy",
                    docs_topics=["fragment_guess", "scf_open_shell"],
                    when_to_use="Use when repeated swap restarts keep returning to the same suspicious state.",
                ),
                _strategy_entry(
                    name="mcscf_seed_or_validation",
                    priority=3,
                    rationale="If DFT keeps returning to the same basin, a multiconfigurational reference can test whether a metal-centered state exists nearby.",
                    tool="suggest_nwchem_state_recovery_strategy",
                    docs_topics=["mcscf"],
                    when_to_use="Use for transition-metal wrong-state cases where Fe/Co/etc. d-manifold character is important.",
                ),
            ]
        )
    elif failure_class == "scf_nonconvergence":
        strategy_family = "scf_recovery"
        if trend == "oscillatory":
            strategies.extend(
                [
                    _strategy_entry(
                        name="damp_and_smear_restart",
                        priority=1,
                        rationale=f"SCF is oscillatory after {iteration_count} iterations, so damping/smearing is more appropriate than blindly increasing maxiter.",
                        tool="draft_nwchem_scf_stabilization_input",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use when energies or densities bounce rather than trend steadily.",
                    ),
                    _strategy_entry(
                        name="different_guess_source",
                        priority=2,
                        rationale="Oscillatory open-shell runs often need a different orbital guess, not just stronger DIIS stabilization.",
                        tool="suggest_nwchem_state_recovery_strategy",
                        docs_topics=["fragment_guess", "mcscf"],
                        when_to_use="Use when oscillation persists after one stabilization-style retry.",
                    ),
                ]
            )
        elif trend == "stalled":
            strategies.extend(
                [
                    _strategy_entry(
                        name="stabilization_restart",
                        priority=1,
                        rationale=f"SCF appears stalled after {iteration_count} iterations, so a restart with damping/ncydp/smearing is the first conservative fix.",
                        tool="draft_nwchem_scf_stabilization_input",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use when errors flatten without reaching threshold.",
                    ),
                    _strategy_entry(
                        name="change_guess_or_state_model",
                        priority=2,
                        rationale="A stalled open-shell run may indicate the current guess or state model is poor, especially for transition-metal chemistry.",
                        tool="suggest_nwchem_state_recovery_strategy",
                        docs_topics=["fragment_guess", "mcscf"],
                        when_to_use="Use when one stabilization retry does not materially change the trend.",
                    ),
                ]
            )
        elif trend in {"slow_improving", "nearly_converged"} and hit_max:
            strategies.extend(
                [
                    _strategy_entry(
                        name="gentle_iteration_extension",
                        priority=1,
                        rationale=f"SCF was still improving near the iteration limit ({iteration_count} iterations), so a modest extension is justified.",
                        tool="draft_nwchem_scf_stabilization_input",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use only when the SCF pattern is monotonic or nearly converged, not oscillatory.",
                    ),
                    _strategy_entry(
                        name="light_restart_from_latest_vectors",
                        priority=2,
                        rationale="Restarting from the latest vectors can finish a nearly converged SCF more cleanly than rerunning from scratch.",
                        tool="draft_nwchem_scf_stabilization_input",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use when the max-iteration stop happened late and the density/error trend is already small.",
                    ),
                ]
            )
        else:
            strategies.extend(
                [
                    _strategy_entry(
                        name="stabilization_restart",
                        priority=1,
                        rationale="A conservative SCF stabilization restart is the safest first step when the failure pattern is not yet well classified.",
                        tool="draft_nwchem_scf_stabilization_input",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use as the generic first retry for SCF nonconvergence.",
                    ),
                    _strategy_entry(
                        name="review_open_shell_syntax_and_guess",
                        priority=2,
                        rationale="Early or low-information SCF failures can come from state specification or an unsuitable guess rather than ordinary DIIS instability.",
                        tool="review_nwchem_case",
                        docs_topics=["scf_open_shell", "fragment_guess"],
                        when_to_use="Use when there are too few iterations to trust a trend classification.",
                    ),
                ]
            )
        if state_check.get("assessment") in {"metal_state_mismatch_suspected", "somo_count_mismatch"}:
            notes.append("state_signals_also_suggest_guess_or_state_problem")
    else:
        strategies.append(
            _strategy_entry(
                name="manual_review",
                priority=1,
                rationale="No SCF-specific automatic recovery path matches this case yet.",
                tool="review_nwchem_case",
                docs_topics=["scf_open_shell"],
                when_to_use="Use when the task failed outside the SCF loop or the failure is primarily not electronic.",
            )
        )

    return {
        "output_file": output_path,
        "input_file": input_path,
        "failure_class": failure_class,
        "task_outcome": diagnosis.get("task_outcome"),
        "scf_status": scf.get("status"),
        "scf_pattern": trend,
        "iteration_count": iteration_count,
        "hit_max_iterations": hit_max,
        "state_assessment": state_check.get("assessment"),
        "strategy_family": strategy_family,
        "primary_strategy": strategies[0]["name"] if strategies else None,
        "strategies": strategies,
        "notes": notes,
    }


def suggest_nwchem_state_recovery_strategy(
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
    state = check_spin_charge_state(
        output_path=output_path,
        input_path=input_path,
        expected_metal_elements=expected_metal_elements,
        expected_somo_count=expected_somo_count,
    )
    state_check = diagnosis.get("state_check") or {}
    spin_summary = state_check.get("spin_density_summary") or {}
    dominant_site = spin_summary.get("dominant_site")
    metal_like = state_check.get("metal_like_somo_count") or 0
    ligand_like = state_check.get("ligand_like_somo_count") or 0
    observed_somos = state_check.get("somo_count")
    expected_somos = state.get("expected_somo_count")

    regime = "manual_review"
    notes: list[str] = []
    strategies: list[dict[str, Any]] = []

    if state["assessment"] == "plausible":
        regime = "no_state_recovery_needed"
        strategies.append(
            _strategy_entry(
                name="accept_or_verify_state",
                priority=1,
                rationale="The current spin/frontier signals are internally consistent with the requested state.",
                tool="review_nwchem_case",
                docs_topics=["scf_open_shell"],
                when_to_use="Use when the main remaining question is chemical interpretation, not state rescue.",
            )
        )
    else:
        covalent_candidate = bool(
            dominant_site is not None
            and expected_metal_elements
            and dominant_site.get("element") in set(expected_metal_elements)
            and metal_like == 0
            and ligand_like > 0
        )
        if expected_somos is not None and observed_somos is not None and observed_somos != expected_somos:
            regime = "occupancy_or_multiplicity_mismatch"
            strategies.extend(
                [
                    _strategy_entry(
                        name="review_multiplicity_or_charge_model",
                        priority=1,
                        rationale="The observed SOMO count does not match the requested state, so multiplicity/charge may be wrong before deeper recovery attempts.",
                        tool="review_nwchem_input_request",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use when the electron count implied by the target state is inconsistent with the solution found.",
                    ),
                    _strategy_entry(
                        name="vectors_swap_restart",
                        priority=2,
                        rationale="If the target multiplicity is still chemically correct, a swap restart may recover the intended occupation pattern.",
                        tool="prepare_nwchem_next_step",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use when the mismatch is small and the target state is otherwise well motivated.",
                    ),
                ]
            )
        elif covalent_candidate:
            regime = "covalent_ligand_hole_candidate"
            notes.append("do_not_treat_ligand_dominated_somos_as_automatic_garbage")
            strategies.extend(
                [
                    _strategy_entry(
                        name="fragment_guess_validation",
                        priority=1,
                        rationale="A true fragment guess can test whether the high-spin state is merely a bad guess artifact or a robust covalent basin.",
                        tool="suggest_nwchem_scf_fix_strategy",
                        docs_topics=["fragment_guess", "scf_open_shell"],
                        when_to_use="Use when SOMOs are ligand-dominated but most total spin remains on the metal.",
                    ),
                    _strategy_entry(
                        name="mcscf_validation",
                        priority=2,
                        rationale="MCSCF is a strong next step when DFT high-spin solutions look covalent and the metal d-manifold needs explicit validation.",
                        tool="suggest_nwchem_scf_fix_strategy",
                        docs_topics=["mcscf"],
                        when_to_use="Use when you need to determine whether a metal-centered high-spin state exists near the DFT solution.",
                    ),
                    _strategy_entry(
                        name="method_or_multiplicity_scan",
                        priority=3,
                        rationale="If the state is robustly covalent across guesses, changing method or multiplicity is more informative than repeating swap restarts.",
                        tool="create_nwchem_dft_input_from_request",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use when multiple restarts collapse into the same covalent high-spin state.",
                    ),
                ]
            )
        else:
            regime = "metal_state_mismatch"
            strategies.extend(
                [
                    _strategy_entry(
                        name="vectors_swap_restart",
                        priority=1,
                        rationale="The current state looks electronically wrong, but a nearby occupation pattern may be reachable by swapping buried metal-centered orbitals into the SOMO window.",
                        tool="prepare_nwchem_next_step",
                        docs_topics=["scf_open_shell"],
                        when_to_use="Use when metal-centered occupied orbitals exist below ligand-like SOMOs.",
                    ),
                    _strategy_entry(
                        name="fragment_guess_seed",
                        priority=2,
                        rationale="If swap restarts do not redirect the state, a fragment guess gives a stronger initial bias toward the desired basin.",
                        tool="suggest_nwchem_scf_fix_strategy",
                        docs_topics=["fragment_guess"],
                        when_to_use="Use when the state repeatedly reconverges to the same suspicious pattern.",
                    ),
                    _strategy_entry(
                        name="mcscf_seed_or_reference",
                        priority=3,
                        rationale="When DFT is not preserving the desired open-shell character, MCSCF can supply a better state model or at least a diagnostic reference.",
                        tool="suggest_nwchem_scf_fix_strategy",
                        docs_topics=["mcscf"],
                        when_to_use="Use for transition-metal cases where d-orbital character matters more than a single-determinant description.",
                    ),
                    _strategy_entry(
                        name="cube_and_population_validation",
                        priority=4,
                        rationale="Visualizing SOMOs and checking Mulliken/Lowdin spin can confirm whether the suspicious state is actually covalent rather than merely wrong.",
                        tool="draft_nwchem_frontier_cube_input",
                        docs_topics=["fragment_guess"],
                        when_to_use="Use when frontier character and total spin seem contradictory.",
                    ),
                ]
            )

    return {
        "output_file": output_path,
        "input_file": input_path,
        "failure_class": diagnosis.get("failure_class"),
        "task_outcome": diagnosis.get("task_outcome"),
        "state_assessment": state["assessment"],
        "state_check_assessment": state.get("state_check_assessment"),
        "observed_somo_count": observed_somos,
        "expected_somo_count": expected_somos,
        "metal_like_somo_count": metal_like,
        "ligand_like_somo_count": ligand_like,
        "dominant_spin_site": dominant_site,
        "regime": regime,
        "primary_strategy": strategies[0]["name"] if strategies else None,
        "strategies": strategies,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# TCE helpers for summarize_nwchem_case
# ---------------------------------------------------------------------------


__all__ = [
    "suggest_nwchem_scf_fix_strategy",
    "suggest_nwchem_state_recovery_strategy",
]

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from chemtools import (
    analyze_frontier_orbitals,
    analyze_imaginary_modes,
    basis_library_summary,
    check_nwchem_run_status,
    check_spin_charge_state,
    compare_nwchem_runs,
    review_nwchem_followup_outcome,
    review_nwchem_mcscf_followup_outcome,
    review_nwchem_mcscf_case,
    create_nwchem_input,
    create_nwchem_dft_input_from_request,
    create_nwchem_dft_workflow_input,
    diagnose_output,
    displace_geometry_along_mode,
    evaluate_cases,
    draft_nwchem_cube_input,
    draft_nwchem_frontier_cube_input,
    draft_nwchem_imaginary_mode_inputs,
    draft_nwchem_mcscf_input,
    draft_nwchem_mcscf_retry_input,
    draft_nwchem_optimization_followup_input,
    draft_nwchem_property_check_input,
    draft_nwchem_scf_stabilization_input,
    draft_nwchem_vectors_swap_input,
    get_basis_blocks,
    inspect_nwchem_geometry,
    inspect_input,
    inspect_runner_profiles,
    find_restart_assets,
    launch_nwchem_run,
    lint_nwchem_input,
    parse_freq,
    parse_mcscf_output,
    parse_mos,
    parse_population_analysis,
    parse_output,
    parse_scf,
    parse_tasks,
    parse_trajectory,
    parse_cube,
    prepare_nwchem_next_step,
    prepare_nwchem_run,
    render_basis_block,
    render_basis_block_from_geometry,
    render_ecp_block,
    render_nwchem_basis_setup,
    review_nwchem_progress,
    watch_nwchem_run,
    review_nwchem_input_request,
    review_nwchem_case,
    resolve_basis,
    resolve_basis_setup,
    resolve_ecp,
    suggest_nwchem_scf_fix_strategy,
    suggest_nwchem_mcscf_active_space,
    suggest_nwchem_state_recovery_strategy,
    suggest_vectors_swaps,
    summarize_cube,
    summarize_nwchem_case,
    summarize_output,
    tail_nwchem_output,
    terminate_nwchem_run,
)


DEFAULT_BASIS_LIBRARY = Path("/Users/charlie/test/mytest/nwchem-test/nwchem_basis_library")


def _basis_library_path(path: str | None = None) -> str:
    if path:
        return path
    configured = os.environ.get("CHEMTOOLS_BASIS_LIBRARY")
    if configured:
        return configured
    return str(DEFAULT_BASIS_LIBRARY)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone chemistry output parsing tools.")
    parser.add_argument("--pretty", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_tasks_parser = subparsers.add_parser("parse-tasks")
    parse_tasks_parser.add_argument("file")
    parse_tasks_parser.add_argument("--pretty", action="store_true")

    parse_mos_parser = subparsers.add_parser("parse-mos")
    parse_mos_parser.add_argument("file")
    parse_mos_parser.add_argument("--top-n", type=int, default=5)
    parse_mos_parser.add_argument("--include-coefficients", action="store_true")
    parse_mos_parser.add_argument("--pretty", action="store_true")

    parse_population_parser = subparsers.add_parser("parse-population")
    parse_population_parser.add_argument("file")
    parse_population_parser.add_argument("--pretty", action="store_true")

    parse_mcscf_parser = subparsers.add_parser("parse-mcscf")
    parse_mcscf_parser.add_argument("file")
    parse_mcscf_parser.add_argument("--pretty", action="store_true")

    review_mcscf_parser = subparsers.add_parser("review-nwchem-mcscf")
    review_mcscf_parser.add_argument("output_file")
    review_mcscf_parser.add_argument("--input-file")
    review_mcscf_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    review_mcscf_parser.add_argument("--pretty", action="store_true")

    review_mcscf_followup_parser = subparsers.add_parser("review-nwchem-mcscf-followup")
    review_mcscf_followup_parser.add_argument("reference_output_file")
    review_mcscf_followup_parser.add_argument("candidate_output_file")
    review_mcscf_followup_parser.add_argument("--reference-input-file")
    review_mcscf_followup_parser.add_argument("--candidate-input-file")
    review_mcscf_followup_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    review_mcscf_followup_parser.add_argument("--output-dir")
    review_mcscf_followup_parser.add_argument("--base-name")
    review_mcscf_followup_parser.add_argument("--pretty", action="store_true")

    parse_freq_parser = subparsers.add_parser("parse-freq")
    parse_freq_parser.add_argument("file")
    parse_freq_parser.add_argument("--include-displacements", action="store_true")
    parse_freq_parser.add_argument("--pretty", action="store_true")

    analyze_frontier_parser = subparsers.add_parser("analyze-frontier")
    analyze_frontier_parser.add_argument("output_file")
    analyze_frontier_parser.add_argument("--input-file")
    analyze_frontier_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    analyze_frontier_parser.add_argument("--expected-somos", type=int)
    analyze_frontier_parser.add_argument("--pretty", action="store_true")

    suggest_swaps_parser = subparsers.add_parser("suggest-vectors-swaps")
    suggest_swaps_parser.add_argument("output_file")
    suggest_swaps_parser.add_argument("--input-file")
    suggest_swaps_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    suggest_swaps_parser.add_argument("--expected-somos", type=int)
    suggest_swaps_parser.add_argument("--vectors-input")
    suggest_swaps_parser.add_argument("--vectors-output")
    suggest_swaps_parser.add_argument("--pretty", action="store_true")

    analyze_imag_parser = subparsers.add_parser("analyze-imaginary")
    analyze_imag_parser.add_argument("file")
    analyze_imag_parser.add_argument("--significant-threshold-cm1", type=float, default=20.0)
    analyze_imag_parser.add_argument("--top-atoms", type=int, default=4)
    analyze_imag_parser.add_argument("--pretty", action="store_true")

    displace_mode_parser = subparsers.add_parser("displace-mode")
    displace_mode_parser.add_argument("file")
    displace_mode_parser.add_argument("--mode-number", type=int)
    displace_mode_parser.add_argument("--amplitude-angstrom", type=float, default=0.15)
    displace_mode_parser.add_argument("--significant-threshold-cm1", type=float, default=20.0)
    displace_mode_parser.add_argument("--pretty", action="store_true")

    draft_mode_parser = subparsers.add_parser("draft-nwchem-imaginary-inputs")
    draft_mode_parser.add_argument("output_file")
    draft_mode_parser.add_argument("--input-file", required=True)
    draft_mode_parser.add_argument("--mode-number", type=int)
    draft_mode_parser.add_argument("--amplitude-angstrom", type=float, default=0.15)
    draft_mode_parser.add_argument("--significant-threshold-cm1", type=float, default=20.0)
    draft_mode_parser.add_argument(
        "--task-strategy",
        choices=["auto", "optimize_only", "optimize_then_freq"],
        default="auto",
    )
    draft_mode_parser.add_argument("--output-dir")
    draft_mode_parser.add_argument("--base-name")
    draft_mode_parser.add_argument("--write-files", action="store_true")
    draft_mode_parser.add_argument("--noautosym", action=argparse.BooleanOptionalAction, default=True)
    draft_mode_parser.add_argument("--symmetry-c1", action=argparse.BooleanOptionalAction, default=True)
    draft_mode_parser.add_argument("--pretty", action="store_true")

    draft_swap_parser = subparsers.add_parser("draft-nwchem-swap-input")
    draft_swap_parser.add_argument("output_file")
    draft_swap_parser.add_argument("--input-file", required=True)
    draft_swap_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    draft_swap_parser.add_argument("--expected-somos", type=int)
    draft_swap_parser.add_argument("--vectors-input")
    draft_swap_parser.add_argument("--vectors-output")
    draft_swap_parser.add_argument("--task-operation", default="energy")
    draft_swap_parser.add_argument("--iterations", type=int, default=500)
    draft_swap_parser.add_argument("--smear", type=float, default=0.001)
    draft_swap_parser.add_argument("--convergence-damp", type=int, default=30)
    draft_swap_parser.add_argument("--convergence-ncydp", type=int, default=30)
    draft_swap_parser.add_argument("--population-print", default="mulliken")
    draft_swap_parser.add_argument("--output-dir")
    draft_swap_parser.add_argument("--base-name")
    draft_swap_parser.add_argument("--title")
    draft_swap_parser.add_argument("--write-file", action="store_true")
    draft_swap_parser.add_argument("--pretty", action="store_true")

    draft_property_parser = subparsers.add_parser("draft-nwchem-property-input")
    draft_property_parser.add_argument("--input-file", required=True)
    draft_property_parser.add_argument("--reference-output-file")
    draft_property_parser.add_argument("--vectors-input")
    draft_property_parser.add_argument("--vectors-output")
    draft_property_parser.add_argument("--property-keywords", default="mulliken")
    draft_property_parser.add_argument("--task-strategy", choices=["auto", "property", "energy"], default="auto")
    draft_property_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    draft_property_parser.add_argument("--expected-somos", type=int)
    draft_property_parser.add_argument("--iterations", type=int, default=1)
    draft_property_parser.add_argument("--convergence-energy", default="1e3")
    draft_property_parser.add_argument("--smear", type=float)
    draft_property_parser.add_argument("--output-dir")
    draft_property_parser.add_argument("--base-name")
    draft_property_parser.add_argument("--title")
    draft_property_parser.add_argument("--write-file", action="store_true")
    draft_property_parser.add_argument("--pretty", action="store_true")

    draft_stabilize_parser = subparsers.add_parser("draft-nwchem-scf-stabilization")
    draft_stabilize_parser.add_argument("--input-file", required=True)
    draft_stabilize_parser.add_argument("--reference-output-file")
    draft_stabilize_parser.add_argument("--vectors-input")
    draft_stabilize_parser.add_argument("--vectors-output")
    draft_stabilize_parser.add_argument("--task-operation", default="energy")
    draft_stabilize_parser.add_argument("--iterations", type=int)
    draft_stabilize_parser.add_argument("--smear", type=float)
    draft_stabilize_parser.add_argument("--convergence-damp", type=int)
    draft_stabilize_parser.add_argument("--convergence-ncydp", type=int)
    draft_stabilize_parser.add_argument("--population-print")
    draft_stabilize_parser.add_argument("--output-dir")
    draft_stabilize_parser.add_argument("--base-name")
    draft_stabilize_parser.add_argument("--title")
    draft_stabilize_parser.add_argument("--write-file", action="store_true")
    draft_stabilize_parser.add_argument("--pretty", action="store_true")

    draft_optimization_parser = subparsers.add_parser("draft-nwchem-optimization-followup")
    draft_optimization_parser.add_argument("output_file")
    draft_optimization_parser.add_argument("--input-file", required=True)
    draft_optimization_parser.add_argument(
        "--task-strategy",
        choices=["auto", "optimize_only", "freq_only", "optimize_then_freq"],
        default="auto",
    )
    draft_optimization_parser.add_argument("--output-dir")
    draft_optimization_parser.add_argument("--base-name")
    draft_optimization_parser.add_argument("--title")
    draft_optimization_parser.add_argument("--write-file", action="store_true")
    draft_optimization_parser.add_argument("--pretty", action="store_true")

    draft_cube_parser = subparsers.add_parser("draft-nwchem-cube-input")
    draft_cube_parser.add_argument("--input-file", required=True)
    draft_cube_parser.add_argument("--vectors-input", required=True)
    draft_cube_parser.add_argument("--orbital-vectors", default="")
    draft_cube_parser.add_argument("--density-modes", default="")
    draft_cube_parser.add_argument("--orbital-spin", default="total")
    draft_cube_parser.add_argument("--extent-angstrom", type=float, default=6.0)
    draft_cube_parser.add_argument("--grid-points", type=int, default=120)
    draft_cube_parser.add_argument("--gaussian", action=argparse.BooleanOptionalAction, default=True)
    draft_cube_parser.add_argument("--output-dir")
    draft_cube_parser.add_argument("--base-name")
    draft_cube_parser.add_argument("--title")
    draft_cube_parser.add_argument("--write-file", action="store_true")
    draft_cube_parser.add_argument("--pretty", action="store_true")

    draft_frontier_cube_parser = subparsers.add_parser("draft-nwchem-frontier-cubes")
    draft_frontier_cube_parser.add_argument("output_file")
    draft_frontier_cube_parser.add_argument("--input-file", required=True)
    draft_frontier_cube_parser.add_argument("--vectors-input")
    draft_frontier_cube_parser.add_argument("--include-somos", action=argparse.BooleanOptionalAction, default=True)
    draft_frontier_cube_parser.add_argument("--include-homo", action=argparse.BooleanOptionalAction, default=True)
    draft_frontier_cube_parser.add_argument("--include-lumo", action=argparse.BooleanOptionalAction, default=True)
    draft_frontier_cube_parser.add_argument("--include-density-modes", default="")
    draft_frontier_cube_parser.add_argument("--extent-angstrom", type=float, default=6.0)
    draft_frontier_cube_parser.add_argument("--grid-points", type=int, default=120)
    draft_frontier_cube_parser.add_argument("--gaussian", action=argparse.BooleanOptionalAction, default=True)
    draft_frontier_cube_parser.add_argument("--output-dir")
    draft_frontier_cube_parser.add_argument("--base-name")
    draft_frontier_cube_parser.add_argument("--title")
    draft_frontier_cube_parser.add_argument("--write-file", action="store_true")
    draft_frontier_cube_parser.add_argument("--pretty", action="store_true")

    parse_traj_parser = subparsers.add_parser("parse-trajectory")
    parse_traj_parser.add_argument("file")
    parse_traj_parser.add_argument("--include-positions", action="store_true")
    parse_traj_parser.add_argument("--pretty", action="store_true")

    parse_cube_parser = subparsers.add_parser("parse-cube")
    parse_cube_parser.add_argument("file")
    parse_cube_parser.add_argument("--include-values", action="store_true")
    parse_cube_parser.add_argument("--pretty", action="store_true")

    summarize_cube_parser = subparsers.add_parser("summarize-cube")
    summarize_cube_parser.add_argument("file")
    summarize_cube_parser.add_argument("--top-atoms", type=int, default=5)
    summarize_cube_parser.add_argument("--pretty", action="store_true")

    inspect_profiles_parser = subparsers.add_parser("inspect-runner-profiles")
    inspect_profiles_parser.add_argument("--profiles-path")
    inspect_profiles_parser.add_argument("--pretty", action="store_true")

    prepare_run_parser = subparsers.add_parser("prepare-nwchem-run")
    prepare_run_parser.add_argument("input_file")
    prepare_run_parser.add_argument("--profile", required=True)
    prepare_run_parser.add_argument("--profiles-path")
    prepare_run_parser.add_argument("--job-name")
    prepare_run_parser.add_argument("--resource-overrides", default="")
    prepare_run_parser.add_argument("--env-overrides", default="")
    prepare_run_parser.add_argument("--pretty", action="store_true")

    launch_run_parser = subparsers.add_parser("launch-nwchem-run")
    launch_run_parser.add_argument("input_file")
    launch_run_parser.add_argument("--profile", required=True)
    launch_run_parser.add_argument("--profiles-path")
    launch_run_parser.add_argument("--job-name")
    launch_run_parser.add_argument("--resource-overrides", default="")
    launch_run_parser.add_argument("--env-overrides", default="")
    launch_run_parser.add_argument("--no-write-script", dest="write_script", action="store_false")
    launch_run_parser.set_defaults(write_script=True)
    launch_run_parser.add_argument("--pretty", action="store_true")

    status_parser = subparsers.add_parser("check-nwchem-run-status")
    status_parser.add_argument("--output-file")
    status_parser.add_argument("--input-file")
    status_parser.add_argument("--error-file")
    status_parser.add_argument("--process-id", type=int)
    status_parser.add_argument("--profile")
    status_parser.add_argument("--job-id")
    status_parser.add_argument("--profiles-path")
    status_parser.add_argument("--pretty", action="store_true")

    progress_parser = subparsers.add_parser("review-nwchem-progress")
    progress_parser.add_argument("--output-file", required=True)
    progress_parser.add_argument("--input-file")
    progress_parser.add_argument("--error-file")
    progress_parser.add_argument("--process-id", type=int)
    progress_parser.add_argument("--profile")
    progress_parser.add_argument("--job-id")
    progress_parser.add_argument("--profiles-path")
    progress_parser.add_argument("--pretty", action="store_true")

    watch_parser = subparsers.add_parser("watch-nwchem-run")
    watch_parser.add_argument("--output-file")
    watch_parser.add_argument("--input-file")
    watch_parser.add_argument("--error-file")
    watch_parser.add_argument("--process-id", type=int)
    watch_parser.add_argument("--profile")
    watch_parser.add_argument("--job-id")
    watch_parser.add_argument("--profiles-path")
    watch_parser.add_argument("--poll-interval-seconds", type=float, default=10.0)
    watch_parser.add_argument("--adaptive-polling", dest="adaptive_polling", action="store_true", default=True)
    watch_parser.add_argument("--fixed-polling", dest="adaptive_polling", action="store_false")
    watch_parser.add_argument("--max-poll-interval-seconds", type=float, default=60.0)
    watch_parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    watch_parser.add_argument("--max-polls", type=int)
    watch_parser.add_argument("--history-limit", type=int, default=8)
    watch_parser.add_argument("--pretty", action="store_true")

    tail_parser = subparsers.add_parser("tail-nwchem-output")
    tail_parser.add_argument("file")
    tail_parser.add_argument("--lines", type=int, default=30)
    tail_parser.add_argument("--max-characters", type=int, default=4000)
    tail_parser.add_argument("--pretty", action="store_true")

    terminate_parser = subparsers.add_parser("terminate-nwchem-run")
    terminate_parser.add_argument("--process-id", required=True, type=int)
    terminate_parser.add_argument("--signal-name", choices=["term", "kill"], default="term")
    terminate_parser.add_argument("--pretty", action="store_true")

    compare_parser = subparsers.add_parser("compare-nwchem-runs")
    compare_parser.add_argument("reference_output")
    compare_parser.add_argument("candidate_output")
    compare_parser.add_argument("--reference-input")
    compare_parser.add_argument("--candidate-input")
    compare_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    compare_parser.add_argument("--expected-somos", type=int)
    compare_parser.add_argument("--pretty", action="store_true")

    followup_review_parser = subparsers.add_parser("review-nwchem-followup")
    followup_review_parser.add_argument("reference_output")
    followup_review_parser.add_argument("candidate_output")
    followup_review_parser.add_argument("--reference-input")
    followup_review_parser.add_argument("--candidate-input")
    followup_review_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    followup_review_parser.add_argument("--expected-somos", type=int)
    followup_review_parser.add_argument("--output-dir")
    followup_review_parser.add_argument("--base-name")
    followup_review_parser.add_argument("--pretty", action="store_true")

    eval_cases_parser = subparsers.add_parser("eval-cases")
    eval_cases_parser.add_argument("path")
    eval_cases_parser.add_argument("--pretty", action="store_true")

    next_step_parser = subparsers.add_parser("prepare-nwchem-next-step")
    next_step_parser.add_argument("output_file")
    next_step_parser.add_argument("--input-file")
    next_step_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    next_step_parser.add_argument("--expected-somos", type=int)
    next_step_parser.add_argument("--output-dir")
    next_step_parser.add_argument("--base-name")
    next_step_parser.add_argument("--write-files", action="store_true")
    next_step_parser.add_argument("--include-property-check", action=argparse.BooleanOptionalAction, default=True)
    next_step_parser.add_argument("--include-frontier-cubes", action=argparse.BooleanOptionalAction, default=False)
    next_step_parser.add_argument("--include-density-modes", default="")
    next_step_parser.add_argument("--cube-extent-angstrom", type=float, default=6.0)
    next_step_parser.add_argument("--cube-grid-points", type=int, default=120)
    next_step_parser.add_argument("--pretty", action="store_true")

    parse_output_parser = subparsers.add_parser("parse-output")
    parse_output_parser.add_argument("file")
    parse_output_parser.add_argument(
        "--sections",
        default="tasks,mos,freq,trajectory",
        help="Comma-separated list of sections",
    )
    parse_output_parser.add_argument("--top-n", type=int, default=5)
    parse_output_parser.add_argument("--include-coefficients", action="store_true")
    parse_output_parser.add_argument("--include-displacements", action="store_true")
    parse_output_parser.add_argument("--include-positions", action="store_true")
    parse_output_parser.add_argument("--pretty", action="store_true")

    basis_summary_parser = subparsers.add_parser("basis-library-summary")
    basis_summary_parser.add_argument("library_path")
    basis_summary_parser.add_argument("--pretty", action="store_true")

    resolve_basis_parser = subparsers.add_parser("resolve-basis")
    resolve_basis_parser.add_argument("basis_name")
    resolve_basis_parser.add_argument("--elements", required=True, help="Comma-separated element list, e.g. Fe,C,N")
    resolve_basis_parser.add_argument("--library-path", required=True)
    resolve_basis_parser.add_argument("--pretty", action="store_true")

    resolve_ecp_parser = subparsers.add_parser("resolve-ecp")
    resolve_ecp_parser.add_argument("ecp_name")
    resolve_ecp_parser.add_argument("--elements", required=True, help="Comma-separated element list, e.g. U,Th")
    resolve_ecp_parser.add_argument("--library-path", required=True)
    resolve_ecp_parser.add_argument("--pretty", action="store_true")

    basis_blocks_parser = subparsers.add_parser("get-basis-blocks")
    basis_blocks_parser.add_argument("basis_name")
    basis_blocks_parser.add_argument("--elements", required=True, help="Comma-separated element list")
    basis_blocks_parser.add_argument("--library-path", required=True)
    basis_blocks_parser.add_argument("--pretty", action="store_true")

    inspect_geometry_parser = subparsers.add_parser("inspect-nwchem-geometry")
    inspect_geometry_parser.add_argument("file")
    inspect_geometry_parser.add_argument("--pretty", action="store_true")

    inspect_input_parser = subparsers.add_parser("inspect-nwchem-input")
    inspect_input_parser.add_argument("file")
    inspect_input_parser.add_argument("--pretty", action="store_true")

    lint_input_parser = subparsers.add_parser("lint-nwchem-input")
    lint_input_parser.add_argument("input_file")
    lint_input_parser.add_argument("--library-path")
    lint_input_parser.add_argument("--pretty", action="store_true")

    find_assets_parser = subparsers.add_parser("find-restart-assets")
    find_assets_parser.add_argument("path")
    find_assets_parser.add_argument("--pretty", action="store_true")

    render_basis_parser = subparsers.add_parser("render-nwchem-basis")
    render_basis_parser.add_argument("basis_name")
    render_basis_parser.add_argument("--library-path", required=True)
    render_basis_parser.add_argument("--elements", help="Comma-separated element list")
    render_basis_parser.add_argument("--geometry-file")
    render_basis_parser.add_argument("--block-name", default="ao basis")
    render_basis_parser.add_argument("--mode")
    render_basis_parser.add_argument("--pretty", action="store_true")

    render_ecp_parser = subparsers.add_parser("render-nwchem-ecp")
    render_ecp_parser.add_argument("ecp_name")
    render_ecp_parser.add_argument("--library-path", required=True)
    render_ecp_parser.add_argument("--elements", required=True, help="Comma-separated element list")
    render_ecp_parser.add_argument("--pretty", action="store_true")

    resolve_setup_parser = subparsers.add_parser("resolve-nwchem-basis-setup")
    resolve_setup_parser.add_argument("geometry_file")
    resolve_setup_parser.add_argument("--library-path", required=True)
    resolve_setup_parser.add_argument("--basis-map", required=True, help="Comma-separated element=basis entries")
    resolve_setup_parser.add_argument("--ecp-map", default="", help="Comma-separated element=ecp entries")
    resolve_setup_parser.add_argument("--default-basis")
    resolve_setup_parser.add_argument("--default-ecp")
    resolve_setup_parser.add_argument("--pretty", action="store_true")

    render_setup_parser = subparsers.add_parser("render-nwchem-basis-setup")
    render_setup_parser.add_argument("geometry_file")
    render_setup_parser.add_argument("--library-path", required=True)
    render_setup_parser.add_argument("--basis-map", required=True, help="Comma-separated element=basis entries")
    render_setup_parser.add_argument("--ecp-map", default="", help="Comma-separated element=ecp entries")
    render_setup_parser.add_argument("--default-basis")
    render_setup_parser.add_argument("--default-ecp")
    render_setup_parser.add_argument("--basis-mode")
    render_setup_parser.add_argument("--block-name", default="ao basis")
    render_setup_parser.add_argument("--pretty", action="store_true")

    create_input_parser = subparsers.add_parser("create-nwchem-input")
    create_input_parser.add_argument("geometry_file")
    create_input_parser.add_argument("--library-path", required=True)
    create_input_parser.add_argument("--basis-map", required=True, help="Comma-separated element=basis entries")
    create_input_parser.add_argument("--ecp-map", default="", help="Comma-separated element=ecp entries")
    create_input_parser.add_argument("--default-basis")
    create_input_parser.add_argument("--default-ecp")
    create_input_parser.add_argument("--basis-mode")
    create_input_parser.add_argument("--block-name", default="ao basis")
    create_input_parser.add_argument("--module", required=True)
    create_input_parser.add_argument("--task-operation")
    create_input_parser.add_argument("--charge", type=int)
    create_input_parser.add_argument("--multiplicity", type=int)
    create_input_parser.add_argument("--memory")
    create_input_parser.add_argument("--title")
    create_input_parser.add_argument("--start-name")
    create_input_parser.add_argument("--vectors-input")
    create_input_parser.add_argument("--vectors-output")
    create_input_parser.add_argument("--module-setting", action="append", default=[])
    create_input_parser.add_argument("--extra-block", action="append", default=[])
    create_input_parser.add_argument("--output-dir")
    create_input_parser.add_argument("--write-file", action="store_true")
    create_input_parser.add_argument("--pretty", action="store_true")

    review_input_request_parser = subparsers.add_parser("review-nwchem-input-request")
    review_input_request_parser.add_argument("--formula")
    review_input_request_parser.add_argument("--geometry-file")
    review_input_request_parser.add_argument("--library-path")
    review_input_request_parser.add_argument("--basis-map", default="", help="Comma-separated element=basis entries")
    review_input_request_parser.add_argument("--ecp-map", default="", help="Comma-separated element=ecp entries")
    review_input_request_parser.add_argument("--default-basis")
    review_input_request_parser.add_argument("--default-ecp")
    review_input_request_parser.add_argument("--module", default="dft")
    review_input_request_parser.add_argument("--tasks", default="energy")
    review_input_request_parser.add_argument("--functional")
    review_input_request_parser.add_argument("--charge", type=int)
    review_input_request_parser.add_argument("--multiplicity", type=int)
    review_input_request_parser.add_argument("--pretty", action="store_true")

    create_dft_workflow_parser = subparsers.add_parser("create-nwchem-dft-workflow")
    create_dft_workflow_parser.add_argument("geometry_file")
    create_dft_workflow_parser.add_argument("--library-path", required=True)
    create_dft_workflow_parser.add_argument("--basis-map", required=True, help="Comma-separated element=basis entries")
    create_dft_workflow_parser.add_argument("--ecp-map", default="", help="Comma-separated element=ecp entries")
    create_dft_workflow_parser.add_argument("--default-basis")
    create_dft_workflow_parser.add_argument("--default-ecp")
    create_dft_workflow_parser.add_argument("--block-name", default="ao basis")
    create_dft_workflow_parser.add_argument("--basis-mode")
    create_dft_workflow_parser.add_argument("--xc-functional", required=True)
    create_dft_workflow_parser.add_argument("--tasks", default="energy")
    create_dft_workflow_parser.add_argument("--charge", type=int)
    create_dft_workflow_parser.add_argument("--multiplicity", type=int)
    create_dft_workflow_parser.add_argument("--dft-setting", action="append", default=[])
    create_dft_workflow_parser.add_argument("--extra-block", action="append", default=[])
    create_dft_workflow_parser.add_argument("--memory")
    create_dft_workflow_parser.add_argument("--title")
    create_dft_workflow_parser.add_argument("--start-name")
    create_dft_workflow_parser.add_argument("--vectors-input")
    create_dft_workflow_parser.add_argument("--vectors-output")
    create_dft_workflow_parser.add_argument("--output-dir")
    create_dft_workflow_parser.add_argument("--write-file", action="store_true")
    create_dft_workflow_parser.add_argument("--pretty", action="store_true")

    create_dft_from_request_parser = subparsers.add_parser("create-nwchem-dft-from-request")
    create_dft_from_request_parser.add_argument("--formula")
    create_dft_from_request_parser.add_argument("--geometry-file")
    create_dft_from_request_parser.add_argument("--library-path")
    create_dft_from_request_parser.add_argument("--basis-map", default="", help="Comma-separated element=basis entries")
    create_dft_from_request_parser.add_argument("--ecp-map", default="", help="Comma-separated element=ecp entries")
    create_dft_from_request_parser.add_argument("--default-basis")
    create_dft_from_request_parser.add_argument("--default-ecp")
    create_dft_from_request_parser.add_argument("--xc-functional")
    create_dft_from_request_parser.add_argument("--tasks", default="energy")
    create_dft_from_request_parser.add_argument("--charge", type=int)
    create_dft_from_request_parser.add_argument("--multiplicity", type=int)
    create_dft_from_request_parser.add_argument("--dft-setting", action="append", default=[])
    create_dft_from_request_parser.add_argument("--extra-block", action="append", default=[])
    create_dft_from_request_parser.add_argument("--memory")
    create_dft_from_request_parser.add_argument("--title")
    create_dft_from_request_parser.add_argument("--start-name")
    create_dft_from_request_parser.add_argument("--vectors-input")
    create_dft_from_request_parser.add_argument("--vectors-output")
    create_dft_from_request_parser.add_argument("--output-dir")
    create_dft_from_request_parser.add_argument("--write-file", action="store_true")
    create_dft_from_request_parser.add_argument("--pretty", action="store_true")

    parse_scf_parser = subparsers.add_parser("parse-scf")
    parse_scf_parser.add_argument("file")
    parse_scf_parser.add_argument("--pretty", action="store_true")

    diagnose_parser = subparsers.add_parser("diagnose-nwchem")
    diagnose_parser.add_argument("output_file")
    diagnose_parser.add_argument("--input-file")
    diagnose_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    diagnose_parser.add_argument("--expected-somos", type=int)
    diagnose_parser.add_argument("--pretty", action="store_true")

    summarize_parser = subparsers.add_parser("summarize-nwchem")
    summarize_parser.add_argument("output_file")
    summarize_parser.add_argument("--input-file")
    summarize_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    summarize_parser.add_argument("--expected-somos", type=int)
    summarize_parser.add_argument("--pretty", action="store_true")

    spin_state_parser = subparsers.add_parser("check-spin-charge-state")
    spin_state_parser.add_argument("output_file")
    spin_state_parser.add_argument("--input-file")
    spin_state_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    spin_state_parser.add_argument("--expected-somos", type=int)
    spin_state_parser.add_argument("--pretty", action="store_true")

    scf_strategy_parser = subparsers.add_parser("suggest-nwchem-scf-fix-strategy")
    scf_strategy_parser.add_argument("output_file")
    scf_strategy_parser.add_argument("--input-file")
    scf_strategy_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    scf_strategy_parser.add_argument("--expected-somos", type=int)
    scf_strategy_parser.add_argument("--pretty", action="store_true")

    mcscf_space_parser = subparsers.add_parser("suggest-nwchem-mcscf-active-space")
    mcscf_space_parser.add_argument("output_file")
    mcscf_space_parser.add_argument("--input-file")
    mcscf_space_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    mcscf_space_parser.add_argument("--expected-somos", type=int)
    mcscf_space_parser.add_argument("--pretty", action="store_true")

    state_strategy_parser = subparsers.add_parser("suggest-nwchem-state-recovery-strategy")
    state_strategy_parser.add_argument("output_file")
    state_strategy_parser.add_argument("--input-file")
    state_strategy_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    state_strategy_parser.add_argument("--expected-somos", type=int)
    state_strategy_parser.add_argument("--pretty", action="store_true")

    draft_mcscf_parser = subparsers.add_parser("draft-nwchem-mcscf-input")
    draft_mcscf_parser.add_argument("reference_output_file")
    draft_mcscf_parser.add_argument("input_file")
    draft_mcscf_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    draft_mcscf_parser.add_argument("--expected-somos", type=int)
    draft_mcscf_parser.add_argument("--active-space-mode", choices=["minimal", "expanded"], default="minimal")
    draft_mcscf_parser.add_argument("--vectors-input")
    draft_mcscf_parser.add_argument("--vectors-output")
    draft_mcscf_parser.add_argument("--state-label")
    draft_mcscf_parser.add_argument("--symmetry", type=int)
    draft_mcscf_parser.add_argument("--hessian", choices=["exact", "onel"], default="exact")
    draft_mcscf_parser.add_argument("--maxiter", type=int, default=80)
    draft_mcscf_parser.add_argument("--thresh", type=float, default=1.0e-5)
    draft_mcscf_parser.add_argument("--level", type=float, default=0.6)
    draft_mcscf_parser.add_argument("--no-lock-vectors", action="store_true")
    draft_mcscf_parser.add_argument("--output-dir")
    draft_mcscf_parser.add_argument("--base-name")
    draft_mcscf_parser.add_argument("--title")
    draft_mcscf_parser.add_argument("--write-file", action="store_true")
    draft_mcscf_parser.add_argument("--pretty", action="store_true")

    draft_mcscf_retry_parser = subparsers.add_parser("draft-nwchem-mcscf-retry")
    draft_mcscf_retry_parser.add_argument("output_file")
    draft_mcscf_retry_parser.add_argument("input_file")
    draft_mcscf_retry_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    draft_mcscf_retry_parser.add_argument("--active-space-mode", choices=["auto", "minimal", "expanded"], default="auto")
    draft_mcscf_retry_parser.add_argument("--vectors-input")
    draft_mcscf_retry_parser.add_argument("--vectors-output")
    draft_mcscf_retry_parser.add_argument("--state-label")
    draft_mcscf_retry_parser.add_argument("--symmetry", type=int)
    draft_mcscf_retry_parser.add_argument("--hessian", choices=["exact", "onel"])
    draft_mcscf_retry_parser.add_argument("--maxiter", type=int)
    draft_mcscf_retry_parser.add_argument("--thresh", type=float)
    draft_mcscf_retry_parser.add_argument("--level", type=float)
    draft_mcscf_retry_parser.add_argument("--no-lock-vectors", action="store_true")
    draft_mcscf_retry_parser.add_argument("--output-dir")
    draft_mcscf_retry_parser.add_argument("--base-name")
    draft_mcscf_retry_parser.add_argument("--title")
    draft_mcscf_retry_parser.add_argument("--write-file", action="store_true")
    draft_mcscf_retry_parser.add_argument("--pretty", action="store_true")

    summarize_case_parser = subparsers.add_parser("summarize-nwchem-case")
    summarize_case_parser.add_argument("output_file")
    summarize_case_parser.add_argument("--input-file")
    summarize_case_parser.add_argument("--library-path")
    summarize_case_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    summarize_case_parser.add_argument("--expected-somos", type=int)
    summarize_case_parser.add_argument("--output-dir")
    summarize_case_parser.add_argument("--base-name")
    summarize_case_parser.add_argument("--compact", action="store_true")
    summarize_case_parser.add_argument("--pretty", action="store_true")

    review_case_parser = subparsers.add_parser("review-nwchem-case")
    review_case_parser.add_argument("output_file")
    review_case_parser.add_argument("--input-file")
    review_case_parser.add_argument("--library-path")
    review_case_parser.add_argument("--expected-metals", help="Comma-separated list such as Fe,Co")
    review_case_parser.add_argument("--expected-somos", type=int)
    review_case_parser.add_argument("--output-dir")
    review_case_parser.add_argument("--base-name")
    review_case_parser.add_argument("--pretty", action="store_true")

    args = parser.parse_args()

    if args.command == "parse-tasks":
        payload = parse_tasks(args.file)
    elif args.command == "parse-mos":
        payload = parse_mos(
            args.file,
            top_n=args.top_n,
            include_coefficients=args.include_coefficients,
        )
    elif args.command == "parse-population":
        payload = parse_population_analysis(args.file)
    elif args.command == "parse-mcscf":
        payload = parse_mcscf_output(args.file)
    elif args.command == "review-nwchem-mcscf":
        payload = review_nwchem_mcscf_case(
            output_path=args.output_file,
            input_path=args.input_file,
            expected_metal_elements=_parse_csv_arg(args.expected_metals),
        )
    elif args.command == "review-nwchem-mcscf-followup":
        payload = review_nwchem_mcscf_followup_outcome(
            reference_output_path=args.reference_output_file,
            candidate_output_path=args.candidate_output_file,
            reference_input_path=args.reference_input_file,
            candidate_input_path=args.candidate_input_file,
            expected_metal_elements=_parse_csv_arg(args.expected_metals),
            output_dir=args.output_dir,
            base_name=args.base_name,
        )
    elif args.command == "parse-freq":
        payload = parse_freq(args.file, include_displacements=args.include_displacements)
    elif args.command == "analyze-frontier":
        payload = analyze_frontier_orbitals(
            output_path=args.output_file,
            input_path=args.input_file,
            expected_metal_elements=[
                element.strip() for element in (args.expected_metals or "").split(",") if element.strip()
            ]
            or None,
            expected_somo_count=args.expected_somos,
        )
    elif args.command == "suggest-vectors-swaps":
        payload = suggest_vectors_swaps(
            output_path=args.output_file,
            input_path=args.input_file,
            expected_metal_elements=[
                element.strip() for element in (args.expected_metals or "").split(",") if element.strip()
            ]
            or None,
            expected_somo_count=args.expected_somos,
            vectors_input=args.vectors_input,
            vectors_output=args.vectors_output,
        )
    elif args.command == "analyze-imaginary":
        payload = analyze_imaginary_modes(
            args.file,
            significant_threshold_cm1=args.significant_threshold_cm1,
            top_atoms=args.top_atoms,
        )
    elif args.command == "displace-mode":
        payload = displace_geometry_along_mode(
            args.file,
            mode_number=args.mode_number,
            amplitude_angstrom=args.amplitude_angstrom,
            significant_threshold_cm1=args.significant_threshold_cm1,
        )
    elif args.command == "draft-nwchem-imaginary-inputs":
        payload = draft_nwchem_imaginary_mode_inputs(
            output_path=args.output_file,
            input_path=args.input_file,
            mode_number=args.mode_number,
            amplitude_angstrom=args.amplitude_angstrom,
            significant_threshold_cm1=args.significant_threshold_cm1,
            add_noautosym=args.noautosym,
            enforce_symmetry_c1=args.symmetry_c1,
            task_strategy=args.task_strategy,
            output_dir=args.output_dir,
            base_name=args.base_name,
            write_files=args.write_files,
        )
    elif args.command == "draft-nwchem-swap-input":
        payload = draft_nwchem_vectors_swap_input(
            output_path=args.output_file,
            input_path=args.input_file,
            expected_metal_elements=[
                element.strip() for element in (args.expected_metals or "").split(",") if element.strip()
            ]
            or None,
            expected_somo_count=args.expected_somos,
            vectors_input=args.vectors_input,
            vectors_output=args.vectors_output,
            task_operation=args.task_operation,
            iterations=args.iterations,
            smear=args.smear,
            convergence_damp=args.convergence_damp,
            convergence_ncydp=args.convergence_ncydp,
            population_print=args.population_print,
            output_dir=args.output_dir,
            base_name=args.base_name,
            title=args.title,
            write_file=args.write_file,
        )
    elif args.command == "draft-nwchem-property-input":
        payload = draft_nwchem_property_check_input(
            input_path=args.input_file,
            reference_output_path=args.reference_output_file,
            vectors_input=args.vectors_input,
            vectors_output=args.vectors_output,
            property_keywords=[
                keyword.strip() for keyword in (args.property_keywords or "").split(",") if keyword.strip()
            ]
            or None,
            task_strategy=args.task_strategy,
            expected_metal_elements=[
                element.strip() for element in (args.expected_metals or "").split(",") if element.strip()
            ]
            or None,
            expected_somo_count=args.expected_somos,
            iterations=args.iterations,
            convergence_energy=args.convergence_energy,
            smear=args.smear,
            output_dir=args.output_dir,
            base_name=args.base_name,
            title=args.title,
            write_file=args.write_file,
        )
    elif args.command == "draft-nwchem-scf-stabilization":
        payload = draft_nwchem_scf_stabilization_input(
            input_path=args.input_file,
            reference_output_path=args.reference_output_file,
            vectors_input=args.vectors_input,
            vectors_output=args.vectors_output,
            task_operation=args.task_operation,
            iterations=args.iterations,
            smear=args.smear,
            convergence_damp=args.convergence_damp,
            convergence_ncydp=args.convergence_ncydp,
            population_print=args.population_print,
            output_dir=args.output_dir,
            base_name=args.base_name,
            title=args.title,
            write_file=args.write_file,
        )
    elif args.command == "draft-nwchem-optimization-followup":
        payload = draft_nwchem_optimization_followup_input(
            output_path=args.output_file,
            input_path=args.input_file,
            task_strategy=args.task_strategy,
            output_dir=args.output_dir,
            base_name=args.base_name,
            title=args.title,
            write_file=args.write_file,
        )
    elif args.command == "draft-nwchem-cube-input":
        payload = draft_nwchem_cube_input(
            input_path=args.input_file,
            vectors_input=args.vectors_input,
            orbital_vectors=[
                int(value.strip()) for value in (args.orbital_vectors or "").split(",") if value.strip()
            ]
            or None,
            density_modes=[
                value.strip() for value in (args.density_modes or "").split(",") if value.strip()
            ]
            or None,
            orbital_spin=args.orbital_spin,
            extent_angstrom=args.extent_angstrom,
            grid_points=args.grid_points,
            gaussian=args.gaussian,
            output_dir=args.output_dir,
            base_name=args.base_name,
            title=args.title,
            write_file=args.write_file,
        )
    elif args.command == "draft-nwchem-frontier-cubes":
        payload = draft_nwchem_frontier_cube_input(
            output_path=args.output_file,
            input_path=args.input_file,
            vectors_input=args.vectors_input,
            include_somos=args.include_somos,
            include_homo=args.include_homo,
            include_lumo=args.include_lumo,
            include_density_modes=[
                value.strip() for value in (args.include_density_modes or "").split(",") if value.strip()
            ]
            or None,
            extent_angstrom=args.extent_angstrom,
            grid_points=args.grid_points,
            gaussian=args.gaussian,
            output_dir=args.output_dir,
            base_name=args.base_name,
            title=args.title,
            write_file=args.write_file,
        )
    elif args.command == "parse-trajectory":
        payload = parse_trajectory(args.file, include_positions=args.include_positions)
    elif args.command == "parse-cube":
        payload = parse_cube(args.file, include_values=args.include_values)
    elif args.command == "summarize-cube":
        payload = summarize_cube(args.file, top_atoms=args.top_atoms)
    elif args.command == "inspect-runner-profiles":
        payload = inspect_runner_profiles(args.profiles_path)
    elif args.command == "prepare-nwchem-run":
        payload = prepare_nwchem_run(
            input_path=args.input_file,
            profile=args.profile,
            profiles_path=args.profiles_path,
            job_name=args.job_name,
            resource_overrides=_parse_key_value_overrides(args.resource_overrides),
            env_overrides=_parse_string_overrides(args.env_overrides),
        )
    elif args.command == "launch-nwchem-run":
        payload = launch_nwchem_run(
            input_path=args.input_file,
            profile=args.profile,
            profiles_path=args.profiles_path,
            job_name=args.job_name,
            resource_overrides=_parse_key_value_overrides(args.resource_overrides),
            env_overrides=_parse_string_overrides(args.env_overrides),
            write_script=args.write_script,
        )
    elif args.command == "check-nwchem-run-status":
        payload = check_nwchem_run_status(
            output_path=args.output_file,
            input_path=args.input_file,
            error_path=args.error_file,
            process_id=args.process_id,
            profile=args.profile,
            job_id=args.job_id,
            profiles_path=args.profiles_path,
        )
    elif args.command == "review-nwchem-progress":
        payload = review_nwchem_progress(
            output_path=args.output_file,
            input_path=args.input_file,
            error_path=args.error_file,
            process_id=args.process_id,
            profile=args.profile,
            job_id=args.job_id,
            profiles_path=args.profiles_path,
        )
    elif args.command == "watch-nwchem-run":
        payload = watch_nwchem_run(
            output_path=args.output_file,
            input_path=args.input_file,
            error_path=args.error_file,
            process_id=args.process_id,
            profile=args.profile,
            job_id=args.job_id,
            profiles_path=args.profiles_path,
            poll_interval_seconds=args.poll_interval_seconds,
            adaptive_polling=args.adaptive_polling,
            max_poll_interval_seconds=args.max_poll_interval_seconds,
            timeout_seconds=args.timeout_seconds,
            max_polls=args.max_polls,
            history_limit=args.history_limit,
        )
    elif args.command == "tail-nwchem-output":
        payload = tail_nwchem_output(
            args.file,
            lines=args.lines,
            max_characters=args.max_characters,
        )
    elif args.command == "terminate-nwchem-run":
        payload = terminate_nwchem_run(
            process_id=args.process_id,
            signal_name=args.signal_name,
        )
    elif args.command == "compare-nwchem-runs":
        payload = compare_nwchem_runs(
            reference_output_path=args.reference_output,
            candidate_output_path=args.candidate_output,
            reference_input_path=args.reference_input,
            candidate_input_path=args.candidate_input,
            expected_metal_elements=[
                element.strip() for element in (args.expected_metals or "").split(",") if element.strip()
            ]
            or None,
            expected_somo_count=args.expected_somos,
        )
    elif args.command == "review-nwchem-followup":
        payload = review_nwchem_followup_outcome(
            reference_output_path=args.reference_output,
            candidate_output_path=args.candidate_output,
            reference_input_path=args.reference_input,
            candidate_input_path=args.candidate_input,
            expected_metal_elements=[
                element.strip() for element in (args.expected_metals or "").split(",") if element.strip()
            ]
            or None,
            expected_somo_count=args.expected_somos,
            output_dir=args.output_dir,
            base_name=args.base_name,
        )
    elif args.command == "eval-cases":
        payload = evaluate_cases(args.path)
    elif args.command == "prepare-nwchem-next-step":
        payload = prepare_nwchem_next_step(
            output_path=args.output_file,
            input_path=args.input_file,
            expected_metal_elements=[
                element.strip() for element in (args.expected_metals or "").split(",") if element.strip()
            ]
            or None,
            expected_somo_count=args.expected_somos,
            output_dir=args.output_dir,
            base_name=args.base_name,
            write_files=args.write_files,
            include_property_check=args.include_property_check,
            include_frontier_cubes=args.include_frontier_cubes,
            include_density_modes=[
                value.strip() for value in (args.include_density_modes or "").split(",") if value.strip()
            ]
            or None,
            cube_extent_angstrom=args.cube_extent_angstrom,
            cube_grid_points=args.cube_grid_points,
        )
    elif args.command == "basis-library-summary":
        payload = basis_library_summary(_basis_library_path(args.library_path))
    elif args.command == "resolve-basis":
        payload = resolve_basis(
            args.basis_name,
            elements=[element.strip() for element in args.elements.split(",") if element.strip()],
            library_path=_basis_library_path(args.library_path),
        )
    elif args.command == "resolve-ecp":
        payload = resolve_ecp(
            args.ecp_name,
            elements=[element.strip() for element in args.elements.split(",") if element.strip()],
            library_path=_basis_library_path(args.library_path),
        )
    elif args.command == "get-basis-blocks":
        payload = get_basis_blocks(
            args.basis_name,
            elements=[element.strip() for element in args.elements.split(",") if element.strip()],
            library_path=_basis_library_path(args.library_path),
        )
    elif args.command == "inspect-nwchem-geometry":
        payload = inspect_nwchem_geometry(args.file)
    elif args.command == "inspect-nwchem-input":
        payload = inspect_input(args.file)
    elif args.command == "lint-nwchem-input":
        payload = lint_nwchem_input(args.input_file, library_path=_basis_library_path(args.library_path))
    elif args.command == "find-restart-assets":
        payload = find_restart_assets(args.path)
    elif args.command == "render-nwchem-basis":
        if args.geometry_file:
            payload = render_basis_block_from_geometry(
                basis_name=args.basis_name,
                input_path=args.geometry_file,
                library_path=_basis_library_path(args.library_path),
                block_name=args.block_name,
                mode=args.mode,
            )
        else:
            if not args.elements:
                raise SystemExit("render-nwchem-basis requires --elements or --geometry-file")
            payload = render_basis_block(
                basis_name=args.basis_name,
                elements=[element.strip() for element in args.elements.split(",") if element.strip()],
                library_path=_basis_library_path(args.library_path),
                block_name=args.block_name,
                mode=args.mode,
            )
    elif args.command == "render-nwchem-ecp":
        payload = render_ecp_block(
            ecp_name=args.ecp_name,
            elements=[element.strip() for element in args.elements.split(",") if element.strip()],
            library_path=_basis_library_path(args.library_path),
        )
    elif args.command == "resolve-nwchem-basis-setup":
        payload = resolve_basis_setup(
            geometry_path=args.geometry_file,
            library_path=_basis_library_path(args.library_path),
            basis_assignments=_parse_string_overrides(args.basis_map) or {},
            ecp_assignments=_parse_string_overrides(args.ecp_map) or {},
            default_basis=args.default_basis,
            default_ecp=args.default_ecp,
        )
    elif args.command == "render-nwchem-basis-setup":
        payload = render_nwchem_basis_setup(
            geometry_path=args.geometry_file,
            library_path=_basis_library_path(args.library_path),
            basis_assignments=_parse_string_overrides(args.basis_map) or {},
            ecp_assignments=_parse_string_overrides(args.ecp_map) or {},
            default_basis=args.default_basis,
            default_ecp=args.default_ecp,
            basis_mode=args.basis_mode,
            basis_block_name=args.block_name,
        )
    elif args.command == "create-nwchem-input":
        payload = create_nwchem_input(
            geometry_path=args.geometry_file,
            library_path=_basis_library_path(args.library_path),
            basis_assignments=_parse_string_overrides(args.basis_map) or {},
            ecp_assignments=_parse_string_overrides(args.ecp_map) or {},
            default_basis=args.default_basis,
            default_ecp=args.default_ecp,
            basis_mode=args.basis_mode,
            basis_block_name=args.block_name,
            module=args.module,
            task_operation=args.task_operation,
            charge=args.charge,
            multiplicity=args.multiplicity,
            memory=args.memory,
            title=args.title,
            start_name=args.start_name,
            vectors_input=args.vectors_input,
            vectors_output=args.vectors_output,
            module_settings=args.module_setting or None,
            extra_blocks=args.extra_block or None,
            output_dir=args.output_dir,
            write_file=args.write_file,
        )
    elif args.command == "review-nwchem-input-request":
        payload = review_nwchem_input_request(
            formula=args.formula,
            geometry_path=args.geometry_file,
            library_path=_basis_library_path(args.library_path),
            basis_assignments=_parse_mapping_arg(args.basis_map),
            ecp_assignments=_parse_mapping_arg(args.ecp_map),
            default_basis=args.default_basis,
            default_ecp=args.default_ecp,
            module=args.module,
            task_operations=_parse_csv_arg(args.tasks),
            functional=args.functional,
            charge=args.charge,
            multiplicity=args.multiplicity,
        )
    elif args.command == "create-nwchem-dft-workflow":
        payload = create_nwchem_dft_workflow_input(
            geometry_path=args.geometry_file,
            library_path=_basis_library_path(args.library_path),
            basis_assignments=_parse_mapping_arg(args.basis_map),
            ecp_assignments=_parse_mapping_arg(args.ecp_map),
            default_basis=args.default_basis,
            default_ecp=args.default_ecp,
            basis_block_name=args.block_name,
            basis_mode=args.basis_mode,
            xc_functional=args.xc_functional,
            task_operations=_parse_csv_arg(args.tasks),
            charge=args.charge,
            multiplicity=args.multiplicity,
            dft_settings=args.dft_setting,
            extra_blocks=args.extra_block,
            memory=args.memory,
            title=args.title,
            start_name=args.start_name,
            vectors_input=args.vectors_input,
            vectors_output=args.vectors_output,
            output_dir=args.output_dir,
            write_file=args.write_file,
        )
    elif args.command == "create-nwchem-dft-from-request":
        payload = create_nwchem_dft_input_from_request(
            formula=args.formula,
            geometry_path=args.geometry_file,
            library_path=_basis_library_path(args.library_path),
            basis_assignments=_parse_string_overrides(args.basis_map),
            ecp_assignments=_parse_string_overrides(args.ecp_map) or None,
            default_basis=args.default_basis,
            default_ecp=args.default_ecp,
            xc_functional=args.xc_functional,
            task_operations=[task.strip() for task in args.tasks.split(",") if task.strip()],
            charge=args.charge,
            multiplicity=args.multiplicity,
            dft_settings=args.dft_setting,
            extra_blocks=args.extra_block,
            memory=args.memory,
            title=args.title,
            start_name=args.start_name,
            vectors_input=args.vectors_input,
            vectors_output=args.vectors_output,
            output_dir=args.output_dir,
            write_file=args.write_file,
        )
    elif args.command == "parse-scf":
        payload = parse_scf(args.file)
    elif args.command == "diagnose-nwchem":
        payload = diagnose_output(
            output_path=args.output_file,
            input_path=args.input_file,
            expected_metal_elements=[
                element.strip() for element in (args.expected_metals or "").split(",") if element.strip()
            ]
            or None,
            expected_somo_count=args.expected_somos,
        )
    elif args.command == "summarize-nwchem":
        payload = summarize_output(
            output_path=args.output_file,
            input_path=args.input_file,
            expected_metal_elements=[
                element.strip() for element in (args.expected_metals or "").split(",") if element.strip()
            ]
            or None,
            expected_somo_count=args.expected_somos,
        )
    elif args.command == "check-spin-charge-state":
        payload = check_spin_charge_state(
            output_path=args.output_file,
            input_path=args.input_file,
            expected_metal_elements=[
                element.strip() for element in (args.expected_metals or "").split(",") if element.strip()
            ]
            or None,
            expected_somo_count=args.expected_somos,
        )
    elif args.command == "suggest-nwchem-scf-fix-strategy":
        payload = suggest_nwchem_scf_fix_strategy(
            output_path=args.output_file,
            input_path=args.input_file,
            expected_metal_elements=[
                element.strip() for element in (args.expected_metals or "").split(",") if element.strip()
            ]
            or None,
            expected_somo_count=args.expected_somos,
        )
    elif args.command == "suggest-nwchem-mcscf-active-space":
        payload = suggest_nwchem_mcscf_active_space(
            output_path=args.output_file,
            input_path=args.input_file,
            expected_metal_elements=[
                element.strip() for element in (args.expected_metals or "").split(",") if element.strip()
            ]
            or None,
            expected_somo_count=args.expected_somos,
        )
    elif args.command == "suggest-nwchem-state-recovery-strategy":
        payload = suggest_nwchem_state_recovery_strategy(
            output_path=args.output_file,
            input_path=args.input_file,
            expected_metal_elements=[
                element.strip() for element in (args.expected_metals or "").split(",") if element.strip()
            ]
            or None,
            expected_somo_count=args.expected_somos,
        )
    elif args.command == "draft-nwchem-mcscf-input":
        payload = draft_nwchem_mcscf_input(
            reference_output_path=args.reference_output_file,
            input_path=args.input_file,
            expected_metal_elements=[
                element.strip() for element in (args.expected_metals or "").split(",") if element.strip()
            ]
            or None,
            expected_somo_count=args.expected_somos,
            active_space_mode=args.active_space_mode,
            vectors_input=args.vectors_input,
            vectors_output=args.vectors_output,
            state_label=args.state_label,
            symmetry=args.symmetry,
            hessian=args.hessian,
            maxiter=args.maxiter,
            thresh=args.thresh,
            level=args.level,
            lock_vectors=not args.no_lock_vectors,
            output_dir=args.output_dir,
            base_name=args.base_name,
            title=args.title,
            write_file=args.write_file,
        )
    elif args.command == "draft-nwchem-mcscf-retry":
        payload = draft_nwchem_mcscf_retry_input(
            output_path=args.output_file,
            input_path=args.input_file,
            expected_metal_elements=_parse_csv_arg(args.expected_metals),
            active_space_mode=args.active_space_mode,
            vectors_input=args.vectors_input,
            vectors_output=args.vectors_output,
            state_label=args.state_label,
            symmetry=args.symmetry,
            hessian=args.hessian,
            maxiter=args.maxiter,
            thresh=args.thresh,
            level=args.level,
            lock_vectors=not args.no_lock_vectors,
            output_dir=args.output_dir,
            base_name=args.base_name,
            title=args.title,
            write_file=args.write_file,
        )
    elif args.command == "summarize-nwchem-case":
        payload = summarize_nwchem_case(
            output_path=args.output_file,
            input_path=args.input_file,
            library_path=_basis_library_path(args.library_path),
            expected_metal_elements=[
                element.strip() for element in (args.expected_metals or "").split(",") if element.strip()
            ]
            or None,
            expected_somo_count=args.expected_somos,
            output_dir=args.output_dir,
            base_name=args.base_name,
            compact=args.compact,
        )
    elif args.command == "review-nwchem-case":
        payload = review_nwchem_case(
            output_path=args.output_file,
            input_path=args.input_file,
            library_path=_basis_library_path(args.library_path),
            expected_metal_elements=[
                element.strip() for element in (args.expected_metals or "").split(",") if element.strip()
            ]
            or None,
            expected_somo_count=args.expected_somos,
            output_dir=args.output_dir,
            base_name=args.base_name,
        )
    else:
        payload = parse_output(
            args.file,
            sections=[section.strip() for section in args.sections.split(",") if section.strip()],
            top_n=args.top_n,
            include_coefficients=args.include_coefficients,
            include_displacements=args.include_displacements,
            include_positions=args.include_positions,
        )

    if args.pretty:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload))


def _parse_key_value_overrides(raw: str) -> dict[str, object] | None:
    if not raw.strip():
        return None
    payload: dict[str, object] = {}
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value.isdigit():
            payload[key] = int(value)
        else:
            try:
                payload[key] = float(value)
            except ValueError:
                payload[key] = value
    return payload or None


def _parse_mapping_arg(raw: str) -> dict[str, str] | None:
    return _parse_string_overrides(raw)


def _parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_string_overrides(raw: str) -> dict[str, str] | None:
    if not raw.strip():
        return None
    payload: dict[str, str] = {}
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key:
            payload[key] = value.strip()
    return payload or None


if __name__ == "__main__":
    main()

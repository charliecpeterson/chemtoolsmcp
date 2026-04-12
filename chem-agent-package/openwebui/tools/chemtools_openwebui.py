from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from chemtools import (
    analyze_frontier_orbitals,
    analyze_imaginary_modes,
    check_nwchem_run_status as check_nwchem_run_status_payload,
    check_spin_charge_state as check_spin_charge_state_payload,
    compare_nwchem_runs as compare_nwchem_runs_payload,
    review_nwchem_followup_outcome as review_nwchem_followup_outcome_payload,
    review_nwchem_mcscf_followup_outcome as review_nwchem_mcscf_followup_outcome_payload,
    review_nwchem_mcscf_case as review_nwchem_mcscf_case_payload,
    create_nwchem_input as create_nwchem_input_payload,
    create_nwchem_dft_input_from_request as create_nwchem_dft_input_from_request_payload,
    create_nwchem_dft_workflow_input as create_nwchem_dft_workflow_input_payload,
    diagnose_output,
    displace_geometry_along_mode,
    draft_nwchem_cube_input as draft_nwchem_cube_input_payload,
    draft_nwchem_frontier_cube_input as draft_nwchem_frontier_cube_input_payload,
    draft_nwchem_imaginary_mode_inputs as draft_nwchem_imaginary_mode_inputs_payload,
    draft_nwchem_mcscf_input as draft_nwchem_mcscf_input_payload,
    draft_nwchem_mcscf_retry_input as draft_nwchem_mcscf_retry_input_payload,
    draft_nwchem_optimization_followup_input as draft_nwchem_optimization_followup_input_payload,
    draft_nwchem_property_check_input as draft_nwchem_property_check_input_payload,
    draft_nwchem_scf_stabilization_input as draft_nwchem_scf_stabilization_input_payload,
    draft_nwchem_vectors_swap_input as draft_nwchem_vectors_swap_input_payload,
    inspect_nwchem_geometry,
    inspect_input,
    inspect_runner_profiles as inspect_runner_profiles_payload,
    find_restart_assets as find_restart_assets_payload,
    launch_nwchem_run as launch_nwchem_run_payload,
    lint_nwchem_input as lint_nwchem_input_payload,
    parse_mcscf_output,
    parse_mos,
    parse_output,
    parse_population_analysis,
    parse_scf,
    parse_cube,
    prepare_nwchem_next_step as prepare_nwchem_next_step_payload,
    prepare_nwchem_run as prepare_nwchem_run_payload,
    render_basis_block,
    render_basis_block_from_geometry,
    render_ecp_block,
    render_nwchem_basis_setup as render_nwchem_basis_setup_payload,
    review_nwchem_progress as review_nwchem_progress_payload,
    watch_nwchem_run as watch_nwchem_run_payload,
    review_nwchem_input_request as review_nwchem_input_request_payload,
    review_nwchem_case as review_nwchem_case_payload,
    resolve_basis,
    resolve_basis_setup as resolve_basis_setup_payload,
    resolve_ecp,
    suggest_nwchem_scf_fix_strategy as suggest_nwchem_scf_fix_strategy_payload,
    suggest_nwchem_mcscf_active_space as suggest_nwchem_mcscf_active_space_payload,
    suggest_nwchem_state_recovery_strategy as suggest_nwchem_state_recovery_strategy_payload,
    suggest_vectors_swaps,
    summarize_cube,
    summarize_nwchem_case as summarize_nwchem_case_payload,
    summarize_output,
    tail_nwchem_output as tail_nwchem_output_payload,
    terminate_nwchem_run as terminate_nwchem_run_payload,
)


DEFAULT_BASIS_LIBRARY = Path("/Users/charlie/test/mytest/nwchem-test/nwchem_basis_library")
DEFAULT_RUNNER_PROFILES = Path("/Users/charlie/test/mytest/chemtools/runner_profiles.example.json")


def _basis_library_path(path: str | None = None) -> str:
    if path:
        return path
    configured = os.environ.get("CHEMTOOLS_BASIS_LIBRARY")
    if configured:
        return configured
    return str(DEFAULT_BASIS_LIBRARY)


def _runner_profiles_path(path: str | None = None) -> str:
    if path:
        return path
    configured = os.environ.get("CHEMTOOLS_RUNNER_PROFILES")
    if configured:
        return configured
    return str(DEFAULT_RUNNER_PROFILES)


def parse_nwchem_output(file_path: str) -> dict[str, Any]:
    """
    Parse a NWChem output file into structured JSON sections.
    """
    return parse_output(file_path, sections=["tasks", "mos", "freq", "trajectory"])


def parse_nwchem_mos(
    file_path: str,
    top_n: int = 5,
    include_coefficients: bool = False,
) -> dict[str, Any]:
    """
    Parse NWChem molecular orbital analysis, including spin-resolved frontier orbitals and atom/AO contributions.
    """
    return parse_mos(file_path, top_n=top_n, include_coefficients=include_coefficients)


def parse_nwchem_population_analysis(file_path: str) -> dict[str, Any]:
    """
    Parse NWChem Mulliken and Lowdin population analysis blocks, including total and spin density tables.
    """
    return parse_population_analysis(file_path)


def parse_nwchem_mcscf_output(file_path: str) -> dict[str, Any]:
    """
    Parse a NWChem MCSCF output for settings, iteration energies, CI convergence, natural occupations, and active-space Mulliken summaries.
    """
    return parse_mcscf_output(file_path)


def review_nwchem_mcscf_case(
    output_file: str,
    input_file: str | None = None,
    expected_metals: list[str] | None = None,
) -> dict[str, Any]:
    """
    Review a NWChem MCSCF run for convergence quality, active-space health, and likely next action.
    """
    return review_nwchem_mcscf_case_payload(
        output_path=output_file,
        input_path=input_file,
        expected_metal_elements=expected_metals,
    )


def review_nwchem_mcscf_followup(
    reference_output_file: str,
    candidate_output_file: str,
    reference_input_file: str | None = None,
    candidate_input_file: str | None = None,
    expected_metals: list[str] | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
) -> dict[str, Any]:
    """
    Compare a follow-up MCSCF run against a reference MCSCF run and summarize whether convergence or active-space quality improved.
    """
    return review_nwchem_mcscf_followup_outcome_payload(
        reference_output_path=reference_output_file,
        candidate_output_path=candidate_output_file,
        reference_input_path=reference_input_file,
        candidate_input_path=candidate_input_file,
        expected_metal_elements=expected_metals,
        output_dir=output_dir,
        base_name=base_name,
    )


def analyze_nwchem_frontier_orbitals(
    output_file: str,
    input_file: str | None = None,
    expected_metals: list[str] | None = None,
    expected_somos: int | None = None,
) -> dict[str, Any]:
    """
    Analyze frontier orbitals and SOMOs to estimate whether the open-shell character is metal-centered or ligand-centered.
    """
    return analyze_frontier_orbitals(
        output_path=output_file,
        input_path=input_file,
        expected_metal_elements=expected_metals,
        expected_somo_count=expected_somos,
    )


def suggest_nwchem_vectors_swaps(
    output_file: str,
    input_file: str | None = None,
    expected_metals: list[str] | None = None,
    expected_somos: int | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
) -> dict[str, Any]:
    """
    Suggest NWChem vectors swap operations to move buried metal-centered orbitals into the SOMO window.
    """
    return suggest_vectors_swaps(
        output_path=output_file,
        input_path=input_file,
        expected_metal_elements=expected_metals,
        expected_somo_count=expected_somos,
        vectors_input=vectors_input,
        vectors_output=vectors_output,
    )


def draft_nwchem_vectors_swap_input(
    output_file: str,
    input_file: str,
    expected_metals: list[str] | None = None,
    expected_somos: int | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    task_operation: str = "energy",
    iterations: int | None = 500,
    smear: float | None = 0.001,
    convergence_damp: int | None = 30,
    convergence_ncydp: int | None = 30,
    population_print: str | None = "mulliken",
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    """
    Draft a NWChem restart input that applies explicit vectors swaps to push metal-centered orbitals into the SOMO window.
    """
    return draft_nwchem_vectors_swap_input_payload(
        output_path=output_file,
        input_path=input_file,
        expected_metal_elements=expected_metals,
        expected_somo_count=expected_somos,
        vectors_input=vectors_input,
        vectors_output=vectors_output,
        task_operation=task_operation,
        iterations=iterations,
        smear=smear,
        convergence_damp=convergence_damp,
        convergence_ncydp=convergence_ncydp,
        population_print=population_print,
        output_dir=output_dir,
        base_name=base_name,
        title=title,
        write_file=write_file,
    )


def draft_nwchem_property_check_input(
    input_file: str,
    reference_output_file: str | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    property_keywords: list[str] | None = None,
    task_strategy: str = "auto",
    expected_metals: list[str] | None = None,
    expected_somos: int | None = None,
    iterations: int | None = 1,
    convergence_energy: str | None = "1e3",
    smear: float | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    """
    Draft a one-step NWChem property input around a chosen movecs file for Mulliken/Lowdin/state inspection.
    """
    return draft_nwchem_property_check_input_payload(
        input_path=input_file,
        reference_output_path=reference_output_file,
        vectors_input=vectors_input,
        vectors_output=vectors_output,
        property_keywords=property_keywords,
        task_strategy=task_strategy,
        expected_metal_elements=expected_metals,
        expected_somo_count=expected_somos,
        iterations=iterations,
        convergence_energy=convergence_energy,
        smear=smear,
        output_dir=output_dir,
        base_name=base_name,
        title=title,
        write_file=write_file,
    )


def draft_nwchem_scf_stabilization_input(
    input_file: str,
    reference_output_file: str | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    task_operation: str = "energy",
    iterations: int | None = None,
    smear: float | None = None,
    convergence_damp: int | None = None,
    convergence_ncydp: int | None = None,
    population_print: str | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    """
    Draft a safer SCF stabilization restart from an existing movecs path when a state-check or SCF retry still fails.
    """
    return draft_nwchem_scf_stabilization_input_payload(
        input_path=input_file,
        reference_output_path=reference_output_file,
        vectors_input=vectors_input,
        vectors_output=vectors_output,
        task_operation=task_operation,
        iterations=iterations,
        smear=smear,
        convergence_damp=convergence_damp,
        convergence_ncydp=convergence_ncydp,
        population_print=population_print,
        output_dir=output_dir,
        base_name=base_name,
        title=title,
        write_file=write_file,
    )


def draft_nwchem_optimization_followup_input(
    output_file: str,
    input_file: str,
    task_strategy: str = "auto",
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    """
    Draft a NWChem follow-up input from the last optimized geometry, either to continue optimization or to run frequency only.
    """
    return draft_nwchem_optimization_followup_input_payload(
        output_path=output_file,
        input_path=input_file,
        task_strategy=task_strategy,
        output_dir=output_dir,
        base_name=base_name,
        title=title,
        write_file=write_file,
    )


def draft_nwchem_cube_input(
    input_file: str,
    vectors_input: str,
    orbital_vectors: list[int] | None = None,
    density_modes: list[str] | None = None,
    orbital_spin: str = "total",
    extent_angstrom: float = 6.0,
    grid_points: int = 120,
    gaussian: bool = True,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    """
    Draft a NWChem dplot input for orbital, density, or spin-density cube generation from a chosen movecs file.
    """
    return draft_nwchem_cube_input_payload(
        input_path=input_file,
        vectors_input=vectors_input,
        orbital_vectors=orbital_vectors,
        density_modes=density_modes,
        orbital_spin=orbital_spin,
        extent_angstrom=extent_angstrom,
        grid_points=grid_points,
        gaussian=gaussian,
        output_dir=output_dir,
        base_name=base_name,
        title=title,
        write_file=write_file,
    )


def draft_nwchem_frontier_cube_input(
    output_file: str,
    input_file: str,
    vectors_input: str | None = None,
    include_somos: bool = True,
    include_homo: bool = True,
    include_lumo: bool = True,
    include_density_modes: list[str] | None = None,
    extent_angstrom: float = 6.0,
    grid_points: int = 120,
    gaussian: bool = True,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    """
    Draft a NWChem dplot input for SOMO/HOMO/LUMO cubes inferred from parsed frontier orbitals.
    """
    return draft_nwchem_frontier_cube_input_payload(
        output_path=output_file,
        input_path=input_file,
        vectors_input=vectors_input,
        include_somos=include_somos,
        include_homo=include_homo,
        include_lumo=include_lumo,
        include_density_modes=include_density_modes,
        extent_angstrom=extent_angstrom,
        grid_points=grid_points,
        gaussian=gaussian,
        output_dir=output_dir,
        base_name=base_name,
        title=title,
        write_file=write_file,
    )


def parse_cube_file(file_path: str, include_values: bool = False) -> dict[str, Any]:
    """
    Parse a Gaussian/NWChem cube file header and volumetric grid metadata.
    """
    return parse_cube(file_path, include_values=include_values)


def summarize_cube_file(file_path: str, top_atoms: int = 5) -> dict[str, Any]:
    """
    Summarize a cube file by grid statistics and approximate atom-localized density/orbital lobes.
    """
    return summarize_cube(file_path, top_atoms=top_atoms)


def inspect_nwchem_input_geometry(input_file: str) -> dict[str, Any]:
    """
    Return the unique element list found across geometry blocks in a NWChem input.
    """
    return inspect_nwchem_geometry(input_file)


def inspect_nwchem_input(input_file: str) -> dict[str, Any]:
    """
    Return input-side chemistry metadata such as charge, multiplicity, selected tasks, and transition metals.
    """
    return inspect_input(input_file)


def inspect_nwchem_runner_profiles(profiles_path: str | None = None) -> dict[str, Any]:
    """
    List the available NWChem runner profiles and their launcher kinds.
    """
    return inspect_runner_profiles_payload(_runner_profiles_path(profiles_path))


def prepare_nwchem_run(
    input_file: str,
    profile: str,
    profiles_path: str | None = None,
    job_name: str | None = None,
    resource_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Render the command or scheduler script that would be used to run a NWChem input with a named runner profile.
    """
    return prepare_nwchem_run_payload(
        input_path=input_file,
        profile=profile,
        profiles_path=_runner_profiles_path(profiles_path),
        job_name=job_name,
        resource_overrides=resource_overrides,
        env_overrides=env_overrides,
    )


def launch_nwchem_run(
    input_file: str,
    profile: str,
    profiles_path: str | None = None,
    job_name: str | None = None,
    resource_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, str] | None = None,
    write_script: bool = True,
) -> dict[str, Any]:
    """
    Launch a NWChem input through a named runner profile. Use prepare_nwchem_run first when previewing a job is preferred.
    """
    return launch_nwchem_run_payload(
        input_path=input_file,
        profile=profile,
        profiles_path=_runner_profiles_path(profiles_path),
        job_name=job_name,
        resource_overrides=resource_overrides,
        env_overrides=env_overrides,
        write_script=write_script,
    )


def check_nwchem_run_status(
    output_file: str | None = None,
    input_file: str | None = None,
    error_file: str | None = None,
    process_id: int | None = None,
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    """
    Inspect process/file-based status for a NWChem run and summarize parsed output outcome when available.
    """
    return check_nwchem_run_status_payload(
        output_path=output_file,
        input_path=input_file,
        error_path=error_file,
        process_id=process_id,
        profile=profile,
        job_id=job_id,
        profiles_path=_runner_profiles_path(profiles_path) if profiles_path or profile else profiles_path,
    )


def review_nwchem_progress(
    output_file: str,
    input_file: str | None = None,
    error_file: str | None = None,
    process_id: int | None = None,
    profile: str | None = None,
    job_id: str | None = None,
    profiles_path: str | None = None,
) -> dict[str, Any]:
    """
    Return a compact task-aware progress summary for an incomplete or running NWChem job.
    """
    return review_nwchem_progress_payload(
        output_path=output_file,
        input_path=input_file,
        error_path=error_file,
        process_id=process_id,
        profile=profile,
        job_id=job_id,
        profiles_path=_runner_profiles_path(profiles_path) if profiles_path or profile else profiles_path,
    )


def watch_nwchem_run(
    output_file: str | None = None,
    input_file: str | None = None,
    error_file: str | None = None,
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
    """
    Poll NWChem status until the run reaches a terminal state or a timeout/max-poll limit.
    """
    return watch_nwchem_run_payload(
        output_path=output_file,
        input_path=input_file,
        error_path=error_file,
        process_id=process_id,
        profile=profile,
        job_id=job_id,
        profiles_path=_runner_profiles_path(profiles_path) if profiles_path or profile else profiles_path,
        poll_interval_seconds=poll_interval_seconds,
        adaptive_polling=adaptive_polling,
        max_poll_interval_seconds=max_poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        max_polls=max_polls,
        history_limit=history_limit,
    )


def terminate_nwchem_run(
    process_id: int,
    signal_name: str = "term",
) -> dict[str, Any]:
    """
    Send SIGTERM or SIGKILL to a local NWChem process when an intervention has already decided the run should stop.
    """
    return terminate_nwchem_run_payload(process_id=process_id, signal_name=signal_name)


def tail_nwchem_output(
    output_file: str,
    lines: int = 30,
    max_characters: int = 4000,
) -> dict[str, Any]:
    """
    Return the tail of a NWChem output file for quick inspection.
    """
    return tail_nwchem_output_payload(output_file, lines=lines, max_characters=max_characters)


def compare_nwchem_runs(
    reference_output_file: str,
    candidate_output_file: str,
    reference_input_file: str | None = None,
    candidate_input_file: str | None = None,
    expected_metals: list[str] | None = None,
    expected_somos: int | None = None,
) -> dict[str, Any]:
    """
    Compare two NWChem runs by diagnosis, task outcome, and energy change.
    """
    return compare_nwchem_runs_payload(
        reference_output_path=reference_output_file,
        candidate_output_path=candidate_output_file,
        reference_input_path=reference_input_file,
        candidate_input_path=candidate_input_file,
        expected_metal_elements=expected_metals,
        expected_somo_count=expected_somos,
    )


def review_nwchem_followup(
    reference_output_file: str,
    candidate_output_file: str,
    reference_input_file: str | None = None,
    candidate_input_file: str | None = None,
    expected_metals: list[str] | None = None,
    expected_somos: int | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
) -> dict[str, Any]:
    """
    Compare a follow-up run against a reference run and summarize whether SCF behavior or state quality improved.
    """
    return review_nwchem_followup_outcome_payload(
        reference_output_path=reference_output_file,
        candidate_output_path=candidate_output_file,
        reference_input_path=reference_input_file,
        candidate_input_path=candidate_input_file,
        expected_metal_elements=expected_metals,
        expected_somo_count=expected_somos,
        output_dir=output_dir,
        base_name=base_name,
    )


def prepare_nwchem_next_step(
    output_file: str,
    input_file: str | None = None,
    expected_metals: list[str] | None = None,
    expected_somos: int | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
    write_files: bool = False,
    include_property_check: bool = True,
    include_frontier_cubes: bool = False,
    include_density_modes: list[str] | None = None,
    cube_extent_angstrom: float = 6.0,
    cube_grid_points: int = 120,
) -> dict[str, Any]:
    """
    Diagnose a NWChem output and prepare the most likely next artifact, such as swap restarts or imaginary-mode follow-up inputs.
    """
    return prepare_nwchem_next_step_payload(
        output_path=output_file,
        input_path=input_file,
        expected_metal_elements=expected_metals,
        expected_somo_count=expected_somos,
        output_dir=output_dir,
        base_name=base_name,
        write_files=write_files,
        include_property_check=include_property_check,
        include_frontier_cubes=include_frontier_cubes,
        include_density_modes=include_density_modes,
        cube_extent_angstrom=cube_extent_angstrom,
        cube_grid_points=cube_grid_points,
    )


def parse_nwchem_scf(file_path: str) -> dict[str, Any]:
    """
    Parse the NWChem SCF/DFT iteration table and convergence pattern.
    """
    return parse_scf(file_path)


def lint_nwchem_input(
    input_file: str,
    library_path: str | None = None,
) -> dict[str, Any]:
    """
    Lint a NWChem input for missing tasks, missing module blocks, basis/ECP coverage, and movecs policy issues.
    """
    return lint_nwchem_input_payload(
        input_path=input_file,
        library_path=_basis_library_path(library_path),
    )


def find_nwchem_restart_assets(path: str) -> dict[str, Any]:
    """
    Discover related restart assets in a job directory, including movecs, db, xyz, input, output, and cube files.
    """
    return find_restart_assets_payload(path)


def resolve_nwchem_basis(
    basis_name: str,
    elements: list[str],
    library_path: str | None = None,
) -> dict[str, Any]:
    """
    Check whether a requested basis exists in the local NWChem basis library and covers the requested elements.
    """
    return resolve_basis(
        basis_name=basis_name,
        elements=elements,
        library_path=_basis_library_path(library_path),
    )


def resolve_nwchem_ecp(
    ecp_name: str,
    elements: list[str],
    library_path: str | None = None,
) -> dict[str, Any]:
    """
    Resolve a local NWChem ECP library entry for the requested elements.
    """
    return resolve_ecp(
        ecp_name=ecp_name,
        elements=elements,
        library_path=_basis_library_path(library_path),
    )


def render_nwchem_basis_from_elements(
    basis_name: str,
    elements: list[str],
    library_path: str | None = None,
    block_name: str = "ao basis",
    mode: str | None = None,
) -> dict[str, Any]:
    """
    Render an explicit per-element NWChem basis block after validating basis coverage.
    """
    return render_basis_block(
        basis_name=basis_name,
        elements=elements,
        library_path=_basis_library_path(library_path),
        block_name=block_name,
        mode=mode,
    )


def render_nwchem_basis_from_input(
    basis_name: str,
    input_file: str,
    library_path: str | None = None,
    block_name: str = "ao basis",
    mode: str | None = None,
) -> dict[str, Any]:
    """
    Inspect the geometry in an input file and build an explicit per-element NWChem basis block.
    """
    return render_basis_block_from_geometry(
        basis_name=basis_name,
        input_path=input_file,
        library_path=_basis_library_path(library_path),
        block_name=block_name,
        mode=mode,
    )


def render_nwchem_ecp_from_elements(
    ecp_name: str,
    elements: list[str],
    library_path: str | None = None,
) -> dict[str, Any]:
    """
    Render an explicit NWChem ECP block using a local ECP library entry.
    """
    return render_ecp_block(
        ecp_name=ecp_name,
        elements=elements,
        library_path=_basis_library_path(library_path),
    )


def resolve_nwchem_basis_setup(
    geometry_file: str,
    basis_assignments: dict[str, str],
    ecp_assignments: dict[str, str] | None = None,
    default_basis: str | None = None,
    default_ecp: str | None = None,
    library_path: str | None = None,
) -> dict[str, Any]:
    """
    Resolve mixed per-element basis/ECP assignments for a geometry source before drafting an input.
    """
    return resolve_basis_setup_payload(
        geometry_path=geometry_file,
        library_path=_basis_library_path(library_path),
        basis_assignments=basis_assignments,
        ecp_assignments=ecp_assignments,
        default_basis=default_basis,
        default_ecp=default_ecp,
    )


def render_nwchem_basis_setup(
    geometry_file: str,
    basis_assignments: dict[str, str],
    ecp_assignments: dict[str, str] | None = None,
    default_basis: str | None = None,
    default_ecp: str | None = None,
    block_name: str = "ao basis",
    basis_mode: str | None = None,
    library_path: str | None = None,
) -> dict[str, Any]:
    """
    Render mixed per-element NWChem basis and ECP blocks from the local library.
    """
    return render_nwchem_basis_setup_payload(
        geometry_path=geometry_file,
        library_path=_basis_library_path(library_path),
        basis_assignments=basis_assignments,
        ecp_assignments=ecp_assignments,
        default_basis=default_basis,
        default_ecp=default_ecp,
        basis_block_name=block_name,
        basis_mode=basis_mode,
    )


def create_nwchem_input(
    geometry_file: str,
    basis_assignments: dict[str, str],
    module: str,
    task_operation: str | None = None,
    ecp_assignments: dict[str, str] | None = None,
    default_basis: str | None = None,
    default_ecp: str | None = None,
    block_name: str = "ao basis",
    basis_mode: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    module_settings: list[str] | None = None,
    extra_blocks: list[str] | None = None,
    memory: str | None = None,
    title: str | None = None,
    start_name: str | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    output_dir: str | None = None,
    write_file: bool = False,
    library_path: str | None = None,
) -> dict[str, Any]:
    """
    Create a new NWChem input from a geometry source with explicit mixed basis/ECP assignments and automatic movecs output for SCF/DFT.
    """
    return create_nwchem_input_payload(
        geometry_path=geometry_file,
        library_path=_basis_library_path(library_path),
        basis_assignments=basis_assignments,
        ecp_assignments=ecp_assignments,
        default_basis=default_basis,
        default_ecp=default_ecp,
        basis_block_name=block_name,
        basis_mode=basis_mode,
        module=module,
        task_operation=task_operation,
        charge=charge,
        multiplicity=multiplicity,
        module_settings=module_settings,
        extra_blocks=extra_blocks,
        memory=memory,
        title=title,
        start_name=start_name,
        vectors_input=vectors_input,
        vectors_output=vectors_output,
        output_dir=output_dir,
        write_file=write_file,
    )


def review_nwchem_input_request(
    formula: str | None = None,
    geometry_file: str | None = None,
    basis_assignments: dict[str, str] | None = None,
    ecp_assignments: dict[str, str] | None = None,
    default_basis: str | None = None,
    default_ecp: str | None = None,
    module: str = "dft",
    task_operations: list[str] | None = None,
    functional: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    library_path: str | None = None,
) -> dict[str, Any]:
    """
    Review whether an NWChem input request has enough information to draft safely, and identify missing geometry/state details instead of guessing.
    """
    return review_nwchem_input_request_payload(
        formula=formula,
        geometry_path=geometry_file,
        library_path=_basis_library_path(library_path),
        basis_assignments=basis_assignments,
        ecp_assignments=ecp_assignments,
        default_basis=default_basis,
        default_ecp=default_ecp,
        module=module,
        task_operations=task_operations,
        functional=functional,
        charge=charge,
        multiplicity=multiplicity,
    )


def create_nwchem_dft_workflow_input(
    geometry_file: str,
    basis_assignments: dict[str, str],
    xc_functional: str,
    task_operations: list[str],
    ecp_assignments: dict[str, str] | None = None,
    default_basis: str | None = None,
    default_ecp: str | None = None,
    block_name: str = "ao basis",
    basis_mode: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    dft_settings: list[str] | None = None,
    extra_blocks: list[str] | None = None,
    memory: str | None = None,
    title: str | None = None,
    start_name: str | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    output_dir: str | None = None,
    write_file: bool = False,
    library_path: str | None = None,
) -> dict[str, Any]:
    """
    Create a standard NWChem DFT workflow input, such as optimize+freq, with explicit basis/ECP blocks and automatic movecs output.
    """
    return create_nwchem_dft_workflow_input_payload(
        geometry_path=geometry_file,
        library_path=_basis_library_path(library_path),
        basis_assignments=basis_assignments,
        ecp_assignments=ecp_assignments,
        default_basis=default_basis,
        default_ecp=default_ecp,
        basis_block_name=block_name,
        basis_mode=basis_mode,
        xc_functional=xc_functional,
        task_operations=task_operations,
        charge=charge,
        multiplicity=multiplicity,
        dft_settings=dft_settings,
        extra_blocks=extra_blocks,
        memory=memory,
        title=title,
        start_name=start_name,
        vectors_input=vectors_input,
        vectors_output=vectors_output,
        output_dir=output_dir,
        write_file=write_file,
    )


def create_nwchem_dft_input_from_request(
    formula: str | None = None,
    geometry_file: str | None = None,
    basis_assignments: dict[str, str] | None = None,
    ecp_assignments: dict[str, str] | None = None,
    default_basis: str | None = None,
    default_ecp: str | None = None,
    xc_functional: str | None = None,
    task_operations: list[str] | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    dft_settings: list[str] | None = None,
    extra_blocks: list[str] | None = None,
    memory: str | None = None,
    title: str | None = None,
    start_name: str | None = None,
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    output_dir: str | None = None,
    write_file: bool = False,
    library_path: str | None = None,
) -> dict[str, Any]:
    """
    Review a DFT input request and create the deterministic NWChem input only when the request is complete enough.
    """
    return create_nwchem_dft_input_from_request_payload(
        formula=formula,
        geometry_path=geometry_file,
        library_path=_basis_library_path(library_path),
        basis_assignments=basis_assignments,
        ecp_assignments=ecp_assignments,
        default_basis=default_basis,
        default_ecp=default_ecp,
        xc_functional=xc_functional,
        task_operations=task_operations,
        charge=charge,
        multiplicity=multiplicity,
        dft_settings=dft_settings,
        extra_blocks=extra_blocks,
        memory=memory,
        title=title,
        start_name=start_name,
        vectors_input=vectors_input,
        vectors_output=vectors_output,
        output_dir=output_dir,
        write_file=write_file,
    )


def diagnose_nwchem_output(
    output_file: str,
    input_file: str | None = None,
    expected_metals: list[str] | None = None,
    expected_somos: int | None = None,
) -> dict[str, Any]:
    """
    Produce a structured first-pass diagnosis for NWChem outputs, including SCF pattern and likely wrong-state checks.
    """
    return diagnose_output(
        output_path=output_file,
        input_path=input_file,
        expected_metal_elements=expected_metals,
        expected_somo_count=expected_somos,
    )


def summarize_nwchem_output(
    output_file: str,
    input_file: str | None = None,
    expected_metals: list[str] | None = None,
    expected_somos: int | None = None,
) -> dict[str, Any]:
    """
    Produce a concise human-readable summary plus structured diagnosis for a NWChem output.
    """
    return summarize_output(
        output_path=output_file,
        input_path=input_file,
        expected_metal_elements=expected_metals,
        expected_somo_count=expected_somos,
    )


def check_nwchem_spin_charge_state(
    output_file: str,
    input_file: str | None = None,
    expected_metals: list[str] | None = None,
    expected_somos: int | None = None,
) -> dict[str, Any]:
    """
    Check whether the parsed frontier orbitals and spin density are chemically plausible for the requested charge and multiplicity.
    """
    return check_spin_charge_state_payload(
        output_path=output_file,
        input_path=input_file,
        expected_metal_elements=expected_metals,
        expected_somo_count=expected_somos,
    )


def suggest_nwchem_scf_fix_strategy(
    output_file: str,
    input_file: str | None = None,
    expected_metals: list[str] | None = None,
    expected_somos: int | None = None,
) -> dict[str, Any]:
    """
    Suggest ranked SCF recovery strategies for a failed or questionable NWChem run.
    """
    return suggest_nwchem_scf_fix_strategy_payload(
        output_path=output_file,
        input_path=input_file,
        expected_metal_elements=expected_metals,
        expected_somo_count=expected_somos,
    )


def suggest_nwchem_mcscf_active_space(
    output_file: str,
    input_file: str | None = None,
    expected_metals: list[str] | None = None,
    expected_somos: int | None = None,
) -> dict[str, Any]:
    """
    Suggest minimal and expanded MCSCF active spaces from the current MO/state picture.
    """
    return suggest_nwchem_mcscf_active_space_payload(
        output_path=output_file,
        input_path=input_file,
        expected_metal_elements=expected_metals,
        expected_somo_count=expected_somos,
    )


def draft_nwchem_mcscf_input(
    reference_output_file: str,
    input_file: str,
    expected_metals: list[str] | None = None,
    expected_somos: int | None = None,
    active_space_mode: str = "minimal",
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    state_label: str | None = None,
    symmetry: int | None = None,
    hessian: str = "exact",
    maxiter: int = 80,
    thresh: float | None = 1.0e-5,
    level: float | None = 0.6,
    lock_vectors: bool = True,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    """
    Draft a NWChem MCSCF input using the recommended active space and vector reordering plan.
    """
    return draft_nwchem_mcscf_input_payload(
        reference_output_path=reference_output_file,
        input_path=input_file,
        expected_metal_elements=expected_metals,
        expected_somo_count=expected_somos,
        active_space_mode=active_space_mode,
        vectors_input=vectors_input,
        vectors_output=vectors_output,
        state_label=state_label,
        symmetry=symmetry,
        hessian=hessian,
        maxiter=maxiter,
        thresh=thresh,
        level=level,
        lock_vectors=lock_vectors,
        output_dir=output_dir,
        base_name=base_name,
        title=title,
        write_file=write_file,
    )


def draft_nwchem_mcscf_retry_input(
    output_file: str,
    input_file: str,
    expected_metals: list[str] | None = None,
    active_space_mode: str = "auto",
    vectors_input: str | None = None,
    vectors_output: str | None = None,
    state_label: str | None = None,
    symmetry: int | None = None,
    hessian: str | None = None,
    maxiter: int | None = None,
    thresh: float | None = None,
    level: float | None = None,
    lock_vectors: bool = True,
    output_dir: str | None = None,
    base_name: str | None = None,
    title: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    """
    Draft a refined NWChem MCSCF retry input after reviewing a failed or stiff prior MCSCF run.
    """
    return draft_nwchem_mcscf_retry_input_payload(
        output_path=output_file,
        input_path=input_file,
        expected_metal_elements=expected_metals,
        active_space_mode=active_space_mode,
        vectors_input=vectors_input,
        vectors_output=vectors_output,
        state_label=state_label,
        symmetry=symmetry,
        hessian=hessian,
        maxiter=maxiter,
        thresh=thresh,
        level=level,
        lock_vectors=lock_vectors,
        output_dir=output_dir,
        base_name=base_name,
        title=title,
        write_file=write_file,
    )


def suggest_nwchem_state_recovery_strategy(
    output_file: str,
    input_file: str | None = None,
    expected_metals: list[str] | None = None,
    expected_somos: int | None = None,
) -> dict[str, Any]:
    """
    Suggest ranked wrong-state or ambiguous-state recovery strategies for a NWChem run.
    """
    return suggest_nwchem_state_recovery_strategy_payload(
        output_path=output_file,
        input_path=input_file,
        expected_metal_elements=expected_metals,
        expected_somo_count=expected_somos,
    )


def summarize_nwchem_case(
    output_file: str,
    input_file: str | None = None,
    library_path: str | None = None,
    expected_metals: list[str] | None = None,
    expected_somos: int | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    """
    Produce a one-shot NWChem case summary with diagnosis, lint, restart assets, spin-state review, and next-step planning.
    """
    return summarize_nwchem_case_payload(
        output_path=output_file,
        input_path=input_file,
        library_path=_basis_library_path(library_path),
        expected_metal_elements=expected_metals,
        expected_somo_count=expected_somos,
        output_dir=output_dir,
        base_name=base_name,
        compact=compact,
    )


def review_nwchem_case(
    output_file: str,
    input_file: str | None = None,
    library_path: str | None = None,
    expected_metals: list[str] | None = None,
    expected_somos: int | None = None,
    output_dir: str | None = None,
    base_name: str | None = None,
) -> dict[str, Any]:
    """
    Produce the compact agent-facing NWChem case review payload for routine case triage.
    """
    return review_nwchem_case_payload(
        output_path=output_file,
        input_path=input_file,
        library_path=_basis_library_path(library_path),
        expected_metal_elements=expected_metals,
        expected_somo_count=expected_somos,
        output_dir=output_dir,
        base_name=base_name,
    )


def analyze_nwchem_imaginary_modes(
    output_file: str,
    significant_threshold_cm1: float = 20.0,
    top_atoms: int = 4,
) -> dict[str, Any]:
    """
    Analyze significant imaginary modes in a NWChem frequency output and identify the atoms dominating the motion.
    """
    return analyze_imaginary_modes(
        output_file,
        significant_threshold_cm1=significant_threshold_cm1,
        top_atoms=top_atoms,
    )


def displace_nwchem_geometry_along_mode(
    output_file: str,
    mode_number: int | None = None,
    amplitude_angstrom: float = 0.15,
    significant_threshold_cm1: float = 20.0,
) -> dict[str, Any]:
    """
    Generate plus/minus displaced geometries from an imaginary mode in a NWChem output.
    """
    return displace_geometry_along_mode(
        output_file,
        mode_number=mode_number,
        amplitude_angstrom=amplitude_angstrom,
        significant_threshold_cm1=significant_threshold_cm1,
    )


def draft_nwchem_imaginary_mode_inputs(
    output_file: str,
    input_file: str,
    mode_number: int | None = None,
    amplitude_angstrom: float = 0.15,
    significant_threshold_cm1: float = 20.0,
    task_strategy: str = "auto",
    output_dir: str | None = None,
    base_name: str | None = None,
    write_files: bool = False,
    noautosym: bool = True,
    symmetry_c1: bool = True,
) -> dict[str, Any]:
    """
    Draft plus/minus NWChem input texts by replacing the input geometry with displacements along an imaginary mode.
    """
    return draft_nwchem_imaginary_mode_inputs_payload(
        output_path=output_file,
        input_path=input_file,
        mode_number=mode_number,
        amplitude_angstrom=amplitude_angstrom,
        significant_threshold_cm1=significant_threshold_cm1,
        add_noautosym=noautosym,
        enforce_symmetry_c1=symmetry_c1,
        task_strategy=task_strategy,
        output_dir=output_dir,
        base_name=base_name,
        write_files=write_files,
    )

"""NWChem MCP handlers — analysis.

Split from mcp/tools/nwchem.py by category. Shared imports/helpers live in
_nwchem_base (pulled in below); nwchem.py imports this module so its @_tool
handlers register.
"""
from __future__ import annotations

from chemtools.mcp.tools._nwchem_base import *  # noqa: F401,F403
from chemtools.mcp.tools._nwchem_base import _tool, _build_next_actions  # noqa: F401


@_tool("track_nwchem_spin_state")
def _handle_track_spin_state(arguments: dict[str, Any]) -> dict[str, Any]:
    result = track_spin_state_across_optimization(
        output_path=arguments["output_file"],
    )
    result["next_actions"] = _build_next_actions(
        "track_spin_state", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("analyze_nwchem_frontier_orbitals")
def _handle_analyze_nwchem_frontier_orbitals(arguments: dict[str, Any]) -> dict[str, Any]:
    result = analyze_frontier_orbitals(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
    )
    result["next_actions"] = _build_next_actions(
        "frontier_orbitals", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("suggest_nwchem_vectors_swaps")
def _handle_suggest_nwchem_vectors_swaps(arguments: dict[str, Any]) -> dict[str, Any]:
    return suggest_vectors_swaps(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
        vectors_input=arguments.get("vectors_input"),
        vectors_output=arguments.get("vectors_output"),
    )


@_tool("compare_nwchem_runs")
def _handle_compare_nwchem_runs(arguments: dict[str, Any]) -> dict[str, Any]:
    if arguments.get("output_dir") or arguments.get("base_name"):
        result = review_nwchem_followup_outcome(
            reference_output_path=arguments["reference_output_file"],
            candidate_output_path=arguments["candidate_output_file"],
            reference_input_path=arguments.get("reference_input_file"),
            candidate_input_path=arguments.get("candidate_input_file"),
            expected_metal_elements=arguments.get("expected_metals"),
            expected_somo_count=arguments.get("expected_somos"),
            output_dir=arguments.get("output_dir"),
            base_name=arguments.get("base_name"),
        )
    else:
        result = compare_nwchem_runs(
            reference_output_path=arguments["reference_output_file"],
            candidate_output_path=arguments["candidate_output_file"],
            reference_input_path=arguments.get("reference_input_file"),
            candidate_input_path=arguments.get("candidate_input_file"),
            expected_metal_elements=arguments.get("expected_metals"),
            expected_somo_count=arguments.get("expected_somos"),
        )
    result["next_actions"] = _build_next_actions(
        "compare_runs", result,
        output_file=arguments["candidate_output_file"],
        input_file=arguments.get("candidate_input_file", ""),
    )
    return result


@_tool("review_nwchem_mcscf_case")
def _handle_review_nwchem_mcscf_case(arguments: dict[str, Any]) -> dict[str, Any]:
    return review_nwchem_mcscf_case(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
    )


@_tool("review_nwchem_mcscf_followup_outcome")
def _handle_review_nwchem_mcscf_followup_outcome(arguments: dict[str, Any]) -> dict[str, Any]:
    return review_nwchem_mcscf_followup_outcome(
        reference_output_path=arguments["reference_output_file"],
        candidate_output_path=arguments["candidate_output_file"],
        reference_input_path=arguments.get("reference_input_file"),
        candidate_input_path=arguments.get("candidate_input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
    )


@_tool("prepare_nwchem_next_step")
def _handle_prepare_nwchem_next_step(arguments: dict[str, Any]) -> dict[str, Any]:
    return prepare_nwchem_next_step(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        write_files=arguments.get("write_files", False),
        include_property_check=arguments.get("include_property_check", True),
        include_frontier_cubes=arguments.get("include_frontier_cubes", False),
        include_density_modes=arguments.get("include_density_modes"),
        cube_extent_angstrom=arguments.get("cube_extent_angstrom", 6.0),
        cube_grid_points=arguments.get("cube_grid_points", 120),
    )


# ---------------------------------------------------------------------------
# Handlers — basis and ECP
# ---------------------------------------------------------------------------


@_tool("analyze_nwchem_case")
def _handle_analyze_nwchem_case(arguments: dict[str, Any]) -> dict[str, Any]:
    compact = arguments.get("detail", "compact") == "compact"
    result = summarize_nwchem_case(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        err_file=arguments.get("err_file"),
        library_path=basis_library_path(arguments.get("library_path")),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        compact=compact,
    )
    result["next_actions"] = _build_next_actions(
        "analyze_case", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
        profile=arguments.get("profile", ""),
    )
    return result


@_tool("suggest_nwchem_recovery")
def _handle_suggest_nwchem_recovery(arguments: dict[str, Any]) -> dict[str, Any]:
    mode = arguments.get("mode", "auto")
    kwargs = dict(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
    )
    if mode == "scf":
        return suggest_nwchem_scf_fix_strategy(**kwargs)
    if mode == "state":
        return suggest_nwchem_state_recovery_strategy(**kwargs)
    # auto: return both
    result: dict[str, Any] = {}
    try:
        result["scf_strategies"] = suggest_nwchem_scf_fix_strategy(**kwargs)
    except Exception as exc:
        result["scf_strategies"] = {"error": str(exc)}
    try:
        result["state_strategies"] = suggest_nwchem_state_recovery_strategy(**kwargs)
    except Exception as exc:
        result["state_strategies"] = {"error": str(exc)}
    return result


# ---------------------------------------------------------------------------
# Handlers — MCSCF
# ---------------------------------------------------------------------------


@_tool("suggest_nwchem_mcscf_active_space")
def _handle_suggest_nwchem_mcscf_active_space(arguments: dict[str, Any]) -> dict[str, Any]:
    return suggest_nwchem_mcscf_active_space(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
    )


@_tool("check_nwchem_geometry_plausibility")
def _handle_check_nwchem_geometry_plausibility(arguments: dict[str, Any]) -> dict[str, Any]:
    frame_arg = arguments.get("frame", "best")
    try:
        frame_arg = int(frame_arg)
    except (TypeError, ValueError):
        pass
    result = check_nwchem_geometry_plausibility(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        frame=frame_arg,
    )
    result["next_actions"] = _build_next_actions(
        "geometry_plausibility", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("check_nwchem_freq_plausibility")
def _handle_check_nwchem_freq_plausibility(arguments: dict[str, Any]) -> dict[str, Any]:
    result = check_nwchem_freq_plausibility(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expect_minimum=arguments.get("expect_minimum", True),
    )
    result["next_actions"] = _build_next_actions(
        "freq_plausibility", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("prepare_nwchem_freq_restart")
def _handle_prepare_nwchem_freq_restart(arguments: dict[str, Any]) -> dict[str, Any]:
    return prepare_freq_restart(
        input_file=arguments["input_file"],
        output_file=arguments["output_file"],
        profile=arguments.get("profile"),
    )


# ---------------------------------------------------------------------------
# Handlers — imaginary modes
# ---------------------------------------------------------------------------


@_tool("analyze_nwchem_imaginary_modes")
def _handle_analyze_nwchem_imaginary_modes(arguments: dict[str, Any]) -> dict[str, Any]:
    result = analyze_imaginary_modes(
        arguments["output_file"],
        significant_threshold_cm1=arguments.get("significant_threshold_cm1", 20.0),
        top_atoms=arguments.get("top_atoms", 4),
        detail=arguments.get("detail", "compact"),
    )
    result["next_actions"] = _build_next_actions(
        "imaginary_modes", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("displace_nwchem_geometry_along_mode")
def _handle_displace_nwchem_geometry_along_mode(arguments: dict[str, Any]) -> dict[str, Any]:
    return displace_geometry_along_mode(
        arguments["output_file"],
        mode_number=arguments.get("mode_number"),
        amplitude_angstrom=arguments.get("amplitude_angstrom", 0.15),
        significant_threshold_cm1=arguments.get("significant_threshold_cm1", 20.0),
    )


@_tool("validate_nwchem_tce_setup")
def _handle_validate_nwchem_tce_setup(arguments: dict[str, Any]) -> dict[str, Any]:
    result = validate_nwchem_tce_setup(
        tce_input_path=arguments["tce_input_file"],
        scf_output_path=arguments.get("scf_output_file"),
    )
    result["next_actions"] = _build_next_actions(
        "tce_validation", result,
        output_file=arguments.get("scf_output_file", ""),
        input_file=arguments["tce_input_file"],
    )
    return result


@_tool("compute_nwchem_harmonic_frequencies")
def _handle_compute_nwchem_harmonic_frequencies(arguments: dict[str, Any]) -> dict[str, Any]:
    from chemtools.programs.nwchem.binary.hessian import compute_nwchem_harmonic_frequencies
    return compute_nwchem_harmonic_frequencies(
        hessian_path=arguments["hessian_file"],
        elements=arguments["elements"],
        masses_amu=arguments.get("masses_amu"),
    )


@_tool("swap_nwchem_movecs")
def _handle_swap_nwchem_movecs(arguments: dict[str, Any]) -> dict[str, Any]:
    return swap_nwchem_movecs(
        movecs_path=arguments["movecs_file"],
        i=arguments["i"],
        j=arguments["j"],
        output_path=arguments.get("output_file"),
    )


@_tool("suggest_nwchem_tce_freeze")
def _handle_suggest_nwchem_tce_freeze(arguments: dict[str, Any]) -> dict[str, Any]:
    return suggest_tce_freeze_count(
        elements=arguments["elements"],
        ecp_core_electrons=arguments.get("ecp_core_electrons"),
    )


# ---------------------------------------------------------------------------
# Handlers — parallel job monitoring, session log, input versioning
# ---------------------------------------------------------------------------


@_tool("check_nwchem_spin_charge_state")
def _handle_check_spin_charge_state(arguments: dict[str, Any]) -> dict[str, Any]:
    result = check_spin_charge_state(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
    )
    result["next_actions"] = _build_next_actions(
        "spin_charge_state", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result


@_tool("suggest_nwchem_multiplicity_scan")
def _handle_suggest_multiplicity_scan(arguments: dict[str, Any]) -> dict[str, Any]:
    elements = arguments.get("elements")
    charge = arguments.get("charge")
    multiplicity = arguments.get("multiplicity")
    input_file = arguments.get("input_file")
    if input_file and (elements is None or charge is None or multiplicity is None):
        summary = inspect_input(input_file)
        if elements is None:
            # Full atom multiset, not unique elements — the electron-count parity
            # that fixes the scan's multiplicities depends on every atom.
            elements = summary.get("all_elements") or summary.get("elements")
        if charge is None:
            charge = summary.get("charge")
        if multiplicity is None:
            multiplicity = summary.get("multiplicity")
    if not elements:
        return {
            "error": "Provide input_file (to read elements/charge/multiplicity) or an explicit elements list.",
        }
    result = recommend_multiplicity_scan(
        elements=elements,
        charge=charge or 0,
        current_multiplicity=multiplicity,
        metal_oxidation_states=arguments.get("metal_oxidation_states"),
    )
    if result["scan_warranted"] and input_file:
        result["next_actions"] = [{
            "priority": 1,
            "tool": "generate_nwchem_input_batch",
            "params": {
                "template_input": input_file,
                "vary": {"mult": result["recommended_multiplicities"]},
                "output_dir": arguments.get("output_dir") or str(Path(input_file).parent),
            },
            "reason": "Generate one input per candidate multiplicity at the same geometry "
                      "and basis; run them, then take the lowest total energy.",
            "confidence": 0.9,
        }]
    return result


@_tool("review_nwchem_progress")
def _handle_review_nwchem_progress(arguments: dict[str, Any]) -> dict[str, Any]:
    result = review_nwchem_progress(
        output_path=arguments["output_file"],
        input_path=arguments.get("input_file"),
        error_path=arguments.get("error_file"),
        process_id=arguments.get("process_id"),
        profile=arguments.get("profile"),
        job_id=arguments.get("job_id"),
    )
    result["next_actions"] = _build_next_actions(
        "review_progress", result,
        output_file=arguments["output_file"],
        input_file=arguments.get("input_file", ""),
    )
    return result

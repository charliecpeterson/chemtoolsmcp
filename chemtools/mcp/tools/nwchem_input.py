"""NWChem MCP handlers — input.

Split from mcp/tools/nwchem.py by category. Shared imports/helpers live in
_nwchem_base (pulled in below); nwchem.py imports this module so its @_tool
handlers register.
"""
from __future__ import annotations

from chemtools.mcp.tools._nwchem_base import *  # noqa: F401,F403
from chemtools.mcp.tools._nwchem_base import _tool, _build_next_actions  # noqa: F401


@_tool("prepare_nwchem_mcscf_setup")
def _handle_prepare_nwchem_mcscf_setup(arguments: dict[str, Any]) -> dict[str, Any]:
    from chemtools.programs.nwchem.strategy.active_space import prepare_nwchem_mcscf_setup
    return prepare_nwchem_mcscf_setup(
        scf_output_path=arguments["scf_output_path"],
        input_path=arguments.get("input_path"),
        expected_metal_elements=arguments.get("expected_metal_elements"),
        expected_somo_count=arguments.get("expected_somo_count"),
        prefer_expanded=arguments.get("prefer_expanded", False),
    )


@_tool("prepare_nwchem_tce_setup")
def _handle_prepare_nwchem_tce_setup(arguments: dict[str, Any]) -> dict[str, Any]:
    """Thick orchestrator — parses MOs, computes freeze count, checks ordering,
    suggests swaps, and returns a Diagnosis with next_actions."""
    from chemtools.programs.nwchem.strategy.active_space import prepare_nwchem_tce_setup
    return prepare_nwchem_tce_setup(
        scf_output_path=arguments["scf_output_path"],
        target_method=arguments.get("target_method", "ccsd(t)"),
        elements=arguments.get("elements"),
        charge=arguments.get("charge", 0),
        multiplicity=arguments.get("multiplicity", 1),
        expected_metal_elements=arguments.get("expected_metal_elements"),
        expected_somo_count=arguments.get("expected_somo_count"),
        ecp_core_electrons=arguments.get("ecp_core_electrons"),
    )

# ---------------------------------------------------------------------------
# Generic auto-detect tool dispatchers (Phase 4)
# ---------------------------------------------------------------------------


@_tool("draft_nwchem_atom_input")
def _handle_draft_nwchem_atom_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_atom_input(
        element=arguments["element"],
        basis=arguments["basis"],
        method=arguments.get("method", "scf"),
        charge=arguments.get("charge", 0),
        multiplicity=arguments.get("multiplicity"),
        xc_functional=arguments.get("xc_functional", "m06"),
        memory=arguments.get("memory"),
        start_name=arguments.get("start_name"),
        output_dir=arguments.get("output_dir"),
        write_file=arguments.get("write_file", False),
        basis_library=basis_library_path(arguments.get("basis_library")),
    )


@_tool("draft_nwchem_vectors_swap_input")
def _handle_draft_nwchem_vectors_swap_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_vectors_swap_input(
        output_path=arguments["output_file"],
        input_path=arguments["input_file"],
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
        vectors_input=arguments.get("vectors_input"),
        vectors_output=arguments.get("vectors_output"),
        task_operation=arguments.get("task_operation", "energy"),
        iterations=arguments.get("iterations", 500),
        smear=arguments.get("smear", 0.001),
        convergence_damp=arguments.get("convergence_damp", 30),
        convergence_ncydp=arguments.get("convergence_ncydp", 30),
        population_print=arguments.get("population_print", "mulliken"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        title=arguments.get("title"),
        write_file=arguments.get("write_file", False),
    )


@_tool("draft_nwchem_property_check_input")
def _handle_draft_nwchem_property_check_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_property_check_input(
        input_path=arguments["input_file"],
        reference_output_path=arguments.get("reference_output_file"),
        vectors_input=arguments.get("vectors_input"),
        vectors_output=arguments.get("vectors_output"),
        property_keywords=arguments.get("property_keywords"),
        task_strategy=arguments.get("task_strategy", "auto"),
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
        iterations=arguments.get("iterations", 1),
        convergence_energy=arguments.get("convergence_energy", "1e-3"),
        smear=arguments.get("smear"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        title=arguments.get("title"),
        write_file=arguments.get("write_file", False),
    )


@_tool("draft_nwchem_scf_stabilization_input")
def _handle_draft_nwchem_scf_stabilization_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_scf_stabilization_input(
        input_path=arguments["input_file"],
        reference_output_path=arguments.get("reference_output_file"),
        vectors_input=arguments.get("vectors_input"),
        vectors_output=arguments.get("vectors_output"),
        task_operation=arguments.get("task_operation", "energy"),
        iterations=arguments.get("iterations"),
        smear=arguments.get("smear"),
        convergence_damp=arguments.get("convergence_damp"),
        convergence_ncydp=arguments.get("convergence_ncydp"),
        population_print=arguments.get("population_print"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        title=arguments.get("title"),
        write_file=arguments.get("write_file", False),
    )


@_tool("draft_nwchem_optimization_followup_input")
def _handle_draft_nwchem_optimization_followup_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_optimization_followup_input(
        output_path=arguments["output_file"],
        input_path=arguments["input_file"],
        task_strategy=arguments.get("task_strategy", "auto"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        title=arguments.get("title"),
        write_file=arguments.get("write_file", False),
    )


@_tool("draft_nwchem_cube_input")
def _handle_draft_nwchem_cube_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_cube_input(
        input_path=arguments["input_file"],
        vectors_input=arguments["vectors_input"],
        orbital_vectors=arguments.get("orbital_vectors"),
        density_modes=arguments.get("density_modes"),
        orbital_spin=arguments.get("orbital_spin", "total"),
        extent_angstrom=arguments.get("extent_angstrom", 6.0),
        grid_points=arguments.get("grid_points", 120),
        pyscf_compatible_grid_points=arguments.get("pyscf_compatible_grid_points"),
        gaussian=arguments.get("gaussian", True),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        title=arguments.get("title"),
        write_file=arguments.get("write_file", False),
    )


@_tool("draft_nwchem_frontier_cube_input")
def _handle_draft_nwchem_frontier_cube_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_frontier_cube_input(
        output_path=arguments["output_file"],
        input_path=arguments["input_file"],
        vectors_input=arguments.get("vectors_input"),
        include_somos=arguments.get("include_somos", True),
        include_homo=arguments.get("include_homo", True),
        include_lumo=arguments.get("include_lumo", True),
        include_density_modes=arguments.get("include_density_modes"),
        extent_angstrom=arguments.get("extent_angstrom", 6.0),
        grid_points=arguments.get("grid_points", 120),
        pyscf_compatible_grid_points=arguments.get("pyscf_compatible_grid_points"),
        gaussian=arguments.get("gaussian", True),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        title=arguments.get("title"),
        write_file=arguments.get("write_file", False),
    )


# ---------------------------------------------------------------------------
# Handlers — output parsers
# ---------------------------------------------------------------------------


@_tool("inspect_nwchem_input")
def _handle_inspect_nwchem_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return inspect_input(arguments["input_file"])


@_tool("inspect_nwchem_runner_profiles", needs="runner_profile")
def _handle_inspect_nwchem_runner_profiles(arguments: dict[str, Any]) -> dict[str, Any]:
    return inspect_runner_profiles(arguments.get("profiles_path"))


@_tool("lint_nwchem_input")
def _handle_lint_nwchem_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return lint_nwchem_input(
        input_path=arguments["input_file"],
        library_path=basis_library_path(arguments.get("library_path")),
    )


@_tool("render_nwchem_basis_block")
def _handle_render_nwchem_basis_block(arguments: dict[str, Any]) -> dict[str, Any]:
    basis_name = arguments["basis_name"]
    library_path = basis_library_path(arguments.get("library_path"))
    if arguments.get("check_only", False):
        elements = arguments.get("elements") or []
        return resolve_basis(basis_name, elements, library_path)
    if arguments.get("input_file"):
        return render_basis_block_from_geometry(
            basis_name,
            arguments["input_file"],
            library_path,
            block_name=arguments.get("block_name", "ao basis"),
            mode=arguments.get("mode"),
        )
    return render_basis_block(
        basis_name,
        arguments.get("elements", []),
        library_path,
        block_name=arguments.get("block_name", "ao basis"),
        mode=arguments.get("mode"),
    )


@_tool("render_nwchem_ecp_block")
def _handle_render_nwchem_ecp_block(arguments: dict[str, Any]) -> dict[str, Any]:
    ecp_name = arguments["ecp_name"]
    elements = arguments["elements"]
    library_path = basis_library_path(arguments.get("library_path"))
    if arguments.get("check_only", False):
        return resolve_ecp(ecp_name, elements, library_path)
    return render_ecp_block(ecp_name, elements, library_path)


@_tool("render_nwchem_basis_setup")
def _handle_render_nwchem_basis_setup(arguments: dict[str, Any]) -> dict[str, Any]:
    return render_nwchem_basis_setup(
        geometry_path=arguments["geometry_file"],
        library_path=basis_library_path(arguments.get("library_path")),
        basis_assignments=arguments["basis_assignments"],
        ecp_assignments=arguments.get("ecp_assignments"),
        default_basis=arguments.get("default_basis"),
        default_ecp=arguments.get("default_ecp"),
        basis_block_name=arguments.get("block_name", "ao basis"),
        basis_mode=arguments.get("basis_mode"),
    )


# ---------------------------------------------------------------------------
# Handlers — input creation
# ---------------------------------------------------------------------------


@_tool("create_nwchem_input")
def _handle_create_nwchem_input(arguments: dict[str, Any]) -> dict[str, Any]:
    # Translate explicit SCF params into module_settings lines
    module = arguments.get("module", "").strip().lower()
    module_settings: list[str] = []
    if module == "scf":
        scf_type = arguments.get("scf_type")
        nopen = arguments.get("nopen")
        maxiter = arguments.get("maxiter")
        thresh = arguments.get("thresh")
        if scf_type:
            module_settings.append(scf_type)
        if nopen is not None:
            module_settings.append(f"nopen {nopen}")
        if maxiter is not None:
            module_settings.append(f"maxiter {maxiter}")
        if thresh is not None:
            module_settings.append(f"thresh {thresh:.2e}")
    return create_nwchem_input(
        geometry_path=arguments["geometry_file"],
        library_path=basis_library_path(arguments.get("library_path")),
        basis_assignments=arguments["basis_assignments"],
        ecp_assignments=arguments.get("ecp_assignments"),
        default_basis=arguments.get("default_basis"),
        default_ecp=arguments.get("default_ecp"),
        basis_block_name=arguments.get("block_name", "ao basis"),
        basis_mode=arguments.get("basis_mode"),
        module=arguments["module"],
        task_operation=arguments.get("task_operation"),
        charge=arguments.get("charge"),
        multiplicity=arguments.get("multiplicity"),
        module_settings=module_settings or None,
        extra_blocks=arguments.get("extra_blocks"),
        memory=arguments.get("memory"),
        title=arguments.get("title"),
        start_name=arguments.get("start_name"),
        vectors_input=arguments.get("vectors_input"),
        vectors_output=arguments.get("vectors_output"),
        output_dir=arguments.get("output_dir"),
        write_file=arguments.get("write_file", False),
    )


@_tool("create_nwchem_dft_workflow_input")
def _handle_create_nwchem_dft_workflow_input(arguments: dict[str, Any]) -> dict[str, Any]:
    result = create_nwchem_dft_workflow_input(
        geometry_path=arguments["geometry_file"],
        library_path=basis_library_path(arguments.get("library_path")) if arguments.get("library_path") else basis_library_path(),
        basis_assignments=arguments["basis_assignments"],
        ecp_assignments=arguments.get("ecp_assignments"),
        default_basis=arguments.get("default_basis"),
        default_ecp=arguments.get("default_ecp"),
        basis_block_name=arguments.get("block_name", "ao basis"),
        basis_mode=arguments.get("basis_mode"),
        xc_functional=arguments["xc_functional"],
        task_operations=arguments["task_operations"],
        charge=arguments.get("charge"),
        multiplicity=arguments.get("multiplicity"),
        dft_settings=arguments.get("dft_settings"),
        extra_blocks=arguments.get("extra_blocks"),
        geometry_options=arguments.get("geometry_options"),
        memory=arguments.get("memory"),
        title=arguments.get("title"),
        start_name=arguments.get("start_name"),
        vectors_input=arguments.get("vectors_input"),
        vectors_output=arguments.get("vectors_output"),
        output_dir=arguments.get("output_dir"),
        write_file=arguments.get("write_file", False),
    )
    # Strip large basis/ECP text from response when file was written — saves tokens
    if arguments.get("write_file") and result.get("written_file"):
        result.pop("input_text", None)
        bs = result.get("basis_setup")
        if isinstance(bs, dict):
            bs = dict(bs)
            if isinstance(bs.get("basis_block"), dict):
                bb = dict(bs["basis_block"])
                bb.pop("text", None)
                bs["basis_block"] = bb
            if isinstance(bs.get("ecp_block"), dict):
                eb = dict(bs["ecp_block"])
                eb.pop("text", None)
                bs["ecp_block"] = eb
            result["basis_setup"] = bs
    return result


# ---------------------------------------------------------------------------
# Handlers — case analysis and recovery
# ---------------------------------------------------------------------------


@_tool("draft_nwchem_mcscf_input")
def _handle_draft_nwchem_mcscf_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_mcscf_input(
        reference_output_path=arguments["reference_output_file"],
        input_path=arguments["input_file"],
        expected_metal_elements=arguments.get("expected_metals"),
        expected_somo_count=arguments.get("expected_somos"),
        active_space_mode=arguments.get("active_space_mode", "minimal"),
        vectors_input=arguments.get("vectors_input"),
        vectors_output=arguments.get("vectors_output"),
        state_label=arguments.get("state_label"),
        symmetry=arguments.get("symmetry"),
        hessian=arguments.get("hessian", "exact"),
        maxiter=arguments.get("maxiter", 80),
        thresh=arguments.get("thresh", 1.0e-5),
        level=arguments.get("level", 0.6),
        lock_vectors=arguments.get("lock_vectors", True),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        title=arguments.get("title"),
        write_file=arguments.get("write_file", False),
    )


@_tool("draft_nwchem_mcscf_retry_input")
def _handle_draft_nwchem_mcscf_retry_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_mcscf_retry_input(
        output_path=arguments["output_file"],
        input_path=arguments["input_file"],
        expected_metal_elements=arguments.get("expected_metals"),
        active_space_mode=arguments.get("active_space_mode", "auto"),
        vectors_input=arguments.get("vectors_input"),
        vectors_output=arguments.get("vectors_output"),
        state_label=arguments.get("state_label"),
        symmetry=arguments.get("symmetry"),
        hessian=arguments.get("hessian"),
        maxiter=arguments.get("maxiter"),
        thresh=arguments.get("thresh"),
        level=arguments.get("level"),
        lock_vectors=arguments.get("lock_vectors", True),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        title=arguments.get("title"),
        write_file=arguments.get("write_file", False),
    )


# ---------------------------------------------------------------------------
# Handlers — geometry and frequency plausibility
# ---------------------------------------------------------------------------


@_tool("draft_nwchem_imaginary_mode_inputs")
def _handle_draft_nwchem_imaginary_mode_inputs(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_imaginary_mode_inputs(
        output_path=arguments["output_file"],
        input_path=arguments["input_file"],
        mode_number=arguments.get("mode_number"),
        amplitude_angstrom=arguments.get("amplitude_angstrom", 0.15),
        significant_threshold_cm1=arguments.get("significant_threshold_cm1", 20.0),
        task_strategy=arguments.get("task_strategy", "auto"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        write_files=arguments.get("write_files", False),
        add_noautosym=arguments.get("noautosym", True),
        enforce_symmetry_c1=arguments.get("symmetry_c1", True),
    )


# ---------------------------------------------------------------------------
# TCE (Tensor Contraction Engine) handlers
# ---------------------------------------------------------------------------


@_tool("draft_nwchem_tce_input")
def _handle_draft_nwchem_tce_input(arguments: dict[str, Any]) -> dict[str, Any]:
    # swap_pairs comes in as list of [i, j] arrays from JSON
    raw_swaps = arguments.get("swap_pairs")
    swap_pairs = [tuple(pair) for pair in raw_swaps] if raw_swaps else None
    result = draft_nwchem_tce_input(
        scf_output_file=arguments["scf_output_file"],
        input_file=arguments["input_file"],
        method=arguments.get("method", "mp2"),
        freeze_count=arguments.get("freeze_count"),
        swap_pairs=swap_pairs,
        movecs_file=arguments.get("movecs_file"),
        ecp_core_electrons=arguments.get("ecp_core_electrons"),
        basis_library=arguments.get("basis_library"),
        start_name=arguments.get("start_name"),
        memory=arguments.get("memory"),
        output_dir=arguments.get("output_dir"),
        base_name=arguments.get("base_name"),
        write_file=arguments.get("write_file", False),
    )
    return result


@_tool("draft_nwchem_tce_restart_input")
def _handle_draft_nwchem_tce_restart_input(arguments: dict[str, Any]) -> dict[str, Any]:
    return draft_nwchem_tce_restart_input(
        tce_output_file=arguments["tce_output_file"],
        tce_input_file=arguments.get("tce_input_file"),
        max_iterations=arguments.get("max_iterations", 200),
        thresh=arguments.get("thresh", 1e-5),
        copy_amplitudes=arguments.get("copy_amplitudes", True),
        output_dir=arguments.get("output_dir"),
        write_file=arguments.get("write_file", False),
    )


@_tool("create_nwchem_input_variant")
def _handle_create_nwchem_input_variant(arguments: dict[str, Any]) -> dict[str, Any]:
    result = create_nwchem_input_variant(
        source_input=arguments["source_input"],
        changes=arguments["changes"],
        reason=arguments.get("reason", ""),
        output_path=arguments.get("output_path"),
    )
    result.pop("input_text", None)
    return result


# ---------------------------------------------------------------------------
# Handlers — eval + smart input creation (Phase 6)
# ---------------------------------------------------------------------------


@_tool("create_nwchem_dft_input_from_request")
def _handle_create_nwchem_dft_input_from_request(arguments: dict[str, Any]) -> dict[str, Any]:
    result = create_nwchem_dft_input_from_request(
        formula=arguments.get("formula"),
        geometry_path=arguments.get("geometry_file"),
        library_path=basis_library_path(arguments.get("library_path")),
        basis_assignments=arguments.get("basis_assignments"),
        ecp_assignments=arguments.get("ecp_assignments"),
        default_basis=arguments.get("default_basis"),
        default_ecp=arguments.get("default_ecp"),
        xc_functional=arguments.get("xc_functional"),
        task_operations=arguments.get("task_operations"),
        charge=arguments.get("charge"),
        multiplicity=arguments.get("multiplicity"),
        dft_settings=arguments.get("dft_settings"),
        extra_blocks=arguments.get("extra_blocks"),
        geometry_options=arguments.get("geometry_options"),
        memory=arguments.get("memory"),
        title=arguments.get("title"),
        start_name=arguments.get("start_name"),
        output_dir=arguments.get("output_dir"),
        write_file=arguments.get("write_file", False),
    )
    # Don't send full input text through MCP — it can be huge with explicit basis blocks
    if result.get("input_text") and len(result["input_text"]) > 5000:
        result["input_text_truncated"] = result["input_text"][:2000] + "\n... (truncated, see written_file)"
        del result["input_text"]
    return result


# ---------------------------------------------------------------------------
# Handlers — gap-fill tools (Phase 5)
# ---------------------------------------------------------------------------


@_tool("review_nwchem_input_request")
def _handle_review_nwchem_input_request(arguments: dict[str, Any]) -> dict[str, Any]:
    return review_nwchem_input_request(
        formula=arguments.get("formula"),
        geometry_path=arguments.get("geometry_file"),
        library_path=basis_library_path(arguments.get("library_path")),
        basis_assignments=arguments.get("basis_assignments"),
        ecp_assignments=arguments.get("ecp_assignments"),
        default_basis=arguments.get("default_basis"),
        default_ecp=arguments.get("default_ecp"),
        module=arguments.get("module", "dft"),
        task_operations=arguments.get("task_operations"),
        functional=arguments.get("functional"),
        charge=arguments.get("charge"),
        multiplicity=arguments.get("multiplicity"),
    )

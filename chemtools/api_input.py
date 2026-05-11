from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from chemtools.core.common import detect_program, make_metadata, read_text, ELEMENT_TO_Z
from chemtools.programs.nwchem.input.basis_library import (
    extract_basis_blocks,
    extract_nwchem_geometry_elements,
    list_basis_sets,
    render_mixed_nwchem_basis_block,
    render_mixed_nwchem_ecp_block,
    render_nwchem_ecp_block,
    render_nwchem_basis_block,
    render_nwchem_basis_block_from_geometry,
    resolve_ecp_set,
    resolve_mixed_basis_assignments,
    resolve_mixed_ecp_assignments,
    resolve_basis_set,
)
from chemtools.programs.nwchem.strategy.diagnose import (
    analyze_frontier_orbitals as analyze_nwchem_frontier_orbitals,
    diagnose_nwchem_output,
    parse_scf,
    suggest_vectors_swaps as suggest_nwchem_vectors_swaps,
    summarize_nwchem_output,
)
from chemtools.programs.nwchem.parse.input import (
    extract_nwchem_geometry_block,
    extract_nwchem_module_block,
    inspect_all_nwchem_basis_blocks,
    inspect_nwchem_basis_block,
    inspect_nwchem_ecp_block,
    inspect_nwchem_input,
    inspect_nwchem_module_vectors,
    load_geometry_source,
    parse_start_blocks,
    render_nwchem_module_block,
    render_nwchem_geometry_block,
    replace_nwchem_geometry_block,
    replace_nwchem_module_block,
)
# Raw-signature versions of functions that have same-name MCP wrappers
# in this file or in chemtools.programs.nwchem.output. The wrappers below
# take a single `path` argument; the raw versions take `(path, contents, ...)`.
from chemtools.programs.nwchem.parse.mos import parse_mos as _parse_mos_raw
from chemtools.programs.nwchem.parse.freq import (
    parse_trajectory as _parse_trajectory_raw,
    analyze_imaginary_modes as _analyze_imaginary_modes_raw,
    displace_geometry_along_mode as _displace_geometry_along_mode_raw,
)
from chemtools.programs.nwchem.input._utils import (
    _TRANSITION_METALS,
    _COVALENT_RADII,
    _coerce_api_int,
    _coerce_api_float,
    _strategy_entry,
    _summarize_prepared_artifact,
    KEYWORD_LINE_RE,
    CONVERGENCE_DAMP_RE,
    CONVERGENCE_NCYDP_RE,
    ITERATIONS_RE,
    SMEAR_RE,
    PRINT_RE,
    CONVERGENCE_ENERGY_RE,
    VECTORS_RE,
    VECTORS_INPUT_TOKEN_RE,
    VECTORS_OUTPUT_TOKEN_RE,
    _select_primary_task_module,
    _select_scf_stabilization_strategy,
    _select_optimization_follow_up_strategy,
    _build_optimization_follow_up_plan,
    _rewrite_module_body_for_vectors_swap,
    _rewrite_module_body_for_property_check,
    _rewrite_module_body_for_scf_stabilization,
    _extract_vectors_io_from_lines,
    _rewrite_module_body_for_vectors_output,
    _indent_vectors_block_lines,
    _replace_module_block_in_text,
    _ensure_module_vectors_output_in_text,
    _default_optimization_follow_up_base_name,
    _default_optimization_follow_up_title,
    _build_simple_input_file_plan,
    _apply_default_dft_settings,
    _ensure_driver_block,
    _parse_formula_elements,
    _normalize_nwchem_task_operation,
    _replace_or_insert_keyword_line,
    _remove_keyword_blocks,
    _render_named_block,
    _replace_or_insert_named_block,
    _append_named_blocks_before_tasks,
    _render_limitxyz_lines,
    _render_dplot_density_block,
    _render_dplot_orbital_block,
    _build_vectors_swap_file_plan,
    _build_mcscf_reorder_plan,
    _render_mcscf_block,
    _build_cube_file_plan,
    _write_text_file,
    _build_imaginary_follow_up_plan,
    _auto_task_strategy,
    _replace_tasks_in_text,
    _build_imaginary_output_file_plan,
    _write_imaginary_input_files,
)
from chemtools.programs.nwchem.input.basis import render_nwchem_basis_setup
from chemtools.programs.nwchem.input.opt_followup import _select_best_optimization_frame  # used by extract_nwchem_geometry below
from chemtools.programs.nwchem.output import (
    parse_tasks,
    parse_mos,
    parse_trajectory,
    parse_mcscf_output,
    parse_population_analysis,
    summarize_output,
    diagnose_output,
    suggest_vectors_swaps,
    analyze_frontier_orbitals,
    parse_freq,
)
from .api_strategy import (
    check_spin_charge_state,
    suggest_nwchem_mcscf_active_space,
    review_nwchem_mcscf_case,
)


def _normalize_stem_for_match(stem: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def _stem_tokens(stem: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", stem.lower()) if token]




# Imaginary-mode handling drafters moved to programs/nwchem/input/imaginary_modes.py.
from chemtools.programs.nwchem.input.imaginary_modes import (  # noqa: F401, E402
    analyze_imaginary_modes,
    displace_geometry_along_mode,
    draft_nwchem_imaginary_mode_inputs,
)


# Optimization follow-up drafter moved to programs/nwchem/input/opt_followup.py.
from chemtools.programs.nwchem.input.opt_followup import (  # noqa: F401, E402
    draft_nwchem_optimization_followup_input,
)


# DFT workflow drafters moved to programs/nwchem/input/dft.py.
from chemtools.programs.nwchem.input.dft import (  # noqa: F401, E402
    create_nwchem_dft_workflow_input,
    create_nwchem_dft_input_from_request,
)


# Geometry helpers moved to programs/nwchem/input/geometry.py.
from chemtools.programs.nwchem.input.geometry import (  # noqa: F401, E402
    extract_nwchem_geometry,
    draft_initial_geometry,
)


# Re-exports for previously-carved drafter families (lost during the
# geometry carve-out's range delete). Restored for back-compat.
from chemtools.programs.nwchem.input.scf_recovery import (  # noqa: F401, E402
    draft_nwchem_vectors_swap_input,
    draft_nwchem_property_check_input,
    draft_nwchem_scf_stabilization_input,
)
from chemtools.programs.nwchem.input.mcscf import (  # noqa: F401, E402
    draft_nwchem_mcscf_input,
    draft_nwchem_mcscf_retry_input,
)
from chemtools.programs.nwchem.input.cube import (  # noqa: F401, E402
    draft_nwchem_cube_input,
    draft_nwchem_frontier_cube_input,
)


# Lint + restart helpers moved to programs/nwchem/input/lint_restart.py.
from chemtools.programs.nwchem.input.lint_restart import (  # noqa: F401, E402
    inspect_input,
    lint_nwchem_input,
    find_restart_assets,
)
from chemtools.programs.nwchem.input.tce import (  # noqa: F401, E402
    draft_nwchem_tce_input,
    validate_nwchem_tce_setup,
    draft_nwchem_atom_input,
    draft_nwchem_tce_restart_input,
)


# General input drafters moved to programs/nwchem/input/general.py.
from chemtools.programs.nwchem.input.general import (  # noqa: F401, E402
    create_nwchem_input,
    review_nwchem_input_request,
    create_nwchem_input_variant,
)


# Workflow planner moved to programs/nwchem/strategy/workflow_planner.py.
from chemtools.programs.nwchem.strategy.workflow_planner import (  # noqa: F401, E402
    prepare_nwchem_next_step,
    plan_nwchem_workflow,
)

from __future__ import annotations

# api.py — public API surface, re-exports from domain sub-modules
from ._api_utils import _COVALENT_RADII, _TRANSITION_METALS  # noqa: F401
from .api_basis import *  # noqa: F401,F403
from .api_runner import *  # noqa: F401,F403
from .api_runner import render_job_script, watch_multiple_nwchem_runs, init_session_log, append_session_log, next_versioned_path  # noqa: F401
from .api_output import *  # noqa: F401,F403
from .api_output import parse_freq_progress  # noqa: F401
from .api_strategy import *  # noqa: F401,F403
from .api_strategy import check_nwchem_geometry_plausibility, check_nwchem_freq_plausibility, suggest_spin_state, suggest_basis_set, suggest_memory, suggest_resources, suggest_relativistic_correction, prepare_freq_restart, preflight_check, get_nwchem_workflow_state, check_memory_fit, estimate_freq_walltime, suggest_hpc_resources, detect_hpc_accounts  # noqa: F401
from .diagnostics import summarize_electronic_structure, track_spin_state_across_optimization  # noqa: F401
from .registry import (  # noqa: F401
    register_run, update_run_status, list_runs, get_run_summary,
    create_campaign, get_campaign_status, get_campaign_energies,
    create_workflow, advance_workflow, generate_input_batch,
)
from .protocols import plan_calculation, list_protocols  # noqa: F401
from .api_input import *  # noqa: F401,F403
from .api_input import extract_nwchem_geometry, draft_initial_geometry, plan_nwchem_workflow, validate_nwchem_tce_setup, draft_nwchem_tce_restart_input, draft_nwchem_atom_input, create_nwchem_input_variant  # noqa: F401
from .api_output import compute_reaction_energy, parse_nwchem_thermochem  # noqa: F401
from .nwchem_tce import (  # noqa: F401
    parse_nwchem_movecs,
    parse_tce_amplitudes,
    swap_nwchem_movecs,
    suggest_tce_freeze_count,
    analyze_tce_orbital_ordering,
)

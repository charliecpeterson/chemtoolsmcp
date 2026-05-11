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


# Case review family moved to programs/nwchem/strategy/case_review.py.
from chemtools.programs.nwchem.strategy.case_review import (  # noqa: F401, E402
    check_spin_charge_state,
    summarize_nwchem_case,
    review_nwchem_case,
    review_nwchem_mcscf_case,
)


# Pre-job input advisors moved to programs/nwchem/strategy/input_advisors.py.
from chemtools.programs.nwchem.strategy.input_advisors import (  # noqa: F401, E402
    suggest_spin_state,
    suggest_basis_set,
    suggest_relativistic_correction,
)


# Workflow state + preflight + freq restart moved to
# programs/nwchem/strategy/workflow_state.py.
from chemtools.programs.nwchem.strategy.workflow_state import (  # noqa: F401, E402
    prepare_freq_restart,
    preflight_check,
    get_nwchem_workflow_state,
)

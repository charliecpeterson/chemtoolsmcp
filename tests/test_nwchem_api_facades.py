"""Legacy NWChem APIs must remain aliases of their focused owners."""

from __future__ import annotations

import chemtools
from chemtools import api_input, api_strategy
from chemtools.programs.nwchem.input.general import (
    create_nwchem_input,
    review_nwchem_input_request,
)
from chemtools.programs.nwchem.input.geometry import extract_nwchem_geometry
from chemtools.programs.nwchem.input.lint_restart import (
    find_restart_assets,
    lint_nwchem_input,
)
from chemtools.programs.nwchem.input.mcscf import (
    draft_nwchem_mcscf_retry_input,
)
from chemtools.programs.nwchem.strategy.case_review import (
    check_spin_charge_state,
    review_nwchem_mcscf_case,
)
from chemtools.programs.nwchem.strategy.hpc_resources import (
    suggest_hpc_resources,
)
from chemtools.programs.nwchem.strategy.mcscf_active_space import (
    suggest_nwchem_mcscf_active_space,
)
from chemtools.programs.nwchem.strategy.resources import _analyze_job_size
from chemtools.programs.nwchem.strategy.workflow_planner import (
    prepare_nwchem_next_step,
)


def test_api_input_exports_are_direct_owner_aliases():
    assert api_input.create_nwchem_input is create_nwchem_input
    assert api_input.review_nwchem_input_request is review_nwchem_input_request
    assert api_input.extract_nwchem_geometry is extract_nwchem_geometry
    assert api_input.find_restart_assets is find_restart_assets
    assert api_input.lint_nwchem_input is lint_nwchem_input
    assert (
        api_input.draft_nwchem_mcscf_retry_input
        is draft_nwchem_mcscf_retry_input
    )
    assert api_input.prepare_nwchem_next_step is prepare_nwchem_next_step


def test_api_strategy_exports_are_direct_owner_aliases():
    assert api_strategy.check_spin_charge_state is check_spin_charge_state
    assert api_strategy.review_nwchem_mcscf_case is review_nwchem_mcscf_case
    assert api_strategy.suggest_hpc_resources is suggest_hpc_resources
    assert (
        api_strategy.suggest_nwchem_mcscf_active_space
        is suggest_nwchem_mcscf_active_space
    )
    assert api_strategy._analyze_job_size is _analyze_job_size


def test_top_level_exports_remain_direct_owner_aliases():
    assert chemtools.create_nwchem_input is create_nwchem_input
    assert chemtools.extract_nwchem_geometry is extract_nwchem_geometry
    assert chemtools.lint_nwchem_input is lint_nwchem_input
    assert chemtools.prepare_nwchem_next_step is prepare_nwchem_next_step
    assert chemtools.review_nwchem_mcscf_case is review_nwchem_mcscf_case
    assert chemtools.suggest_hpc_resources is suggest_hpc_resources

"""Contracts for GRASP generation lists, ASF prompts, and orbital roles."""

from __future__ import annotations

import pytest

from chemtools.mcp.dispatch import tool_definitions
from chemtools.programs.grasp.input.heredoc import (
    rcsfgenerate_input,
    rmcdhf_input,
    rwfnestimate_input,
)
from chemtools.programs.grasp.parse.sum_file import parse_sum
from chemtools.programs.grasp.strategy.workflows import plan_dhf_workflow
from chemtools.programs.grasp.strategy.runner import validate_csf_block_contract


_TWO_BLOCK_CSF = """Core subshells:

Peel subshells:
  1s   2s   2p-  2p
CSF(s):
  1s ( 2)

         0+
  2s ( 2)

         0+
 *
  2p-( 1)  2p ( 1)
      1/2      3/2
                  1+
"""


def test_rcsfgenerate_encodes_independent_generation_lists():
    stdin = rcsfgenerate_input(
        configurations=["1s(2,*)2p(1,*)"],
        active_orbitals="5s,5p,5d,5f,5g",
        twoj_min=1,
        twoj_max=3,
        excitations=3,
        additional_lists=[{
            "configurations": ["1s(2,*)2p(1,*)"],
            "active_orbitals": "7s,7p,7d,7f,7g,7h,7i",
            "twoj_min": 1,
            "twoj_max": 3,
            "excitations": 2,
        }],
    )

    assert stdin == [
        "*", "0",
        "1s(2,*)2p(1,*)", "", "5s,5p,5d,5f,5g", "1,3", "3", "y",
        "1s(2,*)2p(1,*)", "", "7s,7p,7d,7f,7g,7h,7i", "1,3", "2", "n",
    ]


def test_rcsfgenerate_rejects_incomplete_generate_more_stream():
    with pytest.raises(ValueError, match="additional_lists"):
        rcsfgenerate_input(
            configurations=["1s(1,i)"],
            active_orbitals="1s",
            twoj_min=1,
            twoj_max=1,
            generate_more=True,
        )


def test_rmcdhf_omits_weight_for_one_asf_and_preserves_blank_role():
    assert rmcdhf_input(
        block_level_selections=["1"],
        orbitals_to_optimize="3*",
        spectroscopic_orbitals="",
    ) == ["y", "1", "3*", "", "100"]


def test_rmcdhf_includes_weight_for_multiple_asfs():
    assert rmcdhf_input(
        block_level_selections=["1", "1-2"],
    ) == ["y", "1", "1-2", "5", "*", "*", "100"]


def test_rmcdhf_rejects_invalid_or_empty_asf_selection():
    with pytest.raises(ValueError, match="must not be empty"):
        rmcdhf_input(block_level_selections=[])
    with pytest.raises(ValueError, match="at least one selected ASF"):
        rmcdhf_input(block_level_selections=[""])
    with pytest.raises(ValueError, match="invalid ASF selection range"):
        rmcdhf_input(block_level_selections=["3-1"])


def test_rwfnestimate_documented_file_source_sequence_is_aligned():
    assert rwfnestimate_input(sources=["1", "rwfn.inp", "2"]) == [
        "y", "1", "rwfn.inp", "*", "2", "*",
    ]


def test_correlation_workflow_requires_explicit_orbital_roles():
    with pytest.raises(ValueError, match="correlation workflows require explicit"):
        plan_dhf_workflow(
            z=3,
            a=7,
            configurations=["1s(2,*)2s(1,*)"],
            active_orbitals="3s,3p,3d",
            twoj_min=1,
            twoj_max=1,
            excitations=3,
            block_level_selections=["1"],
            expected_csf_blocks=[{"two_j": 1, "parity": "+", "ncsf": 79}],
            name="li_2s_n3",
        )


def test_correlation_workflow_records_explicit_layer_policy():
    plan = plan_dhf_workflow(
        z=3,
        a=7,
        configurations=["1s(2,*)2s(1,*)"],
        active_orbitals="3s,3p,3d",
        twoj_min=1,
        twoj_max=1,
        excitations=3,
        block_level_selections=["1"],
        expected_csf_blocks=[{"two_j": 1, "parity": "+", "ncsf": 79}],
        orbitals_to_optimize="3*",
        spectroscopic_orbitals="",
        name="li_2s_n3",
    )

    assert plan["orbital_policy"] == {
        "orbitals_to_optimize": "3*",
        "spectroscopic_orbitals": "",
        "correlation_expansion": True,
    }
    rmcdhf = next(step for step in plan["steps"] if step["exe"] == "rmcdhf")
    assert rmcdhf["stdin"] == ["y", "1", "3*", "", "100"]
    generation = next(
        step for step in plan["steps"] if step["exe"] == "rcsfgenerate"
    )
    assert generation["expected_csf_blocks"] == [
        {"two_j": 1, "parity": "+", "ncsf": 79}
    ]


def test_labeled_block_contract_rejects_valid_counts_in_wrong_order(tmp_path):
    csf = tmp_path / "rcsf.inp"
    csf.write_text(_TWO_BLOCK_CSF, encoding="ascii")

    with pytest.raises(ValueError, match="block order does not match"):
        validate_csf_block_contract(
            csf,
            [
                {"two_j": 2, "parity": "+", "ncsf": 1},
                {"two_j": 0, "parity": "+", "ncsf": 2},
            ],
        )


def test_summary_ground_energy_is_minimum_across_j_blocks():
    summary = parse_sum(
        """Eigenenergies:
 Level J Parity Hartrees Kaysers eV
 1 1/2 + -1.000000D+00 -2.0D+05 -2.7D+01
Weights of major contributors to ASF:
Eigenenergies:
 Level J Parity Hartrees Kaysers eV
 1 7/2 - -2.000000D+00 -4.0D+05 -5.4D+01
Weights of major contributors to ASF:
"""
    )

    assert summary["ground_energy_au"] == -2.0


def test_executable_asf_tools_require_labeled_block_contracts():
    definitions = {item["name"]: item for item in tool_definitions()}

    for name in ("run_grasp_rmcdhf", "run_grasp_rci"):
        required = definitions[name]["inputSchema"]["required"]
        assert "expected_csf_blocks" in required
    assert definitions["run_grasp_rsave"]["inputSchema"]["required"] == [
        "working_dir", "name",
    ]

"""Atomic LS and jj census contracts used by preflight MCP tools."""

from __future__ import annotations

import pytest

from chemtools.mcp.decorator import _TOOL_CAPABILITIES, _TOOL_PROGRAMS
from chemtools.mcp.dispatch import dispatch_tool, tool_definitions
from chemtools.reference.atomic_multiplets import (
    AtomicConfigurationError,
    AtomicSubshell,
    analyze_atomic_multiplets,
    analyze_parsed_configuration,
    atomic_configuration,
    extract_j_levels,
    j_shell_m_distribution,
)
from chemtools.reference.fblock import load_fblock_catalog
from chemtools.reference.fblock_configuration import parse_shell_configuration


_ANGULAR_MOMENTUM = {letter: index for index, letter in enumerate("spdfgh")}


def test_p2_terms_levels_and_state_counts_are_exact():
    analysis = analyze_atomic_multiplets("2p2")

    assert analysis["schema_version"] == "chemtools.atomic-multiplets/1"
    assert [
        (term["term"], term["occurrences"])
        for term in analysis["terms"]
    ] == [("3P", 1), ("1D", 1), ("1S", 1)]
    assert analysis["j_parity_blocks"] == [
        {
            "two_j": 0,
            "j": "0",
            "parity": "+",
            "levels": 2,
            "magnetic_sublevels": 2,
        },
        {
            "two_j": 2,
            "j": "1",
            "parity": "+",
            "levels": 1,
            "magnetic_sublevels": 3,
        },
        {
            "two_j": 4,
            "j": "2",
            "parity": "+",
            "levels": 2,
            "magnetic_sublevels": 10,
        },
    ]
    assert analysis["microstate_counts"] == {
        "determinant_weights": 15,
        "binomial_subshell_product": 15,
        "ls_terms": 15,
        "j_levels": 15,
        "consistent": True,
    }
    assert analysis["jj_coupling"]["consistent"] is True


def test_recurring_d3_terms_and_hund_limit_are_explicit():
    analysis = analyze_atomic_multiplets("3d3 4s1")

    terms = {
        term["term"]: term["occurrences"]
        for term in analysis["terms"]
    }
    assert terms["3D"] == 2
    assert terms["1D"] == 2
    assert analysis["hund_ground"] is None
    assert analysis["hund_note"] == (
        "Hund guidance is limited to one open subshell"
    )


def test_j_seven_halves_four_electron_census_matches_reference_table():
    levels = extract_j_levels(j_shell_m_distribution(7, 4))

    assert levels == {0: 1, 4: 2, 8: 2, 10: 1, 12: 1, 16: 1}


def test_every_complete_fblock_reference_matches_independent_multiplet_counts():
    catalog = load_fblock_catalog()
    checked = 0

    for element in catalog.elements:
        for state in element.states:
            if not state.confline:
                continue
            shells = parse_shell_configuration(state.confline)
            configuration = atomic_configuration(
                AtomicSubshell(
                    principal=shell.principal,
                    angular_momentum=_ANGULAR_MOMENTUM[shell.orbital],
                    electrons=shell.electrons,
                )
                for shell in shells
            )
            analysis = analyze_parsed_configuration(configuration)
            predicted = [
                (block["j"], block["parity"], block["levels"])
                for block in analysis["j_parity_blocks"]
            ]
            expected = list(zip(
                state.j_blocks,
                [analysis["parity"]] * len(state.j_blocks),
                state.ncsf,
            ))
            assert predicted == expected, f"{element.symbol}.{state.slug}"
            checked += 1

    assert checked == 616


def test_extended_orbital_letters_keep_spectroscopic_j_omission():
    analysis = analyze_atomic_multiplets("l1")

    assert [(term["term"], term["occurrences"]) for term in analysis["terms"]] == [
        ("2L", 1),
    ]
    assert [block["j"] for block in analysis["j_parity_blocks"]] == [
        "15/2",
        "17/2",
    ]


def test_configuration_parser_rejects_impossible_or_excessive_state_spaces():
    for configuration in ("1p1", "2p7", "p0", "4f7 5f7"):
        with pytest.raises(AtomicConfigurationError):
            analyze_atomic_multiplets(configuration)


def test_multiplet_analysis_is_a_generic_read_only_mcp_tool():
    payload = dispatch_tool("analyze_atomic_multiplets", {"configuration": "4f1"})
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "analyze_atomic_multiplets"
    )

    assert payload["j_parity_blocks"] == [
        {
            "two_j": 5,
            "j": "5/2",
            "parity": "-",
            "levels": 1,
            "magnetic_sublevels": 6,
        },
        {
            "two_j": 7,
            "j": "7/2",
            "parity": "-",
            "levels": 1,
            "magnetic_sublevels": 8,
        },
    ]
    assert definition["inputSchema"]["additionalProperties"] is False
    assert _TOOL_PROGRAMS["analyze_atomic_multiplets"] == "generic"
    assert _TOOL_CAPABILITIES["analyze_atomic_multiplets"] == "none"

"""State-selection safety checks for single-atom NWChem input drafting."""

from __future__ import annotations

import pytest

from chemtools import draft_nwchem_atom_input
from chemtools.mcp.dispatch import dispatch_tool, tool_definitions


def test_charged_atom_requires_an_explicit_multiplicity():
    with pytest.raises(
        ValueError,
        match="electron-count parity does not determine the atomic term",
    ):
        draft_nwchem_atom_input("O", "6-31g", charge=1)


def test_explicit_ionic_multiplicity_is_checked_against_electron_parity():
    with pytest.raises(ValueError, match="wrong parity for 7 electrons"):
        draft_nwchem_atom_input("O", "6-31g", charge=1, multiplicity=3)


def test_atomic_draft_marks_orbital_occupation_as_unconstrained():
    payload = draft_nwchem_atom_input(
        "O",
        "6-31g",
        charge=1,
        multiplicity=4,
    )

    assert payload["multiplicity_source"] == "provided"
    assert payload["nopen"] == 3
    assert payload["occupation_control"] == {
        "status": "unconstrained",
        "catalog_state_supported": False,
        "post_scf_population_check_required": True,
    }
    assert "charge and multiplicity do not uniquely identify" in payload[
        "warnings"
    ][0]


def test_fblock_atomic_draft_warns_that_catalog_occupation_is_not_preserved():
    payload = draft_nwchem_atom_input("La", "def2-svp", multiplicity=2)

    assert len(payload["warnings"]) == 2
    assert "cannot preserve or validate" in payload["warnings"][1]
    assert "symmetry c1" in payload["input_text"]


def test_mcp_schema_says_charged_multiplicity_is_required():
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "draft_nwchem_atom_input"
    )

    assert "required explicitly" in definition["inputSchema"]["properties"][
        "multiplicity"
    ]["description"]
    with pytest.raises(ValueError, match="multiplicity is required"):
        dispatch_tool(
            "draft_nwchem_atom_input",
            {"element": "O", "basis": "6-31g", "charge": 1},
        )

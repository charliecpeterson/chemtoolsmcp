"""Bundled documentation and basis data resolve from the installed package."""

from __future__ import annotations

from chemtools.programs.molcas.docs import list_docs as list_molcas_docs
from chemtools.programs.molcas.input.basis_library import list_basis_sets
from chemtools.programs.nwchem.docs import list_docs as list_nwchem_docs


def test_nwchem_document_inventory_is_complete():
    documents = list_nwchem_docs()

    assert len(documents) == 29
    assert documents[0]["name"] == "01_Intro.pdf.txt"
    assert documents[-1]["name"] == "30-Containers.pdf.txt"


def test_molcas_document_inventory_is_complete():
    documents = list_molcas_docs()

    assert len(documents) == 133
    assert documents[0]["name"] == "advanced_examples/ae.md"
    assert documents[-1]["name"] == "users_guide/ug.md"


def test_molcas_basis_inventory_is_complete():
    basis_sets = list_basis_sets()

    assert len(basis_sets) == 71
    assert basis_sets[:3] == ["3-21G", "4-31G", "5-21G"]
    assert basis_sets[-3:] == ["basis.tbl", "basistype.tbl", "trans.tbl"]

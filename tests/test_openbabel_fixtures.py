"""Contracts for the curated Open Babel conversion fixture corpus."""

from chemtools.integrations.openbabel_contract import (
    CORPUS_SCHEMA,
    load_openbabel_fixture_corpus,
)
from chemtools.science_runner import OPENBABEL_CONVERSION_RESULT_SCHEMA


def test_openbabel_fixture_corpus_covers_conversion_risks():
    corpus = load_openbabel_fixture_corpus()

    assert corpus["schema_version"] == CORPUS_SCHEMA
    assert corpus["recorded_with"] == {
        "openbabel_version": "3.1.0",
        "rdkit_version": "2025.09.5",
        "runner_result_schema": OPENBABEL_CONVERSION_RESULT_SCHEMA,
    }
    assert [case["id"] for case in corpus["cases"]] == [
        "water_smiles_to_molblock",
        "acetate_smiles_to_molblock",
        "benzene_smiles_to_molblock",
        "sodium_chloride_smiles_to_molblock",
        "methyl_radical_smiles_to_molblock",
        "fluoroethanol_chiral_smiles_to_molblock",
        "ethanol_molblock_to_smiles",
    ]


def test_openbabel_fixture_corpus_pins_known_difference_and_warnings():
    cases = {case["id"]: case for case in load_openbabel_fixture_corpus()["cases"]}

    assert cases["fluoroethanol_chiral_smiles_to_molblock"]["expected"] == {
        "status": "completed",
        "comparison_status": "different",
        "coordinate_status": "not_generated",
        "converted_text_contains": ["V2000", "M  END"],
        "source_rdkit": {
            "canonical_smiles": "C[C@H](O)F",
            "formula": "C2H5FO",
            "atom_count": 4,
            "heavy_atom_count": 4,
            "bond_count": 3,
            "formal_charge": 0,
            "radical_electrons": 0,
            "fragment_count": 1,
            "aromatic_atom_count": 0,
            "stereocenter_count": 1,
            "stereo_bond_count": 0,
        },
        "converted_rdkit": {
            "canonical_smiles": "CC(O)F",
            "formula": "C2H5FO",
            "atom_count": 4,
            "heavy_atom_count": 4,
            "bond_count": 3,
            "formal_charge": 0,
            "radical_electrons": 0,
            "fragment_count": 1,
            "aromatic_atom_count": 0,
            "stereocenter_count": 1,
            "stereo_bond_count": 0,
        },
        "difference_fields": ["canonical_smiles"],
        "warning_codes": [],
    }
    assert cases["sodium_chloride_smiles_to_molblock"]["expected"]["warning_codes"] == [
        "multiple_fragments"
    ]
    assert cases["methyl_radical_smiles_to_molblock"]["expected"]["warning_codes"] == [
        "radical_electrons"
    ]

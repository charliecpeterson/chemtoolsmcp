"""Focused contracts for NWChem molecular-orbital classification."""

from chemtools.programs.nwchem.parse.mos import (
    _relabel_unrestricted_somos,
)


def _orbital(spin, vector, energy):
    return {
        "spin": spin,
        "vector_number": vector,
        "occupancy": 1.0,
        "energy_hartree": energy,
        "occupation_label": "singly_occupied",
    }


def test_unrestricted_somos_are_the_majority_spin_frontier_excess():
    orbitals = [
        _orbital("alpha", 1, -2.0),
        _orbital("alpha", 2, -1.0),
        _orbital("alpha", 3, -0.5),
        _orbital("beta", 1, -1.9),
    ]

    _relabel_unrestricted_somos(orbitals)

    assert [
        orbital["vector_number"]
        for orbital in orbitals
        if orbital["occupation_label"] == "singly_occupied"
    ] == [2, 3]


def test_unrestricted_somo_labeling_supports_beta_majority():
    orbitals = [
        _orbital("alpha", 1, -2.0),
        _orbital("beta", 1, -1.9),
        _orbital("beta", 2, -0.7),
    ]

    _relabel_unrestricted_somos(orbitals)

    assert [
        (orbital["spin"], orbital["vector_number"])
        for orbital in orbitals
        if orbital["occupation_label"] == "singly_occupied"
    ] == [("beta", 2)]

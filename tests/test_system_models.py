"""Exact contracts for molecular and periodic scientific systems."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chemtools.core.systems import (
    SCIENTIFIC_SYSTEM_SCHEMA,
    AtomSpec,
    KPointSampling,
    LatticeSpec,
    MolecularSystemSpec,
    PeriodicSpinSpec,
    PeriodicSystemSpec,
    PseudopotentialAssignment,
    molecular_system_from_input_spec,
    molecular_system_to_input_fields,
    scientific_system_from_dict,
)


def _feo_system() -> PeriodicSystemSpec:
    return PeriodicSystemSpec(
        name="FeO vc-relax",
        atoms=(
            AtomSpec(element="Fe", species="Fe", position=(0.0, 0.0, 0.0)),
            AtomSpec(element="Fe", species="Fe", position=(2.2, 2.2, 2.2)),
            AtomSpec(element="O", species="O", position=(2.2, 0.0, 0.0)),
            AtomSpec(element="O", species="O", position=(0.0, 2.2, 2.2)),
        ),
        lattice=LatticeSpec(
            vectors=(
                (4.4, 0.0, 0.0),
                (0.0, 4.4, 0.0),
                (0.0, 0.0, 4.4),
            ),
            units="angstrom",
        ),
        coordinate_mode="cartesian",
        coordinate_units="angstrom",
        k_points=KPointSampling(
            mode="mesh",
            mesh=(6, 6, 6),
            shift=(0.0, 0.0, 0.0),
        ),
        pseudopotentials=(
            PseudopotentialAssignment(
                species="Fe",
                element="Fe",
                format="upf",
                artifact_id="artifact-fe-upf",
                path_hint=Path("Fe.pbe-spn-rrkjus_psl.1.0.0.UPF"),
                family="PSLibrary 1.0.0 PBE",
            ),
            PseudopotentialAssignment(
                species="O",
                element="O",
                format="upf",
                artifact_id="artifact-o-upf",
                path_hint=Path("O.pbe-n-rrkjus_psl.1.0.0.UPF"),
                family="PSLibrary 1.0.0 PBE",
            ),
        ),
        spin=PeriodicSpinSpec(
            mode="collinear",
            starting_magnetization_by_species={"Fe": 0.8, "O": 0.0},
        ),
        net_charge=0.0,
        metadata={"source": "input_examples/qe/FeO/feo.vc-relax.in"},
    )


def test_qe_feo_system_round_trip_is_exact():
    system = _feo_system()

    payload = system.to_dict()
    restored = scientific_system_from_dict(payload)

    assert restored == system
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload
    assert payload["schema"] == SCIENTIFIC_SYSTEM_SCHEMA
    assert payload["system_type"] == "periodic"
    assert payload["lattice"] == {
        "vectors": [
            [4.4, 0.0, 0.0],
            [0.0, 4.4, 0.0],
            [0.0, 0.0, 4.4],
        ],
        "units": "angstrom",
        "periodic": [True, True, True],
    }
    assert payload["k_points"] == {
        "mode": "mesh",
        "mesh": [6, 6, 6],
        "shift": [0.0, 0.0, 0.0],
        "points": [],
        "coordinate_system": "crystal",
    }
    assert payload["spin"] == {
        "mode": "collinear",
        "net_spin": None,
        "starting_magnetization_by_species": {"Fe": 0.8, "O": 0.0},
    }
    assert [
        assignment["artifact_id"]
        for assignment in payload["pseudopotentials"]
    ] == ["artifact-fe-upf", "artifact-o-upf"]


def test_qmcpack_graphene_retains_open_z_boundary_and_xml_pseudo():
    system = PeriodicSystemSpec(
        name="QMCPACK graphene",
        atoms=(
            AtomSpec(element="C", position=(0.0, 0.0, 7.5)),
            AtomSpec(element="C", position=(2.32625287, 1.34306272, 7.5)),
            AtomSpec(element="C", position=(4.65250574, 0.0, 7.5)),
            AtomSpec(element="C", position=(6.97875861, 1.34306272, 7.5)),
            AtomSpec(element="C", position=(-2.3262529, 4.02918816, 7.5)),
            AtomSpec(element="C", position=(-0.00000003, 5.37225088, 7.5)),
            AtomSpec(element="C", position=(2.32625284, 4.02918816, 7.5)),
            AtomSpec(element="C", position=(4.65250571, 5.37225088, 7.5)),
        ),
        lattice=LatticeSpec(
            vectors=(
                (9.30501148, 0.0, 0.0),
                (-4.6525058, 8.05837632, 0.0),
                (0.0, 0.0, 15.0),
            ),
            units="bohr",
            periodic=(True, True, False),
        ),
        coordinate_mode="cartesian",
        coordinate_units="bohr",
        k_points=KPointSampling(
            mode="mesh",
            mesh=(1, 1, 1),
            shift=(0.0, 0.0, 0.0),
        ),
        pseudopotentials=(
            PseudopotentialAssignment(
                species="C",
                element="C",
                format="qmcpack_xml",
                artifact_id="artifact-c-bfd-xml",
                path_hint=Path("pseudopotentials/C.BFD.xml"),
                family="BFD",
            ),
        ),
        spin=PeriodicSpinSpec(mode="unpolarized", net_spin=0.0),
    )

    payload = system.to_dict()

    assert payload["lattice"]["periodic"] == [True, True, False]
    assert payload["coordinate_units"] == "bohr"
    assert payload["pseudopotentials"] == [
        {
            "species": "C",
            "element": "C",
            "format": "qmcpack_xml",
            "artifact_id": "artifact-c-bfd-xml",
            "path_hint": "pseudopotentials/C.BFD.xml",
            "family": "BFD",
        }
    ]
    assert PeriodicSystemSpec.from_dict(payload) == system


def test_fractional_periodic_system_has_no_coordinate_units():
    system = PeriodicSystemSpec(
        name="SrTiO3",
        atoms=(
            AtomSpec(element="Sr", position=(0.0, 0.0, 0.0)),
            AtomSpec(element="Ti", position=(0.5, 0.5, 0.5)),
            AtomSpec(element="O", position=(0.5, 0.5, 0.0)),
            AtomSpec(element="O", position=(0.5, 0.0, 0.5)),
            AtomSpec(element="O", position=(0.0, 0.5, 0.5)),
        ),
        lattice=LatticeSpec(
            vectors=(
                (4.133, 0.0, 0.0),
                (0.0, 4.133, 0.0),
                (0.0, 0.0, 4.133),
            ),
            units="angstrom",
        ),
        coordinate_mode="fractional",
        coordinate_units=None,
        k_points=KPointSampling(
            mode="mesh",
            mesh=(8, 8, 8),
            shift=(0.0, 0.0, 0.0),
        ),
        pseudopotentials=(),
        spin=PeriodicSpinSpec(mode="unpolarized"),
    )

    assert system.to_dict()["coordinate_units"] is None


def test_molecular_input_spec_compatibility_round_trip():
    legacy = {
        "title": "linear UO2 triplet",
        "atoms": [
            {"element": "O", "x": 0.0, "y": 0.0, "z": -1.75},
            {"element": "U", "x": 0.0, "y": 0.0, "z": 0.0},
            {"element": "O", "x": 0.0, "y": 0.0, "z": 1.75},
        ],
        "charge": 0,
        "multiplicity": 3,
        "geometry_units": "angstrom",
        "method": "DFT",
        "basis": "def2-TZVP",
    }

    system = molecular_system_from_input_spec(legacy)

    assert system == MolecularSystemSpec(
        name="linear UO2 triplet",
        atoms=(
            AtomSpec(element="O", position=(0.0, 0.0, -1.75)),
            AtomSpec(element="U", position=(0.0, 0.0, 0.0)),
            AtomSpec(element="O", position=(0.0, 0.0, 1.75)),
        ),
        charge=0,
        multiplicity=3,
        coordinate_units="angstrom",
        metadata={"source": "InputSpec"},
    )
    assert molecular_system_to_input_fields(system) == {
        "atoms": [
            {"element": "O", "x": 0.0, "y": 0.0, "z": -1.75},
            {"element": "U", "x": 0.0, "y": 0.0, "z": 0.0},
            {"element": "O", "x": 0.0, "y": 0.0, "z": 1.75},
        ],
        "charge": 0,
        "multiplicity": 3,
        "geometry_units": "angstrom",
        "title": "linear UO2 triplet",
    }
    assert scientific_system_from_dict(system.to_dict()) == system


def test_periodic_system_requires_complete_pseudopotential_coverage():
    with pytest.raises(
        ValueError,
        match="^missing pseudopotentials for species: \\['O'\\]$",
    ):
        PeriodicSystemSpec(
            atoms=(
                AtomSpec(element="Fe", position=(0.0, 0.0, 0.0)),
                AtomSpec(element="O", position=(0.5, 0.5, 0.5)),
            ),
            lattice=LatticeSpec(
                vectors=(
                    (4.0, 0.0, 0.0),
                    (0.0, 4.0, 0.0),
                    (0.0, 0.0, 4.0),
                ),
                units="angstrom",
            ),
            coordinate_mode="fractional",
            coordinate_units=None,
            k_points=KPointSampling(mode="gamma"),
            pseudopotentials=(
                PseudopotentialAssignment(
                    species="Fe",
                    element="Fe",
                    format="upf",
                    path_hint=Path("Fe.UPF"),
                ),
            ),
            spin=PeriodicSpinSpec(mode="unpolarized"),
        )


def test_periodic_spin_settings_must_reference_present_species():
    system = _feo_system()

    with pytest.raises(
        ValueError,
        match="^spin settings reference unknown species: \\['Fe_down'\\]$",
    ):
        PeriodicSystemSpec(
            atoms=system.atoms,
            lattice=system.lattice,
            coordinate_mode=system.coordinate_mode,
            coordinate_units=system.coordinate_units,
            k_points=system.k_points,
            pseudopotentials=system.pseudopotentials,
            spin=PeriodicSpinSpec(
                mode="collinear",
                starting_magnetization_by_species={"Fe_down": -0.8},
            ),
        )


def test_fractional_coordinates_reject_cartesian_units():
    system = _feo_system()

    with pytest.raises(
        ValueError,
        match="^fractional coordinates cannot declare coordinate units$",
    ):
        PeriodicSystemSpec(
            atoms=system.atoms,
            lattice=system.lattice,
            coordinate_mode="fractional",
            coordinate_units="angstrom",
            k_points=system.k_points,
            pseudopotentials=system.pseudopotentials,
            spin=system.spin,
        )


def test_mesh_sampling_requires_explicit_shift():
    with pytest.raises(
        ValueError,
        match="^mesh sampling requires an explicit shift$",
    ):
        KPointSampling(mode="mesh", mesh=(4, 4, 4))

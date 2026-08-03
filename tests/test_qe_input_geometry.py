"""Structural-review contracts for explicit pw.x input geometries."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from chemtools.application.input_review import review_input
from chemtools.core.units import ANGSTROM_PER_BOHR
from chemtools.programs.qe import QE
from chemtools.programs.qe.input import parse_pw_text
from chemtools.programs.qe.input_geometry import (
    _normalized_cell,
    analyze_pw_input_geometry,
    input_geometry_issues,
    normalize_pw_input_geometry,
)


BAD_MOLECULE_INPUT = """\
&CONTROL
 calculation = 'relax', pseudo_dir = '.'
/
&SYSTEM
 ibrav = 0, nat = 2, ntyp = 2, ecutwfc = 50
/
&ELECTRONS
/
&IONS
 ion_dynamics = 'bfgs'
/
ATOMIC_SPECIES
C 12.011 C.UPF
H 1.008 H.UPF
CELL_PARAMETERS angstrom
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
ATOMIC_POSITIONS angstrom
C 5.0 5.0 5.0
H 5.2 5.0 5.0
K_POINTS gamma
"""


EXTENDED_SOLID_INPUT = """\
&CONTROL
 calculation = 'vc-relax'
/
&SYSTEM
 ibrav = 0, nat = 2, ntyp = 2, ecutwfc = 50
/
&ELECTRONS
/
ATOMIC_SPECIES
Fe 55.845 Fe.UPF
O 15.999 O.UPF
CELL_PARAMETERS angstrom
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
ATOMIC_POSITIONS crystal
Fe 0.0 0.0 0.0
O 0.5 0.5 0.5
K_POINTS gamma
"""


def _write(tmp_path: Path, name: str, contents: str) -> Path:
    path = tmp_path / name
    path.write_text(contents, encoding="utf-8")
    return path


def _write_upf(
    tmp_path: Path,
    element: str,
    z_valence: float,
    *,
    wfc_cutoff: float = 20.0,
) -> None:
    _write(
        tmp_path,
        f"{element}.UPF",
        (
            '<UPF version="2.0.1">\n'
            f'<PP_HEADER element="{element}" pseudo_type="PAW" '
            'relativistic="scalar" functional="PBE" '
            f'z_valence="{z_valence}" wfc_cutoff="{wfc_cutoff}" '
            'rho_cutoff="80"/>\n'
            '</UPF>\n'
        ),
    )


def test_input_geometry_flags_close_contact_with_line_anchored_error():
    parsed = parse_pw_text(BAD_MOLECULE_INPUT)

    analysis = analyze_pw_input_geometry(parsed)

    assert analysis["schema"] == "qe-input-structural-analysis/1"
    assert analysis["coordinate_contract"] == {
        "cell_input_units": "angstrom",
        "position_input_units": "angstrom",
        "normalized_units": "angstrom",
    }
    assert analysis["scope"] == "isolated_molecule"
    assert analysis["initial"]["minimum_pair_distance_angstrom"] == 0.2
    assert analysis["verdict"] == {
        "status": "concerning",
        "origin": "input_geometry",
        "reasons": [
            "The initial geometry has 1 pair(s) closer than 0.60 angstrom."
        ],
        "findings": [{
            "code": "initial_close_contact",
            "origin": "input_geometry",
            "message": (
                "The initial geometry has 1 pair(s) closer than 0.60 angstrom."
            ),
        }],
    }
    assert input_geometry_issues(parsed, analysis) == [{
        "level": "error",
        "message": (
            "Input geometry: The initial geometry has 1 pair(s) closer than "
            "0.60 angstrom."
        ),
        "line": parsed["card_lines"]["atomic_positions"],
        "suggested_fix": (
            "Check the coordinate units and separate overlapping atoms."
        ),
    }]


def test_crystal_positions_in_extended_cell_avoid_molecular_verdict():
    parsed = parse_pw_text(EXTENDED_SOLID_INPUT)

    analysis = analyze_pw_input_geometry(parsed)

    assert analysis["coordinate_contract"]["position_input_units"] == "crystal"
    assert analysis["initial"]["minimum_pair_distance_angstrom"] == (
        3.464101615138
    )
    assert analysis["scope"] == "metrics_only"
    assert analysis["verdict"]["status"] == "not_assessed"
    assert input_geometry_issues(parsed, analysis) == []


def test_bravais_cell_without_a_lattice_parameter_abstains_without_a_lint_issue():
    parsed = parse_pw_text(BAD_MOLECULE_INPUT.replace("ibrav = 0", "ibrav = 2"))

    analysis = analyze_pw_input_geometry(parsed)

    assert analysis == {
        "schema": "qe-input-structural-analysis/1",
        "scope": "not_assessed",
        "verdict": {
            "status": "not_assessed",
            "reasons": [
                "Structural review requires celldm(1) or A for ibrav=2."
            ],
        },
    }
    assert input_geometry_issues(parsed, analysis) == []


def test_fcc_bravais_cell_and_alat_positions_are_normalized():
    parsed = parse_pw_text("""\
&CONTROL
 calculation = 'scf'
/
&SYSTEM
 ibrav = 2, celldm(1) = 10.0, nat = 2, ntyp = 1, ecutwfc = 50
/
&ELECTRONS
/
ATOMIC_SPECIES
Si 28.085 Si.UPF
ATOMIC_POSITIONS alat
Si 0.0 0.0 0.0
Si 0.25 0.25 0.25
K_POINTS gamma
""")

    cell_result = _normalized_cell(parsed)
    assert not isinstance(cell_result, str)
    cell, cell_units, alat_result = cell_result

    alat = 10.0 * ANGSTROM_PER_BOHR
    assert cell_units == "qe_bravais"
    assert alat_result == pytest.approx(alat)
    np.testing.assert_allclose(cell, [
        [-alat / 2.0, 0.0, alat / 2.0],
        [0.0, alat / 2.0, alat / 2.0],
        [-alat / 2.0, alat / 2.0, 0.0],
    ])
    analysis = analyze_pw_input_geometry(parsed)
    assert analysis["coordinate_contract"]["position_input_units"] == "alat"
    assert analysis["initial"]["minimum_pair_distance_angstrom"] == pytest.approx(
        math.sqrt(3.0) * alat / 4.0
    )


def test_normalize_pw_input_geometry_exposes_coordinate_evidence():
    geometry = normalize_pw_input_geometry(parse_pw_text(EXTENDED_SOLID_INPUT))

    assert geometry["status"] == "available"
    assert geometry["atoms"] == [
        {"element": "Fe", "x": 0.0, "y": 0.0, "z": 0.0},
        {"element": "O", "x": 2.0, "y": 2.0, "z": 2.0},
    ]
    np.testing.assert_allclose(geometry["cell_vectors_angstrom"], np.eye(3) * 4.0)


@pytest.mark.parametrize(
    ("lattice_parameter", "expected_alat"),
    (
        ("celldm(1) = 10.0", 10.0 * ANGSTROM_PER_BOHR),
        ("A = 5.0", 5.0),
    ),
)
def test_explicit_alat_cell_is_normalized(
    lattice_parameter: str,
    expected_alat: float,
):
    parsed = parse_pw_text(f"""\\
&CONTROL
 calculation = 'scf'
/
&SYSTEM
 ibrav = 0, {lattice_parameter}, nat = 2, ntyp = 1, ecutwfc = 50
/
&ELECTRONS
/
ATOMIC_SPECIES
Si 28.085 Si.UPF
CELL_PARAMETERS alat
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
ATOMIC_POSITIONS crystal
Si 0.0 0.0 0.0
Si 0.25 0.25 0.25
K_POINTS gamma
""")

    cell_result = _normalized_cell(parsed)

    assert not isinstance(cell_result, str)
    cell, cell_units, alat = cell_result
    assert cell_units == "alat"
    assert alat == pytest.approx(expected_alat)
    np.testing.assert_allclose(cell, np.eye(3) * expected_alat)
    analysis = analyze_pw_input_geometry(parsed)
    assert analysis["coordinate_contract"] == {
        "cell_input_units": "alat",
        "position_input_units": "crystal",
        "normalized_units": "angstrom",
    }
    assert analysis["initial"]["minimum_pair_distance_angstrom"] == pytest.approx(
        math.sqrt(3.0) * expected_alat / 4.0
    )


def test_explicit_alat_cell_without_lattice_parameter_abstains():
    parsed = parse_pw_text(BAD_MOLECULE_INPUT.replace(
        "CELL_PARAMETERS angstrom\n10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0",
        "CELL_PARAMETERS alat\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0",
    ))

    analysis = analyze_pw_input_geometry(parsed)

    assert analysis == {
        "schema": "qe-input-structural-analysis/1",
        "scope": "not_assessed",
        "verdict": {
            "status": "not_assessed",
            "reasons": [
                "CELL_PARAMETERS alat requires celldm(1) or A to normalize "
                "the cell."
            ],
        },
    }


@pytest.mark.parametrize(
    ("lattice_parameter", "expected_units", "expected_scale", "expected_alat"),
    (
        (
            "celldm(1) = 10.0, ",
            "implicit_alat",
            10.0 * ANGSTROM_PER_BOHR,
            10.0 * ANGSTROM_PER_BOHR,
        ),
        ("", "implicit_bohr", ANGSTROM_PER_BOHR, None),
    ),
)
def test_explicit_cell_without_units_uses_qe_deprecated_default(
    lattice_parameter: str,
    expected_units: str,
    expected_scale: float,
    expected_alat: float | None,
):
    parsed = parse_pw_text(f"""\\
&CONTROL
 calculation = 'scf'
/
&SYSTEM
 ibrav = 0, {lattice_parameter}nat = 2, ntyp = 1, ecutwfc = 50
/
&ELECTRONS
/
ATOMIC_SPECIES
Si 28.085 Si.UPF
CELL_PARAMETERS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
ATOMIC_POSITIONS crystal
Si 0.0 0.0 0.0
Si 0.25 0.25 0.25
K_POINTS gamma
""")

    cell_result = _normalized_cell(parsed)

    assert not isinstance(cell_result, str)
    cell, cell_units, alat = cell_result
    assert cell_units == expected_units
    if expected_alat is None:
        assert alat is None
    else:
        assert alat == pytest.approx(expected_alat)
    np.testing.assert_allclose(cell, np.eye(3) * expected_scale)
    analysis = analyze_pw_input_geometry(parsed)
    assert analysis["coordinate_contract"]["cell_input_units"] == expected_units
    assert analysis["initial"]["minimum_pair_distance_angstrom"] == pytest.approx(
        math.sqrt(3.0) * expected_scale / 4.0
    )


def test_explicit_cell_without_units_rejects_ambiguous_lattice_parameter():
    parsed = parse_pw_text("""\\
&CONTROL
 calculation = 'scf'
/
&SYSTEM
 ibrav = 0, celldm(1) = 10.0, A = 5.0, nat = 1, ntyp = 1, ecutwfc = 50
/
&ELECTRONS
/
ATOMIC_SPECIES
Si 28.085 Si.UPF
CELL_PARAMETERS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
ATOMIC_POSITIONS crystal
Si 0.0 0.0 0.0
K_POINTS gamma
""")

    analysis = analyze_pw_input_geometry(parsed)

    assert analysis["verdict"]["reasons"] == [
        "CELL_PARAMETERS without units requires one unambiguous positive "
        "celldm(1) or A."
    ]


def test_unitless_atomic_positions_use_qe_deprecated_alat_default():
    parsed = parse_pw_text("""\\
&CONTROL
 calculation = 'scf'
/
&SYSTEM
 ibrav = 0, celldm(1) = 10.0, nat = 2, ntyp = 1, ecutwfc = 50
/
&ELECTRONS
/
ATOMIC_SPECIES
Si 28.085 Si.UPF
CELL_PARAMETERS alat
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
ATOMIC_POSITIONS
Si 0.0 0.0 0.0
Si 0.25 0.25 0.25
K_POINTS gamma
""")

    analysis = analyze_pw_input_geometry(parsed)

    alat = 10.0 * ANGSTROM_PER_BOHR
    assert analysis["coordinate_contract"] == {
        "cell_input_units": "alat",
        "position_input_units": "implicit_alat",
        "normalized_units": "angstrom",
    }
    assert analysis["initial"]["minimum_pair_distance_angstrom"] == pytest.approx(
        math.sqrt(3.0) * alat / 4.0
    )


def test_unitless_atomic_positions_require_a_lattice_parameter():
    parsed = parse_pw_text(BAD_MOLECULE_INPUT.replace(
        "ATOMIC_POSITIONS angstrom",
        "ATOMIC_POSITIONS",
    ))

    analysis = analyze_pw_input_geometry(parsed)

    assert analysis["verdict"]["reasons"] == [
        "Unitless ATOMIC_POSITIONS uses alat and requires celldm(1) or A."
    ]


def test_crystal_sg_geometry_returns_a_specific_abstention():
    parsed = parse_pw_text("""\\
&CONTROL
 calculation = 'scf'
/
&SYSTEM
 ibrav = 0, nat = 2, ntyp = 1, space_group = 225, ecutwfc = 50
/
&ELECTRONS
/
ATOMIC_SPECIES
Si 28.085 Si.UPF
CELL_PARAMETERS angstrom
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
ATOMIC_POSITIONS crystal_sg
Si 8c 0.25
K_POINTS gamma
""")

    analysis = analyze_pw_input_geometry(parsed)

    assert analysis["verdict"]["reasons"] == [
        "Structural review does not expand ATOMIC_POSITIONS crystal_sg "
        "symmetry records."
    ]


@pytest.mark.parametrize(
    ("ibrav", "parameters", "expected_vectors"),
    (
        (
            1,
            "",
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        (
            3,
            "",
            ((0.5, 0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, -0.5, 0.5)),
        ),
        (
            -3,
            "",
            ((-0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5)),
        ),
        (
            4,
            ", celldm(3) = 1.5",
            (
                (1.0, 0.0, 0.0),
                (-0.5, math.sqrt(3.0) / 2.0, 0.0),
                (0.0, 0.0, 1.5),
            ),
        ),
        (
            8,
            ", celldm(2) = 1.1, celldm(3) = 1.2",
            ((1.0, 0.0, 0.0), (0.0, 1.1, 0.0), (0.0, 0.0, 1.2)),
        ),
        (
            6,
            ", celldm(3) = 1.5",
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.5)),
        ),
        (
            7,
            ", celldm(3) = 1.5",
            ((0.5, -0.5, 0.75), (0.5, 0.5, 0.75), (-0.5, -0.5, 0.75)),
        ),
    ),
)
def test_supported_bravais_cells_use_qe_primitive_vectors(
    ibrav,
    parameters,
    expected_vectors,
):
    parsed = parse_pw_text(f"""\
&CONTROL
 calculation = 'scf'
/
&SYSTEM
 ibrav = {ibrav}, celldm(1) = 10.0{parameters}, nat = 1, ntyp = 1, ecutwfc = 50
/
&ELECTRONS
/
ATOMIC_SPECIES
Si 28.085 Si.UPF
ATOMIC_POSITIONS crystal
Si 0.0 0.0 0.0
K_POINTS gamma
""")

    cell_result = _normalized_cell(parsed)

    assert not isinstance(cell_result, str)
    cell, cell_units, alat_result = cell_result
    alat = 10.0 * ANGSTROM_PER_BOHR
    assert cell_units == "qe_bravais"
    assert alat_result == pytest.approx(alat)
    np.testing.assert_allclose(cell, np.asarray(expected_vectors) * alat)


@pytest.mark.parametrize("ibrav", (5, -5))
def test_rhombohedral_bravais_cells_use_the_documented_axis_choice(ibrav):
    parsed = parse_pw_text(f"""\
&CONTROL
 calculation = 'scf'
/
&SYSTEM
 ibrav = {ibrav}, celldm(1) = 10.0, celldm(4) = 0.2,
 nat = 1, ntyp = 1, ecutwfc = 50
/
&ELECTRONS
/
ATOMIC_SPECIES
Si 28.085 Si.UPF
ATOMIC_POSITIONS crystal
Si 0.0 0.0 0.0
K_POINTS gamma
""")

    cell_result = _normalized_cell(parsed)

    assert not isinstance(cell_result, str)
    cell, _, alat = cell_result
    tx = math.sqrt(0.4)
    ty = math.sqrt(0.8 / 6.0)
    tz = math.sqrt(1.4 / 3.0)
    if ibrav == 5:
        expected = ((tx, -ty, tz), (0.0, 2.0 * ty, tz), (-tx, -ty, tz))
    else:
        u = tz - 2.0 * math.sqrt(2.0) * ty
        v = tz + math.sqrt(2.0) * ty
        scale = 1.0 / math.sqrt(3.0)
        expected = (
            (scale * u, scale * v, scale * v),
            (scale * v, scale * u, scale * v),
            (scale * v, scale * v, scale * u),
        )
    np.testing.assert_allclose(cell, np.asarray(expected) * alat)


@pytest.mark.parametrize(
    ("ibrav", "expected"),
    (
        (9, ((0.5, 0.55, 0.0), (-0.5, 0.55, 0.0), (0.0, 0.0, 1.2))),
        (-9, ((0.5, -0.55, 0.0), (0.5, 0.55, 0.0), (0.0, 0.0, 1.2))),
        (91, ((1.0, 0.0, 0.0), (0.0, 0.55, -0.6), (0.0, 0.55, 0.6))),
        (10, ((0.5, 0.0, 0.6), (0.5, 0.55, 0.0), (0.0, 0.55, 0.6))),
        (11, ((0.5, 0.55, 0.6), (-0.5, 0.55, 0.6), (-0.5, -0.55, 0.6))),
    ),
)
def test_centered_orthorhombic_bravais_cells_use_qe_vectors(ibrav, expected):
    parsed = parse_pw_text(f"""\
&CONTROL
 calculation = 'scf'
/
&SYSTEM
 ibrav = {ibrav}, celldm(1) = 10.0, celldm(2) = 1.1, celldm(3) = 1.2,
 nat = 1, ntyp = 1, ecutwfc = 50
/
&ELECTRONS
/
ATOMIC_SPECIES
Si 28.085 Si.UPF
ATOMIC_POSITIONS crystal
Si 0.0 0.0 0.0
K_POINTS gamma
""")

    cell_result = _normalized_cell(parsed)

    assert not isinstance(cell_result, str)
    cell, _, alat = cell_result
    np.testing.assert_allclose(cell, np.asarray(expected) * alat)


@pytest.mark.parametrize("ibrav", (12, 13, -13, 14))
def test_monoclinic_and_triclinic_bravais_cells_use_qe_angles(ibrav):
    parameters = (
        "celldm(2) = 1.1, celldm(3) = 1.2, celldm(4) = 0.2, "
        "celldm(5) = -0.3, celldm(6) = 0.1"
    )
    parsed = parse_pw_text(f"""\
&CONTROL
 calculation = 'scf'
/
&SYSTEM
 ibrav = {ibrav}, celldm(1) = 10.0, {parameters}, nat = 1, ntyp = 1,
 ecutwfc = 50
/
&ELECTRONS
/
ATOMIC_SPECIES
Si 28.085 Si.UPF
ATOMIC_POSITIONS crystal
Si 0.0 0.0 0.0
K_POINTS gamma
""")

    cell_result = _normalized_cell(parsed)

    assert not isinstance(cell_result, str)
    cell, _, alat = cell_result
    b, c = 1.1, 1.2
    alpha, beta, gamma = 0.2, -0.3, 0.1
    if ibrav == 12:
        expected = ((1.0, 0.0, 0.0), (b * alpha, b * math.sqrt(1 - alpha**2), 0.0), (0.0, 0.0, c))
    elif ibrav == 13:
        expected = ((0.5, 0.0, -c / 2.0), (b * alpha, b * math.sqrt(1 - alpha**2), 0.0), (0.5, 0.0, c / 2.0))
    elif ibrav == -13:
        expected = ((0.5, b / 2.0, 0.0), (-0.5, b / 2.0, 0.0), (c * beta, 0.0, c * math.sqrt(1 - beta**2)))
    else:
        sin_gamma = math.sqrt(1 - gamma**2)
        z = math.sqrt(1 + 2 * alpha * beta * gamma - alpha**2 - beta**2 - gamma**2)
        expected = (
            (1.0, 0.0, 0.0),
            (b * gamma, b * sin_gamma, 0.0),
            (c * beta, c * (alpha - beta * gamma) / sin_gamma, c * z / sin_gamma),
        )
    np.testing.assert_allclose(cell, np.asarray(expected) * alat)


def test_monoclinic_bravais_cell_uses_the_documented_ac_angle():
    parsed = parse_pw_text("""\
&CONTROL
 calculation = 'scf'
/
&SYSTEM
 ibrav = -12, celldm(1) = 10.0, celldm(2) = 1.1, celldm(3) = 1.2,
 celldm(5) = -0.2, nat = 1, ntyp = 1, ecutwfc = 50
/
&ELECTRONS
/
ATOMIC_SPECIES
Zr 91.224 Zr.UPF
ATOMIC_POSITIONS crystal
Zr 0.0 0.0 0.0
K_POINTS gamma
""")

    cell_result = _normalized_cell(parsed)
    assert not isinstance(cell_result, str)
    cell, cell_units, alat_result = cell_result

    alat = 10.0 * ANGSTROM_PER_BOHR
    assert cell_units == "qe_bravais"
    assert alat_result == pytest.approx(alat)
    np.testing.assert_allclose(cell, [
        [alat, 0.0, 0.0],
        [0.0, 1.1 * alat, 0.0],
        [-0.24 * alat, 0.0, 1.2 * math.sqrt(0.96) * alat],
    ])


def test_monoclinic_conventional_parameters_match_celldm_geometry():
    parsed = parse_pw_text("""\
&CONTROL
 calculation = 'scf'
/
&SYSTEM
 ibrav = -12, A = 5.0, B = 5.5, C = 6.0, cosAC = -0.2,
 nat = 1, ntyp = 1, ecutwfc = 50
/
&ELECTRONS
/
ATOMIC_SPECIES
Zr 91.224 Zr.UPF
ATOMIC_POSITIONS crystal
Zr 0.0 0.0 0.0
K_POINTS gamma
""")

    cell_result = _normalized_cell(parsed)

    assert not isinstance(cell_result, str)
    cell, cell_units, alat = cell_result
    assert cell_units == "qe_bravais"
    assert alat == pytest.approx(5.0)
    np.testing.assert_allclose(cell, [
        [5.0, 0.0, 0.0],
        [0.0, 5.5, 0.0],
        [-1.2, 0.0, 6.0 * math.sqrt(0.96)],
    ])


def test_guided_review_rejects_bad_input_geometry_before_execution(tmp_path):
    path = _write(tmp_path, "bad.in", BAD_MOLECULE_INPUT)
    _write_upf(tmp_path, "C", 4.0, wfc_cutoff=60.0)
    _write_upf(tmp_path, "H", 1.0)

    reviewed = review_input(QE, path, resolved_by="content")

    geometry = reviewed["evidence"]["parser"]["result"]["geometry_analysis"]
    assert geometry["verdict"]["status"] == "concerning"
    assert reviewed["evidence"]["lint"]["summary"] == {
        "errors": 1,
        "warnings": 1,
        "info": 0,
    }
    assert reviewed["assessment"] == {
        "verdict": {
            "label": "errors_found",
            "confidence": 0.9,
            "reasons": ["The configured linter found 1 error(s)."],
        }
    }
    assert reviewed["next_actions"] == [
        {
            "action": "edit_input",
            "path": str(path.resolve()),
            "line": 19,
            "suggested_fix": (
                "Check the coordinate units and separate overlapping atoms."
            ),
            "reason": (
                "Input geometry: The initial geometry has 1 pair(s) closer "
                "than 0.60 angstrom."
            ),
            "priority": 1,
        },
        {
            "action": "edit_input",
            "path": str(path.resolve()),
            "line": None,
            "suggested_fix": (
                "Use ecutwfc >= 60 Ry as the starting point for a convergence "
                "study."
            ),
            "reason": (
                "ecutwfc=50 Ry is below the hardest positive UPF suggestion "
                "of 60 Ry."
            ),
            "priority": 2,
        },
    ]

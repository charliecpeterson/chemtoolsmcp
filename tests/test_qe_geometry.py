"""Exact geometry normalization contracts for pw.x output."""

from __future__ import annotations

from pathlib import Path

import pytest

from chemtools.core.units import ANGSTROM_PER_BOHR
from chemtools.programs.qe import QE
from chemtools.programs.qe.geometry import parse_pw_geometry


SCF_GEOMETRY_OUTPUT = """\
     Program PWSCF v.7.5 starts on 31Jul2026 at 10:00:00
     lattice parameter (alat)  =      10.0000  a.u.
     crystal axes: (cart. coord. in units of alat)
               a(1) = (  -0.500000  0.000000  0.500000 )
               a(2) = (   0.000000  0.500000  0.500000 )
               a(3) = (  -0.500000  0.500000  0.000000 )
     number of atoms/cell      =  2
   Cartesian axes
     site n.     atom                  positions (alat units)
         1        Si     tau(   1) = ( 0.0000000 0.0000000 0.0000000 )
         2        Si     tau(   2) = ( 0.2500000 0.2500000 0.2500000 )
     Self-consistent Calculation
!    total energy = -10.00000000 Ry
     convergence has been achieved in 4 iterations
   JOB DONE.
"""


VC_RELAX_CRYSTAL_OUTPUT = """\
     Program PWSCF v.7.5 starts on 31Jul2026 at 10:00:00
     lattice parameter (alat)  =       8.0000  a.u.
     crystal axes: (cart. coord. in units of alat)
               a(1) = ( 1.000000 0.000000 0.000000 )
               a(2) = ( 0.000000 1.000000 0.000000 )
               a(3) = ( 0.000000 0.000000 1.000000 )
     number of atoms/cell = 2
     site n. atom positions (alat units)
       1 Fe tau(1) = ( 0.000000 0.000000 0.000000 )
       2 O  tau(2) = ( 0.500000 0.500000 0.500000 )
     force convergence threshold = 1.0E-3
     BFGS Geometry Optimization
     bfgs converged in 3 scf cycles and 2 bfgs steps
     Final enthalpy = -20.00000000 Ry
Begin final coordinates
CELL_PARAMETERS (bohr)
 8.0 0.0 0.0
 0.0 6.0 0.0
 0.0 0.0 4.0
ATOMIC_POSITIONS (crystal)
Fe 0.0 0.0 0.0
O  0.5 0.5 0.5
End final coordinates
   JOB DONE.
"""


FAILED_RELAX_OUTPUT = """\
     Program PWSCF v.7.5 starts on 31Jul2026 at 10:00:00
     force convergence threshold = 1.0E-3
     BFGS Geometry Optimization
ATOMIC_POSITIONS (angstrom)
C 0.0 0.0 0.0
H 0.0 0.0 1.0
     convergence NOT achieved after 100 iterations: stopping
   JOB DONE.
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_scf_runtime_geometry_is_normalized_from_alat(tmp_path):
    path = _write(tmp_path, "silicon.out", SCF_GEOMETRY_OUTPUT)
    scale = 10.0 * ANGSTROM_PER_BOHR

    geometry = parse_pw_geometry(path)

    assert geometry == {
        "status": "available",
        "role": "calculation_structure",
        "units": "angstrom",
        "atoms": [
            {"element": "Si", "x": 0.0, "y": 0.0, "z": 0.0},
            {
                "element": "Si",
                "x": 0.25 * scale,
                "y": 0.25 * scale,
                "z": 0.25 * scale,
            },
        ],
        "atom_count": 2,
        "elements": {"Si": 2},
        "cell": {
            "vectors_angstrom": [
                [-0.5 * scale, 0.0, 0.5 * scale],
                [0.0, 0.5 * scale, 0.5 * scale],
                [-0.5 * scale, 0.5 * scale, 0.0],
            ],
            "periodic": [True, True, True],
        },
        "source": {"position_line": 9, "cell_line": 4},
    }
    assert QE.parser.get_geometry(str(path)) == geometry["atoms"]


def test_vc_relax_final_crystal_positions_use_final_bohr_cell(tmp_path):
    path = _write(tmp_path, "feo.out", VC_RELAX_CRYSTAL_OUTPUT)
    geometry = parse_pw_geometry(path)

    assert geometry["role"] == "converged_relaxed_structure"
    assert geometry["cell"] == {
        "vectors_angstrom": [
            [8.0 * ANGSTROM_PER_BOHR, 0.0, 0.0],
            [0.0, 6.0 * ANGSTROM_PER_BOHR, 0.0],
            [0.0, 0.0, 4.0 * ANGSTROM_PER_BOHR],
        ],
        "periodic": [True, True, True],
    }
    assert geometry["atoms"] == [
        {"element": "Fe", "x": 0.0, "y": 0.0, "z": 0.0},
        {
            "element": "O",
            "x": 4.0 * ANGSTROM_PER_BOHR,
            "y": 3.0 * ANGSTROM_PER_BOHR,
            "z": 2.0 * ANGSTROM_PER_BOHR,
        },
    ]


def test_parse_output_exposes_small_periodic_geometry_summary(tmp_path):
    scale = 10.0 * ANGSTROM_PER_BOHR
    parsed = QE.parser.parse_output(
        str(_write(tmp_path, "silicon.out", SCF_GEOMETRY_OUTPUT))
    )

    assert parsed["derived"]["qe:geometry"] == {
        "status": "available",
        "role": "calculation_structure",
        "units": "angstrom",
        "atom_count": 2,
        "elements": {"Si": 2},
        "cell": {
            "vectors_angstrom": [
                [-0.5 * scale, 0.0, 0.5 * scale],
                [0.0, 0.5 * scale, 0.5 * scale],
                [-0.5 * scale, 0.5 * scale, 0.0],
            ],
            "periodic": [True, True, True],
        },
        "source": {"position_line": 9, "cell_line": 4},
    }


def test_failed_relaxation_does_not_return_attempted_geometry(tmp_path):
    path = _write(tmp_path, "failed-relax.out", FAILED_RELAX_OUTPUT)

    assert parse_pw_geometry(path) == {
        "status": "unavailable",
        "reason": (
            "PWSCF did not print a converged final-coordinate block for "
            "this relaxation."
        ),
    }
    with pytest.raises(
        ValueError,
        match="did not print a converged final-coordinate block",
    ):
        QE.parser.get_geometry(str(path))


def test_qe_geometry_rejects_nonzero_task_index(tmp_path):
    path = _write(tmp_path, "silicon.out", SCF_GEOMETRY_OUTPUT)

    with pytest.raises(IndexError, match="one summarized task"):
        QE.parser.get_geometry(str(path), task_index=1)

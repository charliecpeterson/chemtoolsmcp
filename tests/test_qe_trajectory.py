"""Geometry-frame contracts for pw.x relaxation output."""

from __future__ import annotations

from pathlib import Path

import pytest

from chemtools.core.units import ANGSTROM_PER_BOHR
from chemtools.programs.qe import QE
from chemtools.programs.qe.trajectory import parse_pw_trajectory


INITIAL_RUNTIME = """\
     Program PWSCF v.7.5 starts on 31Jul2026 at 10:00:00
     lattice parameter (alat)  =      10.0000  a.u.
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
"""

UPDATED_GEOMETRY = """\
CELL_PARAMETERS (bohr)
 8.0 0.0 0.0
 0.0 6.0 0.0
 0.0 0.0 4.0
ATOMIC_POSITIONS (crystal)
Fe 0.0 0.0 0.0
O  0.5 0.5 0.5
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_converged_trajectory_deduplicates_final_coordinate_block(tmp_path):
    output = (
        INITIAL_RUNTIME
        + UPDATED_GEOMETRY
        + "     bfgs converged in 2 scf cycles and 1 bfgs steps\n"
        + "Begin final coordinates\n"
        + UPDATED_GEOMETRY
        + "End final coordinates\n"
        + "   JOB DONE.\n"
    )
    trajectory = parse_pw_trajectory(_write(tmp_path, "feo.out", output))

    assert trajectory["optimization_status"] == "converged"
    assert trajectory["geometry_role"] == "converged_final"
    assert trajectory["geometry_source"] == (
        "step 2 of 2, the converged geometry"
    )
    assert trajectory["frame_count"] == 2
    assert [frame["role"] for frame in trajectory["frames"]] == [
        "initial",
        "converged_final",
    ]
    assert trajectory["frames"][-1]["atoms"] == [
        {"element": "Fe", "x": 0.0, "y": 0.0, "z": 0.0},
        {
            "element": "O",
            "x": 4.0 * ANGSTROM_PER_BOHR,
            "y": 3.0 * ANGSTROM_PER_BOHR,
            "z": 2.0 * ANGSTROM_PER_BOHR,
        },
    ]
    assert trajectory["warnings"] == []


def test_failed_trajectory_preserves_last_attempted_geometry(tmp_path):
    output = (
        INITIAL_RUNTIME
        + UPDATED_GEOMETRY
        + "     convergence NOT achieved after 100 iterations: stopping\n"
        + "   JOB DONE.\n"
    )
    path = _write(tmp_path, "failed.out", output)

    trajectory = QE.parser.get_trajectory(str(path))

    assert trajectory["optimization_status"] == "not_converged"
    assert trajectory["geometry_role"] == "last_attempted"
    assert trajectory["geometry_source"] == (
        "step 2 of 2; the run stopped without converging"
    )
    assert trajectory["frame_count"] == 2
    assert trajectory["frames"][-1]["role"] == "last_attempted"
    assert trajectory["energy_alignment"] == {
        "status": "not_assigned",
        "reason": (
            "PWSCF may print a separate final SCF energy at the relaxed "
            "geometry, so SCF records are not assigned to frames by index."
        ),
    }


def test_trajectory_rejects_single_point_output(tmp_path):
    path = _write(
        tmp_path,
        "scf.out",
        INITIAL_RUNTIME.replace(
            "     force convergence threshold = 1.0E-3\n"
            "     BFGS Geometry Optimization\n",
            "",
        ),
    )

    with pytest.raises(ValueError, match="not a geometry relaxation"):
        QE.parser.get_trajectory(str(path))


def test_trajectory_rejects_nonzero_task_index(tmp_path):
    path = _write(tmp_path, "relax.out", INITIAL_RUNTIME + UPDATED_GEOMETRY)

    with pytest.raises(IndexError, match="one summarized task"):
        QE.parser.get_trajectory(str(path), task_index=1)


def test_truncated_trajectory_is_incomplete(tmp_path):
    path = _write(tmp_path, "truncated.out", INITIAL_RUNTIME + UPDATED_GEOMETRY)

    trajectory = parse_pw_trajectory(path)

    assert trajectory["optimization_status"] == "incomplete"
    assert trajectory["geometry_role"] == "last_attempted"
    assert trajectory["geometry_source"] == (
        "step 2 of 2; the output ended before convergence"
    )

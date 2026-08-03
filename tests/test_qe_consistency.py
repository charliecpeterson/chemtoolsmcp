"""Regression tests for pw.x input-output consistency checks."""

from __future__ import annotations

from pathlib import Path

from chemtools.application.run_inspection import inspect_run
from chemtools.programs.qe import QE


SCF_INPUT = """\
&CONTROL
 calculation = 'scf', pseudo_dir = '.'
/
&SYSTEM
 ibrav = 2, nat = 2, ntyp = 1, ecutwfc = 30.0,
/
&ELECTRONS
 conv_thr = 1.0d-8
/
ATOMIC_SPECIES
Si 28.086 Si.UPF
ATOMIC_POSITIONS crystal
Si 0.00 0.00 0.00
Si 0.25 0.25 0.25
K_POINTS automatic
4 4 4 1 1 1
"""


SCF_OUTPUT = """\
     Program PWSCF v.7.5 starts on 31Jul2026 at 10:00:00
     number of atoms/cell      =  2
     number of atomic types    =  1
     number of electrons       =  8.00
     number of Kohn-Sham states=  4
     kinetic-energy cutoff     =  30.0000  Ry
     charge density cutoff     = 120.0000  Ry
     number of k points=    10
     Self-consistent Calculation
     iteration # 10
     total energy = -93.44697597 Ry
!    total energy = -93.44697597 Ry
     convergence has been achieved in 10 iterations
   JOB DONE.
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _write_upf(tmp_path: Path) -> Path:
    return _write(
        tmp_path,
        "Si.UPF",
        (
            '<UPF version="2.0.1">\n'
            '<PP_HEADER element="Si" pseudo_type="NC" z_valence="4.0" '
            'wfc_cutoff="30.0" rho_cutoff="120.0"/>\n'
            "</UPF>\n"
        ),
    )


def _inspect(tmp_path: Path, input_text: str = SCF_INPUT) -> dict:
    input_path = _write(tmp_path, "silicon.in", input_text)
    output_path = _write(tmp_path, "silicon.out", SCF_OUTPUT)
    _write_upf(tmp_path)
    return inspect_run(
        QE,
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )


def test_qe_consistency_matches_runtime_system_summary(tmp_path):
    inspected = _inspect(tmp_path)

    assert inspected["evidence"]["input_output_consistency"] == {
        "status": "checked",
        "input_path": str((tmp_path / "silicon.in").resolve()),
        "summary": {"match": 6, "mismatch": 0, "not_checked": 1},
        "checks": [
            {
                "field": "calculation_mode",
                "status": "match",
                "input": {
                    "calculation": "scf",
                    "comparison_group": "scf",
                },
                "output": {
                    "calculation_mode": "scf",
                    "basis": "PWSCF calculation markers",
                },
            },
            {
                "field": "atom_count",
                "status": "match",
                "input": 2,
                "output": {"value": 2, "line": 2},
            },
            {
                "field": "atomic_type_count",
                "status": "match",
                "input": 1,
                "output": {"value": 1, "line": 3},
            },
            {
                "field": "electron_count",
                "status": "match",
                "input": {"value": 8.0, "units": "electrons"},
                "output": {
                    "value": 8.0,
                    "line": 4,
                    "units": "electrons",
                },
                "absolute_tolerance": 0.005,
                "basis": (
                    "UPF valence charges and tot_charge compared with "
                    "PWSCF's printed electron count."
                ),
            },
            {
                "field": "ecutwfc",
                "status": "match",
                "input": {"value": 30.0, "units": "Ry"},
                "output": {"value": 30.0, "line": 6, "units": "Ry"},
                "absolute_tolerance": 0.0001,
                "basis": (
                    "The configured wavefunction cutoff compared with the "
                    "PWSCF runtime summary."
                ),
            },
            {
                "field": "ecutrho",
                "status": "match",
                "input": {"value": 120.0, "units": "Ry"},
                "output": {"value": 120.0, "line": 7, "units": "Ry"},
                "absolute_tolerance": 0.0001,
                "basis": (
                    "The explicit density cutoff, or QE's documented 4x "
                    "default, compared with the PWSCF runtime summary."
                ),
            },
            {
                "field": "k_point_count",
                "status": "not_checked",
                "reason": (
                    "The requested input count and PWSCF runtime count are "
                    "not directly comparable after symmetry and "
                    "time-reversal reduction."
                ),
                "input": {
                    "mode": "mesh",
                    "option": "automatic",
                    "mesh": [4, 4, 4],
                    "shift": [1, 1, 1],
                    "requested_full_grid_points": 64,
                },
                "output": {"value": 10, "line": 8},
            },
        ],
    }
    assert inspected["uncertainty"] == []


def test_qe_consistency_reports_cutoff_mismatch(tmp_path):
    changed = SCF_INPUT.replace(
        "ecutwfc = 30.0,",
        "ecutwfc = 40.0, ecutrho = 120.0,",
    )

    inspected = _inspect(tmp_path, changed)
    consistency = inspected["evidence"]["input_output_consistency"]

    assert consistency["status"] == "mismatch"
    assert consistency["summary"] == {
        "match": 5,
        "mismatch": 1,
        "not_checked": 1,
    }
    assert consistency["checks"][4] == {
        "field": "ecutwfc",
        "status": "mismatch",
        "input": {"value": 40.0, "units": "Ry"},
        "output": {"value": 30.0, "line": 6, "units": "Ry"},
        "absolute_tolerance": 0.0001,
        "basis": (
            "The configured wavefunction cutoff compared with the PWSCF "
            "runtime summary."
        ),
    }
    assert inspected["uncertainty"] == [{
        "code": "input_output_mismatch",
        "message": (
            "The explicit input disagrees with output evidence for: ecutwfc."
        ),
        "impact": (
            "Verify that the supplied input and related restart files belong "
            "to this output."
        ),
    }]


def test_qe_consistency_reports_calculation_mode_mismatch(tmp_path):
    inspected = _inspect(
        tmp_path,
        SCF_INPUT.replace("calculation = 'scf'", "calculation = 'bands'"),
    )
    consistency = inspected["evidence"]["input_output_consistency"]

    assert consistency["checks"][0] == {
        "field": "calculation_mode",
        "status": "mismatch",
        "input": {
            "calculation": "bands",
            "comparison_group": "bands_or_nscf",
        },
        "output": {
            "calculation_mode": "scf",
            "basis": "PWSCF calculation markers",
        },
    }
    assert inspected["uncertainty"][0]["message"] == (
        "The explicit input disagrees with output evidence for: "
        "calculation_mode."
    )


def test_qe_consistency_can_compare_gamma_only_count(tmp_path):
    gamma_input = SCF_INPUT.replace(
        "K_POINTS automatic\n4 4 4 1 1 1",
        "K_POINTS gamma",
    )
    gamma_output = SCF_OUTPUT.replace(
        "number of k points=    10",
        "number of k points=     1",
    )
    input_path = _write(tmp_path, "gamma.in", gamma_input)
    output_path = _write(tmp_path, "gamma.out", gamma_output)
    _write_upf(tmp_path)

    inspected = inspect_run(
        QE,
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )
    check = inspected["evidence"]["input_output_consistency"]["checks"][6]

    assert check == {
        "field": "k_point_count",
        "status": "match",
        "input": {
            "mode": "gamma",
            "option": "gamma",
            "mesh": None,
            "shift": None,
            "requested_full_grid_points": 1,
        },
        "output": {"value": 1, "line": 8},
        "basis": "Gamma-only sampling contains one k-point.",
    }


def test_qe_consistency_abstains_without_upf_electron_accounting(tmp_path):
    input_path = _write(tmp_path, "silicon.in", SCF_INPUT)
    output_path = _write(tmp_path, "silicon.out", SCF_OUTPUT)

    inspected = inspect_run(
        QE,
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )
    check = inspected["evidence"]["input_output_consistency"]["checks"][3]

    assert check == {
        "field": "electron_count",
        "status": "not_checked",
        "reason": "Both input and output numeric values are required.",
        "output": {
            "value": 8.0,
            "line": 4,
            "units": "electrons",
        },
        "input_accounting_status": "partial",
    }

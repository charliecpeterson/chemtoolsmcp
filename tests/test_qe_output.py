"""Regression tests for pw.x output parsing and scientific diagnosis."""

from __future__ import annotations

import os
from pathlib import Path

from chemtools.application.run_inspection import inspect_run
from chemtools.core.program import ProgramCapability
from chemtools.programs.qe import QE, _plugin_parser


SCF_OUTPUT = """\
     Program PWSCF v.7.5 starts on 31Jul2026 at 10:00:00
     Reading input from silicon.in
     number of atoms/cell      =  2
     number of atomic types    =  1
     number of electrons       =  8.00
     number of Kohn-Sham states=  4
     kinetic-energy cutoff     =  30.0000  Ry
     charge density cutoff     = 240.0000  Ry
     number of k points=    10
     Self-consistent Calculation
     iteration # 10     ecut=    30.00 Ry
     total energy              =     -93.44697597 Ry
     estimated scf accuracy    <       0.00000010 Ry
!    total energy              =     -93.44697597 Ry
     estimated scf accuracy    <       0.00000001 Ry
     End of self-consistent calculation
     convergence has been achieved in 10 iterations
   JOB DONE.
"""


RELAX_SCF_FAILURE = """\
     Program PWSCF v.7.5 starts on 31Jul2026 at 10:00:00
     Reading input from relax.in
     force convergence threshold = 1.0E-3
     BFGS Geometry Optimization
     Self-consistent Calculation
     iteration # 17
     total energy              =    -114.14300990 Ry
!    total energy              =    -114.14300990 Ry
     convergence has been achieved in 17 iterations
     Self-consistent Calculation
     iteration # 100
     total energy              =    -115.05461747 Ry
     estimated scf accuracy    <       0.00200000 Ry
     End of self-consistent calculation
     convergence NOT achieved after 100 iterations: stopping
   JOB DONE.
"""


VC_RELAX_OUTPUT = """\
     Program PWSCF v.7.5 starts on 31Jul2026 at 10:00:00
     press convergence thresh. = 5.00E-01
     Total force = 0.000000 Total SCF correction = 0.000000
     total stress (Ry/bohr**3) (kbar) P= 0.14
   0.00000219 0.00000000 0.00000000 0.32 0.00 0.00
   0.00000000 0.00000035 0.00000000 0.00 0.05 0.00
   0.00000000 0.00000000 0.00000035 0.00 0.00 0.05
     bfgs converged in 19 scf cycles and 18 bfgs steps
     End of BFGS Geometry Optimization
     Final enthalpy = -560.4576834919 Ry
Begin final coordinates
CELL_PARAMETERS (angstrom)
 4.279240077 0.000000000 0.000000000
 0.000000000 3.057370055 0.000000000
 0.000000000 0.000000000 3.057370055
ATOMIC_POSITIONS (angstrom)
Fe 0.0000000000 0.0000000000 0.0000000000
O  2.1396200387 0.0000000000 0.0000000000
End final coordinates
     Self-consistent Calculation
     iteration # 23
     total energy = -560.44350322 Ry
!    total energy = -560.44350322 Ry
     convergence has been achieved in 23 iterations
   JOB DONE.
"""


READPP_FAILURE = """\
     Program PWSCF v.7.5 starts on 31Jul2026 at 10:00:00
     Reading input from failed.in
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     Error in routine readpp (1):
     file ./Bi.NC.FR.PBEsol.stringent.UPF not found
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     stopping ...
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     Error in routine readpp (1):
     file ./Bi.NC.FR.PBEsol.stringent.UPF not found
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


BANDS_OUTPUT = """\
     Program PWSCF v.7.5 starts on 31Jul2026 at 10:00:00
     Band Structure Calculation
     iteration # 1
     End of band structure calculation
   JOB DONE.
"""


PW2QMCPACK_OUTPUT = """\
     Program pw2qmcpack v.5.0   starts on 6Jun2016 at 19:26:19
     Parallel version (MPI), running on 1 processors
esh5 create pwscf_output/pwscf.pwscf.h5
 inclusive time in compute_qmcpack (s) 10.3299999999999983
"""


PW2QMCPACK_COMPLETED_OUTPUT = f"""\
{PW2QMCPACK_OUTPUT}   JOB DONE.
"""


PW2QMCPACK_WAVEFUNCTION_FAILURE = """\
     Program pw2qmcpack v.7.5 starts on 31Jul2026 at 10:00:00
     read_file_new: Wavefunctions not in collected format?!?
"""


PW2QMCPACK_GAMMA_TRICK_FAILURE = """\
     Program pw2qmcpack v.7.5 starts on 31Jul2026 at 10:00:00
     Using gamma trick results a reduced G space that is not
     supported by QMCPACK.
"""


MOLECULAR_RELAX_FAILURE = """\
     Program PWSCF v.7.5 starts on 31Jul2026 at 10:00:00
     lattice parameter (alat)  =      18.897261  a.u.
     crystal axes: (cart. coord. in units of alat)
               a(1) = ( 1.000000 0.000000 0.000000 )
               a(2) = ( 0.000000 1.000000 0.000000 )
               a(3) = ( 0.000000 0.000000 1.000000 )
     number of atoms/cell = 2
     site n. atom positions (alat units)
       1 C tau(1) = ( 0.500000 0.500000 0.500000 )
       2 H tau(2) = ( 0.520000 0.500000 0.500000 )
     force convergence threshold = 1.0E-3
     BFGS Geometry Optimization
     convergence NOT achieved after 100 iterations: stopping
   JOB DONE.
"""


CELL_CONTRACTION_RELAX = """\
     Program PWSCF v.7.5 starts on 31Jul2026 at 10:00:00
     lattice parameter (alat)  =      18.897261  a.u.
     crystal axes: (cart. coord. in units of alat)
               a(1) = ( 1.000000 0.000000 0.000000 )
               a(2) = ( 0.000000 1.000000 0.000000 )
               a(3) = ( 0.000000 0.000000 1.000000 )
     number of atoms/cell = 1
     site n. atom positions (alat units)
       1 H tau(1) = ( 0.500000 0.500000 0.500000 )
     force convergence threshold = 1.0E-3
     BFGS Geometry Optimization
     bfgs converged in 2 scf cycles and 1 bfgs steps
Begin final coordinates
CELL_PARAMETERS (angstrom)
 7.0 0.0 0.0
 0.0 7.0 0.0
 0.0 0.0 7.0
ATOMIC_POSITIONS (angstrom)
H 3.5 3.5 3.5
End final coordinates
   JOB DONE.
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_qe_backend_declares_output_and_diagnosis_capabilities():
    assert {
        ProgramCapability.OUTPUT_PARSE,
        ProgramCapability.OUTPUT_TASK_INDEX,
        ProgramCapability.DIAGNOSIS_RUN,
    } <= QE.capabilities


def test_parse_converged_scf_keeps_ry_and_hartree_evidence(tmp_path):
    path = _write(tmp_path, "silicon.out", SCF_OUTPUT)

    parsed = QE.parser.parse_output(str(path))

    assert parsed["program_version"] == "7.5"
    assert parsed["tasks"] == [{
        "index": 0,
        "kind": "energy",
        "name": "PWSCF SCF",
        "method": "DFT/PWSCF",
        "basis": "plane waves",
        "energy_hartree": -46.723487985,
        "line_range": (1, 18),
        "outcome": "success",
        "has_usable_data": True,
        "selection_priority": 1,
    }]
    assert parsed["derived"]["qe:final_energy_ry"] == -93.44697597
    assert parsed["derived"]["final_energy_hartree"] == -46.723487985
    assert parsed["derived"]["qe:system"] == {
        "n_atoms": {"value": 2, "line": 3},
        "n_atom_types": {"value": 1, "line": 4},
        "n_electrons": {"value": 8.0, "line": 5},
        "n_kohn_sham_states": {"value": 4, "line": 6},
        "ecutwfc_ry": {"value": 30.0, "line": 7},
        "ecutrho_ry": {"value": 240.0, "line": 8},
        "n_k_points": {"value": 10, "line": 9},
    }
    assert QE.diagnostics.diagnose(parsed)["verdict"] == {
        "label": "scf_converged",
        "confidence": 0.99,
        "reasons": ["pw.x reported SCF convergence and a JOB DONE marker."],
    }


def test_parse_pw2qmcpack_output_reports_hdf5_without_claiming_completion(tmp_path):
    path = _write(tmp_path, "p2q.out", PW2QMCPACK_OUTPUT)

    assert QE.detector.detect(PW2QMCPACK_OUTPUT) is True
    assert QE.detector.detect_version(PW2QMCPACK_OUTPUT) == "5.0"
    parsed = QE.parser.parse_output(str(path))

    assert parsed["program_version"] == "5.0"
    assert parsed["tasks"] == [{
        "index": 0,
        "kind": "unknown",
        "name": "pw2qmcpack Conversion",
        "method": "pw2qmcpack",
        "basis": None,
        "energy_hartree": None,
        "line_range": (1, 4),
        "outcome": "unknown",
        "has_usable_data": True,
        "selection_priority": 1,
    }]
    assert parsed["derived"] == {
        "qe:program": "pw2qmcpack",
        "qe:pw2qmcpack_hdf5_artifacts": [{
            "path": "pwscf_output/pwscf.pwscf.h5",
            "line": 3,
        }],
        "qe:pw2qmcpack_compute_seconds": {
            "seconds": 10.329999999999998,
            "line": 4,
        },
        "qe:job_done": False,
        "qe:job_done_line": None,
    }
    assert QE.diagnostics.diagnose(parsed)["verdict"] == {
        "label": "converter_artifact_reported",
        "confidence": 0.8,
        "reasons": [
            "pw2qmcpack reported creating pwscf_output/pwscf.pwscf.h5."
        ],
    }


def test_inspect_run_accepts_pw2qmcpack_output(tmp_path):
    path = _write(tmp_path, "p2q.out", PW2QMCPACK_OUTPUT)

    inspected = inspect_run(QE, path, resolved_by="content")

    assert inspected["program"] == {
        "name": "qe",
        "version": "5.0",
        "resolved_by": "content",
    }
    assert inspected["assessment"]["verdict"]["label"] == "converter_artifact_reported"
    assert inspected["evidence"]["tasks"][0]["outcome"] == "unknown"


def test_inspect_run_marks_completed_pw2qmcpack_output_successful(tmp_path):
    path = _write(tmp_path, "p2q.out", PW2QMCPACK_COMPLETED_OUTPUT)

    inspected = inspect_run(QE, path, resolved_by="content")

    assert inspected["assessment"]["verdict"] == {
        "label": "converter_completed",
        "confidence": 0.98,
        "reasons": [
            "pw2qmcpack reported creating pwscf_output/pwscf.pwscf.h5, then "
            "printed JOB DONE."
        ],
    }
    assert inspected["evidence"]["tasks"][0]["outcome"] == "success"
    assert inspected["evidence"]["derived"]["qe:job_done_line"] == 5


def test_inspect_run_marks_uncollected_pw2qmcpack_wavefunctions_failed(tmp_path):
    path = _write(tmp_path, "p2q.out", PW2QMCPACK_WAVEFUNCTION_FAILURE)

    inspected = inspect_run(QE, path, resolved_by="content")

    assert inspected["evidence"]["tasks"][0]["outcome"] == "failed"
    assert inspected["assessment"]["verdict"] == {
        "label": "converter_wavefunctions_not_collected",
        "confidence": 0.99,
        "reasons": [
            "pw2qmcpack could not read wavefunctions in collected format."
        ],
    }
    assert inspected["evidence"]["derived"]["qe:pw2qmcpack_errors"] == [{
        "kind": "wavefunctions_not_collected",
        "message": "pw2qmcpack could not read wavefunctions in collected format.",
        "line": 2,
    }]


def test_inspect_run_marks_pw2qmcpack_gamma_trick_failure(tmp_path):
    path = _write(tmp_path, "p2q.out", PW2QMCPACK_GAMMA_TRICK_FAILURE)

    inspected = inspect_run(QE, path, resolved_by="content")

    assert inspected["evidence"]["tasks"][0]["outcome"] == "failed"
    assert inspected["assessment"]["verdict"] == {
        "label": "converter_gamma_trick_unsupported",
        "confidence": 0.99,
        "reasons": [
            "pw2qmcpack rejected QE's gamma-only reduced G-space representation."
        ],
    }


def test_inspect_run_classifies_pw2qmcpack_hdf5_as_binary_metadata(tmp_path):
    output_path = _write(tmp_path, "p2q.out", PW2QMCPACK_OUTPUT)
    artifact_path = tmp_path / "pwscf.pwscf.h5"
    artifact_path.write_bytes(b"HDF5 metadata is outside generic inspection")

    inspected = inspect_run(
        QE,
        output_path,
        resolved_by="content",
        artifact_files=[artifact_path],
    )

    artifact = inspected["evidence"]["artifacts"][1]
    assert artifact["classification"] == {
        "status": "matched",
        "candidates": [{
            "kind": "qe.pw2qmcpack_hdf5",
            "roles": ["checkpoint", "wavefunction"],
            "content_kind": "binary",
                "evidence": "inferred",
                "matched_by": "extension",
                "matched_value": ".pwscf.h5",
            }],
        }
    assert "text_excerpt" not in artifact


def test_inspect_run_compares_pw2qmcpack_hdf5_path_with_input(tmp_path):
    input_path = _write(
        tmp_path,
        "p2q.in",
        "&inputpp\nprefix = 'pwscf'\noutdir = 'pwscf_output'\nwrite_psir = .false.\n/\n",
    )
    output_path = _write(tmp_path, "p2q.out", PW2QMCPACK_OUTPUT)

    inspected = inspect_run(
        QE,
        output_path,
        resolved_by="content",
        artifact_files=[input_path],
    )

    assert inspected["evidence"]["input_output_consistency"] == {
        "status": "checked",
        "input_path": str(input_path.resolve()),
        "summary": {"match": 1, "mismatch": 0, "not_checked": 0},
        "checks": [{
            "field": "pw2qmcpack_hdf5_path",
            "status": "match",
            "input": {
                "prefix": "pwscf",
                "outdir": "pwscf_output",
                "expected_path": "pwscf_output/pwscf.pwscf.h5",
            },
            "output": {
                "reported_paths": ["pwscf_output/pwscf.pwscf.h5"],
                "matching_paths": ["pwscf_output/pwscf.pwscf.h5"],
                "basis": "pw2qmcpack esh5 create output",
            },
        }],
    }


def test_inspect_run_matches_explicit_pw2qmcpack_hdf5_sidecar(tmp_path):
    input_path = _write(
        tmp_path,
        "p2q.in",
        "&inputpp\nprefix = 'pwscf'\noutdir = 'pwscf_output'\nwrite_psir = .false.\n/\n",
    )
    sidecar = tmp_path / "pwscf_output" / "pwscf.pwscf.h5"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"metadata only")
    output_path = _write(tmp_path, "p2q.out", PW2QMCPACK_OUTPUT)

    inspected = inspect_run(
        QE,
        output_path,
        resolved_by="content",
        artifact_files=[input_path, sidecar],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "checked"
    assert consistency["summary"] == {"match": 3, "mismatch": 0, "not_checked": 0}
    assert consistency["checks"][1] == {
        "field": "pw2qmcpack_hdf5_artifact",
        "status": "match",
        "input": {
            "reported_paths": ["pwscf_output/pwscf.pwscf.h5"],
            "resolved_paths": [str(sidecar.resolve())],
            "basis": (
                "Relative pw2qmcpack output paths are resolved against the "
                "converter output directory."
            ),
        },
        "output": {
            "supplied_paths": [str(sidecar.resolve())],
            "matching_paths": [str(sidecar.resolve())],
        },
    }
    assert consistency["checks"][2]["field"] == "pw2qmcpack_hdf5_freshness"
    assert consistency["checks"][2]["status"] == "match"


def test_inspect_run_reports_stale_pw2qmcpack_hdf5_sidecar(tmp_path):
    input_path = _write(
        tmp_path,
        "p2q.in",
        "&inputpp\nprefix = 'pwscf'\noutdir = 'pwscf_output'\nwrite_psir = .false.\n/\n",
    )
    output_path = _write(tmp_path, "p2q.out", PW2QMCPACK_OUTPUT)
    sidecar = tmp_path / "pwscf_output" / "pwscf.pwscf.h5"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"metadata only")
    os.utime(sidecar, ns=(1, 1))

    inspected = inspect_run(
        QE,
        output_path,
        resolved_by="content",
        artifact_files=[input_path, sidecar],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "mismatch"
    assert consistency["summary"] == {"match": 2, "mismatch": 1, "not_checked": 0}
    freshness = consistency["checks"][2]
    assert freshness["field"] == "pw2qmcpack_hdf5_freshness"
    assert freshness["output"]["stale_paths"] == [str(sidecar.resolve())]


def test_inspect_run_reports_nonmatching_pw2qmcpack_hdf5_sidecar(tmp_path):
    input_path = _write(
        tmp_path,
        "p2q.in",
        "&inputpp\nprefix = 'pwscf'\noutdir = 'pwscf_output'\nwrite_psir = .false.\n/\n",
    )
    output_path = _write(tmp_path, "p2q.out", PW2QMCPACK_OUTPUT)
    sidecar = _write(tmp_path, "other.pwscf.h5", "metadata only")

    inspected = inspect_run(
        QE,
        output_path,
        resolved_by="content",
        artifact_files=[input_path, sidecar],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "mismatch"
    assert consistency["summary"] == {"match": 1, "mismatch": 1, "not_checked": 0}
    assert consistency["checks"][1]["output"]["matching_paths"] == []


def test_inspect_run_reports_pw2qmcpack_hdf5_path_mismatch(tmp_path):
    input_path = _write(
        tmp_path,
        "p2q.in",
        "&inputpp\nprefix = 'other'\noutdir = 'pwscf_output'\nwrite_psir = .false.\n/\n",
    )
    output_path = _write(tmp_path, "p2q.out", PW2QMCPACK_OUTPUT)

    inspected = inspect_run(
        QE,
        output_path,
        resolved_by="content",
        artifact_files=[input_path],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "mismatch"
    assert consistency["summary"] == {"match": 0, "mismatch": 1, "not_checked": 0}
    assert consistency["checks"][0]["output"]["matching_paths"] == []


def test_relaxation_failure_does_not_promote_unconverged_energy(tmp_path):
    _write(tmp_path, "relax.in", "&CONTROL\n calculation='relax'\n/\n")
    path = _write(tmp_path, "relax.out", RELAX_SCF_FAILURE)

    parsed = QE.parser.parse_output(str(path))
    diagnosis = QE.diagnostics.diagnose(parsed)

    assert parsed["tasks"][0]["outcome"] == "failed"
    assert parsed["tasks"][0]["energy_hartree"] == -57.07150495
    assert parsed["derived"]["qe:final_energy_ry"] == -114.1430099
    assert parsed["derived"]["qe:last_iterative_energy_ry"] == -115.05461747
    assert parsed["derived"]["scf_converged"] is False
    assert diagnosis["verdict"] == {
        "label": "relaxation_interrupted_by_scf_nonconvergence",
        "confidence": 0.99,
        "reasons": [
            "The final electronic cycle stopped after 100 iterations without convergence."
        ],
    }
    assert diagnosis["next_actions"] == [{
        "tool": "review_input",
        "params": {
            "input_file": str((tmp_path / "relax.in").resolve()),
            "program": "qe",
        },
        "reason": (
            "Review the input associated with this failure. The final "
            "electronic cycle stopped after 100 iterations without convergence."
        ),
        "confidence": 0.95,
        "priority": 1,
    }]


def test_vc_relax_preserves_enthalpy_stress_and_final_scf_energy(tmp_path):
    parsed = QE.parser.parse_output(
        str(_write(tmp_path, "feo.out", VC_RELAX_OUTPUT))
    )

    assert parsed["tasks"][0]["name"] == "PWSCF Variable-Cell Relaxation"
    assert parsed["tasks"][0]["outcome"] == "success"
    assert parsed["derived"]["qe:final_enthalpy_ry"] == -560.4576834919
    assert parsed["derived"]["qe:final_enthalpy_hartree"] == -280.22884174595
    assert parsed["derived"]["qe:final_energy_ry"] == -560.44350322
    assert parsed["derived"]["final_energy_hartree"] == -280.22175161
    assert parsed["derived"]["qe:bfgs"] == {
        "converged": True,
        "scf_cycles": 19,
        "steps": 18,
        "line": 8,
    }
    assert parsed["derived"]["qe:last_stress"] == {
        "pressure_kbar": 0.14,
        "matrix_ry_per_bohr3": [
            [0.00000219, 0.0, 0.0],
            [0.0, 0.00000035, 0.0],
            [0.0, 0.0, 0.00000035],
        ],
        "matrix_kbar": [
            [0.32, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, 0.0, 0.05],
        ],
        "line": 4,
    }
    final_coordinates = parsed["derived"]["qe:final_coordinates_native"]
    assert final_coordinates["cell_parameters"] == {
        "units": "angstrom",
        "vectors": [
            [4.279240077, 0.0, 0.0],
            [0.0, 3.057370055, 0.0],
            [0.0, 0.0, 3.057370055],
        ],
        "line": 12,
    }
    assert final_coordinates["atomic_positions"]["atoms"] == [
        {"label": "Fe", "coordinates": [0.0, 0.0, 0.0]},
        {"label": "O", "coordinates": [2.1396200387, 0.0, 0.0]},
    ]


def test_repeated_mpi_readpp_errors_collapse_to_one_diagnostic(tmp_path):
    input_path = _write(tmp_path, "failed.in", "&CONTROL\n/\n")
    parsed = QE.parser.parse_output(
        str(_write(tmp_path, "failed.out", READPP_FAILURE))
    )
    diagnosis = QE.diagnostics.diagnose(parsed)

    assert parsed["derived"]["qe:runtime_errors"] == [{
        "routine": "readpp",
        "code": "1",
        "message": "file ./Bi.NC.FR.PBEsol.stringent.UPF not found",
        "first_line": 4,
        "last_line": 9,
        "message_line": 5,
        "occurrences": 2,
    }]
    assert parsed["diagnostics"] == [{
        "kind": "error",
        "message": (
            "pw.x readpp error: file ./Bi.NC.FR.PBEsol.stringent.UPF not "
            "found (2 occurrence(s))"
        ),
        "line": 4,
        "file": str((tmp_path / "failed.out").resolve()),
    }]
    assert diagnosis["verdict"] == {
        "label": "pseudopotential_not_found",
        "confidence": 0.98,
        "reasons": [
            "pw.x reported readpp error: file "
            "./Bi.NC.FR.PBEsol.stringent.UPF not found",
            "The same error was repeated by 2 MPI ranks.",
        ],
    }
    assert diagnosis["next_actions"][0]["params"] == {
        "input_file": str(input_path.resolve()),
        "program": "qe",
    }


def test_bands_output_is_completed_but_mode_remains_ambiguous(tmp_path):
    parsed = QE.parser.parse_output(
        str(_write(tmp_path, "bands.out", BANDS_OUTPUT))
    )
    diagnosis = QE.diagnostics.diagnose(parsed)

    assert parsed["tasks"][0]["kind"] == "property"
    assert parsed["tasks"][0]["outcome"] == "success"
    assert parsed["derived"]["qe:calculation_mode"] == "bands_or_nscf"
    assert diagnosis["verdict"] == {
        "label": "bands_or_nscf_completed",
        "confidence": 0.75,
        "reasons": [
            "The band-structure calculation ended cleanly, but the output "
            "alone does not distinguish bands from NSCF."
        ],
    }


def test_clean_relax_without_supported_marker_is_not_called_failed(tmp_path):
    output = """\
     Program PWSCF v.7.5 starts on 31Jul2026 at 10:00:00
     force convergence threshold = 1.0E-3
   JOB DONE.
"""
    parsed = QE.parser.parse_output(
        str(_write(tmp_path, "other-relax.out", output))
    )

    assert parsed["tasks"][0]["outcome"] == "unknown"
    assert QE.diagnostics.diagnose(parsed)["verdict"] == {
        "label": "relaxation_completion_unresolved",
        "confidence": 0.7,
        "reasons": [
            "pw.x ended cleanly, but the output has no supported relaxation "
            "convergence marker."
        ],
    }


def test_guided_inspection_uses_qe_scientific_diagnosis(tmp_path):
    path = _write(tmp_path, "silicon.out", SCF_OUTPUT)

    inspected = inspect_run(QE, path, resolved_by="content")

    assert inspected["program"] == {
        "name": "qe",
        "version": "7.5",
        "resolved_by": "content",
    }
    assert inspected["assessment"] == {
        "source": "backend_diagnosis",
        "verdict": {
            "label": "scf_converged",
            "confidence": 0.99,
            "reasons": [
                "pw.x reported SCF convergence and a JOB DONE marker."
            ],
        },
    }
    assert inspected["uncertainty"] == []


def test_guided_inspection_surfaces_input_geometry_concern(tmp_path):
    path = _write(tmp_path, "molecular-relax.out", MOLECULAR_RELAX_FAILURE)
    input_path = _write(
        tmp_path,
        "molecular-relax.in",
        "&CONTROL\n calculation='relax'\n/\n",
    )

    inspected = inspect_run(QE, path, resolved_by="content")

    trajectory = inspected["evidence"]["derived"]["qe:trajectory"]
    assert trajectory["status"] == "available"
    assert trajectory["frame_count"] == 1
    assert trajectory["geometry_role"] == "last_attempted"
    assert trajectory["structural_analysis"]["verdict"] == {
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
    assert inspected["assessment"] == {
        "source": "backend_diagnosis",
        "verdict": {
            "label": "relaxation_interrupted_by_scf_nonconvergence",
            "confidence": 0.99,
            "reasons": [
                "The final electronic cycle stopped after 100 iterations "
                "without convergence.",
                "Trajectory structural concern (input geometry): The initial "
                "geometry has 1 pair(s) closer than 0.60 angstrom.",
            ],
        },
    }
    assert inspected["next_actions"] == [
        {
            "action": "parse_trajectory",
            "tool": "parse_trajectory",
            "params": {"output_file": str(path.resolve()), "program": "qe"},
            "reason": (
                "Inspect the full geometry history and structural metrics "
                "before accepting or restarting this calculation."
            ),
            "confidence": 0.95,
            "priority": 1,
        },
        {
            "action": "review_input",
            "tool": "review_input",
            "params": {
                "input_file": str(input_path.resolve()),
                "program": "qe",
            },
            "reason": (
                "Review the input associated with this failure. The final "
                "electronic cycle stopped after 100 iterations without "
                "convergence."
            ),
            "confidence": 0.95,
            "priority": 1,
        },
    ]


def test_guided_inspection_surfaces_cell_change_as_observation(tmp_path):
    path = _write(tmp_path, "contracted.out", CELL_CONTRACTION_RELAX)

    inspected = inspect_run(QE, path, resolved_by="content")

    analysis = inspected["evidence"]["derived"]["qe:trajectory"][
        "structural_analysis"
    ]
    assert analysis["verdict"]["status"] == "no_obvious_issue"
    assert analysis["evolution"]["cell_volume_change_percent"] == (
        -65.699998662375
    )
    assert inspected["assessment"] == {
        "source": "backend_diagnosis",
        "verdict": {
            "label": "relaxation_converged",
            "confidence": 0.98,
            "reasons": [
                "BFGS convergence was reported after 1 steps, and pw.x ended "
                "cleanly.",
                "Trajectory observation: The cell volume changes by -65.7 "
                "percent.",
            ],
        },
    }
    assert inspected["next_actions"] == [{
        "action": "parse_trajectory",
        "tool": "parse_trajectory",
        "params": {"output_file": str(path.resolve()), "program": "qe"},
        "reason": (
            "Inspect the full geometry history and structural metrics before "
            "accepting or restarting this calculation."
        ),
        "confidence": 0.9,
        "priority": 2,
    }]


def test_automatic_trajectory_analysis_respects_output_size_limit(
    tmp_path,
    monkeypatch,
):
    path = _write(tmp_path, "molecular-relax.out", MOLECULAR_RELAX_FAILURE)
    monkeypatch.setattr(
        _plugin_parser,
        "_AUTOMATIC_TRAJECTORY_ANALYSIS_LIMIT_BYTES",
        2,
    )

    parsed = QE.parser.parse_output(str(path))

    assert parsed["derived"]["qe:trajectory"] == {
        "status": "not_assessed",
        "reason": (
            f"The {path.stat().st_size}-byte output exceeds the automatic "
            "trajectory-analysis limit of 2 bytes."
        ),
    }

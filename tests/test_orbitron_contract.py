"""Unit tests for Orbitron differential outcomes and QE cross-field checks."""

import json

from chemtools.integrations.orbitron_contract import (
    MANIFEST_SCHEMA,
    _compare_molcas_vibrations,
    _compare_qe_geometry,
    _compare_qe_relax,
    _parse_qe_failed_geometry_reference,
    _parse_qe_geometry_reference,
    _parse_qe_relax_reference,
    report_exit_code,
    run_contract,
)


QE_RELAX_OUTPUT = """\
     number of atoms/cell      =            2
     Self-consistent Calculation
!    total energy              =     -10.00000000 Ry
CELL_PARAMETERS (angstrom)
  2.0 0.0 0.0
  0.0 2.0 0.0
  0.0 0.0 2.0
     Self-consistent Calculation
!    total energy              =     -11.00000000 Ry
CELL_PARAMETERS (angstrom)
  1.9 0.0 0.0
  0.0 1.9 0.0
  0.0 0.0 1.9
"""

QE_GEOMETRY_OUTPUT = """\
     number of atoms/cell      =            2
     lattice parameter (alat)  =      10.0000  a.u.
     crystal axes: (cart. coord. in units of alat)
               a(1) = ( -0.500000  0.000000  0.500000 )
               a(2) = (  0.000000  0.500000  0.500000 )
               a(3) = ( -0.500000  0.500000  0.000000 )
     site n.     atom                  positions (alat units)
         1           Si  tau(   1) = (   0.0000000   0.0000000   0.0000000  )
         2           Si  tau(   2) = (   0.2500000   0.2500000   0.2500000  )
"""

QE_FAILED_GEOMETRY_OUTPUT = """\
     number of atoms/cell      =            2
     lattice parameter (alat)  =      10.0000  a.u.
     crystal axes: (cart. coord. in units of alat)
               a(1) = ( 1.000000 0.000000 0.000000 )
               a(2) = ( 0.000000 1.000000 0.000000 )
               a(3) = ( 0.000000 0.000000 1.000000 )
     site n. atom positions (alat units)
         1 Si tau(1) = ( 0.0000000 0.0000000 0.0000000 )
         2 Si tau(2) = ( 0.2500000 0.2500000 0.2500000 )
     force convergence threshold = 1.0E-3
     Self-consistent Calculation
!    total energy = -10.0 Ry
     convergence has been achieved in 4 iterations
     Self-consistent Calculation
     convergence NOT achieved after 100 iterations: stopping
"""


def _qe_relax_payload(extra_energies):
    tasks = []
    for index, (energy_ry, extra_energy_ry) in enumerate(
        zip([-10.0, -11.0], extra_energies),
        start=1,
    ):
        tasks.append(
            {
                "program": "qe",
                "kind": "scf",
                "energy_hartree": energy_ry / 2.0,
                "line_start": 2 + (index - 1) * 6,
                "extra": {
                    "scf_total_energy_ry": extra_energy_ry,
                    "relax_profile": {
                        "initial_energy_ry": -10.0,
                        "final_energy_ry": -11.0,
                        "step_count": 2,
                    },
                },
            }
        )
    return {
        "program": "qe",
        "detected": "trajectory",
        "frames": [{"atoms": 2}, {"atoms": 2}],
        "program_tasks": tasks,
    }


def test_qe_relax_contract_accepts_per_step_energy_fields():
    reference = _parse_qe_relax_reference(QE_RELAX_OUTPUT)
    checks = _compare_qe_relax(reference, _qe_relax_payload([-10.0, -11.0]))

    assert all(check["agrees"] for check in checks)


def test_qe_relax_contract_detects_stale_extra_energy():
    reference = _parse_qe_relax_reference(QE_RELAX_OUTPUT)
    checks = _compare_qe_relax(reference, _qe_relax_payload([-11.0, -11.0]))
    failed_fields = [check["field"] for check in checks if not check["agrees"]]

    assert failed_fields == ["task.extra.scf_total_energy_ry_sequence"]


def test_qe_geometry_contract_compares_coordinates_and_cell():
    reference = _parse_qe_geometry_reference(QE_GEOMETRY_OUTPUT)
    payload = {
        "atoms": 2,
        "elements": {"Si": 2},
        "distance_unit": "angstrom",
        "geometry_role": "single_point",
        "geometry_source": "the only geometry the run reports",
        "bounding_box": reference["bounding_box"],
        "unit_cell": {
            "a": reference["cell"][0],
            "b": reference["cell"][1],
            "c": [0.0, 0.0, 0.0],
            "periodic": [True, True, True],
        },
    }

    checks = _compare_qe_geometry(reference, payload)
    failed_fields = [check["field"] for check in checks if not check["agrees"]]

    assert failed_fields == ["unit_cell.c"]


def test_qe_failed_geometry_contract_pins_last_attempted_provenance():
    reference = _parse_qe_failed_geometry_reference(QE_FAILED_GEOMETRY_OUTPUT)

    assert reference == {
        "atoms": 2,
        "elements": {"Si": 2},
        "geometry_role": "last_attempted",
        "geometry_source": (
            "step 2 of 2; the run stopped without converging"
        ),
    }


def test_molcas_vibration_contract_detects_duplicated_modes():
    reference = {
        "mode_count": 3,
        "imaginary_count": 1,
        "lowest_frequency": -100.0,
        "highest_frequency": 300.0,
        "mean_frequency": 400.0 / 3,
        "frequency_sample": [-100.0, 200.0, 300.0],
    }
    payload = {
        "mode_count": 6,
        "imaginary_count": 2,
        "lowest_frequency": -100.0,
        "highest_frequency": 300.0,
        "mean_frequency": 400.0 / 3,
        "modes": [
            {"frequency": -100.0},
            {"frequency": -100.0},
            {"frequency": 200.0},
            {"frequency": 200.0},
            {"frequency": 300.0},
            {"frequency": 300.0},
        ],
    }

    checks = _compare_molcas_vibrations(reference, payload)
    failed_fields = [check["field"] for check in checks if not check["agrees"]]

    assert failed_fields == [
        "mode_count",
        "imaginary_count",
        "modes.frequency_sample",
    ]


def test_hash_change_is_no_reference_not_disagreement(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "case.out").write_text("changed\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "cases": [
                    {
                        "id": "changed",
                        "operation": "info",
                        "source": "case.out",
                        "size_bytes": 8,
                        "sha256": "0" * 64,
                        "contract": "qe_scf",
                    }
                ],
            }
        )
    )

    report = run_contract(manifest, corpus, executable=tmp_path / "missing-orbitron")

    assert report["checked_count"] == 0
    assert report["no_reference_count"] == 1
    assert report["records"][0]["outcome"] == "no_reference"
    assert report_exit_code(report) == 3


def test_size_change_is_no_reference_before_hashing_or_orbitron(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "case.out").write_text("changed\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "cases": [
                    {
                        "id": "resized",
                        "operation": "info",
                        "source": "case.out",
                        "size_bytes": 7,
                        "sha256": "0" * 64,
                        "contract": "qe_scf",
                    }
                ],
            }
        )
    )

    report = run_contract(
        manifest,
        corpus,
        executable=tmp_path / "missing-orbitron",
    )

    assert report["records"][0] == {
        "case_id": "resized",
        "contract": "qe_scf",
        "operation": "info",
        "reference_status": None,
        "source": str(corpus / "case.out"),
        "size_bytes": 7,
        "sha256": "0" * 64,
        "outcome": "no_reference",
        "reason": "reference size changed; review the case before comparison",
        "expected_size_bytes": 7,
        "actual_size_bytes": 8,
    }


def test_manifest_path_cannot_escape_corpus_root(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    outside = tmp_path / "outside.out"
    outside.write_text("not part of the corpus\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "cases": [
                    {
                        "id": "outside",
                        "operation": "info",
                        "source": "../outside.out",
                        "size_bytes": 16,
                        "sha256": "0" * 64,
                        "contract": "qe_scf",
                    }
                ],
            }
        )
    )

    report = run_contract(manifest, corpus, executable=tmp_path / "missing-orbitron")

    assert report["records"][0]["outcome"] == "no_reference"
    assert report["records"][0]["reason"] == (
        "reference path escapes the configured corpus root"
    )


def test_exit_code_prioritizes_tool_refusal():
    report = {
        "tool_refused_count": 1,
        "disagree_count": 1,
        "no_reference_count": 1,
    }

    assert report_exit_code(report) == 2

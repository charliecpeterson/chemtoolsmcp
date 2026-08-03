"""Contracts for caller-declared PySCF/reference calculation comparison."""

from __future__ import annotations

import pytest

from chemtools.core.pyscf_comparison import (
    PYSCF_REFERENCE_COMPARISON_SCHEMA,
    compare_pyscf_reference_calculation,
)
from chemtools.mcp.dispatch import dispatch_tool, tool_definitions


def _pyscf_result(**overrides):
    result = {
        "schema_version": "chemtools.pyscf-single-point-result/1",
        "status": "completed",
        "calculation": {
            "method": "rhf",
            "basis": "sto-3g",
            "xc": None,
            "density_fit": False,
            "charge": 0,
            "multiplicity": 1,
            "atom_count": 2,
        },
        "geometry": [
            {"element": "H", "x": 0.0, "y": 0.0, "z": 0.0},
            {"element": "H", "x": 0.0, "y": 0.0, "z": 0.74},
        ],
        "provenance": {
            "pyscf_version": "2.13.1",
            "python_version": "3.12.13",
        },
        "scf": {"converged": True, "cycles": 8},
        "energy": {"total_hartree": -1.1},
        "electrons": {"total": 2, "alpha": 1, "beta": 1},
    }
    result.update(overrides)
    return result


def _reference(**overrides):
    reference = {
        "label": "NWChem RHF/STO-3G",
        "geometry": [
            {"element": "H", "x": 0.0, "y": 0.0, "z": 0.0},
            {"element": "H", "x": 0.0, "y": 0.0, "z": 0.74},
        ],
        "calculation": {
            "method": "rhf",
            "basis": "sto-3g",
            "xc": None,
            "density_fit": False,
            "charge": 0,
            "multiplicity": 1,
        },
        "scf": {"converged": True},
        "energy": {"total_hartree": -1.0},
        "electrons": {"total": 2},
    }
    reference.update(overrides)
    return reference


def _write_cube(tmp_path, name, values, *, title, comment):
    path = tmp_path / name
    path.write_text(
        "\n".join((
            title,
            comment,
            "1 0.0 0.0 0.0",
            "2 1.0 0.0 0.0",
            "2 0.0 1.0 0.0",
            "2 0.0 0.0 1.0",
            "1 1.0 0.0 0.0 0.0",
            " ".join(str(value) for value in values),
            "",
        )),
        encoding="utf-8",
    )
    return path


def test_pyscf_reference_report_preserves_matched_and_different_evidence():
    reference = _reference()
    reference["calculation"] = {
        **reference["calculation"],
        "method": "rks",
        "xc": "pbe",
    }
    report = compare_pyscf_reference_calculation(_pyscf_result(), reference)

    assert report["schema_version"] == PYSCF_REFERENCE_COMPARISON_SCHEMA
    assert report["status"] == "compared"
    assert report["conclusion"] == "evidence_only_no_correctness_verdict"
    assert report["matching"]["geometry"]["status"] == "matched"
    assert report["matching"]["calculation"]["method"] == {
        "status": "different",
        "pyscf": "rhf",
        "reference": "rks",
    }
    assert report["matching"]["calculation"]["basis"]["status"] == "matched"
    assert report["mismatched_settings"] == ["method", "xc"]
    assert report["energy"] == {
        "reference_total_hartree": -1.0,
        "pyscf_total_hartree": -1.1,
        "pyscf_minus_reference_hartree": pytest.approx(-0.1),
        "pyscf_minus_reference_kcal_per_mol": pytest.approx(-62.75094740631),
    }
    assert report["field_comparisons"] == {
        "density": {
            "status": "not_compared",
            "reason": "both_pyscf_and_reference_density_cubes_are_required",
        },
        "orbital": {
            "status": "not_compared",
            "reason": "both_pyscf_and_reference_orbital_cubes_are_required",
        },
    }


def test_pyscf_reference_report_composes_density_and_orbital_evidence(tmp_path):
    reference_density = _write_cube(
        tmp_path,
        "reference_density.cube",
        [1.0] * 8,
        title="Electron density",
        comment="Total density on a common grid",
    )
    pyscf_density = _write_cube(
        tmp_path,
        "pyscf_density.cube",
        [1.0] * 8,
        title="Electron density",
        comment="Total density on a common grid",
    )
    reference_orbital = _write_cube(
        tmp_path,
        "reference_orbital.cube",
        [1.0] * 8,
        title="Molecular orbital",
        comment="Matched orbital on a common grid",
    )
    pyscf_orbital = _write_cube(
        tmp_path,
        "pyscf_orbital.cube",
        [-1.0] * 8,
        title="Molecular orbital",
        comment="Matched orbital on a common grid",
    )
    pyscf_result = _pyscf_result(density_cube={
        "status": "written",
        "path": str(pyscf_density),
        "density_value_unit": "electron_per_bohr3",
    })
    reference = _reference(
        density_cube={
            "path": str(reference_density),
            "density_value_unit": "electron_per_bohr3",
        },
        orbital_cube={
            "path": str(reference_orbital),
            "orbital_label": "HOMO",
        },
    )

    report = compare_pyscf_reference_calculation(
        pyscf_result,
        reference,
        pyscf_orbital_cube={
            "status": "written",
            "path": str(pyscf_orbital),
            "orbital_label": "HOMO",
            "spin": "restricted",
            "orbital_index": 0,
        },
    )

    assert report["field_comparisons"]["density"]["status"] == "comparable"
    assert report["field_comparisons"]["orbital"]["metrics"] == {
        "signed_normalized_overlap": pytest.approx(-1.0),
        "phase_alignment": "flip_candidate_sign",
        "phase_aligned_normalized_overlap": pytest.approx(1.0),
        "phase_aligned_l2_distance": pytest.approx(0.0),
    }


def test_pyscf_reference_report_refuses_an_incomplete_pyscf_result():
    with pytest.raises(ValueError, match="completed status"):
        compare_pyscf_reference_calculation(
            _pyscf_result(status="runtime_error"),
            _reference(),
        )


def test_pyscf_reference_report_is_exposed_through_mcp():
    payload = dispatch_tool("compare_pyscf_reference_calculation", {
        "pyscf_result": _pyscf_result(),
        "reference": _reference(),
    })

    assert payload["status"] == "compared"
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "compare_pyscf_reference_calculation"
    )
    assert definition["inputSchema"]["required"] == ["pyscf_result", "reference"]

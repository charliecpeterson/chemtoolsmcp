"""Contracts for explicitly bounded NWChem-to-PySCF reference drafts."""

from __future__ import annotations

import pytest

from chemtools.core.pyscf_comparison import compare_pyscf_reference_calculation
from chemtools.mcp.dispatch import dispatch_tool, tool_definitions
from chemtools.programs.nwchem.pyscf_reference import (
    NWCHEM_PYSCF_REFERENCE_DRAFT_SCHEMA,
    draft_nwchem_pyscf_reference,
)


def _write_h2_case(tmp_path, *, units="angstrom", basis="sto-3g"):
    input_path = tmp_path / "h2.nw"
    input_path.write_text(
        f"""geometry units {units}
H 0.0 0.0 0.0
H 0.0 0.0 0.74
end
charge 0
basis
  * library {basis}
end
scf
  singlet
end
task scf energy
""",
        encoding="utf-8",
    )
    output_path = tmp_path / "h2.out"
    output_path.write_text(
        """Starting SCF solution at 0.0 seconds
Total SCF energy = -1.1167593074
""",
        encoding="utf-8",
    )
    return input_path, output_path


def test_draft_extracts_nwchem_evidence_without_filling_unsafe_fields(tmp_path):
    input_path, output_path = _write_h2_case(tmp_path)

    draft = draft_nwchem_pyscf_reference(
        str(input_path),
        output_path=str(output_path),
    )

    assert draft["schema_version"] == NWCHEM_PYSCF_REFERENCE_DRAFT_SCHEMA
    assert draft["comparison_ready"] is False
    assert draft["missing_required_fields"] == [
        "calculation.method",
        "calculation.density_fit",
        "electrons.total",
    ]
    assert draft["reference_draft"] == {
        "label": "NWChem reference: h2.nw",
        "geometry": [
            {"element": "H", "x": 0.0, "y": 0.0, "z": 0.0},
            {"element": "H", "x": 0.0, "y": 0.0, "z": 0.74},
        ],
        "calculation": {
            "method": None,
            "basis": "sto-3g",
            "xc": None,
            "density_fit": None,
            "charge": 0,
            "multiplicity": 1,
        },
        "scf": {"converged": True},
        "energy": {"total_hartree": -1.1167593074},
        "electrons": {"total": None},
    }
    assert draft["field_sources"]["calculation.method"]["nwchem_task_module"] == "scf"
    assert draft["field_sources"]["electrons.total"]["status"] == "missing"


def test_draft_is_comparison_ready_only_after_caller_declarations(tmp_path):
    input_path, output_path = _write_h2_case(tmp_path, units="bohr")

    draft = draft_nwchem_pyscf_reference(
        str(input_path),
        output_path=str(output_path),
        label="NWChem H2 RHF/STO-3G",
        pyscf_method="rhf",
        density_fit=False,
        electron_total=2,
    )

    assert draft["comparison_ready"] is True
    assert draft["missing_required_fields"] == []
    assert draft["reference_draft"]["label"] == "NWChem H2 RHF/STO-3G"
    assert draft["reference_draft"]["geometry"][1]["z"] == pytest.approx(
        0.74 * 0.529177210903,
    )
    assert draft["field_sources"]["calculation.method"]["status"] == "caller_declared"
    assert draft["field_sources"]["calculation.density_fit"]["value"] is False
    report = compare_pyscf_reference_calculation(
        {
            "schema_version": "chemtools.pyscf-single-point-result/1",
            "status": "completed",
            "calculation": {
                "method": "rhf",
                "basis": "sto-3g",
                "xc": None,
                "density_fit": False,
                "charge": 0,
                "multiplicity": 1,
            },
            "geometry": draft["reference_draft"]["geometry"],
            "provenance": {"pyscf_version": "test", "python_version": "test"},
            "scf": {"converged": True},
            "energy": {"total_hartree": -1.1167593074},
            "electrons": {"total": 2, "alpha": 1, "beta": 1},
        },
        draft["reference_draft"],
    )
    assert report["status"] == "compared"


def test_draft_refuses_implicit_units_and_ambiguous_basis(tmp_path):
    input_path, _ = _write_h2_case(tmp_path, units="angstrom")
    input_path.write_text(
        input_path.read_text(encoding="utf-8")
        .replace("geometry units angstrom", "geometry")
        .replace("  * library sto-3g", "  H library sto-3g\n  H library 6-31g"),
        encoding="utf-8",
    )

    draft = draft_nwchem_pyscf_reference(str(input_path))

    assert draft["reference_draft"]["geometry"] is None
    assert draft["field_sources"]["geometry"]["reason"] == (
        "The Cartesian geometry block does not declare coordinate units."
    )
    assert draft["reference_draft"]["calculation"]["basis"] is None
    assert draft["field_sources"]["calculation.basis"]["basis_names"] == [
        "6-31g",
        "sto-3g",
    ]


def test_draft_validates_caller_declarations_and_is_exposed_through_mcp(tmp_path):
    input_path, _ = _write_h2_case(tmp_path)

    with pytest.raises(ValueError, match="pyscf_method"):
        draft_nwchem_pyscf_reference(str(input_path), pyscf_method="rohf")
    with pytest.raises(ValueError, match="density_fit"):
        draft_nwchem_pyscf_reference(str(input_path), density_fit=0)
    with pytest.raises(ValueError, match="electron_total"):
        draft_nwchem_pyscf_reference(str(input_path), electron_total=True)

    draft = dispatch_tool("draft_nwchem_pyscf_reference", {
        "input_file": str(input_path),
        "pyscf_method": "rhf",
        "density_fit": False,
        "electron_total": 2,
    })
    assert draft["reference_draft"]["calculation"]["method"] == "rhf"
    definition = next(
        item for item in tool_definitions()
        if item["name"] == "draft_nwchem_pyscf_reference"
    )
    assert definition["inputSchema"]["required"] == ["input_file"]


def test_dft_draft_requires_an_explicit_pyscf_xc_declaration(tmp_path):
    input_path, output_path = _write_h2_case(tmp_path)
    input_path.write_text(
        input_path.read_text(encoding="utf-8")
        .replace("scf\n  singlet\nend\ntask scf energy", "dft\n  mult 1\n  xc pbe0\nend\ntask dft energy"),
        encoding="utf-8",
    )

    draft = draft_nwchem_pyscf_reference(
        str(input_path),
        output_path=str(output_path),
        pyscf_method="rks",
        density_fit=False,
        electron_total=2,
    )

    assert draft["comparison_ready"] is False
    assert draft["missing_required_fields"] == ["calculation.xc"]
    assert draft["field_sources"]["calculation.xc"]["nwchem_xc"] == "pbe0"

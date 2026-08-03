"""Contracts for the composed NWChem-to-PySCF matched-run workflow."""

from __future__ import annotations

import pytest

from chemtools.application import nwchem_pyscf
from chemtools.mcp.dispatch import tool_definitions
from chemtools.mcp.tools import nwchem_analysis


def _write_h2_case(tmp_path, *, geometry_header="geometry units angstrom"):
    input_path = tmp_path / "h2.nw"
    input_path.write_text(
        f"""{geometry_header}
H 0.0 0.0 0.0
H 0.0 0.0 0.74
end
charge 0
basis
  * library sto-3g
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


def _arguments(input_path, output_path, working_directory):
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "working_directory": str(working_directory),
        "pyscf_method": "rhf",
        "density_fit": False,
        "electron_total": 2,
    }


def _write_density_cube(tmp_path, name):
    path = tmp_path / name
    path.write_text(
        "\n".join((
            "Electron density",
            "Total density on a common grid",
            "2 0.0 0.0 0.0",
            "2 1.0 0.0 0.0",
            "2 0.0 1.0 0.0",
            "2 0.0 0.0 1.0",
            "1 1.0 0.0 0.0 0.0",
            "1 1.0 0.0 0.0 1.3983973322",
            "1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0",
            "",
        )),
        encoding="utf-8",
    )
    return path


def test_matched_run_composes_existing_pyscf_runner_and_comparison(tmp_path, monkeypatch):
    input_path, output_path = _write_h2_case(tmp_path)
    captured = {}

    def run(_service, **arguments):
        captured.update(arguments)
        return {
            "execution": {"status": "completed"},
            "result": {
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
                "geometry": [
                    {"element": "H", "x": 0.0, "y": 0.0, "z": 0.0},
                    {"element": "H", "x": 0.0, "y": 0.0, "z": 0.74},
                ],
                "provenance": {"pyscf_version": "test", "python_version": "test"},
                "scf": {"converged": True},
                "energy": {"total_hartree": -1.1},
                "electrons": {"total": 2, "alpha": 1, "beta": 1},
            },
        }

    monkeypatch.setattr(nwchem_pyscf, "run_pyscf_single_point", run)
    response = nwchem_pyscf.run_nwchem_pyscf_matched_reference(
        object(),
        **_arguments(input_path, output_path, tmp_path),
    )

    assert response["status"] == "compared"
    assert captured["atoms"] == response["reference_draft"]["reference_draft"]["geometry"]
    assert captured["method"] == "rhf"
    assert captured["basis"] == "sto-3g"
    assert response["comparison"]["matching"]["geometry"]["status"] == "matched"
    assert response["comparison"]["energy"]["pyscf_minus_reference_hartree"] == -1.1 + 1.1167593074


def test_matched_run_refuses_execution_for_an_incomplete_reference(tmp_path, monkeypatch):
    input_path, output_path = _write_h2_case(tmp_path, geometry_header="geometry")
    monkeypatch.setattr(
        nwchem_pyscf,
        "run_pyscf_single_point",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    response = nwchem_pyscf.run_nwchem_pyscf_matched_reference(
        object(),
        **_arguments(input_path, output_path, tmp_path),
    )

    assert response["status"] == "reference_incomplete"
    assert response["comparison"] is None
    assert "geometry" in response["reference_draft"]["missing_required_fields"]


def test_matched_run_dry_run_validates_reference_before_rendering(tmp_path, monkeypatch):
    input_path, output_path = _write_h2_case(tmp_path)

    def render(**arguments):
        assert arguments["working_directory"] == str(tmp_path)
        assert arguments["charge"] == 0
        return None, None, {"status": "rendered", "method": arguments["method"]}

    monkeypatch.setattr(nwchem_pyscf, "render_pyscf_single_point", render)
    response = nwchem_pyscf.run_nwchem_pyscf_matched_reference(
        object(),
        **_arguments(input_path, output_path, tmp_path),
        dry_run=True,
    )

    assert response["status"] == "previewed"
    assert response["pyscf_launch"] == {"status": "rendered", "method": "rhf"}
    assert response["comparison"] is None


def test_matched_run_compares_declared_nwchem_and_pyscf_density_cubes(
    tmp_path,
    monkeypatch,
):
    input_path, output_path = _write_h2_case(tmp_path)
    reference_cube = _write_density_cube(tmp_path, "nwchem_density.cube")
    pyscf_cube = _write_density_cube(tmp_path, "pyscf_density.cube")

    def run(_service, **arguments):
        assert arguments["density_cube_grid_points"] == 20
        return {
            "execution": {"status": "completed"},
            "result": {
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
                "geometry": [
                    {"element": "H", "x": 0.0, "y": 0.0, "z": 0.0},
                    {"element": "H", "x": 0.0, "y": 0.0, "z": 0.74},
                ],
                "provenance": {"pyscf_version": "test", "python_version": "test"},
                "scf": {"converged": True},
                "energy": {"total_hartree": -1.1},
                "electrons": {"total": 2, "alpha": 1, "beta": 1},
                "density_cube": {
                    "status": "written",
                    "path": str(pyscf_cube),
                    "density_value_unit": "electron_per_bohr3",
                },
            },
        }

    monkeypatch.setattr(nwchem_pyscf, "run_pyscf_single_point", run)
    response = nwchem_pyscf.run_nwchem_pyscf_matched_reference(
        object(),
        **_arguments(input_path, output_path, tmp_path),
        reference_density_cube={
            "path": str(reference_cube),
            "density_value_unit": "electron_per_bohr3",
        },
        density_cube_grid_points=20,
    )

    assert response["status"] == "compared"
    assert response["reference_draft"]["reference_draft"]["density_cube"] == {
        "path": str(reference_cube),
        "density_value_unit": "electron_per_bohr3",
    }
    assert response["comparison"]["field_comparisons"]["density"]["status"] == "comparable"
    assert response["comparison"]["field_comparisons"]["density"]["metrics"]["l1_difference_electrons"] == 0.0


def test_matched_run_requires_a_complete_declared_density_cube_pair(tmp_path):
    input_path, output_path = _write_h2_case(tmp_path)
    reference_cube = _write_density_cube(tmp_path, "nwchem_density.cube")
    arguments = _arguments(input_path, output_path, tmp_path)

    with pytest.raises(ValueError, match="supplied together"):
        nwchem_pyscf.run_nwchem_pyscf_matched_reference(
            object(),
            **arguments,
            reference_density_cube={
                "path": str(reference_cube),
                "density_value_unit": "electron_per_bohr3",
            },
        )
    with pytest.raises(ValueError, match="is not a file"):
        nwchem_pyscf.run_nwchem_pyscf_matched_reference(
            object(),
            **arguments,
            reference_density_cube={
                "path": str(tmp_path / "missing.cube"),
                "density_value_unit": "electron_per_bohr3",
            },
            density_cube_grid_points=20,
        )


def test_matched_run_tool_is_exposed_as_an_executable_nwchem_tool():
    definition = next(
        item for item in tool_definitions()
        if item["name"] == "run_nwchem_pyscf_matched_reference"
    )

    assert definition["inputSchema"]["required"] == [
        "input_file",
        "output_file",
        "working_directory",
        "pyscf_method",
        "density_fit",
        "electron_total",
    ]


def test_matched_run_handler_uses_the_execution_service(monkeypatch):
    captured = {}

    monkeypatch.setattr(nwchem_analysis, "get_execution_service", lambda: "service")

    def run(service, **arguments):
        captured["service"] = service
        captured.update(arguments)
        return {"status": "previewed"}

    monkeypatch.setattr(
        nwchem_analysis,
        "run_nwchem_pyscf_matched_reference",
        run,
    )

    response = nwchem_analysis._handle_run_nwchem_pyscf_matched_reference({
        "input_file": "h2.nw",
        "output_file": "h2.out",
        "working_directory": "/scratch/h2",
        "pyscf_method": "rhf",
        "density_fit": False,
        "electron_total": 2,
        "dry_run": True,
    })

    assert response == {"status": "previewed"}
    assert captured == {
        "service": "service",
        "input_path": "h2.nw",
        "output_path": "h2.out",
        "working_directory": "/scratch/h2",
        "pyscf_method": "rhf",
        "pyscf_xc": None,
        "density_fit": False,
        "electron_total": 2,
        "reference_density_cube": None,
        "density_cube_grid_points": None,
        "label": None,
        "max_cycles": 100,
        "convergence_tolerance": 1e-9,
        "max_memory_mb": 2048,
        "omp_threads": 1,
        "timeout_seconds": 120.0,
        "job_name": "nwchem_pyscf_match",
        "dry_run": True,
    }

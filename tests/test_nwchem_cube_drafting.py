"""Regression coverage for NWChem CUBE grids shared with PySCF."""

from __future__ import annotations

import pytest

from chemtools.core.units import ANGSTROM_PER_BOHR
from chemtools.mcp.dispatch import tool_definitions
from chemtools.programs.nwchem.input.cube import draft_nwchem_cube_input


def _write_h2_input(tmp_path, *, units: str = "angstrom"):
    input_path = tmp_path / "h2.nw"
    input_path.write_text(
        f"""geometry units {units}
  H 0.0 0.0 0.0
  H 0.0 0.0 0.74
end
basis
  * library sto-3g
end
task scf energy
""",
        encoding="utf-8",
    )
    return input_path


def test_draft_derives_a_pyscf_compatible_density_grid(tmp_path):
    input_path = _write_h2_input(tmp_path)

    drafted = draft_nwchem_cube_input(
        str(input_path),
        vectors_input="h2.movecs",
        density_modes=["total"],
        pyscf_compatible_grid_points=20,
    )

    z_upper = 0.74 / ANGSTROM_PER_BOHR + 3.0
    assert drafted["cube_grid"] == {
        "kind": "pyscf_compatible",
        "source_geometry_units": "angstrom",
        "coordinate_unit": "bohr",
        "lower_bounds_bohr": [-3.0, -3.0, -3.0],
        "upper_bounds_bohr": pytest.approx([3.0, 3.0, z_upper]),
        "grid_points": [20, 20, 20],
        "nwchem_spacings": [19, 19, 19],
        "pyscf_margin_bohr": 3.0,
        "preserve_input_cartesian_frame": True,
    }
    assert "  limitxyz units bohr" in drafted["input_text"]
    assert "geometry units angstrom nocenter noautosym noautoz" in drafted["input_text"]
    assert "   -3.000000000000  3.000000000000  19" in drafted["input_text"]
    assert (
        f"   -3.000000000000  {z_upper:.12f}  19" in drafted["input_text"]
    )


def test_draft_uses_bohr_coordinates_without_conversion(tmp_path):
    input_path = _write_h2_input(tmp_path, units="au")

    drafted = draft_nwchem_cube_input(
        str(input_path),
        vectors_input="h2.movecs",
        density_modes=["total"],
        pyscf_compatible_grid_points=20,
    )

    assert drafted["cube_grid"]["source_geometry_units"] == "bohr"
    assert drafted["cube_grid"]["upper_bounds_bohr"] == [3.0, 3.0, 3.74]


def test_pyscf_compatible_grid_requires_explicit_geometry_units(tmp_path):
    input_path = tmp_path / "h2.nw"
    input_path.write_text(
        "geometry\n  H 0 0 0\n  H 0 0 0.74\nend\ntask scf energy\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="explicit geometry units"):
        draft_nwchem_cube_input(
            str(input_path),
            vectors_input="h2.movecs",
            density_modes=["total"],
            pyscf_compatible_grid_points=20,
        )


def test_cube_tools_expose_the_pyscf_compatible_grid_option():
    definitions = {item["name"]: item for item in tool_definitions()}

    for tool_name in (
        "draft_nwchem_cube_input",
        "draft_nwchem_frontier_cube_input",
    ):
        assert definitions[tool_name]["inputSchema"]["properties"][
            "pyscf_compatible_grid_points"
        ] == {
            "type": "integer",
            "minimum": 20,
            "maximum": 120,
            "description": "Derive a PySCF-compatible limitxyz grid from one explicit-unit Cartesian geometry. Overrides the symmetric extent_angstrom/grid_points box.",
        }

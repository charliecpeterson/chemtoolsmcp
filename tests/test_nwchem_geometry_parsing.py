"""NWChem output-coordinate parsing contracts shared by analysis tools."""

from __future__ import annotations

from chemtools.core.units import ANGSTROM_PER_BOHR
from chemtools.programs.nwchem.parse.geometry import (
    OutputGeometryScanner,
    extract_output_geometry,
)


def test_extract_output_geometry_selects_first_or_last_complete_table():
    text = (
        "Output coordinates in angstroms\n"
        "No. Tag Charge X Y Z\n"
        "1 H 1.0 -0.37 0.0 0.0\n"
        "2 H 1.0  0.37 0.0 0.0\n"
        "Atomic Mass\n"
        "intermediate output\n"
        "Output coordinates in a.u.\n"
        "No. Tag Charge X Y Z\n"
        "1 H1 1.0 -1.0 0.0 0.0\n"
        "2 H2 1.0  1.0 0.0 0.0\n"
        "Atomic Mass\n"
    )

    first = extract_output_geometry(
        text.splitlines(),
        which="first",
    )
    last = extract_output_geometry(
        text.splitlines(),
        which="last",
    )

    assert first == {
        "atoms": [
            {
                "label": "H",
                "element": "H",
                "x": -0.37,
                "y": 0.0,
                "z": 0.0,
            },
            {
                "label": "H",
                "element": "H",
                "x": 0.37,
                "y": 0.0,
                "z": 0.0,
            },
        ],
        "atom_count": 2,
        "source_units": "angstrom",
        "units": "angstrom",
    }
    assert last == {
        "atoms": [
            {
                "label": "H1",
                "element": "H",
                "x": -ANGSTROM_PER_BOHR,
                "y": 0.0,
                "z": 0.0,
            },
            {
                "label": "H2",
                "element": "H",
                "x": ANGSTROM_PER_BOHR,
                "y": 0.0,
                "z": 0.0,
            },
        ],
        "atom_count": 2,
        "source_units": "bohr",
        "units": "angstrom",
    }


def test_extract_output_geometry_rejects_incomplete_table():
    text = (
        "Output coordinates in angstroms\n"
        "No. Tag Charge X Y Z\n"
        "1 H 1.0 0.0 0.0 0.0\n"
    )

    assert extract_output_geometry(
        text.splitlines(),
        which="first",
    ) is None


def test_output_geometry_scanner_indexes_named_geometries():
    scanner = OutputGeometryScanner()
    text = (
        'Geometry "fragment" -> ""\n'
        "Output coordinates in angstroms\n"
        "No. Tag Charge X Y Z\n"
        "1 Fe 26.0 0.0 0.0 0.0\n"
        "Atomic Mass\n"
    )

    for line in text.splitlines():
        scanner.feed(line)

    assert scanner.first_by_name["fragment"]["name"] == "fragment"
    assert scanner.first_by_name["fragment"]["atom_count"] == 1
    assert scanner.last_by_name == scanner.first_by_name

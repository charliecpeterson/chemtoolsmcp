"""MCP contracts for fixed, read-only Orbitron analysis operations."""

from __future__ import annotations

import pytest

from chemtools.core.units import HARTREE_TO_EV
from chemtools.integrations.orbitron import (
    OrbitronResponse,
    OrbitronUnavailableError,
    OrbitronVersion,
)
from chemtools.mcp.decorator import _TOOL_PROGRAMS
from chemtools.mcp.dispatch import dispatch_tool, tool_definitions
from chemtools.mcp.tools import orbitron as orbitron_tools


_PRODUCER = {
    "name": "orbitron",
    "version": "0.4.0",
    "commit": "58aa65b3f280",
}
_CANONICAL_PRODUCER = {
    "producer_type": "external_tool",
    **_PRODUCER,
}


def _version():
    return OrbitronVersion(
        version="0.4.0",
        commit="58aa65b3f280",
        raw="orbitron-cli 0.4.0 (58aa65b3f280)",
    )


def _geometry_response(source, **overrides):
    payload = {
        "schema": "orbitron.analyze.geometry/3",
        "producer": _PRODUCER,
        "warnings": [],
        "path": str(source.resolve()),
        "format": "xyz",
        "geometry_role": "input",
        "geometry_source": "the geometry stored in the input file",
        "distance_unit": "angstrom",
        "atoms": 3,
        "bonds": 2,
        "elements": {"H": 2, "O": 1},
        "coordination": {"H:1": 2, "O:2": 1},
        "dangling_atoms": 0,
        "bond_lengths": {
            "count": 2,
            "min": 0.95,
            "max": 0.96,
            "mean": 0.955,
            "std_dev": 0.005,
        },
        "bounding_box": {
            "min": [-0.75, 0.0, -0.47],
            "max": [0.75, 0.0, 0.12],
        },
        "center": [0.0, 0.0, -0.175],
        "span": [1.5, 0.0, 0.59],
        "unit_cell": None,
        **overrides,
    }
    return OrbitronResponse(
        operation="analyze_geometry",
        source=str(source.resolve()),
        schema="orbitron.analyze.geometry/3",
        producer=_PRODUCER,
        warnings=(),
        payload=payload,
        stderr="",
        version=_version(),
    )


def _orbital_entry(label, vector, energy_hartree, occupancy):
    return {
        "label": label,
        "vector": vector,
        "energy_hartree": energy_hartree,
        "energy_ev": energy_hartree * HARTREE_TO_EV,
        "occupancy": occupancy,
        "symmetry": "a",
        "spin": None,
    }


def _orbital_response(source):
    homo = _orbital_entry("HOMO", 1, -0.4, 2.0)
    lumo = _orbital_entry("LUMO", 2, 0.1, 0.0)
    payload = {
        "schema": "orbitron.analyze.orbitals/2",
        "producer": _PRODUCER,
        "warnings": [],
        "path": str(source.resolve()),
        "format": "out",
        "total_orbitals": 2,
        "occupied_count": 1,
        "virtual_count": 1,
        "homo": homo,
        "lumo": lumo,
        "gap_hartree": 0.5,
        "gap_ev": 0.5 * HARTREE_TO_EV,
        "frontier": [homo, lumo],
        "unrestricted": False,
        "occupied_threshold": 0.1,
        "spin_channels": [
            {
                "spin": "restricted",
                "orbital_count": 2,
                "occupied_count": 1,
                "virtual_count": 1,
                "homo": homo,
                "lumo": lumo,
                "gap_hartree": 0.5,
                "gap_ev": 0.5 * HARTREE_TO_EV,
                "frontier": [homo, lumo],
            }
        ],
    }
    return OrbitronResponse(
        operation="analyze_orbitals",
        source=str(source.resolve()),
        schema="orbitron.analyze.orbitals/2",
        producer=_PRODUCER,
        warnings=(),
        payload=payload,
        stderr="",
        version=_version(),
    )


def _population_response(source):
    sodium = {
        "atom_index": 1,
        "element": "Na",
        "charge": 1.0,
    }
    population_warning = "Large partial charge magnitude detected (>1.5 e)."
    payload = {
        "schema": "orbitron.analyze.populations/2",
        "producer": _PRODUCER,
        "warnings": [],
        "path": str(source.resolve()),
        "format": "log",
        "methods": [
            {
                "method": "Mulliken",
                "atom_count": 1,
                "total_charge": 1.0,
                "expected_total_charge": 1.0,
                "expected_charge_source": "declared",
                "charge_residual": 0.0,
                "min_charge": 1.0,
                "max_charge": 1.0,
                "mean_abs_charge": 1.0,
                "charges": [sodium],
                "charges_by_atom": {"1": sodium},
                "top_charges": [sodium],
                "warnings": [population_warning],
            }
        ],
    }
    return OrbitronResponse(
        operation="analyze_populations",
        source=str(source.resolve()),
        schema="orbitron.analyze.populations/2",
        producer=_PRODUCER,
        warnings=(),
        payload=payload,
        stderr="",
        version=_version(),
    )


def _vibration_response(
    source,
    *,
    geometry_role="single_point",
    geometry_source="the only geometry the run reports",
):
    mode = {
        "index": 1,
        "frequency": -100.0,
        "magnitude": 100.0,
        "label": "a",
        "has_displacement": False,
    }
    payload = {
        "schema": "orbitron.analyze.vibrations/4",
        "producer": _PRODUCER,
        "warnings": [],
        "path": str(source.resolve()),
        "format": "out",
        "geometry_role": geometry_role,
        "geometry_source": geometry_source,
        "mode_set": "raw",
        "frequency_unit": "cm^-1",
        "frequency_scale_factor": 1.0,
        "has_displacements": False,
        "displacement_mode_count": 0,
        "mode_count": 1,
        "imaginary_count": 1,
        "lowest_frequency": -100.0,
        "highest_frequency": -100.0,
        "mean_frequency": -100.0,
        "modes": [mode],
        "thermochemistry": {
            "temperature_kelvin": 298.15,
            "pressure_atm": None,
            "zero_point_correction_kcal_mol": 10.0,
            "thermal_correction_energy_kcal_mol": 11.0,
            "thermal_correction_enthalpy_kcal_mol": 11.6,
            "thermal_correction_gibbs_kcal_mol": 1.0,
            "total_entropy_cal_mol_k": 35.0,
            "cv_total_cal_mol_k": 6.0,
            "molecular_weight_amu": 27.0,
            "symmetry_number": 1,
        },
    }
    return OrbitronResponse(
        operation="analyze_vibrations",
        source=str(source.resolve()),
        schema="orbitron.analyze.vibrations/4",
        producer=_PRODUCER,
        warnings=(),
        payload=payload,
        stderr="",
        version=_version(),
    )


def test_mcp_returns_validated_geometry_analysis(tmp_path, monkeypatch):
    source = tmp_path / "water.xyz"
    source.write_text("geometry\n")
    response = _geometry_response(source)

    class Client:
        def analyze_geometry(self, path):
            assert path == str(source)
            return response

    monkeypatch.setattr(orbitron_tools, "OrbitronClient", Client)

    analyzed = dispatch_tool(
        "analyze_geometry_with_orbitron",
        {"path": str(source)},
    )

    assert analyzed == {
        "schema_version": "chemtools.orbitron-geometry-analysis/3",
        "status": "ok",
        "operation": "analyze_geometry",
        "source": str(source.resolve()),
        "orbitron_schema": "orbitron.analyze.geometry/3",
        "producer": _PRODUCER,
        "warnings": [],
        "uncertainty": [],
        "evidence": {
            key: value
            for key, value in response.payload.items()
            if key not in {"schema", "producer", "warnings"}
        },
        "canonical_mapping": {
            "producer": _CANONICAL_PRODUCER,
            "scientific_system": {
                "status": "insufficient_evidence",
                "reason": (
                    "orbitron.analyze.geometry/3 reports geometry summaries "
                    "but not atom identities with coordinates."
                ),
            },
        },
    }


def test_mcp_marks_last_attempted_geometry_as_diagnostic_evidence(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "failed-relax.out"
    source.write_text("failed relaxation\n")
    response = _geometry_response(
        source,
        geometry_role="last_attempted",
        geometry_source="step 8 of 8; the run stopped without converging",
    )

    class Client:
        def analyze_geometry(self, path):
            assert path == str(source)
            return response

    monkeypatch.setattr(orbitron_tools, "OrbitronClient", Client)

    analyzed = dispatch_tool(
        "analyze_geometry_with_orbitron",
        {"path": str(source)},
    )

    assert analyzed["uncertainty"] == [
        {
            "code": "orbitron_geometry_not_converged",
            "message": "step 8 of 8; the run stopped without converging",
            "impact": (
                "Use this geometry to diagnose the failed run, not as a "
                "converged structure."
            ),
        }
    ]


def test_mcp_returns_validated_orbital_analysis(tmp_path, monkeypatch):
    source = tmp_path / "run.out"
    source.write_text("orbitals\n")
    response = _orbital_response(source)

    class Client:
        def analyze_orbitals(self, path):
            assert path == str(source)
            return response

    monkeypatch.setattr(orbitron_tools, "OrbitronClient", Client)

    analyzed = dispatch_tool(
        "analyze_orbitals_with_orbitron",
        {"path": str(source)},
    )

    assert analyzed == {
        "schema_version": "chemtools.orbitron-orbital-analysis/2",
        "status": "ok",
        "operation": "analyze_orbitals",
        "source": str(source.resolve()),
        "orbitron_schema": "orbitron.analyze.orbitals/2",
        "producer": _PRODUCER,
        "parameters": {"frontier_count": 3},
        "warnings": [],
        "uncertainty": [],
        "evidence": {
            key: value
            for key, value in response.payload.items()
            if key not in {"schema", "producer", "warnings"}
        },
        "canonical_mapping": {
            "producer": _CANONICAL_PRODUCER,
            "electronic_structure": {
                "status": "not_mapped",
                "reason": (
                    "Chemtools does not yet define a canonical molecular-"
                    "orbital summary model."
                ),
            },
        },
    }


def test_mcp_returns_validated_population_analysis(tmp_path, monkeypatch):
    source = tmp_path / "sodium.log"
    source.write_text("population analysis\n")
    response = _population_response(source)

    class Client:
        def analyze_populations(self, path):
            assert path == str(source)
            return response

    monkeypatch.setattr(orbitron_tools, "OrbitronClient", Client)

    analyzed = dispatch_tool(
        "analyze_populations_with_orbitron",
        {"path": str(source)},
    )

    assert analyzed == {
        "schema_version": "chemtools.orbitron-population-analysis/2",
        "status": "ok",
        "operation": "analyze_populations",
        "source": str(source.resolve()),
        "orbitron_schema": "orbitron.analyze.populations/2",
        "producer": _PRODUCER,
        "parameters": {"top_count": 8},
        "warnings": [],
        "uncertainty": [
            {
                "code": "orbitron_population_method_warning",
                "method": "Mulliken",
                "message": "Large partial charge magnitude detected (>1.5 e).",
                "impact": (
                    "Review the population data before using this method's "
                    "atomic charges."
                ),
            },
        ],
        "evidence": {
            key: value
            for key, value in response.payload.items()
            if key not in {"schema", "producer", "warnings"}
        },
        "canonical_mapping": {
            "producer": _CANONICAL_PRODUCER,
            "electronic_structure": {
                "status": "not_mapped",
                "reason": (
                    "Chemtools does not yet define a canonical atomic-"
                    "population summary model."
                ),
            },
        },
    }


def test_mcp_marks_population_expected_charge_unknown(tmp_path, monkeypatch):
    source = tmp_path / "unknown-charge.log"
    source.write_text("population analysis\n")
    response = _population_response(source)
    method = response.payload["methods"][0]
    method["expected_total_charge"] = None
    method["expected_charge_source"] = None
    method["charge_residual"] = None
    method["warnings"] = []

    class Client:
        def analyze_populations(self, path):
            assert path == str(source)
            return response

    monkeypatch.setattr(orbitron_tools, "OrbitronClient", Client)

    analyzed = dispatch_tool(
        "analyze_populations_with_orbitron",
        {"path": str(source)},
    )

    assert analyzed["uncertainty"] == [
        {
            "code": "orbitron_population_expected_charge_unknown",
            "method": "Mulliken",
            "message": (
                "The source does not establish the expected total charge for "
                "this population analysis."
            ),
            "impact": (
                "The partial-charge sum cannot be checked against the charge "
                "of the calculated system."
            ),
        }
    ]


def test_mcp_returns_validated_vibration_analysis(tmp_path, monkeypatch):
    source = tmp_path / "ts.out"
    source.write_text("frequency analysis\n")
    response = _vibration_response(source)

    class Client:
        def analyze_vibrations(self, path):
            assert path == str(source)
            return response

    monkeypatch.setattr(orbitron_tools, "OrbitronClient", Client)

    analyzed = dispatch_tool(
        "analyze_vibrations_with_orbitron",
        {"path": str(source)},
    )

    assert analyzed["schema_version"] == "chemtools.orbitron-vibration-analysis/3"
    assert analyzed["status"] == "ok"
    assert analyzed["orbitron_schema"] == "orbitron.analyze.vibrations/4"
    assert analyzed["parameters"] == {"mode_set": "raw", "top_count": 10}
    assert analyzed["evidence"] == {
        key: value
        for key, value in response.payload.items()
        if key not in {"schema", "producer", "warnings"}
    }
    assert [item["code"] for item in analyzed["uncertainty"]] == [
        "orbitron_thermochemistry_standard_state_unknown",
        "orbitron_vibration_displacements_unavailable",
    ]
    assert analyzed["canonical_mapping"] == {
        "producer": _CANONICAL_PRODUCER,
        "vibrations": {
            "status": "not_mapped",
            "reason": (
                "Chemtools does not yet define a program-neutral "
                "vibration-analysis summary model."
            ),
        },
    }


def test_mcp_marks_vibrations_from_an_unconverged_geometry(tmp_path, monkeypatch):
    source = tmp_path / "failed-relaxation.out"
    source.write_text("frequency analysis\n")
    response = _vibration_response(
        source,
        geometry_role="last_attempted",
        geometry_source="step 8 of 8; the run stopped without converging",
    )

    class Client:
        def analyze_vibrations(self, path):
            assert path == str(source)
            return response

    monkeypatch.setattr(orbitron_tools, "OrbitronClient", Client)

    analyzed = dispatch_tool(
        "analyze_vibrations_with_orbitron",
        {"path": str(source)},
    )

    assert analyzed["uncertainty"][0] == {
        "code": "orbitron_vibration_geometry_not_converged",
        "message": (
            "Orbitron identified the frequency geometry as the last attempted "
            "structure: step 8 of 8; the run stopped without converging"
        ),
        "impact": (
            "Do not interpret these frequencies as stationary-point modes "
            "until the geometry is converged."
        ),
    }


@pytest.mark.parametrize(
    ("tool_name", "schema_version"),
    (
        (
            "analyze_geometry_with_orbitron",
            "chemtools.orbitron-geometry-analysis/3",
        ),
        (
            "analyze_orbitals_with_orbitron",
            "chemtools.orbitron-orbital-analysis/2",
        ),
        (
            "analyze_populations_with_orbitron",
            "chemtools.orbitron-population-analysis/2",
        ),
        (
            "analyze_vibrations_with_orbitron",
            "chemtools.orbitron-vibration-analysis/3",
        ),
    ),
)
def test_analysis_tools_report_their_own_failure_schema(
    monkeypatch,
    tool_name,
    schema_version,
):
    class Client:
        def __init__(self):
            raise OrbitronUnavailableError("missing")

    monkeypatch.setattr(orbitron_tools, "OrbitronClient", Client)

    analyzed = dispatch_tool(tool_name, {"path": "run.out"})

    assert analyzed == {
        "schema_version": schema_version,
        "status": "unavailable",
        "error": "orbitron_unavailable",
        "message": "missing",
    }


@pytest.mark.parametrize(
    ("tool_name", "description"),
    (
        (
            "analyze_geometry_with_orbitron",
            "Path to one local file supported by Orbitron. Directories, "
            "remote targets, output paths, and render arguments are not "
            "accepted.",
        ),
        (
            "analyze_orbitals_with_orbitron",
            "Path to one local file with molecular-orbital data supported by "
            "Orbitron. Commands, remote targets, output paths, and frontier "
            "overrides are not accepted.",
        ),
        (
            "analyze_populations_with_orbitron",
            "Path to one local file with atomic population data supported by "
            "Orbitron. Commands, remote targets, output paths, and top-count "
            "overrides are not accepted.",
        ),
        (
            "analyze_vibrations_with_orbitron",
            "Path to one local file with vibrational data supported by "
            "Orbitron. Commands, remote targets, output paths, projected-mode "
            "requests, and top-count overrides are not accepted.",
        ),
    ),
)
def test_analysis_tool_definitions_expose_only_a_local_path(
    tool_name,
    description,
):
    definition = next(
        item for item in tool_definitions() if item["name"] == tool_name
    )

    assert _TOOL_PROGRAMS[tool_name] == "generic"
    assert definition["inputSchema"] == {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "description": description,
            }
        },
        "required": ["path"],
        "additionalProperties": False,
    }

"""MCP tests for companion scientific-runtime availability reporting."""

from chemtools.integrations.science_runtime import (
    ScienceRuntimeProbe,
    ScienceRuntimeUnavailableError,
)
from chemtools.mcp.dispatch import dispatch_tool, tool_definitions
from chemtools.mcp.tools import science_runtime as science_runtime_tools


def test_science_runtime_tool_is_registered_as_generic():
    assert "inspect_science_runtime" in {
        definition["name"] for definition in tool_definitions()
    }


def test_science_runtime_tool_reports_available_packages(monkeypatch):
    class Client:
        def probe(self):
            return ScienceRuntimeProbe(
                python={
                    "executable": "/opt/chemtools-science/bin/python",
                    "implementation": "cpython",
                    "version": "3.12.11",
                },
                packages={
                    "pyscf": {"status": "available", "version": "2.11.0"},
                    "rdkit": {"status": "available", "version": "2026.03.4"},
                    "openbabel": {"status": "available", "version": "3.1.1"},
                    "h5py": {"status": "available", "version": "3.15.1"},
                    "orbitron": {"status": "available", "version": "0.4.0"},
                },
            )

    monkeypatch.setattr(science_runtime_tools, "ScienceRuntimeClient", Client)

    inspected = dispatch_tool("inspect_science_runtime", {})

    assert inspected["status"] == "ok"
    assert inspected["packages"]["pyscf"]["version"] == "2.11.0"


def test_science_runtime_tool_preserves_optional_absence(monkeypatch):
    class Client:
        def probe(self):
            raise ScienceRuntimeUnavailableError("not configured")

    monkeypatch.setattr(science_runtime_tools, "ScienceRuntimeClient", Client)

    assert dispatch_tool("inspect_science_runtime", {}) == {
        "schema_version": "chemtools.science-runtime-probe/1",
        "status": "unavailable",
        "error": "science_runtime_unavailable",
        "message": "not configured",
    }


def test_rdkit_preflight_tool_uses_only_the_fixed_request(monkeypatch):
    class Client:
        def rdkit_preflight(self, request):
            assert request == {
                "schema_version": "chemtools.rdkit-preflight-request/1",
                "format": "smiles",
                "source": "O",
            }
            return {"status": "valid", "formula": "H2O"}

    monkeypatch.setattr(science_runtime_tools, "ScienceRuntimeClient", Client)

    assert dispatch_tool("preflight_molecule_with_rdkit", {
        "format": "smiles",
        "source": "O",
    }) == {"status": "valid", "formula": "H2O"}


def test_openbabel_conversion_tool_uses_only_the_fixed_request(monkeypatch):
    class Client:
        def openbabel_convert(self, request):
            assert request == {
                "schema_version": "chemtools.openbabel-conversion-request/1",
                "format": "smiles",
                "source": "O",
                "output_format": "molblock",
            }
            return {"status": "completed", "comparison": {"status": "matched"}}

    monkeypatch.setattr(science_runtime_tools, "ScienceRuntimeClient", Client)

    assert dispatch_tool("convert_molecule_with_openbabel", {
        "format": "smiles",
        "source": "O",
        "output_format": "molblock",
    }) == {"status": "completed", "comparison": {"status": "matched"}}


def test_orbitron_periodic_inspection_tool_uses_only_the_fixed_request(monkeypatch):
    class Client:
        def orbitron_periodic_electronic_structure(self, request):
            assert request == {
                "schema_version": "chemtools.orbitron-periodic-electronic-structure-request/1",
                "path": "/work/vasprun.xml",
            }
            return {"status": "completed", "fermi_energy_ev": 1.2}

    monkeypatch.setattr(science_runtime_tools, "ScienceRuntimeClient", Client)

    assert dispatch_tool("inspect_periodic_electronic_structure_with_orbitron", {
        "path": "/work/vasprun.xml",
    }) == {"status": "completed", "fermi_energy_ev": 1.2}


def test_orbitron_structure_identity_tool_uses_only_the_fixed_request(monkeypatch):
    class Client:
        def orbitron_structure_identity(self, request):
            assert request == {
                "schema_version": "chemtools.orbitron-structure-identity-request/1",
                "path": "/work/zncl2.xyz",
            }
            return {"status": "completed", "bond_order_counts": {"Dative": 2}}

    monkeypatch.setattr(science_runtime_tools, "ScienceRuntimeClient", Client)

    assert dispatch_tool("inspect_structure_identity_with_orbitron", {
        "path": "/work/zncl2.xyz",
    }) == {"status": "completed", "bond_order_counts": {"Dative": 2}
    }


def test_orbitron_nbo_tool_uses_only_the_fixed_request(monkeypatch):
    class Client:
        def orbitron_nbo(self, request):
            assert request == {
                "schema_version": "chemtools.orbitron-nbo-request/1",
                "path": "/work/uo2-test.nbo",
            }
            return {"status": "completed", "orbital_count": 142}

    monkeypatch.setattr(science_runtime_tools, "ScienceRuntimeClient", Client)

    assert dispatch_tool("inspect_nbo_with_orbitron", {
        "path": "/work/uo2-test.nbo",
    }) == {"status": "completed", "orbital_count": 142}


def test_pyscf_dry_run_does_not_request_execution(monkeypatch):
    def render(**arguments):
        assert arguments["method"] == "rhf"
        return None, None, {"status": "rendered"}

    monkeypatch.setattr(science_runtime_tools, "render_pyscf_single_point", render)

    assert dispatch_tool("run_pyscf_single_point", {
        "atoms": [{"element": "H", "x": 0, "y": 0, "z": 0}],
        "charge": 0,
        "multiplicity": 1,
        "method": "rhf",
        "basis": "sto-3g",
        "working_directory": "/tmp",
        "dry_run": True,
    }) == {"status": "rendered"}

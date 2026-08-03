"""Pure request-boundary tests for the fixed science companion runner."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from chemtools import science_runner


def _pyscf_request(**overrides):
    request = {
        "schema_version": science_runner.PYSCF_SINGLE_POINT_REQUEST_SCHEMA,
        "atoms": [
            {"element": "H", "x": 0.0, "y": 0.0, "z": 0.0},
            {"element": "H", "x": 0.0, "y": 0.0, "z": 0.74},
        ],
        "charge": 0,
        "multiplicity": 1,
        "method": "rhf",
        "basis": "sto-3g",
        "xc": None,
        "density_fit": False,
        "max_cycles": 100,
        "convergence_tolerance": 1e-9,
        "max_memory_mb": 2048,
    }
    request.update(overrides)
    return request


def test_pyscf_request_normalizes_bounded_typed_values():
    parsed = science_runner._pyscf_request(_pyscf_request())

    assert parsed["atoms"][1]["z"] == 0.74
    assert parsed["method"] == "rhf"
    assert parsed["xc"] is None


def test_science_runner_attaches_shared_runtime_provenance(monkeypatch):
    written = []
    monkeypatch.setattr(
        science_runner,
        "_runtime_provenance",
        lambda: {"schema_version": "chemtools.companion-runtime-provenance/1"},
    )
    monkeypatch.setattr(
        science_runner,
        "_write_result",
        lambda result: written.append(result) or 0,
    )

    assert science_runner.main(["not-an-operation"]) == 0
    assert written == [{
        "schema_version": "chemtools.science-runner-error/1",
        "status": "invalid_operation",
        "message": (
            "operation must be rdkit-preflight, openbabel-convert, "
            "orbitron-periodic-electronic-structure, "
            "orbitron-structure-identity, orbitron-nbo, qmcpack-hdf5-inspect, "
            "bse-render, or "
            "pyscf-single-point"
        ),
        "runtime_provenance": {
            "schema_version": "chemtools.companion-runtime-provenance/1",
        },
    }]


def test_runtime_provenance_hashes_the_resolved_lock(tmp_path, monkeypatch):
    lock = tmp_path / "chemtools-science-linux-64.explicit.txt"
    lock.write_text("@EXPLICIT\nhttps://example.invalid/python.conda\n", encoding="utf-8")
    monkeypatch.setattr(science_runner, "SCIENCE_RUNTIME_LOCK_PATH", lock)
    monkeypatch.setattr(
        science_runner,
        "_installed_package_evidence",
        lambda distribution, module_name: {
            "status": "available",
            "version": f"{distribution}-version",
        },
    )

    provenance = science_runner._runtime_provenance()

    assert provenance["schema_version"] == (
        "chemtools.companion-runtime-provenance/1"
    )
    assert provenance["environment_lock"] == {
        "status": "available",
        "identifier": lock.name,
        "sha256": "832934c9b9499c52ec8f8b841ea192736d310b12f3e7d19d025da327b41fd681",
    }
    assert provenance["packages"]["openbabel"] == {
        "status": "available",
        "version": "openbabel-version",
    }


def test_runtime_provenance_uses_module_version_when_conda_lacks_metadata(
    monkeypatch,
):
    def missing_metadata(distribution):
        raise science_runner.importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(
        science_runner.importlib.metadata,
        "version",
        missing_metadata,
    )
    module = type("OpenBabelModule", (), {"__version__": "3.1.0"})()
    monkeypatch.setattr(
        science_runner.importlib,
        "import_module",
        lambda name: module,
    )

    assert science_runner._installed_package_evidence("openbabel", "openbabel") == {
        "status": "available",
        "version": "3.1.0",
    }


def test_runtime_provenance_binds_the_fixed_operation_to_the_request(monkeypatch):
    monkeypatch.setattr(
        science_runner,
        "_runtime_provenance",
        lambda: {"schema_version": "chemtools.companion-runtime-provenance/1"},
    )
    request = {
        "schema_version": science_runner.RDKIT_PREFLIGHT_REQUEST_SCHEMA,
        "format": "smiles",
        "source": "O",
    }

    result = science_runner._with_runtime_provenance(
        {"schema_version": "example/1", "status": "completed"},
        operation="rdkit-preflight",
        request=request,
    )

    assert result["runtime_provenance"] == {
        "schema_version": "chemtools.companion-runtime-provenance/1",
        "runner_operation": "rdkit-preflight",
        "request": {
            "schema_version": "chemtools.rdkit-preflight-request/1",
            "sha256": "ab6ed944cfd15d24477b35c7879672c359cc82d29031627fdf8d508ad0c789e0",
        },
    }


def test_pyscf_request_rejects_restricted_open_shell_method():
    with pytest.raises(ValueError, match="restricted methods"):
        science_runner._pyscf_request(_pyscf_request(multiplicity=2))


def test_pyscf_request_rejects_untyped_extension_field():
    request = _pyscf_request()
    request["python"] = "import os"

    with pytest.raises(ValueError, match="unsupported or missing"):
        science_runner._pyscf_request(request)


def test_science_runner_rejects_nonfinite_json_before_hashing_provenance(
    monkeypatch,
):
    from io import BytesIO

    monkeypatch.setattr(
        science_runner.sys,
        "stdin",
        SimpleNamespace(buffer=BytesIO(b'{"value": NaN}')),
    )

    with pytest.raises(ValueError, match="valid JSON"):
        science_runner._read_request()


def test_pyscf_request_accepts_a_bounded_density_cube_request():
    parsed = science_runner._pyscf_request(
        _pyscf_request(density_cube_grid_points=80)
    )

    assert parsed["density_cube_grid_points"] == 80


def test_pyscf_request_rejects_an_unbounded_density_cube_request():
    with pytest.raises(ValueError, match="between 20 and 120"):
        science_runner._pyscf_request(
            _pyscf_request(density_cube_grid_points=121)
        )


def test_pyscf_request_accepts_bounded_selected_orbital_cubes():
    parsed = science_runner._pyscf_request(_pyscf_request(
        orbital_cube_grid_points=80,
        orbital_cube_requests=[
            {"spin": "restricted", "orbital_index": 0},
            {"spin": "restricted", "orbital_index": 1},
        ],
    ))

    assert parsed["orbital_cube_grid_points"] == 80
    assert parsed["orbital_cube_requests"] == [
        {"spin": "restricted", "orbital_index": 0},
        {"spin": "restricted", "orbital_index": 1},
    ]


def test_pyscf_request_rejects_invalid_selected_orbital_cubes():
    with pytest.raises(ValueError, match="supplied together"):
        science_runner._pyscf_request(_pyscf_request(
            orbital_cube_grid_points=80,
        ))
    with pytest.raises(ValueError, match="restricted"):
        science_runner._pyscf_request(_pyscf_request(
            orbital_cube_grid_points=80,
            orbital_cube_requests=[{"spin": "alpha", "orbital_index": 0}],
        ))
    with pytest.raises(ValueError, match="duplicate"):
        science_runner._pyscf_request(_pyscf_request(
            orbital_cube_grid_points=80,
            orbital_cube_requests=[
                {"spin": "restricted", "orbital_index": 0},
                {"spin": "restricted", "orbital_index": 0},
            ],
        ))


def test_rdkit_request_rejects_unknown_format():
    with pytest.raises(ValueError, match="smiles or molblock"):
        science_runner._rdkit_request({
            "schema_version": science_runner.RDKIT_PREFLIGHT_REQUEST_SCHEMA,
            "format": "xyz",
            "source": "1\nH\nH 0 0 0\n",
        })


def test_openbabel_request_accepts_only_bounded_declared_formats():
    parsed = science_runner._openbabel_request({
        "schema_version": science_runner.OPENBABEL_CONVERSION_REQUEST_SCHEMA,
        "format": "smiles",
        "source": "C[C@H](O)F",
        "output_format": "molblock",
    })

    assert parsed == ("smiles", "C[C@H](O)F", "molblock")

    with pytest.raises(ValueError, match="output_format"):
        science_runner._openbabel_request({
            "schema_version": science_runner.OPENBABEL_CONVERSION_REQUEST_SCHEMA,
            "format": "smiles",
            "source": "O",
            "output_format": "xyz",
        })


def test_openbabel_invalid_request_returns_a_schema_bound_refusal():
    assert science_runner.openbabel_convert({
        "schema_version": science_runner.OPENBABEL_CONVERSION_REQUEST_SCHEMA,
        "format": "xyz",
        "source": "1\nH\nH 0 0 0\n",
        "output_format": "smiles",
    }) == {
        "schema_version": science_runner.OPENBABEL_CONVERSION_RESULT_SCHEMA,
        "status": "invalid_request",
        "message": "Open Babel format must be smiles or molblock",
    }


def test_bse_render_preserves_explicit_text_and_version(monkeypatch):
    calls = []

    class Bse:
        __version__ = "0.12"

        @staticmethod
        def get_basis(name, *, elements, fmt, header):
            calls.append((name, elements, fmt, header))
            return 'BASIS "ao basis" SPHERICAL PRINT\nO S\nEND\n'

    monkeypatch.setitem(sys.modules, "basis_set_exchange", Bse)

    result = science_runner.bse_render({
        "schema_version": science_runner.BSE_RENDER_REQUEST_SCHEMA,
        "basis": "def2-SVP",
        "elements": ["O"],
        "program_format": "nwchem",
    })

    assert calls == [("def2-SVP", ["O"], "nwchem", True)]
    assert result["status"] == "completed"
    assert result["basis"]["text"] == 'BASIS "ao basis" SPHERICAL PRINT\nO S\nEND\n'
    assert result["provenance"] == {"basis_set_exchange_version": "0.12"}


def test_bse_render_rejects_unsupported_program_format():
    with pytest.raises(ValueError, match="program format"):
        science_runner._bse_render_request({
            "schema_version": science_runner.BSE_RENDER_REQUEST_SCHEMA,
            "basis": "def2-SVP",
            "elements": ["O"],
            "program_format": "grasp",
        })


def test_orbitron_periodic_request_and_summary_are_bounded(tmp_path):
    source = tmp_path / "vasprun.xml"
    source.write_text("<modeling />", encoding="utf-8")
    request = {
        "schema_version": science_runner.ORBITRON_PERIODIC_REQUEST_SCHEMA,
        "path": str(source),
    }

    assert science_runner._orbitron_periodic_request(request) == source.resolve()
    summary = science_runner._periodic_electronic_structure_summary({
        "fermi_energy_ev": 1.2,
        "total_magnetization_bohr": None,
        "band_gap": {"value_ev": 7.0, "is_direct": False},
        "band_structure": {
            "sampling": "path",
            "spin_channels": ["total"],
            "kpoints_fractional": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            "eigenvalues_ev": [[[-5.0, 3.0], [-4.0, 4.0]]],
            "kpoint_labels": ["Γ", "X"],
            "segments": [],
        },
        "density_of_states": {
            "spin_channels": ["total"],
            "densities": [[0.1, 0.2, 0.3]],
            "energies_ev": [-5.0, 0.0, 5.0],
            "integrated": [[0.0, 0.1, 0.3]],
        },
    })

    assert summary == {
        "fermi_energy_ev": 1.2,
        "total_magnetization_bohr": None,
        "band_gap": {"status": "available", "value_ev": 7.0, "is_direct": False},
        "band_structure": {
            "status": "available",
            "sampling": "path",
            "spin_channels": ["total"],
            "kpoint_count": 2,
            "band_count_per_spin": [2],
            "label_count": 2,
            "segment_count": 0,
        },
        "density_of_states": {
            "status": "available",
            "spin_channels": ["total"],
            "energy_point_count": 3,
            "energy_min_ev": -5.0,
            "energy_max_ev": 5.0,
            "integrated_available": True,
        },
        "projected_data": "omitted",
    }


def test_orbitron_structure_identity_summary_preserves_dative_evidence(tmp_path):
    source = tmp_path / "zncl2.xyz"
    source.write_text("3\nZnCl2\nZn 0 0 0\nCl 2.3 0 0\nCl -2.3 0 0\n", encoding="utf-8")
    request = {
        "schema_version": science_runner.ORBITRON_STRUCTURE_IDENTITY_REQUEST_SCHEMA,
        "path": str(source),
    }

    assert science_runner._orbitron_source_request(
        request,
        schema=science_runner.ORBITRON_STRUCTURE_IDENTITY_REQUEST_SCHEMA,
        operation="structure-identity",
    ) == source.resolve()

    class Scene:
        def atom_count(self):
            return 3

        def bond_count(self):
            return 2

        def bonds(self):
            return [
                {"a": 1, "b": 2, "order": "Dative"},
                {"a": 1, "b": 3, "order": "Dative"},
            ]

    class Service:
        formula = staticmethod(lambda scene: "Cl2Zn")
        inchi = staticmethod(lambda scene: "InChI=1S/2ClH.Zn/h2*1H;/q;;+2/p-2")
        inchikey = staticmethod(lambda scene: "JIAARYAFYJHUJI-UHFFFAOYSA-L")
        smiles = staticmethod(lambda scene: "[Cl][Zn][Cl]")

    assert science_runner._orbitron_structure_identity_summary(Service(), Scene()) == {
        "atom_count": 3,
        "bond_count": 2,
        "bond_order_counts": {"Dative": 2},
        "identifiers": {
            "formula": {"status": "available", "value": "Cl2Zn"},
            "inchi": {
                "status": "available",
                "value": "InChI=1S/2ClH.Zn/h2*1H;/q;;+2/p-2",
            },
            "inchikey": {
                "status": "available",
                "value": "JIAARYAFYJHUJI-UHFFFAOYSA-L",
            },
            "smiles": {"status": "available", "value": "[Cl][Zn][Cl]"},
        },
    }

    with pytest.raises(ValueError, match="128 KiB"):
        science_runner._openbabel_request({
            "schema_version": science_runner.OPENBABEL_CONVERSION_REQUEST_SCHEMA,
            "format": "smiles",
            "source": "C" * (128 * 1024 + 1),
            "output_format": "molblock",
        })


def test_orbitron_nbo_summary_preserves_bounded_bonding_evidence():
    summary = science_runner._orbitron_nbo_summary({
        "orbitals": [
            {
                "number": 1,
                "label": "BD(1) 1-2",
                "orbital_type": "BD",
                "occupancy": 1.99803,
                "atoms": [
                    {
                        "atom_id": 2,
                        "atom_index": 1,
                        "element": "O",
                        "weight": 0.8107,
                        "is_positive": True,
                    },
                    {
                        "atom_id": 1,
                        "atom_index": 0,
                        "element": "U",
                        "weight": 0.1893,
                        "is_positive": True,
                    },
                ],
            },
            {
                "number": 2,
                "label": "BD*(1) 1-2",
                "orbital_type": "BD*",
                "occupancy": 0.00153,
                "atoms": [
                    {
                        "atom_id": 1,
                        "atom_index": 0,
                        "element": "U",
                        "weight": 0.8107,
                        "is_positive": True,
                    },
                    {
                        "atom_id": 2,
                        "atom_index": 1,
                        "element": "O",
                        "weight": 0.1893,
                        "is_positive": False,
                    },
                ],
            },
        ],
        "per_atom": {"0": [{}, {}], "1": [{}, {}]},
    }, 2)

    assert summary == {
        "orbital_count": 2,
        "orbital_type_counts": {"BD": 1, "BD*": 1},
        "occupancy_range": {"minimum": 0.00153, "maximum": 1.99803},
        "per_atom_entry_counts": [
            {"atom_index": 0, "entry_count": 2},
            {"atom_index": 1, "entry_count": 2},
        ],
        "bonding_orbital_samples": [
            {
                "number": 1,
                "label": "BD(1) 1-2",
                "orbital_type": "BD",
                "occupancy": 1.99803,
                "atoms": [
                    {
                        "atom_index": 1,
                        "element": "O",
                        "weight": 0.8107,
                        "is_positive": True,
                    },
                    {
                        "atom_index": 0,
                        "element": "U",
                        "weight": 0.1893,
                        "is_positive": True,
                    },
                ],
            },
            {
                "number": 2,
                "label": "BD*(1) 1-2",
                "orbital_type": "BD*",
                "occupancy": 0.00153,
                "atoms": [
                    {
                        "atom_index": 0,
                        "element": "U",
                        "weight": 0.8107,
                        "is_positive": True,
                    },
                    {
                        "atom_index": 1,
                        "element": "O",
                        "weight": 0.1893,
                        "is_positive": False,
                    },
                ],
            },
        ],
    }


def test_orbitron_nbo_reports_missing_data_without_a_parser_refusal(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "no-nbo.xyz"
    source.write_text("1\nH\nH 0 0 0\n", encoding="utf-8")

    class Scene:
        def atom_count(self):
            return 1

    class Service:
        def analyze_nbo(self, scene, top_atoms):
            assert top_atoms == 5
            return None

    class Runtime:
        __version__ = "test"
        Orbitron = Service

        @staticmethod
        def load(path):
            assert path == str(source.resolve())
            return Scene()

    monkeypatch.setitem(sys.modules, "orbitron", Runtime)

    response = science_runner.orbitron_nbo({
        "schema_version": science_runner.ORBITRON_NBO_REQUEST_SCHEMA,
        "path": str(source),
    })

    assert response["schema_version"] == science_runner.ORBITRON_NBO_RESULT_SCHEMA
    assert response["status"] == "unavailable_data"
    assert response["message"] == "Orbitron found no Natural Bond Orbital data"


def test_science_runner_remains_standalone_from_a_working_directory(tmp_path):
    completed = subprocess.run(
        [sys.executable, science_runner.__file__, "unsupported"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.startswith("CHEMTOOLS_SCIENCE_RESULT=")

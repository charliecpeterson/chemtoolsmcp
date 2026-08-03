"""Regression coverage for fixed QMCPACK HDF5 layout classification."""

from __future__ import annotations

from typing import Any

from chemtools.mcp.dispatch import dispatch_tool
from chemtools.mcp.tools import qmcpack as qmcpack_tools
from chemtools.programs.qmcpack import hdf5


class _Dataset:
    def __init__(self, value: Any, shape: tuple[int, ...] = ()) -> None:
        self.value = value
        self.shape = shape
        self.nbytes = 8

    def __getitem__(self, key: object) -> Any:
        assert key == ()
        return self.value


class _Document:
    def __init__(self, entries: dict[str, _Dataset], roots: set[str]) -> None:
        self.entries = entries
        self.roots = roots

    def __contains__(self, path: object) -> bool:
        return isinstance(path, str) and (path in self.entries or path in self.roots)

    def get(self, path: str) -> _Dataset | None:
        return self.entries.get(path)

    def keys(self) -> list[str]:
        return sorted(self.roots)


def test_classifies_the_fixed_pwscf_wavefunction_layout():
    document = _Document({
        "application/code": _Dataset(b"pw2qmcpack", (1,)),
        "application/version": _Dataset([7, 5, 0], (3,)),
        "format": _Dataset(b"ES-HDF", (1,)),
        "version": _Dataset([0, 4, 1], (3,)),
        "atoms/number_of_atoms": _Dataset([1], (1,)),
        "atoms/number_of_species": _Dataset([1], (1,)),
        "atoms/positions": _Dataset([], (1, 3)),
        "atoms/species_ids": _Dataset([0], (1,)),
        "atoms/species_0/name": _Dataset(b"O", (1,)),
        "atoms/species_0/atomic_number": _Dataset([8], (1,)),
        "atoms/species_0/valence_charge": _Dataset([6], (1,)),
        "electrons/number_of_electrons": _Dataset([3, 3], (2,)),
        "electrons/number_of_spins": _Dataset([2], (1,)),
        "electrons/number_of_kpoints": _Dataset([1], (1,)),
        "supercell/primitive_vectors": _Dataset([], (3, 3)),
    }, {
        "application", "atoms", "electrons", "supercell", "format", "version",
        "atoms/species_0", "electrons/density",
    })

    inspection = hdf5._recognized_layout(document)

    assert inspection == {
        "artifact_kind": "pwscf_wavefunction",
        "message": (
            "Recognized the QMCPACK electronic-structure HDF5 layout written by "
            "a converter such as pw2qmcpack."
        ),
        "wavefunction": {
            "format": "ES-HDF",
            "version": [0, 4, 1],
            "application": {"code": "pw2qmcpack", "version": [7, 5, 0]},
            "atoms": {
                "count": 1,
                "species_count": 1,
                "positions_shape": [1, 3],
                "species": [{
                    "index": 0,
                    "name": "O",
                    "atomic_number": 8,
                    "valence_charge": 6,
                    "atom_count": 1,
                }],
            },
            "electrons": {
                "spin_populations": [3, 3],
                "spin_count": 2,
                "kpoint_count": 1,
                "density_metadata_present": True,
            },
            "supercell": {"primitive_vectors_shape": [3, 3]},
        },
    }


def test_classifies_variational_sidecars_without_reading_parameter_values():
    document = _Document({
        "name_value_lists/parameter_names": _Dataset([], (24,)),
        "name_value_lists/parameter_values": _Dataset([], (24,)),
        "timestamp": _Dataset(b"2026-08-02 12:00:00", (1,)),
        "version": _Dataset([1, 0, 0], (3,)),
    }, {"name_value_lists", "timestamp", "version"})

    inspection = hdf5._recognized_layout(document)

    assert inspection["artifact_kind"] == "variational_parameters"
    assert inspection["variational_parameters"] == {
        "version": [1, 0, 0],
        "timestamp": "2026-08-02 12:00:00",
        "parameter_name_count": 24,
        "parameter_value_count": 24,
        "name_value_counts_match": True,
    }


def test_qmcpack_hdf5_tool_uses_only_the_fixed_companion_request(monkeypatch):
    class Client:
        def qmcpack_hdf5_inspect(self, request):
            assert request == {
                "schema_version": "chemtools.qmcpack-hdf5-inspection-request/1",
                "path": "/work/O.pwscf.h5",
            }
            return {"status": "recognized", "artifact_kind": "pwscf_wavefunction"}

    monkeypatch.setattr(qmcpack_tools, "ScienceRuntimeClient", Client)

    assert dispatch_tool("inspect_qmcpack_hdf5", {
        "hdf5_file": "/work/O.pwscf.h5",
    }) == {"status": "recognized", "artifact_kind": "pwscf_wavefunction"}

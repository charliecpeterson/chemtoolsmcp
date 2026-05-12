"""DIRAC BinaryReader sub-protocol implementation."""

from __future__ import annotations

from typing import Any

from chemtools.programs.dirac.binary import (
    read_metadata,
    read_geometry,
    read_total_energy,
    read_orbital_summary,
    read_mo_coefficients,
    read_aobasis_info,
    H5PY_AVAILABLE,
)


_SUPPORTED_KINDS = (
    "metadata", "geometry", "energy",
    "orbitals", "mo_coefficients", "aobasis",
)


class _DiracBinaryReader:
    """Adapts the HDF5 reader to the chemtools BinaryReader protocol."""

    def supported_kinds(self) -> list[str]:
        return list(_SUPPORTED_KINDS)

    def parse(self, path: str, kind: str) -> dict[str, Any]:
        k = kind.lower()
        if k == "metadata":
            return read_metadata(path)
        if k == "geometry":
            return read_geometry(path)
        if k == "energy":
            return {"path": path, "total_energy_hartree": read_total_energy(path)}
        if k == "orbitals":
            return {"path": path, "orbitals": read_orbital_summary(path)}
        if k == "mo_coefficients":
            return read_mo_coefficients(path)
        if k == "aobasis":
            return read_aobasis_info(path)
        raise ValueError(
            f"DIRAC binary reader does not support kind={kind!r}; "
            f"supported: {', '.join(_SUPPORTED_KINDS)}"
        )

    def write(self, path: str, kind: str, data: dict[str, Any]) -> None:
        # Writes are post-DA. Reordered MO checkpoints will land here.
        raise NotImplementedError(
            "DIRAC HDF5 writing not implemented yet — phase DE will add "
            "MO-reordered checkpoints."
        )


DIRAC_BINARY = _DiracBinaryReader()

__all__ = ["DIRAC_BINARY"]

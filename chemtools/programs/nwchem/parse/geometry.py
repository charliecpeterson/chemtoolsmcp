"""Parse coordinate tables printed by NWChem outputs.

The scanner accepts an iterable stream so callers can retain the first or
last complete geometry without loading every optimization frame.
"""

from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any, Literal

from chemtools.core.units import ANGSTROM_PER_BOHR
from chemtools.programs.nwchem.input.basis_library import (
    normalize_element_symbol,
)

_GEOMETRY_NAME_RE = re.compile(r'\bGeometry\s+"([^"]+)"\s*->')
GEOMETRY_DISTANCE_TOLERANCE_ANGSTROM = 1.0e-5


class OutputGeometryScanner:
    def __init__(self) -> None:
        self.first_geometry: dict[str, Any] | None = None
        self.last_geometry: dict[str, Any] | None = None
        self.first_by_name: dict[str, dict[str, Any]] = {}
        self.last_by_name: dict[str, dict[str, Any]] = {}
        self._in_geometry = False
        self._source_units = "angstrom"
        self._atoms: list[dict[str, Any]] = []
        self._pending_name: str | None = None
        self._name: str | None = None

    def feed(self, line: str) -> dict[str, Any] | None:
        if match := _GEOMETRY_NAME_RE.search(line):
            self._pending_name = match.group(1).strip()
        if "Output coordinates in angstroms" in line:
            self._start("angstrom")
            return None
        if "Output coordinates in a.u." in line:
            self._start("bohr")
            return None
        if not self._in_geometry:
            return None

        stripped = line.strip()
        if "Atomic Mass" in line:
            return self._finish()
        if (
            not stripped
            or stripped.startswith("----")
            or (
                "No." in line
                and "Tag" in line
                and "Charge" in line
            )
        ):
            return None

        parts = line.split()
        if len(parts) < 6:
            return None
        try:
            int(parts[0])
            x, y, z = (
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
        except ValueError:
            return None
        if self._source_units == "bohr":
            x *= ANGSTROM_PER_BOHR
            y *= ANGSTROM_PER_BOHR
            z *= ANGSTROM_PER_BOHR
        try:
            element = normalize_element_symbol(parts[1])
        except ValueError:
            element = None
        self._atoms.append({
            "label": parts[1],
            "element": element,
            "x": x,
            "y": y,
            "z": z,
        })
        return None

    def _start(self, source_units: str) -> None:
        self._in_geometry = True
        self._source_units = source_units
        self._atoms = []
        self._name = self._pending_name
        self._pending_name = None

    def _finish(self) -> dict[str, Any] | None:
        self._in_geometry = False
        if not self._atoms:
            return None
        geometry = {
            "atoms": self._atoms,
            "atom_count": len(self._atoms),
            "source_units": self._source_units,
            "units": "angstrom",
        }
        if self._name:
            geometry["name"] = self._name
            self.first_by_name.setdefault(self._name, geometry)
            self.last_by_name[self._name] = geometry
        if self.first_geometry is None:
            self.first_geometry = geometry
        self.last_geometry = geometry
        self._atoms = []
        self._name = None
        return geometry


def extract_output_geometry(
    lines: Iterable[str],
    *,
    which: Literal["first", "last"] = "last",
) -> dict[str, Any] | None:
    scanner = OutputGeometryScanner()
    for line in lines:
        scanner.feed(line)
        if which == "first" and scanner.first_geometry is not None:
            return scanner.first_geometry
    return (
        scanner.first_geometry
        if which == "first"
        else scanner.last_geometry
    )


__all__ = [
    "GEOMETRY_DISTANCE_TOLERANCE_ANGSTROM",
    "OutputGeometryScanner",
    "extract_output_geometry",
]

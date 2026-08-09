"""Parse the simple keywords and inline XYZ form used by ORCA inputs."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any


_ELEMENT_BASIS_RE = re.compile(
    r'^NewGTO\s+([A-Za-z]{1,2})\s+"([^"]+)"\s+end$',
    re.IGNORECASE,
)
_MOINP_RE = re.compile(r'^%moinp\s+"([^"]+)"$', re.IGNORECASE)
_GUESS_MODE_RE = re.compile(r"^GuessMode\s+(CMatrix|FMatrix)$", re.IGNORECASE)


def parse_orca_input(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    keywords = []
    block_names = []
    element_basis_sets = []
    moinp = None
    guess_mode = None
    atoms = []
    charge = None
    multiplicity = None
    coordinate_file = None
    coordinate_format = None
    in_xyz = False

    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("!"):
            keywords.extend(stripped[1:].split())
            continue
        if match := _MOINP_RE.match(stripped):
            block_names.append("moinp")
            moinp = match.group(1)
            continue
        if stripped.startswith("%"):
            block_names.append(stripped[1:].split()[0].casefold())
            continue
        if match := _ELEMENT_BASIS_RE.match(stripped):
            element_basis_sets.append({
                "element": match.group(1),
                "basis": match.group(2),
                "line": number,
            })
            continue
        if match := _GUESS_MODE_RE.match(stripped):
            guess_mode = match.group(1)
            continue
        fields = stripped.split()
        if (
            len(fields) >= 5
            and fields[0] == "*"
            and fields[1].casefold() in {"xyzfile", "pdbfile"}
        ):
            charge = int(fields[2])
            multiplicity = int(fields[3])
            coordinate_format = fields[1].casefold().removesuffix("file")
            coordinate_file = fields[4]
            continue
        if len(fields) >= 4 and fields[0] == "*" and fields[1].casefold() == "xyz":
            charge = int(fields[2])
            multiplicity = int(fields[3])
            in_xyz = True
            continue
        if in_xyz and stripped == "*":
            in_xyz = False
            continue
        if in_xyz:
            if len(fields) < 4:
                raise ValueError(f"invalid inline XYZ coordinate at line {number}")
            atoms.append({
                "element": fields[0],
                "x": float(fields[1]),
                "y": float(fields[2]),
                "z": float(fields[3]),
                "line": number,
            })

    return {
        "program": "orca",
        "file": str(source),
        "simple_keywords": keywords,
        "block_names": block_names,
        "element_basis_sets": element_basis_sets,
        "moinp": moinp,
        "guess_mode": guess_mode,
        "charge": charge,
        "multiplicity": multiplicity,
        "coordinate_file": coordinate_file,
        "coordinate_format": coordinate_format,
        "geometry_units": "angstrom" if atoms else None,
        "atoms": atoms,
    }


__all__ = ["parse_orca_input"]

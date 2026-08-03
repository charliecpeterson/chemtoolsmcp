"""Bounded inspection of GRASP2018 radial-wavefunction files.

The record layout follows GRASP2018's ``rwfntotxt.f90`` and
``rwfnrelabel.f90`` sources. Radial arrays are validated but not returned.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import stat
import struct
import tempfile
from typing import BinaryIO

from chemtools.programs.grasp.binary._fortran_records import (
    detect_record_byte_order,
    read_record,
)


GRASP_RWFN_INSPECTION_SCHEMA = "chemtools.grasp-radial-wfn-inspection/1"
GRASP_RWFN_MERGE_SCHEMA = "chemtools.grasp-radial-wfn-merge/1"
MAX_GRASP_RWFN_BYTES = 64 * 1024 * 1024
MAX_GRASP_ORBITALS = 127
MAX_GRASP_GRID_POINTS = 10_000
MAX_GRASP_RWFN_DONORS = 16
_MAX_RECORD_BYTES = 8 * (1 + 2 * MAX_GRASP_GRID_POINTS)
_MAGIC = b"G92RWF"
_ORBITAL_HEADER_BYTES = 20
_ANGULAR_LABELS = "spdfghiklm"


@dataclass(frozen=True)
class _OrbitalRecord:
    n: int
    kappa: int
    energy_au: float
    n_points: int
    a0: float
    radius_minimum: float
    radius_maximum: float
    header: bytes
    components: bytes
    grid: bytes

    @property
    def identity(self) -> tuple[int, int]:
        return self.n, self.kappa

    @property
    def label(self) -> str:
        return _orbital_label(self.n, self.kappa)

    def summary(self, index: int) -> dict[str, object]:
        return {
            "index": index,
            "n": self.n,
            "kappa": self.kappa,
            "label": self.label,
            "energy_au": self.energy_au,
            "n_points": self.n_points,
            "a0": self.a0,
            "radial_grid_au": {
                "minimum": self.radius_minimum,
                "maximum": self.radius_maximum,
            },
        }


@dataclass(frozen=True)
class _RadialWavefunction:
    source: Path
    endian: str
    size_bytes: int
    sha256: str
    orbitals: tuple[_OrbitalRecord, ...]


def inspect_grasp_radial_wfn(path: str | Path) -> dict[str, object]:
    return _inspection_payload(_load_radial_wfn(path))


def merge_grasp_radial_wfns(
    donor_paths: list[str | Path] | tuple[str | Path, ...],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    if isinstance(donor_paths, (str, bytes)):
        raise TypeError("donor_paths must be a list of file paths")
    if not 2 <= len(donor_paths) <= MAX_GRASP_RWFN_DONORS:
        raise ValueError(
            "donor_paths must contain between 2 and "
            f"{MAX_GRASP_RWFN_DONORS} files"
        )
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")

    donors = tuple(_load_radial_wfn(path) for path in donor_paths)
    resolved_paths = tuple(donor.source for donor in donors)
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("donor_paths must resolve to distinct files")
    endian = donors[0].endian
    if any(donor.endian != endian for donor in donors[1:]):
        raise ValueError("GRASP radial-wavefunction donors use mixed byte order")

    destination = Path(output_path).expanduser().resolve()
    if destination in set(resolved_paths):
        raise ValueError("output_path must not replace a donor file")

    selected: list[_OrbitalRecord] = []
    owners: dict[tuple[int, int], Path] = {}
    donor_summaries: list[dict[str, object]] = []
    duplicate_count = 0
    for donor_index, donor in enumerate(donors):
        contributed: list[str] = []
        skipped: list[dict[str, str]] = []
        for orbital in donor.orbitals:
            owner = owners.get(orbital.identity)
            if owner is not None:
                duplicate_count += 1
                skipped.append(
                    {
                        "label": orbital.label,
                        "kept_from": str(owner),
                    }
                )
                continue
            if len(selected) == MAX_GRASP_ORBITALS:
                raise ValueError(
                    f"merged file would exceed {MAX_GRASP_ORBITALS} orbitals"
                )
            owners[orbital.identity] = donor.source
            selected.append(orbital)
            contributed.append(orbital.label)
        if donor_index > 0 and not contributed:
            raise ValueError(
                f"donor {donor.source} contributes no new orbitals"
            )
        donor_summaries.append(
            {
                "precedence": donor_index + 1,
                "path": str(donor.source),
                "sha256": donor.sha256,
                "orbital_count": len(donor.orbitals),
                "contributed_orbitals": contributed,
                "skipped_duplicates": skipped,
            }
        )

    merged_size = _framed_size(len(_MAGIC)) + sum(
        _framed_size(len(payload))
        for orbital in selected
        for payload in (orbital.header, orbital.components, orbital.grid)
    )
    if merged_size > MAX_GRASP_RWFN_BYTES:
        raise ValueError(
            f"merged file would exceed {MAX_GRASP_RWFN_BYTES} bytes"
        )
    _write_atomic(destination, endian, tuple(selected), overwrite=overwrite)
    output = inspect_grasp_radial_wfn(destination)
    return {
        "schema_version": GRASP_RWFN_MERGE_SCHEMA,
        "policy": "first_donor_wins_duplicate_n_kappa",
        "donor_count": len(donors),
        "duplicate_count": duplicate_count,
        "donors": donor_summaries,
        "output": output,
    }


def _load_radial_wfn(path: str | Path) -> _RadialWavefunction:
    source = Path(path).expanduser().resolve()
    try:
        stream = source.open("rb")
    except OSError as error:
        raise ValueError(
            f"cannot open GRASP radial-wavefunction file {source}: {error}"
        ) from error
    with stream:
        initial_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(initial_stat.st_mode):
            raise ValueError(
                f"GRASP radial-wavefunction path is not a regular file: {source}"
            )
        size_bytes = initial_stat.st_size
        if size_bytes > MAX_GRASP_RWFN_BYTES:
            raise ValueError(
                "GRASP radial-wavefunction file exceeds "
                f"{MAX_GRASP_RWFN_BYTES} bytes"
            )
        endian = _read_header(stream)
        orbitals = _read_orbitals(stream, endian)
        stream.seek(0)
        digest = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        final_stat = os.fstat(stream.fileno())
        if (
            final_stat.st_size != initial_stat.st_size
            or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
        ):
            raise ValueError(
                "GRASP radial-wavefunction file changed during inspection"
            )
    return _RadialWavefunction(
        source=source,
        endian=endian,
        size_bytes=size_bytes,
        sha256=digest.hexdigest(),
        orbitals=orbitals,
    )


def _inspection_payload(document: _RadialWavefunction) -> dict[str, object]:
    return {
        "schema_version": GRASP_RWFN_INSPECTION_SCHEMA,
        "path": str(document.source),
        "format": {
            "magic": _MAGIC.decode("ascii"),
            "byte_order": "little" if document.endian == "<" else "big",
            "record_marker_bytes": 4,
            "source_contract": ["rwfntotxt.f90", "rwfnrelabel.f90"],
        },
        "file": {
            "size_bytes": document.size_bytes,
            "sha256": document.sha256,
        },
        "orbital_count": len(document.orbitals),
        "orbitals": [
            orbital.summary(index)
            for index, orbital in enumerate(document.orbitals, start=1)
        ],
        "checks": {
            "complete_record_triples": True,
            "unique_n_kappa": True,
            "finite_radial_values": True,
            "strictly_increasing_radial_grids": True,
        },
    }


def _read_header(stream: BinaryIO) -> str:
    marker = stream.read(4)
    if len(marker) != 4:
        raise ValueError("GRASP radial-wavefunction file has no complete header")
    endian = detect_record_byte_order(marker, len(_MAGIC))
    if endian is None:
        raise ValueError("GRASP radial-wavefunction header record has invalid size")
    payload = stream.read(len(_MAGIC))
    trailer = stream.read(4)
    if payload != _MAGIC:
        raise ValueError("GRASP radial-wavefunction magic must be 'G92RWF'")
    if len(trailer) != 4 or struct.unpack(f"{endian}i", trailer)[0] != len(payload):
        raise ValueError("GRASP radial-wavefunction header markers do not match")
    return endian


def _read_orbitals(
    stream: BinaryIO,
    endian: str,
) -> tuple[_OrbitalRecord, ...]:
    orbitals: list[_OrbitalRecord] = []
    identities: set[tuple[int, int]] = set()
    while True:
        header = _read_record(
            stream,
            endian,
            "orbital header",
            allow_eof=True,
        )
        if header is None:
            break
        if len(orbitals) == MAX_GRASP_ORBITALS:
            raise ValueError(
                f"GRASP radial-wavefunction file exceeds {MAX_GRASP_ORBITALS} orbitals"
            )
        if len(header) != _ORBITAL_HEADER_BYTES:
            raise ValueError(
                "GRASP orbital header must contain two int32 values, one "
                "float64 value, and one int32 value"
            )
        n, kappa, energy_au, n_points = struct.unpack(f"{endian}iidi", header)
        _validate_identity(n, kappa)
        if not math.isfinite(energy_au):
            raise ValueError(f"orbital {(n, kappa)} has non-finite energy")
        if not 1 <= n_points <= MAX_GRASP_GRID_POINTS:
            raise ValueError(
                f"orbital {(n, kappa)} has invalid grid size {n_points}"
            )

        components = _required_record(
            stream,
            endian,
            f"orbital {(n, kappa)} radial components",
        )
        expected_component_bytes = 8 * (1 + 2 * n_points)
        if len(components) != expected_component_bytes:
            raise ValueError(
                f"orbital {(n, kappa)} radial components contain "
                f"{len(components)} bytes; expected {expected_component_bytes}"
            )
        component_values = struct.unpack(
            f"{endian}{1 + 2 * n_points}d",
            components,
        )
        if not all(math.isfinite(value) for value in component_values):
            raise ValueError(f"orbital {(n, kappa)} has non-finite radial values")

        grid = _required_record(
            stream,
            endian,
            f"orbital {(n, kappa)} radial grid",
        )
        expected_grid_bytes = 8 * n_points
        if len(grid) != expected_grid_bytes:
            raise ValueError(
                f"orbital {(n, kappa)} radial grid contains {len(grid)} bytes; "
                f"expected {expected_grid_bytes}"
            )
        radii = struct.unpack(f"{endian}{n_points}d", grid)
        if not all(math.isfinite(radius) for radius in radii):
            raise ValueError(f"orbital {(n, kappa)} has non-finite grid values")
        if radii[0] < 0 or any(
            current <= previous
            for previous, current in zip(radii, radii[1:])
        ):
            raise ValueError(
                f"orbital {(n, kappa)} radial grid must be nonnegative and "
                "strictly increasing"
            )
        identity = (n, kappa)
        if identity in identities:
            raise ValueError(f"duplicate GRASP orbital identity {identity}")
        identities.add(identity)
        orbitals.append(
            _OrbitalRecord(
                n=n,
                kappa=kappa,
                energy_au=energy_au,
                n_points=n_points,
                a0=component_values[0],
                radius_minimum=radii[0],
                radius_maximum=radii[-1],
                header=header,
                components=components,
                grid=grid,
            )
        )
    if not orbitals:
        raise ValueError("GRASP radial-wavefunction file contains no orbitals")
    return tuple(orbitals)


def _write_atomic(
    destination: Path,
    endian: str,
    orbitals: tuple[_OrbitalRecord, ...],
    *,
    overwrite: bool,
) -> None:
    parent = destination.parent
    if not parent.is_dir():
        raise ValueError(f"output directory does not exist: {parent}")
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"output path is not a regular file: {destination}")
        if not overwrite:
            raise ValueError(
                f"output path already exists; set overwrite=true: {destination}"
            )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            _write_record(stream, endian, _MAGIC)
            for orbital in orbitals:
                for payload in (
                    orbital.header,
                    orbital.components,
                    orbital.grid,
                ):
                    _write_record(stream, endian, payload)
            stream.flush()
            os.fsync(stream.fileno())
        inspect_grasp_radial_wfn(temporary_path)
        if overwrite:
            os.replace(temporary_path, destination)
        else:
            try:
                os.link(temporary_path, destination)
            except FileExistsError as error:
                raise ValueError(
                    f"output path already exists; set overwrite=true: {destination}"
                ) from error
            temporary_path.unlink()
        temporary_path = None
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_record(
    stream: BinaryIO,
    endian: str,
    payload: bytes,
) -> None:
    marker = struct.pack(f"{endian}i", len(payload))
    stream.write(marker)
    stream.write(payload)
    stream.write(marker)


def _framed_size(payload_size: int) -> int:
    return 4 + payload_size + 4


def _validate_identity(n: int, kappa: int) -> None:
    if n < 1:
        raise ValueError(f"GRASP orbital principal quantum number must be positive: {n}")
    if kappa == 0 or abs(kappa) > n:
        raise ValueError(f"invalid GRASP orbital identity {(n, kappa)}")
    angular_momentum = kappa if kappa > 0 else -kappa - 1
    if angular_momentum >= len(_ANGULAR_LABELS):
        raise ValueError(
            f"GRASP orbital {(n, kappa)} exceeds the supported angular labels"
        )


def _orbital_label(n: int, kappa: int) -> str:
    angular_momentum = kappa if kappa > 0 else -kappa - 1
    suffix = "-" if kappa > 0 else ""
    return f"{n}{_ANGULAR_LABELS[angular_momentum]}{suffix}"


def _required_record(
    stream: BinaryIO,
    endian: str,
    field: str,
) -> bytes:
    record = _read_record(stream, endian, field)
    if record is None:
        raise ValueError(f"missing {field} record")
    return record


def _read_record(
    stream: BinaryIO,
    endian: str,
    field: str,
    *,
    allow_eof: bool = False,
) -> bytes | None:
    return read_record(
        stream,
        endian,
        field,
        max_record_bytes=_MAX_RECORD_BYTES,
        allow_eof=allow_eof,
    )


__all__ = [
    "GRASP_RWFN_INSPECTION_SCHEMA",
    "GRASP_RWFN_MERGE_SCHEMA",
    "MAX_GRASP_GRID_POINTS",
    "MAX_GRASP_ORBITALS",
    "MAX_GRASP_RWFN_BYTES",
    "MAX_GRASP_RWFN_DONORS",
    "inspect_grasp_radial_wfn",
    "merge_grasp_radial_wfns",
]

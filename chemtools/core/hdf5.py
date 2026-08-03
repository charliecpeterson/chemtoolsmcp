"""Bounded HDF5 identification shared by artifact inspectors.

The probe recognizes HDF5 superblock signatures without decoding datasets.
"""

from __future__ import annotations

from pathlib import Path


HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def hdf5_signature_offset(path: str | Path, size_bytes: int) -> int | None:
    """Return the supported superblock offset, if an HDF5 signature is present."""
    source = Path(path)
    offset = 0
    with source.open("rb") as handle:
        while offset + len(HDF5_SIGNATURE) <= size_bytes:
            handle.seek(offset)
            if handle.read(len(HDF5_SIGNATURE)) == HDF5_SIGNATURE:
                return offset
            offset = 512 if offset == 0 else offset * 2
    return None


__all__ = ["HDF5_SIGNATURE", "hdf5_signature_offset"]

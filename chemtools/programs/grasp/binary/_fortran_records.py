"""Shared framing helpers for GRASP sequential-unformatted records."""

from __future__ import annotations

import struct
from typing import BinaryIO


def detect_record_byte_order(marker: bytes, expected_size: int) -> str | None:
    if len(marker) != 4:
        return None
    if struct.unpack("<i", marker)[0] == expected_size:
        return "<"
    if struct.unpack(">i", marker)[0] == expected_size:
        return ">"
    return None


def read_record(
    stream: BinaryIO,
    endian: str,
    field: str,
    *,
    max_record_bytes: int,
    allow_eof: bool = False,
) -> bytes | None:
    size = read_record_size(
        stream,
        endian,
        field,
        max_record_bytes=max_record_bytes,
        allow_eof=allow_eof,
    )
    if size is None:
        return None
    payload = stream.read(size)
    if len(payload) != size:
        raise ValueError(f"incomplete {field} record")
    finish_record(stream, endian, field, size)
    return payload


def read_record_size(
    stream: BinaryIO,
    endian: str,
    field: str,
    *,
    max_record_bytes: int,
    allow_eof: bool = False,
) -> int | None:
    marker = stream.read(4)
    if not marker and allow_eof:
        return None
    if len(marker) != 4:
        raise ValueError(f"incomplete {field} record marker")
    size = struct.unpack(f"{endian}i", marker)[0]
    if size < 0 or size > max_record_bytes:
        raise ValueError(f"{field} record has invalid size {size}")
    return size


def finish_record(
    stream: BinaryIO,
    endian: str,
    field: str,
    size: int,
) -> None:
    trailer = stream.read(4)
    if len(trailer) != 4:
        raise ValueError(f"incomplete {field} record")
    if struct.unpack(f"{endian}i", trailer)[0] != size:
        raise ValueError(f"{field} record markers do not match")


__all__ = [
    "detect_record_byte_order",
    "finish_record",
    "read_record",
    "read_record_size",
]

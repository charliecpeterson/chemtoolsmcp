"""Read bounded UTF-8 evidence from explicit text artifacts.

The caller chooses eligible files and owns the aggregate byte budget. This
module handles per-file positioning, decoding boundaries, and uncertainty.
"""

from __future__ import annotations

import codecs
from pathlib import Path
from typing import Any


RELATED_TEXT_LIMIT_BYTES = 16 * 1024
RELATED_TEXT_TOTAL_LIMIT_BYTES = 64 * 1024


def read_text_excerpt(
    path: Path,
    size_bytes: int,
    limit_bytes: int,
    *,
    tail_only: bool,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    if tail_only:
        return _read_tail_excerpt(path, size_bytes, limit_bytes)
    return _read_head_tail_excerpt(path, size_bytes, limit_bytes)


def _read_tail_excerpt(
    path: Path,
    size_bytes: int,
    limit_bytes: int,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    start = max(size_bytes - limit_bytes, 0)
    try:
        with path.open("rb") as handle:
            handle.seek(start)
            raw = handle.read(limit_bytes)
    except OSError as exc:
        return None, [{
            "code": "related_artifact_read_failed",
            "message": f"Could not read related stderr artifact {path}: {exc}",
            "impact": "Stderr content could not contribute diagnostic evidence.",
        }]

    text, decode_status, boundary_bytes_discarded = _decode_text_segment(
        raw,
        trim_leading_boundary=start > 0,
        trim_trailing_boundary=False,
    )

    truncated = start > 0
    uncertainty = []
    if truncated:
        uncertainty.append({
            "code": "related_artifact_text_truncated",
            "message": (
                f"Related stderr artifact exceeds "
                f"{limit_bytes} bytes: {path}"
            ),
            "impact": "Only the final bounded excerpt was inspected.",
        })
    if decode_status != "decoded":
        uncertainty.append({
            "code": "related_artifact_decode_replaced",
            "message": (
                f"Related stderr artifact is not valid UTF-8: {path}"
            ),
            "impact": (
                "Invalid byte sequences were replaced in the text excerpt."
            ),
        })
    return {
        "role": "stderr",
        "position": "tail",
        "limit_bytes": limit_bytes,
        "bytes_read": len(raw),
        "boundary_bytes_discarded": boundary_bytes_discarded,
        "truncated": truncated,
        "encoding": "utf-8",
        "decode_status": decode_status,
        "text": text,
    }, uncertainty


def _read_head_tail_excerpt(
    path: Path,
    size_bytes: int,
    limit_bytes: int,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    truncated = size_bytes > limit_bytes
    try:
        with path.open("rb") as handle:
            if not truncated:
                segments = [(
                    "whole",
                    0,
                    handle.read(limit_bytes),
                    False,
                    False,
                )]
            else:
                head_limit = limit_bytes // 2
                tail_limit = limit_bytes - head_limit
                head = handle.read(head_limit)
                tail_offset = size_bytes - tail_limit
                handle.seek(tail_offset)
                tail = handle.read(tail_limit)
                segments = [
                    ("head", 0, head, False, True),
                    ("tail", tail_offset, tail, True, False),
                ]
    except OSError as exc:
        return None, [{
            "code": "related_artifact_read_failed",
            "message": f"Could not read related text artifact {path}: {exc}",
            "impact": "Its content could not contribute inspection evidence.",
        }]

    decoded_segments = []
    decode_status = "decoded"
    for (
        position,
        byte_offset,
        raw,
        trim_leading,
        trim_trailing,
    ) in segments:
        text, segment_status, boundary_bytes_discarded = (
            _decode_text_segment(
                raw,
                trim_leading_boundary=trim_leading,
                trim_trailing_boundary=trim_trailing,
            )
        )
        if segment_status != "decoded":
            decode_status = "replacement_characters"
        decoded_segments.append({
            "position": position,
            "byte_offset": byte_offset,
            "bytes_read": len(raw),
            "boundary_bytes_discarded": boundary_bytes_discarded,
            "text": text,
        })

    uncertainty = []
    if truncated:
        uncertainty.append({
            "code": "related_artifact_text_truncated",
            "message": (
                f"Related text artifact exceeds {limit_bytes} bytes: {path}"
            ),
            "impact": (
                "Only bounded head and tail segments were inspected."
            ),
        })
    if decode_status != "decoded":
        uncertainty.append({
            "code": "related_artifact_decode_replaced",
            "message": f"Related text artifact is not valid UTF-8: {path}",
            "impact": (
                "Invalid byte sequences were replaced in the text excerpt."
            ),
        })
    return {
        "position": "head_tail" if truncated else "whole",
        "limit_bytes": limit_bytes,
        "bytes_read": sum(
            segment["bytes_read"] for segment in decoded_segments
        ),
        "truncated": truncated,
        "encoding": "utf-8",
        "decode_status": decode_status,
        "segments": decoded_segments,
    }, uncertainty


def _decode_text_segment(
    raw: bytes,
    *,
    trim_leading_boundary: bool,
    trim_trailing_boundary: bool,
) -> tuple[str, str, int]:
    leading_bytes_discarded = 0
    if trim_leading_boundary:
        while (
            leading_bytes_discarded < min(3, len(raw))
            and raw[leading_bytes_discarded] & 0xC0 == 0x80
        ):
            leading_bytes_discarded += 1
    text_bytes = raw[leading_bytes_discarded:]

    decoder_type = codecs.getincrementaldecoder("utf-8")
    decoder = decoder_type(errors="strict")
    try:
        text = decoder.decode(
            text_bytes,
            final=not trim_trailing_boundary,
        )
        trailing_bytes_discarded = (
            len(decoder.getstate()[0])
            if trim_trailing_boundary
            else 0
        )
        decode_status = "decoded"
    except UnicodeDecodeError:
        decoder = decoder_type(errors="replace")
        text = decoder.decode(
            text_bytes,
            final=not trim_trailing_boundary,
        )
        trailing_bytes_discarded = (
            len(decoder.getstate()[0])
            if trim_trailing_boundary
            else 0
        )
        decode_status = "replacement_characters"
    return (
        text,
        decode_status,
        leading_bytes_discarded + trailing_bytes_discarded,
    )


__all__ = [
    "RELATED_TEXT_LIMIT_BYTES",
    "RELATED_TEXT_TOTAL_LIMIT_BYTES",
    "read_text_excerpt",
]

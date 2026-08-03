"""Bounded inspection of GRASP2018 mixing-coefficient files.

The reader validates every energy and CSF coefficient but returns only
caller-bounded per-level and per-component summaries.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import os
from pathlib import Path
import stat
import struct
from typing import BinaryIO, cast

from chemtools.programs.grasp.binary._fortran_records import (
    detect_record_byte_order,
    finish_record,
    read_record_size,
)
from chemtools.programs.grasp.parse.csf import (
    CsfDocument,
    load_grasp_csf_list,
)


GRASP_MIXING_INSPECTION_SCHEMA = "chemtools.grasp-mixing-inspection/1"
MAX_GRASP_MIXING_BYTES = 512 * 1024 * 1024
MAX_GRASP_MIXING_BLOCKS = 1_024
MAX_GRASP_MIXING_LEVELS = 20_000
MAX_GRASP_MIXING_LEVEL_LIMIT = 2_000
DEFAULT_GRASP_MIXING_LEVEL_LIMIT = 256
MAX_GRASP_MIXING_COMPONENT_LIMIT = 50
DEFAULT_GRASP_MIXING_COMPONENT_LIMIT = 5
MIXING_NORMALIZATION_TOLERANCE = 1e-8
_MAGIC = b"G92MIX"
_GLOBAL_HEADER_BYTES = 24
_BLOCK_HEADER_BYTES = 20
_MAX_ORBITALS = 127
_COEFFICIENT_CHUNK = 8_192


def inspect_grasp_mixing(
    path: str | Path,
    *,
    level_limit: int = DEFAULT_GRASP_MIXING_LEVEL_LIMIT,
    component_limit: int = DEFAULT_GRASP_MIXING_COMPONENT_LIMIT,
    csf_path: str | Path | None = None,
) -> dict[str, object]:
    if (
        isinstance(level_limit, bool)
        or not isinstance(level_limit, int)
        or not 0 <= level_limit <= MAX_GRASP_MIXING_LEVEL_LIMIT
    ):
        raise ValueError(
            "level_limit must be an integer between 0 and "
            f"{MAX_GRASP_MIXING_LEVEL_LIMIT}"
        )
    if (
        isinstance(component_limit, bool)
        or not isinstance(component_limit, int)
        or not 1 <= component_limit <= MAX_GRASP_MIXING_COMPONENT_LIMIT
    ):
        raise ValueError(
            "component_limit must be an integer between 1 and "
            f"{MAX_GRASP_MIXING_COMPONENT_LIMIT}"
        )

    source = Path(path).expanduser().resolve()
    try:
        stream = source.open("rb")
    except OSError as error:
        raise ValueError(
            f"cannot open GRASP mixing-coefficient file {source}: {error}"
        ) from error

    with stream:
        initial_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(initial_stat.st_mode):
            raise ValueError(
                f"GRASP mixing-coefficient path is not a regular file: {source}"
            )
        if initial_stat.st_size > MAX_GRASP_MIXING_BYTES:
            raise ValueError(
                "GRASP mixing-coefficient file exceeds "
                f"{MAX_GRASP_MIXING_BYTES} bytes"
            )

        endian = _read_header(stream)
        global_values = _read_global_header(stream, endian)
        (
            electron_count,
            csf_count,
            orbital_count,
            eigenstate_count,
            coefficient_count,
            block_count,
        ) = global_values
        _validate_global_header(global_values)

        blocks: list[dict[str, object]] = []
        parsed_csfs = 0
        parsed_eigenstates = 0
        parsed_coefficients = 0
        levels_returned = 0
        all_vectors_normalized = True
        max_norm_deviation = 0.0

        for expected_block in range(1, block_count + 1):
            block_header = _read_sized_record(
                stream,
                endian,
                f"block {expected_block} header",
                _BLOCK_HEADER_BYTES,
            )
            block_index, block_csfs, block_levels, j_code, parity_code = (
                struct.unpack(f"{endian}5i", block_header)
            )
            _validate_block_header(
                expected_block,
                block_index,
                block_csfs,
                block_levels,
                j_code,
                parity_code,
                csf_count,
            )

            parsed_csfs += block_csfs
            parsed_eigenstates += block_levels
            parsed_coefficients += block_csfs * block_levels
            if block_levels == 0:
                blocks.append(
                    {
                        "index": block_index,
                        "csf_count": block_csfs,
                        "eigenstate_count": 0,
                        "two_j": None,
                        "j": None,
                        "j_label": None,
                        "parity": None,
                        "average_energy_au": None,
                        "levels_returned": 0,
                        "levels_omitted": 0,
                        "levels": [],
                    }
                )
                continue

            state_indices = _read_state_indices(
                stream,
                endian,
                expected_block,
                block_csfs,
                block_levels,
            )
            average_energy, relative_energies = _read_energies(
                stream,
                endian,
                expected_block,
                block_levels,
            )
            summaries, block_max_deviation, block_normalized = (
                _read_coefficient_summaries(
                    stream,
                    endian,
                    expected_block,
                    block_csfs,
                    state_indices,
                    average_energy,
                    relative_energies,
                    level_limit=max(0, level_limit - levels_returned),
                    component_limit=component_limit,
                )
            )
            levels_returned += len(summaries)
            max_norm_deviation = max(max_norm_deviation, block_max_deviation)
            all_vectors_normalized = (
                all_vectors_normalized and block_normalized
            )
            blocks.append(
                {
                    "index": block_index,
                    "csf_count": block_csfs,
                    "eigenstate_count": block_levels,
                    "two_j": j_code - 1,
                    "j": (j_code - 1) / 2,
                    "j_label": _j_label(j_code - 1),
                    "parity": "+" if parity_code == 1 else "-",
                    "average_energy_au": average_energy,
                    "levels_returned": len(summaries),
                    "levels_omitted": block_levels - len(summaries),
                    "levels": summaries,
                }
            )

        if stream.read(1):
            raise ValueError(
                "GRASP mixing-coefficient file has data after the declared blocks"
            )
        if parsed_csfs != csf_count:
            raise ValueError(
                f"block CSF total {parsed_csfs} does not match header {csf_count}"
            )
        if parsed_eigenstates != eigenstate_count:
            raise ValueError(
                "block eigenstate total "
                f"{parsed_eigenstates} does not match header {eigenstate_count}"
            )
        if parsed_coefficients != coefficient_count:
            raise ValueError(
                "block coefficient total "
                f"{parsed_coefficients} does not match header {coefficient_count}"
            )

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
                "GRASP mixing-coefficient file changed during inspection"
            )

    report: dict[str, object] = {
        "schema_version": GRASP_MIXING_INSPECTION_SCHEMA,
        "path": str(source),
        "format": {
            "magic": _MAGIC.decode("ascii"),
            "byte_order": "little" if endian == "<" else "big",
            "record_marker_bytes": 4,
            "filename_producer_convention": _producer_convention(source),
            "source_contract": [
                "rmcdhf90/setmix.f90",
                "rmcdhf90/matrix.f90",
                "rci90/setmix.f90",
                "rci90/lodmix.f90",
                "jj2lsj90/getmixblock.f90",
            ],
        },
        "file": {
            "size_bytes": initial_stat.st_size,
            "sha256": digest.hexdigest(),
        },
        "header": {
            "electron_count": electron_count,
            "csf_count": csf_count,
            "orbital_count": orbital_count,
            "eigenstate_count": eigenstate_count,
            "coefficient_count": coefficient_count,
            "block_count": block_count,
        },
        "response": {
            "level_limit": level_limit,
            "component_limit": component_limit,
            "levels_returned": levels_returned,
            "levels_omitted": eigenstate_count - levels_returned,
        },
        "blocks": blocks,
        "checks": {
            "complete_block_records": True,
            "header_totals_match_blocks": True,
            "finite_energies": True,
            "finite_coefficients": True,
            "unique_level_indices_per_block": True,
            "all_vectors_normalized": all_vectors_normalized,
            "normalization_tolerance": MIXING_NORMALIZATION_TOLERANCE,
            "maximum_norm_deviation": max_norm_deviation,
        },
    }
    if csf_path is not None:
        report["csf_mapping"] = _attach_csf_mapping(
            load_grasp_csf_list(csf_path),
            blocks,
            electron_count=electron_count,
            orbital_count=orbital_count,
            csf_count=csf_count,
        )
    return report


def _attach_csf_mapping(
    document: CsfDocument,
    mixing_blocks: list[dict[str, object]],
    *,
    electron_count: int,
    orbital_count: int,
    csf_count: int,
) -> dict[str, object]:
    if document.electron_count != electron_count:
        raise ValueError(
            "CSF electron count "
            f"{document.electron_count} does not match mixing header "
            f"{electron_count}"
        )
    if document.orbital_count != orbital_count:
        raise ValueError(
            "CSF subshell count "
            f"{document.orbital_count} does not match mixing header "
            f"{orbital_count}"
        )
    if document.csf_count != csf_count:
        raise ValueError(
            f"CSF list total {document.csf_count} does not match mixing "
            f"header {csf_count}"
        )
    if len(document.blocks) != len(mixing_blocks):
        raise ValueError(
            f"CSF block count {len(document.blocks)} does not match mixing "
            f"header {len(mixing_blocks)}"
        )

    for csf_block, mixing_block in zip(document.blocks, mixing_blocks):
        mixing_csf_count = int(mixing_block["csf_count"])
        if len(csf_block.entries) != mixing_csf_count:
            raise ValueError(
                f"CSF block {csf_block.index} contains "
                f"{len(csf_block.entries)} configurations; mixing file "
                f"declares {mixing_csf_count}"
            )
        if int(mixing_block["eigenstate_count"]) > 0 and (
            mixing_block["two_j"] != csf_block.two_j
            or mixing_block["parity"] != csf_block.parity
        ):
            raise ValueError(
                f"CSF block {csf_block.index} symmetry "
                f"{csf_block.j_label}{csf_block.parity} does not match "
                "the mixing file"
            )
        mixing_block["csf_symmetry"] = {
            "two_j": csf_block.two_j,
            "j": csf_block.two_j / 2,
            "j_label": csf_block.j_label,
            "parity": csf_block.parity,
        }
        levels = cast(list[dict[str, object]], mixing_block["levels"])
        for level in levels:
            components = cast(
                list[dict[str, object]],
                level["components"],
            )
            for component in components:
                component_index = int(component["csf_index"])
                component["csf"] = csf_block.entries[
                    component_index - 1
                ].summary()
            level["dominant_csf"] = components[0]["csf"]

    return {
        "path": str(document.source),
        "file": {
            "size_bytes": document.size_bytes,
            "sha256": document.sha256,
        },
        "source_contract": [
            "lib9290/lodcsh.f90",
            "rmcdhf90/lodcsh2GG.f90",
        ],
        "core_subshells": list(document.core_subshells),
        "peel_subshells": list(document.peel_subshells),
        "electron_count": document.electron_count,
        "orbital_count": document.orbital_count,
        "csf_count": document.csf_count,
        "block_count": len(document.blocks),
        "checks": {
            "electron_count_matches": True,
            "orbital_count_matches": True,
            "csf_count_matches": True,
            "block_counts_match": True,
            "populated_block_symmetries_match": True,
            "dominant_indices_resolved": True,
            "returned_component_indices_resolved": True,
        },
    }


def _read_header(stream: BinaryIO) -> str:
    marker = stream.read(4)
    if len(marker) != 4:
        raise ValueError("GRASP mixing-coefficient file has no complete header")
    endian = detect_record_byte_order(marker, len(_MAGIC))
    if endian is None:
        raise ValueError("GRASP mixing-coefficient header record has invalid size")
    payload = stream.read(len(_MAGIC))
    trailer = stream.read(4)
    if payload != _MAGIC:
        raise ValueError("GRASP mixing-coefficient magic must be 'G92MIX'")
    if len(trailer) != 4 or struct.unpack(f"{endian}i", trailer)[0] != len(payload):
        raise ValueError("GRASP mixing-coefficient header markers do not match")
    return endian


def _read_global_header(
    stream: BinaryIO,
    endian: str,
) -> tuple[int, int, int, int, int, int]:
    payload = _read_sized_record(
        stream,
        endian,
        "GRASP mixing global header",
        _GLOBAL_HEADER_BYTES,
    )
    return struct.unpack(f"{endian}6i", payload)


def _validate_global_header(values: tuple[int, int, int, int, int, int]) -> None:
    nelec, ncftot, nw, nvectot, nvecsiz, nblock = values
    if nelec < 1:
        raise ValueError(f"GRASP mixing electron count must be positive: {nelec}")
    if ncftot < 1:
        raise ValueError(f"GRASP mixing CSF count must be positive: {ncftot}")
    if not 1 <= nw <= _MAX_ORBITALS:
        raise ValueError(f"GRASP mixing orbital count is invalid: {nw}")
    if not 0 <= nvectot <= MAX_GRASP_MIXING_LEVELS:
        raise ValueError(f"GRASP mixing eigenstate count is invalid: {nvectot}")
    if not 0 <= nvecsiz <= MAX_GRASP_MIXING_BYTES // 8:
        raise ValueError(f"GRASP mixing coefficient count is invalid: {nvecsiz}")
    if not 1 <= nblock <= MAX_GRASP_MIXING_BLOCKS:
        raise ValueError(f"GRASP mixing block count is invalid: {nblock}")


def _validate_block_header(
    expected_block: int,
    block_index: int,
    block_csfs: int,
    block_levels: int,
    j_code: int,
    parity_code: int,
    total_csfs: int,
) -> None:
    if block_index != expected_block:
        raise ValueError(
            f"GRASP mixing block index {block_index} does not match "
            f"position {expected_block}"
        )
    if not 1 <= block_csfs <= total_csfs:
        raise ValueError(
            f"block {block_index} has invalid CSF count {block_csfs}"
        )
    if not 0 <= block_levels <= block_csfs:
        raise ValueError(
            f"block {block_index} has invalid eigenstate count {block_levels}"
        )
    if block_levels == 0:
        if (j_code, parity_code) != (999, 999):
            raise ValueError(
                f"empty block {block_index} must use GRASP sentinel symmetry"
            )
        return
    if j_code < 1:
        raise ValueError(f"block {block_index} has invalid 2J+1 code {j_code}")
    if parity_code not in (-1, 1):
        raise ValueError(
            f"block {block_index} has invalid parity code {parity_code}"
        )


def _read_state_indices(
    stream: BinaryIO,
    endian: str,
    block_index: int,
    block_csfs: int,
    block_levels: int,
) -> tuple[int, ...]:
    expected_bytes = 4 * block_levels
    payload = _read_sized_record(
        stream,
        endian,
        f"block {block_index} level indices",
        expected_bytes,
    )
    indices = struct.unpack(f"{endian}{block_levels}i", payload)
    if any(index < 1 or index > block_csfs for index in indices):
        raise ValueError(f"block {block_index} has an out-of-range level index")
    if len(indices) != len(set(indices)):
        raise ValueError(f"block {block_index} has duplicate level indices")
    return indices


def _read_energies(
    stream: BinaryIO,
    endian: str,
    block_index: int,
    block_levels: int,
) -> tuple[float, tuple[float, ...]]:
    expected_bytes = 8 * (1 + block_levels)
    payload = _read_sized_record(
        stream,
        endian,
        f"block {block_index} energies",
        expected_bytes,
    )
    values = struct.unpack(f"{endian}{1 + block_levels}d", payload)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"block {block_index} has non-finite energies")
    if not all(math.isfinite(values[0] + value) for value in values[1:]):
        raise ValueError(f"block {block_index} has overflowing absolute energies")
    return values[0], values[1:]


def _read_coefficient_summaries(
    stream: BinaryIO,
    endian: str,
    block_index: int,
    block_csfs: int,
    state_indices: tuple[int, ...],
    average_energy: float,
    relative_energies: tuple[float, ...],
    *,
    level_limit: int,
    component_limit: int,
) -> tuple[list[dict[str, object]], float, bool]:
    field = f"block {block_index} coefficients"
    expected_bytes = 8 * block_csfs * len(state_indices)
    size = read_record_size(
        stream,
        endian,
        field,
        max_record_bytes=MAX_GRASP_MIXING_BYTES,
    )
    if size != expected_bytes:
        raise ValueError(
            f"{field} contain {size} bytes; expected {expected_bytes}"
        )

    summaries: list[dict[str, object]] = []
    max_deviation = 0.0
    all_normalized = True
    for level_position, (state_index, relative_energy) in enumerate(
        zip(state_indices, relative_energies),
        start=1,
    ):
        norm = 0.0
        dominant_coefficient = 0.0
        dominant_csf_index = 1
        leading_components: list[tuple[float, int, float]] = []
        consumed = 0
        while consumed < block_csfs:
            chunk_count = min(_COEFFICIENT_CHUNK, block_csfs - consumed)
            payload = stream.read(8 * chunk_count)
            if len(payload) != 8 * chunk_count:
                raise ValueError(f"incomplete {field} record")
            coefficients = struct.unpack(f"{endian}{chunk_count}d", payload)
            if not all(math.isfinite(value) for value in coefficients):
                raise ValueError(
                    f"block {block_index} level {state_index} has "
                    "non-finite coefficients"
                )
            squares = tuple(value * value for value in coefficients)
            if not all(math.isfinite(value) for value in squares):
                raise ValueError(
                    f"block {block_index} level {state_index} has "
                    "an overflowing coefficient norm"
                )
            norm += math.fsum(squares)
            if not math.isfinite(norm):
                raise ValueError(
                    f"block {block_index} level {state_index} has "
                    "an overflowing coefficient norm"
                )
            for offset, coefficient in enumerate(coefficients):
                csf_index = consumed + offset + 1
                if abs(coefficient) > abs(dominant_coefficient):
                    dominant_coefficient = coefficient
                    dominant_csf_index = csf_index
                if level_position <= level_limit:
                    candidate = (abs(coefficient), -csf_index, coefficient)
                    if len(leading_components) < component_limit:
                        heapq.heappush(leading_components, candidate)
                    elif candidate > leading_components[0]:
                        heapq.heapreplace(leading_components, candidate)
            consumed += chunk_count

        deviation = abs(norm - 1.0)
        max_deviation = max(max_deviation, deviation)
        normalized = deviation <= MIXING_NORMALIZATION_TOLERANCE
        all_normalized = all_normalized and normalized
        if level_position <= level_limit:
            components = [
                {
                    "csf_index": -negative_index,
                    "coefficient": coefficient,
                    "weight": coefficient**2,
                }
                for _, negative_index, coefficient in sorted(
                    leading_components,
                    key=lambda item: (-item[0], -item[1]),
                )
            ]
            returned_weight = math.fsum(
                float(component["weight"]) for component in components
            )
            summaries.append(
                {
                    "level_index": state_index,
                    "relative_energy_au": relative_energy,
                    "energy_au": average_energy + relative_energy,
                    "coefficient_norm": norm,
                    "normalized": normalized,
                    "dominant_csf_index": dominant_csf_index,
                    "dominant_coefficient": dominant_coefficient,
                    "dominant_weight": dominant_coefficient**2,
                    "component_count": block_csfs,
                    "components_returned": len(components),
                    "returned_component_weight": returned_weight,
                    "omitted_component_weight": max(
                        0.0,
                        norm - returned_weight,
                    ),
                    "components": components,
                }
            )

    finish_record(stream, endian, field, size)
    return summaries, max_deviation, all_normalized


def _read_sized_record(
    stream: BinaryIO,
    endian: str,
    field: str,
    expected_bytes: int,
) -> bytes:
    size = read_record_size(
        stream,
        endian,
        field,
        max_record_bytes=MAX_GRASP_MIXING_BYTES,
    )
    if size != expected_bytes:
        raise ValueError(
            f"{field} contains {size} bytes; expected {expected_bytes}"
        )
    payload = stream.read(size)
    if len(payload) != size:
        raise ValueError(f"incomplete {field} record")
    finish_record(stream, endian, field, size)
    return payload


def _j_label(two_j: int) -> str:
    return str(two_j // 2) if two_j % 2 == 0 else f"{two_j}/2"


def _producer_convention(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".m":
        return "rmcdhf"
    if suffix == ".cm":
        return "rci"
    return "unknown"


__all__ = [
    "DEFAULT_GRASP_MIXING_COMPONENT_LIMIT",
    "DEFAULT_GRASP_MIXING_LEVEL_LIMIT",
    "GRASP_MIXING_INSPECTION_SCHEMA",
    "MAX_GRASP_MIXING_BYTES",
    "MAX_GRASP_MIXING_COMPONENT_LIMIT",
    "MAX_GRASP_MIXING_LEVEL_LIMIT",
    "inspect_grasp_mixing",
]

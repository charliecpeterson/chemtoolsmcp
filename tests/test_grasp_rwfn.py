"""Binary-format and MCP contracts for GRASP radial wavefunctions."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct

import pytest

from chemtools.mcp.decorator import _TOOL_PROGRAMS
from chemtools.mcp.dispatch import dispatch_tool, tool_definitions
from chemtools.programs.grasp._plugin_binary import GRASP_BINARY
from chemtools.programs.grasp.binary import (
    inspect_grasp_radial_wfn,
    merge_grasp_radial_wfns,
)
from chemtools.programs.grasp.binary import rwfn


def test_inspector_returns_bounded_orbital_metadata(tmp_path):
    path = tmp_path / "atom.w"
    path.write_bytes(_rwfn_bytes("<", [(1, -1), (2, 1)]))

    inspected = inspect_grasp_radial_wfn(path)

    assert inspected == {
        "schema_version": "chemtools.grasp-radial-wfn-inspection/1",
        "path": str(path.resolve()),
        "format": {
            "magic": "G92RWF",
            "byte_order": "little",
            "record_marker_bytes": 4,
            "source_contract": ["rwfntotxt.f90", "rwfnrelabel.f90"],
        },
        "file": {
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "orbital_count": 2,
        "orbitals": [
            {
                "index": 1,
                "n": 1,
                "kappa": -1,
                "label": "1s",
                "energy_au": 0.5,
                "n_points": 3,
                "a0": 1.25,
                "radial_grid_au": {"minimum": 0.0, "maximum": 0.2},
            },
            {
                "index": 2,
                "n": 2,
                "kappa": 1,
                "label": "2p-",
                "energy_au": 0.5,
                "n_points": 3,
                "a0": 1.25,
                "radial_grid_au": {"minimum": 0.0, "maximum": 0.2},
            },
        ],
        "checks": {
            "complete_record_triples": True,
            "unique_n_kappa": True,
            "finite_radial_values": True,
            "strictly_increasing_radial_grids": True,
        },
    }


def test_inspector_accepts_big_endian_fortran_records(tmp_path):
    path = tmp_path / "big-endian.w"
    path.write_bytes(_rwfn_bytes(">", [(3, -3)]))

    inspected = inspect_grasp_radial_wfn(path)

    assert inspected["format"]["byte_order"] == "big"
    assert inspected["orbitals"] == [{
        "index": 1,
        "n": 3,
        "kappa": -3,
        "label": "3d",
        "energy_au": 0.5,
        "n_points": 3,
        "a0": 1.25,
        "radial_grid_au": {"minimum": 0.0, "maximum": 0.2},
    }]


def test_inspector_rejects_duplicate_orbital_identity(tmp_path):
    path = tmp_path / "duplicate.w"
    path.write_bytes(_rwfn_bytes("<", [(2, -2), (2, -2)]))

    with pytest.raises(ValueError, match=r"duplicate GRASP orbital identity \(2, -2\)"):
        inspect_grasp_radial_wfn(path)


def test_inspector_rejects_nonfinite_radial_values(tmp_path):
    path = tmp_path / "nan.w"
    path.write_bytes(_rwfn_bytes("<", [(1, -1)], component_value=math.nan))

    with pytest.raises(ValueError, match="non-finite radial values"):
        inspect_grasp_radial_wfn(path)


def test_inspector_rejects_nonmonotonic_radial_grid(tmp_path):
    path = tmp_path / "bad-grid.w"
    path.write_bytes(_rwfn_bytes("<", [(1, -1)], radii=(0.0, 0.2, 0.1)))

    with pytest.raises(ValueError, match="strictly increasing"):
        inspect_grasp_radial_wfn(path)


def test_inspector_rejects_truncated_record_triple(tmp_path):
    path = tmp_path / "truncated.w"
    complete = _rwfn_bytes("<", [(1, -1)])
    grid_record_bytes = 4 + 3 * 8 + 4
    path.write_bytes(complete[:-grid_record_bytes])

    with pytest.raises(
        ValueError,
        match=r"incomplete orbital \(1, -1\) radial grid record marker",
    ):
        inspect_grasp_radial_wfn(path)


def test_inspector_rejects_mismatched_record_markers(tmp_path):
    path = tmp_path / "bad-marker.w"
    payload = bytearray(_rwfn_bytes("<", [(1, -1)]))
    payload[-4:] = struct.pack("<i", 999)
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="radial grid record markers do not match"):
        inspect_grasp_radial_wfn(path)


def test_inspector_enforces_file_size_limit(tmp_path, monkeypatch):
    path = tmp_path / "atom.w"
    path.write_bytes(_rwfn_bytes("<", [(1, -1)]))
    monkeypatch.setattr(rwfn, "MAX_GRASP_RWFN_BYTES", path.stat().st_size - 1)

    with pytest.raises(ValueError, match="file exceeds"):
        inspect_grasp_radial_wfn(path)


def test_merger_uses_first_donor_for_duplicate_identity(tmp_path):
    first = tmp_path / "first.w"
    second = tmp_path / "second.w"
    merged = tmp_path / "merged.w"
    first.write_bytes(
        _rwfn_bytes("<", [(1, -1), (2, 1)], energy_au=0.5)
    )
    second.write_bytes(
        _rwfn_bytes("<", [(1, -1), (2, -2)], energy_au=9.5)
    )

    report = merge_grasp_radial_wfns([first, second], merged)

    assert report["schema_version"] == "chemtools.grasp-radial-wfn-merge/1"
    assert report["policy"] == "first_donor_wins_duplicate_n_kappa"
    assert report["donor_count"] == 2
    assert report["duplicate_count"] == 1
    assert report["donors"] == [
        {
            "precedence": 1,
            "path": str(first.resolve()),
            "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
            "orbital_count": 2,
            "contributed_orbitals": ["1s", "2p-"],
            "skipped_duplicates": [],
        },
        {
            "precedence": 2,
            "path": str(second.resolve()),
            "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
            "orbital_count": 2,
            "contributed_orbitals": ["2p"],
            "skipped_duplicates": [
                {"label": "1s", "kept_from": str(first.resolve())}
            ],
        },
    ]
    assert report["output"]["path"] == str(merged.resolve())
    assert report["output"]["orbital_count"] == 3
    assert [
        (orbital["label"], orbital["energy_au"])
        for orbital in report["output"]["orbitals"]
    ] == [("1s", 0.5), ("2p-", 0.5), ("2p", 9.5)]


def test_merger_rejects_later_donor_with_no_new_orbitals(tmp_path):
    first = tmp_path / "first.w"
    second = tmp_path / "second.w"
    merged = tmp_path / "merged.w"
    first.write_bytes(_rwfn_bytes("<", [(1, -1), (2, 1)]))
    second.write_bytes(_rwfn_bytes("<", [(1, -1)]))

    with pytest.raises(ValueError, match="contributes no new orbitals"):
        merge_grasp_radial_wfns([first, second], merged)

    assert merged.exists() is False


def test_merger_rejects_mixed_byte_order_before_writing(tmp_path):
    first = tmp_path / "little.w"
    second = tmp_path / "big.w"
    merged = tmp_path / "merged.w"
    first.write_bytes(_rwfn_bytes("<", [(1, -1)]))
    second.write_bytes(_rwfn_bytes(">", [(2, -2)]))

    with pytest.raises(ValueError, match="mixed byte order"):
        merge_grasp_radial_wfns([first, second], merged)

    assert merged.exists() is False


def test_merger_refuses_existing_output_without_overwrite(tmp_path):
    first = tmp_path / "first.w"
    second = tmp_path / "second.w"
    merged = tmp_path / "merged.w"
    first.write_bytes(_rwfn_bytes("<", [(1, -1)]))
    second.write_bytes(_rwfn_bytes("<", [(2, -2)]))
    merged.write_bytes(b"keep me")

    with pytest.raises(ValueError, match="output path already exists"):
        merge_grasp_radial_wfns([first, second], merged)

    assert merged.read_bytes() == b"keep me"
    report = merge_grasp_radial_wfns(
        [first, second],
        merged,
        overwrite=True,
    )
    assert report["output"]["orbital_count"] == 2


def test_merger_never_replaces_a_donor(tmp_path):
    first = tmp_path / "first.w"
    second = tmp_path / "second.w"
    first_bytes = _rwfn_bytes("<", [(1, -1)])
    first.write_bytes(first_bytes)
    second.write_bytes(_rwfn_bytes("<", [(2, -2)]))

    with pytest.raises(ValueError, match="must not replace a donor"):
        merge_grasp_radial_wfns(
            [first, second],
            first,
            overwrite=True,
        )

    assert first.read_bytes() == first_bytes


def test_corrupt_donor_leaves_no_output(tmp_path):
    first = tmp_path / "first.w"
    corrupt = tmp_path / "corrupt.w"
    merged = tmp_path / "merged.w"
    first.write_bytes(_rwfn_bytes("<", [(1, -1)]))
    corrupt.write_bytes(b"not a GRASP radial wavefunction")

    with pytest.raises(ValueError, match="header record has invalid size"):
        merge_grasp_radial_wfns([first, corrupt], merged)

    assert merged.exists() is False


def test_grasp_binary_provider_reads_and_merges(tmp_path):
    path = tmp_path / "atom.w"
    donor = tmp_path / "donor.w"
    merged = tmp_path / "merged.w"
    path.write_bytes(_rwfn_bytes("<", [(1, -1)]))
    donor.write_bytes(_rwfn_bytes("<", [(2, -2)]))

    assert GRASP_BINARY.supported_kinds() == ["radial_wfn", "mixing"]
    assert GRASP_BINARY.parse(str(path), "radial_wfn")["orbital_count"] == 1
    with pytest.raises(ValueError, match="does not support"):
        GRASP_BINARY.parse(str(path), "checkpoint")
    GRASP_BINARY.write(
        str(merged),
        "radial_wfn",
        {"donor_paths": [str(path), str(donor)]},
    )
    assert GRASP_BINARY.parse(str(merged), "radial_wfn")["orbital_count"] == 2
    with pytest.raises(NotImplementedError, match="does not support"):
        GRASP_BINARY.write(str(merged), "mixing", {})


def test_mcp_inspector_is_grasp_scoped_and_analysis_safe(tmp_path):
    path = tmp_path / "atom.w"
    path.write_bytes(_rwfn_bytes("<", [(1, -1)]))

    payload = dispatch_tool("inspect_grasp_radial_wfn", {"path": str(path)})
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "inspect_grasp_radial_wfn"
    )

    assert payload["orbital_count"] == 1
    assert payload["orbitals"][0]["label"] == "1s"
    assert _TOOL_PROGRAMS["inspect_grasp_radial_wfn"] == "grasp"
    assert definition["inputSchema"]["additionalProperties"] is False


def test_mcp_merger_is_grasp_scoped_and_analysis_safe(tmp_path):
    first = tmp_path / "first.w"
    second = tmp_path / "second.w"
    merged = tmp_path / "merged.w"
    first.write_bytes(_rwfn_bytes("<", [(1, -1)]))
    second.write_bytes(_rwfn_bytes("<", [(2, -2)]))

    payload = dispatch_tool(
        "merge_grasp_radial_wfns",
        {
            "donor_paths": [str(first), str(second)],
            "output_path": str(merged),
        },
    )
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "merge_grasp_radial_wfns"
    )

    assert payload["output"]["orbital_count"] == 2
    assert _TOOL_PROGRAMS["merge_grasp_radial_wfns"] == "grasp"
    assert definition["inputSchema"]["properties"]["donor_paths"] == {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 2,
        "maxItems": 16,
        "uniqueItems": True,
        "description": "GRASP2018 radial-wavefunction files in precedence order.",
    }
    assert definition["inputSchema"]["additionalProperties"] is False


def _rwfn_bytes(
    endian: str,
    identities: list[tuple[int, int]],
    *,
    component_value: float = 0.1,
    energy_au: float = 0.5,
    radii: tuple[float, ...] = (0.0, 0.1, 0.2),
) -> bytes:
    records = [_record(b"G92RWF", endian)]
    for n, kappa in identities:
        header = struct.pack(
            f"{endian}iidi",
            n,
            kappa,
            energy_au,
            len(radii),
        )
        components = struct.pack(
            f"{endian}{1 + 2 * len(radii)}d",
            1.25,
            *([component_value] * (2 * len(radii))),
        )
        grid = struct.pack(f"{endian}{len(radii)}d", *radii)
        records.extend(
            _record(payload, endian)
            for payload in (header, components, grid)
        )
    return b"".join(records)


def _record(payload: bytes, endian: str) -> bytes:
    marker = struct.pack(f"{endian}i", len(payload))
    return marker + payload + marker

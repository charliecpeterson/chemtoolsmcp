"""Binary-format contracts for GRASP mixing-coefficient inspection."""

from __future__ import annotations

import hashlib
import math
import struct

import pytest

from chemtools.mcp.decorator import _TOOL_PROGRAMS
from chemtools.mcp.dispatch import dispatch_tool, tool_definitions
from chemtools.programs.grasp._plugin_binary import GRASP_BINARY
from chemtools.programs.grasp.binary.mixing import inspect_grasp_mixing
from chemtools.programs.grasp.binary import mixing
from chemtools.programs.grasp.parse import csf
from chemtools.programs.grasp.parse.csf import load_grasp_csf_list


def test_inspector_returns_bounded_level_summaries(tmp_path):
    path = tmp_path / "atom.m"
    path.write_bytes(
        _mixing_bytes(
            "<",
            [
                {
                    "csfs": 3,
                    "j_code": 2,
                    "parity": -1,
                    "indices": [1, 3],
                    "average_energy": -10.0,
                    "relative_energies": [-0.5, 0.25],
                    "vectors": [[0.8, 0.6, 0.0], [0.0, 0.0, 1.0]],
                },
                {"csfs": 2},
            ],
        )
    )

    inspected = inspect_grasp_mixing(
        path,
        level_limit=1,
        component_limit=2,
    )

    assert inspected == {
        "schema_version": "chemtools.grasp-mixing-inspection/1",
        "path": str(path.resolve()),
        "format": {
            "magic": "G92MIX",
            "byte_order": "little",
            "record_marker_bytes": 4,
            "filename_producer_convention": "rmcdhf",
            "source_contract": [
                "rmcdhf90/setmix.f90",
                "rmcdhf90/matrix.f90",
                "rci90/setmix.f90",
                "rci90/lodmix.f90",
                "jj2lsj90/getmixblock.f90",
            ],
        },
        "file": {
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "header": {
            "electron_count": 4,
            "csf_count": 5,
            "orbital_count": 4,
            "eigenstate_count": 2,
            "coefficient_count": 6,
            "block_count": 2,
        },
        "response": {
            "level_limit": 1,
            "component_limit": 2,
            "levels_returned": 1,
            "levels_omitted": 1,
        },
        "blocks": [
            {
                "index": 1,
                "csf_count": 3,
                "eigenstate_count": 2,
                "two_j": 1,
                "j": 0.5,
                "j_label": "1/2",
                "parity": "-",
                "average_energy_au": -10.0,
                "levels_returned": 1,
                "levels_omitted": 1,
                "levels": [
                    {
                        "level_index": 1,
                        "relative_energy_au": -0.5,
                        "energy_au": -10.5,
                        "coefficient_norm": 1.0,
                        "normalized": True,
                        "dominant_csf_index": 1,
                        "dominant_coefficient": 0.8,
                        "dominant_weight": 0.6400000000000001,
                        "component_count": 3,
                        "components_returned": 2,
                        "returned_component_weight": 1.0,
                        "omitted_component_weight": 0.0,
                        "components": [
                            {
                                "csf_index": 1,
                                "coefficient": 0.8,
                                "weight": 0.6400000000000001,
                            },
                            {
                                "csf_index": 2,
                                "coefficient": 0.6,
                                "weight": 0.36,
                            },
                        ],
                    }
                ],
            },
            {
                "index": 2,
                "csf_count": 2,
                "eigenstate_count": 0,
                "two_j": None,
                "j": None,
                "j_label": None,
                "parity": None,
                "average_energy_au": None,
                "levels_returned": 0,
                "levels_omitted": 0,
                "levels": [],
            },
        ],
        "checks": {
            "complete_block_records": True,
            "header_totals_match_blocks": True,
            "finite_energies": True,
            "finite_coefficients": True,
            "unique_level_indices_per_block": True,
            "all_vectors_normalized": True,
            "normalization_tolerance": 1e-8,
            "maximum_norm_deviation": 0.0,
        },
    }


def test_inspector_accepts_big_endian_rci_convention(tmp_path):
    path = tmp_path / "atom.cm"
    path.write_bytes(_mixing_bytes(">", [_one_level_block()]))

    inspected = inspect_grasp_mixing(path)

    assert inspected["format"]["byte_order"] == "big"
    assert inspected["format"]["filename_producer_convention"] == "rci"
    assert inspected["blocks"][0]["levels"][0]["energy_au"] == -10.5


def test_inspector_reports_non_normalized_vector_without_rejecting(tmp_path):
    path = tmp_path / "atom.m"
    block = _one_level_block()
    block["vectors"] = [[0.5, 0.0]]
    path.write_bytes(_mixing_bytes("<", [block]))

    inspected = inspect_grasp_mixing(path)

    assert inspected["blocks"][0]["levels"][0]["coefficient_norm"] == 0.25
    assert inspected["blocks"][0]["levels"][0]["normalized"] is False
    assert inspected["checks"]["all_vectors_normalized"] is False
    assert inspected["checks"]["maximum_norm_deviation"] == 0.75


def test_inspector_orders_equal_components_and_reports_omitted_weight(tmp_path):
    path = tmp_path / "atom.m"
    path.write_bytes(
        _mixing_bytes(
            "<",
            [
                {
                    "csfs": 4,
                    "j_code": 1,
                    "parity": 1,
                    "indices": [1],
                    "average_energy": -10.0,
                    "relative_energies": [-0.5],
                    "vectors": [[0.5, -0.5, 0.5, -0.5]],
                }
            ],
        )
    )

    inspected = inspect_grasp_mixing(path, component_limit=2)
    level = inspected["blocks"][0]["levels"][0]

    assert level["component_count"] == 4
    assert level["components_returned"] == 2
    assert level["returned_component_weight"] == 0.5
    assert level["omitted_component_weight"] == 0.5
    assert level["components"] == [
        {"csf_index": 1, "coefficient": 0.5, "weight": 0.25},
        {"csf_index": 2, "coefficient": -0.5, "weight": 0.25},
    ]


def test_inspector_rejects_truncated_coefficient_record(tmp_path):
    path = tmp_path / "truncated.m"
    path.write_bytes(_mixing_bytes("<", [_one_level_block()])[:-4])

    with pytest.raises(
        ValueError,
        match="incomplete block 1 coefficients record",
    ):
        inspect_grasp_mixing(path)


def test_inspector_rejects_header_total_mismatch(tmp_path):
    path = tmp_path / "bad-total.m"
    path.write_bytes(
        _mixing_bytes(
            "<",
            [_one_level_block()],
            coefficient_count=3,
        )
    )

    with pytest.raises(
        ValueError,
        match="block coefficient total 2 does not match header 3",
    ):
        inspect_grasp_mixing(path)


def test_inspector_rejects_duplicate_level_indices(tmp_path):
    path = tmp_path / "duplicate.m"
    path.write_bytes(
        _mixing_bytes(
            "<",
            [
                {
                    "csfs": 2,
                    "j_code": 1,
                    "parity": 1,
                    "indices": [1, 1],
                    "average_energy": -10.0,
                    "relative_energies": [-0.5, 0.25],
                    "vectors": [[1.0, 0.0], [0.0, 1.0]],
                }
            ],
        )
    )

    with pytest.raises(ValueError, match="block 1 has duplicate level indices"):
        inspect_grasp_mixing(path)


def test_inspector_rejects_nonfinite_coefficient(tmp_path):
    path = tmp_path / "nan.m"
    block = _one_level_block()
    block["vectors"] = [[math.nan, 0.0]]
    path.write_bytes(_mixing_bytes("<", [block]))

    with pytest.raises(ValueError, match="non-finite coefficients"):
        inspect_grasp_mixing(path)


def test_inspector_validates_level_limit(tmp_path):
    path = tmp_path / "atom.m"
    path.write_bytes(_mixing_bytes("<", [_one_level_block()]))

    for value in (True, -1, 2001, 1.5):
        with pytest.raises(ValueError, match="level_limit"):
            inspect_grasp_mixing(path, level_limit=value)


def test_inspector_validates_component_limit(tmp_path):
    path = tmp_path / "atom.m"
    path.write_bytes(_mixing_bytes("<", [_one_level_block()]))

    for value in (True, 0, 51, 1.5):
        with pytest.raises(ValueError, match="component_limit"):
            inspect_grasp_mixing(path, component_limit=value)


def test_inspector_enforces_file_size_limit(tmp_path, monkeypatch):
    path = tmp_path / "atom.m"
    path.write_bytes(_mixing_bytes("<", [_one_level_block()]))
    monkeypatch.setattr(
        mixing,
        "MAX_GRASP_MIXING_BYTES",
        path.stat().st_size - 1,
    )

    with pytest.raises(ValueError, match="file exceeds"):
        inspect_grasp_mixing(path)


def test_inspector_maps_dominant_coefficients_to_csf_configurations(tmp_path):
    mixing_path = tmp_path / "atom.m"
    csf_path = tmp_path / "atom.c"
    mixing_path.write_bytes(
        _mixing_bytes(
            "<",
            [
                {
                    "csfs": 3,
                    "j_code": 1,
                    "parity": 1,
                    "indices": [1],
                    "average_energy": -10.0,
                    "relative_energies": [-0.5],
                    "vectors": [[0.0, 1.0, 0.0]],
                },
                {
                    "csfs": 2,
                    "j_code": 3,
                    "parity": 1,
                    "indices": [1],
                    "average_energy": -9.0,
                    "relative_energies": [-0.25],
                    "vectors": [[1.0, 0.0]],
                },
            ],
            electron_count=2,
        )
    )
    csf_path.write_text(_two_block_csf_text(), encoding="ascii")

    inspected = inspect_grasp_mixing(mixing_path, csf_path=csf_path)

    assert inspected["csf_mapping"]["checks"] == {
        "electron_count_matches": True,
        "orbital_count_matches": True,
        "csf_count_matches": True,
        "block_counts_match": True,
        "populated_block_symmetries_match": True,
        "dominant_indices_resolved": True,
        "returned_component_indices_resolved": True,
    }
    assert inspected["blocks"][0]["csf_symmetry"] == {
        "two_j": 0,
        "j": 0.0,
        "j_label": "0",
        "parity": "+",
    }
    assert inspected["blocks"][0]["levels"][0]["dominant_csf"] == {
        "block_index": 1,
        "index_within_block": 2,
        "global_index": 2,
        "configuration": "2s(2)",
        "occupations": [{"subshell": "2s", "electrons": 2}],
        "subshell_quantum_numbers": "",
        "coupling_and_symmetry": "         0+",
        "two_j": 0,
        "j_label": "0",
        "parity": "+",
        "source_lines": ["  2s ( 2)", "", "         0+"],
    }
    assert inspected["blocks"][1]["levels"][0]["dominant_csf"][
        "global_index"
    ] == 4
    assert inspected["blocks"][0]["levels"][0]["components"][0]["csf"][
        "configuration"
    ] == "2s(2)"
    assert inspected["blocks"][1]["levels"][0]["components"][0]["csf"][
        "global_index"
    ] == 4


def test_inspector_rejects_csf_symmetry_mismatch(tmp_path):
    mixing_path = tmp_path / "atom.m"
    csf_path = tmp_path / "atom.c"
    mixing_path.write_bytes(
        _mixing_bytes("<", [_one_level_block()], electron_count=4)
    )
    csf_path.write_text(
        _one_block_csf_text().replace("         0+", "         1+"),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match=r"symmetry 1\+ does not match"):
        inspect_grasp_mixing(mixing_path, csf_path=csf_path)


def test_inspector_rejects_csf_count_mismatch(tmp_path):
    mixing_path = tmp_path / "atom.m"
    csf_path = tmp_path / "atom.c"
    mixing_path.write_bytes(_mixing_bytes("<", [_one_level_block()]))
    csf_path.write_text(
        _one_block_csf_text().split("  1s ( 2)  2p-( 2)")[0],
        encoding="ascii",
    )

    with pytest.raises(
        ValueError,
        match="CSF list total 1 does not match mixing header 2",
    ):
        inspect_grasp_mixing(mixing_path, csf_path=csf_path)


def test_csf_loader_rejects_inconsistent_electron_counts(tmp_path):
    mixing_path = tmp_path / "atom.m"
    csf_path = tmp_path / "atom.c"
    mixing_path.write_bytes(_mixing_bytes("<", [_one_level_block()]))
    csf_path.write_text(
        _one_block_csf_text().replace(
            "  1s ( 2)  2p-( 2)",
            "  1s ( 1)  2p-( 2)",
        ),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="do not have one electron count"):
        inspect_grasp_mixing(mixing_path, csf_path=csf_path)


def test_csf_loader_enforces_text_boundaries(tmp_path, monkeypatch):
    non_ascii = tmp_path / "non-ascii.c"
    non_ascii.write_bytes(_one_block_csf_text().encode("ascii") + b"\xff")
    with pytest.raises(ValueError, match="must contain ASCII text"):
        load_grasp_csf_list(non_ascii)

    long_line = tmp_path / "long-line.c"
    long_line.write_text(
        _one_block_csf_text().replace(
            "  1s   2s   2p-  2p",
            " " * 257,
        ),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="must not exceed 256 characters"):
        load_grasp_csf_list(long_line)

    valid = tmp_path / "valid.c"
    valid.write_text(_one_block_csf_text(), encoding="ascii")
    monkeypatch.setattr(csf, "MAX_GRASP_CSF_BYTES", valid.stat().st_size - 1)
    with pytest.raises(ValueError, match="file exceeds"):
        load_grasp_csf_list(valid)


def test_grasp_binary_provider_reads_mixing_file(tmp_path):
    path = tmp_path / "atom.m"
    path.write_bytes(_mixing_bytes("<", [_one_level_block()]))

    inspected = GRASP_BINARY.parse(str(path), "mixing")

    assert GRASP_BINARY.supported_kinds() == ["radial_wfn", "mixing"]
    assert inspected["header"]["eigenstate_count"] == 1
    assert inspected["blocks"][0]["levels"][0]["dominant_csf_index"] == 1


def test_mcp_inspector_is_grasp_scoped_and_bounded(tmp_path):
    path = tmp_path / "atom.m"
    csf_path = tmp_path / "atom.c"
    path.write_bytes(_mixing_bytes("<", [_one_level_block()]))
    csf_path.write_text(_one_block_csf_text(), encoding="ascii")

    inspected = dispatch_tool(
        "inspect_grasp_mixing",
        {
            "path": str(path),
            "csf_path": str(csf_path),
            "level_limit": 1,
            "component_limit": 2,
        },
    )
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "inspect_grasp_mixing"
    )

    assert inspected["blocks"][0]["levels"][0]["dominant_csf"][
        "configuration"
    ] == "1s(2) 2s(2)"
    assert _TOOL_PROGRAMS["inspect_grasp_mixing"] == "grasp"
    assert definition["inputSchema"]["properties"]["level_limit"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 2000,
        "default": 256,
        "description": (
            "Maximum per-level summaries to return across all blocks. "
            "Every level is still validated."
        ),
    }
    assert definition["inputSchema"]["properties"]["csf_path"] == {
        "type": "string",
        "description": (
            "Optional matching GRASP .c file. When supplied, validate its "
            "block ordering and resolve every returned component to its CSF."
        ),
    }
    assert definition["inputSchema"]["properties"]["component_limit"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 50,
        "default": 5,
        "description": (
            "Maximum leading CSF components returned per included level. "
            "Every coefficient is still validated."
        ),
    }
    assert definition["inputSchema"]["additionalProperties"] is False


def _one_level_block() -> dict[str, object]:
    return {
        "csfs": 2,
        "j_code": 1,
        "parity": 1,
        "indices": [1],
        "average_energy": -10.0,
        "relative_energies": [-0.5],
        "vectors": [[0.8, 0.6]],
    }


def _mixing_bytes(
    endian: str,
    blocks: list[dict[str, object]],
    *,
    coefficient_count: int | None = None,
    electron_count: int = 4,
    orbital_count: int = 4,
) -> bytes:
    populated = [block for block in blocks if block.get("indices")]
    csf_count = sum(int(block["csfs"]) for block in blocks)
    eigenstate_count = sum(len(block["indices"]) for block in populated)
    actual_coefficients = sum(
        int(block["csfs"]) * len(block["indices"])
        for block in populated
    )
    global_header = struct.pack(
        f"{endian}6i",
        electron_count,
        csf_count,
        orbital_count,
        eigenstate_count,
        actual_coefficients if coefficient_count is None else coefficient_count,
        len(blocks),
    )
    records = [_record(b"G92MIX", endian), _record(global_header, endian)]
    for index, block in enumerate(blocks, start=1):
        indices = list(block.get("indices", []))
        block_header = struct.pack(
            f"{endian}5i",
            index,
            int(block["csfs"]),
            len(indices),
            int(block.get("j_code", 999)),
            int(block.get("parity", 999)),
        )
        records.append(_record(block_header, endian))
        if not indices:
            continue
        records.append(
            _record(struct.pack(f"{endian}{len(indices)}i", *indices), endian)
        )
        energy_values = [
            float(block["average_energy"]),
            *[float(value) for value in block["relative_energies"]],
        ]
        records.append(
            _record(
                struct.pack(f"{endian}{len(energy_values)}d", *energy_values),
                endian,
            )
        )
        coefficients = [
            float(value)
            for vector in block["vectors"]
            for value in vector
        ]
        records.append(
            _record(
                struct.pack(f"{endian}{len(coefficients)}d", *coefficients),
                endian,
            )
        )
    return b"".join(records)


def _record(payload: bytes, endian: str) -> bytes:
    marker = struct.pack(f"{endian}i", len(payload))
    return marker + payload + marker


def _one_block_csf_text() -> str:
    return """Core subshells:

Peel subshells:
  1s   2s   2p-  2p
CSF(s):
  1s ( 2)  2s ( 2)

         0+
  1s ( 2)  2p-( 2)

         0+
"""


def _two_block_csf_text() -> str:
    return """Core subshells:

Peel subshells:
  1s   2s   2p-  2p
CSF(s):
  1s ( 2)

         0+
  2s ( 2)

         0+
  2p-( 2)

         0+
 *
  2p-( 1)  2p ( 1)
      1/2      3/2
                  1+
  1s ( 1)  2s ( 1)
      1/2      1/2
                  1+
"""

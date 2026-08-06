"""Catalog-bound validation of generated GRASP CSF and mixing artifacts."""

from __future__ import annotations

import pytest

import chemtools.reference.fblock_grasp as fblock_grasp
from chemtools.mcp.decorator import _TOOL_CAPABILITIES, _TOOL_PROGRAMS
from chemtools.mcp.dispatch import dispatch_tool, tool_definitions
from chemtools.reference import validate_grasp_fblock_artifacts


_PA_CSF = """Core subshells:
  1s   2s   2p-  2p   3s   3p-  3p   3d-  3d   4s   4p-  4p   4d-  4d   5s   5p-  5p
Peel subshells:
  4f-  4f   5d-  5d   5f-  5f   6s   6p-  6p   7s
CSF(s):
  4f-( 6)  4f ( 8)  5d-( 4)  5d ( 6)  5f-( 1)  6s ( 2)  6p-( 2)  6p ( 4)  7s ( 1)
                                          5/2                                 1/2
                                                                                 2-
 *
  4f-( 6)  4f ( 8)  5d-( 4)  5d ( 6)  5f ( 1)  6s ( 2)  6p-( 2)  6p ( 4)  7s ( 1)
                                          7/2                                 1/2
                                                                                 3-
  4f-( 6)  4f ( 8)  5d-( 4)  5d ( 6)  5f-( 1)  6s ( 2)  6p-( 2)  6p ( 4)  7s ( 1)
                                          5/2                                 1/2
                                                                                 3-
 *
  4f-( 6)  4f ( 8)  5d-( 4)  5d ( 6)  5f ( 1)  6s ( 2)  6p-( 2)  6p ( 4)  7s ( 1)
                                          7/2                                 1/2
                                                                                 4-
"""


def test_generated_pa_csfs_match_corrected_ion_and_every_catalog_block(tmp_path):
    path = tmp_path / "rcsf.out"
    path.write_text(_PA_CSF, encoding="ascii")

    payload = validate_grasp_fblock_artifacts(
        "Pa",
        "ion3_5f17s1",
        path,
    )

    assert payload["schema_version"] == "chemtools.fblock-grasp-validation/1"
    assert payload["valid"] is True
    assert payload["csf"]["electron_count"] == 88
    assert payload["csf"]["csf_count"] == 4
    assert payload["csf"]["blocks"] == [
        {"j": "2", "parity": "-", "ncsf": 1},
        {"j": "3", "parity": "-", "ncsf": 2},
        {"j": "4", "parity": "-", "ncsf": 1},
    ]


def test_catalog_validator_rejects_a_plausible_wrong_asf_symmetry(tmp_path):
    path = tmp_path / "wrong.c"
    path.write_text(_PA_CSF.replace("2-\n *", "1-\n *", 1), encoding="ascii")

    with pytest.raises(ValueError, match="CSF blocks do not match"):
        validate_grasp_fblock_artifacts("Pa", "ion3_5f17s1", path)


def test_catalog_validator_rejects_partial_asf_manifold(tmp_path, monkeypatch):
    path = tmp_path / "partial.c"
    path.write_text(_PA_CSF, encoding="ascii")
    monkeypatch.setattr(
        fblock_grasp,
        "inspect_grasp_mixing",
        lambda *args, **kwargs: {
            "blocks": [
                {"eigenstate_count": 1},
                {"eigenstate_count": 1},
                {"eigenstate_count": 1},
            ],
            "checks": {},
        },
    )

    with pytest.raises(ValueError, match="requires all 2"):
        validate_grasp_fblock_artifacts(
            "Pa",
            "ion3_5f17s1",
            path,
            mixing_path=tmp_path / "partial.m",
        )


def test_catalog_validator_is_exposed_as_a_read_only_grasp_tool(tmp_path):
    path = tmp_path / "rcsf.out"
    path.write_text(_PA_CSF, encoding="ascii")

    payload = dispatch_tool(
        "validate_grasp_fblock_artifacts",
        {"element": "Pa", "state": "ion3_5f17s1", "csf_path": str(path)},
    )
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "validate_grasp_fblock_artifacts"
    )

    assert payload["valid"] is True
    assert definition["inputSchema"]["additionalProperties"] is False
    assert _TOOL_PROGRAMS["validate_grasp_fblock_artifacts"] == "grasp"
    assert _TOOL_CAPABILITIES["validate_grasp_fblock_artifacts"] == "none"

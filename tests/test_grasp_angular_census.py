"""Independent jj-coupling checks for represented GRASP configurations."""

from __future__ import annotations

import pytest

from chemtools.mcp.decorator import _TOOL_CAPABILITIES, _TOOL_PROGRAMS
from chemtools.mcp.dispatch import dispatch_tool, tool_definitions
from chemtools.reference.grasp_angular_census import (
    validate_grasp_csf_angular_census,
)


_P2_CSF = """Core subshells:
  1s
Peel subshells:
  2p-  2p
CSF(s):
  2p-( 2)
  0
  0+
  2p ( 2)
  0
  0+
 *
  2p-( 1)  2p ( 1)
  1/2  3/2
  1+
 *
  2p-( 1)  2p ( 1)
  1/2  3/2
  2+
  2p ( 2)
  3/2
  2+
"""


def test_grasp_p2_configurations_have_exact_jj_multiplicities(tmp_path):
    path = tmp_path / "p2.c"
    path.write_text(_P2_CSF, encoding="ascii")

    census = validate_grasp_csf_angular_census(path)

    assert census["schema_version"] == "chemtools.grasp-angular-census/1"
    assert census["electron_count"] == 4
    assert census["csf_count"] == 5
    assert census["configuration_count"] == 3
    assert census["full_j_manifold_present"] is True
    assert {
        row["configuration"]: row["complete_j_levels"]
        for row in census["configurations"]
    } == {
        "2p(2)": [
            {"two_j": 0, "j": "0", "csfs": 1},
            {"two_j": 4, "j": "2", "csfs": 1},
        ],
        "2p-(1) 2p(1)": [
            {"two_j": 2, "j": "1", "csfs": 1},
            {"two_j": 4, "j": "2", "csfs": 1},
        ],
        "2p-(2)": [
            {"two_j": 0, "j": "0", "csfs": 1},
        ],
    }


def test_grasp_census_reports_an_intentionally_restricted_j_manifold(tmp_path):
    path = tmp_path / "p2-j0.c"
    path.write_text(_P2_CSF.split(" *\n", 1)[0], encoding="ascii")

    census = validate_grasp_csf_angular_census(path)

    assert census["valid"] is True
    assert census["full_j_manifold_present"] is False
    p_upper = next(
        row
        for row in census["configurations"]
        if row["configuration"] == "2p(2)"
    )
    assert p_upper["present_j_levels"] == [
        {"two_j": 0, "j": "0", "csfs": 1},
    ]
    assert p_upper["full_j_manifold_present"] is False


def test_grasp_census_rejects_duplicate_coupling_path(tmp_path):
    path = tmp_path / "duplicate.c"
    path.write_text(
        _P2_CSF.replace(
            "  2p ( 2)\n  0\n  0+\n *",
            "  2p ( 2)\n  0\n  0+\n  2p ( 2)\n  0\n  0+\n *",
        ),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="independent jj coupling requires 1"):
        validate_grasp_csf_angular_census(path)


def test_grasp_angular_census_is_a_read_only_grasp_tool(tmp_path):
    path = tmp_path / "p2.c"
    path.write_text(_P2_CSF, encoding="ascii")

    payload = dispatch_tool(
        "validate_grasp_csf_angular_census",
        {"csf_path": str(path)},
    )
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "validate_grasp_csf_angular_census"
    )

    assert payload["valid"] is True
    assert definition["inputSchema"]["additionalProperties"] is False
    assert _TOOL_PROGRAMS["validate_grasp_csf_angular_census"] == "grasp"
    assert _TOOL_CAPABILITIES["validate_grasp_csf_angular_census"] == "none"

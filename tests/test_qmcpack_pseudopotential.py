"""Regression coverage for semilocal QMCPACK pseudopotential inspection."""

from __future__ import annotations

from pathlib import Path

import pytest

from chemtools.mcp.tools.qmcpack import (
    _handle_inspect_qmcpack_pseudopotential,
    _handle_inspect_qmcpack_referenced_pseudopotentials,
)
from chemtools.programs.qmcpack.includes import inspect_xml_includes
from chemtools.programs.qmcpack.input import parse_qmcpack_input
from chemtools.programs.qmcpack.pseudopotential import (
    inspect_qmcpack_pseudopotential,
    inspect_referenced_pseudopotentials,
)


PSEUDOPOTENTIAL = """\
<pseudo version="0.5">
  <header symbol="O" atomic-number="8" zval="6" creator="ppconvert" flavor="Troullier-Martins"/>
  <grid type="linear" units="bohr" ri="0" rf="10" npts="5"/>
  <semilocal units="hartree" format="r*V" npots-down="2" npots-up="0" l-local="1">
    <vps principal-n="0" l="s" spin="-1" cutoff="1.3"><radfunc><grid type="linear" units="bohr" ri="0" rf="10" npts="5"/><data>0 -2 -4 -6 -6</data></radfunc></vps>
    <vps principal-n="0" l="p" spin="-1" cutoff="1.4"><radfunc><grid type="linear" units="bohr" ri="0" rf="10" npts="5"/><data>0 -3 -5 -6 -6</data></radfunc></vps>
  </semilocal>
</pseudo>
"""


def _write(tmp_path: Path, text: str = PSEUDOPOTENTIAL) -> Path:
    path = tmp_path / "O.xml"
    path.write_text(text, encoding="utf-8")
    return path


def test_inspect_qmcpack_pseudopotential_reports_channels_and_tail(tmp_path):
    path = _write(tmp_path)

    inspected = inspect_qmcpack_pseudopotential(path)

    assert inspected["header"] == {
        "symbol": "O",
        "atomic_number": 8,
        "zval": 6.0,
        "relativistic": None,
        "flavor": "Troullier-Martins",
        "creator": "ppconvert",
    }
    assert inspected["semilocal"]["local_channel"] == 1
    assert [channel["l"] for channel in inspected["semilocal"]["channels"]] == [
        "s", "p",
    ]
    assert inspected["tail_check"] == {
        "expected_r_times_v_hartree": -6.0,
        "samples_per_channel": 3,
        "channels": [
            {
                "l": "s",
                "values_hartree": [-4.0, -6.0, -6.0],
                "mean_hartree": pytest.approx(-16 / 3),
                "difference_from_expected_hartree": pytest.approx(2 / 3),
            },
            {
                "l": "p",
                "values_hartree": [-5.0, -6.0, -6.0],
                "mean_hartree": pytest.approx(-17 / 3),
                "difference_from_expected_hartree": pytest.approx(1 / 3),
            },
        ],
    }
    assert inspected["structural_evidence"] == {
        "units_are_hartree": True,
        "format_is_r_times_v": True,
        "all_channel_grids_are_linear": True,
        "all_declared_grid_counts_match_data": True,
        "local_channel_has_vps": True,
        "channel_labels_are_recognized": True,
        "channel_spin_pairs_are_unique": True,
    }
    assert inspected["warnings"] == []
    assert _handle_inspect_qmcpack_pseudopotential({
        "pseudopotential_file": str(path),
    }) == inspected


def test_inspect_qmcpack_pseudopotential_rejects_incomplete_semilocal_card(tmp_path):
    path = _write(tmp_path, "<pseudo><header zval=\"6\"/></pseudo>")

    with pytest.raises(ValueError, match="missing <semilocal>"):
        inspect_qmcpack_pseudopotential(path)


def test_inspect_qmcpack_pseudopotential_reports_structural_mismatches(tmp_path):
    path = _write(tmp_path, PSEUDOPOTENTIAL.replace(
        'units="hartree" format="r*V" npots-down="2" npots-up="0" l-local="1"',
        'units="rydberg" format="V" npots-down="2" npots-up="0" l-local="2"',
    ).replace(
        'type="linear" units="bohr" ri="0" rf="10" npts="5"/><data>0 -2 -4 -6 -6',
        'type="log" units="bohr" ri="0" rf="10" npts="4"/><data>0 -2 -4 -6 -6',
        1,
    ))

    inspected = inspect_qmcpack_pseudopotential(path)

    assert inspected["structural_evidence"] == {
        "units_are_hartree": False,
        "format_is_r_times_v": False,
        "all_channel_grids_are_linear": False,
        "all_declared_grid_counts_match_data": False,
        "local_channel_has_vps": False,
        "channel_labels_are_recognized": True,
        "channel_spin_pairs_are_unique": True,
    }
    assert inspected["warnings"] == [
        "<semilocal> units are not 'hartree'.",
        "<semilocal> format is not 'r*V'.",
        "At least one <vps> channel does not declare a linear grid.",
        "At least one <vps> grid npts value does not match its data count.",
        "The declared l-local channel has no matching <vps> channel.",
    ]


def test_inspect_qmcpack_pseudopotential_rejects_ambiguous_channels(tmp_path):
    path = _write(tmp_path, PSEUDOPOTENTIAL.replace(
        'l="p" spin="-1" cutoff="1.4"',
        'l="s" spin="-1" cutoff="1.4"',
    ))

    inspected = inspect_qmcpack_pseudopotential(path)

    assert inspected["structural_evidence"]["channel_spin_pairs_are_unique"] is False
    assert inspected["warnings"][-1] == (
        "Multiple <vps> channels use the same angular momentum and spin."
    )


def test_inspect_qmcpack_pseudopotential_rejects_unknown_channel_label(tmp_path):
    path = _write(tmp_path, PSEUDOPOTENTIAL.replace('l="p"', 'l="h"'))

    inspected = inspect_qmcpack_pseudopotential(path)

    assert inspected["structural_evidence"]["channel_labels_are_recognized"] is False
    assert "At least one <vps> channel label is not a supported angular momentum." in (
        inspected["warnings"]
    )


def test_referenced_pseudopotentials_check_declared_element_against_header(tmp_path):
    pseudopotential = _write(tmp_path)
    input_path = tmp_path / "qmc.xml"
    input_path.write_text("""\
<simulation>
  <hamiltonian target="e">
    <pairpot type="pseudo"><pseudo elementType="O" href="O.xml"/></pairpot>
  </hamiltonian>
</simulation>
""", encoding="utf-8")
    parsed_input = parse_qmcpack_input(input_path)
    include_review = inspect_xml_includes(input_path, parsed_input)

    inspected = inspect_referenced_pseudopotentials(
        parsed_input,
        include_review,
        input_path,
    )

    assert inspected["status"] == "pass"
    assert inspected["observed"]["inspections"][0]["reference_identity"] == {
        "status": "pass",
        "declared_elements": ["O"],
        "header_symbol": "O",
        "matches_header_symbol": True,
    }
    tool_response = _handle_inspect_qmcpack_referenced_pseudopotentials({
        "qmcpack_input": str(input_path),
    })
    assert tool_response["status"] == "pass"
    assert tool_response["inspection"] == inspected

    input_path.write_text(
        input_path.read_text(encoding="utf-8").replace('elementType="O"', 'elementType="F"'),
        encoding="utf-8",
    )
    mismatch = _handle_inspect_qmcpack_referenced_pseudopotentials({
        "qmcpack_input": str(input_path),
    })
    assert mismatch["status"] == "not_ready"
    assert mismatch["inspection"]["observed"]["inspections"][0]["reference_identity"] == {
        "status": "not_ready",
        "declared_elements": ["F"],
        "header_symbol": "O",
        "matches_header_symbol": False,
    }

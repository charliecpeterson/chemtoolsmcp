"""Molcas task selection and scientific analysis stay outside MCP handlers."""

import pytest

from chemtools.mcp.tools import molcas
from chemtools.programs.molcas.parse import geometry, rassi


def test_molcas_inspection_handlers_only_translate_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(
        molcas,
        "_get_orbitals",
        lambda path, task_index: calls.append(
            ("orbitals", path, task_index)
        ) or {"kind": "orbitals"},
    )
    monkeypatch.setattr(
        molcas,
        "_analyze_active_space_source",
        lambda **kwargs: calls.append(("active_space", kwargs))
        or {"kind": "active_space"},
    )
    monkeypatch.setattr(
        molcas,
        "_validate_caspt2_output",
        lambda path: calls.append(("caspt2", path))
        or {"kind": "caspt2"},
    )
    monkeypatch.setattr(
        molcas,
        "_suggest_orbital_swaps_from_output",
        lambda path, **kwargs: calls.append(("swaps", path, kwargs))
        or {"kind": "swaps"},
    )

    assert molcas._handle_get_molcas_orbitals({
        "output_file": "run.out",
        "task_index": 3,
    }) == {"kind": "orbitals"}
    assert molcas._handle_analyze_molcas_active_space({
        "output_file": "run.out",
    }) == {"kind": "active_space"}
    assert molcas._handle_validate_molcas_caspt2_setup({
        "output_file": "run.out",
    }) == {"kind": "caspt2"}
    assert molcas._handle_suggest_molcas_orbital_swaps({
        "output_file": "run.out",
        "target_atom_pattern": "Fe",
        "target_ao_pattern": "3d",
    }) == {"kind": "swaps"}
    assert calls == [
        ("orbitals", "run.out", 3),
        (
            "active_space",
            {"output_file": "run.out", "orbital_file": None},
        ),
        ("caspt2", "run.out"),
        (
            "swaps",
            "run.out",
            {
                "target_atom_pattern": "Fe",
                "target_ao_pattern": "3d",
                "symmetry": 1,
                "top_dominant_aos": 1,
            },
        ),
    ]


def test_molcas_geometry_selection_owns_block_choice(monkeypatch):
    blocks = [
        {"atoms": [{"symbol": "H"}], "units": "angstrom"},
        {"atoms": [{"symbol": "O"}], "units": "angstrom"},
    ]
    monkeypatch.setattr(geometry, "parse_cartesian_blocks", lambda text: blocks)
    monkeypatch.setattr(
        geometry,
        "parse_final_geometry",
        lambda text: {"atoms": [{"symbol": "N"}]},
    )

    assert geometry.select_geometry("output") == {
        "atoms": [{"symbol": "N"}],
    }
    assert geometry.select_geometry("output", 1) == blocks[1]
    with pytest.raises(geometry.GeometryBlockIndexError) as caught:
        geometry.select_geometry("output", 2)
    assert caught.value.block_count == 2


def test_molcas_rassi_module_selection_bounds_parser_input(monkeypatch):
    parsed_text = []
    monkeypatch.setattr(
        rassi,
        "parse_rassi",
        lambda text: parsed_text.append(text) or {"states": 2},
    )
    output = (
        "before\n"
        "--- Start Module: rassi\n"
        "state table\n"
        "--- Stop Module: rassi\n"
        "after\n"
    )

    assert rassi.parse_rassi_module(output) == {"states": 2}
    assert parsed_text == [
        "--- Start Module: rassi\nstate table\n"
    ]
    assert rassi.parse_rassi_module("no module") is None

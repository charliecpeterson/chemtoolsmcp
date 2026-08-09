"""Application and guided-tool contracts for chemistry input drafting."""

from __future__ import annotations

import pytest

from chemtools.application.input_drafting import InputDraftError, draft_input
from chemtools.core import registry
from chemtools.mcp.catalog import (
    BUILTIN_BACKENDS,
    load_backend,
    register_builtin_backends,
)
from chemtools.mcp.tools import guided


H2_SPEC = {
    "atoms": [
        {"element": "H", "x": 0.0, "y": 0.0, "z": 0.0},
        {"element": "H", "x": 0.0, "y": 0.0, "z": 0.74},
    ],
    "charge": 0,
    "multiplicity": 1,
    "method": "HF",
    "basis": "STO-3G",
    "task": "energy",
    "title": "H2",
}


def test_nwchem_draft_is_deterministic_and_checked():
    backend = load_backend(BUILTIN_BACKENDS[0])

    first = draft_input(backend, H2_SPEC)
    second = draft_input(backend, H2_SPEC)

    assert first["schema_version"] == "chemtools.draft-input/1"
    assert first["program"] == {"name": "nwchem"}
    assert first["assessment"]["verdict"] == {
        "label": "draft_ready",
        "confidence": 0.8,
        "reasons": [
            "The input was rendered and the configured linter found no "
            "errors or warnings."
        ],
    }
    assert first["evidence"]["lint"]["status"] == "completed"
    assert first["evidence"]["lint"]["summary"] == {
        "errors": 0,
        "warnings": 0,
        "info": 2,
    }
    assert [
        issue["message"]
        for issue in first["evidence"]["lint"]["issues"]
    ] == [
        "Multiplicity is not explicitly set in the input.",
        "Basis block contains manual basis data; library validation was skipped.",
    ]
    text = first["evidence"]["rendered_input"]["text"]
    assert text == second["evidence"]["rendered_input"]["text"]
    assert "start chemtools_job\n" in text
    assert "vectors output chemtools_job.movecs" in text
    assert text.endswith("task scf energy\n")
    assert first["next_actions"] == [{
        "action": "save_input",
        "reason": "Save the rendered text to a new input file after review.",
        "priority": 1,
    }]


def test_molcas_draft_uses_the_same_application_contract():
    drafted = draft_input(load_backend(BUILTIN_BACKENDS[1]), H2_SPEC)

    assert drafted["program"] == {"name": "molcas"}
    assert drafted["assessment"]["verdict"]["label"] == "draft_ready"
    assert drafted["evidence"]["request"] == {
        "atom_count": 2,
        "charge": 0,
        "multiplicity": 1,
        "method": "HF",
        "basis": "STO-3G",
        "task": "energy",
        "functional": None,
        "geometry_units": "angstrom",
        "program_option_keys": [],
    }
    text = drafted["evidence"]["rendered_input"]["text"]
    assert "&SEWARD &END" in text
    assert "H.STO-3G...1s." in text
    assert "&SCF &END" in text


def test_feo_quintet_draft_preserves_the_pinned_state_specification():
    drafted = draft_input(
        load_backend(BUILTIN_BACKENDS[0]),
        {
            "atoms": [
                {"element": "Fe", "x": 0.0, "y": 0.0, "z": 0.0},
                {"element": "O", "x": 0.0, "y": 0.0, "z": 1.62},
            ],
            "charge": 0,
            "multiplicity": 5,
            "method": "DFT",
            "basis": "def2-TZVP",
            "task": "energy",
            "functional": "b3lyp",
            "title": "FeO quintet",
            "program_options": {
                "start_name": "feo_quintet",
                "module_settings": [
                    "grid xfine",
                    "direct",
                    "iterations 100",
                ],
            },
        },
    )

    assert drafted["assessment"]["verdict"]["label"] == "draft_ready"
    assert drafted["evidence"]["request"] == {
        "atom_count": 2,
        "charge": 0,
        "multiplicity": 5,
        "method": "DFT",
        "basis": "def2-TZVP",
        "task": "energy",
        "functional": "b3lyp",
        "geometry_units": "angstrom",
        "program_option_keys": ["module_settings", "start_name"],
    }
    text = drafted["evidence"]["rendered_input"]["text"]
    assert "start feo_quintet" in text
    assert "  O   0.00000000  0.00000000  1.62000000" in text
    assert "  xc b3lyp" in text
    assert "  mult 5" in text
    assert "  grid xfine" in text
    assert "  direct" in text
    assert "  iterations 100" in text
    assert "vectors output feo_quintet.movecs" in text
    assert text.endswith("task dft energy\n")


def test_nwchem_draft_converts_bohr_and_maps_frequency_operation():
    specification = {
        **H2_SPEC,
        "geometry_units": "bohr",
        "task": "frequency",
        "atoms": [
            {"element": "H", "x": 0.0, "y": 0.0, "z": 0.0},
            {"element": "H", "x": 0.0, "y": 0.0, "z": 1.4},
        ],
    }

    drafted = draft_input(
        load_backend(BUILTIN_BACKENDS[0]),
        specification,
    )

    text = drafted["evidence"]["rendered_input"]["text"]
    assert "  H   0.00000000  0.00000000  0.74084810" in text
    assert text.endswith("task scf freq\n")


@pytest.mark.parametrize(
    ("program_index", "specification", "message"),
    [
        (
            0,
            {**H2_SPEC, "solvent": {"model": "cosmo"}},
            "NWChem InputSpec solvent rendering is not implemented",
        ),
        (
            0,
            {**H2_SPEC, "program_options": {"made_up": True}},
            "Unsupported NWChem program_options: made_up",
        ),
        (
            1,
            {**H2_SPEC, "task": "optimize"},
            "molcas guided drafting supports tasks: energy",
        ),
        (
            1,
            {**H2_SPEC, "ecp": {"H": "example"}},
            "OpenMolcas InputSpec ECP rendering is not implemented",
        ),
        (
            0,
            {**H2_SPEC, "method": "DFT"},
            "DFT input drafting requires a functional",
        ),
        (
            1,
            {**H2_SPEC, "functional": "b3lyp"},
            "functional is accepted only for DFT methods",
        ),
    ],
)
def test_draft_input_rejects_fields_the_backend_would_ignore(
    program_index,
    specification,
    message,
):
    with pytest.raises(InputDraftError, match=message):
        draft_input(
            load_backend(BUILTIN_BACKENDS[program_index]),
            specification,
        )


def test_guided_draft_input_reports_provider_validation_error():
    if not registry.has("nwchem"):
        register_builtin_backends()

    response = guided._handle_draft_input({
        **H2_SPEC,
        "program": "nwchem",
        "method": "unsupported-method",
    })

    assert response["error"] == "invalid_input_specification"
    assert response["program"] == "nwchem"
    assert "nwchem guided drafting supports methods: dft, hf, scf" in (
        response["message"]
    )


def test_nwchem_open_shell_hf_draft_sets_rohf_occupation():
    drafted = draft_input(
        load_backend(BUILTIN_BACKENDS[0]),
        {**H2_SPEC, "multiplicity": 3},
    )

    text = drafted["evidence"]["rendered_input"]["text"]
    assert "scf\n  rohf\n  nopen 2" in text


def test_guided_draft_input_validates_inline_geometry_before_provider():
    if not registry.has("nwchem"):
        register_builtin_backends()

    response = guided._handle_draft_input({
        **H2_SPEC,
        "program": "nwchem",
        "atoms": [],
    })

    assert response == {
        "error": "invalid_input_specification",
        "message": "atoms must contain between 1 and 2048 entries",
        "program": "nwchem",
    }


def test_guided_draft_input_uses_application_contract():
    if not registry.has("nwchem"):
        register_builtin_backends()

    response = guided._handle_draft_input({
        **H2_SPEC,
        "program": "nwchem",
    })

    assert response["schema_version"] == "chemtools.draft-input/1"
    assert response["program"] == {"name": "nwchem"}
    assert response["assessment"]["verdict"]["label"] == "draft_ready"
    assert response["evidence"]["rendered_input"]["text"].endswith(
        "task scf energy\n"
    )


def test_guided_draft_input_refuses_backend_without_draft_capability():
    if not registry.has("qe"):
        register_builtin_backends()

    response = guided._handle_draft_input({
        **H2_SPEC,
        "program": "qe",
    })

    assert response == {
        "error": "unsupported_capability",
        "message": "'qe' does not support input drafting",
        "program": "qe",
    }

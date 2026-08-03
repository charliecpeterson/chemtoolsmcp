"""Exact guided input-review contracts across current program capabilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from chemtools.application.input_review import (
    InputReviewError,
    detect_input_content_candidates,
    detect_input_backend,
    review_input,
)
from chemtools.core import registry
from chemtools.mcp.catalog import (
    BUILTIN_BACKENDS,
    load_backend,
    register_builtin_backends,
)
from chemtools.mcp.tools import guided


FIXTURES = Path(__file__).parent / "golden" / "mcp" / "fixtures"
BACKENDS = tuple(load_backend(spec) for spec in BUILTIN_BACKENDS)


def test_review_input_combines_nwchem_parse_and_lint_evidence():
    path = FIXTURES / "nwchem_h2.nw"

    reviewed = review_input(
        BACKENDS[0],
        path,
        resolved_by="content",
    )

    assert reviewed["schema_version"] == "chemtools.review-input/1"
    assert reviewed["program"] == {
        "name": "nwchem",
        "resolved_by": "content",
    }
    assert reviewed["assessment"] == {
        "verdict": {
            "label": "review_required",
            "confidence": 0.8,
            "reasons": ["The configured linter found 1 warning(s)."],
        },
    }
    assert reviewed["evidence"]["parser"]["status"] == "completed"
    assert reviewed["evidence"]["parser"]["result"]["atom_count"] == 2
    assert reviewed["evidence"]["parser"]["result"]["tasks"] == [{
        "module": "scf",
        "operation": "energy",
    }]
    assert reviewed["evidence"]["parser"]["result"]["multiplicity"] == 1
    assert reviewed["evidence"]["lint"]["summary"] == {
        "errors": 0,
        "warnings": 1,
        "info": 2,
    }
    assert reviewed["next_actions"] == [{
        "action": "edit_input",
        "path": str(path.resolve()),
        "line": None,
        "suggested_fix": None,
        "reason": "Module 'scf' does not explicitly write a movecs file.",
        "priority": 2,
    }]
    assert reviewed["uncertainty"] == []


def test_review_input_runs_molcas_lint_without_claiming_parse_support():
    path = FIXTURES / "molcas_scf.input"

    reviewed = review_input(
        BACKENDS[1],
        path,
        resolved_by="extension",
    )

    assert reviewed["program"] == {
        "name": "molcas",
        "resolved_by": "extension",
    }
    assert reviewed["assessment"] == {
        "verdict": {
            "label": "checks_passed",
            "confidence": 0.8,
            "reasons": [
                "The configured linter found no errors or warnings."
            ],
        },
    }
    assert reviewed["evidence"]["parser"] == {
        "status": "unsupported",
        "result": None,
    }
    assert reviewed["evidence"]["lint"] == {
        "status": "completed",
        "summary": {"errors": 0, "warnings": 0, "info": 0},
        "issues": [],
    }
    assert reviewed["uncertainty"] == [{
        "code": "input_parser_unavailable",
        "message": "molcas has no declared input parser.",
        "impact": "The review cannot report a normalized input summary.",
    }]
    assert reviewed["next_actions"] == []


def test_review_input_returns_exact_molcas_error_fixes(tmp_path):
    path = tmp_path / "broken.input"
    path.write_text("&RASSCF\nSpin\n1\n", encoding="utf-8")

    reviewed = review_input(
        BACKENDS[1],
        path,
        resolved_by="extension",
    )

    assert reviewed["assessment"] == {
        "verdict": {
            "label": "errors_found",
            "confidence": 0.9,
            "reasons": ["The configured linter found 2 error(s)."],
        },
    }
    assert reviewed["evidence"]["lint"]["summary"] == {
        "errors": 2,
        "warnings": 0,
        "info": 0,
    }
    assert reviewed["next_actions"] == [
        {
            "action": "edit_input",
            "path": str(path.resolve()),
            "line": 1,
            "suggested_fix": "End of input",
            "reason": (
                "&RASSCF block at line 1 has no matching 'End of input'"
            ),
            "priority": 1,
        },
        {
            "action": "edit_input",
            "path": str(path.resolve()),
            "line": 1,
            "suggested_fix": (
                "Nactel\n"
                "   N   0   0   (replace N with active electron count)"
            ),
            "reason": "RASSCF block at line 1 has no Nactel directive",
            "priority": 1,
        },
    ]


def test_review_input_marks_dirac_as_parsed_but_unchecked():
    path = FIXTURES / "dirac_scf.inp"

    reviewed = review_input(
        BACKENDS[2],
        path,
        resolved_by="content",
    )

    assert reviewed["assessment"] == {
        "verdict": {
            "label": "parsed_unchecked",
            "confidence": 0.45,
            "reasons": [
                "The input parsed, but no declared linter reviewed it."
            ],
        },
    }
    assert reviewed["evidence"]["parser"]["status"] == "completed"
    assert reviewed["evidence"]["parser"]["result"]["has_scf"] is True
    assert reviewed["evidence"]["parser"]["result"]["has_dft"] is False
    assert reviewed["evidence"]["lint"]["status"] == "unsupported"
    assert reviewed["uncertainty"] == [{
        "code": "input_linter_unavailable",
        "message": "dirac has no declared input linter.",
        "impact": "Parsing alone does not establish input correctness.",
    }]
    assert reviewed["next_actions"] == [{
        "action": "manual_scientific_review",
        "path": str(path.resolve()),
        "reason": "dirac has no declared input linter.",
        "priority": 1,
    }]


@pytest.mark.parametrize(
    ("filename", "program", "method"),
    (
        ("nwchem_h2.nw", "nwchem", "content"),
        ("molcas_scf.input", "molcas", "content"),
        ("dirac_scf.inp", "dirac", "content"),
    ),
)
def test_input_detection_uses_exact_content_signatures(
    filename,
    program,
    method,
):
    backend, resolved_by = detect_input_backend(
        BACKENDS,
        FIXTURES / filename,
    )

    assert backend.name == program
    assert resolved_by == method
    assert detect_input_content_candidates(
        BACKENDS,
        FIXTURES / filename,
    ) == (program,)


def test_input_content_detection_reads_only_the_bounded_head(tmp_path, monkeypatch):
    path = tmp_path / "pw.in"
    path.write_text(
        "&control\n/\n&system\n/\nATOMIC_SPECIES\nH 1.0 H.upf\n",
        encoding="utf-8",
    )

    def unexpected_full_read(*args, **kwargs):
        raise AssertionError("content detection must not read the whole input")

    monkeypatch.setattr(Path, "read_text", unexpected_full_read)

    assert detect_input_content_candidates((BACKENDS[4],), path) == ("qe",)


@pytest.mark.parametrize(
    ("filename", "selected_program", "detected_program"),
    (
        ("nwchem_h2.nw", "molcas", "nwchem"),
        ("molcas_scf.input", "nwchem", "molcas"),
        ("dirac_scf.inp", "molcas", "dirac"),
    ),
)
def test_review_input_handler_rejects_conflicting_program_override(
    filename,
    selected_program,
    detected_program,
):
    if not registry.has("nwchem"):
        register_builtin_backends()

    assert guided._handle_review_input({
        "input_file": str(FIXTURES / filename),
        "program": selected_program,
    }) == {
        "error": "program_content_mismatch",
        "message": (
            f"chemistry input content matches {detected_program}, but "
            f"program override selected {selected_program}"
        ),
        "program": selected_program,
        "detected_programs": [detected_program],
    }


def test_review_input_handler_allows_detector_negative_explicit_input(tmp_path):
    path = tmp_path / "fragment.inp"
    path.write_text("unrecognized partial input\n", encoding="utf-8")
    if not registry.has("nwchem"):
        register_builtin_backends()

    reviewed = guided._handle_review_input({
        "input_file": str(path),
        "program": "dirac",
    })

    assert "error" not in reviewed
    assert reviewed["program"] == {
        "name": "dirac",
        "resolved_by": "explicit",
    }


def test_review_input_handler_detects_unsupported_qe_companion_input(tmp_path):
    path = tmp_path / "bands.in"
    path.write_text("&BANDS\n/\n", encoding="utf-8")
    if not registry.has("qe"):
        register_builtin_backends()

    reviewed = guided._handle_review_input({"input_file": str(path)})

    assert reviewed["program"] == {
        "name": "qe",
        "resolved_by": "content",
    }
    assert reviewed["assessment"]["verdict"]["label"] == "errors_found"
    assert reviewed["evidence"]["lint"] == {
        "status": "completed",
        "summary": {"errors": 1, "warnings": 0, "info": 0},
        "issues": [{
            "level": "error",
            "message": (
                "This is a Quantum ESPRESSO bands.x input; the current "
                "Chemtools QE reviewer supports pw.x inputs only."
            ),
            "line": None,
            "suggested_fix": None,
        }],
    }


def test_input_detection_refuses_ambiguous_inp_extension(tmp_path):
    path = tmp_path / "unknown.inp"
    path.write_text("unrecognized input\n", encoding="utf-8")

    with pytest.raises(InputReviewError) as caught:
        detect_input_backend(BACKENDS, path)

    assert caught.value.as_dict() == {
        "error": "program_detection_ambiguous",
        "message": (
            "input extension '.inp' matches multiple programs; "
            "pass program explicitly"
        ),
        "candidates": ["molcas", "dirac"],
    }


def test_review_input_reports_grasp_single_file_as_unsupported(tmp_path):
    path = tmp_path / "rcsfgenerate.stdin"
    path.write_text("n\nTh\n", encoding="utf-8")

    reviewed = review_input(
        BACKENDS[3],
        path,
        resolved_by="explicit",
    )

    assert reviewed["assessment"] == {
        "verdict": {
            "label": "unsupported",
            "confidence": 1.0,
            "reasons": [
                (
                    "This backend has no declared parser or linter "
                    "for this input."
                )
            ],
        },
    }
    assert [item["code"] for item in reviewed["uncertainty"]] == [
        "input_parser_unavailable",
        "input_linter_unavailable",
        "artifact_kind_unmatched",
    ]
    assert reviewed["next_actions"] == [{
        "action": "use_program_specific_input_builder",
        "path": str(path.resolve()),
        "reason": (
            "grasp does not expose a standalone input review contract."
        ),
        "priority": 1,
    }]


def test_review_input_handler_rejects_missing_source(tmp_path):
    missing = tmp_path / "missing.nw"

    assert guided._handle_review_input(
        {"input_file": str(missing)}
    ) == {
        "error": "source_not_file",
        "message": f"chemistry input is not a readable file: {missing}",
    }

"""Exact tests for bounded, backend-scoped artifact classification."""

from pathlib import Path

import pytest

from chemtools.core.artifact_classification import (
    ArtifactCandidate,
    classify_artifact,
    classify_artifacts,
)
from chemtools.core.artifacts import (
    ArtifactLocation,
    ArtifactRole,
    ExpectedArtifact,
    StepRef,
)
from chemtools.mcp.catalog import BUILTIN_BACKENDS, load_backend


BACKENDS = {
    spec.name: load_backend(spec)
    for spec in BUILTIN_BACKENDS
}


def test_artifact_candidate_defaults_legacy_constructors_to_unknown_content():
    candidate = ArtifactCandidate(
        "test.output",
        frozenset({ArtifactRole.PRIMARY_OUTPUT}),
        "inferred",
        "extension",
        ".out",
    )

    assert candidate.content_kind == "unknown"


@pytest.mark.parametrize(
    ("program", "path", "kind", "roles", "content_kind", "matched_by"),
    (
        (
            "nwchem",
            "job.movecs",
            "nwchem.movecs",
            frozenset({ArtifactRole.CHECKPOINT, ArtifactRole.ORBITAL}),
            "binary",
            "extension",
        ),
        (
            "molcas",
            "INPORB",
            "molcas.orbitals",
            frozenset({ArtifactRole.CHECKPOINT, ArtifactRole.ORBITAL}),
            "text",
            "filename",
        ),
        (
            "dirac",
            "DFCOEF",
            "dirac.orbitals",
            frozenset({ArtifactRole.CHECKPOINT, ArtifactRole.ORBITAL}),
            "binary",
            "filename",
        ),
        (
            "grasp",
            "atom.w",
            "grasp.radial_wfn",
            frozenset({ArtifactRole.ORBITAL, ArtifactRole.WAVEFUNCTION}),
            "binary",
            "extension",
        ),
        (
            "qe",
            "oxygen.pwscf.h5",
            "qe.pw2qmcpack_hdf5",
            frozenset({ArtifactRole.CHECKPOINT, ArtifactRole.WAVEFUNCTION}),
            "binary",
            "extension",
        ),
    ),
)
def test_representative_backend_artifacts_match_declared_kinds(
    program,
    path,
    kind,
    roles,
    content_kind,
    matched_by,
):
    classification = classify_artifact(BACKENDS[program], path)

    assert classification.program == program
    assert classification.path == Path(path)
    assert classification.status == "matched"
    assert len(classification.candidates) == 1
    candidate = classification.candidates[0]
    assert candidate.kind == kind
    assert candidate.roles == roles
    assert candidate.content_kind == content_kind
    assert candidate.evidence == "inferred"
    assert candidate.matched_by == matched_by
    assert candidate.producing_step is None
    assert candidate.expectation_id is None


@pytest.mark.parametrize(
    ("path", "kind", "matched_value"),
    (
        ("run.sum", "grasp.rmcdhf_summary", ".sum"),
        ("run.csum", "grasp.rci_summary", ".csum"),
        ("levels.hlsj", "grasp.hfs", ".hlsj"),
        ("shift.i", "grasp.isotope_shift", ".i"),
        ("lines.t.lsj", "grasp.transition", ".t.lsj"),
    ),
)
def test_grasp_scientific_outputs_have_one_specific_kind(
    path,
    kind,
    matched_value,
):
    classification = classify_artifact(BACKENDS["grasp"], path)

    assert classification.status == "matched"
    assert len(classification.candidates) == 1
    assert classification.candidates[0].kind == kind
    assert classification.candidates[0].matched_value == matched_value


def test_exact_expectation_overrides_suffix_ambiguity():
    step = StepRef(run_uid="run-grasp-1", step_id="rmcdhf")
    expectation = ExpectedArtifact(
        expectation_id="expected-rmcdhf-summary",
        roles=frozenset({ArtifactRole.PRIMARY_OUTPUT}),
        kind="grasp.rmcdhf_summary",
        location=ArtifactLocation(
            path=Path("run.sum"),
            entry_type="file",
        ),
        required=True,
        producing_step=step,
    )

    classification = classify_artifact(
        BACKENDS["grasp"],
        "run.sum",
        (expectation,),
    )

    assert classification.status == "matched"
    assert len(classification.candidates) == 1
    candidate = classification.candidates[0]
    assert candidate.kind == "grasp.rmcdhf_summary"
    assert candidate.content_kind == "text"
    assert candidate.evidence == "declared"
    assert candidate.matched_by == "expectation"
    assert candidate.matched_value == "run.sum"
    assert candidate.producing_step == step
    assert candidate.expectation_id == "expected-rmcdhf-summary"
    assert classification.to_dict() == {
        "program": "grasp",
        "path": "run.sum",
        "status": "matched",
        "candidates": [
            {
                "kind": "grasp.rmcdhf_summary",
                "roles": ["primary_output"],
                "content_kind": "text",
                "evidence": "declared",
                "matched_by": "expectation",
                "matched_value": "run.sum",
                "producing_step": {
                    "run_uid": "run-grasp-1",
                    "step_id": "rmcdhf",
                },
                "expectation_id": "expected-rmcdhf-summary",
            }
        ],
    }


def test_selected_backend_controls_shared_output_suffix():
    path = Path("calculation.out")

    assert classify_artifact(
        BACKENDS["nwchem"],
        path,
    ).candidates[0].kind == "nwchem.output"
    assert classify_artifact(
        BACKENDS["dirac"],
        path,
    ).candidates[0].kind == "dirac.output"


def test_unmatched_path_does_not_invent_a_kind():
    classification = classify_artifact(
        BACKENDS["molcas"],
        "unknown.checkpoint",
    )

    assert classification.status == "unmatched"
    assert classification.candidates == ()


@pytest.mark.parametrize(
    "program",
    ("nwchem", "molcas", "dirac", "grasp", "qe"),
)
def test_stderr_artifacts_have_one_backend_declared_role(program):
    classification = classify_artifact(
        BACKENDS[program],
        "calculation.err",
    )

    assert classification.status == "matched"
    assert len(classification.candidates) == 1
    assert classification.candidates[0].kind == f"{program}.error"
    assert classification.candidates[0].roles == frozenset({
        ArtifactRole.STDERR
    })
    assert classification.candidates[0].content_kind == "text"


def test_molcas_jobiph_is_separate_from_formatted_orbitals():
    classification = classify_artifact(BACKENDS["molcas"], "JOBIPH")

    assert classification.status == "matched"
    assert len(classification.candidates) == 1
    candidate = classification.candidates[0]
    assert candidate.kind == "molcas.jobiph"
    assert candidate.roles == frozenset({
        ArtifactRole.CHECKPOINT,
        ArtifactRole.ORBITAL,
        ArtifactRole.WAVEFUNCTION,
    })
    assert candidate.content_kind == "binary"
    assert candidate.matched_by == "filename"
    assert candidate.matched_value == "JOBIPH"


def test_bulk_classification_preserves_input_order_without_filesystem_access():
    paths = (
        "/does/not/exist/job.movecs",
        "/does/not/exist/job.nw",
        "/does/not/exist/unknown",
    )

    classifications = classify_artifacts(BACKENDS["nwchem"], paths)

    assert tuple(item.path for item in classifications) == tuple(
        Path(path) for path in paths
    )
    assert tuple(item.status for item in classifications) == (
        "matched",
        "matched",
        "unmatched",
    )

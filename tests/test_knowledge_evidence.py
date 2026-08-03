"""Exact provenance checks behind knowledge-card independence claims."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from chemtools.core.artifacts import RunArtifacts
from chemtools.knowledge.evidence import (
    assess_direct_producer_independence,
    assess_starting_guess_class_diversity,
)


FIXTURES = (
    Path(__file__).parent / "fixtures" / "knowledge" / "evidence_independence"
)


def _collection(filename: str) -> RunArtifacts:
    payload = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
    return RunArtifacts.from_dict(payload)


def test_same_producer_outputs_fail_independence_check():
    assessment = assess_direct_producer_independence(
        _collection("shared_producer.json"),
        ("artifact-level-count", "artifact-block-count"),
    )

    assert assessment.to_dict() == {
        "artifact_ids": ["artifact-level-count", "artifact-block-count"],
        "verdict": "correlated",
        "producer_event_ids": [
            "event-generate-level-count",
            "event-generate-block-count",
        ],
        "producer_refs": [
            "step:run-shared-producer/generate-manifold",
            "step:run-shared-producer/generate-manifold",
        ],
        "shared_producer_refs": [
            "step:run-shared-producer/generate-manifold"
        ],
        "reasons": [
            "Compared artifacts share a direct producer step or event and "
            "cannot independently validate that producer."
        ],
    }


def test_distinct_recorded_producers_pass_narrow_independence_check():
    assessment = assess_direct_producer_independence(
        _collection("distinct_producers.json"),
        (
            "artifact-relativistic-ordering",
            "artifact-nonrelativistic-ordering",
        ),
    )

    assert assessment.to_dict() == {
        "artifact_ids": [
            "artifact-relativistic-ordering",
            "artifact-nonrelativistic-ordering",
        ],
        "verdict": "independent",
        "producer_event_ids": [
            "event-grasp-ordering",
            "event-atsp2k-ordering",
        ],
        "producer_refs": [
            "step:run-distinct-producers/grasp-ordering",
            "step:run-distinct-producers/atsp2k-ordering",
        ],
        "shared_producer_refs": [],
        "reasons": [
            "Compared artifacts have distinct recorded direct producers. "
            "This does not establish broader method or data independence."
        ],
    }


def test_inferred_distinct_producer_abstains():
    collection = _collection("distinct_producers.json")
    collection = replace(
        collection,
        provenance=(
            collection.provenance[0],
            replace(collection.provenance[1], evidence="inferred"),
        ),
    )

    assessment = assess_direct_producer_independence(
        collection,
        (
            "artifact-relativistic-ordering",
            "artifact-nonrelativistic-ordering",
        ),
    )

    assert assessment.verdict == "not_established"
    assert assessment.reasons == (
        "Distinct direct producers are not all backed by recorded provenance: "
        "event-atsp2k-ordering.",
    )


def test_missing_direct_producer_abstains():
    collection = _collection("distinct_producers.json")
    collection = replace(
        collection,
        provenance=(collection.provenance[0],),
    )

    assessment = assess_direct_producer_independence(
        collection,
        (
            "artifact-relativistic-ordering",
            "artifact-nonrelativistic-ordering",
        ),
    )

    assert assessment.verdict == "not_established"
    assert assessment.reasons == (
        "artifact-nonrelativistic-ordering has no direct producer record.",
    )


def test_multiple_direct_producers_abstain():
    collection = _collection("distinct_producers.json")
    duplicate = replace(
        collection.provenance[0],
        event_id="event-grasp-ordering-duplicate",
        step_id="grasp-ordering-duplicate",
    )
    collection = replace(
        collection,
        provenance=(*collection.provenance, duplicate),
    )

    assessment = assess_direct_producer_independence(
        collection,
        (
            "artifact-relativistic-ordering",
            "artifact-nonrelativistic-ordering",
        ),
    )

    assert assessment.verdict == "not_established"
    assert assessment.reasons == (
        "artifact-relativistic-ordering has 2 direct producer records.",
    )


def test_same_starting_guess_class_counts_as_one_measurement():
    assessment = assess_starting_guess_class_diversity(
        _collection("same_starting_guess_class.json"),
        ("artifact-grasp-tf-run-a", "artifact-grasp-tf-run-b"),
    )

    assert assessment.to_dict() == {
        "artifact_ids": [
            "artifact-grasp-tf-run-a",
            "artifact-grasp-tf-run-b",
        ],
        "verdict": "same_recorded_class",
        "producer_event_ids": [
            "event-grasp-tf-run-a",
            "event-grasp-tf-run-b",
        ],
        "starting_guess_classes": ["thomas_fermi", "thomas_fermi"],
        "repeated_classes": ["thomas_fermi"],
        "reasons": [
            "Compared runs reuse starting-guess class(es): thomas_fermi. "
            "Their agreement counts as one starting-class measurement."
        ],
    }


def test_distinct_starting_guess_classes_pass_class_diversity_check():
    assessment = assess_starting_guess_class_diversity(
        _collection("distinct_producers.json"),
        (
            "artifact-relativistic-ordering",
            "artifact-nonrelativistic-ordering",
        ),
    )

    assert assessment.to_dict() == {
        "artifact_ids": [
            "artifact-relativistic-ordering",
            "artifact-nonrelativistic-ordering",
        ],
        "verdict": "distinct_recorded_classes",
        "producer_event_ids": [
            "event-grasp-ordering",
            "event-atsp2k-ordering",
        ],
        "starting_guess_classes": [
            "thomas_fermi",
            "nonrelativistic_hartree_fock",
        ],
        "repeated_classes": [],
        "reasons": [
            "Compared runs record distinct starting-guess classes. This "
            "checks class diversity, not scientific correctness."
        ],
    }


@pytest.mark.parametrize(
    "parameters",
    ({}, {"starting_guess_class": "Thomas-Fermi"}),
)
def test_missing_or_invalid_starting_guess_class_abstains(parameters):
    collection = _collection("distinct_producers.json")
    collection = replace(
        collection,
        provenance=(
            replace(collection.provenance[0], parameters=parameters),
            collection.provenance[1],
        ),
    )

    assessment = assess_starting_guess_class_diversity(
        collection,
        (
            "artifact-relativistic-ordering",
            "artifact-nonrelativistic-ordering",
        ),
    )

    assert assessment.verdict == "not_established"
    assert assessment.starting_guess_classes == (
        None,
        "nonrelativistic_hartree_fock",
    )
    assert assessment.reasons == (
        "Recorded starting_guess_class is missing or invalid for: "
        "artifact-relativistic-ordering.",
    )


def test_shared_producer_cannot_establish_starting_class_diversity():
    collection = _collection("shared_producer.json")
    collection = replace(
        collection,
        provenance=(
            replace(
                collection.provenance[0],
                parameters={"starting_guess_class": "thomas_fermi"},
            ),
            replace(
                collection.provenance[1],
                parameters={
                    "starting_guess_class": "screened_hydrogenic",
                },
            ),
        ),
    )

    assessment = assess_starting_guess_class_diversity(
        collection,
        ("artifact-level-count", "artifact-block-count"),
    )

    assert assessment.verdict == "not_established"
    assert assessment.reasons == (
        "Starting-guess class independence requires distinct direct producers; "
        "shared producer(s): step:run-shared-producer/generate-manifold.",
    )


def test_inferred_starting_guess_class_abstains():
    collection = _collection("distinct_producers.json")
    collection = replace(
        collection,
        provenance=(
            collection.provenance[0],
            replace(collection.provenance[1], evidence="inferred"),
        ),
    )

    assessment = assess_starting_guess_class_diversity(
        collection,
        (
            "artifact-relativistic-ordering",
            "artifact-nonrelativistic-ordering",
        ),
    )

    assert assessment.verdict == "not_established"
    assert assessment.starting_guess_classes == (None, None)
    assert assessment.reasons == (
        "Starting-guess classes require recorded provenance; non-recorded "
        "events: event-atsp2k-ordering.",
    )


@pytest.mark.parametrize(
    ("artifact_ids", "message"),
    (
        (("artifact-level-count",), "at least two artifacts"),
        (
            ("artifact-level-count", "artifact-level-count"),
            "must not contain duplicates",
        ),
        (
            ("artifact-level-count", "artifact-missing"),
            "unknown artifact IDs",
        ),
    ),
)
def test_direct_producer_check_rejects_invalid_comparisons(
    artifact_ids,
    message,
):
    with pytest.raises(ValueError, match=message):
        assess_direct_producer_independence(
            _collection("shared_producer.json"),
            artifact_ids,
        )

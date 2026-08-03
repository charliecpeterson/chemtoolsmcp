"""Deterministic provenance checks used by curated knowledge cards.

Direct producer records can disprove independence. They establish only the
narrow fact that compared artifacts came from distinct recorded steps or
events.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from chemtools.core.artifacts import ProvenanceRecord, RunArtifacts


DirectProducerVerdict = Literal[
    "independent",
    "correlated",
    "not_established",
]
StartingGuessClassVerdict = Literal[
    "distinct_recorded_classes",
    "same_recorded_class",
    "not_established",
]
_STARTING_GUESS_CLASS_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class _DirectProducerResolution:
    artifact_ids: tuple[str, ...]
    events: tuple[ProvenanceRecord | None, ...]
    event_ids: tuple[str | None, ...]
    producer_refs: tuple[str | None, ...]
    problems: tuple[str, ...]


@dataclass(frozen=True)
class DirectProducerAssessment:
    artifact_ids: tuple[str, ...]
    verdict: DirectProducerVerdict
    producer_event_ids: tuple[str | None, ...]
    producer_refs: tuple[str | None, ...]
    shared_producer_refs: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_ids": list(self.artifact_ids),
            "verdict": self.verdict,
            "producer_event_ids": list(self.producer_event_ids),
            "producer_refs": list(self.producer_refs),
            "shared_producer_refs": list(self.shared_producer_refs),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class StartingGuessClassAssessment:
    artifact_ids: tuple[str, ...]
    verdict: StartingGuessClassVerdict
    producer_event_ids: tuple[str | None, ...]
    starting_guess_classes: tuple[str | None, ...]
    repeated_classes: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_ids": list(self.artifact_ids),
            "verdict": self.verdict,
            "producer_event_ids": list(self.producer_event_ids),
            "starting_guess_classes": list(self.starting_guess_classes),
            "repeated_classes": list(self.repeated_classes),
            "reasons": list(self.reasons),
        }


def assess_direct_producer_independence(
    collection: RunArtifacts,
    artifact_ids: tuple[str, ...],
) -> DirectProducerAssessment:
    resolution = _resolve_direct_producers(collection, artifact_ids)
    if resolution.problems:
        return DirectProducerAssessment(
            artifact_ids=resolution.artifact_ids,
            verdict="not_established",
            producer_event_ids=resolution.event_ids,
            producer_refs=resolution.producer_refs,
            shared_producer_refs=(),
            reasons=resolution.problems,
        )

    resolved_refs = [
        ref for ref in resolution.producer_refs if ref is not None
    ]
    shared_producer_refs = tuple(sorted({
        ref
        for ref in resolved_refs
        if resolved_refs.count(ref) > 1
    }))
    if shared_producer_refs:
        return DirectProducerAssessment(
            artifact_ids=resolution.artifact_ids,
            verdict="correlated",
            producer_event_ids=resolution.event_ids,
            producer_refs=resolution.producer_refs,
            shared_producer_refs=shared_producer_refs,
            reasons=(
                "Compared artifacts share a direct producer step or event "
                "and cannot independently validate that producer.",
            ),
        )

    non_recorded = tuple(
        event.event_id
        for event in resolution.events
        if event is not None and event.evidence != "recorded"
    )
    if non_recorded:
        return DirectProducerAssessment(
            artifact_ids=resolution.artifact_ids,
            verdict="not_established",
            producer_event_ids=resolution.event_ids,
            producer_refs=resolution.producer_refs,
            shared_producer_refs=(),
            reasons=(
                "Distinct direct producers are not all backed by recorded "
                f"provenance: {', '.join(non_recorded)}.",
            ),
        )

    return DirectProducerAssessment(
        artifact_ids=resolution.artifact_ids,
        verdict="independent",
        producer_event_ids=resolution.event_ids,
        producer_refs=resolution.producer_refs,
        shared_producer_refs=(),
        reasons=(
            "Compared artifacts have distinct recorded direct producers. "
            "This does not establish broader method or data independence.",
        ),
    )


def assess_starting_guess_class_diversity(
    collection: RunArtifacts,
    artifact_ids: tuple[str, ...],
) -> StartingGuessClassAssessment:
    resolution = _resolve_direct_producers(collection, artifact_ids)
    if resolution.problems:
        return StartingGuessClassAssessment(
            artifact_ids=resolution.artifact_ids,
            verdict="not_established",
            producer_event_ids=resolution.event_ids,
            starting_guess_classes=tuple(None for _ in resolution.events),
            repeated_classes=(),
            reasons=resolution.problems,
        )

    resolved_refs = [
        ref for ref in resolution.producer_refs if ref is not None
    ]
    shared_producer_refs = tuple(sorted({
        ref
        for ref in resolved_refs
        if resolved_refs.count(ref) > 1
    }))
    if shared_producer_refs:
        return StartingGuessClassAssessment(
            artifact_ids=resolution.artifact_ids,
            verdict="not_established",
            producer_event_ids=resolution.event_ids,
            starting_guess_classes=tuple(None for _ in resolution.events),
            repeated_classes=(),
            reasons=(
                "Starting-guess class independence requires distinct direct "
                "producers; shared producer(s): "
                f"{', '.join(shared_producer_refs)}.",
            ),
        )

    non_recorded = tuple(
        event.event_id
        for event in resolution.events
        if event is not None and event.evidence != "recorded"
    )
    if non_recorded:
        return StartingGuessClassAssessment(
            artifact_ids=resolution.artifact_ids,
            verdict="not_established",
            producer_event_ids=resolution.event_ids,
            starting_guess_classes=tuple(None for _ in resolution.events),
            repeated_classes=(),
            reasons=(
                "Starting-guess classes require recorded provenance; "
                f"non-recorded events: {', '.join(non_recorded)}.",
            ),
        )

    guess_classes = tuple(
        _starting_guess_class(event)
        if event is not None
        else None
        for event in resolution.events
    )
    missing = tuple(
        artifact_id
        for artifact_id, guess_class in zip(
            resolution.artifact_ids,
            guess_classes,
        )
        if guess_class is None
    )
    if missing:
        return StartingGuessClassAssessment(
            artifact_ids=resolution.artifact_ids,
            verdict="not_established",
            producer_event_ids=resolution.event_ids,
            starting_guess_classes=guess_classes,
            repeated_classes=(),
            reasons=(
                "Recorded starting_guess_class is missing or invalid for: "
                f"{', '.join(missing)}.",
            ),
        )

    resolved_classes = [
        guess_class
        for guess_class in guess_classes
        if guess_class is not None
    ]
    repeated_classes = tuple(sorted({
        guess_class
        for guess_class in resolved_classes
        if resolved_classes.count(guess_class) > 1
    }))
    if repeated_classes:
        return StartingGuessClassAssessment(
            artifact_ids=resolution.artifact_ids,
            verdict="same_recorded_class",
            producer_event_ids=resolution.event_ids,
            starting_guess_classes=guess_classes,
            repeated_classes=repeated_classes,
            reasons=(
                "Compared runs reuse starting-guess class(es): "
                f"{', '.join(repeated_classes)}. Their agreement counts as "
                "one starting-class measurement.",
            ),
        )

    return StartingGuessClassAssessment(
        artifact_ids=resolution.artifact_ids,
        verdict="distinct_recorded_classes",
        producer_event_ids=resolution.event_ids,
        starting_guess_classes=guess_classes,
        repeated_classes=(),
        reasons=(
            "Compared runs record distinct starting-guess classes. This "
            "checks class diversity, not scientific correctness.",
        ),
    )


def _resolve_direct_producers(
    collection: RunArtifacts,
    artifact_ids: tuple[str, ...],
) -> _DirectProducerResolution:
    compared = tuple(artifact_ids)
    if len(compared) < 2:
        raise ValueError("at least two artifacts are required")
    if len(compared) != len(set(compared)):
        raise ValueError("artifact_ids must not contain duplicates")

    known_artifacts = {
        artifact.artifact_id for artifact in collection.artifacts
    }
    unknown = sorted(set(compared) - known_artifacts)
    if unknown:
        raise ValueError(f"unknown artifact IDs: {unknown}")

    events_by_artifact: dict[str, dict[str, ProvenanceRecord]] = {
        artifact_id: {} for artifact_id in compared
    }
    for event in collection.provenance:
        for snapshot in event.outputs:
            if snapshot.artifact_id in events_by_artifact:
                events_by_artifact[snapshot.artifact_id][event.event_id] = event

    producer_events: list[ProvenanceRecord | None] = []
    problems = []
    for artifact_id in compared:
        events = tuple(events_by_artifact[artifact_id].values())
        if len(events) == 1:
            producer_events.append(events[0])
        else:
            producer_events.append(None)
            problems.append(
                f"{artifact_id} has no direct producer record."
                if not events
                else f"{artifact_id} has {len(events)} direct producer records."
            )

    events_tuple = tuple(producer_events)
    return _DirectProducerResolution(
        artifact_ids=compared,
        events=events_tuple,
        event_ids=tuple(
            event.event_id if event is not None else None
            for event in events_tuple
        ),
        producer_refs=tuple(
            _producer_ref(event) if event is not None else None
            for event in events_tuple
        ),
        problems=tuple(problems),
    )


def _producer_ref(event: ProvenanceRecord) -> str:
    if event.run_uid is not None and event.step_id is not None:
        return f"step:{event.run_uid}/{event.step_id}"
    return f"event:{event.event_id}"


def _starting_guess_class(event: ProvenanceRecord) -> str | None:
    value = event.parameters.get("starting_guess_class")
    if (
        not isinstance(value, str)
        or not _STARTING_GUESS_CLASS_RE.fullmatch(value)
    ):
        return None
    return value


__all__ = [
    "DirectProducerAssessment",
    "DirectProducerVerdict",
    "StartingGuessClassAssessment",
    "StartingGuessClassVerdict",
    "assess_direct_producer_independence",
    "assess_starting_guess_class_diversity",
]

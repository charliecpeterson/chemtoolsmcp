"""Immutable artifact, observation, provenance, and freshness models.

The versioned JSON boundary keeps artifact identity separate from paths and
from the point-in-time observations used by parsers and later runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Union


JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[
    JsonScalar,
    tuple["JsonValue", ...],
    Mapping[str, "JsonValue"],
]

EntryType = Literal["file", "directory"]
HashStatus = Literal[
    "not_requested",
    "verified",
    "unavailable",
]
ProvenanceEvidence = Literal[
    "recorded",
    "declared",
    "inferred",
]
ProducerType = Literal[
    "program",
    "chemtools",
    "external_tool",
    "manual",
]
FreshnessVerdict = Literal[
    "current",
    "stale",
    "changed",
    "missing",
    "unknown",
]

RUN_ARTIFACTS_SCHEMA = "chemtools.run-artifacts/1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactRole(str, Enum):
    PRIMARY_INPUT = "primary_input"
    PRIMARY_OUTPUT = "primary_output"
    AUXILIARY_INPUT = "auxiliary_input"
    AUXILIARY_OUTPUT = "auxiliary_output"
    STDOUT = "stdout"
    STDERR = "stderr"
    CHECKPOINT = "checkpoint"
    ORBITAL = "orbital"
    WAVEFUNCTION = "wavefunction"
    WAVEFUNCTION_SEED = "wavefunction_seed"
    PSEUDOPOTENTIAL = "pseudopotential"
    VOLUMETRIC_DATA = "volumetric_data"
    SCHEDULER_SCRIPT = "scheduler_script"


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a UTC offset")


def _validate_sha256(value: str, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _freeze_json(value: Any, field_name: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} contains a non-string key")
            frozen[key] = _freeze_json(item, f"{field_name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{field_name} contains a non-JSON value of type "
        f"{type(value).__name__}"
    )


def _freeze_mapping(
    value: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, JsonValue]:
    frozen = _freeze_json(value, field_name)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return frozen


def _plain_json(value: JsonValue) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _roles(values: Any) -> frozenset[ArtifactRole]:
    roles = frozenset(ArtifactRole(value) for value in values)
    if not roles:
        raise ValueError("artifact roles must not be empty")
    return roles


def _unique_ids(values: tuple[Any, ...], field_name: str) -> None:
    identifiers = [getattr(value, field_name) for value in values]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate {field_name} values are not allowed")


@dataclass(frozen=True)
class StepRef:
    run_uid: str
    step_id: str

    def __post_init__(self) -> None:
        _require_text(self.run_uid, "run_uid")
        _require_text(self.step_id, "step_id")

    def to_dict(self) -> dict[str, str]:
        return {"run_uid": self.run_uid, "step_id": self.step_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StepRef:
        return cls(run_uid=value["run_uid"], step_id=value["step_id"])


@dataclass(frozen=True)
class ArtifactLocation:
    path: Path
    entry_type: EntryType
    root_name: str | None = None
    relative_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.entry_type not in ("file", "directory"):
            raise ValueError("entry_type must be 'file' or 'directory'")
        if self.root_name is not None:
            _require_text(self.root_name, "root_name")
        if self.relative_path is not None:
            relative_path = Path(self.relative_path)
            if relative_path.is_absolute():
                raise ValueError("relative_path must not be absolute")
            object.__setattr__(self, "relative_path", relative_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "entry_type": self.entry_type,
            "root_name": self.root_name,
            "relative_path": (
                str(self.relative_path)
                if self.relative_path is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactLocation:
        relative_path = value.get("relative_path")
        return cls(
            path=Path(value["path"]),
            entry_type=value["entry_type"],
            root_name=value.get("root_name"),
            relative_path=(
                Path(relative_path) if relative_path is not None else None
            ),
        )


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    roles: frozenset[ArtifactRole]
    kind: str
    producing_step: StepRef | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.artifact_id, "artifact_id")
        _require_text(self.kind, "kind")
        object.__setattr__(self, "roles", _roles(self.roles))
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, "metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "roles": sorted(role.value for role in self.roles),
            "kind": self.kind,
            "producing_step": (
                self.producing_step.to_dict()
                if self.producing_step is not None
                else None
            ),
            "metadata": _plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactRef:
        producing_step = value.get("producing_step")
        return cls(
            artifact_id=value["artifact_id"],
            roles=_roles(value["roles"]),
            kind=value["kind"],
            producing_step=(
                StepRef.from_dict(producing_step)
                if producing_step is not None
                else None
            ),
            metadata=value.get("metadata") or {},
        )


@dataclass(frozen=True)
class ArtifactObservation:
    observation_id: str
    artifact_id: str
    observed_at: datetime
    location: ArtifactLocation
    exists: bool
    size_bytes: int | None = None
    modified_ns: int | None = None
    sha256: str | None = None
    hash_status: HashStatus = "not_requested"
    directory_manifest_schema: str | None = None
    directory_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.observation_id, "observation_id")
        _require_text(self.artifact_id, "artifact_id")
        _require_aware_datetime(self.observed_at, "observed_at")
        if self.hash_status not in (
            "not_requested",
            "verified",
            "unavailable",
        ):
            raise ValueError("invalid hash_status")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if self.modified_ns is not None and self.modified_ns < 0:
            raise ValueError("modified_ns must be non-negative")
        if self.sha256 is not None:
            _validate_sha256(self.sha256, "sha256")
        if self.hash_status == "verified" and self.sha256 is None:
            raise ValueError("verified observations require sha256")
        if self.hash_status != "verified" and self.sha256 is not None:
            raise ValueError("sha256 requires hash_status='verified'")
        if not self.exists and any(
            value is not None
            for value in (
                self.size_bytes,
                self.modified_ns,
                self.sha256,
                self.directory_manifest_schema,
                self.directory_manifest_sha256,
            )
        ):
            raise ValueError("missing observations cannot contain file metadata")
        manifest_fields = (
            self.directory_manifest_schema,
            self.directory_manifest_sha256,
        )
        if (manifest_fields[0] is None) != (manifest_fields[1] is None):
            raise ValueError(
                "directory manifest schema and digest must be provided together"
            )
        if self.directory_manifest_schema is not None:
            if self.location.entry_type != "directory":
                raise ValueError("directory manifests require a directory location")
            _require_text(
                self.directory_manifest_schema,
                "directory_manifest_schema",
            )
            _validate_sha256(
                self.directory_manifest_sha256,
                "directory_manifest_sha256",
            )
        if self.location.entry_type == "directory" and self.sha256 is not None:
            raise ValueError(
                "directories use directory_manifest_sha256, not sha256"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "artifact_id": self.artifact_id,
            "observed_at": self.observed_at.isoformat(),
            "location": self.location.to_dict(),
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "modified_ns": self.modified_ns,
            "sha256": self.sha256,
            "hash_status": self.hash_status,
            "directory_manifest_schema": self.directory_manifest_schema,
            "directory_manifest_sha256": self.directory_manifest_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactObservation:
        return cls(
            observation_id=value["observation_id"],
            artifact_id=value["artifact_id"],
            observed_at=datetime.fromisoformat(value["observed_at"]),
            location=ArtifactLocation.from_dict(value["location"]),
            exists=value["exists"],
            size_bytes=value.get("size_bytes"),
            modified_ns=value.get("modified_ns"),
            sha256=value.get("sha256"),
            hash_status=value["hash_status"],
            directory_manifest_schema=value.get(
                "directory_manifest_schema"
            ),
            directory_manifest_sha256=value.get(
                "directory_manifest_sha256"
            ),
        )


@dataclass(frozen=True)
class ArtifactSnapshotRef:
    artifact_id: str
    observation_id: str

    def __post_init__(self) -> None:
        _require_text(self.artifact_id, "artifact_id")
        _require_text(self.observation_id, "observation_id")

    def to_dict(self) -> dict[str, str]:
        return {
            "artifact_id": self.artifact_id,
            "observation_id": self.observation_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactSnapshotRef:
        return cls(
            artifact_id=value["artifact_id"],
            observation_id=value["observation_id"],
        )


@dataclass(frozen=True)
class ProducerIdentity:
    producer_type: ProducerType
    name: str
    version: str | None = None
    commit: str | None = None

    def __post_init__(self) -> None:
        if self.producer_type not in (
            "program",
            "chemtools",
            "external_tool",
            "manual",
        ):
            raise ValueError("invalid producer_type")
        _require_text(self.name, "name")
        if self.version is not None:
            _require_text(self.version, "version")
        if self.commit is not None:
            _require_text(self.commit, "commit")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "producer_type": self.producer_type,
            "name": self.name,
            "version": self.version,
            "commit": self.commit,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProducerIdentity:
        return cls(
            producer_type=value["producer_type"],
            name=value["name"],
            version=value.get("version"),
            commit=value.get("commit"),
        )


@dataclass(frozen=True)
class ProvenanceRecord:
    event_id: str
    event_type: str
    occurred_at: datetime
    actor: ProducerIdentity
    inputs: tuple[ArtifactSnapshotRef, ...]
    outputs: tuple[ArtifactSnapshotRef, ...]
    evidence: ProvenanceEvidence
    run_uid: str | None = None
    step_id: str | None = None
    parameters: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.event_id, "event_id")
        _require_text(self.event_type, "event_type")
        _require_aware_datetime(self.occurred_at, "occurred_at")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        if not self.outputs:
            raise ValueError(
                "provenance events require at least one output snapshot"
            )
        if self.evidence not in ("recorded", "declared", "inferred"):
            raise ValueError("invalid provenance evidence")
        if self.run_uid is not None:
            _require_text(self.run_uid, "run_uid")
        if self.step_id is not None:
            _require_text(self.step_id, "step_id")
            if self.run_uid is None:
                raise ValueError("step_id requires run_uid")
        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(self.parameters, "parameters"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at.isoformat(),
            "actor": self.actor.to_dict(),
            "run_uid": self.run_uid,
            "step_id": self.step_id,
            "inputs": [snapshot.to_dict() for snapshot in self.inputs],
            "outputs": [snapshot.to_dict() for snapshot in self.outputs],
            "evidence": self.evidence,
            "parameters": _plain_json(self.parameters),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProvenanceRecord:
        return cls(
            event_id=value["event_id"],
            event_type=value["event_type"],
            occurred_at=datetime.fromisoformat(value["occurred_at"]),
            actor=ProducerIdentity.from_dict(value["actor"]),
            run_uid=value.get("run_uid"),
            step_id=value.get("step_id"),
            inputs=tuple(
                ArtifactSnapshotRef.from_dict(snapshot)
                for snapshot in value["inputs"]
            ),
            outputs=tuple(
                ArtifactSnapshotRef.from_dict(snapshot)
                for snapshot in value["outputs"]
            ),
            evidence=value["evidence"],
            parameters=value.get("parameters") or {},
        )


@dataclass(frozen=True)
class ExpectedArtifact:
    expectation_id: str
    roles: frozenset[ArtifactRole]
    kind: str
    location: ArtifactLocation
    required: bool
    producing_step: StepRef | None = None

    def __post_init__(self) -> None:
        _require_text(self.expectation_id, "expectation_id")
        _require_text(self.kind, "kind")
        object.__setattr__(self, "roles", _roles(self.roles))

    def to_dict(self) -> dict[str, Any]:
        return {
            "expectation_id": self.expectation_id,
            "roles": sorted(role.value for role in self.roles),
            "kind": self.kind,
            "location": self.location.to_dict(),
            "required": self.required,
            "producing_step": (
                self.producing_step.to_dict()
                if self.producing_step is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExpectedArtifact:
        producing_step = value.get("producing_step")
        return cls(
            expectation_id=value["expectation_id"],
            roles=_roles(value["roles"]),
            kind=value["kind"],
            location=ArtifactLocation.from_dict(value["location"]),
            required=value["required"],
            producing_step=(
                StepRef.from_dict(producing_step)
                if producing_step is not None
                else None
            ),
        )


@dataclass(frozen=True)
class FreshnessAssessment:
    verdict: FreshnessVerdict
    artifact_id: str
    observation_id: str | None
    compared_with: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.verdict not in (
            "current",
            "stale",
            "changed",
            "missing",
            "unknown",
        ):
            raise ValueError("invalid freshness verdict")
        _require_text(self.artifact_id, "artifact_id")
        if self.observation_id is not None:
            _require_text(self.observation_id, "observation_id")
        object.__setattr__(self, "compared_with", tuple(self.compared_with))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        for value in self.compared_with:
            _require_text(value, "compared_with entry")
        for value in self.evidence:
            _require_text(value, "evidence entry")
        if self.verdict == "missing" and self.observation_id is not None:
            raise ValueError("missing freshness cannot cite an observation")
        if self.verdict != "unknown" and not self.evidence:
            raise ValueError(
                "non-unknown freshness verdicts require supporting evidence"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "artifact_id": self.artifact_id,
            "observation_id": self.observation_id,
            "compared_with": list(self.compared_with),
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FreshnessAssessment:
        return cls(
            verdict=value["verdict"],
            artifact_id=value["artifact_id"],
            observation_id=value.get("observation_id"),
            compared_with=tuple(value.get("compared_with") or ()),
            evidence=tuple(value.get("evidence") or ()),
        )


@dataclass(frozen=True)
class RunArtifacts:
    run_uid: str
    artifacts: tuple[ArtifactRef, ...] = ()
    observations: tuple[ArtifactObservation, ...] = ()
    expectations: tuple[ExpectedArtifact, ...] = ()
    provenance: tuple[ProvenanceRecord, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.run_uid, "run_uid")
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "expectations", tuple(self.expectations))
        object.__setattr__(self, "provenance", tuple(self.provenance))

        _unique_ids(self.artifacts, "artifact_id")
        _unique_ids(self.observations, "observation_id")
        _unique_ids(self.expectations, "expectation_id")
        _unique_ids(self.provenance, "event_id")

        artifact_ids = {
            artifact.artifact_id for artifact in self.artifacts
        }
        observation_pairs = set()
        for observation in self.observations:
            if observation.artifact_id not in artifact_ids:
                raise ValueError(
                    f"observation {observation.observation_id!r} references "
                    f"unknown artifact {observation.artifact_id!r}"
                )
            observation_pairs.add(
                (observation.artifact_id, observation.observation_id)
            )

        for event in self.provenance:
            for snapshot in (*event.inputs, *event.outputs):
                pair = (snapshot.artifact_id, snapshot.observation_id)
                if pair not in observation_pairs:
                    raise ValueError(
                        f"provenance event {event.event_id!r} references "
                        f"unknown snapshot {pair!r}"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUN_ARTIFACTS_SCHEMA,
            "run": {"run_uid": self.run_uid},
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "observations": [
                observation.to_dict()
                for observation in self.observations
            ],
            "expectations": [
                expectation.to_dict()
                for expectation in self.expectations
            ],
            "provenance": [event.to_dict() for event in self.provenance],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunArtifacts:
        if value.get("schema") != RUN_ARTIFACTS_SCHEMA:
            raise ValueError(
                f"unsupported run-artifacts schema: {value.get('schema')!r}"
            )
        run = value.get("run")
        if not isinstance(run, Mapping):
            raise TypeError("run must be a mapping")
        return cls(
            run_uid=run["run_uid"],
            artifacts=tuple(
                ArtifactRef.from_dict(artifact)
                for artifact in value.get("artifacts") or ()
            ),
            observations=tuple(
                ArtifactObservation.from_dict(observation)
                for observation in value.get("observations") or ()
            ),
            expectations=tuple(
                ExpectedArtifact.from_dict(expectation)
                for expectation in value.get("expectations") or ()
            ),
            provenance=tuple(
                ProvenanceRecord.from_dict(event)
                for event in value.get("provenance") or ()
            ),
        )


__all__ = [
    "RUN_ARTIFACTS_SCHEMA",
    "ArtifactRole",
    "StepRef",
    "ArtifactLocation",
    "ArtifactRef",
    "ArtifactObservation",
    "ArtifactSnapshotRef",
    "ProducerIdentity",
    "ProvenanceRecord",
    "ExpectedArtifact",
    "FreshnessAssessment",
    "RunArtifacts",
]

"""Validated review ledger for unresolved f-block donor aliases.

The catalog contains logical donor labels whose exact source artifacts were
not committed. This module keeps those gaps explicit and tied to consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from chemtools.reference.fblock import (
    _exact_fields,
    _list,
    _load_json,
    _object,
    _text,
    bundled_fblock_directory,
    load_fblock_catalog,
)
from chemtools.reference.fblock_models import (
    FBlockCatalog,
    FBlockCatalogLoadError,
)


FBLOCK_DONOR_ALIAS_SCHEMA = "chemtools.fblock-donor-alias-manifest/1"
MAX_DONOR_ALIAS_MANIFEST_BYTES = 64 * 1024
_ALIAS_RE = re.compile(r"^donor_[A-Za-z0-9]+$")


@dataclass(frozen=True)
class FBlockDonorAliasRecord:
    element: str
    consumer_state: str
    alias: str
    status: str


@dataclass(frozen=True)
class FBlockDonorAliasManifest:
    dataset_id: str
    dataset_version: str
    catalog_sha256: str
    status: str
    resolution_policy: str
    unresolved_reason: str
    records: tuple[FBlockDonorAliasRecord, ...]
    schema_version: str = FBLOCK_DONOR_ALIAS_SCHEMA

    def record(
        self,
        element: str,
        consumer_state: str,
        alias: str,
    ) -> FBlockDonorAliasRecord:
        matches = tuple(
            record
            for record in self.records
            if (
                record.element,
                record.consumer_state,
                record.alias,
            ) == (element, consumer_state, alias)
        )
        if len(matches) != 1:
            raise FBlockCatalogLoadError(
                "donor alias manifest must contain one record for "
                f"{element}.{consumer_state}:{alias}"
            )
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "catalog_sha256": self.catalog_sha256,
            "record_count": len(self.records),
            "unresolved_count": len(self.records),
            "resolution_policy": self.resolution_policy,
        }


def load_fblock_donor_alias_manifest(
    directory: str | Path | None = None,
    *,
    catalog: FBlockCatalog | None = None,
) -> FBlockDonorAliasManifest:
    dataset_directory = (
        Path(directory) if directory is not None else bundled_fblock_directory()
    )
    try:
        active_catalog = catalog or load_fblock_catalog(dataset_directory)
        payload = _load_json(
            dataset_directory / "donor-aliases.json",
            MAX_DONOR_ALIAS_MANIFEST_BYTES,
        )
        manifest = _manifest(payload)
        _validate_dataset_link(manifest, active_catalog)
        _validate_record_coverage(manifest, active_catalog)
        return manifest
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        if isinstance(error, FBlockCatalogLoadError):
            raise
        raise FBlockCatalogLoadError(
            f"invalid f-block donor alias manifest at {dataset_directory}: "
            f"{error}"
        ) from error


def _manifest(value: object) -> FBlockDonorAliasManifest:
    document = _object(value, "donor alias manifest")
    _exact_fields(
        document,
        {
            "schema_version",
            "dataset",
            "status",
            "resolution_policy",
            "unresolved_reason",
            "records",
        },
        "donor alias manifest",
    )
    if document["schema_version"] != FBLOCK_DONOR_ALIAS_SCHEMA:
        raise ValueError(
            "unsupported donor alias schema "
            f"{document['schema_version']!r}"
        )
    status = _text(document["status"], "status")
    if status != "scientific_review_required":
        raise ValueError(f"unsupported donor alias manifest status {status!r}")

    dataset = _object(document["dataset"], "dataset")
    _exact_fields(dataset, {"id", "version", "catalog_sha256"}, "dataset")
    records = tuple(
        _record(item, index)
        for index, item in enumerate(_list(document["records"], "records"))
    )
    if not records:
        raise ValueError("records must not be empty")
    keys = [
        (record.element, record.consumer_state, record.alias)
        for record in records
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("records must not contain duplicate consumer aliases")

    return FBlockDonorAliasManifest(
        dataset_id=_text(dataset["id"], "dataset.id"),
        dataset_version=_text(dataset["version"], "dataset.version"),
        catalog_sha256=_text(dataset["catalog_sha256"], "dataset.catalog_sha256"),
        status=status,
        resolution_policy=_text(
            document["resolution_policy"],
            "resolution_policy",
        ),
        unresolved_reason=_text(
            document["unresolved_reason"],
            "unresolved_reason",
        ),
        records=records,
    )


def _record(value: object, index: int) -> FBlockDonorAliasRecord:
    field = f"records[{index}]"
    document = _object(value, field)
    _exact_fields(
        document,
        {"element", "consumer_state", "alias", "status"},
        field,
    )
    alias = _text(document["alias"], f"{field}.alias")
    if not _ALIAS_RE.fullmatch(alias):
        raise ValueError(f"{field}.alias has invalid syntax {alias!r}")
    status = _text(document["status"], f"{field}.status")
    if status != "unresolved":
        raise ValueError(f"{field}.status must be 'unresolved'")
    return FBlockDonorAliasRecord(
        element=_text(document["element"], f"{field}.element"),
        consumer_state=_text(
            document["consumer_state"],
            f"{field}.consumer_state",
        ),
        alias=alias,
        status=status,
    )


def _validate_dataset_link(
    manifest: FBlockDonorAliasManifest,
    catalog: FBlockCatalog,
) -> None:
    metadata = catalog.metadata
    expected = (
        metadata.dataset_id,
        metadata.dataset_version,
        metadata.catalog_sha256,
    )
    actual = (
        manifest.dataset_id,
        manifest.dataset_version,
        manifest.catalog_sha256,
    )
    if actual != expected:
        raise ValueError(
            "manifest dataset link changed: expected "
            f"{expected}, found {actual}"
        )


def _validate_record_coverage(
    manifest: FBlockDonorAliasManifest,
    catalog: FBlockCatalog,
) -> None:
    expected: set[tuple[str, str, str]] = set()
    for element in catalog.elements:
        local_states = {state.slug for state in element.states}
        for state in element.states:
            for donor in _donors(state.estimate_from):
                if donor not in local_states:
                    expected.add((element.symbol, state.slug, donor))
    actual = {
        (record.element, record.consumer_state, record.alias)
        for record in manifest.records
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "donor alias coverage changed: "
            f"missing={missing}, extra={extra}"
        )


def _donors(value: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return value


__all__ = [
    "FBLOCK_DONOR_ALIAS_SCHEMA",
    "MAX_DONOR_ALIAS_MANIFEST_BYTES",
    "FBlockDonorAliasManifest",
    "FBlockDonorAliasRecord",
    "load_fblock_donor_alias_manifest",
]

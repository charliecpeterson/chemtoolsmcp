"""Coverage and failure contracts for the f-block donor alias ledger."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from chemtools.reference import (
    FBlockCatalogLoadError,
    bundled_fblock_directory,
    load_fblock_catalog,
    load_fblock_donor_alias_manifest,
)


def test_manifest_pins_every_external_donor_occurrence():
    manifest = load_fblock_donor_alias_manifest()

    assert manifest.to_dict() == {
        "schema_version": "chemtools.fblock-donor-alias-manifest/1",
        "status": "scientific_review_required",
        "dataset_id": "fblock.atomic_seeds",
        "dataset_version": "3",
        "catalog_sha256": (
            "ba3397c7ff0634c489cc0dded2a857ef0b8151897618c7b2daa4d90502aadbec"
        ),
        "record_count": 132,
        "unresolved_count": 132,
        "resolution_policy": (
            "Resolve an alias only from a recorded donor artifact or an "
            "explicit scientific review; never infer a target from the "
            "alias spelling."
        ),
    }
    assert len({record.alias for record in manifest.records}) == 41
    assert len({record.element for record in manifest.records}) == 25
    assert Counter(record.alias for record in manifest.records).most_common(1) == [
        ("donor_closed", 15)
    ]
    assert manifest.records[0].element == "La"
    assert manifest.records[0].consumer_state == "ion2_4f1"
    assert manifest.records[0].alias == "donor_Cef1"
    assert manifest.records[-1].element == "Lr"
    assert manifest.records[-1].consumer_state == "ion0_5f147s27p1"
    assert manifest.records[-1].alias == "donor_7p"


def test_manifest_lookup_is_consumer_scoped():
    manifest = load_fblock_donor_alias_manifest()

    record = manifest.record("La", "ion2_4f1", "donor_Cef1")

    assert record.status == "unresolved"
    with pytest.raises(FBlockCatalogLoadError, match="must contain one record"):
        manifest.record("Ce", "ion2_4f1", "donor_Cef1")


def test_manifest_rejects_duplicate_records(tmp_path):
    payload = _manifest_payload()
    payload["records"].append(dict(payload["records"][0]))
    _write_manifest(tmp_path, payload)

    with pytest.raises(FBlockCatalogLoadError, match="duplicate consumer aliases"):
        load_fblock_donor_alias_manifest(
            tmp_path,
            catalog=load_fblock_catalog(),
        )


def test_manifest_rejects_unknown_fields(tmp_path):
    payload = _manifest_payload()
    payload["records"][0]["guessed_target"] = "Ce.ion3_4f1"
    _write_manifest(tmp_path, payload)

    with pytest.raises(FBlockCatalogLoadError, match="unknown fields"):
        load_fblock_donor_alias_manifest(
            tmp_path,
            catalog=load_fblock_catalog(),
        )


def test_manifest_rejects_catalog_coverage_drift(tmp_path):
    payload = _manifest_payload()
    payload["records"] = payload["records"][1:]
    _write_manifest(tmp_path, payload)

    with pytest.raises(FBlockCatalogLoadError, match="donor alias coverage changed"):
        load_fblock_donor_alias_manifest(
            tmp_path,
            catalog=load_fblock_catalog(),
        )


def _manifest_payload() -> dict[str, object]:
    return json.loads(
        (bundled_fblock_directory() / "donor-aliases.json").read_text(
            encoding="utf-8"
        )
    )


def _write_manifest(directory: Path, payload: dict[str, object]) -> None:
    (directory / "donor-aliases.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

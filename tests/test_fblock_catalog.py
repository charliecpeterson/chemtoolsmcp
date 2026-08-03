"""Installed-resource and scientific-contract tests for the f-block catalog."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chemtools.reference.fblock import (
    FBlockCatalogLoadError,
    bundled_fblock_directory,
    load_fblock_catalog,
)


def test_bundled_fblock_catalog_has_exact_versioned_coverage():
    catalog = load_fblock_catalog()

    assert catalog.metadata.to_dict() == {
        "schema_version": "chemtools.fblock-dataset/1",
        "dataset_id": "fblock.atomic_seeds",
        "dataset_version": "2",
        "rebuild_date": "2026-07-28",
        "element_count": 31,
        "state_count": 633,
        "role_counts": {"fit": 493, "holdout": 140},
        "seed_class_counts": {
            "atsp_hf": 60,
            "cold": 36,
            "donor": 427,
            "multi_donor": 110,
        },
        "staged_birth_state_count": 51,
        "catalog_sha256": (
            "6b2a59951c11ab141bbe0dfe4806f9b1fc4248b80f8cf2eb780ac1b426495eeb"
        ),
    }
    assert catalog.metadata.redistribution.status == "allowed"
    assert catalog.metadata.redistribution.license_identifier == "MIT"
    assert [program.name for program in catalog.metadata.programs] == [
        "GRASP2018",
        "ATSP2K",
        "DIRAC",
    ]


def test_thorium_false_vacuum_reference_preserves_method_and_seed_scope():
    catalog = load_fblock_catalog()
    thorium = catalog.element("th")
    state = thorium.state("ion0_6d27s2")

    assert catalog.element("Y").atomic_number == 39
    assert thorium.atomic_number == 90
    assert thorium.mass_number == 232
    assert thorium.atsp_hf_seed_default is True
    assert state.config == "6d(2)7s(2)"
    assert state.j_blocks == ("0", "1", "2", "3", "4")
    assert state.ncsf == (2, 1, 3, 1, 2)
    assert state.energy_dcb_au == -26475.800396508253
    assert state.seed_class == "atsp_hf"
    assert state.seeding == (
        "ATSP-hf seed (non-relativistic orbitals, converted)"
    )


def test_multi_donor_lineage_remains_typed_and_immutable():
    catalog = load_fblock_catalog()
    state = next(
        state
        for element in catalog.elements
        for state in element.states
        if state.seed_class == "multi_donor"
    )

    assert isinstance(state.estimate_from, tuple)
    assert len(state.estimate_from) == 2
    assert all(isinstance(donor, str) for donor in state.estimate_from)


def test_bundled_access_uses_package_data_and_old_copy_is_absent():
    directory = bundled_fblock_directory()

    assert directory.parts[-2:] == ("data", "fblock")
    assert (directory / "metadata.json").is_file()
    assert (directory / "grasp" / "fblock-all.json").is_file()
    repository_root = Path(__file__).resolve().parents[1]
    assert not (repository_root / "notes/fblock/atomic-library").exists()


def test_catalog_hash_drift_is_rejected(tmp_path):
    source = bundled_fblock_directory()
    target = tmp_path / "fblock"
    target.mkdir()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    metadata["components"] = [{
        **metadata["components"][0],
        "paths": ["grasp/fblock-all.json"],
    }]
    (target / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    catalog_target = target / "grasp" / "fblock-all.json"
    catalog_target.parent.mkdir()
    payload = (source / "grasp" / "fblock-all.json").read_bytes()
    catalog_target.write_bytes(payload[:-1] + b" ")

    with pytest.raises(FBlockCatalogLoadError, match="catalog SHA-256 changed"):
        load_fblock_catalog(target)


def test_catalog_rejects_misaligned_j_and_ncsf_arrays(tmp_path):
    source = bundled_fblock_directory()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    metadata["components"] = [{
        **metadata["components"][0],
        "paths": ["grasp/fblock-all.json"],
    }]
    full_catalog = json.loads(
        (source / "grasp" / "fblock-all.json").read_text(encoding="utf-8")
    )
    element = full_catalog["Th"]
    state = dict(element["states"][0])
    state["ncsf"] = state["ncsf"][:-1]
    element = {**element, "states": [state]}
    payload = json.dumps({"Th": element}, separators=(",", ":")).encode()

    metadata["catalog"]["size_bytes"] = len(payload)
    metadata["catalog"]["sha256"] = hashlib.sha256(payload).hexdigest()
    metadata["coverage"].update({
        "elements": ["Th"],
        "element_count": 1,
        "state_count": 1,
        "role_counts": {"fit": 1, "holdout": 0},
        "seed_class_counts": {
            "donor": 0,
            "multi_donor": 0,
            "atsp_hf": 1,
            "cold": 0,
        },
        "staged_birth_state_count": 0,
    })
    target = tmp_path / "fblock"
    (target / "grasp").mkdir(parents=True)
    (target / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (target / "grasp" / "fblock-all.json").write_bytes(payload)

    with pytest.raises(
        FBlockCatalogLoadError,
        match="J_blocks and ncsf lengths differ",
    ):
        load_fblock_catalog(target)

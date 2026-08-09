"""Exact access-boundary tests for named external reference artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from chemtools.reference.external_corpus import (
    REFERENCE_CORPUS_SCHEMA,
    ReferenceManifestError,
    load_reference_manifest,
    verify_reference_case,
)


NWCHEM_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "chemtools"
    / "data"
    / "reference_cases"
    / "nwchem_behavior_cases.json"
)
NON_NWCHEM_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "chemtools"
    / "data"
    / "reference_cases"
    / "non_nwchem_review_cases.json"
)
ORCA_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "chemtools"
    / "data"
    / "reference_cases"
    / "orca_experimental_cases.json"
)


def _artifact(contents: bytes) -> dict:
    return {
        "id": "primary_output",
        "roles": ["primary_output"],
        "kind": "nwchem.output",
        "path": "nwchem/case.out",
        "storage_tier": "external_reference",
        "size_bytes": len(contents),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "required": True,
        "redistribution": "review_required",
        "source": "local-research",
        "attribution": "Charles Peterson",
        "license": {"identifier": None, "terms": None},
        "permission_evidence": None,
    }


def _manifest(contents: bytes) -> dict:
    return {
        "schema": REFERENCE_CORPUS_SCHEMA,
        "cases": [{
            "id": "nwchem.example",
            "programs": ["nwchem"],
            "status": "exploratory",
            "purposes": ["scientific_regression"],
            "artifacts": [_artifact(contents)],
            "expected": {},
            "review": {
                "reviewed_by": None,
                "reviewed_at": None,
                "scope": "Artifact identity only",
            },
            "tags": ["nwchem"],
        }],
    }


def test_verify_reference_case_returns_exact_verified_artifact(tmp_path):
    contents = b"Total DFT energy = -1.0\n"
    corpus = tmp_path / "corpus"
    source = corpus / "nwchem" / "case.out"
    source.parent.mkdir(parents=True)
    source.write_bytes(contents)

    verified = verify_reference_case(_manifest(contents), "nwchem.example", corpus)

    assert verified == {
        "case_id": "nwchem.example",
        "status": "exploratory",
        "purposes": ["scientific_regression"],
        "outcome": "verified",
        "total_size_bytes": len(contents),
        "artifacts": [{
            "id": "primary_output",
            "path": str(source),
            "roles": ["primary_output"],
            "kind": "nwchem.output",
            "size_bytes": len(contents),
            "sha256": hashlib.sha256(contents).hexdigest(),
        }],
    }


def test_size_change_is_rejected_before_hashing(tmp_path, monkeypatch):
    expected = b"expected\n"
    corpus = tmp_path / "corpus"
    source = corpus / "nwchem" / "case.out"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"changed and longer\n")

    def unexpected_hash(path):
        raise AssertionError(f"must not hash resized artifact {path}")

    monkeypatch.setattr(
        "chemtools.reference.external_corpus._sha256",
        unexpected_hash,
    )

    checked = verify_reference_case(_manifest(expected), "nwchem.example", corpus)

    assert checked["outcome"] == "no_reference"
    assert checked["artifact_id"] == "primary_output"
    assert checked["reason"] == (
        "reference size changed; review the case before use"
    )
    assert checked["expected_size_bytes"] == len(expected)
    assert checked["actual_size_bytes"] == len(b"changed and longer\n")


def test_case_budget_is_rejected_before_filesystem_access(tmp_path):
    checked = verify_reference_case(
        _manifest(b"12345678"),
        "nwchem.example",
        tmp_path / "missing",
        byte_limit=7,
    )

    assert checked == {
        "case_id": "nwchem.example",
        "status": "exploratory",
        "purposes": ["scientific_regression"],
        "outcome": "no_reference",
        "reason": "case exceeds the configured byte budget",
        "expected_size_bytes": 8,
        "byte_limit": 7,
    }


def test_symlink_escape_is_rejected(tmp_path):
    contents = b"outside\n"
    corpus = tmp_path / "corpus"
    outside = tmp_path / "outside.out"
    (corpus / "nwchem").mkdir(parents=True)
    outside.write_bytes(contents)
    (corpus / "nwchem" / "case.out").symlink_to(outside)

    checked = verify_reference_case(_manifest(contents), "nwchem.example", corpus)

    assert checked["outcome"] == "no_reference"
    assert checked["reason"] == (
        "reference path escapes the configured corpus root"
    )


@pytest.mark.parametrize("path", ["../case.out", "/case.out"])
def test_manifest_rejects_uncontained_paths(tmp_path, path):
    payload = _manifest(b"output\n")
    payload["cases"][0]["artifacts"][0]["path"] = path
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ReferenceManifestError,
        match="path must be a contained relative path",
    ):
        load_reference_manifest(manifest)


def test_manifest_rejects_duplicate_case_ids(tmp_path):
    payload = _manifest(b"output\n")
    payload["cases"].append(payload["cases"][0])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ReferenceManifestError,
        match="duplicate case IDs",
    ):
        load_reference_manifest(manifest)


def test_manifest_rejects_unknown_scientific_status(tmp_path):
    payload = _manifest(b"output\n")
    payload["cases"][0]["status"] = "looks_good"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ReferenceManifestError,
        match="status has unsupported value 'looks_good'",
    ):
        load_reference_manifest(manifest)


def test_allowed_redistribution_requires_permission_evidence(tmp_path):
    payload = _manifest(b"output\n")
    artifact = payload["cases"][0]["artifacts"][0]
    artifact["redistribution"] = "allowed"
    artifact["license"]["identifier"] = "MIT"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ReferenceManifestError,
        match="requires license terms and permission evidence",
    ):
        load_reference_manifest(manifest)


def test_nwchem_behavior_manifest_has_five_exploratory_cases():
    manifest = load_reference_manifest(NWCHEM_MANIFEST)

    assert [case["id"] for case in manifest["cases"]] == [
        "nwchem.fecn6_lowspin_fragment",
        "nwchem.hexaaquairon_swap_chain",
        "nwchem.feo_spin_comparison",
        "nwchem.ferrocene_basis_stepping",
        "nwchem.crco6_bailar_twist",
    ]
    assert {case["status"] for case in manifest["cases"]} == {"exploratory"}


def test_non_nwchem_manifest_is_an_exploratory_review_queue():
    manifest = load_reference_manifest(NON_NWCHEM_MANIFEST)

    assert [case["id"] for case in manifest["cases"]] == [
        "molcas.nactel_parity_failure",
        "molcas.hcn_transition_state_frequency",
        "dirac.h2o_x2c_4c_comparison",
        "dirac.uranium_open_shell",
        "grasp.thorium_relativistic_limit",
        "grasp.lithium_e1_transition",
        "qe.feo_vc_relaxation",
        "qe.iron_spin_scf",
        "qmcpack.hydrogen_vmc_statistics",
        "qmcpack.oxygen_dmc_autocorrection",
    ]
    assert {case["status"] for case in manifest["cases"]} == {
        "exploratory"
    }
    assert {
        program: sum(case["programs"] == [program] for case in manifest["cases"])
        for program in ("molcas", "dirac", "grasp", "qe", "qmcpack")
    } == {
        "molcas": 2,
        "dirac": 2,
        "grasp": 2,
        "qe": 2,
        "qmcpack": 2,
    }
    assert all(
        case["expected"]["review_state"] == "pending"
        and case["review"]["reviewed_by"] is None
        and case["review"]["reviewed_at"] is None
        for case in manifest["cases"]
    )


def test_orca_manifest_pins_nineteen_experiments():
    manifest = load_reference_manifest(ORCA_MANIFEST)

    assert [case["id"] for case in manifest["cases"]] == [
        "orca.h2_hf_serial_smoke",
        "orca.water_r2scan3c_opt_freq",
        "orca.o2_triplet_pbe0",
        "orca.formaldehyde_wb97x_d4",
        "orca.cucl4_doublet_pbe0",
        "orca.uranyl_singlet_pbe0_zora",
        "orca.cucl4_interrupted_moread_restart",
        "orca.fe_macrocycle_scf_algorithms",
        "orca.water_dlpno_ccsdt_tightpno",
        "orca.water_pbe0_rijcosx",
        "orca.water_pentamer_qmmm",
        "orca.nacl6_ionic_crystal_qmmm",
        "orca.alpha_glycine_molecular_crystal_qmmm",
        "orca.n2_stretched_casscf_nevpt2",
        "orca.formaldehyde_mrci_excited_states",
        "orca.formaldehyde_pbe0_tddft",
        "orca.formaldehyde_eom_ccsd",
        "orca.n2_excited_casscf_caspt2",
        "orca.formaldehyde_esd_spectra_and_rates",
    ]
    assert {case["status"] for case in manifest["cases"]} == {
        "exploratory",
        "regression_failure",
    }
    assert all(
        case["expected"]["review_state"] == "pending"
        and case["review"]["reviewed_by"] is None
        and case["review"]["reviewed_at"] is None
        for case in manifest["cases"]
    )


@pytest.mark.skipif(
    not os.environ.get("CHEMTOOLS_REFERENCE_CORPUS"),
    reason="CHEMTOOLS_REFERENCE_CORPUS is not configured",
)
def test_nwchem_behavior_manifest_matches_external_corpus():
    manifest = load_reference_manifest(NWCHEM_MANIFEST)

    checked = [
        verify_reference_case(
            manifest,
            case["id"],
            os.environ["CHEMTOOLS_REFERENCE_CORPUS"],
        )
        for case in manifest["cases"]
    ]

    assert [item["outcome"] for item in checked] == ["verified"] * 5


@pytest.mark.skipif(
    not os.environ.get("CHEMTOOLS_REFERENCE_CORPUS"),
    reason="CHEMTOOLS_REFERENCE_CORPUS is not configured",
)
def test_non_nwchem_review_manifest_matches_external_corpus():
    manifest = load_reference_manifest(NON_NWCHEM_MANIFEST)

    checked = [
        verify_reference_case(
            manifest,
            case["id"],
            os.environ["CHEMTOOLS_REFERENCE_CORPUS"],
        )
        for case in manifest["cases"]
    ]

    assert [item["outcome"] for item in checked] == ["verified"] * 10


@pytest.mark.skipif(
    not os.environ.get("CHEMTOOLS_REFERENCE_CORPUS"),
    reason="CHEMTOOLS_REFERENCE_CORPUS is not configured",
)
def test_orca_manifest_matches_external_corpus():
    manifest = load_reference_manifest(ORCA_MANIFEST)

    checked = [
        verify_reference_case(
            manifest,
            case["id"],
            os.environ["CHEMTOOLS_REFERENCE_CORPUS"],
        )
        for case in manifest["cases"]
    ]

    assert [item["outcome"] for item in checked] == ["verified"] * 19

"""Regression contracts for reviewed NWChem-to-PySCF reference evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chemtools.application import nwchem_pyscf
from chemtools.programs.nwchem.pyscf_reference import (
    draft_nwchem_pyscf_reference,
)
from chemtools.programs.nwchem.strategy.diagnose import parse_scf


FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "nwchem_pyscf"
CORPUS_PATH = FIXTURE_DIRECTORY / "cases.json"


def _corpus() -> dict:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nwchem_pyscf_fixture_corpus_pins_real_case_provenance_and_artifacts():
    corpus = _corpus()

    assert corpus["schema_version"] == "chemtools.nwchem-pyscf-fixtures/1"
    assert corpus["recorded_with"] == {
        "nwchem_version": "7.2.2",
        "container": "/home/charlie/mycontainers/nwchem_7.2.2.sif",
        "executor": "apptainer exec with mpirun -np 2",
        "host_class": "linux-4090",
    }
    assert [case["id"] for case in corpus["cases"]] == [
        "h2_rhf_sto3g",
        "h2o_rhf_sto3g",
        "o2_uhf_triplet_sto3g",
        "h2_rks_b3lyp_sto3g",
        "h2o_rhf_sto3g_maxiter1",
    ]
    for case in corpus["cases"]:
        assert _sha256(FIXTURE_DIRECTORY / case["input_file"]) == case["input_sha256"]
        assert _sha256(FIXTURE_DIRECTORY / case["output_file"]) == case["output_sha256"]
        assert (FIXTURE_DIRECTORY / case["error_file"]).is_file()


def test_nwchem_pyscf_fixture_outputs_and_reference_drafts_agree_on_evidence():
    for case in _corpus()["cases"]:
        expected = case["expected"]
        input_path = FIXTURE_DIRECTORY / case["input_file"]
        output_path = FIXTURE_DIRECTORY / case["output_file"]
        scf = parse_scf(str(output_path))
        draft = draft_nwchem_pyscf_reference(
            str(input_path),
            output_path=str(output_path),
            pyscf_method=expected["pyscf_method"],
            pyscf_xc=expected["pyscf_xc"],
            density_fit=expected["density_fit"],
            electron_total=expected["electron_total"],
        )

        assert scf["status"] == expected["scf_status"]
        assert scf["total_energy_hartree"] == pytest.approx(
            expected.get("total_hartree", expected.get("printed_total_hartree")),
        )
        assert draft["reference_draft"]["calculation"]["multiplicity"] == expected["multiplicity"]
        assert draft["reference_draft"]["scf"]["converged"] is expected["scf_converged"]

        if expected["scf_converged"]:
            assert draft["comparison_ready"] is True
            assert draft["reference_draft"]["energy"]["total_hartree"] == pytest.approx(
                expected["total_hartree"],
            )
        else:
            assert draft["comparison_ready"] is False
            assert draft["reference_draft"]["energy"]["total_hartree"] is None
            assert "energy.total_hartree" in draft["missing_required_fields"]

    dft_draft = draft_nwchem_pyscf_reference(
        str(FIXTURE_DIRECTORY / "h2_rks_b3lyp_sto3g.nw"),
        output_path=str(FIXTURE_DIRECTORY / "h2_rks_b3lyp_sto3g.out"),
        pyscf_method="rks",
        pyscf_xc="b3lyp",
        density_fit=False,
        electron_total=2,
    )
    assert dft_draft["field_sources"]["calculation.xc"] == {
        "status": "caller_declared",
        "value": "b3lyp",
        "nwchem_xc": "b3lyp",
        "nwchem_evidence": {
            "status": "extracted",
            "value": "b3lyp",
            "block_selection": "last",
        },
        "reason": (
            "The NWChem xc declaration is retained as evidence but is not treated "
            "as a PySCF functional-equivalence mapping."
        ),
    }


def test_failed_nwchem_fixture_refuses_to_launch_pyscf(tmp_path, monkeypatch):
    case = next(
        case for case in _corpus()["cases"]
        if case["id"] == "h2o_rhf_sto3g_maxiter1"
    )
    expected = case["expected"]
    monkeypatch.setattr(
        nwchem_pyscf,
        "run_pyscf_single_point",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    response = nwchem_pyscf.run_nwchem_pyscf_matched_reference(
        object(),
        input_path=str(FIXTURE_DIRECTORY / case["input_file"]),
        output_path=str(FIXTURE_DIRECTORY / case["output_file"]),
        working_directory=str(tmp_path),
        pyscf_method=expected["pyscf_method"],
        pyscf_xc=expected["pyscf_xc"],
        density_fit=expected["density_fit"],
        electron_total=expected["electron_total"],
    )

    assert response["status"] == "reference_incomplete"
    assert response["comparison"] is None
    assert "energy.total_hartree" in response["reference_draft"]["missing_required_fields"]

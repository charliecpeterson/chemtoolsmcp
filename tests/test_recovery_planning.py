"""Application and guided-tool contracts for read-only recovery plans."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chemtools.application.recovery_planning import (
    ApplyRecoveryResolutionError,
    RecoveryPlanError,
    plan_recovery,
    resolve_apply_recovery_program,
)
from chemtools.application.input_review import InputReviewError
from chemtools.application import recovery_planning
from chemtools.core import registry
from chemtools.mcp.catalog import (
    BUILTIN_BACKENDS,
    load_backend,
    register_builtin_backends,
)
from chemtools.mcp.tools import generic, guided


FIXTURES = Path(__file__).parent / "golden" / "mcp" / "fixtures"
BACKENDS = tuple(load_backend(spec) for spec in BUILTIN_BACKENDS)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("nwchem_h2.nw", "nwchem"),
        ("molcas_scf.input", "molcas"),
    ],
)
def test_apply_recovery_program_uses_shared_input_detection(
    filename,
    expected,
):
    assert resolve_apply_recovery_program(
        BACKENDS,
        input_file=FIXTURES / filename,
    ) == expected


def test_apply_recovery_program_rejects_conflicting_input():
    with pytest.raises(ApplyRecoveryResolutionError) as caught:
        resolve_apply_recovery_program(
            BACKENDS,
            input_file=FIXTURES / "nwchem_h2.nw",
            selected_program="molcas",
        )

    assert caught.value.as_dict() == {
        "error": "program_content_mismatch",
        "message": (
            "recovery input content matches nwchem, but selected program "
            "is molcas"
        ),
        "program": "molcas",
        "detected_programs": ["nwchem"],
    }


def test_apply_recovery_program_preserves_missing_input_compatibility(tmp_path):
    missing = tmp_path / "missing.input"
    with pytest.raises(ApplyRecoveryResolutionError) as automatic:
        resolve_apply_recovery_program(BACKENDS, input_file=missing)
    assert automatic.value.code == "program_detection_failed"

    with pytest.raises(InputReviewError) as explicit:
        resolve_apply_recovery_program(
            BACKENDS,
            input_file=missing,
            selected_program="molcas",
        )
    assert explicit.value.code == "source_not_file"


def test_apply_recovery_handler_delegates_source_resolution(monkeypatch):
    calls = []

    def resolve(backends, **kwargs):
        calls.append((sorted(backend.name for backend in backends), kwargs))
        return "nwchem"

    monkeypatch.setattr(generic, "resolve_apply_recovery_program", resolve)

    recovered = generic._handle_apply_recovery_generic({
        "program": "molcas",
        "input_file": "run.input",
    })

    assert recovered["program"] == "nwchem"
    assert recovered["verdict"] == "not_implemented_for_program"
    assert calls == [(
        ["dirac", "grasp", "molcas", "nwchem", "orca", "qe", "qmcpack"],
        {
            "input_file": "run.input",
            "selected_program": "molcas",
        },
    )]


def test_apply_recovery_falls_back_from_sparse_output_to_input(tmp_path):
    if not registry.has("nwchem"):
        register_builtin_backends()
    output_path = tmp_path / "sparse.out"
    output_path.write_text("partial calculation record\n", encoding="utf-8")

    recovered = generic._handle_apply_recovery_generic({
        "output_file": str(output_path),
        "input_file": str(FIXTURES / "nwchem_h2.nw"),
    })

    assert recovered["program"] == "nwchem"
    assert recovered["verdict"] == "not_implemented_for_program"


def test_completed_nwchem_run_needs_no_recovery():
    planned = plan_recovery(
        load_backend(BUILTIN_BACKENDS[0]),
        FIXTURES / "nwchem_scf.out",
        input_file=FIXTURES / "nwchem_h2.nw",
    )

    assert planned["schema_version"] == "chemtools.plan-recovery/1"
    assert planned["program"] == {"name": "nwchem"}
    assert planned["target"] == {
        "expected_charge": None,
        "expected_multiplicity": None,
        "expected_metal_elements": [],
        "expected_somo_count": None,
        "somo_count_source": None,
    }
    assert planned["assessment"]["verdict"]["label"] == (
        "no_recovery_needed"
    )
    assert planned["evidence"]["plan_kind"] == "verification_only"
    assert planned["evidence"]["input_output_consistency"]["status"] == (
        "checked"
    )
    assert planned["evidence"]["prepared_artifacts"] == []
    assert planned["evidence"]["files_written"] == []
    assert planned["uncertainty"] == []
    assert planned["next_actions"] == []


def test_target_multiplicity_derives_somo_count():
    planned = plan_recovery(
        load_backend(BUILTIN_BACKENDS[0]),
        FIXTURES / "nwchem_scf.out",
        input_file=FIXTURES / "nwchem_h2.nw",
        target={"expected_multiplicity": 1},
    )

    assert planned["target"]["expected_somo_count"] == 0
    assert planned["target"]["somo_count_source"] == (
        "derived_from_multiplicity"
    )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ([], "recovery target must be an object"),
        ({"expected_multiplicity": 0}, "positive integer"),
        ({"expected_somo_count": -1}, "nonnegative integer"),
        ({"expected_metal_elements": [""]}, "element symbols"),
        ({"unknown": 1}, "unsupported recovery target fields"),
    ],
)
def test_recovery_target_validation_is_explicit(target, message):
    with pytest.raises(RecoveryPlanError, match=message):
        plan_recovery(
            load_backend(BUILTIN_BACKENDS[0]),
            FIXTURES / "nwchem_scf.out",
            target=target,
        )


def test_guided_plan_recovery_uses_application_contract():
    if not registry.has("nwchem"):
        register_builtin_backends()

    response = guided._handle_plan_recovery({
        "output_file": str(FIXTURES / "nwchem_scf.out"),
        "input_file": str(FIXTURES / "nwchem_h2.nw"),
        "program": "nwchem",
    })

    assert response["schema_version"] == "chemtools.plan-recovery/1"
    assert response["program"] == {"name": "nwchem"}
    assert response["assessment"]["verdict"]["label"] == (
        "no_recovery_needed"
    )
    assert response["evidence"]["files_written"] == []


@pytest.mark.parametrize(
    ("can_prepare", "expected_label", "expected_actions"),
    [
        (
            True,
            "source_consistency_required",
            ["confirm_source_artifacts"],
        ),
        (False, "manual_review_required", ["manual_recovery_review"]),
    ],
)
def test_recovery_consistency_mismatch_blocks_only_prepared_candidates(
    monkeypatch,
    can_prepare,
    expected_label,
    expected_actions,
):
    class Diagnostics:
        def plan_recovery(self, output_path, input_path, target):
            return {
                "assessment": {
                    "verdict": {
                        "label": (
                            "recovery_plan_ready"
                            if can_prepare
                            else "manual_review_required"
                        ),
                        "confidence": 0.8,
                        "reasons": [],
                    }
                },
                "evidence": {
                    "plan_kind": "test_recovery",
                    "can_prepare": can_prepare,
                    "prepared_artifacts": ([{
                        "kind": "candidate",
                        "candidate_drafts": [{"text": "task dft energy\n"}],
                    }] if can_prepare else []),
                    "files_written": [],
                },
                "uncertainty": [],
                "next_actions": [{
                    "action": "manual_recovery_review",
                    "priority": 1,
                }],
            }

    consistency = {
        "status": "mismatch",
        "checks": [{
            "field": "restart_artifacts",
            "status": "mismatch",
        }],
    }
    mismatch = {
        "code": "input_output_mismatch",
        "message": "The explicit input disagrees with output evidence.",
        "impact": "Confirm the source pair.",
    }
    monkeypatch.setattr(
        recovery_planning,
        "_check_source_consistency",
        lambda *_: (consistency, [mismatch]),
    )
    backend = replace(
        load_backend(BUILTIN_BACKENDS[0]),
        diagnostics=Diagnostics(),
    )

    planned = plan_recovery(
        backend,
        FIXTURES / "nwchem_scf.out",
        input_file=FIXTURES / "nwchem_h2.nw",
    )

    assert planned["assessment"]["verdict"]["label"] == expected_label
    assert planned["evidence"]["input_output_consistency"] == consistency
    assert planned["uncertainty"] == [mismatch]
    assert [action["action"] for action in planned["next_actions"]] == (
        expected_actions
    )
    if can_prepare:
        assert planned["evidence"]["can_prepare"] is False
        assert planned["evidence"]["proposed_plan_kind"] == "test_recovery"
        assert planned["evidence"]["prepared_artifacts"] == []
    else:
        assert "proposed_plan_kind" not in planned["evidence"]

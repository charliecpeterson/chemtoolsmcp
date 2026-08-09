"""Opt-in application contracts for pinned expert NWChem workflows."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chemtools.application.input_review import review_input
from chemtools.application.run_comparison import compare_runs
from chemtools.application.run_inspection import inspect_run
from chemtools.application.recovery_planning import plan_recovery
from chemtools.mcp.catalog import BUILTIN_BACKENDS, load_backend
from chemtools.programs.nwchem.parse.mos import parse_mos
from chemtools.reference.external_corpus import (
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


GUIDED_CASE_EXPECTATIONS = {
    "nwchem.fecn6_lowspin_fragment": {
        "reviews": {
            "failed_input": ("checks_passed", []),
            "solution_input": ("review_required", ["edit_input"]),
        },
        "inspections": {
            "failed_output": (
                "success",
                [],
                ["analyze_nwchem_frontier_orbitals"],
            ),
            "solution_output": (
                "success",
                [],
                ["analyze_nwchem_frontier_orbitals"],
            ),
        },
    },
    "nwchem.hexaaquairon_swap_chain": {
        "reviews": {
            "default_input": ("review_required", ["edit_input"]),
            "fragment_input": ("review_required", ["edit_input"]),
            "swap_input": ("review_required", ["edit_input"]),
        },
        "inspections": {
            "default_output": (
                "wrong_state_convergence",
                [],
                ["analyze_nwchem_case"],
            ),
            "fragment_output": (
                "success",
                [],
                ["analyze_nwchem_frontier_orbitals"],
            ),
            "swap_output": (
                "wrong_state_convergence",
                ["input_output_mismatch"],
                ["analyze_nwchem_case"],
            ),
        },
    },
    "nwchem.feo_spin_comparison": {
        "reviews": {
            "triplet_input": ("checks_passed", []),
            "quintet_input": ("checks_passed", []),
        },
        "inspections": {
            "triplet_output": (
                "success",
                [],
                ["analyze_nwchem_frontier_orbitals"],
            ),
            "quintet_output": (
                "success",
                [],
                ["analyze_nwchem_frontier_orbitals"],
            ),
        },
    },
    "nwchem.ferrocene_basis_stepping": {
        "reviews": {
            "failed_input": ("checks_passed", []),
            "small_basis_input": ("checks_passed", []),
            "solution_input": ("checks_passed", []),
        },
        "inspections": {
            "failed_output": (
                "success",
                [],
                ["analyze_nwchem_frontier_orbitals"],
            ),
            "small_basis_output": (
                "success",
                [],
                ["analyze_nwchem_frontier_orbitals"],
            ),
            "solution_output": (
                "success",
                ["input_output_mismatch"],
                ["analyze_nwchem_frontier_orbitals"],
            ),
        },
    },
    "nwchem.crco6_bailar_twist": {
        "reviews": {
            "saddle_input": ("checks_passed", []),
            "minimum_input": ("checks_passed", []),
        },
        "inspections": {
            "saddle_output": (
                "frequency_interpretation_required",
                [],
                ["analyze_nwchem_case"],
            ),
            "minimum_output": (
                "success",
                [],
                ["analyze_nwchem_frontier_orbitals"],
            ),
        },
    },
}


@pytest.mark.skipif(
    not os.environ.get("CHEMTOOLS_REFERENCE_CORPUS"),
    reason="CHEMTOOLS_REFERENCE_CORPUS is not configured",
)
def test_guided_review_and_inspection_map_covers_all_five_cases():
    manifest = load_reference_manifest(NWCHEM_MANIFEST)
    backend = load_backend(BUILTIN_BACKENDS[0])

    for case_id, expected in GUIDED_CASE_EXPECTATIONS.items():
        verified = verify_reference_case(
            manifest,
            case_id,
            os.environ["CHEMTOOLS_REFERENCE_CORPUS"],
        )
        assert verified["outcome"] == "verified"
        artifacts = {
            artifact["id"]: artifact["path"]
            for artifact in verified["artifacts"]
        }

        for artifact_id, (verdict, actions) in expected["reviews"].items():
            reviewed = review_input(
                backend,
                artifacts[artifact_id],
                resolved_by="explicit",
            )
            assert reviewed["assessment"]["verdict"]["label"] == verdict
            assert _action_names(reviewed["next_actions"]) == actions

        for artifact_id, expected_result in expected["inspections"].items():
            prefix = artifact_id.removesuffix("_output")
            inspected = inspect_run(
                backend,
                artifacts[artifact_id],
                resolved_by="explicit",
                artifact_files=[artifacts[f"{prefix}_input"]],
            )
            verdict, uncertainty_codes, actions = expected_result
            assert inspected["assessment"]["verdict"]["label"] == verdict
            assert [
                item["code"] for item in inspected["uncertainty"]
            ] == uncertainty_codes
            assert _action_names(inspected["next_actions"]) == actions


def _action_names(actions):
    return [
        action.get("tool") or action.get("action")
        for action in actions
    ]


@pytest.mark.skipif(
    not os.environ.get("CHEMTOOLS_REFERENCE_CORPUS"),
    reason="CHEMTOOLS_REFERENCE_CORPUS is not configured",
)
def test_hexaaquairon_lock_blocks_mismatched_swap_provenance():
    manifest = load_reference_manifest(NWCHEM_MANIFEST)
    verified = verify_reference_case(
        manifest,
        "nwchem.hexaaquairon_swap_chain",
        os.environ["CHEMTOOLS_REFERENCE_CORPUS"],
    )
    assert verified["outcome"] == "verified"
    artifacts = {
        artifact["id"]: artifact["path"]
        for artifact in verified["artifacts"]
    }
    backend = load_backend(BUILTIN_BACKENDS[0])

    plans = {
        prefix: plan_recovery(
            backend,
            artifacts[f"{prefix}_output"],
            input_file=artifacts[f"{prefix}_input"],
        )
        for prefix in ("default", "fragment", "swap")
    }

    assert {
        prefix: plan["assessment"]["verdict"]["label"]
        for prefix, plan in plans.items()
    } == {
        "default": "recovery_plan_ready",
        "fragment": "stability_hardening_requires_manual_edit",
        "swap": "source_consistency_required",
    }
    assert plans["default"]["evidence"]["plan_kind"] == (
        "wrong_state_swap_recovery"
    )
    assert len(plans["default"]["evidence"]["prepared_artifacts"]) == 1
    assert plans["fragment"]["evidence"]["prepared_artifacts"] == []

    blocked = plans["swap"]
    assert blocked["evidence"]["proposed_plan_kind"] == (
        "wrong_state_swap_recovery"
    )
    assert blocked["evidence"]["prepared_artifacts"] == []
    assert blocked["evidence"]["input_output_consistency"]["status"] == (
        "mismatch"
    )
    assert [item["code"] for item in blocked["uncertainty"]] == [
        "input_output_mismatch"
    ]
    assert blocked["next_actions"] == [{
        "action": "confirm_source_artifacts",
        "reason": (
            "Confirm the input, output, and referenced restart artifacts "
            "belong to the same calculation before preparing a retry."
        ),
        "priority": 1,
    }]


@pytest.mark.skipif(
    not os.environ.get("CHEMTOOLS_REFERENCE_CORPUS"),
    reason="CHEMTOOLS_REFERENCE_CORPUS is not configured",
)
def test_fecn6_lock_recovers_low_spin_state_and_frontier_character():
    manifest = load_reference_manifest(NWCHEM_MANIFEST)
    verified = verify_reference_case(
        manifest,
        "nwchem.fecn6_lowspin_fragment",
        os.environ["CHEMTOOLS_REFERENCE_CORPUS"],
    )
    assert verified["outcome"] == "verified"
    artifacts = {
        artifact["id"]: artifact["path"]
        for artifact in verified["artifacts"]
    }
    backend = load_backend(BUILTIN_BACKENDS[0])

    compared = compare_runs(
        backend,
        artifacts["failed_output"],
        artifacts["solution_output"],
        reference_input_file=artifacts["failed_input"],
        candidate_input_file=artifacts["solution_input"],
    )

    assert compared["assessment"]["verdict"]["label"] == (
        "candidate_lower_energy"
    )
    assert compared["assessment"]["comparability"] == {
        "status": "partially_checked",
        "blocking_fields": [],
        "unchecked_fields": ["composition", "geometry"],
    }
    assert compared["evidence"]["energy"] == {
        "status": "checked",
        "unit": "hartree",
        "reference": -1819.496662763585,
        "candidate": -1820.002049685715,
        "candidate_minus_reference": pytest.approx(-0.5053869221301284),
        "candidate_minus_reference_kcal_per_mol": pytest.approx(
            -317.13508170424575
        ),
        "equality_tolerance_hartree": 1.0e-8,
        "lower_energy_run": "candidate",
    }
    assert compared["evidence"]["reference"]["multiplicity"] == 6
    assert compared["evidence"]["candidate"]["multiplicity"] == 2
    assert compared["evidence"]["reference"]["verdict"]["label"] == (
        "success"
    )
    assert compared["evidence"]["candidate"]["verdict"]["label"] == (
        "success"
    )

    failed_output = Path(artifacts["failed_output"])
    solution_output = Path(artifacts["solution_output"])
    failed_mos = parse_mos(
        str(failed_output),
        failed_output.read_text(
            encoding="utf-8",
            errors="replace",
        ),
    )
    solution_mos = parse_mos(
        str(solution_output),
        solution_output.read_text(
            encoding="utf-8",
            errors="replace",
        ),
    )
    assert failed_mos["somo_count"] == 5
    assert [
        (orbital["vector_number"], orbital["symmetry"])
        for orbital in failed_mos["somos"]
    ] == [
        (55, "eg"),
        (56, "eg"),
        (52, "t2g"),
        (53, "t2g"),
        (54, "t2g"),
    ]
    assert solution_mos["somo_count"] == 1
    assert solution_mos["somos"][0]["vector_number"] == 52
    assert solution_mos["somos"][0]["symmetry"] == "t2g"
    assert solution_mos["somos"][0]["top_atom_contributions"][0][
        "element"
    ] == "Fe"
    assert solution_mos["somos"][0]["top_atom_contributions"][0][
        "fraction_of_visible"
    ] == pytest.approx(0.8696950641323992)

    failed_plan = plan_recovery(
        backend,
        artifacts["failed_output"],
        input_file=artifacts["failed_input"],
        target={
            "expected_charge": -3,
            "expected_multiplicity": 2,
            "expected_metal_elements": ["Fe"],
        },
    )
    assert failed_plan["assessment"]["verdict"]["label"] == (
        "target_state_rebuild_required"
    )
    assert failed_plan["evidence"]["plan_kind"] == "target_state_rebuild"
    assert failed_plan["evidence"]["input_state"]["multiplicity"] == 6
    assert failed_plan["evidence"]["state_check"] == {
        "assessment": "somo_count_mismatch",
        "observed_somo_count": 5,
        "expected_somo_count": 1,
        "metal_like_somo_count": 3,
        "ligand_like_somo_count": 0,
    }
    assert failed_plan["evidence"]["target_mismatches"] == [{
        "field": "multiplicity",
        "input": 6,
        "target": 2,
    }]
    assert failed_plan["evidence"]["prepared_artifacts"] == []
    assert failed_plan["evidence"]["files_written"] == []

    solution_plan = plan_recovery(
        backend,
        artifacts["solution_output"],
        input_file=artifacts["solution_input"],
        target={
            "expected_charge": -3,
            "expected_multiplicity": 2,
            "expected_metal_elements": ["Fe"],
        },
    )
    assert solution_plan["assessment"]["verdict"]["label"] == (
        "stability_hardening_requires_manual_edit"
    )
    assert solution_plan["evidence"]["state_check"][
        "observed_somo_count"
    ] == 1
    assert solution_plan["evidence"]["prepared_artifacts"] == []
    assert solution_plan["evidence"]["trigger_evidence"][
        "multi_stage_input"
    ] is True


@pytest.mark.skipif(
    not os.environ.get("CHEMTOOLS_REFERENCE_CORPUS"),
    reason="CHEMTOOLS_REFERENCE_CORPUS is not configured",
)
def test_ferrocene_lock_distinguishes_unstable_and_controlled_scf_paths():
    manifest = load_reference_manifest(NWCHEM_MANIFEST)
    verified = verify_reference_case(
        manifest,
        "nwchem.ferrocene_basis_stepping",
        os.environ["CHEMTOOLS_REFERENCE_CORPUS"],
    )
    assert verified["outcome"] == "verified"
    artifacts = {
        artifact["id"]: artifact["path"]
        for artifact in verified["artifacts"]
    }
    backend = load_backend(BUILTIN_BACKENDS[0])

    inspections = {
        name: inspect_run(
            backend,
            artifacts[f"{name}_output"],
            resolved_by="explicit",
            artifact_files=[artifacts[f"{name}_input"]],
        )
        for name in ("failed", "small_basis", "solution")
    }

    assert {
        name: inspection["assessment"]["verdict"]["label"]
        for name, inspection in inspections.items()
    } == {
        "failed": "success",
        "small_basis": "success",
        "solution": "success",
    }
    assert [
        anchor["message"]
        for anchor in inspections["failed"]["evidence"]["diagnosis_anchors"]
    ] == [
        "SCF converged after a transient +16.700000 Ha energy increase at "
        "iteration 10 (DIIS error 133); the convergence path was unstable."
    ]
    assert [
        anchor["message"]
        for anchor in inspections["small_basis"]["evidence"]["diagnosis_anchors"]
    ] == [
        "SCF converged after a transient +17.200000 Ha energy increase at "
        "iteration 10 (DIIS error 133); the convergence path was unstable."
    ]
    assert inspections["solution"]["evidence"]["diagnosis_anchors"] == []

    recovery_plans = {
        name: plan_recovery(
            backend,
            artifacts[f"{name}_output"],
            input_file=artifacts[f"{name}_input"],
        )
        for name in ("failed", "small_basis", "solution")
    }
    assert {
        name: plan["assessment"]["verdict"]["label"]
        for name, plan in recovery_plans.items()
    } == {
        "failed": "stability_hardening_available",
        "small_basis": "stability_hardening_available",
        "solution": "no_recovery_needed",
    }
    assert {
        name: [item["code"] for item in plan["uncertainty"]]
        for name, plan in recovery_plans.items()
    } == {
        "failed": [],
        "small_basis": [],
        "solution": ["input_output_mismatch"],
    }
    assert {
        name: plan["evidence"]["trigger_evidence"].get(
            "scf_instability",
            {},
        ).get("delta_e_hartree")
        for name, plan in recovery_plans.items()
    } == {
        "failed": 16.7,
        "small_basis": 17.2,
        "solution": None,
    }

    failed_prepared = recovery_plans["failed"]["evidence"][
        "prepared_artifacts"
    ]
    assert len(failed_prepared) == 1
    assert failed_prepared[0]["summary"]["stabilization_strategy"] == (
        "converged_instability_hardening"
    )
    candidate = failed_prepared[0]["candidate_drafts"][0]["text"]
    assert "convergence damp 70" in candidate
    assert "convergence ncydp 25" in candidate
    assert "smear" not in candidate
    assert "vectors input ferrocene_tzvp_failed.movecs" in candidate
    assert "#" not in candidate
    assert recovery_plans["failed"]["evidence"]["files_written"] == []

    alternatives = recovery_plans["small_basis"]["evidence"][
        "trigger_evidence"
    ]["strategy_options"]
    assert [option["name"] for option in alternatives] == [
        "reuse_converged_vectors_with_damping",
        "smaller_basis_projection",
    ]
    assert alternatives[1]["status"] == "manual_fallback"


@pytest.mark.skipif(
    not os.environ.get("CHEMTOOLS_REFERENCE_CORPUS"),
    reason="CHEMTOOLS_REFERENCE_CORPUS is not configured",
)
def test_crco6_lock_distinguishes_saddle_from_minimum():
    manifest = load_reference_manifest(NWCHEM_MANIFEST)
    verified = verify_reference_case(
        manifest,
        "nwchem.crco6_bailar_twist",
        os.environ["CHEMTOOLS_REFERENCE_CORPUS"],
    )
    assert verified["outcome"] == "verified"
    artifacts = {
        artifact["id"]: artifact["path"]
        for artifact in verified["artifacts"]
    }
    backend = load_backend(BUILTIN_BACKENDS[0])

    saddle = inspect_run(
        backend,
        artifacts["saddle_output"],
        resolved_by="explicit",
        artifact_files=[artifacts["saddle_input"]],
    )
    minimum = inspect_run(
        backend,
        artifacts["minimum_output"],
        resolved_by="explicit",
        artifact_files=[artifacts["minimum_input"]],
    )

    assert saddle["assessment"]["verdict"] == {
        "label": "frequency_interpretation_required",
        "confidence": 0.6,
        "reasons": ["imaginary modes present", "stage: frequency"],
    }
    assert saddle["evidence"]["derived"] == {
        "n_tasks": 2,
        "final_energy_hartree": -1723.872731354654,
        "primary_energy_hartree": -1723.872731354654,
        "n_imaginary_modes": 1,
        "significant_imaginary_frequencies_cm1": [-234.96],
        "n_near_zero_modes": 6,
    }
    assert minimum["assessment"]["verdict"]["label"] == "success"
    assert minimum["evidence"]["derived"] == {
        "n_tasks": 2,
        "final_energy_hartree": -1723.925434241469,
        "primary_energy_hartree": -1723.925434241469,
        "n_imaginary_modes": 0,
        "significant_imaginary_frequencies_cm1": [],
        "n_near_zero_modes": 6,
    }

    recovery = plan_recovery(
        backend,
        artifacts["saddle_output"],
        input_file=artifacts["saddle_input"],
    )
    assert recovery["assessment"]["verdict"]["label"] == (
        "recovery_plan_ready"
    )
    assert recovery["evidence"]["plan_kind"] == (
        "imaginary_mode_follow_up"
    )
    assert recovery["evidence"]["files_written"] == []
    prepared = recovery["evidence"]["prepared_artifacts"]
    assert len(prepared) == 1
    assert prepared[0]["kind"] == "imaginary_mode_restarts"
    assert prepared[0]["selected_mode"]["frequency_cm1"] == -234.96
    assert [draft["label"] for draft in prepared[0]["candidate_drafts"]] == [
        "plus",
        "minus",
    ]
    assert all(
        draft["text"].endswith("task dft optimize\n")
        for draft in prepared[0]["candidate_drafts"]
    )

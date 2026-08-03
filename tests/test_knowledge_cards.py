"""Exact contracts for curated knowledge cards and their YAML boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from chemtools.knowledge.cards import (
    KNOWLEDGE_CARD_SCHEMA,
    KnowledgeCard,
    KnowledgeCardLoadError,
    bundled_card_directory,
    load_knowledge_card,
    load_knowledge_cards,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _accepted_card(**changes):
    card = {
        "schema_version": KNOWLEDGE_CARD_SCHEMA,
        "id": "cross_program.test_rule",
        "programs": ["*"],
        "workflows": ["run_validation"],
        "kind": "validation",
        "status": "accepted",
        "confidence": "high",
        "applies_when": {"reported_status": "success"},
        "claim": "Require independent evidence.",
        "check": {"requirement": "independent_evidence"},
        "failure": {"severity": "warning"},
        "sources": ["notes/fblock/README.md#the-five-things-that-generalize"],
        "tests": ["tests/test_knowledge_cards.py"],
    }
    card.update(changes)
    return card


def _reference_path(reference: str) -> Path:
    path_text = reference.split("#", 1)[0].split("::", 1)[0]
    return REPOSITORY_ROOT / path_text


def test_bundled_cards_are_valid_and_traceable():
    cards = load_knowledge_cards()

    assert [
        (card.id, card.status, card.programs)
        for card in cards
    ] == [
        (
            "cross_program.cheap_invariants_find_wrong_basins",
            "draft",
            ("*",),
        ),
        (
            "cross_program.optimizer_failure_sentinel_must_lose",
            "accepted",
            ("*",),
        ),
        (
            "cross_program.same_producer_is_correlated",
            "accepted",
            ("*",),
        ),
        (
            "cross_program.same_starting_guess_class_is_one_measurement",
            "accepted",
            ("*",),
        ),
        ("cross_program.silent_success", "draft", ("*",)),
        (
            "grasp.rmcdhf.zero_exit_requires_convergence",
            "accepted",
            ("grasp",),
        ),
        (
            "pyscf.electron_spin_consistency_is_runtime_required",
            "accepted",
            ("pyscf",),
        ),
        (
            "pyscf.scf_convergence_is_separate_from_execution",
            "accepted",
            ("pyscf",),
        ),
        (
            "qmcpack.determinant_only_vmc_offsets",
            "accepted",
            ("qmcpack",),
        ),
        (
            "qmcpack.fblock_dmc_reference_protocol",
            "accepted",
            ("qmcpack",),
        ),
        (
            "qmcpack.jastrow_vmc_energy_gate",
            "accepted",
            ("qmcpack",),
        ),
        (
            "qmcpack.variational_parameter_sidecar",
            "accepted",
            ("qmcpack",),
        ),
    ]
    by_id = {card.id: card for card in cards}
    cheap_invariants = by_id[
        "cross_program.cheap_invariants_find_wrong_basins"
    ]
    assert cheap_invariants.applies_when == {
        "expected_relation": "scientifically_scoped",
        "compared_values": "recorded_and_comparable",
    }
    optimizer_sentinel = by_id[
        "cross_program.optimizer_failure_sentinel_must_lose"
    ]
    assert optimizer_sentinel.applies_when == {
        "failed_evaluation": "represented_by_finite_numeric_sentinel",
        "valid_objective_bounds": "scientifically_scoped",
    }
    silent_success = by_id["cross_program.silent_success"]
    assert silent_success.schema_version == KNOWLEDGE_CARD_SCHEMA
    assert silent_success.workflows == ("run_validation",)
    assert silent_success.confidence == "high"
    assert silent_success.applies_when == {
        "reported_status": "success",
        "independent_verification": "absent",
    }
    grasp = by_id["grasp.rmcdhf.zero_exit_requires_convergence"]
    assert grasp.applies_when == {
        "executable": "rmcdhf",
        "process_exit_code": 0,
    }
    pyscf_convergence = by_id[
        "pyscf.scf_convergence_is_separate_from_execution"
    ]
    assert pyscf_convergence.applies_when == {
        "runner_operation": "pyscf_single_point",
        "execution_status": "completed",
    }
    pyscf_electrons = by_id[
        "pyscf.electron_spin_consistency_is_runtime_required"
    ]
    assert pyscf_electrons.applies_when == {
        "runner_operation": "pyscf_single_point",
        "electronic_state": "charge_and_multiplicity_declared",
    }
    provenance = by_id["cross_program.same_producer_is_correlated"]
    assert provenance.applies_when == {
        "compared_artifacts": "multiple",
        "direct_producer_records": "available",
    }
    starting_class = by_id[
        "cross_program.same_starting_guess_class_is_one_measurement"
    ]
    assert starting_class.applies_when == {
        "compared_runs": "multiple",
        "starting_guess_class": "recorded",
    }
    determinant_only_vmc = by_id[
        "qmcpack.determinant_only_vmc_offsets"
    ]
    assert determinant_only_vmc.applies_when == {
        "wavefunction": "determinant_only",
        "compared_states": "multiple",
        "trial_scf_energies": "matched",
    }
    assert by_id["qmcpack.fblock_dmc_reference_protocol"].kind == "workflow"
    for card in cards:
        assert all(
            _reference_path(source).is_file() for source in card.sources
        )
        assert all(_reference_path(test).is_file() for test in card.tests)


def test_knowledge_card_round_trip_is_exact():
    card = KnowledgeCard.from_dict(_accepted_card())

    assert KnowledgeCard.from_dict(card.to_dict()) == card
    assert card.to_dict() == _accepted_card()


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"schema_version": "chemtools.knowledge-card/2"}, "unsupported"),
        ({"programs": ["*", "nwchem"]}, "cannot be combined"),
        ({"applies_when": {}}, "at least one condition"),
        ({"sources": ["../private-note.md"]}, "inside the repository"),
        ({"tests": []}, "accepted cards must cite at least one test"),
        ({"unexpected": True}, "unknown knowledge-card fields"),
    ),
)
def test_knowledge_card_rejects_unsafe_or_incomplete_fields(changes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        KnowledgeCard.from_dict(_accepted_card(**changes))


def test_yaml_loader_rejects_non_json_yaml_types(tmp_path):
    path = tmp_path / "dated.yaml"
    path.write_text(
        "\n".join((
            "schema_version: chemtools.knowledge-card/1",
            "id: cross_program.dated",
            'programs: ["*"]',
            "workflows: [run_validation]",
            "kind: validation",
            "status: draft",
            "confidence: low",
            "applies_when:",
            "  observed_on: 2026-07-31",
            "claim: YAML dates are not JSON values.",
            "sources: []",
            "tests: []",
        )),
        encoding="utf-8",
    )

    with pytest.raises(KnowledgeCardLoadError, match="unsupported YAML type date"):
        load_knowledge_card(path)


def test_directory_loader_rejects_filename_id_mismatch(tmp_path):
    source = bundled_card_directory() / "cross_program.silent_success.yaml"
    path = tmp_path / "wrong-name.yaml"
    path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(KnowledgeCardLoadError, match="must be named"):
        load_knowledge_cards(tmp_path)

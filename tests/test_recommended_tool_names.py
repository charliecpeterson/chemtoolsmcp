"""Keep first-party recommendation payloads on canonical MCP tool names."""

from __future__ import annotations

import ast
from pathlib import Path

from chemtools.programs.nwchem.strategy import recovery


ROOT = Path(__file__).resolve().parents[1]
DEPRECATED_RECOMMENDATION_NAMES = {
    "review_nwchem_case",
    "suggest_nwchem_scf_fix_strategy",
    "suggest_nwchem_state_recovery_strategy",
}


def _deprecated_recommendations(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if (
                    keyword.arg == "tool"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value in DEPRECATED_RECOMMENDATION_NAMES
                ):
                    found.append((keyword.value.lineno, keyword.value.value))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "tool"
                    and isinstance(value, ast.Constant)
                    and value.value in DEPRECATED_RECOMMENDATION_NAMES
                ):
                    found.append((value.lineno, value.value))

    return found


def test_first_party_recommendations_do_not_emit_deprecated_tool_names():
    violations = {}
    for path in sorted((ROOT / "chemtools").rglob("*.py")):
        found = _deprecated_recommendations(path)
        if found:
            violations[str(path.relative_to(ROOT))] = found

    assert violations == {}


def test_scf_strategy_routes_state_recovery_through_canonical_tool(monkeypatch):
    monkeypatch.setattr(
        recovery,
        "diagnose_nwchem_output",
        lambda **_: {
            "failure_class": "wrong_state_convergence",
            "task_outcome": "success",
            "scf": {"status": "success"},
            "state_check": {},
        },
    )

    planned = recovery.suggest_nwchem_scf_fix_strategy("run.out")

    assert [
        (strategy["tool"], strategy.get("params"))
        for strategy in planned["strategies"]
    ] == [
        ("prepare_nwchem_next_step", None),
        ("suggest_nwchem_recovery", {"mode": "state"}),
        ("suggest_nwchem_recovery", {"mode": "state"}),
    ]


def test_state_strategy_routes_scf_recovery_through_canonical_tool(monkeypatch):
    monkeypatch.setattr(
        recovery,
        "diagnose_nwchem_output",
        lambda **_: {
            "failure_class": "wrong_state_convergence",
            "task_outcome": "success",
            "state_check": {
                "spin_density_summary": {
                    "dominant_site": {"element": "Fe", "atom_index": 1},
                },
                "metal_like_somo_count": 0,
                "ligand_like_somo_count": 2,
                "somo_count": 2,
            },
        },
    )
    monkeypatch.setattr(
        recovery,
        "check_spin_charge_state",
        lambda **_: {
            "assessment": "suspicious",
            "expected_somo_count": 2,
            "state_check_assessment": "metal_state_mismatch_suspected",
        },
    )

    planned = recovery.suggest_nwchem_state_recovery_strategy(
        "run.out",
        expected_metal_elements=["Fe"],
    )

    assert planned["regime"] == "covalent_ligand_hole_candidate"
    assert [
        (strategy["tool"], strategy.get("params"))
        for strategy in planned["strategies"]
    ] == [
        ("suggest_nwchem_recovery", {"mode": "scf"}),
        ("suggest_nwchem_recovery", {"mode": "scf"}),
        ("create_nwchem_dft_input_from_request", None),
    ]


def test_plausible_state_review_uses_canonical_analysis_tool(monkeypatch):
    monkeypatch.setattr(
        recovery,
        "diagnose_nwchem_output",
        lambda **_: {
            "failure_class": "no_clear_failure_detected",
            "task_outcome": "success",
            "state_check": {},
        },
    )
    monkeypatch.setattr(
        recovery,
        "check_spin_charge_state",
        lambda **_: {
            "assessment": "plausible",
            "expected_somo_count": 0,
            "state_check_assessment": "plausible",
        },
    )

    planned = recovery.suggest_nwchem_state_recovery_strategy("run.out")

    assert planned["strategies"] == [
        {
            "name": "accept_or_verify_state",
            "priority": 1,
            "rationale": (
                "The current spin/frontier signals are internally consistent "
                "with the requested state."
            ),
            "tool": "analyze_nwchem_case",
            "docs_topics": ["scf_open_shell"],
            "when_to_use": (
                "Use when the main remaining question is chemical "
                "interpretation, not state rescue."
            ),
        }
    ]

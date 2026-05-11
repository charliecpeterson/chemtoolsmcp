"""NWChem Strategist sub-protocol implementation.

Adapter layer that translates the existing analysis functions in
chemtools.programs.nwchem.strategy.* and chemtools.api_strategy into the
program-neutral Diagnosis envelope from chemtools.core.types.

The Strategist is where the "thick tools, thin LLM" goal lands: every method
that an agent calls returns a verdict + ready-to-execute next_actions, so a
small model can chain tool calls deterministically.

This is a minimal first pass — it wires up the existing functions and produces
correctly-shaped Diagnosis envelopes. Enriching next_actions[] with concrete
tool invocations is follow-up work tracked in REFACTOR.md.
"""

from __future__ import annotations
from typing import Any

from chemtools.core.types import (
    ParsedRun,
    Diagnosis,
    NextAction,
    Verdict,
    DiagnosticAnchor,
)
from chemtools.programs.nwchem.strategy.diagnose import diagnose_nwchem_output


# Map the existing diagnose_nwchem_output.task_outcome / failure_class /
# recommended_next_action triple into a Verdict + NextAction list.

_CONFIDENCE_MAP: dict[str, float] = {
    "high":   0.85,
    "medium": 0.6,
    "low":    0.35,
}


def _build_verdict(diag: dict[str, Any]) -> Verdict:
    """Synthesize a Verdict from the existing diagnosis result."""
    outcome = diag.get("task_outcome") or "unknown"
    failure_class = diag.get("failure_class")
    likely_cause = diag.get("likely_cause")

    # Label: prefer failure_class when it's informative, else task_outcome.
    if failure_class and failure_class != "no_clear_failure_detected":
        label = failure_class
    else:
        label = outcome

    reasons: list[str] = []
    if likely_cause and likely_cause != "run_completed_normally":
        reasons.append(likely_cause.replace("_", " "))
    stage = diag.get("stage")
    if stage and stage != "other":
        reasons.append(f"stage: {stage}")

    return {
        "label": label,
        "confidence": _CONFIDENCE_MAP.get((diag.get("confidence") or "").lower(), 0.5),
        "reasons": reasons,
    }


def _build_next_actions(diag: dict[str, Any]) -> list[NextAction]:
    """Build a primary recommended action from the diagnosis.

    The existing diagnose_nwchem_output returns `recommended_next_action` as
    a short string like "verify_state_quality_before_accepting_result". We
    surface it as a NextAction with priority 1; the agent must still
    translate the action label into a concrete tool call (until the
    suggest_recovery follow-up below provides the concrete params).
    """
    rec = diag.get("recommended_next_action")
    if not rec or rec == "no_action_required":
        return []
    return [
        {
            "tool": _action_to_tool(rec),
            "params": {},
            "reason": rec.replace("_", " "),
            "confidence": _CONFIDENCE_MAP.get((diag.get("confidence") or "").lower(), 0.5),
            "priority": 1,
        }
    ]


# Map the existing recommended_next_action enum to the MCP tool that
# implements it. Incomplete — expand as new actions surface.
_ACTION_TO_TOOL: dict[str, str] = {
    "verify_state_quality_before_accepting_result": "analyze_nwchem_frontier_orbitals",
    "rerun_with_damping": "suggest_nwchem_recovery",
    "swap_vectors_and_rerun": "draft_nwchem_vectors_swap_input",
    "check_active_space": "suggest_nwchem_mcscf_active_space",
    "restart_freq": "prepare_nwchem_freq_restart",
    "displace_along_imaginary_mode": "displace_nwchem_geometry_along_mode",
}


def _action_to_tool(action: str) -> str:
    return _ACTION_TO_TOOL.get(action, "analyze_nwchem_case")


class _NwchemStrategist:
    """Implements chemtools.core.program.Strategist for NWChem."""

    def diagnose(self, parsed: ParsedRun) -> Diagnosis:
        path = parsed.get("file")
        if not path:
            return {
                "verdict": {"label": "missing_file_path", "confidence": 0.0, "reasons": []},
                "next_actions": [],
                "anchors": [],
            }
        diag = diagnose_nwchem_output(path)

        verdict = _build_verdict(diag)
        next_actions = _build_next_actions(diag)

        anchors: list[DiagnosticAnchor] = []
        # Pass through any existing line-anchored diagnostics from ParsedRun.
        for d in parsed.get("diagnostics") or []:
            anchors.append({
                "kind": d.get("kind", "info"),
                "message": d.get("message", ""),
                "line": d.get("line"),
                "file": d.get("file") or path,
            })

        return {
            "verdict": verdict,
            "next_actions": next_actions,
            "anchors": anchors,
        }

    def suggest_recovery(
        self, parsed: ParsedRun, diagnosis: Diagnosis
    ) -> list[NextAction]:
        # Lazy import — api_strategy still flat, splits later.
        from chemtools.api_strategy import suggest_nwchem_recovery
        path = parsed.get("file")
        if not path:
            return []
        result = suggest_nwchem_recovery(path)
        # Existing tool returns a dict of strategies; surface each as a NextAction.
        actions: list[NextAction] = []
        for category in ("scf_strategies", "state_strategies"):
            for entry in (result.get(category) or {}).get("entries", []) or []:
                actions.append({
                    "tool": entry.get("tool") or "suggest_nwchem_recovery",
                    "params": entry.get("params") or {},
                    "reason": entry.get("description") or entry.get("rationale") or category,
                    "confidence": 0.6,
                    "priority": entry.get("priority") or 2,
                })
        return actions

    def suggest_resources(
        self, input_path: str, profile: dict[str, Any]
    ) -> dict[str, Any]:
        from chemtools.api_strategy import suggest_hpc_resources
        profile_name = profile.get("name") if isinstance(profile, dict) else profile
        # Some existing implementations take a profile name; pass through.
        return suggest_hpc_resources(input_path=input_path, profile=profile_name)

    def progress_summary(self, output_path: str) -> dict[str, Any]:
        # _build_nwchem_progress_summary lives in core/runner.py (it's the
        # function the planned progress_summary_fn callback will replace).
        from chemtools.core.runner import _build_nwchem_progress_summary
        from chemtools.core.common import read_text
        contents = read_text(output_path)
        return _build_nwchem_progress_summary(output_path, contents, None)


NWCHEM_STRATEGIST = _NwchemStrategist()


__all__ = ["NWCHEM_STRATEGIST"]

"""NWChem Strategist sub-protocol implementation.

Adapter layer that translates the existing analysis and workflow functions
into program-neutral diagnosis and recovery-plan envelopes.

The Strategist is where the "thick tools, thin LLM" goal lands: every method
that an agent calls returns a verdict + ready-to-execute next_actions, so a
small model can chain tool calls deterministically.

Recovery planning is read-only. Candidate input text may be returned, but the
adapter never writes it to disk.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

from chemtools.core.types import (
    ParsedRun,
    Diagnosis,
    NextAction,
    Verdict,
    DiagnosticAnchor,
)
from chemtools.programs.nwchem.strategy.diagnose import diagnose_nwchem_output
from chemtools.programs.nwchem.parse.input import inspect_nwchem_input
from chemtools.programs.nwchem.strategy.diagnose import summarize_nwchem_output
from chemtools.programs.nwchem.strategy.workflow_planner import (
    prepare_nwchem_next_step,
)
from chemtools.programs.nwchem.scf_quality import (
    find_converged_scf_excursion,
)


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
    translate the action label into a concrete tool call. Recovery requests
    should use the target-aware plan_recovery method below.
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


def _build_anchors(diag: dict[str, Any], path: str) -> list[DiagnosticAnchor]:
    anchors: list[DiagnosticAnchor] = []
    for diagnostic in (diag.get("tasks") or {}).get(
        "program_summary",
        {},
    ).get("diagnostics", []):
        anchors.append({
            "kind": diagnostic.get("kind", "info"),
            "message": diagnostic.get("message", ""),
            "line": diagnostic.get("line"),
            "file": path,
        })

    instability = find_converged_scf_excursion(diag.get("scf") or {})
    if instability is None:
        return anchors
    message = (
        "SCF converged after a transient "
        f"+{instability['delta_e_hartree']:.6f} Ha energy increase at iteration "
        f"{instability['iteration']}"
    )
    diis_error = instability.get("diis_error")
    if isinstance(diis_error, (int, float)):
        message += f" (DIIS error {diis_error:g})"
    message += "; the convergence path was unstable."
    anchors.append({
        "kind": "warning",
        "message": message,
        "line": None,
        "file": path,
    })
    return anchors


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


def _compact_diagnosis(diagnosis: dict[str, Any]) -> dict[str, Any]:
    return {
        key: diagnosis.get(key)
        for key in (
            "stage",
            "task_outcome",
            "failure_class",
            "likely_cause",
            "recommended_next_action",
            "confidence",
        )
    }


def _compact_state_check(diagnosis: dict[str, Any]) -> dict[str, Any]:
    state = diagnosis.get("state_check") or {}
    return {
        "assessment": state.get("assessment"),
        "observed_somo_count": state.get("somo_count"),
        "expected_somo_count": state.get("expected_somo_count"),
        "metal_like_somo_count": state.get("metal_like_somo_count"),
        "ligand_like_somo_count": state.get("ligand_like_somo_count"),
    }


def _input_state(input_path: str | None) -> dict[str, Any] | None:
    if input_path is None:
        return None
    inspected = inspect_nwchem_input(input_path)
    task_states = inspected.get("task_states") or []
    final_task_state = task_states[-1] if task_states else {}
    charge = final_task_state.get("charge")
    multiplicity = final_task_state.get("multiplicity")
    multiplicity_source = final_task_state.get("multiplicity_source")
    return {
        "charge": charge if charge is not None else inspected.get("charge"),
        "multiplicity": (
            multiplicity
            if multiplicity is not None
            else inspected.get("multiplicity")
        ),
        "multiplicity_source": (
            multiplicity_source
            if multiplicity_source is not None
            else inspected.get("multiplicity_source")
        ),
        "transition_metals": inspected.get("transition_metals") or [],
    }


def _target_state_mismatches(
    input_state: dict[str, Any] | None,
    target: dict[str, Any],
) -> list[dict[str, Any]]:
    if input_state is None:
        return []
    mismatches = []
    for field in ("charge", "multiplicity"):
        expected = target.get(f"expected_{field}")
        observed = input_state.get(field)
        if expected is not None and observed is not None and expected != observed:
            mismatches.append({
                "field": field,
                "input": observed,
                "target": expected,
            })
    return mismatches


def _draft_payload(label: str, text: str) -> dict[str, Any]:
    return {
        "label": label,
        "text": text,
        "size_bytes": len(text.encode("utf-8")),
        "line_count": len(text.splitlines()),
    }


def _normalize_prepared_artifacts(
    workflow: dict[str, Any],
) -> list[dict[str, Any]]:
    normalized = []
    prepared = workflow.get("prepared_artifacts") or {}
    summaries = workflow.get("prepared_artifact_summaries") or {}
    for name in workflow.get("artifact_order") or []:
        payload = prepared.get(name) or {}
        entry: dict[str, Any] = {
            "kind": name,
            "summary": summaries.get(name) or {},
            "candidate_drafts": [],
        }
        for key, label in (
            ("input_text", "candidate"),
            ("plus_input_text", "plus"),
            ("minus_input_text", "minus"),
        ):
            text = payload.get(key)
            if isinstance(text, str) and text.strip():
                entry["candidate_drafts"].append(
                    _draft_payload(label, text)
                )
        mode = payload.get("selected_mode")
        if isinstance(mode, dict):
            entry["selected_mode"] = {
                key: mode.get(key)
                for key in (
                    "mode_number",
                    "frequency_cm1",
                    "is_imaginary",
                    "dominant_axis",
                    "axis_character",
                    "locality",
                    "motion_type",
                    "motion_rationale",
                    "recommended_action",
                )
            }
        vectors_input = payload.get("vectors_input")
        if isinstance(vectors_input, str) and vectors_input != "atomic":
            source_input = Path(payload["input_file"])
            vectors_path = Path(vectors_input)
            if not vectors_path.is_absolute():
                vectors_path = source_input.parent / vectors_path
            vectors_path = vectors_path.resolve()
            entry["required_artifacts"] = [{
                "kind": "vectors_input",
                "path": str(vectors_path),
                "exists": vectors_path.is_file(),
            }]
        else:
            entry["required_artifacts"] = []
        entry["ready_to_run"] = all(
            artifact["exists"]
            for artifact in entry["required_artifacts"]
        )
        normalized.append(entry)
    return normalized


def _workflow_assessment(workflow: dict[str, Any]) -> dict[str, Any]:
    selected = workflow["selected_workflow"]
    if selected == "scf_stability_hardening":
        instability = workflow["trigger_evidence"]["scf_instability"]
        return {
            "verdict": {
                "label": (
                    "stability_hardening_available"
                    if workflow["can_auto_prepare"]
                    else "stability_hardening_requires_manual_edit"
                ),
                "confidence": 0.85,
                "reasons": [
                    "The SCF converged, but its largest transient energy "
                    f"increase was +{instability['delta_e_hartree']:.6f} Ha."
                ],
            }
        }
    if workflow["can_auto_prepare"]:
        return {
            "verdict": {
                "label": "recovery_plan_ready",
                "confidence": 0.8,
                "reasons": [
                    f"Prepared the read-only {selected} candidate workflow."
                ],
            }
        }
    if selected == "verification_only":
        return {
            "verdict": {
                "label": "no_recovery_needed",
                "confidence": 0.8,
                "reasons": ["No matching failure requires an automatic repair."],
            }
        }
    return {
        "verdict": {
            "label": "manual_review_required",
            "confidence": 0.5,
            "reasons": [
                f"The {selected} workflow could not prepare a candidate input."
            ],
        }
    }


def _workflow_actions(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    if workflow["selected_workflow"] == "scf_stability_hardening":
        actions = [{
            "action": "accept_result_with_stability_warning",
            "reason": (
                "The task completed; hardening is for a repeat, batch, "
                "or less forgiving related system."
            ),
            "priority": 1,
        }]
        actions.append({
            "action": (
                "review_stability_hardening_candidate"
                if workflow["can_auto_prepare"]
                else "edit_multistage_stability_hardening_manually"
            ),
            "reason": (
                "Reuse the converged vectors with damping if a controlled "
                "repeat is scientifically useful."
            ),
            "priority": 2,
        })
        return actions
    if workflow["can_auto_prepare"]:
        return [
            {
                "action": "review_candidate_drafts",
                "reason": (
                    "Check the proposed input text and scientific assumptions "
                    "before saving or running it."
                ),
                "priority": 1,
            },
            {
                "action": "save_selected_input",
                "reason": "Save only the reviewed candidate chosen for the retry.",
                "priority": 2,
            },
        ]
    if workflow["selected_workflow"] == "verification_only":
        return []
    return [{
        "action": "manual_recovery_review",
        "reason": "The available evidence does not support an automatic draft.",
        "priority": 1,
    }]


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

        anchors = _build_anchors(diag, path)

        return {
            "verdict": verdict,
            "next_actions": next_actions,
            "anchors": anchors,
        }

    def plan_recovery(
        self,
        output_path: str,
        input_path: str | None,
        target: dict[str, Any],
    ) -> dict[str, Any]:
        expected_metals = target.get("expected_metal_elements") or None
        expected_somos = target.get("expected_somo_count")
        summary = summarize_nwchem_output(
            output_path=output_path,
            input_path=input_path,
            expected_metal_elements=expected_metals,
            expected_somo_count=expected_somos,
            detail_level="full",
        )
        diagnosis = summary["diagnosis"]
        input_state = _input_state(input_path)
        mismatches = _target_state_mismatches(input_state, target)

        if mismatches:
            fields = ", ".join(item["field"] for item in mismatches)
            return {
                "assessment": {
                    "verdict": {
                        "label": "target_state_rebuild_required",
                        "confidence": 0.95,
                        "reasons": [
                            f"The supplied input does not match the target {fields}."
                        ],
                    }
                },
                "evidence": {
                    "plan_kind": "target_state_rebuild",
                    "can_prepare": False,
                    "diagnosis": _compact_diagnosis(diagnosis),
                    "input_state": input_state,
                    "state_check": _compact_state_check(diagnosis),
                    "target_mismatches": mismatches,
                    "prepared_artifacts": [],
                    "files_written": [],
                },
                "uncertainty": [{
                    "code": "fragment_guess_builder_unavailable",
                    "message": (
                        "Chemtools can inspect and lint fragment-guess inputs, "
                        "but it does not have a general fragment-guess builder."
                    ),
                    "impact": (
                        "The target-state input must be rebuilt and reviewed "
                        "instead of applying an automatic vectors swap."
                    ),
                }],
                "next_actions": [
                    {
                        "action": "rebuild_input_for_target_state",
                        "reason": (
                            "Charge or multiplicity changes require a fresh "
                            "state specification; an orbital swap preserves them."
                        ),
                        "priority": 1,
                    },
                    {
                        "action": "review_fragment_guess_initialization",
                        "reason": (
                            "Use a documented fragment guess when ordinary "
                            "atomic initialization returns to the wrong basin."
                        ),
                        "priority": 2,
                    },
                ],
            }

        workflow = prepare_nwchem_next_step(
            output_path=output_path,
            input_path=input_path,
            expected_metal_elements=expected_metals,
            expected_somo_count=expected_somos,
            write_files=False,
            include_property_check=False,
            include_frontier_cubes=False,
            _precomputed_summary=summary,
        )
        prepared_artifacts = _normalize_prepared_artifacts(workflow)
        uncertainty = []
        if (
            input_path is None
            and not workflow["can_auto_prepare"]
            and workflow["selected_workflow"] != "verification_only"
        ):
            uncertainty.append({
                "code": "input_file_not_supplied",
                "message": "No matching NWChem input file was supplied.",
                "impact": "Recovery input text could not be prepared.",
            })
        missing_checkpoints = [
            artifact
            for prepared in prepared_artifacts
            for artifact in prepared["required_artifacts"]
            if not artifact["exists"]
        ]
        if missing_checkpoints:
            uncertainty.append({
                "code": "required_checkpoint_missing",
                "message": (
                    "A candidate input references a checkpoint that is not "
                    "present beside the supplied input."
                ),
                "impact": (
                    "The candidate can be reviewed but cannot run until the "
                    "declared vectors file is supplied or regenerated."
                ),
            })
        return {
            "assessment": _workflow_assessment(workflow),
            "evidence": {
                "plan_kind": workflow["selected_workflow"],
                "can_prepare": workflow["can_auto_prepare"],
                "diagnosis": workflow["diagnosis"],
                "input_state": input_state,
                "state_check": _compact_state_check(diagnosis),
                "prepared_artifacts": prepared_artifacts,
                "trigger_evidence": workflow.get("trigger_evidence") or {},
                "notes": workflow["notes"],
                "files_written": [],
            },
            "uncertainty": uncertainty,
            "next_actions": _workflow_actions(workflow),
        }

    def suggest_resources(
        self, input_path: str, profile: dict[str, Any]
    ) -> dict[str, Any]:
        from chemtools.programs.nwchem.strategy.hpc_resources import (
            suggest_hpc_resources,
        )
        profile_name = profile.get("name") if isinstance(profile, dict) else profile
        # Some existing implementations take a profile name; pass through.
        return suggest_hpc_resources(input_path=input_path, profile=profile_name)

    def progress_summary(self, output_path: str) -> dict[str, Any]:
        from chemtools.core.common import read_text
        from chemtools.programs.nwchem.strategy.progress import (
            build_progress_summary,
            parse_progress_state,
        )

        contents = read_text(output_path)
        parsed = parse_progress_state(contents, output_path)
        progress = build_progress_summary(contents, parsed)
        program_summary = parsed.get("program_summary") or {}
        progress["outcome"] = program_summary.get("outcome")
        progress["task_count"] = program_summary.get("task_count")
        return progress


NWCHEM_STRATEGIST = _NwchemStrategist()


__all__ = ["NWCHEM_STRATEGIST"]

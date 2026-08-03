"""Turn parsed pw.x evidence into deterministic run diagnoses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from chemtools.core.types import Diagnosis, ParsedRun


class _QeDiagnostics:
    def diagnose(self, parsed: ParsedRun) -> Diagnosis:
        diagnosis = self._diagnose_run(parsed)
        return _attach_trajectory_findings(diagnosis, parsed)

    def _diagnose_run(self, parsed: ParsedRun) -> Diagnosis:
        derived = parsed.get("derived") or {}
        anchors = list(parsed.get("diagnostics") or [])
        if derived.get("qe:program") == "pw2qmcpack":
            errors = derived.get("qe:pw2qmcpack_errors") or []
            if errors:
                primary = errors[0]
                labels = {
                    "wavefunctions_not_collected": (
                        "converter_wavefunctions_not_collected"
                    ),
                    "gamma_trick_unsupported": "converter_gamma_trick_unsupported",
                }
                return {
                    "verdict": {
                        "label": labels.get(
                            primary.get("kind"), "converter_runtime_error"
                        ),
                        "confidence": 0.99,
                        "reasons": [primary["message"]],
                    },
                    "next_actions": [],
                    "anchors": anchors,
                }
            products = derived.get("qe:pw2qmcpack_hdf5_artifacts") or []
            if products and derived.get("qe:job_done"):
                return {
                    "verdict": {
                        "label": "converter_completed",
                        "confidence": 0.98,
                        "reasons": [
                            "pw2qmcpack reported creating "
                            f"{product['path']}, then printed JOB DONE."
                            for product in products
                        ],
                    },
                    "next_actions": [],
                    "anchors": anchors,
                }
            if products:
                return {
                    "verdict": {
                        "label": "converter_artifact_reported",
                        "confidence": 0.8,
                        "reasons": [
                            "pw2qmcpack reported creating "
                            f"{product['path']}."
                            for product in products
                        ],
                    },
                    "next_actions": [],
                    "anchors": anchors,
                }
            return {
                "verdict": {
                    "label": "converter_output_incomplete",
                    "confidence": 0.85,
                    "reasons": [
                        "pw2qmcpack output did not report an emitted HDF5 artifact."
                    ],
                },
                "next_actions": [],
                "anchors": anchors,
            }
        errors = derived.get("qe:runtime_errors") or []
        mode = derived.get("qe:calculation_mode") or "unknown"

        if errors:
            primary = errors[0]
            pseudo_missing = (
                str(primary.get("routine", "")).lower() == "readpp"
                and "not found" in str(primary.get("message", "")).lower()
            )
            label = (
                "pseudopotential_not_found"
                if pseudo_missing
                else "pw_runtime_error"
            )
            reasons = [
                f"pw.x reported {primary.get('routine', 'a runtime')} error: "
                f"{primary.get('message', 'unspecified error')}"
            ]
            if primary.get("occurrences", 1) > 1:
                reasons.append(
                    "The same error was repeated by "
                    f"{primary['occurrences']} MPI ranks."
                )
            return {
                "verdict": {
                    "label": label,
                    "confidence": 0.98 if pseudo_missing else 0.95,
                    "reasons": reasons,
                },
                "next_actions": _review_input_action(parsed, derived, reasons[0]),
                "anchors": anchors,
            }

        nonconvergence = derived.get("qe:scf_nonconvergence")
        if nonconvergence is not None:
            label = (
                "relaxation_interrupted_by_scf_nonconvergence"
                if mode in {"relax", "vc-relax"}
                else "scf_not_converged"
            )
            reason = (
                "The final electronic cycle stopped after "
                f"{nonconvergence['iterations']} iterations without convergence."
            )
            return {
                "verdict": {
                    "label": label,
                    "confidence": 0.99,
                    "reasons": [reason],
                },
                "next_actions": _review_input_action(parsed, derived, reason),
                "anchors": anchors,
            }

        job_done = bool(derived.get("qe:job_done"))
        tasks = parsed.get("tasks") or []
        outcome = tasks[0].get("outcome") if tasks else "unknown"
        if mode in {"relax", "vc-relax"} and outcome == "success":
            bfgs = derived.get("qe:bfgs") or {}
            return {
                "verdict": {
                    "label": "relaxation_converged",
                    "confidence": 0.98,
                    "reasons": [
                        "BFGS convergence was reported after "
                        f"{bfgs.get('steps')} steps, and pw.x ended cleanly."
                    ],
                },
                "next_actions": [],
                "anchors": anchors,
            }
        if mode == "scf" and outcome == "success":
            return {
                "verdict": {
                    "label": "scf_converged",
                    "confidence": 0.99,
                    "reasons": [
                        "pw.x reported SCF convergence and a JOB DONE marker."
                    ],
                },
                "next_actions": [],
                "anchors": anchors,
            }
        if mode == "bands_or_nscf" and outcome == "success":
            return {
                "verdict": {
                    "label": "bands_or_nscf_completed",
                    "confidence": 0.75,
                    "reasons": [
                        "The band-structure calculation ended cleanly, but the "
                        "output alone does not distinguish bands from NSCF."
                    ],
                },
                "next_actions": [],
                "anchors": anchors,
            }
        if (
            job_done
            and mode in {"relax", "vc-relax"}
            and derived.get("qe:relaxation_algorithm") == "bfgs"
        ):
            reason = "pw.x ended cleanly without reporting BFGS convergence."
            return {
                "verdict": {
                    "label": "relaxation_not_converged",
                    "confidence": 0.9,
                    "reasons": [reason],
                },
                "next_actions": _review_input_action(parsed, derived, reason),
                "anchors": anchors,
            }
        if job_done and mode in {"relax", "vc-relax"}:
            return {
                "verdict": {
                    "label": "relaxation_completion_unresolved",
                    "confidence": 0.7,
                    "reasons": [
                        "pw.x ended cleanly, but the output has no supported "
                        "relaxation convergence marker."
                    ],
                },
                "next_actions": [],
                "anchors": anchors,
            }
        return {
            "verdict": {
                "label": "incomplete",
                "confidence": 0.85,
                "reasons": ["The output does not establish scientific completion."],
            },
            "next_actions": [],
            "anchors": anchors,
        }


def _attach_trajectory_findings(
    diagnosis: Diagnosis,
    parsed: Mapping[str, Any],
) -> Diagnosis:
    derived = parsed.get("derived") or {}
    trajectory = derived.get("qe:trajectory") or {}
    analysis = trajectory.get("structural_analysis") or {}
    structural_verdict = analysis.get("verdict") or {}
    concerning = structural_verdict.get("status") == "concerning"
    findings = []
    if concerning:
        findings.extend(
            "Trajectory structural concern "
            f"({finding['origin'].replace('_', ' ')}): {finding['message']}"
            for finding in structural_verdict["findings"]
        )
    findings.extend(
        f"Trajectory observation: {observation['message']}"
        for observation in analysis.get("observations") or []
        if observation.get("message")
    )
    if not findings:
        return diagnosis

    diagnosis["verdict"]["reasons"].extend(findings)
    output_file = parsed.get("file")
    if output_file:
        action = {
            "tool": "parse_trajectory",
            "params": {
                "output_file": str(output_file),
                "program": "qe",
            },
            "reason": (
                "Inspect the full geometry history and structural metrics "
                "before accepting or restarting this calculation."
            ),
            "confidence": 0.95 if concerning else 0.9,
            "priority": 1 if concerning else 2,
        }
        if concerning:
            diagnosis["next_actions"].insert(0, action)
        else:
            diagnosis["next_actions"].append(action)
    return diagnosis


def _review_input_action(
    parsed: Mapping[str, Any],
    derived: Mapping[str, Any],
    reason: str,
) -> list[dict[str, Any]]:
    input_path = _related_input_path(parsed, derived)
    if input_path is None:
        return []
    return [
        {
            "tool": "review_input",
            "params": {"input_file": str(input_path), "program": "qe"},
            "reason": (
                f"Review the input associated with this failure. {reason}"
            ),
            "confidence": 0.95,
            "priority": 1,
        }
    ]


def _related_input_path(
    parsed: Mapping[str, Any], derived: Mapping[str, Any]
) -> Path | None:
    output = Path(str(parsed.get("file", "")))
    input_name = derived.get("qe:input_file")
    candidates: list[Path] = []
    if input_name and str(input_name).lower() != "standard input":
        named = Path(str(input_name))
        candidates.append(named if named.is_absolute() else output.parent / named)
    candidates.append(output.with_suffix(".in"))
    return next((path.resolve() for path in candidates if path.is_file()), None)


QE_DIAGNOSTICS = _QeDiagnostics()


__all__ = ["QE_DIAGNOSTICS"]

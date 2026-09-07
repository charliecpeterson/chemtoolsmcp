"""GRASP atomic MCDHF and RCI strategy provider."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

from chemtools.core.common import ELEMENT_TO_Z


_HAMILTONIANS = {
    "dirac_coulomb",
    "dirac_coulomb_plus_zero_frequency_breit",
    "dirac_coulomb_plus_transverse_interaction",
}
_SUBSTITUTIONS = {"none", "S", "SD"}
_SUBSHELL_RE = re.compile(r"^[1-9][0-9]*([spdfghi])(-|\+)?$")
_L_INDEX = {label: index for index, label in enumerate("spdfghi")}

_REQUIRED_OPTIONS = (
    (
        "isotope_mass_number",
        "Which isotope mass number should rnucleus use?",
        "The isotope fixes the default nuclear radius and mass data.",
    ),
    (
        "nuclear_model",
        "What Fermi-nucleus parameters and mass treatment should be used?",
        "Both codes need the same final nuclear charge distribution.",
    ),
    (
        "reference_configurations",
        "Which configurations define the DHF and minimal-CAS references?",
        "Configuration labels determine the relativistic CSF space.",
    ),
    (
        "inactive_core_subshells",
        "Which relativistic subshells stay inactive?",
        "The frozen-core boundary must match the comparison calculation.",
    ),
    (
        "inactive_core_electrons",
        "How many electrons are assigned to the inactive core?",
        "Core and active electron counts must reproduce the atomic charge.",
    ),
    (
        "active_subshells",
        "Which relativistic subshells form the minimal active space?",
        "Relativistic subshells, spinors, determinants, and CSFs are distinct counts.",
    ),
    (
        "active_electrons",
        "How many electrons occupy the active spinors?",
        "This defines the CAS electron count.",
    ),
    (
        "target",
        "Which 2J, parity, and ASF roots should be targeted?",
        "Multiplicity alone does not select a relativistic atomic block.",
    ),
    (
        "correlation",
        "Which substitutions and active layers should add correlation?",
        "The model must distinguish the minimal CAS from added S/SD correlation.",
    ),
    (
        "orbital_optimization",
        "Which radial orbitals are varied or frozen at each layer?",
        "MCDHF lowering and fixed-orbital CI lowering answer different questions.",
    ),
    (
        "hamiltonian_variants",
        "Which matched Dirac-Coulomb and Breit variants should be run?",
        "Breit and QED terms must be reported as separate Hamiltonian choices.",
    ),
    (
        "numerical_controls",
        "Which radial-grid and SCF controls should be held fixed?",
        "Numerical-accuracy comparisons need explicit controls and stop diagnostics.",
    ),
)


class _GraspCalculationPlanner:
    def plan_calculation(
        self,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if len(request["elements"]) != 1:
            raise ValueError("GRASP planning requires exactly one atomic element")
        if tuple(request["stages"]) != ("energy",):
            raise ValueError(
                "GRASP planning currently supports stages=['energy']; "
                "atomic geometry optimization and frequencies do not apply"
            )
        method = str(request.get("method") or "").casefold()
        if method and not any(
            token in method for token in ("dhf", "mcdhf", "rci", "grasp")
        ):
            raise ValueError(
                "GRASP planning supports DHF, MCDHF, and RCI methods; "
                f"received {request['method']!r}"
            )

        options = dict(request.get("program_options") or {})
        _validate_supplied_options(options)
        decisions = _required_decisions(options)
        model_space = _model_space(request, options) if not decisions else None
        planned_stages = _stages(options)
        return {
            "protocol": {
                "name": "atomic_mcdhf_rci_accuracy_ladder",
                "description": (
                    "Build a DHF reference, a minimal CAS-MCDHF state, "
                    "explicit correlation layers, and fixed-orbital RCI "
                    "Hamiltonian variants."
                ),
                "model_space": model_space,
                "result_artifacts": {
                    "rmcdhf": ".sum",
                    "rci": ".csum",
                    "execution_logs": ["rmcdhf stdout", "rci stdout"],
                },
            },
            "stages": planned_stages,
            "required_decisions": decisions,
            "assumptions": _assumptions(request),
            "verdict": {
                "label": (
                    "needs_scientific_decisions"
                    if decisions
                    else "ready_for_grasp_workflow_planning"
                ),
                "confidence": 0.95,
                "reasons": [
                    (
                        f"{len(decisions)} GRASP model decision(s) remain."
                        if decisions
                        else (
                            "The atomic state, electron partition, radial "
                            "policy, numerical controls, and Hamiltonian "
                            "variants are explicit."
                        )
                    )
                ],
            },
            "next_actions": [
                {
                    "action": "prepare_grasp_workflow_stages",
                    "tools": [
                        "plan_grasp_dhf_workflow",
                        "run_grasp_workflow",
                    ],
                    "reason": (
                        "Generate and verify each CSF block before accepting its "
                        "dimension, then preserve .sum and .csum as the result "
                        "artifacts."
                    ),
                    "priority": 1,
                }
            ],
        }


def _required_decisions(options: Mapping[str, Any]) -> list[dict[str, str]]:
    decisions = [
        {"field": f"program_options.{field}", "question": question, "reason": reason}
        for field, question, reason in _REQUIRED_OPTIONS
        if field not in options
    ]
    nucleus = options.get("nuclear_model")
    if isinstance(nucleus, Mapping):
        for field, question in (
            ("type", "Which nuclear charge-distribution model should be used?"),
            ("mass_treatment", "Should the nucleus be static or finite mass?"),
            ("bohr_radius_fm", "Which Bohr-radius conversion should be matched?"),
        ):
            if field not in nucleus:
                decisions.append(
                    {
                        "field": f"program_options.nuclear_model.{field}",
                        "question": question,
                        "reason": "Record the exact nuclear convention used by both codes.",
                    }
                )
        has_final_fermi = all(
            field in nucleus for field in ("fermi_a_fm", "fermi_c_fm")
        )
        has_radius_inputs = all(
            field in nucleus for field in ("skin_thickness_fm", "rms_charge_radius_fm")
        )
        if not has_final_fermi and not has_radius_inputs:
            decisions.append(
                {
                    "field": "program_options.nuclear_model.fermi_parameters",
                    "question": (
                        "What final Fermi a/c values, or RMS radius plus skin "
                        "thickness, should be matched?"
                    ),
                    "reason": "The isotope label and skin thickness alone do not fix c.",
                }
            )
        if nucleus.get("mass_treatment") == "finite" and "mass_amu" not in nucleus:
            decisions.append(
                {
                    "field": "program_options.nuclear_model.mass_amu",
                    "question": "Which finite nuclear mass should GRASP use?",
                    "reason": "The recoil convention must be explicit.",
                }
            )
    target = options.get("target")
    if isinstance(target, Mapping):
        for field in ("two_j", "parity", "asf_selections"):
            if field not in target:
                decisions.append(
                    {
                        "field": f"program_options.target.{field}",
                        "question": f"What target {field} should GRASP use?",
                        "reason": "The requested relativistic ASF block must be explicit.",
                    }
                )
    correlation = options.get("correlation")
    if isinstance(correlation, Mapping):
        for field in ("substitutions", "active_layers"):
            if field not in correlation:
                decisions.append(
                    {
                        "field": f"program_options.correlation.{field}",
                        "question": f"What correlation {field} should GRASP use?",
                        "reason": (
                            "Use an explicit none value when no post-CAS "
                            "correlation layer is intended."
                        ),
                    }
                )
    return decisions


def _validate_supplied_options(options: Mapping[str, Any]) -> None:
    _positive_integer(options, "isotope_mass_number")
    _positive_integer(options, "inactive_core_electrons", allow_zero=True)
    _positive_integer(options, "active_electrons")
    for field in (
        "reference_configurations",
        "inactive_core_subshells",
        "active_subshells",
        "hamiltonian_variants",
    ):
        if field in options and not _string_list(
            options[field], allow_empty=(field == "inactive_core_subshells")
        ):
            raise ValueError(f"program_options.{field} must be a list of strings")
    for field in ("inactive_core_subshells", "active_subshells"):
        labels = options.get(field)
        if not isinstance(labels, list):
            continue
        if len(labels) != len(set(labels)):
            raise ValueError(f"program_options.{field} contains duplicate subshells")
        for label in labels:
            _subshell_capacity(label)
    inactive = set(options.get("inactive_core_subshells") or [])
    active = set(options.get("active_subshells") or [])
    overlap = sorted(inactive & active)
    if overlap:
        raise ValueError("inactive and active subshells overlap: " + ", ".join(overlap))
    if "nuclear_model" in options:
        nucleus = options["nuclear_model"]
        if not isinstance(nucleus, Mapping):
            raise ValueError("program_options.nuclear_model must be an object")
        if nucleus.get("type") not in (None, "fermi"):
            raise ValueError("program_options.nuclear_model.type must be 'fermi'")
        if nucleus.get("mass_treatment") not in (None, "static", "finite"):
            raise ValueError(
                "program_options.nuclear_model.mass_treatment must be "
                "'static' or 'finite'"
            )
        for field in (
            "bohr_radius_fm",
            "fermi_a_fm",
            "fermi_c_fm",
            "skin_thickness_fm",
            "rms_charge_radius_fm",
            "mass_amu",
        ):
            _positive_number(nucleus, field, prefix="nuclear_model.")
    if "target" in options:
        target = options["target"]
        if not isinstance(target, Mapping):
            raise ValueError("program_options.target must be an object")
        two_j = target.get("two_j")
        if two_j is not None and (
            isinstance(two_j, bool) or not isinstance(two_j, int) or two_j < 0
        ):
            raise ValueError(
                "program_options.target.two_j must be a nonnegative integer"
            )
        if target.get("parity") not in (None, "+", "-"):
            raise ValueError("program_options.target.parity must be '+' or '-'")
        selections = target.get("asf_selections")
        if selections is not None and not _string_list(selections):
            raise ValueError(
                "program_options.target.asf_selections must be a list of strings"
            )
    if "correlation" in options:
        correlation = options["correlation"]
        if not isinstance(correlation, Mapping):
            raise ValueError("program_options.correlation must be an object")
        substitutions = correlation.get("substitutions")
        if substitutions not in (None, *_SUBSTITUTIONS):
            raise ValueError("correlation substitutions must be one of: none, S, SD")
        layers = correlation.get("active_layers")
        if layers is not None and (
            not isinstance(layers, list)
            or any(not _string_list(layer) for layer in layers)
        ):
            raise ValueError(
                "correlation active_layers must be lists of subshell labels"
            )
        if isinstance(layers, list):
            for layer in layers:
                if len(layer) != len(set(layer)):
                    raise ValueError(
                        "correlation active_layers contain duplicate subshells"
                    )
                for label in layer:
                    _subshell_capacity(label)
    if "hamiltonian_variants" in options:
        variants = options["hamiltonian_variants"]
        if len(variants) != len(set(variants)):
            raise ValueError("hamiltonian_variants contains duplicates")
        unknown = sorted(set(variants) - _HAMILTONIANS)
        if unknown:
            raise ValueError(f"unsupported GRASP Hamiltonian variants: {unknown}")
    for field in ("orbital_optimization", "numerical_controls"):
        if field in options and not isinstance(options[field], Mapping):
            raise ValueError(f"program_options.{field} must be an object")


def _model_space(
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> dict[str, Any]:
    element = request["elements"][0]
    total_electrons = ELEMENT_TO_Z[element] - request["charge"]
    inactive_electrons = int(options["inactive_core_electrons"])
    active_electrons = int(options["active_electrons"])
    if inactive_electrons + active_electrons != total_electrons:
        raise ValueError(
            "inactive_core_electrons + active_electrons must equal "
            f"Z - charge ({total_electrons})"
        )
    active_spinors = sum(
        _subshell_capacity(label) for label in options["active_subshells"]
    )
    if active_electrons > active_spinors:
        raise ValueError("active_electrons exceeds the active spinor count")
    return {
        "element": element,
        "atomic_number": ELEMENT_TO_Z[element],
        "charge": request["charge"],
        "electrons": total_electrons,
        "inactive_core_subshells": list(options["inactive_core_subshells"]),
        "inactive_core_electrons": inactive_electrons,
        "active_subshells": list(options["active_subshells"]),
        "active_electrons": active_electrons,
        "active_spinors": active_spinors,
        "raw_active_determinants": math.comb(active_spinors, active_electrons),
        "grasp_csf_count": "verify_after_rcsfgenerate",
        "target": dict(options["target"]),
    }


def _stages(options: Mapping[str, Any]) -> list[dict[str, Any]]:
    stages = [
        _stage(1, "nuclear_setup", None, ["fermi_parameters_recorded"]),
        _stage(2, "dhf_reference", "nuclear_setup", ["scf_converged", "sum_parsed"]),
        _stage(
            3,
            "minimal_cas_mcscf",
            "dhf_reference",
            ["csf_count_recorded", "sum_parsed"],
        ),
    ]
    previous = "minimal_cas_mcscf"
    correlation = options.get("correlation")
    layers = (
        correlation.get("active_layers", []) if isinstance(correlation, Mapping) else []
    )
    for index, layer in enumerate(layers, start=1):
        stage_id = f"correlation_layer_{index}"
        stages.append(
            {
                **_stage(
                    len(stages) + 1,
                    stage_id,
                    previous,
                    ["csf_count_recorded", "scf_converged", "sum_parsed"],
                ),
                "active_layer": list(layer),
                "substitutions": correlation.get("substitutions"),
            }
        )
        previous = stage_id
    stages.append(
        {
            **_stage(
                len(stages) + 1,
                "fixed_orbital_rci_variants",
                previous,
                [
                    "paired_model_space_verified",
                    "csum_parsed",
                    "hamiltonian_terms_recorded",
                ],
            ),
            "hamiltonian_variants": list(options.get("hamiltonian_variants", [])),
        }
    )
    return stages


def _stage(
    sequence: int,
    stage_id: str,
    depends_on: str | None,
    completion_checks: list[str],
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "id": stage_id,
        "kind": "energy",
        "depends_on": depends_on,
        "completion_checks": completion_checks,
    }


def _assumptions(request: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "code": "relativistic_state_labels_control_sector",
            "message": (
                f"The requested multiplicity {request['multiplicity']} is "
                "recorded, but GRASP uses 2J, parity, and ASF selections."
            ),
            "impact": "Compare the same relativistic state sector in MPQC.",
        },
        {
            "code": "summary_files_are_authoritative",
            "message": "Final RMCDHF and RCI energies come from .sum and .csum files.",
            "impact": "Stdout is used for convergence and execution diagnostics.",
        },
        {
            "code": "rci_orbitals_are_fixed",
            "message": "Each RCI Hamiltonian variant uses the preceding MCDHF orbitals.",
            "impact": "RCI differences isolate Hamiltonian terms at a fixed model space.",
        },
    ]


def _positive_integer(
    options: Mapping[str, Any],
    field: str,
    *,
    allow_zero: bool = False,
) -> None:
    if field not in options:
        return
    value = options[field]
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"program_options.{field} must be an integer >= {minimum}")


def _positive_number(
    values: Mapping[str, Any],
    field: str,
    *,
    prefix: str = "",
) -> None:
    if field not in values:
        return
    value = values[field]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"program_options.{prefix}{field} must be a positive finite number"
        )


def _string_list(value: Any, *, allow_empty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def _subshell_capacity(label: str) -> int:
    match = _SUBSHELL_RE.fullmatch(label.strip())
    if match is None:
        raise ValueError(f"invalid relativistic subshell label: {label!r}")
    angular_momentum = _L_INDEX[match.group(1)]
    if angular_momentum == 0:
        if match.group(2) == "-":
            raise ValueError("an s subshell cannot use the '-' branch")
        return 2
    return 2 * angular_momentum if match.group(2) == "-" else 2 * angular_momentum + 2


GRASP_CALCULATION_PLANNER = _GraspCalculationPlanner()


__all__ = ["GRASP_CALCULATION_PLANNER"]

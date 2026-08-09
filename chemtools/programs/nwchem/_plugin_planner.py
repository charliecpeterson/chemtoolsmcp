"""NWChem calculation-strategy provider for pre-input planning."""

from __future__ import annotations

from typing import Any, Mapping

from chemtools.core.common import ELEMENT_TO_Z
from chemtools.programs.nwchem.protocols import PROTOCOLS


_PROTOCOL_BY_STAGES = {
    ("energy",): "single_point_dft",
    ("optimize",): "geometry_opt_dft",
    ("frequency",): "freq_only",
    ("optimize", "frequency"): "thermochem_dft",
}

_PURPOSE_BY_STAGE = {
    "energy": "Evaluate the electronic energy at the supplied geometry.",
    "optimize": "Locate a stationary geometry for the requested electronic state.",
    "frequency": (
        "Evaluate the Hessian at the optimized geometry and classify the "
        "stationary point."
    ),
}


class _NwchemCalculationPlanner:
    def plan_calculation(
        self,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        method = str(request.get("method") or "").strip().casefold()
        if method and method != "dft":
            raise ValueError(
                "the current NWChem planning recipes support method='dft'; "
                f"received {request['method']!r}"
            )
        stages = tuple(request["stages"])
        protocol_name = _PROTOCOL_BY_STAGES.get(stages)
        if protocol_name is None:
            supported = [list(item) for item in _PROTOCOL_BY_STAGES]
            raise ValueError(
                f"unsupported NWChem stage sequence {list(stages)!r}; "
                f"supported sequences: {supported!r}"
            )

        protocol = PROTOCOLS[protocol_name]
        decisions = _required_decisions(request)
        return {
            "protocol": {
                "name": protocol_name,
                "description": protocol["description"],
            },
            "stages": _stages(protocol, stages),
            "required_decisions": decisions,
            "assumptions": _assumptions(request),
            "verdict": {
                "label": (
                    "needs_scientific_decisions"
                    if decisions
                    else "ready_for_input_drafting"
                ),
                "confidence": 0.9,
                "reasons": [
                    (
                        f"{len(decisions)} scientific decision(s) remain "
                        "before input drafting."
                    )
                    if decisions
                    else (
                        "The requested stages and required scientific choices "
                        "are specified."
                    )
                ],
            },
        }


def _stages(
    protocol: Mapping[str, Any],
    requested: tuple[str, ...],
) -> list[dict[str, Any]]:
    planned = []
    for index, (stage_name, recipe) in enumerate(
        zip(requested, protocol["steps"]),
        start=1,
    ):
        checks = []
        if stage_name == "optimize":
            checks = ["optimization_converged", "geometry_is_plausible"]
        elif stage_name == "frequency":
            checks = [
                "frequency_task_completed",
                "stationary_point_classified",
            ]
        elif stage_name == "energy":
            checks = ["electronic_structure_converged", "energy_reported"]
        planned.append({
            "sequence": index,
            "id": recipe["id"],
            "kind": stage_name,
            "program_task": recipe.get("task"),
            "depends_on": recipe.get("depends_on"),
            "purpose": _PURPOSE_BY_STAGE[stage_name],
            "completion_checks": checks,
        })
    return planned


def _required_decisions(request: Mapping[str, Any]) -> list[dict[str, str]]:
    decisions = []
    if not request.get("geometry_source"):
        decisions.append({
            "field": "geometry_source",
            "question": "Which starting geometry should be used?",
            "reason": "The first calculation stage still needs coordinates and units.",
        })
    if not request.get("method"):
        decisions.append({
            "field": "method",
            "question": "Which electronic-structure method should all stages use?",
            "reason": "The NWChem recipe is a stage template, not a method choice.",
        })
    method = str(request.get("method") or "").casefold()
    if (not method or "dft" in method) and not request.get("functional"):
        decisions.append({
            "field": "functional",
            "question": "Which density functional should be used?",
            "reason": (
                "The protocol library's B3LYP value is a default, not an "
                "accepted choice."
            ),
        })
    if not request.get("basis"):
        decisions.append({
            "field": "basis",
            "question": "Which basis should be assigned to each element?",
            "reason": (
                "Basis quality and element coverage must be settled before "
                "drafting."
            ),
        })

    heavy_elements = [
        symbol
        for symbol in request["elements"]
        if ELEMENT_TO_Z[symbol] > 36
    ]
    if (
        heavy_elements
        and not request.get("relativistic")
        and not request.get("ecp")
    ):
        decisions.append({
            "field": "relativistic",
            "question": (
                "How should relativistic effects and core electrons be treated "
                f"for {', '.join(heavy_elements)}?"
            ),
            "reason": "The relativistic model, ECP, and basis must be compatible.",
        })
    if request["multiplicity"] > 1 and not request.get("state_strategy"):
        decisions.append({
            "field": "state_strategy",
            "question": (
                "How will the requested open-shell state be initialized and "
                "checked?"
            ),
            "reason": (
                "Multiplicity alone does not establish that optimization and "
                "frequency stages follow the intended SCF solution."
            ),
        })
    return decisions


def _assumptions(request: Mapping[str, Any]) -> list[dict[str, str]]:
    assumptions = []
    if not request.get("solvent"):
        assumptions.append({
            "code": "gas_phase_assumed",
            "message": (
                "No solvent model was specified, so the plan assumes gas phase."
            ),
            "impact": (
                "Add a solvent choice before drafting if the target is not gas "
                "phase."
            ),
        })
    if "frequency" in request["stages"]:
        assumptions.append({
            "code": "harmonic_frequency_assumed",
            "message": "The frequency stage is treated as a harmonic analysis.",
            "impact": (
                "Temperature, pressure, scaling, and standard-state choices "
                "remain separate if thermochemical corrections are needed."
            ),
        })
    return assumptions


NWCHEM_CALCULATION_PLANNER = _NwchemCalculationPlanner()


__all__ = ["NWCHEM_CALCULATION_PLANNER"]

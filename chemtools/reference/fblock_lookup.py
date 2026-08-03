"""Public projections for exact lookup in the f-block atomic catalog.

The lookup keeps method scope and review status beside every scientific value.
It does not resolve approximate configurations or infer missing seed lineage.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from chemtools.reference.fblock import load_fblock_catalog
from chemtools.reference.fblock_models import (
    FBlockComponent,
    FBlockDatasetMetadata,
    FBlockElement,
    FBlockState,
)


FBLOCK_LOOKUP_SCHEMA = "chemtools.fblock-reference-lookup/1"
MAX_STATE_SUMMARIES = 64
_CATALOG_COMPONENT_ID = "grasp_v2_catalog"
_STATE_SLUG_RE = re.compile(r"^ion[0-9]+_[a-z0-9]+$")


@dataclass(frozen=True)
class FBlockLookupResult:
    metadata: FBlockDatasetMetadata
    component: FBlockComponent
    element: FBlockElement
    state: FBlockState | None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": FBLOCK_LOOKUP_SCHEMA,
            "query": {
                "element": self.element.symbol,
                "state": self.state.slug if self.state is not None else None,
            },
            "reference": _reference_dict(self.metadata, self.component),
            "element": _element_dict(self.element),
        }
        if self.state is None:
            states = self.element.states[:MAX_STATE_SUMMARIES]
            payload["state_index"] = {
                "total_count": len(self.element.states),
                "returned_count": len(states),
                "truncated": len(states) < len(self.element.states),
                "states": [_state_summary(state) for state in states],
            }
        else:
            payload["state"] = _state_dict(self.state)
        return payload


def lookup_grasp_fblock_state(
    element: str,
    state: str | None = None,
) -> FBlockLookupResult:
    state_slug = _state_slug(state)
    catalog = load_fblock_catalog()
    try:
        element_record = catalog.element(element)
    except KeyError:
        available = ", ".join(item.symbol for item in catalog.elements)
        raise ValueError(
            f"no f-block reference for element {element!r}; "
            f"available elements: {available}"
        ) from None

    state_record = None
    if state_slug is not None:
        try:
            state_record = element_record.state(state_slug)
        except KeyError:
            available = ", ".join(item.slug for item in element_record.states)
            raise ValueError(
                f"no f-block reference for state "
                f"{element_record.symbol}.{state_slug}; available states: "
                f"{available}"
            ) from None

    return FBlockLookupResult(
        metadata=catalog.metadata,
        component=catalog.metadata.component(_CATALOG_COMPONENT_ID),
        element=element_record,
        state=state_record,
    )


def _state_slug(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("state must be a string")
    if not _STATE_SLUG_RE.fullmatch(value):
        raise ValueError(
            "state must be an exact catalog slug such as 'ion0_6d27s2'"
        )
    return value


def _reference_dict(
    metadata: FBlockDatasetMetadata,
    component: FBlockComponent,
) -> dict[str, object]:
    redistribution = metadata.redistribution
    return {
        "status": component.status,
        "recommendation_eligible": component.status == "validated_reference",
        "dataset": {
            "id": metadata.dataset_id,
            "version": metadata.dataset_version,
            "rebuild_date": metadata.rebuild_date,
            "payload_schema": metadata.catalog_payload_schema,
            "catalog_sha256": metadata.catalog_sha256,
        },
        "component": {
            "id": component.id,
            "purposes": list(component.purposes),
        },
        "review": {
            "reviewed_by": component.reviewed_by,
            "reviewed_at": component.reviewed_at,
            "scope": component.review_scope,
        },
        "programs": [
            {
                "name": program.name,
                "version": program.version,
                "role": program.role,
            }
            for program in metadata.programs
        ],
        "method_scope": dict(metadata.method_scope),
        "redistribution": {
            "status": redistribution.status,
            "source": redistribution.source,
            "attribution": redistribution.attribution,
            "license_identifier": redistribution.license_identifier,
        },
        "known_limitations": list(metadata.known_limitations),
    }


def _element_dict(element: FBlockElement) -> dict[str, object]:
    return {
        "symbol": element.symbol,
        "atomic_number": element.atomic_number,
        "mass_number": element.mass_number,
        "core_label": element.core_label,
        "hamiltonian": element.hamiltonian,
        "nucleus": element.nucleus,
        "comment": element.comment,
        "atsp_hf_seed_default": element.atsp_hf_seed_default,
        "state_count": len(element.states),
    }


def _state_summary(state: FBlockState) -> dict[str, object]:
    return {
        "slug": state.slug,
        "ion": state.ion,
        "config": state.config,
        "role": state.role,
        "seed_class": state.seed_class,
        "staged_birth": state.vary_first is not None,
        "energy_relative_au": state.energy_relative_au,
    }


def _state_dict(state: FBlockState) -> dict[str, object]:
    estimate_from = state.estimate_from
    if isinstance(estimate_from, str):
        donors = [estimate_from]
    elif estimate_from is None:
        donors = []
    else:
        donors = list(estimate_from)
    return {
        "slug": state.slug,
        "ion": state.ion,
        "config": state.config,
        "core": state.core,
        "confline": state.confline,
        "role": state.role,
        "note": state.note,
        "seed": {
            "class": state.seed_class,
            "instruction": state.seeding,
            "hf_seed": state.hf_seed,
            "estimate_from": donors,
            "vary_first": state.vary_first,
        },
        "grasp": {
            "core_menu": state.core_menu,
            "active_set": state.active_set,
            "jrange": state.jrange,
            "j_blocks": list(state.j_blocks),
            "ncsf": list(state.ncsf),
        },
        "energies_au": {
            "dirac_coulomb": state.energy_dc_au,
            "dirac_coulomb_breit": state.energy_dcb_au,
            "relative_to_anchor": state.energy_relative_au,
        },
    }


__all__ = [
    "FBLOCK_LOOKUP_SCHEMA",
    "MAX_STATE_SUMMARIES",
    "FBlockLookupResult",
    "lookup_grasp_fblock_state",
]

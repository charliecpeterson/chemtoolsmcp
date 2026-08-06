"""Validated SCF-transfer semantics tied to the bundled f-block catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chemtools.reference.fblock import (
    _exact_fields,
    _list,
    _load_json,
    _object,
    _text,
    _text_tuple,
    bundled_fblock_directory,
    load_fblock_catalog,
)
from chemtools.reference.fblock_configuration import occupancy_projection
from chemtools.reference.fblock_models import (
    FBlockCatalog,
    FBlockCatalogLoadError,
    FBlockElement,
    FBlockState,
)


FBLOCK_STATE_SEMANTICS_SCHEMA = "chemtools.fblock-state-semantics/1"
MAX_STATE_SEMANTICS_BYTES = 32 * 1024
HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class ClosedAnchorRegime:
    series: str
    bistable: tuple[str, ...]
    constraint_required: tuple[str, ...]


@dataclass(frozen=True)
class FBlockStateSemanticsManifest:
    dataset_id: str
    dataset_version: str
    catalog_sha256: str
    source_path: str
    source_recorded_at: str
    source_scope: str
    regimes: tuple[ClosedAnchorRegime, ...]
    cross_program_policy: str
    d2h_f_p_separation: str
    reproducibility: str
    schema_version: str = FBLOCK_STATE_SEMANTICS_SCHEMA

    def closed_anchor_status(self, element: str) -> str:
        for regime in self.regimes:
            if element in regime.constraint_required:
                return "constraint_required"
            if element in regime.bistable:
                return "bistable"
        return "not_assessed"


def load_fblock_state_semantics(
    directory: str | Path | None = None,
    *,
    catalog: FBlockCatalog | None = None,
) -> FBlockStateSemanticsManifest:
    root = Path(directory) if directory is not None else bundled_fblock_directory()
    try:
        active_catalog = catalog or load_fblock_catalog(root)
        payload = _load_json(
            root / "state-semantics.json",
            MAX_STATE_SEMANTICS_BYTES,
        )
        manifest = _manifest(payload)
        expected = (
            active_catalog.metadata.dataset_id,
            active_catalog.metadata.dataset_version,
            active_catalog.metadata.catalog_sha256,
        )
        actual = (
            manifest.dataset_id,
            manifest.dataset_version,
            manifest.catalog_sha256,
        )
        if actual != expected:
            raise ValueError(
                f"state-semantics dataset link changed: expected {expected}, "
                f"found {actual}"
            )
        known = {element.symbol for element in active_catalog.elements}
        classified = [
            symbol
            for regime in manifest.regimes
            for symbol in regime.bistable + regime.constraint_required
        ]
        if len(classified) != len(set(classified)):
            raise ValueError("closed-anchor regimes contain duplicate elements")
        unknown = sorted(set(classified) - known)
        if unknown:
            raise ValueError(
                f"closed-anchor regimes contain unknown elements: {unknown}"
            )
        return manifest
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        if isinstance(error, FBlockCatalogLoadError):
            raise
        raise FBlockCatalogLoadError(
            f"invalid f-block state semantics at {root}: {error}"
        ) from error


def state_semantics_dict(
    element: FBlockElement,
    state: FBlockState,
    manifest: FBlockStateSemanticsManifest,
) -> dict[str, object]:
    if state.config == "closed":
        state_class = "closed_anchor"
        risk = manifest.closed_anchor_status(element.symbol)
        reason = {
            "constraint_required": (
                "The recorded closed shell is an excited constrained state; "
                "an unconstrained SCF is expected to occupy f orbitals."
            ),
            "bistable": (
                "The recorded closed shell is in a bistable crossing region; "
                "an unconstrained SCF may reach a different occupation."
            ),
            "not_assessed": (
                "The source note does not classify this closed anchor's "
                "unconstrained-SCF basin."
            ),
        }[risk]
    elif state.config in {"4f(1)", "5f(1)", "5d(1)", "6d(1)"}:
        state_class = "one_electron_f_d"
        risk = "constraint_required"
        reason = (
            "D2h occupation constraints do not separate p and f character; "
            "use atomic symmetry with per-(l,m) control and verify populations."
        )
    else:
        state_class = "other_configuration"
        risk = "not_assessed"
        reason = (
            "No cross-program occupation constraint has been validated for "
            "this catalog configuration."
        )
    return {
        "state_class": state_class,
        "unconstrained_scf_risk": risk,
        "reason": reason,
        "occupancy": occupancy_projection(
            config=state.config,
            confline=state.confline,
            core=state.core,
        ),
        "cross_program_transfer": {
            "eligible": False,
            "policy": manifest.cross_program_policy,
            "d2h_f_p_separation": manifest.d2h_f_p_separation,
        },
        "f_d_separation": _f_d_separation(element, state),
        "evidence": {
            "source": manifest.source_path,
            "recorded_at": manifest.source_recorded_at,
            "scope": manifest.source_scope,
        },
        "reproducibility": manifest.reproducibility,
    }


def _f_d_separation(
    element: FBlockElement,
    selected: FBlockState,
) -> dict[str, object] | None:
    pairs = (("4f(1)", "5d(1)"), ("5f(1)", "6d(1)"))
    for f_config, d_config in pairs:
        if selected.config not in {f_config, d_config}:
            continue
        f_states = [
            state
            for state in element.states
            if state.ion == selected.ion and state.config == f_config
        ]
        d_states = [
            state
            for state in element.states
            if state.ion == selected.ion and state.config == d_config
        ]
        if len(f_states) != 1 or len(d_states) != 1:
            return None
        f_state = f_states[0]
        d_state = d_states[0]
        delta_au = d_state.energy_dcb_au - f_state.energy_dcb_au
        return {
            "definition": "E_d_minus_E_f",
            "f_state": f_state.slug,
            "d_state": d_state.slug,
            "delta_au": delta_au,
            "delta_ev": delta_au * HARTREE_TO_EV,
            "energy_field": "dirac_coulomb_breit",
        }
    return None


def _manifest(value: object) -> FBlockStateSemanticsManifest:
    document = _object(value, "state semantics")
    _exact_fields(
        document,
        {
            "schema_version",
            "dataset",
            "source",
            "closed_anchor_regimes",
            "policies",
        },
        "state semantics",
    )
    if document["schema_version"] != FBLOCK_STATE_SEMANTICS_SCHEMA:
        raise ValueError(
            f"unsupported state-semantics schema {document['schema_version']!r}"
        )
    dataset = _object(document["dataset"], "dataset")
    _exact_fields(dataset, {"id", "version", "catalog_sha256"}, "dataset")
    source = _object(document["source"], "source")
    _exact_fields(source, {"path", "recorded_at", "scope"}, "source")
    policies = _object(document["policies"], "policies")
    _exact_fields(
        policies,
        {"cross_program_transfer", "d2h_f_p_separation", "reproducibility"},
        "policies",
    )
    regimes = tuple(
        _regime(item, index)
        for index, item in enumerate(
            _list(document["closed_anchor_regimes"], "closed_anchor_regimes")
        )
    )
    if {regime.series for regime in regimes} != {"4f", "5f"}:
        raise ValueError("closed-anchor regimes must contain 4f and 5f series")
    return FBlockStateSemanticsManifest(
        dataset_id=_text(dataset["id"], "dataset.id"),
        dataset_version=_text(dataset["version"], "dataset.version"),
        catalog_sha256=_text(dataset["catalog_sha256"], "dataset.catalog_sha256"),
        source_path=_text(source["path"], "source.path"),
        source_recorded_at=_text(source["recorded_at"], "source.recorded_at"),
        source_scope=_text(source["scope"], "source.scope"),
        regimes=regimes,
        cross_program_policy=_text(
            policies["cross_program_transfer"],
            "policies.cross_program_transfer",
        ),
        d2h_f_p_separation=_text(
            policies["d2h_f_p_separation"],
            "policies.d2h_f_p_separation",
        ),
        reproducibility=_text(policies["reproducibility"], "policies.reproducibility"),
    )


def _regime(value: object, index: int) -> ClosedAnchorRegime:
    field = f"closed_anchor_regimes[{index}]"
    document = _object(value, field)
    _exact_fields(
        document,
        {"series", "bistable", "constraint_required"},
        field,
    )
    return ClosedAnchorRegime(
        series=_text(document["series"], f"{field}.series"),
        bistable=_text_tuple(
            document["bistable"],
            f"{field}.bistable",
            allow_empty=True,
        ),
        constraint_required=_text_tuple(
            document["constraint_required"],
            f"{field}.constraint_required",
        ),
    )


__all__ = [
    "FBLOCK_STATE_SEMANTICS_SCHEMA",
    "FBlockStateSemanticsManifest",
    "load_fblock_state_semantics",
    "state_semantics_dict",
]

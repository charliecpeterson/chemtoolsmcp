"""Immutable contracts for the committed f-block atomic dataset."""

from __future__ import annotations

from dataclasses import dataclass


FBLOCK_DATASET_SCHEMA = "chemtools.fblock-dataset/1"


class FBlockCatalogLoadError(ValueError):
    """The committed f-block dataset violates its versioned contract."""


@dataclass(frozen=True)
class FBlockProgram:
    name: str
    version: str
    role: str


@dataclass(frozen=True)
class FBlockComponent:
    id: str
    paths: tuple[str, ...]
    status: str
    purposes: tuple[str, ...]
    reviewed_by: str
    reviewed_at: str
    review_scope: str


@dataclass(frozen=True)
class FBlockRedistribution:
    status: str
    source: str
    attribution: str
    license_identifier: str
    license_evidence: str
    permission_evidence: str
    third_party_components: tuple[str, ...]


@dataclass(frozen=True)
class FBlockDatasetMetadata:
    dataset_id: str
    dataset_version: str
    rebuild_date: str
    description: str
    catalog_relative_path: str
    catalog_payload_schema: str
    catalog_size_bytes: int
    catalog_sha256: str
    element_symbols: tuple[str, ...]
    element_count: int
    state_count: int
    role_counts: tuple[tuple[str, int], ...]
    seed_class_counts: tuple[tuple[str, int], ...]
    staged_birth_state_count: int
    programs: tuple[FBlockProgram, ...]
    method_scope: tuple[tuple[str, str], ...]
    components: tuple[FBlockComponent, ...]
    redistribution: FBlockRedistribution
    known_limitations: tuple[str, ...]
    schema_version: str = FBLOCK_DATASET_SCHEMA

    def component(self, component_id: str) -> FBlockComponent:
        matches = tuple(
            component
            for component in self.components
            if component.id == component_id
        )
        if len(matches) != 1:
            raise FBlockCatalogLoadError(
                f"dataset must contain one {component_id!r} component"
            )
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "rebuild_date": self.rebuild_date,
            "element_count": self.element_count,
            "state_count": self.state_count,
            "role_counts": dict(self.role_counts),
            "seed_class_counts": dict(self.seed_class_counts),
            "staged_birth_state_count": self.staged_birth_state_count,
            "catalog_sha256": self.catalog_sha256,
        }


@dataclass(frozen=True)
class FBlockState:
    slug: str
    ion: int
    config: str
    core: str | None
    confline: str
    role: str
    note: str | None
    hf_seed: bool | None
    estimate_from: str | tuple[str, ...] | None
    vary_first: str | None
    core_menu: int | None
    active_set: str
    jrange: str
    j_blocks: tuple[str, ...]
    ncsf: tuple[int, ...]
    energy_dc_au: float
    energy_dcb_au: float
    energy_relative_au: float
    seeding: str

    @property
    def seed_class(self) -> str:
        if self.seeding.startswith("cold"):
            return "cold"
        if self.seeding.startswith("ATSP"):
            return "atsp_hf"
        if self.seeding.startswith("multi-donor"):
            return "multi_donor"
        if self.seeding.startswith("donor"):
            return "donor"
        raise FBlockCatalogLoadError(
            f"state {self.slug!r} has unknown seeding class {self.seeding!r}"
        )


@dataclass(frozen=True)
class FBlockElement:
    symbol: str
    atomic_number: int
    mass_number: int
    core_label: str
    hamiltonian: str
    nucleus: str
    comment: str
    atsp_hf_seed_default: bool
    states: tuple[FBlockState, ...]

    def state(self, slug: str) -> FBlockState:
        for state in self.states:
            if state.slug == slug:
                return state
        raise KeyError(f"unknown f-block state {self.symbol}.{slug}")


@dataclass(frozen=True)
class FBlockCatalog:
    metadata: FBlockDatasetMetadata
    elements: tuple[FBlockElement, ...]

    def element(self, symbol: str) -> FBlockElement:
        normalized = _element_symbol(symbol)
        for element in self.elements:
            if element.symbol == normalized:
                return element
        raise KeyError(f"unknown f-block element {normalized}")

    def state(self, symbol: str, slug: str) -> FBlockState:
        return self.element(symbol).state(slug)


def _element_symbol(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("element symbol must be a non-empty string")
    normalized = value[0].upper() + value[1:].lower()
    if not normalized[0].isupper():
        raise ValueError(f"invalid element symbol {value!r}")
    if len(normalized) == 2 and not normalized[1].islower():
        raise ValueError(f"invalid element symbol {value!r}")
    if not 1 <= len(normalized) <= 2 or not normalized.isalpha():
        raise ValueError(f"invalid element symbol {value!r}")
    return normalized


__all__ = [
    "FBLOCK_DATASET_SCHEMA",
    "FBlockCatalog",
    "FBlockCatalogLoadError",
    "FBlockComponent",
    "FBlockDatasetMetadata",
    "FBlockElement",
    "FBlockProgram",
    "FBlockRedistribution",
    "FBlockState",
]

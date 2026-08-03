"""Exact, provenance-aware plans for cataloged f-block atomic states.

Plans expose recorded inputs and dependencies. Unresolved donor aliases and
unsupported seed preparation remain explicit manual requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from chemtools.reference.fblock import bundled_fblock_directory
from chemtools.reference.fblock_donors import (
    FBlockDonorAliasManifest,
    FBlockDonorAliasRecord,
    load_fblock_donor_alias_manifest,
)
from chemtools.reference.fblock_lookup import (
    FBlockLookupResult,
    lookup_grasp_fblock_state,
)
from chemtools.reference.fblock_models import (
    FBlockCatalogLoadError,
    FBlockComponent,
    FBlockElement,
    FBlockState,
)


FBLOCK_PLAN_SCHEMA = "chemtools.fblock-atomic-plan/1"
MAX_ATSP_RECIPE_BYTES = 64 * 1024
_ATSP_COMPONENT_ID = "atsp_hf_recipes"
_ATSP_METADATA_RE = re.compile(
    r"^`Z = (?P<z>[0-9]+)`, `Z_eff = (?P<z_eff>[0-9.]+)`, "
    r"`n_core = (?P<n_core>[0-9]+)`, `l_max = (?P<l_max>[0-9]+)`  "
    r"\(module `(?P<module>[a-z0-9_]+)`\)$",
    re.MULTILINE,
)
_ATSP_SEED_RE = re.compile(
    r"^Seed state \(run first every evaluation, seeds all others\): "
    r"`(?P<slug>ion[0-9]+_[a-z0-9]+)`$",
    re.MULTILINE,
)
_ATSP_PIN_RE = re.compile(
    r"^- `(?P<state>ion[0-9]+_[a-z0-9]+)` "
    r"\N{LEFTWARDS ARROW} `(?P<donor>ion[0-9]+_[a-z0-9]+)`$",
    re.MULTILINE,
)
_ATSP_ROW_RE = re.compile(
    r"^\| `(?P<slug>ion[0-9]+_[a-z0-9]+)` "
    r"\| `(?P<closed>[^`]*)` \| `(?P<open>[^`]*)` \|$",
    re.MULTILINE,
)
_CLOSED_SHELL_RE = re.compile(r"(?:  [0-9][spdfgh])+")
_OPEN_CONFIG_RE = re.compile(r"(?:[0-9][spdfgh]\([0-9]+\))+")


@dataclass(frozen=True)
class ATSPStateRecipe:
    slug: str
    closed_shells: str
    open_configuration: str

    def stdin_lines(self, symbol: str, atomic_number: int) -> list[str]:
        return [
            f"{symbol},AV,{atomic_number}.",
            self.closed_shells,
            self.open_configuration,
            "all",
            "y",
            "n",
            "y",
            "y",
            "n",
            "99 2",
            "y",
            "n",
            "n",
        ]


@dataclass(frozen=True)
class ATSPElementRecipes:
    symbol: str
    atomic_number: int
    effective_charge: float
    core_electrons: int
    l_max: int
    module: str
    seed_state: str
    donor_pins: tuple[tuple[str, str], ...]
    states: tuple[ATSPStateRecipe, ...]

    def state(self, slug: str) -> ATSPStateRecipe:
        for state in self.states:
            if state.slug == slug:
                return state
        raise FBlockCatalogLoadError(
            f"ATSP2K recipes contain no state {self.symbol}.{slug}"
        )


@dataclass(frozen=True)
class FBlockAtomicPlan:
    lookup: FBlockLookupResult
    atsp: ATSPElementRecipes
    prerequisites: tuple[FBlockState, ...]
    donor_alias_manifest: FBlockDonorAliasManifest
    unresolved_donors: tuple[FBlockDonorAliasRecord, ...]

    def to_dict(self) -> dict[str, object]:
        if self.lookup.state is None:
            raise FBlockCatalogLoadError("atomic plans require an exact state")
        state = self.lookup.state
        lookup_payload = self.lookup.to_dict()
        atsp_component = self.lookup.metadata.component(_ATSP_COMPONENT_ID)
        missing_grasp_fields = _missing_grasp_fields(state)
        requirements = _automation_requirements(
            state,
            self.prerequisites,
            self.unresolved_donors,
            missing_grasp_fields,
        )
        if missing_grasp_fields:
            plan_status = "incomplete_reference_input"
        elif self.unresolved_donors:
            plan_status = "needs_donor_mapping"
        else:
            plan_status = "complete"
        return {
            "schema_version": FBLOCK_PLAN_SCHEMA,
            "query": lookup_payload["query"],
            "plan_status": plan_status,
            "reference": lookup_payload["reference"],
            "target": {
                "element": lookup_payload["element"],
                "state": lookup_payload["state"],
            },
            "dependencies": _dependency_dict(
                self.lookup.element,
                state,
                self.prerequisites,
                self.donor_alias_manifest,
                self.unresolved_donors,
            ),
            "atsp2k": _atsp_dict(
                self.atsp,
                state,
                atsp_component,
            ),
            "grasp2018": _grasp_dict(
                self.lookup.element,
                state,
                missing_grasp_fields,
            ),
            "automation": {
                "status": (
                    "unavailable"
                    if missing_grasp_fields
                    else (
                        "manual_steps_required"
                        if requirements
                        else "input_ready"
                    )
                ),
                "requirements": requirements,
            },
        }


def plan_fblock_atomic_state(
    element: str,
    state: str,
) -> FBlockAtomicPlan:
    if not isinstance(state, str):
        raise TypeError("state must be a string")
    lookup = lookup_grasp_fblock_state(element, state)
    if lookup.state is None:
        raise FBlockCatalogLoadError("atomic plans require an exact state")
    atsp = load_atsp_element_recipes(lookup.element)
    donor_alias_manifest = load_fblock_donor_alias_manifest()
    prerequisites, unresolved = _dependency_order(
        lookup.element,
        lookup.state,
        donor_alias_manifest,
    )
    return FBlockAtomicPlan(
        lookup=lookup,
        atsp=atsp,
        prerequisites=prerequisites,
        donor_alias_manifest=donor_alias_manifest,
        unresolved_donors=unresolved,
    )


def load_atsp_element_recipes(
    element: FBlockElement,
    directory: str | Path | None = None,
) -> ATSPElementRecipes:
    root = Path(directory) if directory is not None else bundled_fblock_directory()
    path = root / "atsp" / f"{element.symbol}.md"
    try:
        size = path.stat().st_size
        if size > MAX_ATSP_RECIPE_BYTES:
            raise ValueError(
                f"ATSP2K recipe exceeds {MAX_ATSP_RECIPE_BYTES} bytes"
            )
        text = path.read_text(encoding="utf-8")
        metadata = _one_match(_ATSP_METADATA_RE, text, "ATSP2K metadata")
        seed = _one_match(_ATSP_SEED_RE, text, "ATSP2K seed state")
        states = tuple(_atsp_state(match) for match in _ATSP_ROW_RE.finditer(text))
        if not states:
            raise ValueError("ATSP2K recipe table is empty")
        slugs = [state.slug for state in states]
        if len(slugs) != len(set(slugs)):
            raise ValueError("ATSP2K recipe table contains duplicate states")
        expected = {state.slug for state in element.states}
        if set(slugs) != expected:
            raise ValueError(
                "ATSP2K state coverage changed: expected "
                f"{sorted(expected)}, found {sorted(slugs)}"
            )
        atomic_number = int(metadata.group("z"))
        if atomic_number != element.atomic_number:
            raise ValueError(
                f"ATSP2K Z={atomic_number} does not match {element.atomic_number}"
            )
        seed_state = seed.group("slug")
        if seed_state not in set(slugs):
            raise ValueError(f"unknown ATSP2K seed state {seed_state!r}")
        donor_pins = tuple(
            (match.group("state"), match.group("donor"))
            for match in _ATSP_PIN_RE.finditer(text)
        )
        unknown_consumers = [
            state
            for state, _donor in donor_pins
            if state not in set(slugs)
        ]
        if unknown_consumers:
            raise ValueError(
                "ATSP2K donor pins reference unknown consumer states: "
                f"{unknown_consumers}"
            )
        return ATSPElementRecipes(
            symbol=element.symbol,
            atomic_number=atomic_number,
            effective_charge=float(metadata.group("z_eff")),
            core_electrons=int(metadata.group("n_core")),
            l_max=int(metadata.group("l_max")),
            module=metadata.group("module"),
            seed_state=seed_state,
            donor_pins=donor_pins,
            states=states,
        )
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        if isinstance(error, FBlockCatalogLoadError):
            raise
        raise FBlockCatalogLoadError(
            f"invalid ATSP2K recipe for {element.symbol}: {error}"
        ) from error


def _one_match(pattern: re.Pattern[str], text: str, field: str) -> re.Match[str]:
    matches = tuple(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"{field} must appear exactly once")
    return matches[0]


def _atsp_state(match: re.Match[str]) -> ATSPStateRecipe:
    closed_shells = match.group("closed")
    open_configuration = match.group("open")
    if not _CLOSED_SHELL_RE.fullmatch(closed_shells):
        raise ValueError(
            f"ATSP2K closed shells have invalid fixed fields: {closed_shells!r}"
        )
    if open_configuration == "(none)":
        open_configuration = ""
    elif not _OPEN_CONFIG_RE.fullmatch(open_configuration):
        raise ValueError(
            f"ATSP2K open configuration is invalid: {open_configuration!r}"
        )
    return ATSPStateRecipe(
        slug=match.group("slug"),
        closed_shells=closed_shells,
        open_configuration=open_configuration,
    )


def _dependency_order(
    element: FBlockElement,
    target: FBlockState,
    donor_alias_manifest: FBlockDonorAliasManifest,
) -> tuple[
    tuple[FBlockState, ...],
    tuple[FBlockDonorAliasRecord, ...],
]:
    by_slug = {state.slug: state for state in element.states}
    ordered: list[FBlockState] = []
    unresolved: list[FBlockDonorAliasRecord] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(state: FBlockState) -> None:
        if state.slug in visiting:
            raise FBlockCatalogLoadError(
                f"donor lineage contains a cycle at {element.symbol}.{state.slug}"
            )
        if state.slug in visited:
            return
        visiting.add(state.slug)
        for donor in _donors(state):
            donor_state = by_slug.get(donor)
            if donor_state is None:
                record = donor_alias_manifest.record(
                    element.symbol,
                    state.slug,
                    donor,
                )
                if record not in unresolved:
                    unresolved.append(record)
                continue
            visit(donor_state)
        visiting.remove(state.slug)
        visited.add(state.slug)
        if state.slug != target.slug:
            ordered.append(state)

    visit(target)
    return tuple(ordered), tuple(unresolved)


def _donors(state: FBlockState) -> tuple[str, ...]:
    if state.estimate_from is None:
        return ()
    if isinstance(state.estimate_from, str):
        return (state.estimate_from,)
    return state.estimate_from


def _dependency_dict(
    element: FBlockElement,
    target: FBlockState,
    prerequisites: tuple[FBlockState, ...],
    donor_alias_manifest: FBlockDonorAliasManifest,
    unresolved: tuple[FBlockDonorAliasRecord, ...],
) -> dict[str, object]:
    by_slug = {state.slug: state for state in element.states}
    return {
        "direct_donors": [
            {
                "identifier": donor,
                "kind": "catalog_state" if donor in by_slug else "external_alias",
            }
            for donor in _donors(target)
        ],
        "ordered_prerequisites": [
            {
                "state": state.slug,
                "ion": state.ion,
                "config": state.config,
                "seed_class": state.seed_class,
                "donors": list(_donors(state)),
                "vary_first": state.vary_first,
            }
            for state in prerequisites
        ],
        "unresolved_donor_aliases": [
            {
                "element": record.element,
                "consumer_state": record.consumer_state,
                "alias": record.alias,
                "status": record.status,
                "reason": donor_alias_manifest.unresolved_reason,
            }
            for record in unresolved
        ],
        "donor_alias_manifest": donor_alias_manifest.to_dict(),
    }


def _atsp_dict(
    atsp: ATSPElementRecipes,
    state: FBlockState,
    component: FBlockComponent,
) -> dict[str, object]:
    recipe = atsp.state(state.slug)
    donor_pin = dict(atsp.donor_pins).get(state.slug)
    return {
        "component_status": component.status,
        "reviewed_by": component.reviewed_by,
        "reviewed_at": component.reviewed_at,
        "required_for_grasp_seed": state.seed_class == "atsp_hf",
        "implementation_scope": (
            "Modified ATSP2K with the campaign libecp port; stock ECP support "
            "is not established."
        ),
        "ecp_card_included": False,
        "element": {
            "symbol": atsp.symbol,
            "atomic_number": atsp.atomic_number,
            "effective_charge": atsp.effective_charge,
            "core_electrons": atsp.core_electrons,
            "l_max": atsp.l_max,
            "module": atsp.module,
            "campaign_seed_state": atsp.seed_state,
        },
        "state": {
            "slug": recipe.slug,
            "closed_shells": recipe.closed_shells,
            "open_configuration": recipe.open_configuration,
            "pinned_donor": donor_pin,
        },
        "stdin_lines": recipe.stdin_lines(atsp.symbol, atsp.atomic_number),
    }


def _missing_grasp_fields(state: FBlockState) -> tuple[str, ...]:
    missing = tuple(
        field
        for field in ("confline", "active_set", "jrange")
        if not getattr(state, field)
    )
    if state.core_menu is None:
        missing += ("core_menu",)
    return missing


def _grasp_dict(
    element: FBlockElement,
    state: FBlockState,
    missing_fields: tuple[str, ...],
) -> dict[str, object]:
    expected = {
        "blocks": [
            {"j": j, "ncsf": ncsf}
            for j, ncsf in zip(state.j_blocks, state.ncsf)
        ],
        "energies_au": {
            "dirac_coulomb": state.energy_dc_au,
            "dirac_coulomb_breit": state.energy_dcb_au,
            "relative_to_anchor": state.energy_relative_au,
        },
    }
    if missing_fields:
        return {
            "availability": "incomplete_reference_input",
            "missing_fields": list(missing_fields),
            "workflow_scope": (
                "Energy and J/CSF reference only; GRASP interactive inputs "
                "cannot be generated from this row."
            ),
            "serial_rmcdhf_required": True,
            "inputs": None,
            "artifact_transfers": [],
            "expected": expected,
            "checks": [
                "Do not infer missing GRASP prompt fields from the ATSP2K recipe.",
            ],
        }
    try:
        twoj_min, twoj_max = (int(part) for part in state.jrange.split(","))
    except (TypeError, ValueError) as error:
        raise FBlockCatalogLoadError(
            f"invalid 2J range for {element.symbol}.{state.slug}: {state.jrange!r}"
        ) from error
    if state.core_menu is None:
        raise FBlockCatalogLoadError(
            f"missing GRASP core menu for {element.symbol}.{state.slug}"
        )
    selections = ["1" if count == 1 else f"1-{count}" for count in state.ncsf]
    rmcdhf_tail = ["5"] if sum(state.ncsf) > 1 else []
    rmcdhf_tail.extend(["*", "*", "100"])
    rmcdhf_input = ["y", *selections, *rmcdhf_tail]
    stage_input = None
    if state.vary_first is not None:
        stage_tail = ["5"] if sum(state.ncsf) > 1 else []
        stage_tail.extend([state.vary_first, "*", "100"])
        stage_input = ["y", *selections, *stage_tail]
    seeded = state.seed_class != "cold"
    inputs = {
        "rnucleus": [
            str(element.atomic_number),
            str(element.mass_number),
            "n",
            "0",
            "0.5",
            "1",
            "1",
        ],
        "rcsfgenerate": [
            "*",
            str(state.core_menu),
            state.confline,
            "",
            state.active_set,
            f"{twoj_min},{twoj_max}",
            "0",
            "n",
        ],
        "rangular": ["y"],
        "rwfnestimate": (
            ["y", "1", "prev.w", "*", "3", "*"]
            if seeded
            else ["y", "2", "*"]
        ),
        "rmcdhf": rmcdhf_input,
        "rci": [
            "y",
            "ref",
            "y",
            "y",
            "1.d-6",
            "n",
            "n",
            "n",
            "n",
            *selections,
        ],
    }
    if stage_input is not None:
        inputs["rmcdhf_stage"] = stage_input
    return {
        "availability": "input_ready",
        "missing_fields": [],
        "workflow_scope": "single-configuration DC+Breit configuration average",
        "serial_rmcdhf_required": True,
        "inputs": inputs,
        "artifact_transfers": [
            {"source": "rcsf.out", "destination": "rcsf.inp"},
            {"source": "rcsf.inp", "destination": "ref.c"},
            {"source": "rwfn.out", "destination": "ref.w"},
        ],
        "expected": expected,
        "checks": [
            "Require the rcsfgenerate block table to match every expected J and CSF count.",
            "Require positive RMCDHF convergence evidence; process exit 0 is insufficient.",
            "Select every ASF in every block and use (2J+1) weights for the configuration average.",
            "Run RCI with low-frequency transverse photon enabled and QED and mass shifts disabled.",
            "Use the serial RMCDHF build for these small diffuse outer-shell references.",
        ],
    }


def _automation_requirements(
    state: FBlockState,
    prerequisites: tuple[FBlockState, ...],
    unresolved: tuple[FBlockDonorAliasRecord, ...],
    missing_grasp_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    requirements: list[dict[str, object]] = []
    if missing_grasp_fields:
        requirements.append({
            "kind": "missing_grasp_reference_input",
            "fields": list(missing_grasp_fields),
        })
    if state.seed_class == "atsp_hf":
        requirements.append({
            "kind": "atsp2k_seed_conversion",
            "detail": (
                "Run the recorded 13-line input with the modified ATSP2K "
                "ECP build, provide ecp.inp, then convert wfn.out with "
                "rwfnmchfmcdf into prev.w."
            ),
        })
    elif state.seed_class == "donor":
        requirements.append({
            "kind": "converged_donor_orbitals",
            "donors": list(_donors(state)),
        })
    elif state.seed_class == "multi_donor":
        requirements.append({
            "kind": "multi_donor_orbital_merge",
            "donors": list(_donors(state)),
            "detail": (
                "Merge donor radial-wavefunction records by orbital identity; "
                "the first donor wins duplicate orbitals."
            ),
        })
    if state.vary_first is not None:
        requirements.append({
            "kind": "staged_orbital_birth",
            "orbitals": state.vary_first,
            "detail": (
                "Run the staged RMCDHF input first, then warm-start the full "
                "RMCDHF input from that pass."
            ),
        })
    prerequisite_preparation = [
        {
            "state": prerequisite.slug,
            "seed_class": prerequisite.seed_class,
            "donors": list(_donors(prerequisite)),
            "vary_first": prerequisite.vary_first,
        }
        for prerequisite in prerequisites
        if prerequisite.seed_class != "cold"
        or prerequisite.vary_first is not None
    ]
    if prerequisite_preparation:
        requirements.append({
            "kind": "prerequisite_seed_preparation",
            "states": prerequisite_preparation,
        })
    if unresolved:
        requirements.append({
            "kind": "external_donor_mapping",
            "aliases": [record.alias for record in unresolved],
        })
    return requirements


__all__ = [
    "ATSPElementRecipes",
    "ATSPStateRecipe",
    "FBLOCK_PLAN_SCHEMA",
    "MAX_ATSP_RECIPE_BYTES",
    "FBlockAtomicPlan",
    "load_atsp_element_recipes",
    "plan_fblock_atomic_state",
]

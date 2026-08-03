"""Immutable, validated access to the bundled f-block atomic dataset.

The loader preserves the scientific payload verbatim and combines it with the
adjacent version, provenance, review, and redistribution metadata.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from chemtools.reference.fblock_models import (
    FBLOCK_DATASET_SCHEMA,
    FBlockCatalog,
    FBlockCatalogLoadError,
    FBlockComponent,
    FBlockDatasetMetadata,
    FBlockElement,
    FBlockProgram,
    FBlockRedistribution,
    FBlockState,
)

MAX_METADATA_BYTES = 128 * 1024
MAX_CATALOG_BYTES = 2 * 1024 * 1024
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_STATUSES = frozenset({
    "validated_reference",
    "regression_failure",
    "exploratory",
    "shelved",
})
_PURPOSES = frozenset({
    "parser_contract",
    "differential_contract",
    "scientific_regression",
    "workflow_recipe",
    "failure_diagnosis",
    "methodology_warning",
})
_STATE_FIELDS = frozenset({
    "slug",
    "ion",
    "config",
    "core",
    "confline",
    "role",
    "note",
    "hf_seed",
    "estimate_from",
    "vary_first",
    "core_menu",
    "active_set",
    "jrange",
    "J_blocks",
    "ncsf",
    "E_DC_au",
    "E_DCB_au",
    "E_rel_au",
    "seeding",
})
_ELEMENT_FIELDS = frozenset({
    "Z",
    "A",
    "core_label",
    "hamiltonian",
    "nucleus",
    "comment",
    "states",
    "atsp_hf_seed_default",
})


def bundled_fblock_directory() -> Path:
    resource = files("chemtools").joinpath("data/fblock")
    return Path(str(resource))


def load_fblock_catalog(
    directory: str | Path | None = None,
) -> FBlockCatalog:
    dataset_directory = (
        Path(directory) if directory is not None else bundled_fblock_directory()
    )
    metadata_path = dataset_directory / "metadata.json"
    try:
        metadata_payload = _load_json(metadata_path, MAX_METADATA_BYTES)
        metadata = _metadata(metadata_payload)
        _validate_component_paths(dataset_directory, metadata.components)
        catalog_path = _contained_path(
            dataset_directory,
            metadata.catalog_relative_path,
        )
        catalog_bytes = catalog_path.read_bytes()
        if len(catalog_bytes) > MAX_CATALOG_BYTES:
            raise ValueError(
                f"catalog exceeds {MAX_CATALOG_BYTES} bytes"
            )
        if len(catalog_bytes) != metadata.catalog_size_bytes:
            raise ValueError(
                "catalog byte count changed: expected "
                f"{metadata.catalog_size_bytes}, found {len(catalog_bytes)}"
            )
        actual_sha256 = hashlib.sha256(catalog_bytes).hexdigest()
        if actual_sha256 != metadata.catalog_sha256:
            raise ValueError(
                "catalog SHA-256 changed: expected "
                f"{metadata.catalog_sha256}, found {actual_sha256}"
            )
        catalog_payload = json.loads(
            catalog_bytes.decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
        elements = _elements(catalog_payload)
        _validate_coverage(metadata, elements)
        return FBlockCatalog(metadata=metadata, elements=elements)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        if isinstance(error, FBlockCatalogLoadError):
            raise
        raise FBlockCatalogLoadError(
            f"invalid f-block dataset at {dataset_directory}: {error}"
        ) from error


def _load_json(path: Path, byte_limit: int) -> Any:
    if not path.is_file():
        raise ValueError(f"required dataset file is missing: {path.name}")
    size = path.stat().st_size
    if size > byte_limit:
        raise ValueError(f"{path.name} exceeds {byte_limit} bytes")
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _metadata(value: Any) -> FBlockDatasetMetadata:
    document = _object(value, "metadata")
    required = {
        "schema_version",
        "dataset_id",
        "dataset_version",
        "rebuild_date",
        "description",
        "catalog",
        "coverage",
        "programs",
        "method_scope",
        "components",
        "validation",
        "redistribution",
        "known_limitations",
    }
    _exact_fields(document, required, "metadata")
    if document["schema_version"] != FBLOCK_DATASET_SCHEMA:
        raise ValueError(
            f"unsupported dataset schema {document['schema_version']!r}"
        )
    rebuild_date = _text(document["rebuild_date"], "rebuild_date")
    if not _DATE_RE.fullmatch(rebuild_date):
        raise ValueError("rebuild_date must use YYYY-MM-DD")

    catalog = _object(document["catalog"], "catalog")
    _exact_fields(
        catalog,
        {"relative_path", "payload_schema", "size_bytes", "sha256"},
        "catalog",
    )
    relative_path = _relative_path(catalog["relative_path"], "catalog.relative_path")
    sha256 = _text(catalog["sha256"], "catalog.sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("catalog.sha256 must be 64 lowercase hex characters")

    coverage = _object(document["coverage"], "coverage")
    _exact_fields(
        coverage,
        {
            "elements",
            "element_count",
            "state_count",
            "role_counts",
            "seed_class_counts",
            "staged_birth_state_count",
        },
        "coverage",
    )
    element_symbols = _text_tuple(coverage["elements"], "coverage.elements")
    if len(element_symbols) != len(set(element_symbols)):
        raise ValueError("coverage.elements must not contain duplicates")
    role_counts = _count_pairs(
        coverage["role_counts"],
        "coverage.role_counts",
        {"fit", "holdout"},
    )
    seed_counts = _count_pairs(
        coverage["seed_class_counts"],
        "coverage.seed_class_counts",
        {"donor", "multi_donor", "atsp_hf", "cold"},
    )

    programs = tuple(
        _program(item, f"programs[{index}]")
        for index, item in enumerate(_list(document["programs"], "programs"))
    )
    if not programs:
        raise ValueError("programs must not be empty")
    method_scope = _string_pairs(document["method_scope"], "method_scope")
    components = tuple(
        _component(item, f"components[{index}]")
        for index, item in enumerate(
            _list(document["components"], "components")
        )
    )
    if not components:
        raise ValueError("components must not be empty")
    component_ids = [component.id for component in components]
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("component IDs must not contain duplicates")
    _validation(document["validation"])

    return FBlockDatasetMetadata(
        dataset_id=_text(document["dataset_id"], "dataset_id"),
        dataset_version=_text(document["dataset_version"], "dataset_version"),
        rebuild_date=rebuild_date,
        description=_text(document["description"], "description"),
        catalog_relative_path=relative_path,
        catalog_payload_schema=_text(
            catalog["payload_schema"],
            "catalog.payload_schema",
        ),
        catalog_size_bytes=_positive_int(
            catalog["size_bytes"],
            "catalog.size_bytes",
        ),
        catalog_sha256=sha256,
        element_symbols=element_symbols,
        element_count=_positive_int(
            coverage["element_count"],
            "coverage.element_count",
        ),
        state_count=_positive_int(
            coverage["state_count"],
            "coverage.state_count",
        ),
        role_counts=role_counts,
        seed_class_counts=seed_counts,
        staged_birth_state_count=_nonnegative_int(
            coverage["staged_birth_state_count"],
            "coverage.staged_birth_state_count",
        ),
        programs=programs,
        method_scope=method_scope,
        components=components,
        redistribution=_redistribution(document["redistribution"]),
        known_limitations=_text_tuple(
            document["known_limitations"],
            "known_limitations",
        ),
    )


def _program(value: Any, field: str) -> FBlockProgram:
    document = _object(value, field)
    _exact_fields(document, {"name", "version", "role"}, field)
    return FBlockProgram(
        name=_text(document["name"], f"{field}.name"),
        version=_text(document["version"], f"{field}.version"),
        role=_text(document["role"], f"{field}.role"),
    )


def _component(value: Any, field: str) -> FBlockComponent:
    document = _object(value, field)
    _exact_fields(
        document,
        {"id", "paths", "status", "purposes", "review"},
        field,
    )
    status = _text(document["status"], f"{field}.status")
    if status not in _STATUSES:
        raise ValueError(f"{field}.status has unsupported value {status!r}")
    purposes = _text_tuple(document["purposes"], f"{field}.purposes")
    unknown_purposes = sorted(set(purposes) - _PURPOSES)
    if unknown_purposes:
        raise ValueError(f"{field}.purposes has unsupported values {unknown_purposes}")
    paths = tuple(
        _relative_path(path, f"{field}.paths")
        for path in _text_tuple(document["paths"], f"{field}.paths")
    )
    review = _object(document["review"], f"{field}.review")
    _exact_fields(
        review,
        {"reviewed_by", "reviewed_at", "scope"},
        f"{field}.review",
    )
    reviewed_at = _text(review["reviewed_at"], f"{field}.review.reviewed_at")
    if not _DATE_RE.fullmatch(reviewed_at):
        raise ValueError(f"{field}.review.reviewed_at must use YYYY-MM-DD")
    return FBlockComponent(
        id=_text(document["id"], f"{field}.id"),
        paths=paths,
        status=status,
        purposes=purposes,
        reviewed_by=_text(
            review["reviewed_by"],
            f"{field}.review.reviewed_by",
        ),
        reviewed_at=reviewed_at,
        review_scope=_text(review["scope"], f"{field}.review.scope"),
    )


def _redistribution(value: Any) -> FBlockRedistribution:
    document = _object(value, "redistribution")
    required = {
        "status",
        "source",
        "attribution",
        "license_identifier",
        "license_evidence",
        "permission_evidence",
        "third_party_components",
    }
    _exact_fields(document, required, "redistribution")
    status = _text(document["status"], "redistribution.status")
    if status not in {"allowed", "restricted", "review_required"}:
        raise ValueError(f"unsupported redistribution status {status!r}")
    return FBlockRedistribution(
        status=status,
        source=_text(document["source"], "redistribution.source"),
        attribution=_text(
            document["attribution"],
            "redistribution.attribution",
        ),
        license_identifier=_text(
            document["license_identifier"],
            "redistribution.license_identifier",
        ),
        license_evidence=_text(
            document["license_evidence"],
            "redistribution.license_evidence",
        ),
        permission_evidence=_text(
            document["permission_evidence"],
            "redistribution.permission_evidence",
        ),
        third_party_components=_text_tuple(
            document["third_party_components"],
            "redistribution.third_party_components",
            allow_empty=True,
        ),
    )


def _validation(value: Any) -> None:
    document = _object(value, "validation")
    required = {
        "catalog_sha256_pinned",
        "unique_state_slugs_per_element",
        "j_block_and_ncsf_lengths_match",
        "energies_are_finite",
        "seed_class_totals_pinned",
    }
    _exact_fields(document, required, "validation")
    if any(document[key] is not True for key in required):
        raise ValueError("all declared dataset validation gates must be true")


def _elements(value: Any) -> tuple[FBlockElement, ...]:
    document = _object(value, "catalog")
    elements = tuple(
        _element(symbol, payload)
        for symbol, payload in document.items()
    )
    atomic_numbers = [element.atomic_number for element in elements]
    if len(atomic_numbers) != len(set(atomic_numbers)):
        raise ValueError("catalog atomic numbers must be unique")
    return elements


def _element(symbol: str, value: Any) -> FBlockElement:
    normalized_symbol = _element_symbol(symbol)
    document = _object(value, f"element {symbol}")
    _exact_fields(document, _ELEMENT_FIELDS, f"element {symbol}")
    states = tuple(
        _state(item, f"{symbol}.states[{index}]")
        for index, item in enumerate(_list(document["states"], f"{symbol}.states"))
    )
    if not states:
        raise ValueError(f"element {symbol} must contain states")
    slugs = [state.slug for state in states]
    if len(slugs) != len(set(slugs)):
        raise ValueError(f"element {symbol} contains duplicate state slugs")
    return FBlockElement(
        symbol=normalized_symbol,
        atomic_number=_positive_int(document["Z"], f"{symbol}.Z"),
        mass_number=_positive_int(document["A"], f"{symbol}.A"),
        core_label=_text(document["core_label"], f"{symbol}.core_label"),
        hamiltonian=_text(document["hamiltonian"], f"{symbol}.hamiltonian"),
        nucleus=_text(document["nucleus"], f"{symbol}.nucleus"),
        comment=_text(document["comment"], f"{symbol}.comment"),
        atsp_hf_seed_default=_boolean(
            document["atsp_hf_seed_default"],
            f"{symbol}.atsp_hf_seed_default",
        ),
        states=states,
    )


def _state(value: Any, field: str) -> FBlockState:
    document = _object(value, field)
    _exact_fields(document, _STATE_FIELDS, field)
    role = _text(document["role"], f"{field}.role")
    if role not in {"fit", "holdout"}:
        raise ValueError(f"{field}.role must be 'fit' or 'holdout'")
    j_blocks = _text_tuple(document["J_blocks"], f"{field}.J_blocks")
    ncsf = tuple(
        _positive_int(item, f"{field}.ncsf[{index}]")
        for index, item in enumerate(_list(document["ncsf"], f"{field}.ncsf"))
    )
    if len(j_blocks) != len(ncsf):
        raise ValueError(f"{field} J_blocks and ncsf lengths differ")
    estimate_from = document["estimate_from"]
    if isinstance(estimate_from, list):
        estimate_from = _text_tuple(estimate_from, f"{field}.estimate_from")
    else:
        estimate_from = _optional_text(
            estimate_from,
            f"{field}.estimate_from",
        )
    state = FBlockState(
        slug=_text(document["slug"], f"{field}.slug"),
        ion=_nonnegative_int(document["ion"], f"{field}.ion"),
        config=_text(document["config"], f"{field}.config"),
        core=_optional_text(document["core"], f"{field}.core"),
        confline=_string(document["confline"], f"{field}.confline"),
        role=role,
        note=_optional_text(document["note"], f"{field}.note"),
        hf_seed=_optional_boolean(document["hf_seed"], f"{field}.hf_seed"),
        estimate_from=estimate_from,
        vary_first=_optional_text(
            document["vary_first"],
            f"{field}.vary_first",
        ),
        core_menu=_optional_int(
            document["core_menu"],
            f"{field}.core_menu",
        ),
        active_set=_string(document["active_set"], f"{field}.active_set"),
        jrange=_string(document["jrange"], f"{field}.jrange"),
        j_blocks=j_blocks,
        ncsf=ncsf,
        energy_dc_au=_finite_number(document["E_DC_au"], f"{field}.E_DC_au"),
        energy_dcb_au=_finite_number(
            document["E_DCB_au"],
            f"{field}.E_DCB_au",
        ),
        energy_relative_au=_finite_number(
            document["E_rel_au"],
            f"{field}.E_rel_au",
        ),
        seeding=_text(document["seeding"], f"{field}.seeding"),
    )
    state.seed_class
    return state


def _validate_coverage(
    metadata: FBlockDatasetMetadata,
    elements: tuple[FBlockElement, ...],
) -> None:
    symbols = tuple(element.symbol for element in elements)
    if symbols != metadata.element_symbols:
        raise ValueError(
            f"element coverage changed: expected {metadata.element_symbols}, "
            f"found {symbols}"
        )
    if len(elements) != metadata.element_count:
        raise ValueError("element_count does not match the catalog")
    states = tuple(state for element in elements for state in element.states)
    if len(states) != metadata.state_count:
        raise ValueError("state_count does not match the catalog")
    role_counts = {
        role: sum(state.role == role for state in states)
        for role, _count in metadata.role_counts
    }
    if role_counts != dict(metadata.role_counts):
        raise ValueError(f"role counts changed: {role_counts}")
    seed_counts = {
        seed_class: sum(state.seed_class == seed_class for state in states)
        for seed_class, _count in metadata.seed_class_counts
    }
    if seed_counts != dict(metadata.seed_class_counts):
        raise ValueError(f"seed-class counts changed: {seed_counts}")
    staged_birth_count = sum(state.vary_first is not None for state in states)
    if staged_birth_count != metadata.staged_birth_state_count:
        raise ValueError(
            f"staged-birth count changed: {staged_birth_count}"
        )


def _contained_path(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    resolved = (resolved_root / relative_path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("catalog path escapes the dataset directory") from error
    if not resolved.is_file():
        raise ValueError(f"catalog file is missing: {relative_path}")
    return resolved


def _validate_component_paths(
    root: Path,
    components: tuple[FBlockComponent, ...],
) -> None:
    resolved_root = root.resolve()
    for component in components:
        for pattern in component.paths:
            matches = tuple(path for path in root.glob(pattern) if path.is_file())
            if not matches:
                raise ValueError(
                    f"component {component.id!r} path matches no files: {pattern}"
                )
            for match in matches:
                try:
                    match.resolve().relative_to(resolved_root)
                except ValueError as error:
                    raise ValueError(
                        f"component {component.id!r} path escapes the dataset"
                    ) from error


def _relative_path(value: Any, field: str) -> str:
    path_text = _text(value, field)
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must stay inside the dataset directory")
    return path_text


def _element_symbol(value: Any) -> str:
    text = _text(value, "element symbol")
    normalized = text[0].upper() + text[1:].lower()
    if not re.fullmatch(r"[A-Z][a-z]?", normalized):
        raise ValueError(f"invalid element symbol {value!r}")
    return normalized


def _exact_fields(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    field: str,
) -> None:
    missing = sorted(set(expected) - set(value))
    unknown = sorted(set(value) - set(expected))
    if missing:
        raise ValueError(f"{field} is missing fields {missing}")
    if unknown:
        raise ValueError(f"{field} has unknown fields {unknown}")


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise TypeError(f"{field} must be a string")
    return value


def _text(value: Any, field: str) -> str:
    text = _string(value, field)
    if not text.strip():
        raise ValueError(f"{field} must not be empty")
    return text


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _text_tuple(
    value: Any,
    field: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    items = tuple(
        _text(item, f"{field}[{index}]")
        for index, item in enumerate(_list(value, field))
    )
    if not items and not allow_empty:
        raise ValueError(f"{field} must not be empty")
    return items


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean")
    return value


def _optional_boolean(value: Any, field: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, field)


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return value


def _positive_int(value: Any, field: str) -> int:
    number = _nonnegative_int(value, field)
    if number == 0:
        raise ValueError(f"{field} must be positive")
    return number


def _optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, field)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _count_pairs(
    value: Any,
    field: str,
    expected_keys: set[str],
) -> tuple[tuple[str, int], ...]:
    document = _object(value, field)
    _exact_fields(document, expected_keys, field)
    return tuple(
        (key, _nonnegative_int(document[key], f"{field}.{key}"))
        for key in sorted(document)
    )


def _string_pairs(value: Any, field: str) -> tuple[tuple[str, str], ...]:
    document = _object(value, field)
    if not document:
        raise ValueError(f"{field} must not be empty")
    return tuple(
        (_text(key, f"{field} key"), _text(item, f"{field}.{key}"))
        for key, item in sorted(document.items())
    )


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
    "bundled_fblock_directory",
    "load_fblock_catalog",
]

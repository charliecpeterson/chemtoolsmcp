"""Resolve referenced QMCPACK HDF5 sidecars without decoding their datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from chemtools.core.hdf5 import hdf5_signature_offset
from chemtools.core.types import LintIssue


_PWSCF_H5_REFERENCE_KINDS = frozenset({"determinantset", "sposet_collection"})


def inspect_hdf5_sidecars(
    input_path: str | Path,
    parsed_input: Mapping[str, Any],
    include_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(input_path).expanduser().resolve()
    references = _hdf5_references(parsed_input, include_review, source)
    if not references:
        return {
            "status": "not_applicable",
            "resolution": {
                "base_path": str(source.parent),
                "basis": "input_directory_assumption",
            },
            "entries": [],
        }

    grouped: dict[tuple[str, str], list[str]] = {}
    for reference in references:
        grouped.setdefault(
            (reference["source_path"], reference["href"]),
            [],
        ).append(reference["kind"])
    entries = [
        _inspect_sidecar(source, Path(source_path), href, kinds)
        for (source_path, href), kinds in grouped.items()
    ]
    return {
        "status": (
            "reviewed"
            if all(entry["status"] == "present" for entry in entries)
            else "incomplete"
        ),
        "resolution": {
            "base_path": str(source.parent),
            "basis": "input_directory_assumption",
        },
        "entries": entries,
    }


def find_referenced_hdf5(
    review: Mapping[str, Any],
    path: str | Path,
) -> list[dict[str, Any]]:
    resolved = str(Path(path).expanduser().resolve())
    return [
        dict(entry)
        for entry in review.get("entries", [])
        if isinstance(entry, Mapping) and entry.get("path") == resolved
    ]


def inspect_pwscf_h5_reference(
    review: Mapping[str, Any],
    include_review: Mapping[str, Any],
    pwscf_h5: str | Path,
) -> dict[str, Any]:
    declared_path = str(Path(pwscf_h5).expanduser().resolve())
    entries = review["entries"]
    matching_entries = find_referenced_hdf5(review, pwscf_h5)
    orbital_references = [
        entry
        for entry in matching_entries
        if _PWSCF_H5_REFERENCE_KINDS.intersection(entry["reference_kinds"])
    ]
    valid_orbital_references = [
        entry for entry in orbital_references if entry["status"] == "present"
    ]
    if valid_orbital_references:
        return {
            "name": "qmcpack_pwscf_h5_reference",
            "status": "pass",
            "observed": {
                "declared_path": declared_path,
                "matching_references": valid_orbital_references,
            },
            "message": (
                "The QMCPACK XML deck resolves an HDF5 reference to the declared "
                "QE conversion artifact."
            ),
        }
    if orbital_references:
        return {
            "name": "qmcpack_pwscf_h5_reference",
            "status": "not_ready",
            "observed": {
                "declared_path": declared_path,
                "matching_references": orbital_references,
            },
            "message": (
                "The QMCPACK XML deck names the declared QE conversion artifact, "
                "but that HDF5 sidecar is missing, unreadable, or invalid."
            ),
        }
    if include_review["status"] == "incomplete":
        return {
            "name": "qmcpack_pwscf_h5_reference",
            "status": "review_required",
            "observed": {
                "declared_path": declared_path,
                "include_review_status": include_review["status"],
                "resolved_hdf5_references": entries,
                "non_orbital_matching_references": matching_entries,
            },
            "message": (
                "The QMCPACK include graph is incomplete, so the declared "
                ".pwscf.h5 reference cannot be established."
            ),
        }
    return {
        "name": "qmcpack_pwscf_h5_reference",
        "status": "not_ready",
        "observed": {
            "declared_path": declared_path,
            "resolved_hdf5_references": entries,
            "non_orbital_matching_references": matching_entries,
        },
        "message": (
            "No QMCPACK orbital reference resolves to the declared QE conversion "
            "artifact."
        ),
    }


def hdf5_sidecar_issues(review: Mapping[str, Any]) -> list[LintIssue]:
    issues: list[LintIssue] = []
    for entry_value in review.get("entries", []):
        if not isinstance(entry_value, Mapping):
            continue
        href = str(entry_value.get("href") or "")
        kinds = entry_value.get("reference_kinds") or []
        authoritative = "override_variational_parameters" in kinds
        if entry_value.get("status") == "missing":
            if authoritative:
                issues.append(_issue(
                    "error",
                    (
                        f"Authoritative variational-parameter sidecar {href!r} "
                        "is missing."
                    ),
                    suggested_fix=(
                        "Provide the referenced vp.h5 file or remove the "
                        "override and re-optimize the Jastrow."
                    ),
                ))
            else:
                issues.append(_issue(
                    "warning",
                    f"Referenced HDF5 sidecar {href!r} was not found relative to the reviewed input.",
                    suggested_fix=(
                        "Provide the sidecar in the input directory or correct "
                        "the XML href."
                    ),
                ))
        elif entry_value.get("status") == "not_file":
            issues.append(_issue(
                "error",
                f"Referenced HDF5 sidecar {href!r} resolves to a non-file path.",
            ))
        elif entry_value.get("status") == "unreadable":
            issues.append(_issue(
                "warning",
                f"Referenced HDF5 sidecar {href!r} could not be inspected.",
            ))
        elif entry_value.get("status") == "invalid":
            if authoritative:
                issues.append(_issue(
                    "error",
                    (
                        f"Authoritative variational-parameter sidecar {href!r} "
                        "does not contain an HDF5 signature."
                    ),
                    suggested_fix=(
                        "Provide the HDF5 variational-parameter sidecar written "
                        "by QMCPACK or remove the override and re-optimize the Jastrow."
                    ),
                ))
            else:
                issues.append(_issue(
                    "warning",
                    (
                        f"Referenced HDF5 sidecar {href!r} does not contain an "
                        "HDF5 signature."
                    ),
                ))
        elif entry_value.get("freshness") == "older_than_input":
            issues.append(_issue(
                "warning",
                (
                    f"HDF5 sidecar {href!r} is older than the XML input; "
                    "confirm that the reference still matches this input."
                ),
            ))
    return issues


def _hdf5_references(
    parsed_input: Mapping[str, Any],
    include_review: Mapping[str, Any] | None,
    source: Path,
) -> list[dict[str, str]]:
    references = []
    values = list(parsed_input.get("references", []))
    if include_review is not None:
        values.extend(include_review.get("discovered_references", []))
    for value in values:
        if not isinstance(value, Mapping):
            continue
        href = value.get("href")
        kind = value.get("kind")
        if (
            isinstance(href, str)
            and href.lower().endswith(".h5")
            and isinstance(kind, str)
        ):
            reference = {"href": href, "kind": kind}
            source_path = value.get("source_path")
            reference["source_path"] = (
                source_path if isinstance(source_path, str) else str(source)
            )
            references.append(reference)
    return references


def _inspect_sidecar(
    primary_source: Path,
    source: Path,
    href: str,
    kinds: list[str],
) -> dict[str, Any]:
    configured = Path(href).expanduser()
    path = configured.resolve() if configured.is_absolute() else (
        source.parent / configured
    ).resolve()
    common = {
        "href": href,
        "path": str(path),
        "reference_kinds": sorted(set(kinds)),
        **({"source_path": str(source)} if source != primary_source else {}),
    }
    try:
        metadata = path.stat()
    except FileNotFoundError:
        return {**common, "status": "missing", "freshness": "missing"}
    except OSError as error:
        return {
            **common,
            "status": "unreadable",
            "freshness": "not_assessed",
            "reason": f"{type(error).__name__}: {error}",
        }
    if not path.is_file():
        return {**common, "status": "not_file", "freshness": "not_assessed"}

    input_modified_ns = source.stat().st_mtime_ns
    freshness = (
        "older_than_input"
        if metadata.st_mtime_ns < input_modified_ns
        else "not_older_than_input"
    )
    try:
        signature_offset = hdf5_signature_offset(path, metadata.st_size)
    except OSError as error:
        return {
            **common,
            "status": "unreadable",
            "freshness": "not_assessed",
            "reason": f"{type(error).__name__}: {error}",
        }
    if signature_offset is None:
        return {
            **common,
            "status": "invalid",
            "freshness": freshness,
            "size_bytes": metadata.st_size,
            "modified_ns": metadata.st_mtime_ns,
            "hdf5_signature_offset": None,
        }
    return {
        **common,
        "status": "present",
        "freshness": freshness,
        "size_bytes": metadata.st_size,
        "modified_ns": metadata.st_mtime_ns,
        "hdf5_signature_offset": signature_offset,
    }


def _issue(
    level: str,
    message: str,
    *,
    suggested_fix: str | None = None,
) -> LintIssue:
    return {
        "level": level,
        "message": message,
        "line": None,
        "suggested_fix": suggested_fix,
    }


__all__ = [
    "find_referenced_hdf5",
    "hdf5_sidecar_issues",
    "inspect_hdf5_sidecars",
    "inspect_pwscf_h5_reference",
]

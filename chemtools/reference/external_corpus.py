"""Validate named artifacts in a read-only external reference corpus.

The verifier establishes artifact identity and containment only. Scientific
expectations remain the responsibility of the application-level consumer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


REFERENCE_CORPUS_SCHEMA = "chemtools.reference-corpus/1"
DEFAULT_CASE_BYTE_LIMIT = 32 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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


class ReferenceManifestError(ValueError):
    pass


def load_reference_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
        _validate_manifest(payload)
    except (OSError, TypeError, ValueError) as error:
        if isinstance(error, ReferenceManifestError):
            raise
        raise ReferenceManifestError(
            f"invalid reference manifest at {manifest_path}: {error}"
        ) from error
    return payload


def verify_reference_case(
    manifest: dict[str, Any],
    case_id: str,
    corpus_root: str | Path | None,
    *,
    byte_limit: int = DEFAULT_CASE_BYTE_LIMIT,
) -> dict[str, Any]:
    """Verify one case without scanning or reading unlisted corpus files."""
    _validate_manifest(manifest)
    case = next(
        (item for item in manifest["cases"] if item["id"] == case_id),
        None,
    )
    if case is None:
        raise ReferenceManifestError(f"unknown reference case {case_id!r}")
    if isinstance(byte_limit, bool) or not isinstance(byte_limit, int):
        raise ValueError("byte_limit must be a non-negative integer")
    if byte_limit < 0:
        raise ValueError("byte_limit must be a non-negative integer")

    required = [item for item in case["artifacts"] if item["required"]]
    total_bytes = sum(item["size_bytes"] for item in required)
    if total_bytes > byte_limit:
        return _no_reference(
            case,
            "case exceeds the configured byte budget",
            expected_size_bytes=total_bytes,
            byte_limit=byte_limit,
        )
    if corpus_root is None:
        return _no_reference(
            case,
            "set CHEMTOOLS_REFERENCE_CORPUS or pass a corpus root",
        )

    root = Path(corpus_root).expanduser().resolve()
    if not root.is_dir():
        return _no_reference(case, "reference corpus root is not a directory")

    verified = []
    for artifact in required:
        source = (root / artifact["path"]).resolve()
        try:
            source.relative_to(root)
        except ValueError:
            return _artifact_failure(
                case,
                artifact,
                source,
                "reference path escapes the configured corpus root",
            )
        if not source.is_file():
            return _artifact_failure(
                case,
                artifact,
                source,
                "reference artifact is missing or is not a file",
            )
        actual_size = source.stat().st_size
        if actual_size != artifact["size_bytes"]:
            return _artifact_failure(
                case,
                artifact,
                source,
                "reference size changed; review the case before use",
                expected_size_bytes=artifact["size_bytes"],
                actual_size_bytes=actual_size,
            )
        actual_sha256 = _sha256(source)
        if actual_sha256 != artifact["sha256"]:
            return _artifact_failure(
                case,
                artifact,
                source,
                "reference hash changed; review the case before use",
                expected_sha256=artifact["sha256"],
                actual_sha256=actual_sha256,
            )
        verified.append({
            "id": artifact["id"],
            "path": str(source),
            "roles": list(artifact["roles"]),
            "kind": artifact["kind"],
            "size_bytes": actual_size,
            "sha256": actual_sha256,
        })

    return {
        "case_id": case["id"],
        "status": case["status"],
        "purposes": list(case["purposes"]),
        "outcome": "verified",
        "total_size_bytes": total_bytes,
        "artifacts": verified,
    }


def _validate_manifest(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ReferenceManifestError("manifest must be an object")
    if payload.get("schema") != REFERENCE_CORPUS_SCHEMA:
        raise ReferenceManifestError(
            f"unsupported reference manifest schema {payload.get('schema')!r}"
        )
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ReferenceManifestError("manifest must contain at least one case")
    case_ids = []
    for index, case in enumerate(cases):
        field = f"cases[{index}]"
        if not isinstance(case, dict):
            raise ReferenceManifestError(f"{field} must be an object")
        for required_field in (
            "id",
            "programs",
            "status",
            "purposes",
            "artifacts",
            "expected",
            "review",
        ):
            if required_field not in case:
                raise ReferenceManifestError(
                    f"{field} is missing {required_field!r}"
                )
        case_id = _nonempty_text(case["id"], f"{field}.id")
        case_ids.append(case_id)
        programs = _nonempty_text_list(
            case["programs"],
            f"{field}.programs",
        )
        if len(programs) != len(set(programs)):
            raise ReferenceManifestError(
                f"{field}.programs must not contain duplicates"
            )
        status = _nonempty_text(case["status"], f"{field}.status")
        if status not in _STATUSES:
            raise ReferenceManifestError(
                f"{field}.status has unsupported value {status!r}"
            )
        purposes = _nonempty_text_list(
            case["purposes"],
            f"{field}.purposes",
        )
        unknown_purposes = sorted(set(purposes) - _PURPOSES)
        if unknown_purposes:
            raise ReferenceManifestError(
                f"{field}.purposes has unsupported values {unknown_purposes}"
            )
        if len(purposes) != len(set(purposes)):
            raise ReferenceManifestError(
                f"{field}.purposes must not contain duplicates"
            )
        if not isinstance(case["expected"], dict):
            raise ReferenceManifestError(f"{field}.expected must be an object")
        _validate_review(case["review"], status, f"{field}.review")
        if "tags" in case:
            tags = _nonempty_text_list(case["tags"], f"{field}.tags")
            if len(tags) != len(set(tags)):
                raise ReferenceManifestError(
                    f"{field}.tags must not contain duplicates"
                )
        artifacts = case["artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise ReferenceManifestError(
                f"{field}.artifacts must be a non-empty array"
            )
        artifact_ids = []
        for artifact_index, artifact in enumerate(artifacts):
            artifact_field = f"{field}.artifacts[{artifact_index}]"
            _validate_artifact(artifact, artifact_field)
            artifact_ids.append(artifact["id"])
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ReferenceManifestError(
                f"{field}.artifacts contains duplicate IDs"
            )
    if len(case_ids) != len(set(case_ids)):
        raise ReferenceManifestError("manifest contains duplicate case IDs")


def _validate_artifact(artifact: Any, field: str) -> None:
    if not isinstance(artifact, dict):
        raise ReferenceManifestError(f"{field} must be an object")
    required_fields = {
        "id",
        "roles",
        "kind",
        "path",
        "storage_tier",
        "size_bytes",
        "sha256",
        "required",
        "redistribution",
        "source",
        "attribution",
        "license",
        "permission_evidence",
    }
    missing = required_fields - artifact.keys()
    if missing:
        raise ReferenceManifestError(f"{field} is missing {sorted(missing)}")
    _nonempty_text(artifact["id"], f"{field}.id")
    _nonempty_text_list(artifact["roles"], f"{field}.roles")
    _nonempty_text(artifact["kind"], f"{field}.kind")
    path = PurePosixPath(_nonempty_text(artifact["path"], f"{field}.path"))
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ReferenceManifestError(
            f"{field}.path must be a contained relative path"
        )
    if artifact["storage_tier"] != "external_reference":
        raise ReferenceManifestError(
            f"{field}.storage_tier must be 'external_reference'"
        )
    size_bytes = artifact["size_bytes"]
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
    ):
        raise ReferenceManifestError(
            f"{field}.size_bytes must be a non-negative integer"
        )
    sha256 = artifact["sha256"]
    if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
        raise ReferenceManifestError(
            f"{field}.sha256 must be 64 lowercase hex characters"
        )
    if not isinstance(artifact["required"], bool):
        raise ReferenceManifestError(f"{field}.required must be boolean")
    if artifact["redistribution"] not in {
        "allowed",
        "restricted",
        "review_required",
    }:
        raise ReferenceManifestError(
            f"{field}.redistribution has an unsupported value"
        )
    _nonempty_text(artifact["source"], f"{field}.source")
    _nonempty_text(artifact["attribution"], f"{field}.attribution")
    license_record = artifact["license"]
    if not isinstance(license_record, dict):
        raise ReferenceManifestError(f"{field}.license must be an object")
    if set(license_record) != {"identifier", "terms"}:
        raise ReferenceManifestError(
            f"{field}.license must contain only 'identifier' and 'terms'"
        )
    for key in ("identifier", "terms"):
        value = license_record[key]
        if value is not None:
            _nonempty_text(value, f"{field}.license.{key}")
    permission_evidence = artifact["permission_evidence"]
    if permission_evidence is not None:
        _nonempty_text(permission_evidence, f"{field}.permission_evidence")
    if artifact["redistribution"] == "allowed":
        if not any(license_record.values()) or permission_evidence is None:
            raise ReferenceManifestError(
                f"{field} marked allowed requires license terms and "
                "permission evidence"
            )


def _validate_review(value: Any, status: str, field: str) -> None:
    if not isinstance(value, dict):
        raise ReferenceManifestError(f"{field} must be an object")
    required = {"reviewed_by", "reviewed_at", "scope"}
    if set(value) != required:
        raise ReferenceManifestError(
            f"{field} must contain only {sorted(required)}"
        )
    _nonempty_text(value["scope"], f"{field}.scope")
    reviewer = value["reviewed_by"]
    reviewed_at = value["reviewed_at"]
    if status == "validated_reference":
        _nonempty_text(reviewer, f"{field}.reviewed_by")
        date = _nonempty_text(reviewed_at, f"{field}.reviewed_at")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            raise ReferenceManifestError(
                f"{field}.reviewed_at must use YYYY-MM-DD"
            )
    else:
        for item, name in (
            (reviewer, "reviewed_by"),
            (reviewed_at, "reviewed_at"),
        ):
            if item is not None:
                _nonempty_text(item, f"{field}.{name}")


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReferenceManifestError(f"{field} must be non-empty text")
    return value


def _nonempty_text_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ReferenceManifestError(f"{field} must be a non-empty array")
    return [
        _nonempty_text(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    ]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ReferenceManifestError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _artifact_failure(
    case: dict[str, Any],
    artifact: dict[str, Any],
    source: Path,
    reason: str,
    **details: Any,
) -> dict[str, Any]:
    return _no_reference(
        case,
        reason,
        artifact_id=artifact["id"],
        source=str(source),
        **details,
    )


def _no_reference(
    case: dict[str, Any],
    reason: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "status": case["status"],
        "purposes": list(case["purposes"]),
        "outcome": "no_reference",
        "reason": reason,
        **details,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "DEFAULT_CASE_BYTE_LIMIT",
    "REFERENCE_CORPUS_SCHEMA",
    "ReferenceManifestError",
    "load_reference_manifest",
    "verify_reference_case",
]

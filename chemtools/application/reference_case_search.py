"""Search packaged reference-case metadata without scanning external corpora."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable

from chemtools.reference.external_corpus import load_reference_manifest


REFERENCE_CASE_SEARCH_SCHEMA = "chemtools.find-reference-case/1"
MAX_REFERENCE_QUERY_LENGTH = 256
MAX_REFERENCE_RESULTS = 10
REFERENCE_CASE_STATUSES = frozenset({
    "validated_reference",
    "regression_failure",
    "exploratory",
    "shelved",
})
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class ReferenceCaseSearchError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)

    def as_dict(self) -> dict[str, str]:
        return {"error": self.code, "message": str(self)}


def bundled_reference_manifest_paths() -> tuple[Path, ...]:
    directory = Path(__file__).resolve().parents[1] / "data" / "reference_cases"
    return tuple(sorted(directory.glob("*.json")))


def find_reference_cases(
    query: str,
    *,
    program: str | None = None,
    scientific_status: str | None = None,
    limit: int = 5,
    manifest_paths: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Return ranked case metadata; artifact pins are never opened here."""
    normalized_query = _normalize_query(query)
    normalized_program = _normalize_program(program)
    normalized_status = _normalize_status(scientific_status)
    normalized_limit = _normalize_limit(limit)
    paths = tuple(
        Path(path) for path in (
            manifest_paths
            if manifest_paths is not None
            else bundled_reference_manifest_paths()
        )
    )
    if not paths:
        raise ReferenceCaseSearchError(
            "reference_manifests_unavailable",
            "no packaged reference-case manifests were found",
        )

    query_tokens = frozenset(_TOKEN_RE.findall(normalized_query.casefold()))
    ranked = []
    seen_case_ids = set()
    for path in paths:
        manifest = load_reference_manifest(path)
        for case in manifest["cases"]:
            case_id = case["id"]
            if case_id in seen_case_ids:
                raise ReferenceCaseSearchError(
                    "duplicate_reference_case",
                    f"reference case {case_id!r} occurs in more than one manifest",
                )
            seen_case_ids.add(case_id)
            if normalized_program not in (None, *case["programs"]):
                continue
            if normalized_status != case["status"]:
                continue
            score = _match_score(query_tokens, case)
            if score == 0:
                continue
            ranked.append((score, case_id, _case_record(case)))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    matches = [record for _, _, record in ranked[:normalized_limit]]
    uncertainties = []
    if matches:
        uncertainties.append({
            "code": "artifact_availability_not_checked",
            "message": (
                "Artifact paths, sizes, and hashes are pinned, but this metadata "
                "search does not open the external corpus."
            ),
            "impact": (
                "Verify a selected case against CHEMTOOLS_REFERENCE_CORPUS "
                "before using its files."
            ),
        })
    else:
        uncertainties.append({
            "code": "no_matching_reference_case",
            "message": "No packaged reference case matched every filter.",
            "impact": (
                "Broaden the query or scientific-status filter; do not treat "
                "an exploratory case as a validated reference."
            ),
        })
    if any(
        match["scientific_status"] != "validated_reference"
        for match in matches
    ):
        uncertainties.append({
            "code": "scientific_review_incomplete",
            "message": (
                "One or more returned cases have not been approved as "
                "validated scientific references."
            ),
            "impact": (
                "Treat recorded expectations as review evidence, not accepted "
                "scientific truth."
            ),
        })

    return {
        "schema_version": REFERENCE_CASE_SEARCH_SCHEMA,
        "query": {
            "text": normalized_query,
            "program": normalized_program,
            "scientific_status": normalized_status,
            "limit": normalized_limit,
        },
        "match_count": len(matches),
        "matches": matches,
        "uncertainty": uncertainties,
        "next_actions": ([{
            "action": "verify_reference_case",
            "reason": (
                "Confirm the selected required artifacts still match their "
                "recorded sizes and SHA-256 hashes before comparison."
            ),
            "priority": 1,
        }] if matches else []),
    }


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ReferenceCaseSearchError(
            "invalid_reference_query",
            "query must be non-empty text",
        )
    normalized = " ".join(query.split())
    if len(normalized) > MAX_REFERENCE_QUERY_LENGTH:
        raise ReferenceCaseSearchError(
            "invalid_reference_query",
            f"query must be at most {MAX_REFERENCE_QUERY_LENGTH} characters",
        )
    return normalized


def _normalize_program(program: str | None) -> str | None:
    if program is None:
        return None
    if not isinstance(program, str) or not program.strip():
        raise ReferenceCaseSearchError(
            "invalid_reference_program",
            "program must be non-empty text",
        )
    return program.strip().casefold()


def _normalize_status(status: str | None) -> str:
    if status is None:
        return "validated_reference"
    if not isinstance(status, str) or status not in REFERENCE_CASE_STATUSES:
        raise ReferenceCaseSearchError(
            "invalid_reference_status",
            "scientific_status must be one of: "
            + ", ".join(sorted(REFERENCE_CASE_STATUSES)),
        )
    return status


def _normalize_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ReferenceCaseSearchError(
            "invalid_reference_limit",
            "limit must be an integer",
        )
    if not 1 <= limit <= MAX_REFERENCE_RESULTS:
        raise ReferenceCaseSearchError(
            "invalid_reference_limit",
            f"limit must be between 1 and {MAX_REFERENCE_RESULTS}",
        )
    return limit


def _match_score(query_tokens: frozenset[str], case: dict[str, Any]) -> int:
    fields = [
        case["id"],
        *case["programs"],
        case["status"],
        *case["purposes"],
        *case.get("tags", []),
    ]
    for artifact in case["artifacts"]:
        if not artifact["required"]:
            continue
        fields.extend((artifact["id"], artifact["kind"], *artifact["roles"]))
    indexed_tokens = frozenset(
        token
        for field in fields
        for token in _TOKEN_RE.findall(field.casefold())
    )
    return sum(3 for token in query_tokens if token in indexed_tokens)


def _case_record(case: dict[str, Any]) -> dict[str, Any]:
    required = [
        artifact for artifact in case["artifacts"] if artifact["required"]
    ]
    return {
        "case_id": case["id"],
        "programs": list(case["programs"]),
        "scientific_status": case["status"],
        "purposes": list(case["purposes"]),
        "tags": list(case.get("tags", [])),
        "review": dict(case["review"]),
        "pinning": {
            "required_artifact_count": len(required),
            "total_size_bytes": sum(item["size_bytes"] for item in required),
            "required_artifacts": [{
                "id": artifact["id"],
                "roles": list(artifact["roles"]),
                "kind": artifact["kind"],
                "relative_path": artifact["path"],
                "size_bytes": artifact["size_bytes"],
                "sha256": artifact["sha256"],
                "redistribution": artifact["redistribution"],
            } for artifact in required],
        },
    }


__all__ = [
    "MAX_REFERENCE_QUERY_LENGTH",
    "MAX_REFERENCE_RESULTS",
    "REFERENCE_CASE_SEARCH_SCHEMA",
    "REFERENCE_CASE_STATUSES",
    "ReferenceCaseSearchError",
    "bundled_reference_manifest_paths",
    "find_reference_cases",
]

"""Resolve and lint bounded QMCPACK XML include graphs without merging them."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree

from chemtools.core.types import LintIssue
from chemtools.programs.qmcpack.input import lint_qmcpack_fragment


_MAX_INCLUDE_FILES = 64
_MAX_INCLUDED_XML_BYTES = 8 * 1024 * 1024


def inspect_xml_includes(
    input_path: str | Path,
    parsed_input: Mapping[str, Any],
) -> dict[str, Any]:
    source = Path(input_path).expanduser().resolve()
    includes = [
        value
        for value in parsed_input.get("includes", [])
        if isinstance(value, str) and value
    ]
    if not includes:
        return {
            "status": "not_applicable",
            "resolution": {
                "base_path": str(source.parent),
                "basis": "input_directory_assumption",
            },
            "entries": [],
        }

    entries, references = _inspect_include_graph(source, includes)
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
        **(
            {"discovered_references": references}
            if references else {}
        ),
    }


def include_issues(review: Mapping[str, Any]) -> list[LintIssue]:
    issues: list[LintIssue] = []
    for entry_value in review.get("entries", []):
        if not isinstance(entry_value, Mapping):
            continue
        href = str(entry_value.get("href") or "")
        if entry_value.get("status") == "missing":
            issues.append(_issue(
                "error",
                f"Referenced XML include {href!r} was not found relative to the reviewed input.",
                suggested_fix=(
                    "Provide the included XML file in the input directory or "
                    "correct the include href."
                ),
            ))
        elif entry_value.get("status") == "not_file":
            issues.append(_issue(
                "error",
                f"Referenced XML include {href!r} resolves to a non-file path.",
            ))
        elif entry_value.get("status") == "unreadable":
            issues.append(_issue(
                "warning",
                f"Referenced XML include {href!r} could not be inspected.",
            ))
        elif entry_value.get("status") == "invalid":
            issues.append(_issue(
                "error",
                f"Referenced XML include {href!r} is not well-formed XML.",
            ))
        elif entry_value.get("status") == "cycle":
            issues.append(_issue(
                "error",
                f"Referenced XML include {href!r} creates an include cycle.",
            ))
        elif entry_value.get("status") == "limit_reached":
            issues.append(_issue(
                "warning",
                "Nested XML includes exceeded the review limit of 64 files.",
            ))
        elif entry_value.get("scan_status") == "too_large":
            issues.append(_issue(
                "warning",
                (
                    f"Referenced XML include {href!r} exceeds the 8 MiB "
                    "nested-review limit."
                ),
            ))
    return issues


def included_xml_lint_issues(review: Mapping[str, Any]) -> list[LintIssue]:
    issues: list[LintIssue] = []
    seen_paths: set[str] = set()
    for entry_value in review.get("entries", []):
        if not isinstance(entry_value, Mapping):
            continue
        if entry_value.get("status") != "present":
            continue
        if entry_value.get("scan_status") == "too_large":
            continue
        path = entry_value.get("path")
        href = entry_value.get("href")
        if not isinstance(path, str) or not isinstance(href, str):
            continue
        if path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for issue in lint_qmcpack_fragment(text):
            issues.append({
                **issue,
                "message": f"Included QMCPACK XML {href!r}: {issue['message']}",
            })
    return issues


def _inspect_include(source: Path, href: str) -> dict[str, Any]:
    configured = Path(href).expanduser()
    path = configured.resolve() if configured.is_absolute() else (
        source.parent / configured
    ).resolve()
    common = {"href": href, "path": str(path)}
    try:
        metadata = path.stat()
    except FileNotFoundError:
        return {**common, "status": "missing"}
    except OSError as error:
        return {
            **common,
            "status": "unreadable",
            "reason": f"{type(error).__name__}: {error}",
        }
    if not path.is_file():
        return {**common, "status": "not_file"}
    return {
        **common,
        "status": "present",
        "size_bytes": metadata.st_size,
        "modified_ns": metadata.st_mtime_ns,
    }


def _inspect_include_graph(
    source: Path,
    includes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    entries = []
    references = []
    pending = [(source, href, (source,)) for href in includes]
    while pending:
        parent, href, ancestors = pending.pop(0)
        if len(entries) >= _MAX_INCLUDE_FILES:
            entries.append({"href": href, "status": "limit_reached"})
            break
        configured = Path(href).expanduser()
        candidate = configured.resolve() if configured.is_absolute() else (
            parent.parent / configured
        ).resolve()
        if candidate in ancestors:
            entries.append({
                "href": href,
                "path": str(candidate),
                "status": "cycle",
            })
            continue
        entry = _inspect_include(parent, href)
        entries.append(entry)
        if entry["status"] != "present":
            continue
        if entry["size_bytes"] > _MAX_INCLUDED_XML_BYTES:
            entry["scan_status"] = "too_large"
            continue
        try:
            root = ElementTree.parse(candidate).getroot()
        except (ElementTree.ParseError, OSError):
            entry["status"] = "invalid"
            continue
        for element in root.iter():
            reference = _attribute_reference(element)
            if reference is None:
                continue
            if reference["kind"] == "include":
                pending.append((candidate, reference["href"], (*ancestors, candidate)))
            else:
                references.append({
                    **reference,
                    "source_path": str(candidate),
                })
    return entries, references


def _attribute_reference(
    element: ElementTree.Element,
) -> dict[str, str] | None:
    href = element.get("href")
    if href is None or not href.strip():
        return None
    tag = element.tag.rsplit("}", 1)[-1]
    reference = {"kind": "include" if tag == "include" else tag, "href": href.strip()}
    if tag == "pseudo":
        element_type = element.get("elementType")
        if element_type is not None and element_type.strip():
            reference["element"] = element_type.strip()
    return reference


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


__all__ = ["include_issues", "included_xml_lint_issues", "inspect_xml_includes"]

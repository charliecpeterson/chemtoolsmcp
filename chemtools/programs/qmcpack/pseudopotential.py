"""Inspect the semilocal QMCPACK pseudopotential XML needed for DMC review."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


_MAX_PSEUDOPOTENTIAL_BYTES = 8 * 1024 * 1024
_TAIL_SAMPLE_COUNT = 3
_ANGULAR_MOMENTA = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}


def inspect_qmcpack_pseudopotential(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.stat().st_size > _MAX_PSEUDOPOTENTIAL_BYTES:
        raise ValueError(
            f"{source} exceeds the {_MAX_PSEUDOPOTENTIAL_BYTES}-byte inspection limit."
        )
    try:
        root = ElementTree.parse(source).getroot()
    except ElementTree.ParseError as error:
        raise ValueError(f"QMCPACK pseudopotential XML is malformed: {error}.") from error
    if _tag_name(root.tag) != "pseudo":
        raise ValueError(f"QMCPACK pseudopotential root must be <pseudo>, found <{_tag_name(root.tag)}>.")

    header = _required_child(root, "header")
    semilocal = _required_child(root, "semilocal")
    zval = _required_float(header, "zval")
    channels = [_channel_summary(vps, zval) for vps in _children(semilocal, "vps")]
    if not channels:
        raise ValueError("QMCPACK semilocal pseudopotential contains no <vps> channels.")

    semilocal_summary = {
        "units": semilocal.get("units"),
        "format": semilocal.get("format"),
        "npots_down": _optional_int(semilocal, "npots-down"),
        "npots_up": _optional_int(semilocal, "npots-up"),
        "local_channel": _optional_int(semilocal, "l-local"),
        "channels": channels,
    }
    structural_evidence = _semilocal_structural_evidence(semilocal_summary)
    return {
        "schema_version": "chemtools.qmcpack-pseudopotential/1",
        "path": str(source),
        "header": {
            "symbol": header.get("symbol"),
            "atomic_number": _optional_int(header, "atomic-number"),
            "zval": zval,
            "relativistic": header.get("relativistic"),
            "flavor": header.get("flavor"),
            "creator": header.get("creator"),
        },
        "grid": _grid_summary(_required_child(root, "grid")),
        "semilocal": semilocal_summary,
        "structural_evidence": structural_evidence,
        "tail_check": {
            "expected_r_times_v_hartree": -zval,
            "samples_per_channel": _TAIL_SAMPLE_COUNT,
            "channels": [channel["tail"] for channel in channels],
        },
        "warnings": _structural_warnings(structural_evidence),
        "scope_limit": (
            "This reports XML structure and observed r*V tail evidence. It does "
            "not establish pseudopotential transferability or DMC compatibility."
        ),
    }


def collect_pseudopotential_references(
    parsed_input: dict[str, Any],
    include_review: dict[str, Any],
    input_path: str | Path,
) -> list[dict[str, Any]]:
    source = Path(input_path).expanduser().resolve()
    references = [*parsed_input["references"]]
    references.extend(include_review.get("discovered_references", []))
    resolved: dict[str, dict[str, Any]] = {}
    for reference in references:
        if reference.get("kind") != "pseudo":
            continue
        href = reference.get("href")
        if not isinstance(href, str) or not href:
            continue
        source_path = Path(reference.get("source_path", source))
        configured = Path(href).expanduser()
        path = configured.resolve() if configured.is_absolute() else (
            source_path.parent / configured
        ).resolve()
        entry = resolved.setdefault(str(path), {"path": str(path), "references": []})
        entry["references"].append({
            "href": href,
            "source_path": str(source_path),
            "element": reference.get("element"),
        })
    return list(resolved.values())


def inspect_referenced_pseudopotentials(
    parsed_input: dict[str, Any],
    include_review: dict[str, Any],
    input_path: str | Path,
) -> dict[str, Any]:
    references = collect_pseudopotential_references(
        parsed_input,
        include_review,
        input_path,
    )
    if not references:
        return {
            "name": "qmcpack_pseudopotential_structure",
            "status": "review_required",
            "observed": {"references": []},
            "message": (
                "The QMCPACK deck declares no pseudopotential XML reference; "
                "confirm whether an all-electron Hamiltonian is intended."
            ),
        }

    inspections = []
    statuses = []
    for reference in references:
        try:
            inspection = inspect_qmcpack_pseudopotential(reference["path"])
        except (OSError, ValueError) as error:
            inspections.append({
                **reference,
                "status": "not_ready",
                "error": str(error),
            })
            statuses.append("not_ready")
            continue
        evidence = inspection["structural_evidence"]
        structure_status = (
            "not_ready"
            if any(value is False for value in evidence.values())
            else "review_required"
            if any(value is None for value in evidence.values())
            else "pass"
        )
        identity = _reference_identity(reference, inspection)
        status = (
            "not_ready"
            if "not_ready" in {structure_status, identity["status"]}
            else "review_required"
            if "review_required" in {structure_status, identity["status"]}
            else "pass"
        )
        inspections.append({
            **reference,
            "status": status,
            "inspection": inspection,
            "reference_identity": identity,
        })
        statuses.append(status)

    status = _assessment_status(statuses)
    if include_review["status"] == "incomplete" and status == "pass":
        status = "review_required"
    return {
        "name": "qmcpack_pseudopotential_structure",
        "status": status,
        "observed": {
            "include_review_status": include_review["status"],
            "inspections": inspections,
        },
        "message": {
            "pass": (
                "Every declared QMCPACK pseudopotential has the supported "
                "semilocal structural evidence and matching declared element."
            ),
            "review_required": (
                "QMCPACK pseudopotential structure or declared-element identity "
                "is incomplete or not fully established by the bounded XML review."
            ),
            "not_ready": (
                "At least one declared QMCPACK pseudopotential is missing, "
                "malformed, has a declared-element mismatch, or fails the "
                "supported structural checks."
            ),
        }[status],
    }


def _assessment_status(statuses: list[str]) -> str:
    if "not_ready" in statuses:
        return "not_ready"
    if "review_required" in statuses:
        return "review_required"
    return "pass"


def _reference_identity(
    reference: dict[str, Any],
    inspection: dict[str, Any],
) -> dict[str, Any]:
    declared_elements = sorted({
        element.strip()
        for item in reference["references"]
        if isinstance((element := item.get("element")), str) and element.strip()
    })
    header_symbol = inspection["header"].get("symbol")
    if not declared_elements:
        return {
            "status": "review_required",
            "declared_elements": [],
            "header_symbol": header_symbol,
            "reason": "No pseudopotential elementType declaration was available.",
        }
    if not isinstance(header_symbol, str) or not header_symbol.strip():
        return {
            "status": "review_required",
            "declared_elements": declared_elements,
            "header_symbol": header_symbol,
            "reason": "The pseudopotential header has no symbol attribute.",
        }
    normalized_symbol = header_symbol.strip().casefold()
    matches = all(element.casefold() == normalized_symbol for element in declared_elements)
    return {
        "status": "pass" if matches else "not_ready",
        "declared_elements": declared_elements,
        "header_symbol": header_symbol,
        "matches_header_symbol": matches,
    }


def _channel_summary(element: ElementTree.Element, zval: float) -> dict[str, Any]:
    radfunc = _required_child(element, "radfunc")
    data = _required_child(radfunc, "data")
    try:
        values = [float(value) for value in (data.text or "").split()]
    except ValueError as error:
        raise ValueError("QMCPACK pseudopotential <data> contains a non-numeric value.") from error
    if not values:
        raise ValueError("QMCPACK pseudopotential <data> contains no values.")
    if not all(isfinite(value) for value in values):
        raise ValueError("QMCPACK pseudopotential <data> contains a non-finite value.")
    grid = _grid_summary(_required_child(radfunc, "grid"))
    tail_values = values[-_TAIL_SAMPLE_COUNT:]
    tail_mean = sum(tail_values) / len(tail_values)
    return {
        "l": element.get("l"),
        "angular_momentum": _angular_momentum(element.get("l")),
        "principal_n": _optional_int(element, "principal-n"),
        "spin": _optional_int(element, "spin"),
        "cutoff": _optional_float(element, "cutoff"),
        "data_point_count": len(values),
        "grid": grid,
        "declared_grid_count_matches_data": (
            grid["npts"] == len(values) if grid["npts"] is not None else None
        ),
        "tail": {
            "l": element.get("l"),
            "values_hartree": tail_values,
            "mean_hartree": tail_mean,
            "difference_from_expected_hartree": tail_mean + zval,
        },
    }


def _required_child(
    element: ElementTree.Element,
    name: str,
) -> ElementTree.Element:
    for child in element:
        if _tag_name(child.tag) == name:
            return child
    raise ValueError(f"QMCPACK pseudopotential is missing <{name}>.")


def _children(element: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [child for child in element if _tag_name(child.tag) == name]


def _grid_summary(element: ElementTree.Element) -> dict[str, Any]:
    return {
        "type": element.get("type"),
        "units": element.get("units"),
        "ri": _optional_float(element, "ri"),
        "rf": _optional_float(element, "rf"),
        "npts": _optional_int(element, "npts"),
    }


def _semilocal_structural_evidence(
    semilocal: dict[str, Any],
) -> dict[str, bool | None]:
    channels = semilocal["channels"]
    angular_momenta = [channel["angular_momentum"] for channel in channels]
    local_channel = semilocal["local_channel"]
    declared_grid_matches = [
        channel["declared_grid_count_matches_data"] for channel in channels
    ]
    channel_labels_are_recognized = all(
        angular_momentum is not None for angular_momentum in angular_momenta
    )
    channel_spin_pairs = [
        (channel["angular_momentum"], channel["spin"])
        for channel in channels
    ]
    return {
        "units_are_hartree": semilocal["units"] == "hartree",
        "format_is_r_times_v": semilocal["format"] == "r*V",
        "all_channel_grids_are_linear": all(
            channel["grid"]["type"] == "linear" for channel in channels
        ),
        "all_declared_grid_counts_match_data": (
            all(declared_grid_matches)
            if all(match is not None for match in declared_grid_matches)
            else None
        ),
        "local_channel_has_vps": (
            local_channel in angular_momenta
            if local_channel is not None and channel_labels_are_recognized
            else None
        ),
        "channel_labels_are_recognized": channel_labels_are_recognized,
        "channel_spin_pairs_are_unique": (
            len(channel_spin_pairs) == len(set(channel_spin_pairs))
            if channel_labels_are_recognized
            else None
        ),
    }


def _structural_warnings(evidence: dict[str, bool | None]) -> list[str]:
    warnings = []
    if evidence["units_are_hartree"] is False:
        warnings.append("<semilocal> units are not 'hartree'.")
    if evidence["format_is_r_times_v"] is False:
        warnings.append("<semilocal> format is not 'r*V'.")
    if evidence["all_channel_grids_are_linear"] is False:
        warnings.append("At least one <vps> channel does not declare a linear grid.")
    if evidence["all_declared_grid_counts_match_data"] is False:
        warnings.append("At least one <vps> grid npts value does not match its data count.")
    if evidence["local_channel_has_vps"] is False:
        warnings.append("The declared l-local channel has no matching <vps> channel.")
    if evidence["channel_labels_are_recognized"] is False:
        warnings.append("At least one <vps> channel label is not a supported angular momentum.")
    if evidence["channel_spin_pairs_are_unique"] is False:
        warnings.append("Multiple <vps> channels use the same angular momentum and spin.")
    return warnings


def _angular_momentum(label: str | None) -> int | None:
    if label in _ANGULAR_MOMENTA:
        return _ANGULAR_MOMENTA[label]
    try:
        return int(label) if label is not None else None
    except ValueError:
        return None


def _required_float(element: ElementTree.Element, name: str) -> float:
    value = _optional_float(element, name)
    if value is None:
        raise ValueError(f"QMCPACK pseudopotential <{_tag_name(element.tag)}> needs numeric {name!r}.")
    return value


def _optional_float(element: ElementTree.Element, name: str) -> float | None:
    value = element.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(
            f"QMCPACK pseudopotential <{_tag_name(element.tag)}> has non-numeric {name!r}."
        ) from error


def _optional_int(element: ElementTree.Element, name: str) -> int | None:
    value = element.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            f"QMCPACK pseudopotential <{_tag_name(element.tag)}> has non-integer {name!r}."
        ) from error


def _tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


__all__ = [
    "collect_pseudopotential_references",
    "inspect_qmcpack_pseudopotential",
    "inspect_referenced_pseudopotentials",
]

"""Inspect referenced UPF headers and compare their cutoff suggestions.

Only the bounded UPF preamble is read. Suggested cutoffs are reported as
screening evidence, never as proof that a calculation is converged.
"""

from __future__ import annotations

from html import unescape
from pathlib import Path
import re
from typing import Any, Mapping

from chemtools.core.types import LintIssue
from chemtools.programs.qe._elements import element_from_label


_HEADER_LIMIT_BYTES = 256 * 1024
_UPF_VERSION_RE = re.compile(
    r"<UPF\b[^>]*\bversion\s*=\s*(['\"])(.*?)\1",
    re.IGNORECASE | re.DOTALL,
)
_PP_HEADER_RE = re.compile(r"<PP_HEADER\b(.*?)>", re.IGNORECASE | re.DOTALL)
_PP_BETA_RE = re.compile(
    r"<PP_BETA(?:\.\d+)?\b(.*?)>",
    re.IGNORECASE | re.DOTALL,
)
_ATTRIBUTE_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(['\"])(.*?)\2",
    re.DOTALL,
)


class UpfHeaderError(ValueError):
    """Raised when the bounded UPF preamble has no usable PP_HEADER."""


def parse_upf_header(path: str | Path) -> dict[str, Any]:
    """Parse the metadata attributes needed for pre-execution review."""
    source = Path(path).expanduser().resolve()
    with source.open("rb") as handle:
        preamble = handle.read(_HEADER_LIMIT_BYTES).decode(
            "utf-8",
            errors="replace",
        )

    header_match = _PP_HEADER_RE.search(preamble)
    if header_match is None:
        raise UpfHeaderError(
            f"no PP_HEADER found in the first {_HEADER_LIMIT_BYTES} bytes"
        )
    attributes = {
        name.lower(): unescape(value).strip()
        for name, _, value in _ATTRIBUTE_RE.findall(header_match.group(1))
    }
    if not attributes:
        raise UpfHeaderError("PP_HEADER contains no readable attributes")

    version_match = _UPF_VERSION_RE.search(preamble)
    projector_count = _nonnegative_int(attributes.get("number_of_proj"))
    return {
        "schema_version": "chemtools.qe-upf-header/1",
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "upf_version": version_match.group(2).strip() if version_match else None,
        "element": _optional_text(attributes.get("element")),
        "pseudo_type": _optional_text(attributes.get("pseudo_type")),
        "relativistic": _optional_text(attributes.get("relativistic")),
        "functional": _normalized_text(attributes.get("functional")),
        "z_valence": _optional_float(attributes.get("z_valence")),
        "local_channel": _optional_int(attributes.get("l_local")),
        "projector_count": projector_count,
        "projector_channel_evidence": _projector_channel_evidence(
            preamble,
            projector_count,
        ),
        "suggested_ecutwfc_ry": _positive_float(attributes.get("wfc_cutoff")),
        "suggested_ecutrho_ry": _positive_float(attributes.get("rho_cutoff")),
        "is_ultrasoft": _optional_bool(attributes.get("is_ultrasoft")),
        "is_paw": _optional_bool(attributes.get("is_paw")),
        "has_spin_orbit": _optional_bool(attributes.get("has_so")),
        "core_correction": _optional_bool(attributes.get("core_correction")),
    }


def inspect_input_pseudopotentials(
    input_path: str | Path,
    parsed_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve referenced UPFs and summarize the hardest cutoff suggestion."""
    source = Path(input_path).expanduser().resolve()
    control = _mapping(parsed_input.get("namelists", {})).get("control", {})
    pseudo_dir = _mapping(control).get("pseudo_dir")
    if not isinstance(pseudo_dir, str) or not pseudo_dir.strip():
        return {
            "status": "unresolved",
            "resolution": {
                "pseudo_dir": None,
                "base_path": None,
                "basis": "qe_runtime_default",
            },
            "entries": [],
            "cutoff_review": _cutoff_review(parsed_input, []),
        }

    configured = Path(pseudo_dir).expanduser()
    if configured.is_absolute():
        base = configured.resolve()
        basis = "absolute_pseudo_dir"
    else:
        base = (source.parent / configured).resolve()
        basis = "input_directory_assumption"

    entries = [
        _inspect_species(base, species)
        for species in parsed_input.get("atomic_species", [])
        if isinstance(species, Mapping)
    ]
    parsed_headers = [
        entry["upf"]
        for entry in entries
        if entry["status"] == "parsed"
    ]
    if entries and all(entry["status"] == "parsed" for entry in entries):
        status = "parsed"
    elif any(entry["status"] == "parsed" for entry in entries):
        status = "partial"
    else:
        status = "unresolved"
    return {
        "status": status,
        "resolution": {
            "pseudo_dir": pseudo_dir,
            "base_path": str(base),
            "basis": basis,
        },
        "entries": entries,
        "cutoff_review": _cutoff_review(parsed_input, parsed_headers),
    }


def pseudopotential_issues(review: Mapping[str, Any]) -> list[LintIssue]:
    """Convert path, identity, and cutoff findings into guided-review issues."""
    issues: list[LintIssue] = []
    resolution = _mapping(review.get("resolution"))
    if resolution.get("basis") == "qe_runtime_default":
        issues.append(_issue(
            "warning",
            (
                "pseudo_dir is not explicit, so referenced pseudopotentials "
                "could not be inspected before execution."
            ),
            suggested_fix="Set pseudo_dir to the directory containing the UPF files.",
        ))

    for entry_value in review.get("entries", []):
        entry = _mapping(entry_value)
        status = entry.get("status")
        label = entry.get("species_label")
        filename = entry.get("filename")
        line = entry.get("line") if isinstance(entry.get("line"), int) else None
        if status == "missing":
            issues.append(_issue(
                "warning",
                (
                    f"Pseudopotential {filename!r} for species {label!r} was "
                    "not found relative to the reviewed input."
                ),
                line=line,
                suggested_fix=(
                    "Confirm the execution working directory or correct "
                    "pseudo_dir and the ATOMIC_SPECIES filename."
                ),
            ))
        elif status == "invalid":
            issues.append(_issue(
                "warning",
                f"Pseudopotential {filename!r} has no usable UPF PP_HEADER.",
                line=line,
            ))
        elif status == "unreadable":
            issues.append(_issue(
                "warning",
                f"Pseudopotential {filename!r} could not be read.",
                line=line,
            ))
        if status != "parsed":
            continue
        upf = _mapping(entry.get("upf"))
        element = upf.get("element")
        if isinstance(element, str) and not _label_matches_element(str(label), element):
            issues.append(_issue(
                "error",
                (
                    f"Species label {label!r} references a UPF whose "
                    f"PP_HEADER element is {element!r}."
                ),
                line=line,
                suggested_fix="Use a pseudopotential generated for this species.",
            ))

    cutoff = _mapping(review.get("cutoff_review"))
    if cutoff.get("wavefunction_status") == "below_suggestion":
        issues.append(_issue(
            "warning",
            (
                f"ecutwfc={cutoff['ecutwfc_ry']:g} Ry is below the hardest "
                f"positive UPF suggestion of {cutoff['suggested_ecutwfc_ry']:g} Ry."
            ),
            suggested_fix=(
                f"Use ecutwfc >= {cutoff['suggested_ecutwfc_ry']:g} Ry as the "
                "starting point for a convergence study."
            ),
        ))
    if cutoff.get("density_status") == "below_suggestion":
        source = cutoff.get("ecutrho_source")
        issues.append(_issue(
            "warning",
            (
                f"The {source} ecutrho={cutoff['effective_ecutrho_ry']:g} Ry "
                "is below the hardest positive UPF suggestion of "
                f"{cutoff['suggested_ecutrho_ry']:g} Ry."
            ),
            suggested_fix=(
                f"Use ecutrho >= {cutoff['suggested_ecutrho_ry']:g} Ry as the "
                "starting point for a convergence study."
            ),
        ))
    return issues


def _inspect_species(base: Path, species: Mapping[str, Any]) -> dict[str, Any]:
    filename = str(species.get("pseudopotential") or "")
    path = (base / filename).resolve()
    common = {
        "species_label": species.get("label"),
        "filename": filename,
        "path": str(path),
        "line": species.get("line"),
    }
    if not path.is_file():
        return {**common, "status": "missing", "upf": None}
    try:
        header = parse_upf_header(path)
    except UpfHeaderError as exc:
        return {
            **common,
            "status": "invalid",
            "reason": str(exc),
            "upf": None,
        }
    except OSError as exc:
        return {
            **common,
            "status": "unreadable",
            "reason": f"{type(exc).__name__}: {exc}",
            "upf": None,
        }
    return {**common, "status": "parsed", "upf": header}


def _cutoff_review(
    parsed_input: Mapping[str, Any],
    headers: list[Mapping[str, Any]],
) -> dict[str, Any]:
    system = _mapping(parsed_input.get("system"))
    ecutwfc = _optional_float(system.get("ecutwfc_ry"))
    explicit_ecutrho = _optional_float(system.get("ecutrho_ry"))
    effective_ecutrho = (
        explicit_ecutrho
        if explicit_ecutrho is not None
        else 4.0 * ecutwfc if ecutwfc is not None else None
    )
    wfc_suggestion = _hardest_suggestion(headers, "suggested_ecutwfc_ry")
    rho_suggestion = _hardest_suggestion(headers, "suggested_ecutrho_ry")
    suggested_wfc = wfc_suggestion.get("value_ry") if wfc_suggestion else None
    suggested_rho = rho_suggestion.get("value_ry") if rho_suggestion else None
    return {
        "ecutwfc_ry": ecutwfc,
        "effective_ecutrho_ry": effective_ecutrho,
        "ecutrho_source": "explicit" if explicit_ecutrho is not None else "qe_default_4x",
        "suggested_ecutwfc_ry": suggested_wfc,
        "suggested_ecutrho_ry": suggested_rho,
        "suggested_ecutwfc_source": wfc_suggestion,
        "suggested_ecutrho_source": rho_suggestion,
        "wavefunction_status": _comparison(ecutwfc, suggested_wfc),
        "density_status": _comparison(effective_ecutrho, suggested_rho),
        "convergence_established": False,
    }


def _comparison(actual: float | None, suggested: float | None) -> str:
    if actual is None or suggested is None:
        return "suggestion_unavailable"
    return "meets_suggestion" if actual >= suggested else "below_suggestion"


def _hardest_suggestion(
    headers: list[Mapping[str, Any]],
    key: str,
) -> dict[str, Any] | None:
    candidates = [
        header
        for header in headers
        if isinstance(header.get(key), (int, float))
        and not isinstance(header.get(key), bool)
        and header[key] > 0
    ]
    if not candidates:
        return None
    hardest = max(candidates, key=lambda header: float(header[key]))
    return {
        "value_ry": float(hardest[key]),
        "element": hardest.get("element"),
        "path": hardest.get("path"),
    }


def _label_matches_element(label: str, element: str) -> bool:
    return element_from_label(label) == element.strip().capitalize()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_text(value: str | None) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _normalized_text(value: str | None) -> str | None:
    text = _optional_text(value)
    return " ".join(text.split()) if text else None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        return float(value.replace("D", "e").replace("d", "e"))
    except ValueError:
        return None


def _positive_float(value: Any) -> float | None:
    number = _optional_float(value)
    return number if number is not None and number > 0 else None


def _optional_int(value: str | None) -> int | None:
    number = _optional_float(value)
    return int(number) if number is not None and number.is_integer() else None


def _nonnegative_int(value: str | None) -> int | None:
    number = _optional_int(value)
    return number if number is not None and number >= 0 else None


def _projector_channel_evidence(
    preamble: str,
    declared_total: int | None,
) -> dict[str, Any]:
    angular_momenta = []
    invalid_count = 0
    for attributes in _PP_BETA_RE.findall(preamble):
        values = {
            name.lower(): unescape(value).strip()
            for name, _, value in _ATTRIBUTE_RE.findall(attributes)
        }
        angular_momentum = _nonnegative_int(values.get("angular_momentum"))
        if angular_momentum is None:
            invalid_count += 1
        else:
            angular_momenta.append(angular_momentum)

    observed_total = len(angular_momenta) + invalid_count
    complete = (
        declared_total is not None
        and observed_total == declared_total
        and invalid_count == 0
    )
    return {
        "status": (
            "complete"
            if complete
            else "partial"
            if observed_total
            else "not_available"
        ),
        "declared_total": declared_total,
        "observed_total": observed_total,
        "invalid_angular_momentum_count": invalid_count,
        "counts_by_angular_momentum": {
            str(angular_momentum): angular_momenta.count(angular_momentum)
            for angular_momentum in sorted(set(angular_momenta))
        },
        "declared_total_matches_observed": (
            declared_total == observed_total if declared_total is not None else None
        ),
    }


def _optional_bool(value: str | None) -> bool | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"true", ".true.", "t"}:
        return True
    if normalized in {"false", ".false.", "f"}:
        return False
    return None


def _issue(
    level: str,
    message: str,
    *,
    line: int | None = None,
    suggested_fix: str | None = None,
) -> LintIssue:
    return {
        "level": level,  # type: ignore[typeddict-item]
        "message": message,
        "line": line,
        "suggested_fix": suggested_fix,
    }


__all__ = [
    "UpfHeaderError",
    "inspect_input_pseudopotentials",
    "parse_upf_header",
    "pseudopotential_issues",
]

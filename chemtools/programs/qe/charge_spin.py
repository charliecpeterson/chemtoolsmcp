"""Review deterministic charge and spin relationships in pw.x inputs.

The checks identify internally inconsistent controls and summarize electron
accounting from inspected UPF headers. They do not select a physical spin state.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

from chemtools.core.types import LintIssue


_INDEXED_VALUE_RE = re.compile(r"^(starting_magnetization|starting_charge)\((\d+)\)$")


def inspect_charge_spin(
    parsed_input: Mapping[str, Any],
    pseudopotential_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize charge, spin controls, and UPF-based electron accounting."""
    namelists = _mapping(parsed_input.get("namelists"))
    system = _mapping(namelists.get("system"))
    assignment_lines = _mapping(parsed_input.get("assignment_lines"))
    system_lines = _mapping(assignment_lines.get("system"))
    indexed = _indexed_values(system, system_lines)
    nspin_raw = system.get("nspin", 1)
    nspin = _integer(nspin_raw)
    noncolin = system.get("noncolin", False)
    lspinorb = system.get("lspinorb", False)
    tot_charge = _real(system.get("tot_charge", 0.0))
    tot_magnetization = (
        _real(system.get("tot_magnetization"))
        if "tot_magnetization" in system
        else None
    )
    constrained = str(system.get("constrained_magnetization", "none")).lower()

    return {
        "schema_version": "chemtools.qe-charge-spin-review/1",
        "source": {
            "nat": _mapping(parsed_input.get("system")).get("nat"),
            "ntyp": _mapping(parsed_input.get("system")).get("ntyp"),
            "assignment_lines": dict(system_lines),
        },
        "spin": {
            "mode": _spin_mode(nspin, noncolin, lspinorb),
            "nspin": nspin_raw,
            "nspin_explicit": "nspin" in system,
            "noncolin": noncolin,
            "lspinorb": lspinorb,
            "tot_magnetization": tot_magnetization,
            "tot_magnetization_explicit": "tot_magnetization" in system,
            "constrained_magnetization": constrained,
            "starting_magnetization": indexed["starting_magnetization"],
        },
        "charge": {
            "tot_charge": tot_charge,
            "tot_charge_raw": system.get("tot_charge", 0.0),
            "starting_charge": indexed["starting_charge"],
        },
        "electron_accounting": _electron_accounting(
            parsed_input,
            pseudopotential_review,
            tot_charge,
        ),
        "spin_orbit_pseudopotentials": _spin_orbit_summary(
            pseudopotential_review
        ),
        "scope": (
            "Internal consistency only; the requested charge and physical "
            "spin state are not validated."
        ),
    }


def charge_spin_issues(review: Mapping[str, Any]) -> list[LintIssue]:
    """Convert contradictory or incomplete charge and spin controls to issues."""
    spin = _mapping(review.get("spin"))
    charge = _mapping(review.get("charge"))
    issues: list[LintIssue] = []

    nspin = _integer(spin.get("nspin"))
    nspin_explicit = spin.get("nspin_explicit") is True
    noncolin = spin.get("noncolin")
    lspinorb = spin.get("lspinorb")
    starting_magnetization = _indexed_entries(spin.get("starting_magnetization"))
    starting_charge = _indexed_entries(charge.get("starting_charge"))

    if nspin is None:
        issues.append(_issue(
            "error",
            "nspin must be an integer pw.x spin mode.",
            line=_line(review, "nspin"),
            suggested_fix="Use nspin=1 or nspin=2, or use noncolin=.true. without nspin.",
        ))
    elif noncolin is True and nspin_explicit:
        issues.append(_issue(
            "error",
            "noncolin=.true. must be used without an explicit nspin value.",
            line=_line(review, "noncolin"),
            suggested_fix="Remove nspin and retain noncolin=.true.",
        ))
    elif noncolin is not True and nspin not in {1, 2}:
        issues.append(_issue(
            "error",
            f"nspin={nspin} is not a supported collinear pw.x spin mode.",
            line=_line(review, "nspin"),
            suggested_fix="Use nspin=1 or nspin=2; select noncollinear mode with noncolin=.true.",
        ))

    if not isinstance(noncolin, bool):
        issues.append(_logical_issue("noncolin", _line(review, "noncolin")))
    if not isinstance(lspinorb, bool):
        issues.append(_logical_issue("lspinorb", _line(review, "lspinorb")))
    elif lspinorb and noncolin is not True:
        issues.append(_issue(
            "error",
            "lspinorb=.true. requires the noncollinear calculation path.",
            line=_line(review, "lspinorb"),
            suggested_fix="Set noncolin=.true. and remove nspin.",
        ))

    if charge.get("tot_charge") is None:
        issues.append(_issue(
            "error",
            "tot_charge must be a real number when specified.",
            line=_line(review, "tot_charge"),
            suggested_fix="Set tot_charge to the net cell charge in electron-charge units.",
        ))

    ntyp = _integer_from_review(review, "ntyp")
    if ntyp is not None:
        issues.extend(_index_issues(starting_magnetization, "starting_magnetization", ntyp))
        issues.extend(_index_issues(starting_charge, "starting_charge", ntyp))

    invalid_magnetization = [
        entry["index"]
        for entry in starting_magnetization
        if _real(entry.get("value")) is None
    ]
    if invalid_magnetization:
        issues.append(_issue(
            "error",
            f"starting_magnetization must be real for species indices {invalid_magnetization}.",
            line=_entry_line(starting_magnetization, invalid_magnetization[0]),
        ))

    invalid_charge = [
        entry["index"]
        for entry in starting_charge
        if _real(entry.get("value")) is None
    ]
    if invalid_charge:
        issues.append(_issue(
            "error",
            f"starting_charge must be real for species indices {invalid_charge}.",
            line=_entry_line(starting_charge, invalid_charge[0]),
        ))

    if spin.get("tot_magnetization_explicit") and spin.get("tot_magnetization") is None:
        issues.append(_issue(
            "error",
            "tot_magnetization must be a real number when specified.",
            line=_line(review, "tot_magnetization"),
        ))
    elif spin.get("tot_magnetization_explicit") and starting_magnetization:
        issues.append(_issue(
            "error",
            "tot_magnetization and starting_magnetization must not be specified together.",
            line=_line(review, "tot_magnetization"),
            suggested_fix=(
                "Keep the fixed total magnetization or the species starting "
                "values, not both."
            ),
        ))

    if noncolin is True and spin.get("tot_magnetization_explicit"):
        issues.append(_issue(
            "error",
            "tot_magnetization is the collinear LSDA control, not a noncollinear constraint.",
            line=_line(review, "tot_magnetization"),
            suggested_fix="Use constrained_magnetization for a noncollinear constraint.",
        ))

    has_nonzero_seed = any(
        (value := _real(entry.get("value"))) is not None
        and not math.isclose(value, 0.0, abs_tol=1e-15)
        for entry in starting_magnetization
    )
    constrained = spin.get("constrained_magnetization")
    has_constraint = isinstance(constrained, str) and constrained != "none"
    has_fixed_total = spin.get("tot_magnetization_explicit") is True

    if noncolin is not True and nspin == 1 and starting_magnetization:
        issues.append(_issue(
            "warning",
            "starting_magnetization is present in a non-spin-polarized nspin=1 calculation.",
            line=_entry_line(starting_magnetization, starting_magnetization[0]["index"]),
            suggested_fix="Remove it or select the intended spin-polarized mode.",
        ))
    elif noncolin is not True and nspin == 2 and not (has_nonzero_seed or has_fixed_total):
        issues.append(_issue(
            "warning",
            "nspin=2 has no nonzero starting_magnetization or fixed tot_magnetization.",
            line=_line(review, "nspin"),
            suggested_fix=(
                "Set a nonzero starting_magnetization for at least one "
                "species if a magnetic solution is intended."
            ),
        ))
    elif noncolin is True and not (has_nonzero_seed or has_constraint):
        if lspinorb is True:
            issues.append(_issue(
                "info",
                (
                    "The spin-orbit calculation has no nonzero magnetic seed "
                    "or constraint, so it retains time-reversal symmetry and "
                    "zero magnetization."
                ),
                line=_line(review, "noncolin"),
                suggested_fix=(
                    "Set a nonzero starting_magnetization or an appropriate "
                    "constrained_magnetization only if a magnetic solution is intended."
                ),
            ))
        else:
            issues.append(_issue(
                "warning",
                (
                    "The noncollinear calculation has no nonzero magnetic "
                    "seed or magnetization constraint."
                ),
                line=_line(review, "noncolin"),
                suggested_fix=(
                    "Set a nonzero starting_magnetization or an appropriate "
                    "constrained_magnetization if a magnetic solution is intended."
                ),
            ))

    spin_orbit = _mapping(review.get("spin_orbit_pseudopotentials"))
    if (
        lspinorb is True
        and spin_orbit.get("inspected")
        and not spin_orbit.get("any_has_spin_orbit")
    ):
        issues.append(_issue(
            "warning",
            "lspinorb=.true., but none of the inspected UPFs advertises spin-orbit data.",
            line=_line(review, "lspinorb"),
            suggested_fix="Confirm that the intended fully relativistic UPFs are referenced.",
        ))

    accounting = _mapping(review.get("electron_accounting"))
    electron_count = _real(accounting.get("electron_count"))
    if (
        accounting.get("status") == "complete"
        and electron_count is not None
        and electron_count < 0
    ):
        issues.append(_issue(
            "error",
            f"UPF valence accounting gives {electron_count:g} electrons after tot_charge.",
            line=_line(review, "tot_charge"),
            suggested_fix="Check tot_charge, atom counts, and the referenced pseudopotentials.",
        ))
    elif accounting.get("status") == "complete" and electron_count == 0:
        issues.append(_issue(
            "warning",
            "UPF valence accounting gives zero electrons after tot_charge.",
            line=_line(review, "tot_charge"),
            suggested_fix=(
                "Confirm that removing every UPF valence electron is intentional."
            ),
        ))
    return issues


def _indexed_values(
    system: Mapping[str, Any],
    lines: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    values: dict[str, list[dict[str, Any]]] = {
        "starting_magnetization": [],
        "starting_charge": [],
    }
    for key, value in system.items():
        match = _INDEXED_VALUE_RE.match(key)
        if match is None:
            continue
        values[match.group(1)].append({
            "index": int(match.group(2)),
            "value": value,
            "line": lines.get(key),
        })
    for entries in values.values():
        entries.sort(key=lambda entry: entry["index"])
    return values


def _electron_accounting(
    parsed_input: Mapping[str, Any],
    pseudopotential_review: Mapping[str, Any] | None,
    tot_charge: float | None,
) -> dict[str, Any]:
    entries = _pseudo_entries(pseudopotential_review)
    z_valence_by_label: dict[str, float] = {}
    for entry in entries:
        upf = _mapping(entry.get("upf"))
        label = entry.get("species_label")
        z_valence = _real(upf.get("z_valence"))
        if entry.get("status") == "parsed" and isinstance(label, str) and z_valence is not None:
            z_valence_by_label[label] = z_valence

    positions = _mapping(parsed_input.get("atomic_positions"))
    atoms = [
        atom for atom in positions.get("atoms", [])
        if isinstance(atom, Mapping) and isinstance(atom.get("label"), str)
    ]
    labels = [str(atom["label"]) for atom in atoms]
    missing = sorted(set(labels) - set(z_valence_by_label))
    if not atoms or not entries:
        status = "unavailable"
    elif missing or tot_charge is None:
        status = "partial"
    else:
        status = "complete"

    valence = (
        sum(z_valence_by_label[label] for label in labels)
        if status == "complete"
        else None
    )
    return {
        "status": status,
        "basis": "UPF PP_HEADER z_valence summed over ATOMIC_POSITIONS",
        "valence_electrons_before_charge": valence,
        "tot_charge": tot_charge,
        "electron_count": (
            valence - tot_charge
            if valence is not None and tot_charge is not None
            else None
        ),
        "missing_species": missing,
    }


def _spin_orbit_summary(
    pseudopotential_review: Mapping[str, Any] | None,
) -> dict[str, Any]:
    headers = [
        _mapping(entry.get("upf"))
        for entry in _pseudo_entries(pseudopotential_review)
        if entry.get("status") == "parsed"
    ]
    return {
        "inspected": bool(headers),
        "parsed_count": len(headers),
        "any_has_spin_orbit": any(
            header.get("has_spin_orbit") is True for header in headers
        ),
    }


def _spin_mode(nspin: int | None, noncolin: Any, lspinorb: Any) -> str:
    if noncolin is True:
        return "spin_orbit" if lspinorb is True else "noncollinear"
    if nspin == 1:
        return "non_spin_polarized"
    if nspin == 2:
        return "collinear"
    return "invalid"


def _index_issues(
    entries: list[Mapping[str, Any]],
    name: str,
    ntyp: int,
) -> list[LintIssue]:
    invalid = [entry["index"] for entry in entries if not 1 <= entry["index"] <= ntyp]
    if not invalid:
        return []
    return [_issue(
        "error",
        f"{name} species indices {invalid} fall outside 1..ntyp ({ntyp}).",
        line=_entry_line(entries, invalid[0]),
        suggested_fix=f"Use {name}(i) only for species indices 1 through {ntyp}.",
    )]


def _integer_from_review(review: Mapping[str, Any], key: str) -> int | None:
    source = _mapping(review.get("source"))
    return _integer(source.get(key))


def _pseudo_entries(
    review: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    if review is None:
        return []
    return [entry for entry in review.get("entries", []) if isinstance(entry, Mapping)]


def _indexed_entries(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, Mapping)]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _real(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _line(review: Mapping[str, Any], key: str) -> int | None:
    source = _mapping(review.get("source"))
    lines = _mapping(source.get("assignment_lines"))
    value = lines.get(key)
    return value if isinstance(value, int) else None


def _entry_line(entries: list[Mapping[str, Any]], index: int) -> int | None:
    for entry in entries:
        if entry.get("index") == index and isinstance(entry.get("line"), int):
            return entry["line"]
    return None


def _logical_issue(name: str, line: int | None) -> LintIssue:
    return _issue(
        "error",
        f"{name} must be a Fortran logical value.",
        line=line,
        suggested_fix=f"Set {name}=.true. or {name}=.false.",
    )


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


__all__ = ["charge_spin_issues", "inspect_charge_spin"]

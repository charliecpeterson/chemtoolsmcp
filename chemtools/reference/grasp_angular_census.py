"""Check GRASP jj-coupled CSFs against independent angular combinatorics.

The validator works from occupations present in a parsed CSF list. It checks
each represented configuration/J pair without assuming an excitation recipe.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

from chemtools.programs.grasp.parse.csf import (
    CsfDocument,
    load_grasp_csf_list,
)
from chemtools.reference.atomic_multiplets import (
    combine_j_distributions,
    extract_j_levels,
    format_half,
    j_shell_m_distribution,
)


GRASP_ANGULAR_CENSUS_SCHEMA = "chemtools.grasp-angular-census/1"
_ORBITAL_LETTERS = "spdfghiklm"
_RELATIVISTIC_SUBSHELL_RE = re.compile(
    rf"^(?P<n>[1-9]\d?)(?P<orbital>[{_ORBITAL_LETTERS}])(?P<lower>-)?$"
)


def validate_grasp_csf_angular_census(
    path: str | Path,
) -> dict[str, object]:
    """Validate every represented relativistic configuration/J multiplicity."""
    return validate_csf_angular_census(load_grasp_csf_list(path))


def validate_csf_angular_census(
    document: CsfDocument,
) -> dict[str, object]:
    actual: dict[
        tuple[tuple[tuple[str, int], ...], str],
        dict[int, int],
    ] = defaultdict(lambda: defaultdict(int))
    for block in document.blocks:
        for entry in block.entries:
            actual[(entry.occupations, entry.parity)][entry.two_j] += 1

    configurations = []
    all_full = True
    for (occupations, parity), actual_levels in sorted(
        actual.items(),
        key=lambda item: (item[0][1], item[0][0]),
    ):
        expected_parity = _configuration_parity(occupations)
        if parity != expected_parity:
            label = _configuration_label(occupations)
            raise ValueError(
                f"GRASP CSF configuration {label} has parity {parity}; "
                f"occupations require {expected_parity}"
            )
        expected_levels = _configuration_j_levels(occupations)
        for two_j, actual_count in actual_levels.items():
            expected_count = expected_levels.get(two_j, 0)
            if actual_count != expected_count:
                label = _configuration_label(occupations)
                raise ValueError(
                    f"GRASP CSF configuration {label} has {actual_count} "
                    f"CSFs at J={format_half(two_j)}{parity}; independent "
                    f"jj coupling requires {expected_count}"
                )
        full_j_manifold = actual_levels == expected_levels
        all_full = all_full and full_j_manifold
        configurations.append({
            "configuration": _configuration_label(occupations),
            "parity": parity,
            "present_j_levels": _level_rows(actual_levels),
            "complete_j_levels": _level_rows(expected_levels),
            "full_j_manifold_present": full_j_manifold,
        })

    return {
        "schema_version": GRASP_ANGULAR_CENSUS_SCHEMA,
        "path": str(document.source),
        "size_bytes": document.size_bytes,
        "sha256": document.sha256,
        "electron_count": document.electron_count,
        "csf_count": document.csf_count,
        "configuration_count": len(configurations),
        "configurations": configurations,
        "full_j_manifold_present": all_full,
        "checks": {
            "configuration_parities_match": True,
            "present_configuration_j_multiplicities_complete": True,
        },
        "valid": True,
        "scope": {
            "provides": (
                "Independent jj-coupled multiplicities for every relativistic "
                "occupation and J pair represented in the CSF file."
            ),
            "does_not_prove": [
                "that every requested configuration was generated",
                "that an intentionally restricted J range is complete",
                "LS term assignments for ASFs",
                "level energies or spin-orbit mixing",
            ],
        },
    }


def _configuration_j_levels(
    occupations: tuple[tuple[str, int], ...],
) -> dict[int, int]:
    distributions = []
    for label, electrons in occupations:
        two_j, capacity = _subshell_j(label)
        if not 1 <= electrons <= capacity:
            raise ValueError(
                f"GRASP occupation {label}({electrons}) exceeds capacity "
                f"{capacity}"
            )
        distributions.append(j_shell_m_distribution(two_j, electrons))
    return extract_j_levels(combine_j_distributions(distributions))


def _configuration_parity(
    occupations: tuple[tuple[str, int], ...],
) -> str:
    exponent = 0
    for label, electrons in occupations:
        match = _relativistic_subshell(label)
        exponent += _ORBITAL_LETTERS.index(match.group("orbital")) * electrons
    return "-" if exponent % 2 else "+"


def _subshell_j(label: str) -> tuple[int, int]:
    match = _relativistic_subshell(label)
    angular_momentum = _ORBITAL_LETTERS.index(match.group("orbital"))
    lower = match.group("lower") is not None
    if angular_momentum == 0 and lower:
        raise ValueError(f"invalid relativistic subshell {label!r}")
    two_j = (
        2 * angular_momentum - 1
        if lower
        else 2 * angular_momentum + 1
    )
    return two_j, two_j + 1


def _relativistic_subshell(label: str) -> re.Match[str]:
    match = _RELATIVISTIC_SUBSHELL_RE.fullmatch(label)
    if match is None:
        raise ValueError(f"invalid relativistic subshell {label!r}")
    return match


def _configuration_label(
    occupations: tuple[tuple[str, int], ...],
) -> str:
    return " ".join(
        f"{label}({electrons})"
        for label, electrons in occupations
    )


def _level_rows(levels: dict[int, int]) -> list[dict[str, object]]:
    return [
        {
            "two_j": two_j,
            "j": format_half(two_j),
            "csfs": count,
        }
        for two_j, count in sorted(levels.items())
    ]


__all__ = [
    "GRASP_ANGULAR_CENSUS_SCHEMA",
    "validate_csf_angular_census",
    "validate_grasp_csf_angular_census",
]

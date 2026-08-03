"""Collect QMCPACK particle sets from one bounded XML include review."""

from __future__ import annotations

from typing import Any, Mapping

from chemtools.programs.qmcpack.input import (
    parse_qmcpack_ion_geometries,
    parse_qmcpack_particle_sets,
)


def collect_particle_sets(
    parsed_input: Mapping[str, Any],
    include_review: Mapping[str, Any],
) -> list[dict[str, Any]]:
    particle_sets = list(parsed_input.get("particle_sets", []))
    for entry in include_review.get("entries", []):
        if (
            not isinstance(entry, Mapping)
            or entry.get("status") != "present"
            or entry.get("scan_status") == "too_large"
        ):
            continue
        path = entry.get("path")
        if not isinstance(path, str):
            continue
        try:
            particle_sets.extend(parse_qmcpack_particle_sets(path))
        except (OSError, ValueError):
            continue
    return particle_sets


def collect_ion_geometries(
    parsed_input: Mapping[str, Any],
    include_review: Mapping[str, Any],
) -> list[dict[str, Any]]:
    geometries = list(parsed_input.get("ion_geometries", []))
    for entry in include_review.get("entries", []):
        if (
            not isinstance(entry, Mapping)
            or entry.get("status") != "present"
            or entry.get("scan_status") == "too_large"
        ):
            continue
        path = entry.get("path")
        if not isinstance(path, str):
            continue
        try:
            included = parse_qmcpack_ion_geometries(path)
        except (OSError, ValueError):
            continue
        geometries.extend({**geometry, "source_path": path} for geometry in included)
    return geometries


def electron_particle_count(
    parsed_input: Mapping[str, Any],
    include_review: Mapping[str, Any],
) -> dict[str, Any]:
    targets = sorted({
        hamiltonian["target"]
        for hamiltonian in parsed_input.get("hamiltonians", [])
        if isinstance(hamiltonian, Mapping) and hamiltonian.get("target")
    })
    selected = [
        particle_set
        for particle_set in collect_particle_sets(parsed_input, include_review)
        if particle_set.get("name") in targets
    ]
    observed = {
        "hamiltonian_targets": targets,
        "matching_particle_sets": selected,
        "include_review_status": include_review.get("status"),
    }
    if len(targets) != 1 or len(selected) != 1:
        return {"status": "incomplete", **observed}
    group_sizes = [group.get("size") for group in selected[0].get("groups", [])]
    if (
        not group_sizes
        or any(not isinstance(size, str) or not size.isdigit() for size in group_sizes)
    ):
        return {"status": "incomplete", **observed}
    return {
        "status": "complete",
        "electron_count": sum(int(size) for size in group_sizes),
        **observed,
    }


def electron_spin_population(
    parsed_input: Mapping[str, Any],
    include_review: Mapping[str, Any],
) -> dict[str, Any]:
    electron_count = electron_particle_count(parsed_input, include_review)
    selected = electron_count["matching_particle_sets"]
    if electron_count["status"] != "complete" or len(selected) != 1:
        return {"status": "incomplete", **electron_count}
    groups = selected[0]["groups"]
    if len(groups) != 2 or {group.get("name") for group in groups} != {"u", "d"}:
        return {"status": "incomplete", **electron_count}
    populations = {group["name"]: int(group["size"]) for group in groups}
    return {
        "status": "complete",
        "up_electrons": populations["u"],
        "down_electrons": populations["d"],
        "spin_imbalance": populations["u"] - populations["d"],
        **electron_count,
    }


def non_electron_particle_sets(
    parsed_input: Mapping[str, Any],
    include_review: Mapping[str, Any],
) -> dict[str, Any]:
    targets = {
        hamiltonian["target"]
        for hamiltonian in parsed_input.get("hamiltonians", [])
        if isinstance(hamiltonian, Mapping) and hamiltonian.get("target")
    }
    particle_sets = [
        particle_set
        for particle_set in collect_particle_sets(parsed_input, include_review)
        if particle_set.get("name") not in targets
    ]
    sizes = [particle_set.get("size") for particle_set in particle_sets]
    observed = {
        "hamiltonian_targets": sorted(targets),
        "qmcpack_non_electron_particle_sets": particle_sets,
        "include_review_status": include_review.get("status"),
    }
    if (
        not particle_sets
        or any(not isinstance(size, str) or not size.isdigit() for size in sizes)
    ):
        return {"status": "incomplete", **observed}
    return {
        "status": "complete",
        "particle_count": sum(int(size) for size in sizes),
        **observed,
    }


__all__ = [
    "collect_ion_geometries",
    "collect_particle_sets",
    "electron_particle_count",
    "electron_spin_population",
    "non_electron_particle_sets",
]

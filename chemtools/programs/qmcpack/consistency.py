"""Compare declared QMCPACK QMC methods with primary-log section evidence."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Mapping

from chemtools.programs.qmcpack.scalar import scalar_filename_identity


_METHOD_TO_SECTION = {
    "linear": "QMCFixedSampleLinearOptimize",
    "vmc": "VMC",
    "dmc": "DMC",
}
_SUPPORTED_SECTIONS = frozenset(_METHOD_TO_SECTION.values())


class QmcpackRunConsistency:
    def compare_input_output(
        self,
        input_path: str,
        _output_path: str,
        parsed_input: Mapping[str, Any],
        parsed_output: Mapping[str, Any],
        artifact_paths: tuple[str, ...],
    ) -> Mapping[str, Any]:
        declared = [
            str(block.get("method", "")).casefold()
            for block in parsed_input.get("qmc_blocks") or []
            if isinstance(block, Mapping)
            and str(block.get("method", "")).casefold() in _METHOD_TO_SECTION
        ]
        expected_sections = _unique(_METHOD_TO_SECTION[method] for method in declared)
        observed_sections = _unique(
            str(task.get("method", ""))
            for task in parsed_output.get("tasks") or []
            if isinstance(task, Mapping)
            and task.get("has_usable_data")
            and str(task.get("method", "")) in _SUPPORTED_SECTIONS
        )
        if not expected_sections:
            return {
                "status": "not_checked",
                "input_path": input_path,
                "reason": "The input declares no supported QMC method block.",
            }
        if not observed_sections:
            return {
                "status": "not_checked",
                "input_path": input_path,
                "reason": "The output contains no supported QMC section.",
            }

        checks = [
            _method_check(section, expected_sections, observed_sections)
            for section in _unique((*expected_sections, *observed_sections))
        ]
        project_check = _project_check(parsed_input, parsed_output)
        if project_check is not None:
            checks.append(project_check)
        checks.extend(_particle_set_checks(parsed_input, parsed_output))
        checks.extend(_scalar_filename_project_checks(parsed_output, artifact_paths))
        summary = {
            status: sum(check["status"] == status for check in checks)
            for status in ("match", "mismatch", "not_checked")
        }
        return {
            "status": "mismatch" if summary["mismatch"] else "checked",
            "input_path": input_path,
            "summary": summary,
            "checks": checks,
        }


def _method_check(
    section: str,
    expected_sections: list[str],
    observed_sections: list[str],
) -> dict[str, Any]:
    expected = section in expected_sections
    observed = section in observed_sections
    return {
        "field": f"qmc_method:{section}",
        "status": "match" if expected and observed else "mismatch",
        "input": {
            "declared": expected,
            "supported_sections": expected_sections,
        },
        "output": {
            "observed": observed,
            "supported_sections": observed_sections,
            "basis": (
                "Repeated log sections are internal optimizer iterations; "
                "this check compares supported method presence only."
            ),
        },
    }


def _project_check(
    parsed_input: Mapping[str, Any],
    parsed_output: Mapping[str, Any],
) -> dict[str, Any] | None:
    project = parsed_input.get("project")
    if not isinstance(project, Mapping) or not isinstance(project.get("id"), str):
        return None
    output_project_id = (parsed_output.get("derived") or {}).get(
        "qmcpack:project_id"
    )
    if not isinstance(output_project_id, str):
        return None
    input_project_id = project["id"]
    return {
        "field": "project_id",
        "status": "match" if input_project_id == output_project_id else "mismatch",
        "input": {"project_id": input_project_id},
        "output": {
            "project_id": output_project_id,
            "line": (parsed_output.get("derived") or {}).get(
                "qmcpack:project_line"
            ),
            "basis": (
                "This compares QMCPACK's printed project label only; it does not "
                "establish input controls or output provenance."
            ),
        },
    }


def _particle_set_checks(
    parsed_input: Mapping[str, Any],
    parsed_output: Mapping[str, Any],
) -> list[dict[str, Any]]:
    input_counts = _input_particle_set_counts(parsed_input)
    duplicate_input_names = _duplicate_input_particle_set_names(parsed_input)
    runtime_sets = (parsed_output.get("derived") or {}).get(
        "qmcpack:runtime_particle_sets"
    )
    if not isinstance(runtime_sets, list):
        return []

    output_sets = _unique_runtime_particle_sets(runtime_sets)
    checks = [
        {
            "field": f"particle_set:{name}",
            "status": "not_checked",
            "input": {
                "particle_set_name": name,
                "basis": "The direct XML declares this particle-set name more than once.",
            },
            "reason": "The input particle-set declaration is ambiguous.",
        }
        for name in sorted(duplicate_input_names)
    ]
    for name, input_count in input_counts.items():
        input_evidence = {
            "particle_count": input_count,
            "basis": (
                "The count is from a direct XML particle set or its explicit "
                "group sizes."
            ),
        }
        output_set = output_sets.get(name)
        if output_set is None:
            checks.append({
                "field": f"particle_set:{name}",
                "status": "not_checked",
                "input": input_evidence,
                "reason": (
                    "The primary log has no unambiguous matching runtime particle set."
                ),
            })
            continue
        checks.append({
            "field": f"particle_set:{name}",
            "status": (
                "match"
                if input_count == output_set["particle_count"]
                else "mismatch"
            ),
            "input": input_evidence,
            "output": {
                "particle_count": output_set["particle_count"],
                "line": output_set.get("line"),
                "basis": "The count is from QMCPACK's runtime particle-pool summary.",
            },
        })
    duplicate_input_group_names = _duplicate_input_particle_set_group_names(
        parsed_input
    )
    for particle_set_name, input_groups in _input_particle_set_group_counts(
        parsed_input
    ).items():
        output_set = output_sets.get(particle_set_name)
        if output_set is None:
            continue
        checks.extend(_particle_group_checks(
            particle_set_name,
            input_groups,
            duplicate_input_group_names.get(particle_set_name, set()),
            output_set,
        ))
    return checks


def _particle_group_checks(
    particle_set_name: str,
    input_groups: Mapping[str, int],
    duplicate_input_group_names: set[str],
    output_set: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks = [
        {
            "field": f"particle_group:{particle_set_name}:{group_name}",
            "status": "not_checked",
            "input": {
                "particle_set_name": particle_set_name,
                "group_name": group_name,
                "basis": "The direct XML declares this group name more than once.",
            },
            "reason": "The input particle-group declaration is ambiguous.",
        }
        for group_name in sorted(duplicate_input_group_names)
    ]
    if output_set.get("group_particle_count_matches") is not True:
        for group_name, input_count in input_groups.items():
            checks.append({
                "field": f"particle_group:{particle_set_name}:{group_name}",
                "status": "not_checked",
                "input": _input_particle_group_evidence(input_count),
                "output": {
                    "particle_count": output_set["particle_count"],
                    "group_particle_count": output_set.get("group_particle_count"),
                    "line": output_set.get("line"),
                    "basis": (
                        "The named runtime group counts do not sum to QMCPACK's "
                        "printed particle-set total."
                    ),
                },
                "reason": "The runtime group counts are internally inconsistent.",
            })
        return checks

    output_groups = _runtime_particle_set_group_counts(output_set)
    duplicate_output_group_names = _duplicate_named_group_names(
        output_set.get("groups")
    )
    for group_name, input_count in input_groups.items():
        if group_name in duplicate_input_group_names:
            continue
        if group_name in duplicate_output_group_names:
            checks.append({
                "field": f"particle_group:{particle_set_name}:{group_name}",
                "status": "not_checked",
                "input": _input_particle_group_evidence(input_count),
                "output": {
                    "line": output_set.get("line"),
                    "basis": (
                        "The runtime particle-pool summary declares this group "
                        "name more than once."
                    ),
                },
                "reason": "The runtime particle-group declaration is ambiguous.",
            })
            continue
        output_count = output_groups.get(group_name)
        if output_count is None:
            checks.append({
                "field": f"particle_group:{particle_set_name}:{group_name}",
                "status": "not_checked",
                "input": _input_particle_group_evidence(input_count),
                "output": {
                    "line": output_set.get("line"),
                    "basis": "The runtime particle-pool summary has no matching named group.",
                },
                "reason": "The primary log has no matching runtime particle group.",
            })
            continue
        checks.append({
            "field": f"particle_group:{particle_set_name}:{group_name}",
            "status": "match" if input_count == output_count else "mismatch",
            "input": _input_particle_group_evidence(input_count),
            "output": {
                "particle_count": output_count,
                "line": output_set.get("line"),
                "basis": (
                    "The count is from QMCPACK's named runtime particle-pool "
                    "group."
                ),
            },
        })
    return checks


def _input_particle_group_evidence(input_count: int) -> dict[str, Any]:
    return {
        "particle_count": input_count,
        "basis": "The count is from a direct XML particle group.",
    }


def _scalar_filename_project_checks(
    parsed_output: Mapping[str, Any],
    artifact_paths: tuple[str, ...],
) -> list[dict[str, Any]]:
    output_project_id = (parsed_output.get("derived") or {}).get(
        "qmcpack:project_id"
    )
    output_project_line = (parsed_output.get("derived") or {}).get(
        "qmcpack:project_line"
    )
    checks = []
    for artifact_path in artifact_paths:
        path = Path(artifact_path)
        identity = scalar_filename_identity(path)
        if identity is None or not path.name.endswith(".scalar.dat"):
            continue
        if identity.get("status") != "recognized":
            checks.append({
                "field": f"scalar_filename_project:{path.name}",
                "status": "not_checked",
                "artifact": {
                    "path": str(path),
                    "filename": path.name,
                    "basis": identity["scope_limit"],
                },
                "reason": (
                    "The scalar filename has no recognized project label to compare."
                ),
            })
            continue
        scalar_project_id = identity["project_id"]
        artifact = {
            "path": str(path),
            "project_id": scalar_project_id,
            "series_index": identity["series_index"],
            "basis": identity["scope_limit"],
        }
        if not isinstance(output_project_id, str):
            checks.append({
                "field": f"scalar_filename_project:{path.name}",
                "status": "not_checked",
                "artifact": artifact,
                "reason": (
                    "The primary log has no unambiguous project label to compare."
                ),
            })
            continue
        checks.append({
            "field": f"scalar_filename_project:{path.name}",
            "status": (
                "match" if scalar_project_id == output_project_id else "mismatch"
            ),
            "artifact": artifact,
            "output": {
                "project_id": output_project_id,
                "line": output_project_line,
                "basis": (
                    "This compares the scalar filename project label with the "
                    "primary log label; it does not establish source-run or "
                    "QMC-block lineage."
                ),
            },
        })
    return checks


def _unique_runtime_particle_sets(
    runtime_sets: list[object],
) -> dict[str, Mapping[str, Any]]:
    particle_sets: dict[str, Mapping[str, Any]] = {}
    duplicate_names: set[str] = set()
    for particle_set in runtime_sets:
        if (
            not isinstance(particle_set, Mapping)
            or not isinstance(particle_set.get("name"), str)
            or not isinstance(particle_set.get("particle_count"), int)
        ):
            continue
        name = particle_set["name"]
        if name in particle_sets:
            duplicate_names.add(name)
        else:
            particle_sets[name] = particle_set
    return {
        name: particle_set
        for name, particle_set in particle_sets.items()
        if name not in duplicate_names
    }


def _input_particle_set_counts(parsed_input: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, particle_set in _unique_input_particle_sets(parsed_input).items():
        count = _explicit_particle_set_count(particle_set)
        if count is not None:
            counts[name] = count
    return counts


def _input_particle_set_group_counts(
    parsed_input: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    particle_set_groups: dict[str, dict[str, int]] = {}
    for name, particle_set in _unique_input_particle_sets(parsed_input).items():
        groups = _named_group_counts(particle_set.get("groups"), "size")
        if groups:
            particle_set_groups[name] = groups
    return particle_set_groups


def _duplicate_input_particle_set_group_names(
    parsed_input: Mapping[str, Any],
) -> dict[str, set[str]]:
    return {
        name: duplicate_names
        for name, particle_set in _unique_input_particle_sets(parsed_input).items()
        if (duplicate_names := _duplicate_named_group_names(particle_set.get("groups")))
    }


def _unique_input_particle_sets(
    parsed_input: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    particle_sets: dict[str, Mapping[str, Any]] = {}
    duplicate_names: set[str] = set()
    for particle_set in parsed_input.get("particle_sets") or []:
        if not isinstance(particle_set, Mapping) or not isinstance(
            particle_set.get("name"), str
        ):
            continue
        name = particle_set["name"]
        if name in particle_sets:
            duplicate_names.add(name)
        else:
            particle_sets[name] = particle_set
    return {
        name: particle_set
        for name, particle_set in particle_sets.items()
        if name not in duplicate_names
    }


def _duplicate_input_particle_set_names(
    parsed_input: Mapping[str, Any],
) -> set[str]:
    names: set[str] = set()
    duplicates: set[str] = set()
    for particle_set in parsed_input.get("particle_sets") or []:
        if not isinstance(particle_set, Mapping) or not isinstance(
            particle_set.get("name"), str
        ):
            continue
        name = particle_set["name"]
        if name in names:
            duplicates.add(name)
        names.add(name)
    return duplicates


def _runtime_particle_set_group_counts(particle_set: Mapping[str, Any]) -> dict[str, int]:
    if particle_set.get("group_particle_count_matches") is not True:
        return {}
    return _named_group_counts(particle_set.get("groups"), "count")


def _duplicate_named_group_names(groups: object) -> set[str]:
    if not isinstance(groups, list):
        return set()
    names: set[str] = set()
    duplicates: set[str] = set()
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("name"), str):
            continue
        name = group["name"]
        if name in names:
            duplicates.add(name)
        names.add(name)
    return duplicates


def _named_group_counts(groups: object, count_field: str) -> dict[str, int]:
    if not isinstance(groups, list):
        return {}
    counts: dict[str, int] = {}
    duplicates: set[str] = set()
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("name"), str):
            continue
        count = _nonnegative_integer(group.get(count_field))
        if count is None:
            continue
        name = group["name"]
        if name in counts:
            duplicates.add(name)
        else:
            counts[name] = count
    return {name: count for name, count in counts.items() if name not in duplicates}


def _explicit_particle_set_count(particle_set: Mapping[str, Any]) -> int | None:
    if (count := _nonnegative_integer(particle_set.get("size"))) is not None:
        return count
    groups = particle_set.get("groups")
    if not isinstance(groups, list) or not groups:
        return None
    group_counts = [
        _nonnegative_integer(group.get("size"))
        for group in groups
        if isinstance(group, Mapping)
    ]
    if len(group_counts) != len(groups) or any(count is None for count in group_counts):
        return None
    return sum(group_counts)


def _nonnegative_integer(value: object) -> int | None:
    if isinstance(value, int):
        return value if value >= 0 else None
    return int(value) if isinstance(value, str) and value.isdigit() else None


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


QMCPACK_RUN_CONSISTENCY = QmcpackRunConsistency()


__all__ = ["QMCPACK_RUN_CONSISTENCY"]

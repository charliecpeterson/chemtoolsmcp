"""Parse bounded completion and warning evidence from QMCPACK primary logs."""

from __future__ import annotations

import re
from math import isfinite
from pathlib import Path
from typing import Any


_BANNER_RE = re.compile(r"^\s*QMCPACK\s+(\d[^\s]*)", re.MULTILINE)
_TOTAL_TIME_RE = re.compile(
    r"^\s*Total Execution time\s*=\s*([^\s]+)\s+secs\s*$",
    re.MULTILINE,
)
_WARNING_RE = re.compile(r"^\s*QMCPACK WARNING\s+(.*\S)\s*$", re.MULTILINE)
_MINWALKERS_WARNING_RE = re.compile(
    r'^\s*QMCPACK WARNING\s+Smaller than the user specified threshold '
    r'"minwalkers"\s*=\s*([^\s]+)\s*$'
)
_INPUT_PARAMETER_CORRECTION_RE = re.compile(
    r'^\s*QMCPACK WARNING\s+Input parameter "([^"]+)" must be positive! '
    r'Set to ([^\s]+)\. User input value ([^\s]+)\s*$'
)
_EFFECTIVE_WEIGHT_RE = re.compile(
    r"^\s*Effective weight of all the samples measured by correlated sampling "
    r"is\s+([^\s]+)\s*$"
)
_COST_FUNCTION_INVALID_RE = re.compile(r"^\s*Cost Function is Invalid\.")
_REVERTING_PARAMETERS_RE = re.compile(
    r"^\s*(?:ERROR\s+)?Revert{1,2}ing to old Parameters\s*$"
)
_EFFECTIVE_WALKERS_TOO_SMALL_RE = re.compile(
    r"^\s*ERROR\s+CostFunction->\s+Number of Effective Walkers is too small\s+"
    r"([^\s]+)(?:\s|$)"
)
_FAILED_LINEAR_OPTIMIZATION_STEP_RE = re.compile(
    r"^\s*Failed Step\.\s+Largest LM parameter change:\s*([^\s]+)\s*$"
)
_GOOD_LINEAR_OPTIMIZATION_STEP_RE = re.compile(
    r"^\s*Good Step\.\s+Largest LM parameter change:\s*([^\s]+)\s*$"
)
_PROJECT_RE = re.compile(r"^\s*Project\s*=\s*(.+?\S)\s*$", re.MULTILINE)
_PARTICLE_SET_RE = re.compile(
    r"^\s*ParticleSet\s+'([^']+)'\s+contains\s+(\d+)\s+particles\s*:\s*(.*?)\s*$"
)
_PARTICLE_GROUP_RE = re.compile(r"([^\s()]+)\((\d+)\)")
_LEGACY_PARTICLE_SET_RE = re.compile(
    r"^\s*ParticleSet\s+(\S+)\s*:\s*((?:\d+\s*)+)$"
)
_SECTION_START_RE = re.compile(
    r"^\s*Start\s+(QMCFixedSampleLinearOptimize|VMC|DMC|VMCSingleOMP|DMCOMP)\s*$"
)
_SECTION_TIME_RE = re.compile(
    r"^\s*(QMCFixedSampleLinearOptimize|VMC|DMC)\s+Execution time\s*=\s*"
    r"([^\s]+)\s+secs\s*$"
)
_LEGACY_SECTION_TIME_RE = re.compile(
    r"^\s*QMC Execution time\s*=\s*([^\s]+)\s+secs\s*$"
)
_SUCCESS_MARKER = "QMCPACK execution completed successfully"
_SECTION_NAMES = {
    "VMCSingleOMP": "VMC",
    "DMCOMP": "DMC",
}


def parse_qmcpack_output(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    return parse_qmcpack_output_text(
        source.read_text(encoding="utf-8", errors="replace")
    )


def parse_qmcpack_output_text(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    banners = list(_BANNER_RE.finditer(text))
    if not banners:
        raise ValueError("QMCPACK output does not contain a QMCPACK version banner.")
    banner = banners[-1]
    run_start_line = _line_number(text, banner.start())
    run_lines = lines[run_start_line - 1:]
    run_text = "\n".join(run_lines)
    warnings = _warnings(run_lines, run_start_line)
    total_time_matches = [
        match
        for match in _TOTAL_TIME_RE.finditer(text)
        if _line_number(text, match.start()) >= run_start_line
    ]
    total_time = _float_or_none(total_time_matches[-1].group(1)) if total_time_matches else None
    total_time_line = (
        _line_number(text, total_time_matches[-1].start())
        if total_time_matches
        else None
    )
    success_lines = [
        number
        for number, line in enumerate(lines, start=1)
        if number >= run_start_line and line.strip() == _SUCCESS_MARKER
    ]
    success_line = success_lines[-1] if success_lines else None
    linear_optimization_steps = _linear_optimization_steps(run_lines, run_start_line)
    project_labels = _project_labels(run_text, run_start_line)
    runtime_particle_sets = _runtime_particle_sets(run_lines, run_start_line)
    input_parameter_corrections = _input_parameter_corrections(
        run_lines,
        run_start_line,
    )
    return {
        "program_version": banner.group(1),
        "line_count": len(lines),
        "completion": {
            "success_marker": success_line is not None,
            "line": success_line,
        },
        "total_execution_time_seconds": total_time,
        "last_total_execution_time_line": total_time_line,
        "project": _unambiguous_project(project_labels),
        **({"project_labels": project_labels} if len(project_labels) > 1 else {}),
        **({"last_run": {"start_line": run_start_line}} if len(banners) > 1 else {}),
        "sections": _sections(run_lines, run_start_line),
        "warnings": warnings,
        "minwalkers_threshold_warnings": _minwalkers_threshold_warnings(
            run_lines, run_start_line
        ),
        **(
            {"input_parameter_corrections": input_parameter_corrections}
            if input_parameter_corrections
            else {}
        ),
        "optimization_messages": _optimization_messages(
            run_lines,
            linear_optimization_steps.get("failed"),
            run_start_line,
        ),
        "linear_optimization_steps": linear_optimization_steps,
        **({"runtime_particle_sets": runtime_particle_sets} if runtime_particle_sets else {}),
    }


def _project_labels(text: str, line_offset: int = 1) -> list[dict[str, Any]]:
    return [
        {
            "id": match.group(1).strip(),
            "line": line_offset + text[:match.start()].count("\n"),
        }
        for match in _PROJECT_RE.finditer(text)
    ]


def _line_number(text: str, position: int) -> int:
    return text[:position].count("\n") + 1


def _unambiguous_project(
    project_labels: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not project_labels or len({project["id"] for project in project_labels}) != 1:
        return None
    return project_labels[-1]


def _runtime_particle_sets(
    lines: list[str], line_offset: int = 1
) -> list[dict[str, Any]]:
    particle_sets = []
    for number, line in enumerate(lines, start=line_offset):
        match = _PARTICLE_SET_RE.match(line)
        if match is not None:
            groups = [
                {"name": name, "count": int(count)}
                for name, count in _PARTICLE_GROUP_RE.findall(match.group(3))
            ]
            particle_count = int(match.group(2))
            group_particle_count = sum(group["count"] for group in groups)
            particle_sets.append({
                "name": match.group(1),
                "particle_count": particle_count,
                "groups": groups,
                "group_particle_count": group_particle_count if groups else None,
                "group_particle_count_matches": (
                    group_particle_count == particle_count if groups else None
                ),
                "line": number,
            })
            continue
        legacy_match = _LEGACY_PARTICLE_SET_RE.match(line)
        if legacy_match is None:
            continue
        offsets = [int(value) for value in legacy_match.group(2).split()]
        if len(offsets) < 2:
            continue
        valid_offsets = offsets[0] == 0 and all(
            end >= start for start, end in zip(offsets, offsets[1:])
        )
        groups = [
            {"name": None, "count": end - start}
            for start, end in zip(offsets, offsets[1:])
        ]
        group_particle_count = sum(group["count"] for group in groups)
        particle_sets.append({
            "name": legacy_match.group(1),
            "particle_count": offsets[-1],
            "groups": groups,
            "group_offsets": offsets,
            "group_particle_count": (
                group_particle_count if valid_offsets else None
            ),
            "group_particle_count_matches": (
                group_particle_count == offsets[-1] if valid_offsets else False
            ),
            "line": number,
        })
    return particle_sets


def _warnings(lines: list[str], line_offset: int = 1) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(lines, start=line_offset):
        match = _WARNING_RE.match(line)
        if match is None:
            continue
        message = match.group(1)
        existing = seen.get(message)
        if existing is not None:
            existing["occurrences"] += 1
            continue
        warning = {"message": message, "line": number, "occurrences": 1}
        seen[message] = warning
        warnings.append(warning)
    return warnings


def _minwalkers_threshold_warnings(
    lines: list[str], line_offset: int = 1
) -> list[dict[str, Any]]:
    warnings: dict[float, dict[str, Any]] = {}
    previous_effective_weight = None
    previous_effective_weight_line = None
    for number, line in enumerate(lines, start=line_offset):
        effective_weight = _EFFECTIVE_WEIGHT_RE.match(line)
        if effective_weight is not None:
            previous_effective_weight = _float_or_none(effective_weight.group(1))
            previous_effective_weight_line = number
            continue
        match = _MINWALKERS_WARNING_RE.match(line)
        if match is None:
            continue
        threshold = _float_or_none(match.group(1))
        if threshold is None:
            continue
        existing = warnings.get(threshold)
        if existing is not None:
            existing["occurrences"] += 1
            existing["last_line"] = number
        else:
            existing = {
                "threshold": threshold,
                "occurrences": 1,
                "first_line": number,
                "last_line": number,
                "minimum_immediately_preceding_effective_weight": None,
                "immediately_preceding_effective_weight_count": 0,
            }
            warnings[threshold] = existing
        if (
            previous_effective_weight is not None
            and previous_effective_weight_line == number - 1
        ):
            current_minimum = existing[
                "minimum_immediately_preceding_effective_weight"
            ]
            existing["minimum_immediately_preceding_effective_weight"] = (
                previous_effective_weight
                if current_minimum is None
                else min(current_minimum, previous_effective_weight)
            )
            existing["immediately_preceding_effective_weight_count"] += 1
    return list(warnings.values())


def _input_parameter_corrections(
    lines: list[str], line_offset: int = 1
) -> list[dict[str, Any]]:
    corrections: dict[tuple[str, float, float, str | None], dict[str, Any]] = {}
    current_section = None
    for number, line in enumerate(lines, start=line_offset):
        section_start = _SECTION_START_RE.match(line)
        if section_start is not None:
            current_section = _SECTION_NAMES.get(
                section_start.group(1),
                section_start.group(1),
            )
        match = _INPUT_PARAMETER_CORRECTION_RE.match(line)
        if match is None:
            continue
        corrected_value = _float_or_none(match.group(2))
        requested_value = _float_or_none(match.group(3))
        if corrected_value is None or requested_value is None:
            continue
        key = (match.group(1), corrected_value, requested_value, current_section)
        existing = corrections.get(key)
        if existing is not None:
            existing["occurrences"] += 1
            existing["last_line"] = number
            continue
        corrections[key] = {
            "parameter": match.group(1),
            "requested_value": requested_value,
            "corrected_value": corrected_value,
            "occurrences": 1,
            "first_line": number,
            "last_line": number,
            **({"section": current_section} if current_section is not None else {}),
        }
    return list(corrections.values())


def _optimization_messages(
    lines: list[str],
    failed_step: dict[str, Any] | None,
    line_offset: int = 1,
) -> list[dict[str, Any]]:
    patterns = (
        ("cost_function_invalid", _COST_FUNCTION_INVALID_RE),
        ("reverting_to_old_parameters", _REVERTING_PARAMETERS_RE),
    )
    messages: dict[str, dict[str, Any]] = {}
    current_section: dict[str, Any] | None = None

    def record_section(message: dict[str, Any]) -> None:
        if current_section is None:
            return
        sections = message.setdefault("sections", [])
        if not sections or sections[-1] != current_section:
            sections.append(current_section)

    for number, line in enumerate(lines, start=line_offset):
        section_start = _SECTION_START_RE.match(line)
        if section_start is not None:
            current_section = {
                "name": _SECTION_NAMES.get(
                    section_start.group(1),
                    section_start.group(1),
                ),
                "start_line": number,
            }
        for code, pattern in patterns:
            if pattern.match(line) is None:
                continue
            existing = messages.get(code)
            if existing is not None:
                existing["occurrences"] += 1
                existing["last_line"] = number
                record_section(existing)
                continue
            message = {
                "code": code,
                "message": line.strip(),
                "occurrences": 1,
                "first_line": number,
                "last_line": number,
            }
            record_section(message)
            messages[code] = message
        effective_walkers = _EFFECTIVE_WALKERS_TOO_SMALL_RE.match(line)
        if effective_walkers is not None:
            value = _float_or_none(effective_walkers.group(1))
            existing = messages.get("effective_walkers_too_small")
            if existing is not None:
                existing["occurrences"] += 1
                existing["last_line"] = number
                if value is not None:
                    current_minimum = existing[
                        "minimum_reported_effective_walkers"
                    ]
                    existing["minimum_reported_effective_walkers"] = (
                        value
                        if current_minimum is None
                        else min(current_minimum, value)
                    )
                record_section(existing)
            else:
                message = {
                    "code": "effective_walkers_too_small",
                    "message": line.strip(),
                    "occurrences": 1,
                    "first_line": number,
                    "last_line": number,
                    "minimum_reported_effective_walkers": value,
                }
                record_section(message)
                messages["effective_walkers_too_small"] = message
    if failed_step is not None:
        messages["linear_optimization_failed_step"] = {
            "code": "linear_optimization_failed_step",
            **failed_step,
        }
    return sorted(messages.values(), key=lambda message: message["first_line"])


def _linear_optimization_steps(
    lines: list[str], line_offset: int = 1
) -> dict[str, dict[str, Any]]:
    patterns = (
        ("failed", _FAILED_LINEAR_OPTIMIZATION_STEP_RE),
        ("good", _GOOD_LINEAR_OPTIMIZATION_STEP_RE),
    )
    steps: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(lines, start=line_offset):
        for outcome, pattern in patterns:
            match = pattern.match(line)
            if match is None:
                continue
            value = _float_or_none(match.group(1))
            existing = steps.get(outcome)
            if existing is None:
                steps[outcome] = {
                    "message": line.strip(),
                    "occurrences": 1,
                    "first_line": number,
                    "last_line": number,
                    "largest_reported_parameter_change": value,
                }
                continue
            existing["occurrences"] += 1
            existing["last_line"] = number
            if value is None:
                continue
            current_maximum = existing["largest_reported_parameter_change"]
            existing["largest_reported_parameter_change"] = (
                value if current_maximum is None else max(current_maximum, value)
            )
    return steps


def _sections(lines: list[str], line_offset: int = 1) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    optimizing = False
    for number, line in enumerate(lines, start=1):
        match = _SECTION_START_RE.match(line)
        if match is None:
            continue
        name = _SECTION_NAMES.get(match.group(1), match.group(1))
        if name == "QMCFixedSampleLinearOptimize":
            optimizing = True
            sections.append({"name": name, "start_line": number})
        elif not optimizing:
            sections.append({"name": name, "start_line": number})

        if name == "DMC":
            optimizing = False

    for index, section in enumerate(sections):
        next_start = (
            sections[index + 1]["start_line"]
            if index + 1 < len(sections)
            else len(lines) + 1
        )
        end_line = None
        execution_time = None
        for number in range(section["start_line"], next_start):
            match = _SECTION_TIME_RE.match(lines[number - 1])
            if match is not None and match.group(1) == section["name"]:
                end_line = number
                execution_time = _float_or_none(match.group(2))
                break
            if section["name"] in {
                "QMCFixedSampleLinearOptimize",
                "VMC",
                "DMC",
            }:
                legacy_match = _LEGACY_SECTION_TIME_RE.match(lines[number - 1])
                if legacy_match is not None:
                    end_line = number
                    execution_time = _float_or_none(legacy_match.group(1))
                    break
        section["end_line"] = end_line
        section["execution_time_seconds"] = execution_time
        section["start_line"] += line_offset - 1
        if section["end_line"] is not None:
            section["end_line"] += line_offset - 1
    return sections


def _float_or_none(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if isfinite(parsed) else None


__all__ = ["parse_qmcpack_output", "parse_qmcpack_output_text"]

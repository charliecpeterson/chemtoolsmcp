"""Parse the bounded QMCPACK XML input subset used by Chemtools review."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from chemtools.core.types import LintIssue


_SUPPORTED_ROOTS = frozenset({"simulation", "qmcsystem"})
_F_BLOCK_TMOVE_LADDERS = (
    (0.005, 0.0025, 0.00125, 0.000625),
    (0.005, 0.0025, 0.00125, 0.000625, 0.0003125),
)
_WARMUP_STEP_NAMES = ("warmupSteps", "warmupsteps")


def parse_qmcpack_input(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    return parse_qmcpack_text(source.read_text(encoding="utf-8", errors="replace"))


def parse_qmcpack_particle_sets(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    root = _parse_root(source.read_text(encoding="utf-8", errors="replace"))
    return _particle_sets(root)


def parse_qmcpack_ion_geometries(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    root = _parse_root(source.read_text(encoding="utf-8", errors="replace"))
    return _ion_geometries(root)


def parse_qmcpack_text(text: str) -> dict[str, Any]:
    root = _parse_root(text)
    root_name = _tag_name(root.tag)
    if root_name not in _SUPPORTED_ROOTS:
        raise ValueError(
            "QMCPACK input must use a <simulation> or <qmcsystem> root; "
            f"found <{root_name}>."
        )

    qmc_blocks = [
        _qmc_block(element)
        for element in root.iter()
        if _tag_name(element.tag) == "qmc"
    ]
    qmc_loop_maxes = _qmc_loop_maxes(root)
    references = _references(root)
    return {
        "format": "qmcpack-input/1",
        "root": root_name,
        "project": _project(root),
        "includes": [
            reference["href"]
            for reference in references
            if reference["kind"] == "include" and reference["href"]
        ],
        "references": references,
        "hdf5_sidecars": [
            reference["href"]
            for reference in references
            if reference["href"].lower().endswith(".h5")
        ],
        "particle_sets": _particle_sets(root),
        "ion_geometries": _ion_geometries(root),
        "hamiltonians": _hamiltonians(root),
        "qmc_blocks": qmc_blocks,
        "dmc_campaign": _dmc_campaign(qmc_blocks, qmc_loop_maxes),
    }


def lint_qmcpack_input(text: str) -> list[LintIssue]:
    return _lint_qmcpack_xml(text, allow_fragment=False)


def lint_qmcpack_fragment(text: str) -> list[LintIssue]:
    return _lint_qmcpack_xml(text, allow_fragment=True)


def _lint_qmcpack_xml(text: str, *, allow_fragment: bool) -> list[LintIssue]:
    try:
        root = _parse_root(text)
    except ValueError as error:
        return [_issue("error", str(error))]

    root_name = _tag_name(root.tag)
    if root_name not in _SUPPORTED_ROOTS and not allow_fragment:
        return [_issue(
            "error",
            "QMCPACK input must use a <simulation> or <qmcsystem> root; "
            f"found <{root_name}>.",
        )]

    issues: list[LintIssue] = []
    for element in root.iter():
        tag = _tag_name(element.tag)
        if tag in {"include", "override_variational_parameters"}:
            if not _attribute(element, "href"):
                issues.append(_issue(
                    "error",
                    f"<{tag}> requires a non-empty href attribute.",
                ))
        if tag == "pseudo":
            if not _attribute(element, "elementType"):
                issues.append(_issue(
                    "error",
                    "<pseudo> requires an elementType attribute.",
                ))
            if not _attribute(element, "href"):
                issues.append(_issue(
                    "error",
                    "<pseudo> requires a non-empty href attribute.",
                ))
        if tag == "qmc" and not _attribute(element, "method"):
            issues.append(_issue(
                "error",
                "<qmc> requires a method attribute.",
            ))
        if tag == "qmc":
            _qmc_parameter_issues(element, issues)
        if tag == "determinantset":
            _determinant_set_issues(element, issues)
        if tag == "particleset":
            _particle_set_issues(element, issues)

    _variational_parameter_override_issues(root, issues)

    if not allow_fragment and root_name == "simulation" and not any(
        _tag_name(element.tag) == "qmc" for element in root.iter()
    ):
        issues.append(_issue(
            "info",
            "This simulation input contains no <qmc> block; it may be an included fragment.",
        ))
    return issues


def _parse_root(text: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(text)
    except ElementTree.ParseError as error:
        line, column = error.position
        raise ValueError(
            f"QMCPACK XML is not well formed at line {line}, column {column}: {error}."
        ) from error


def _tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _attribute(element: ElementTree.Element, name: str) -> str | None:
    value = element.get(name)
    return value.strip() if value and value.strip() else None


def _project(root: ElementTree.Element) -> dict[str, str | None] | None:
    for element in root:
        if _tag_name(element.tag) == "project":
            return {
                "id": _attribute(element, "id"),
                "series": _attribute(element, "series"),
            }
    return None


def _references(root: ElementTree.Element) -> list[dict[str, str]]:
    references = []
    for element in root.iter():
        href = _attribute(element, "href")
        if href is None:
            continue
        tag = _tag_name(element.tag)
        reference = {
            "kind": "include" if tag == "include" else tag,
            "href": href,
        }
        if tag == "pseudo":
            element_type = _attribute(element, "elementType")
            if element_type is not None:
                reference["element"] = element_type
        references.append(reference)
    return references


def _particle_sets(root: ElementTree.Element) -> list[dict[str, Any]]:
    particle_sets = []
    for element in root.iter():
        if _tag_name(element.tag) != "particleset":
            continue
        groups = []
        for group in element:
            if _tag_name(group.tag) != "group":
                continue
            parameters = {
                name: value
                for child in group
                if _tag_name(child.tag) == "parameter"
                and (name := _attribute(child, "name")) is not None
                and (value := (child.text or "").strip())
            }
            groups.append({
                "name": _attribute(group, "name"),
                "size": _attribute(group, "size"),
                **({"parameters": parameters} if parameters else {}),
            })
        particle_sets.append({
            "name": _attribute(element, "name"),
            "size": _attribute(element, "size"),
            "groups": groups,
        })
    return particle_sets


def _ion_geometries(root: ElementTree.Element) -> list[dict[str, Any]]:
    cell = _simulation_cell(root)
    geometries = []
    for particle_set in root.iter():
        if _tag_name(particle_set.tag) != "particleset":
            continue
        groups = [
            group
            for group in particle_set
            if _tag_name(group.tag) == "group"
        ]
        atoms = []
        issue = None
        for group in groups:
            positions = _group_positions(group)
            if positions is None:
                continue
            size = _group_size(group, particle_set, len(groups))
            if size is None or len(positions) != size:
                issue = "Ion position records do not match their declared group size."
                break
            label = _attribute(group, "name")
            if label is None:
                issue = "An ion position record has no group name."
                break
            atoms.extend({"label": label, "coordinates": position} for position in positions)
        if not atoms:
            continue
        geometries.append({
            "particle_set": _attribute(particle_set, "name"),
            "cell": cell,
            "atoms": atoms,
            "status": "complete" if cell is not None and issue is None else "incomplete",
            **({"reason": issue} if issue else {}),
        })
    return geometries


def _simulation_cell(root: ElementTree.Element) -> dict[str, Any] | None:
    for cell in root.iter():
        if _tag_name(cell.tag) != "simulationcell":
            continue
        lattice = None
        boundary_conditions = None
        for parameter in cell:
            if _tag_name(parameter.tag) != "parameter":
                continue
            name = _attribute(parameter, "name")
            if name == "lattice":
                values = _numeric_values(parameter.text)
                if values is None or len(values) != 9:
                    return None
                lattice = {
                    "units": _attribute(parameter, "units"),
                    "vectors": [values[index:index + 3] for index in range(0, 9, 3)],
                }
            elif name == "bconds":
                boundary_conditions = (parameter.text or "").split()
        if lattice is not None:
            return {"lattice": lattice, "boundary_conditions": boundary_conditions}
    return None


def _group_positions(group: ElementTree.Element) -> list[list[float]] | None:
    for child in group:
        if (
            _tag_name(child.tag) == "attrib"
            and _attribute(child, "name") == "position"
        ):
            values = _numeric_values(child.text)
            if values is None or len(values) % 3:
                return []
            return [values[index:index + 3] for index in range(0, len(values), 3)]
    return None


def _group_size(
    group: ElementTree.Element,
    particle_set: ElementTree.Element,
    group_count: int,
) -> int | None:
    configured = _attribute(group, "size")
    if configured is None and group_count == 1:
        configured = _attribute(particle_set, "size")
    if configured is None or not configured.isdigit():
        return None
    size = int(configured)
    return size if size > 0 else None


def _numeric_values(text: str | None) -> list[float] | None:
    try:
        values = [float(value) for value in (text or "").split()]
    except ValueError:
        return None
    return values if all(isfinite(value) for value in values) else None


def _hamiltonians(root: ElementTree.Element) -> list[dict[str, Any]]:
    hamiltonians = []
    for element in root.iter():
        if _tag_name(element.tag) != "hamiltonian":
            continue
        pseudopotentials = []
        for pseudo in element.iter():
            if _tag_name(pseudo.tag) != "pseudo":
                continue
            pseudopotentials.append({
                "element": _attribute(pseudo, "elementType"),
                "href": _attribute(pseudo, "href"),
            })
        hamiltonians.append({
            "name": _attribute(element, "name"),
            "target": _attribute(element, "target"),
            "pseudopotentials": pseudopotentials,
        })
    return hamiltonians


def _qmc_block(element: ElementTree.Element) -> dict[str, Any]:
    parameters = {}
    costs = []
    for child in element:
        tag = _tag_name(child.tag)
        if tag == "cost":
            name = _attribute(child, "name")
            value = (child.text or "").strip()
            if name and value:
                costs.append({"name": name, "value": value})
            continue
        if tag != "parameter":
            continue
        name = _attribute(child, "name")
        value = (child.text or "").strip()
        if name and value:
            parameters[name] = value
    return {
        "method": _attribute(element, "method"),
        "move": _attribute(element, "move"),
        "checkpoint": _attribute(element, "checkpoint"),
        "parameters": parameters,
        "costs": costs,
    }


def _qmc_loop_maxes(root: ElementTree.Element) -> list[int | None]:
    loop_maxes: list[int | None] = []

    def visit(element: ElementTree.Element, enclosing_loop_max: int | None) -> None:
        tag = _tag_name(element.tag)
        loop_max = (
            _positive_int(_attribute(element, "max"))
            if tag == "loop"
            else enclosing_loop_max
        )
        if tag == "qmc":
            loop_maxes.append(loop_max)
        for child in element:
            visit(child, loop_max)

    visit(root, None)
    return loop_maxes


def _dmc_campaign(
    qmc_blocks: list[dict[str, Any]],
    qmc_loop_maxes: list[int | None],
) -> dict[str, Any] | None:
    dmc_blocks = [
        _dmc_block(index, block)
        for index, block in enumerate(qmc_blocks)
        if block["method"] == "dmc"
    ]
    if not dmc_blocks:
        return None
    tmove_blocks = [block for block in dmc_blocks if block["nonlocalmoves"] is True]
    no_tmove_blocks = [
        block for block in dmc_blocks if block["nonlocalmoves"] is False
    ]
    return {
        "dmc_blocks": dmc_blocks,
        "production_protocol": _production_protocol(qmc_blocks, qmc_loop_maxes),
        "tmove_ladder": _tmove_ladder(tmove_blocks),
        "no_tmove_control": _no_tmove_control(
            no_tmove_blocks,
            tmove_blocks,
            qmc_blocks,
        ),
        "declared_target_walkers": sorted({
            block["target_walkers"]
            for block in dmc_blocks
            if block["target_walkers"] is not None
        }),
    }


def _production_protocol(
    qmc_blocks: list[dict[str, Any]],
    qmc_loop_maxes: list[int | None],
) -> dict[str, Any]:
    methods = [block["method"] for block in qmc_blocks]
    indices = {
        method: [
            index for index, observed in enumerate(methods) if observed == method
        ]
        for method in ("linear", "vmc", "dmc")
    }
    if any(not indices[method] for method in indices):
        return {
            "status": "not_assessed",
            "reason": (
                "the reference production order needs linear optimization, VMC, "
                "and DMC blocks"
            ),
            "observed_methods": methods,
        }
    linear_before_vmc = max(indices["linear"]) < min(indices["vmc"])
    vmc_before_dmc = max(indices["vmc"]) < min(indices["dmc"])
    return {
        "status": "assessed",
        "linear_qmc_block_indices": indices["linear"],
        "vmc_qmc_block_indices": indices["vmc"],
        "dmc_qmc_block_indices": indices["dmc"],
        "linear_before_vmc": linear_before_vmc,
        "vmc_before_dmc": vmc_before_dmc,
        "linear_optimization_loop": _linear_optimization_loop(
            indices["linear"],
            qmc_loop_maxes,
        ),
        "linear_optimization_settings": _linear_optimization_settings(
            indices["linear"],
            qmc_blocks,
        ),
        "matches_reference_order": linear_before_vmc and vmc_before_dmc,
    }


def _linear_optimization_loop(
    linear_indices: list[int],
    qmc_loop_maxes: list[int | None],
) -> dict[str, Any]:
    loop_maxes = [qmc_loop_maxes[index] for index in linear_indices]
    if any(loop_max is None for loop_max in loop_maxes):
        return {
            "status": "not_assessed",
            "reason": (
                "each linear QMC block needs an enclosing <loop> with a positive "
                "max attribute"
            ),
        }
    return {
        "status": "assessed",
        "loop_maxes": loop_maxes,
        "all_loop_maxes_in_reference_range": all(
            6 <= loop_max <= 8 for loop_max in loop_maxes
        ),
    }


def _linear_optimization_settings(
    linear_indices: list[int],
    qmc_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    settings = [
        {
            "qmc_block_index": index,
            "min_method": qmc_blocks[index]["parameters"].get("MinMethod"),
            "energy_cost": _cost_value(qmc_blocks[index], "energy"),
            "unreweighted_variance_cost": _cost_value(
                qmc_blocks[index],
                "unreweightedvariance",
            ),
        }
        for index in linear_indices
    ]
    if any(
        setting["min_method"] is None
        or setting["energy_cost"] is None
        or setting["unreweighted_variance_cost"] is None
        for setting in settings
    ):
        return {
            "status": "not_assessed",
            "reason": (
                "each linear QMC block needs MinMethod plus numeric energy and "
                "unreweightedvariance costs"
            ),
        }
    return {
        "status": "assessed",
        "settings": settings,
        "all_settings_match_reference": all(
            setting["min_method"] == "OneShiftOnly"
            and setting["energy_cost"] == 0.1
            and setting["unreweighted_variance_cost"] == 0.9
            for setting in settings
        ),
    }


def _cost_value(block: dict[str, Any], name: str) -> float | None:
    for cost in block["costs"]:
        if cost["name"] == name:
            return _finite_float(cost["value"])
    return None


def _dmc_block(index: int, block: dict[str, Any]) -> dict[str, Any]:
    parameters = block["parameters"]
    return {
        "qmc_block_index": index,
        "timestep": _positive_float(parameters.get("timestep")),
        "blocks": _positive_int(parameters.get("blocks")),
        "target_walkers": _positive_int(
            parameters.get("targetWalkers") or parameters.get("total_walkers")
        ),
        "nonlocalmoves": _boolean_parameter(parameters.get("nonlocalmoves")),
    }


def _tmove_ladder(tmove_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    if len(tmove_blocks) < 2:
        return {
            "status": "not_assessed",
            "reason": "fewer than two DMC blocks explicitly set nonlocalmoves=yes",
            "blocks": tmove_blocks,
        }
    if any(
        block["timestep"] is None or block["blocks"] is None
        for block in tmove_blocks
    ):
        return {
            "status": "not_assessed",
            "reason": "each T-move DMC block needs positive timestep and blocks values",
            "blocks": tmove_blocks,
        }
    ordered = sorted(tmove_blocks, key=lambda block: block["timestep"], reverse=True)
    timesteps = [block["timestep"] for block in ordered]
    block_counts = [block["blocks"] for block in ordered]
    return {
        "status": "assessed",
        "blocks": ordered,
        "timesteps_strictly_decrease": all(
            earlier > later for earlier, later in zip(timesteps, timesteps[1:])
        ),
        "block_counts_strictly_increase": all(
            earlier < later
            for earlier, later in zip(block_counts, block_counts[1:])
        ),
        "matches_fblock_reference_timestep_ladder": (
            tuple(timesteps) in _F_BLOCK_TMOVE_LADDERS
        ),
    }


def _no_tmove_control(
    no_tmove_blocks: list[dict[str, Any]],
    tmove_blocks: list[dict[str, Any]],
    qmc_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    tmove_timesteps = {
        block["timestep"] for block in tmove_blocks if block["timestep"] is not None
    }
    matching_timesteps = {
        block["timestep"]
        for block in no_tmove_blocks
        if block["timestep"] in tmove_timesteps
    }
    return {
        "blocks": no_tmove_blocks,
        "matching_tmove_timestep": (
            bool(matching_timesteps)
            if no_tmove_blocks and tmove_timesteps
            else None
        ),
        "middle_timestep_control": _middle_timestep_control(
            no_tmove_blocks,
            tmove_timesteps,
            matching_timesteps,
        ),
        "matching_tmove_settings": _matching_tmove_settings(
            no_tmove_blocks,
            tmove_blocks,
            qmc_blocks,
        ),
    }


def _matching_tmove_settings(
    no_tmove_blocks: list[dict[str, Any]],
    tmove_blocks: list[dict[str, Any]],
    qmc_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    comparisons = []
    for control in no_tmove_blocks:
        matches = [
            block
            for block in tmove_blocks
            if control["timestep"] is not None
            and block["timestep"] == control["timestep"]
        ]
        if not matches:
            continue
        fields = {
            "block_count_match": "blocks",
            "steps_match": "steps",
            "warmup_steps_match": "warmup_steps",
            "target_walkers_match": "target_walkers",
            "move_match": "move",
            "checkpoint_match": "checkpoint",
        }
        comparison = {
            name: _matching_field(
                _dmc_setting(control, field, qmc_blocks),
                [_dmc_setting(block, field, qmc_blocks) for block in matches],
            )
            for name, field in fields.items()
        }
        values = list(comparison.values())
        comparisons.append({
            "no_tmove_qmc_block_index": control["qmc_block_index"],
            "tmove_qmc_block_indices": [
                block["qmc_block_index"] for block in matches
            ],
            **comparison,
            "all_declared_settings_match": (
                all(values) if all(value is not None for value in values) else None
            ),
        })
    if not comparisons:
        return {
            "status": "not_assessed",
            "reason": (
                "a no-T-move control with a matching T-move timestep is required"
            ),
        }
    return {
        "status": "assessed",
        "comparisons": comparisons,
    }


def _dmc_setting(
    block: dict[str, Any],
    field: str,
    qmc_blocks: list[dict[str, Any]],
) -> int | str | None:
    if field in {"blocks", "target_walkers"}:
        return block[field]
    qmc_block = qmc_blocks[block["qmc_block_index"]]
    if field == "steps":
        return _positive_int(qmc_block["parameters"].get("steps"))
    if field == "warmup_steps":
        return _nonnegative_parameter_alias(
            qmc_block["parameters"],
            _WARMUP_STEP_NAMES,
        )
    return qmc_block[field]


def _matching_field(value: object, matches: list[object]) -> bool | None:
    if value is None or any(match is None for match in matches):
        return None
    return all(value == match for match in matches)


def _middle_timestep_control(
    no_tmove_blocks: list[dict[str, Any]],
    tmove_timesteps: set[float],
    matching_timesteps: set[float],
) -> dict[str, Any]:
    if not no_tmove_blocks or not tmove_timesteps:
        return {
            "status": "not_assessed",
            "reason": "a no-T-move control and T-move time steps are both required",
        }
    ordered_timesteps = sorted(tmove_timesteps, reverse=True)
    if len(ordered_timesteps) < 3:
        return {
            "status": "not_assessed",
            "reason": (
                "at least three distinct T-move time steps are needed to identify "
                "an interior control point"
            ),
        }
    interior_timesteps = set(ordered_timesteps[1:-1])
    return {
        "status": "assessed",
        "matching_tmove_timesteps": sorted(matching_timesteps, reverse=True),
        "interior_tmove_timesteps": ordered_timesteps[1:-1],
        "control_count_matches_reference": len(no_tmove_blocks) == 1,
        "all_controls_match_interior_tmove_timestep": all(
            block["timestep"] in interior_timesteps for block in no_tmove_blocks
        ),
    }


def _positive_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if isfinite(parsed) else None


def _positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _nonnegative_parameter_alias(
    parameters: dict[str, str],
    names: tuple[str, ...],
) -> int | None:
    values = [
        _nonnegative_int(parameters[name])
        for name in names
        if name in parameters
    ]
    if not values or any(value is None for value in values):
        return None
    return values[0] if all(value == values[0] for value in values) else None


def _nonnegative_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _boolean_parameter(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.casefold()
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    return None


def _particle_set_issues(
    element: ElementTree.Element,
    issues: list[LintIssue],
) -> None:
    for group in element:
        if _tag_name(group.tag) != "group":
            continue
        size = _attribute(group, "size")
        if size is None:
            continue
        try:
            positive = int(size) > 0
        except ValueError:
            positive = False
        if not positive:
            issues.append(_issue(
                "error",
                "<group> size must be a positive integer when it is present.",
            ))


def _qmc_parameter_issues(
    element: ElementTree.Element,
    issues: list[LintIssue],
) -> None:
    walker_counts: dict[str, int | None] = {}
    warmup_steps: dict[str, int | None] = {}
    for child in element:
        if _tag_name(child.tag) != "parameter":
            continue
        name = _attribute(child, "name")
        if name == "nonlocalpp":
            issues.append(_issue(
                "warning",
                '<qmc> parameter "nonlocalpp" is deprecated and does not affect '
                "QMCPACK execution; remove it.",
            ))
            continue
        if name == "nonlocalmoves":
            value = (child.text or "").strip()
            if _boolean_parameter(value) is None:
                issues.append(_issue(
                    "warning",
                    '<qmc> parameter "nonlocalmoves" is not a recognized boolean; '
                    "use yes/no, true/false, or 1/0.",
                ))
            continue
        if name == "timestep":
            value = (child.text or "").strip()
            parsed = _positive_float(value)
            if parsed is not None:
                continue
            if _finite_float(value) is None:
                issues.append(_issue(
                    "warning",
                    '<qmc> parameter "timestep" is non-numeric and cannot be validated.',
                ))
            else:
                issues.append(_issue(
                    "error",
                    '<qmc> parameter "timestep" must be a positive finite number when it is present.',
                ))
            continue
        if name not in {
            "blocks",
            "steps",
            "targetWalkers",
            "total_walkers",
            *_WARMUP_STEP_NAMES,
        }:
            continue
        value = (child.text or "").strip()
        try:
            parsed = int(value)
        except ValueError:
            issues.append(_issue(
                "warning",
                f'<qmc> parameter "{name}" is non-numeric and cannot be validated.',
            ))
            continue
        if name in _WARMUP_STEP_NAMES:
            if parsed < 0:
                issues.append(_issue(
                    "error",
                    f'<qmc> parameter "{name}" must be a nonnegative integer when it is present.',
                ))
                continue
            warmup_steps[name] = parsed
            continue
        if parsed <= 0:
            issues.append(_issue(
                "error",
                f'<qmc> parameter "{name}" must be a positive integer when it is present.',
            ))
            continue
        if name in {"targetWalkers", "total_walkers"}:
            walker_counts[name] = parsed
    if (
        len(walker_counts) == 2
        and walker_counts["targetWalkers"] != walker_counts["total_walkers"]
    ):
        issues.append(_issue(
            "error",
            '<qmc> parameters "targetWalkers" and "total_walkers" disagree; '
            "declare one target or matching values.",
        ))
    if (
        len(warmup_steps) == 2
        and warmup_steps["warmupSteps"] != warmup_steps["warmupsteps"]
    ):
        issues.append(_issue(
            "error",
            '<qmc> parameters "warmupSteps" and "warmupsteps" disagree; '
            "declare one spelling or matching values.",
        ))


def _determinant_set_issues(
    element: ElementTree.Element,
    issues: list[LintIssue],
) -> None:
    if _attribute(element, "twistnum") and not _attribute(element, "twist"):
        issues.append(_issue(
            "warning",
            "<determinantset> has twistnum but no twist attribute; specify twist "
            "to avoid an ambiguous selection.",
        ))
    if any(_tag_name(child.tag) == "slaterdeterminant" for child in element):
        issues.append(_issue(
            "warning",
            "<determinantset> contains a legacy inline <slaterdeterminant>; "
            "move SPO setup to a top-level <sposet_collection>.",
        ))


def _variational_parameter_override_issues(
    root: ElementTree.Element,
    issues: list[LintIssue],
) -> None:
    has_override = any(
        _tag_name(element.tag) == "override_variational_parameters"
        and _attribute(element, "href")
        for element in root.iter()
    )
    has_inline_coefficients = any(
        _tag_name(element.tag) == "coefficients"
        and (element.text or "").strip()
        for element in root.iter()
    )
    if has_override and has_inline_coefficients:
        issues.append(_issue(
            "warning",
            "An override_variational_parameters sidecar is present with inline "
            "<coefficients> values. The sidecar is authoritative; the inline "
            "values may be stale display values.",
        ))


def _issue(level: str, message: str) -> LintIssue:
    return {
        "level": level,
        "message": message,
        "line": None,
        "suggested_fix": None,
    }


__all__ = [
    "lint_qmcpack_fragment",
    "lint_qmcpack_input",
    "parse_qmcpack_input",
    "parse_qmcpack_ion_geometries",
    "parse_qmcpack_particle_sets",
    "parse_qmcpack_text",
]

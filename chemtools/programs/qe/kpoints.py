"""Summarize pw.x k-point sampling and propose bounded mesh refinements.

The candidate series is a Chemtools heuristic. Only executed calculations and
user-selected tolerances can establish k-point convergence.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from chemtools.core.types import LintIssue


_SYMMETRY_FLAGS = (
    "nosym",
    "nosym_evc",
    "noinv",
    "no_t_rev",
    "force_symmorphic",
)
_PATH_OPTIONS = frozenset({"crystal_b", "tpiba_b"})


def inspect_k_points(parsed_input: Mapping[str, Any]) -> dict[str, Any]:
    """Return sampling, symmetry, and convergence-planning evidence."""
    namelists = _mapping(parsed_input.get("namelists"))
    system = _mapping(namelists.get("system"))
    k_points = _mapping(parsed_input.get("k_points"))
    option = str(k_points.get("option") or "unknown").lower()
    flags = {
        name: system.get(name, False)
        for name in _SYMMETRY_FLAGS
    }
    sampling = _sampling_summary(k_points, option)
    return {
        "schema_version": "chemtools.qe-k-point-review/1",
        "sampling": sampling,
        "symmetry": {
            "flags": flags,
            "effects": _symmetry_effects(flags),
            "irreducible_k_point_count": None,
            "count_requires_pw_output": True,
        },
        "occupations": system.get("occupations"),
        "convergence_plan": _convergence_plan(
            calculation=str(parsed_input.get("calculation") or "scf").lower(),
            sampling=sampling,
        ),
        "source": {
            "k_points_line": _card_line(parsed_input, "k_points"),
            "system_assignment_lines": dict(
                _system_assignment_lines(parsed_input)
            ),
        },
        "scope": (
            "The requested mesh and candidate refinements are known. Actual "
            "irreducible counts and convergence require pw.x outputs."
        ),
    }


def k_point_issues(review: Mapping[str, Any]) -> list[LintIssue]:
    """Return documented occupation and symmetry compatibility findings."""
    sampling = _mapping(review.get("sampling"))
    symmetry = _mapping(review.get("symmetry"))
    flags = _mapping(symmetry.get("flags"))
    issues: list[LintIssue] = []

    for name in _SYMMETRY_FLAGS:
        if not isinstance(flags.get(name), bool):
            issues.append(_issue(
                "error",
                f"{name} must be a Fortran logical value.",
                line=_system_line(review, name),
                suggested_fix=f"Set {name}=.true. or {name}=.false.",
            ))

    option = sampling.get("option")
    occupations = review.get("occupations")
    if occupations == "tetrahedra" and option != "automatic":
        issues.append(_issue(
            "error",
            "occupations='tetrahedra' requires K_POINTS automatic.",
            line=_k_points_line(review),
            suggested_fix=(
                "Use an automatically generated uniform grid, or choose an "
                "occupation method compatible with the intended sampling."
            ),
        ))
    elif occupations == "tetrahedra" and any(sampling.get("shift") or []):
        issues.append(_issue(
            "warning",
            (
                "The tetrahedron calculation uses an offset automatic grid; "
                "QE warns that some offset grids lack the full crystal symmetry."
            ),
            line=_k_points_line(review),
            suggested_fix=(
                "Confirm the generated grid has the full crystal symmetry in "
                "the pw.x output, or use a compatible unshifted grid."
            ),
        ))
    return issues


def _sampling_summary(
    k_points: Mapping[str, Any],
    option: str,
) -> dict[str, Any]:
    if option == "automatic":
        mesh = _integer_triplet(k_points.get("grid"))
        shift = _integer_triplet(k_points.get("shift"))
        if mesh and any(value <= 0 for value in mesh):
            mesh = None
        if shift and any(value not in {0, 1} for value in shift):
            shift = None
        return {
            "mode": "mesh",
            "option": option,
            "mesh": list(mesh) if mesh else None,
            "shift": list(shift) if shift else None,
            "requested_full_grid_points": math.prod(mesh) if mesh else None,
        }
    if option == "gamma":
        return {
            "mode": "gamma",
            "option": option,
            "mesh": None,
            "shift": None,
            "requested_full_grid_points": 1,
        }
    return {
        "mode": "path" if option in _PATH_OPTIONS else "explicit",
        "option": option,
        "declared_count": k_points.get("declared_count"),
        "parsed_count": len(k_points.get("points", [])),
        "points": k_points.get("points", []),
    }


def _convergence_plan(
    *,
    calculation: str,
    sampling: Mapping[str, Any],
) -> dict[str, Any]:
    observables = ["total_energy_per_atom"]
    if calculation in {"relax", "vc-relax"}:
        observables.append("maximum_force")
    if calculation == "vc-relax":
        observables.append("stress")

    mesh = _integer_triplet(sampling.get("mesh"))
    shift = _integer_triplet(sampling.get("shift"))
    if sampling.get("mode") == "mesh" and mesh and shift:
        candidates = _mesh_candidates(mesh, shift)
        return {
            "status": "candidate_series",
            "convergence_established": False,
            "candidate_meshes": candidates,
            "compare": observables,
            "controlled_inputs": [
                "structure",
                "pseudopotentials",
                "ecutwfc",
                "ecutrho",
                "occupations",
                "smearing",
                "degauss",
            ],
            "stopping_rule": "User-selected observable tolerances are required.",
            "generation_basis": (
                "Chemtools heuristic: preserve shift and mesh parity, target "
                "25% and 50% refinement, keep stages distinct, and retain "
                "axes currently at 1."
            ),
            "assumptions": [
                (
                    "An axis currently sampled with one point is treated as "
                    "intentionally unrefined; confirm the system dimensionality."
                ),
                (
                    "The actual irreducible k-point counts and crystal-symmetry "
                    "compatibility must be read from each pw.x output."
                ),
            ],
        }

    if sampling.get("mode") == "path":
        status = "not_applicable_to_band_path"
        reason = "Band paths do not test Brillouin-zone integration convergence."
    elif sampling.get("mode") == "gamma":
        status = "sampling_design_required"
        reason = (
            "The input gives no dimensionality evidence for choosing which "
            "automatic-mesh axes to refine."
        )
    else:
        status = "sampling_design_required"
        reason = "An explicit point list cannot be refined safely from point count alone."
    return {
        "status": status,
        "convergence_established": False,
        "candidate_meshes": [],
        "compare": observables,
        "reason": reason,
        "stopping_rule": "User-selected observable tolerances are required.",
    }


def _mesh_candidates(
    mesh: tuple[int, int, int],
    shift: tuple[int, int, int],
) -> list[dict[str, Any]]:
    meshes = [mesh]
    previous = mesh
    for factor in (1.25, 1.5):
        refined = tuple(
            _refined_axis(current, factor, prior)
            for current, prior in zip(mesh, previous)
        )
        meshes.append(refined)
        previous = refined
    return [
        {
            "stage": stage,
            "mesh": list(candidate),
            "shift": list(shift),
            "requested_full_grid_points": math.prod(candidate),
        }
        for stage, candidate in zip(("current", "refine_1", "refine_2"), meshes)
    ]


def _refined_axis(current: int, factor: float, previous: int) -> int:
    if current == 1:
        return 1
    target = math.ceil(current * factor)
    if (target - current) % 2:
        target += 1
    if target <= previous:
        target = previous + 2
    return target


def _symmetry_effects(flags: Mapping[str, Any]) -> list[str]:
    effects: list[str] = []
    if flags.get("nosym") is True:
        effects.append("Uniform grids expand over the full Brillouin zone.")
    if flags.get("nosym_evc") is True:
        effects.append(
            "K-points are completed to the full symmetry of the Bravais lattice."
        )
    if flags.get("noinv") is True:
        effects.append("k and -k are not treated as equivalent during generation.")
    if flags.get("no_t_rev") is True:
        effects.append("Rotation plus time-reversal magnetic symmetries are disabled.")
    if flags.get("force_symmorphic") is True:
        effects.append("Symmetry operations with fractional translations are disabled.")
    return effects


def _integer_triplet(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return None
    return value[0], value[1], value[2]


def _system_assignment_lines(
    parsed_input: Mapping[str, Any],
) -> Mapping[str, Any]:
    assignment_lines = _mapping(parsed_input.get("assignment_lines"))
    return _mapping(assignment_lines.get("system"))


def _card_line(parsed_input: Mapping[str, Any], name: str) -> int | None:
    card_lines = _mapping(parsed_input.get("card_lines"))
    line = card_lines.get(name)
    return line if isinstance(line, int) else None


def _k_points_line(review: Mapping[str, Any]) -> int | None:
    source = _mapping(review.get("source"))
    line = source.get("k_points_line")
    return line if isinstance(line, int) else None


def _system_line(review: Mapping[str, Any], name: str) -> int | None:
    source = _mapping(review.get("source"))
    lines = _mapping(source.get("system_assignment_lines"))
    line = lines.get(name)
    return line if isinstance(line, int) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _issue(
    level: str,
    message: str,
    *,
    line: int | None,
    suggested_fix: str | None = None,
) -> LintIssue:
    return {
        "level": level,  # type: ignore[typeddict-item]
        "message": message,
        "line": line,
        "suggested_fix": suggested_fix,
    }


__all__ = ["inspect_k_points", "k_point_issues"]

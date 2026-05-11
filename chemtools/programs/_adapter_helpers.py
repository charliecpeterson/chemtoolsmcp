"""Shared adapter helpers for Program plugin Parser implementations.

Every program's parse_tasks function returns a `generic_tasks` list with a
common cross-program shape. This module turns that into the `TaskSummary`
TypedDicts from `chemtools.core.types`, computes pre-derived scalars, and
auto-picks a primary task. Per-program Parser plugins (`_plugin_parser.py`
in each program package) reuse these helpers.

The generic_tasks shape is:

    {
        "program":       str,                # "nwchem" | "molpro" | "molcas"
        "kind":          str,                # operation label
        "label":         str,                # human-friendly name
        "energy_hartree": float | None,
        "line_start":    int | None,         # 1-based, optional
        "line_end":      int | None,         # 1-based, optional
        "extra":         dict[str, Any],     # program-specific fields
    }

Plus an optional per-program `raw` task dict (NWChem provides one) that may
contain `boundary` (byte offsets), `outcome`, and other program-specific
metadata. The helper accepts either; missing fields default to safe values.
"""

from __future__ import annotations
from typing import Any

from chemtools.core.types import TaskSummary


# Auto-pick priority by operation kind. Higher = preferred when picking the
# "main" task from a multi-task output file.
SELECTION_PRIORITY: dict[str, int] = {
    "optimize":  3,
    "saddle":    3,
    "frequency": 2,
    "gradient":  1,
    "energy":    1,
    "property":  1,
    "raman":     1,
    "dynamics":  1,
    "unknown":   0,
}

# Normalize legacy / synonym kind labels to the TaskKind enum from core.types.
_KIND_ALIASES: dict[str, str] = {
    "single_point": "energy",
    "optimization": "optimize",
}


def normalize_kind(raw_kind: str | None) -> str:
    """Map a parser's kind label to a canonical TaskKind value."""
    if raw_kind is None:
        return "unknown"
    return _KIND_ALIASES.get(raw_kind, raw_kind)


def to_task_summary(idx: int, generic: dict[str, Any], raw: dict[str, Any] | None = None) -> TaskSummary:
    """Convert one generic_tasks entry (plus optional program-raw entry) to TaskSummary.

    Line-range resolution order:
      1. generic.line_start / generic.line_end  (Molpro pattern)
      2. raw.boundary.line_start / .line_end    (future-proof)
      3. raw.boundary.start_byte / .end_byte    (NWChem pattern — byte offsets used as placeholders)
      4. (0, 0) fallback
    """
    raw = raw or {}
    kind = normalize_kind(generic.get("kind") or raw.get("kind"))
    extra = generic.get("extra") or {}

    outcome = extra.get("outcome") or raw.get("outcome") or "unknown"
    if outcome not in {"success", "failed", "incomplete", "unknown"}:
        outcome = "unknown"

    # Line range — prefer 1-based line numbers from generic; fall back to byte boundaries.
    line_start = generic.get("line_start")
    line_end = generic.get("line_end")
    if line_start is None or line_end is None:
        boundary = raw.get("boundary") or {}
        line_start = boundary.get("line_start") or boundary.get("start_byte") or 0
        line_end = boundary.get("line_end") or boundary.get("end_byte") or 0

    has_data = (
        generic.get("energy_hartree") is not None
        or bool(extra.get("frame_count"))
        or bool(extra.get("mode_count"))
    )

    return {
        "index": idx,
        "kind": kind,
        "name": generic.get("label") or raw.get("label") or "Unknown Task",
        "method": (
            raw.get("method")
            or extra.get("method")
            or extra.get("method_hint")
            or extra.get("program")  # Molpro: extra.program is the method-ish module name
        ),
        "basis": raw.get("basis") or extra.get("basis"),
        "energy_hartree": generic.get("energy_hartree") or raw.get("total_energy_hartree"),
        "line_range": (int(line_start or 0), int(line_end or 0)),
        "outcome": outcome,
        "has_usable_data": bool(has_data),
        "selection_priority": SELECTION_PRIORITY.get(kind, 0),
    }


def pick_primary(tasks: list[TaskSummary]) -> int | None:
    """Choose the "main" task: highest selection_priority, ties broken by latest index."""
    if not tasks:
        return None
    best = max(tasks, key=lambda t: (t.get("selection_priority", 0), t.get("index", 0)))
    return best.get("index")


def compute_derived(
    tasks: list[TaskSummary], raw_tasks: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Pre-compute scalars an agent typically wants without re-parsing.

    Always includes n_tasks. Adds final_energy_hartree if the last task with an
    energy has one. Adds n_imaginary_modes if the program emits per-task
    frequency_modes lists (NWChem does).
    """
    raw_tasks = raw_tasks or []
    derived: dict[str, Any] = {"n_tasks": len(tasks)}

    final_energy = None
    for t in reversed(tasks):
        if t.get("energy_hartree") is not None:
            final_energy = t["energy_hartree"]
            break
    if final_energy is not None:
        derived["final_energy_hartree"] = final_energy

    n_imag = 0
    for task in raw_tasks:
        for mode in task.get("frequency_modes") or []:
            f = mode.get("frequency_cm1")
            if f is not None and f < 0:
                n_imag += 1
    if n_imag > 0:
        derived["n_imaginary_modes"] = n_imag

    return derived


__all__ = [
    "SELECTION_PRIORITY",
    "normalize_kind",
    "to_task_summary",
    "pick_primary",
    "compute_derived",
]

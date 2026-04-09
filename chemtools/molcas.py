from __future__ import annotations

import re
from typing import Any

from .common import make_metadata

MODULE_RE = re.compile(r"---\s+Start Module:\s+([A-Za-z0-9_]+)")
INTERNAL_MODULES = {"last_energy", "last_atoms", "emil"}


def classify_module(name: str) -> str:
    upper = name.upper()
    if upper in {"SEWARD", "GATEWAY"}:
        return "integrals"
    if upper == "SCF":
        return "scf"
    if upper == "RASSCF":
        return "rasscf"
    if upper == "CASPT2":
        return "caspt2"
    if upper == "RASSI":
        return "rassi"
    if upper in {"SLAPAF", "ALASKA", "NUMGRAD"}:
        return "optimization"
    if upper == "VIBROT":
        return "frequency"
    if "SO" in upper or "SPIN" in upper:
        return "spin_orbit"
    return "other"


def parse_tasks(path: str, contents: str) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    last_line = 0

    for line_number, raw_line in enumerate(contents.splitlines(), start=1):
        last_line = line_number
        match = MODULE_RE.search(raw_line.rstrip())
        if not match:
            continue
        module = match.group(1).strip().upper()
        if module.lower() in INTERNAL_MODULES:
            continue
        if current is not None:
            current["line_end"] = line_number - 1
            tasks.append(current)
        current = {
            "kind": classify_module(module),
            "module": module,
            "line_start": line_number,
            "line_end": line_number,
        }

    if current is not None:
        current["line_end"] = max(last_line, current["line_start"])
        tasks.append(current)

    generic_tasks = []
    for task in tasks:
        generic_kind = {
            "optimization": "optimization",
            "frequency": "frequency",
            "scf": "scf",
        }.get(task["kind"], "other")
        generic_tasks.append(
            {
                "program": "molcas",
                "kind": generic_kind,
                "label": task["kind"].replace("_", " ").title(),
                "energy_hartree": None,
                "line_start": task["line_start"],
                "line_end": task["line_end"],
                "extra": {"module": task["module"]},
            }
        )

    return {
        "metadata": make_metadata(path, contents, "molcas"),
        "generic_tasks": generic_tasks,
        "program_summary": {
            "kind": "molcas",
            "task_count": len(tasks),
            "raw": {"tasks": tasks},
        },
    }


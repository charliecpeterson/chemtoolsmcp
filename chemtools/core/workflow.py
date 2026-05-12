"""Program-agnostic workflow DAG engine.

Multi-step calculation protocols (opt → freq, opt → CCSD(T), spin scans,
reaction-energy chains) follow a common shape: a step list with
``depends_on`` edges, optional dynamic step generators, and per-step
pre / launch / post action stanzas the agent walks.

The engine itself is generic. The *protocol library* (the set of named
recipes) and the *tool names* the actions reference are
program-specific. Both NWChem and Molcas can plug their own dicts into
``list_protocols`` and ``plan_calculation``.

Migration note: the NWChem-specific entry points in
``chemtools/programs/nwchem/protocols.py`` are thin wrappers over these
two functions — passing ``PROTOCOLS`` and the NWChem tool-name mapping.
A future Molcas protocol library would do the same.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable


# Default tool-name mapping. Programs override this with their own names
# (e.g. NWChem passes ``check_nwchem_freq_plausibility`` for ``check_freq``).
_DEFAULT_TOOL_NAMES: dict[str, str] = {
    "check_freq": "check_freq_plausibility",
    "check_geom": "check_geometry_plausibility",
    "workflow_state": "get_workflow_state",
    "extract_geom": "extract_geometry",
    "input_variant": "create_input_variant",
    "launch": "launch_run",
}


def list_protocols(protocols: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """Return ``[{name, description}, ...]`` for every protocol in the library."""
    return [
        {"name": name, "description": proto["description"]}
        for name, proto in protocols.items()
    ]


def plan_calculation(
    protocols: dict[str, dict[str, Any]],
    input_file: str,
    protocol: str,
    profile: str = "",
    output_dir: str | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    tool_names: dict[str, str] | None = None,
    dynamic_generators: dict[str, Callable[[str, dict[str, Any]], list[dict[str, Any]]]] | None = None,
) -> dict[str, Any]:
    """Build a step plan for ``protocol`` against ``input_file``.

    Parameters
    ----------
    protocols
        Program's protocol library (dict keyed by protocol name).
    input_file
        Path to the original input deck — anchors output filenames.
    protocol
        Name of the protocol to plan.
    profile
        Runner-profile name forwarded into per-step ``launch_action.params``.
    output_dir
        Where step outputs land (defaults to ``input_file``'s directory).
    overrides
        Caller-supplied tweaks. Forwarded to dynamic generators (e.g.
        ``multiplicities`` for spin scans).
    tool_names
        Maps abstract tool tags (``check_freq``, ``extract_geom``,
        ``launch``, ...) to the program's actual MCP tool names. Falls
        back to bare defaults if not supplied — useful for tests.
    dynamic_generators
        Maps a protocol's ``dynamic_generator`` string to a function that
        returns the step list. The function signature is
        ``(input_file, overrides) -> list[step_dict]``.

    Returns
    -------
    dict
        Workflow plan — same shape as the historical NWChem version
        (n_steps, steps, post_checks, parallel_independent, ...).
    """
    if protocol not in protocols:
        available = ", ".join(sorted(protocols.keys()))
        raise ValueError(f"Unknown protocol '{protocol}'. Available: {available}")

    proto = protocols[protocol]
    inp = Path(input_file)
    if output_dir is None:
        output_dir = str(inp.parent)
    base_stem = re.sub(r"_v\d+$", "", inp.stem)
    overrides = overrides or {}
    tn = {**_DEFAULT_TOOL_NAMES, **(tool_names or {})}
    generators = dynamic_generators or {}

    steps = proto.get("steps", [])
    if proto.get("dynamic"):
        gen_name = proto.get("dynamic_generator")
        gen = generators.get(gen_name) if gen_name else None
        if gen is not None:
            steps = gen(input_file, overrides)

    plan_steps: list[dict[str, Any]] = []
    for step in steps:
        step_id = step["id"]
        task_str = step.get("task", "energy")
        depends = step.get("depends_on")

        step_input = input_file if depends is None else f"<from step '{depends}'>"
        step_output = str(Path(output_dir) / f"{base_stem}_{step_id}.out")

        tool_params: dict[str, Any] = {
            "input_file": step_input,
            "profile": profile,
        }
        if step.get("basis_override"):
            tool_params["basis_override"] = step["basis_override"]

        pre_actions: list[dict[str, Any]] = []
        auto_input_action = step.get("auto_input")
        if auto_input_action and depends:
            if "extract_geometry" in auto_input_action:
                pre_actions.append({
                    "tool": tn["extract_geom"],
                    "params": {"output_file": f"<output of step '{depends}'>", "frame": "best"},
                    "purpose": "Get optimized geometry for next step",
                })
            if "freq" in auto_input_action:
                pre_actions.append({
                    "tool": tn["input_variant"],
                    "params": {
                        "source_input": input_file,
                        "changes": {"task": task_str},
                        "reason": f"Switch task to {task_str} for protocol step '{step_id}'",
                    },
                    "purpose": f"Create input for {task_str}",
                })

        plan_steps.append({
            "step_id": step_id,
            "task": task_str,
            "depends_on": depends,
            "expected_output": step_output,
            "pre_actions": pre_actions,
            "launch_action": {
                "tool": tn["launch"],
                "params": tool_params,
            },
            "post_actions": _post_actions_for_step(step, proto, step_output, input_file, tn),
        })

    post_checks: list[dict[str, Any]] = []
    for check_name in proto.get("post_process", []):
        post_checks.append({
            "tool": check_name,
            "params": {"output_file": "<final output>"},
        })

    return {
        "protocol": protocol,
        "description": proto["description"],
        "input_file": input_file,
        "output_dir": output_dir,
        "profile": profile,
        "n_steps": len(plan_steps),
        "parallel_independent": proto.get("parallel_independent", False),
        "steps": plan_steps,
        "post_checks": post_checks,
        "on_imaginary_modes": proto.get("on_imaginary_modes"),
    }


def _post_actions_for_step(
    step: dict[str, Any],
    proto: dict[str, Any],  # noqa: ARG001 — reserved for protocol-level post actions
    output_file: str,
    input_file: str,
    tool_names: dict[str, str],
) -> list[dict[str, Any]]:
    """Build the per-step post-completion action list."""
    actions: list[dict[str, Any]] = []
    task = step.get("task", "")
    if "freq" in task:
        actions.append({
            "tool": tool_names["check_freq"],
            "params": {"output_file": output_file},
            "purpose": "Verify frequencies are physically reasonable",
        })
    elif "optimize" in task:
        actions.append({
            "tool": tool_names["check_geom"],
            "params": {"output_file": output_file},
            "purpose": "Verify optimized geometry is reasonable",
        })
    actions.append({
        "tool": tool_names["workflow_state"],
        "params": {"input_file": input_file, "output_file": output_file},
        "purpose": "Check workflow state and determine if step succeeded",
    })
    return actions

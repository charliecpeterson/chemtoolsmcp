"""Generic rule-walker framework for program-specific recovery suggesters.

Each program registers a list of rules — callables of the form

    rule(text: str, parsed: dict | None) -> dict | None

that either return a recovery record (failure_class, severity, root_cause,
fix_recipe, next_actions, ...) when the rule matches, or None.

This module provides the dispatch + result-shaping logic; each program
provides its rule list (in ``programs/<prog>/strategy/recovery.py``) and
program-specific "did this run complete cleanly" semantics. The
``apply_recovery`` patcher (regex edits on input decks) stays in each
program's module because the input syntax differs.

Typical usage:

    from chemtools.core.recovery import dispatch_rules

    _RULES = [_rule_seward_angstrom_symmetry, _rule_missing_basis_in_loop, ...]

    def suggest_recovery(output_file, return_all_matches=False):
        text = read_text(output_file)
        parsed = parse_output_full(...)
        ran_clean = my_program_clean_check(text)
        return dispatch_rules(
            rules=_RULES,
            text=text,
            parsed=parsed,
            output_file=output_file,
            ran_clean=ran_clean,
            last_module=...,
            last_rc=...,
            return_all_matches=return_all_matches,
            unknown_failure_next_actions=[...],
        )
"""

from __future__ import annotations

from typing import Any, Callable


# A rule is text + parsed → recovery dict OR None
RuleFn = Callable[[str, "dict | None"], "dict | None"]


def dispatch_rules(
    *,
    rules: list[RuleFn],
    text: str,
    parsed: dict | None,
    output_file: str,
    ran_clean: bool,
    last_module: str | None,
    last_rc: str | None,
    return_all_matches: bool = False,
    unknown_failure_next_actions: list[dict] | None = None,
) -> dict[str, Any]:
    """Walk a priority-ordered rule list and synthesize a recovery result.

    Returns a dict with verdict ∈ {success_no_recovery_needed,
    recovery_suggested, unknown_failure}. The shape matches what each
    program's suggest_recovery historically returned — agents that
    chain on the verdict + recovery dict won't see a change.
    """
    matches: list[dict] = []
    for rule in rules:
        m = rule(text, parsed)
        if m is not None:
            matches.append(m)
            if not return_all_matches:
                break

    if not matches:
        verdict = "success_no_recovery_needed" if ran_clean else "unknown_failure"
        result: dict[str, Any] = {
            "verdict": verdict,
            "output_file": output_file,
            "ran_to_completion": ran_clean,
            "last_module": last_module,
            "last_rc": last_rc,
            "recovery": None,
        }
        if verdict == "unknown_failure" and unknown_failure_next_actions:
            result["next_actions"] = unknown_failure_next_actions
        return result

    primary = matches[0]
    result = {
        "verdict": "recovery_suggested",
        "output_file": output_file,
        "ran_to_completion": ran_clean,
        "last_module": last_module,
        "last_rc": last_rc,
        "recovery": primary,
        "failure_class": primary.get("failure_class"),
        "severity": primary.get("severity"),
        "next_actions": primary.get("next_actions", []),
    }
    if return_all_matches:
        result["all_matches"] = matches
    return result

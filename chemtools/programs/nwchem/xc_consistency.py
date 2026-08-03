"""Compare named NWChem DFT functionals with runtime XC evidence."""

from __future__ import annotations

import re
from typing import Any, Mapping


_XC_ALIASES = {
    "b3lyp": "b3lyp",
    "bhlyp": "bhlyp",
    "m062x": "m06-2x",
    "pbe0": "pbe0",
    "scan": "scan",
}


def canonical_xc_alias(value: str) -> str | None:
    key = re.sub(r"[^a-z0-9]", "", value.casefold())
    return _XC_ALIASES.get(key)


def compare_xc_functional(
    input_state: Mapping[str, Any],
    output_state: Mapping[str, Any],
) -> dict[str, Any]:
    input_xc = input_state.get("xc")
    labels = list(dict.fromkeys(
        output_state.get("xc_functional_labels") or []
    ))
    names = list(dict.fromkeys(
        output_state.get("xc_functional_names") or []
    ))
    output_evidence = {
        "names": names,
        "labels": labels,
    }
    if input_state.get("module") not in {"dft", "tddft"}:
        return {
            "status": "not_checked",
            "reason": "The input task does not use the DFT XC state.",
            "output": output_evidence,
        }
    if not input_xc:
        return {
            "status": "not_checked",
            "reason": "The input does not select an explicit XC functional.",
            "output": output_evidence,
        }
    if input_xc.get("source") != "explicit_alias":
        return {
            "status": "not_checked",
            "reason": (
                "The explicit XC expression is not a supported named alias."
            ),
            "input": dict(input_xc),
            "output": output_evidence,
        }
    if len(names) != 1:
        return {
            "status": "not_checked",
            "reason": (
                "The task does not expose one supported runtime XC alias."
            ),
            "input": dict(input_xc),
            "output": output_evidence,
        }
    return {
        "status": (
            "match" if input_xc["name"] == names[0] else "mismatch"
        ),
        "input": dict(input_xc),
        "output": {
            "name": names[0],
            "labels": labels,
        },
    }


__all__ = ["canonical_xc_alias", "compare_xc_functional"]

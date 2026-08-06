"""Compare the MCP multiplet engine with the original standalone implementation.

The reference repository is supplied explicitly so the project does not gain a
runtime dependency on a developer checkout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from chemtools.reference.atomic_multiplets import analyze_atomic_multiplets


_CORPUS = tuple(
    f"{orbital}{electrons}"
    for angular_momentum, orbital in enumerate("spdf")
    for electrons in range(1, 2 * (2 * angular_momentum + 1) + 1)
) + (
    "2p1 3s1",
    "3d2 4s1",
    "3d3 4s1",
    "4f2 5d1",
    "5f2 6d1 7s2",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
        help="Checkout containing multiplet_generator.py and grasp_preflight.py",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    try:
        reference = _load_reference(arguments.reference_root)
    except (ImportError, OSError, ValueError) as error:
        print(f"reference unavailable: {error}", file=sys.stderr)
        return 2

    records = []
    counts = {
        "agree": 0,
        "disagree": 0,
        "tool_refused": 0,
        "no_reference": 0,
    }
    for configuration in _CORPUS:
        try:
            expected = _reference_record(reference, configuration)
        except (RuntimeError, ValueError) as error:
            counts["no_reference"] += 1
            records.append({
                "configuration": configuration,
                "status": "no_reference",
                "error": str(error),
            })
            continue
        try:
            observed = _target_record(configuration)
        except (RuntimeError, ValueError) as error:
            status = "tool_refused"
            record: dict[str, Any] = {
                "configuration": configuration,
                "status": status,
                "error": str(error),
                "reference": expected,
            }
        else:
            status = "agree" if observed == expected else "disagree"
            record = {
                "configuration": configuration,
                "status": status,
            }
            if status == "disagree":
                record["reference"] = expected
                record["target"] = observed
        counts[status] += 1
        records.append(record)

    report = {
        "schema_version": "chemtools.atomic-multiplet-differential/1",
        "reference_root": str(arguments.reference_root.resolve()),
        "checked": len(records),
        **counts,
        "records": records,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.write_text(rendered, encoding="utf-8")
        print(arguments.output)
    compared = counts["agree"] + counts["disagree"] + counts["tool_refused"]
    if compared == 0:
        return 2
    return 1 if counts["disagree"] or counts["tool_refused"] else 0


def _load_reference(root: Path):
    module_path = root / "multiplet_generator.py"
    preflight_path = root / "grasp_preflight.py"
    if not module_path.is_file() or not preflight_path.is_file():
        raise ValueError(
            f"{root} must contain multiplet_generator.py and grasp_preflight.py"
        )
    sys.path.insert(0, str(root.resolve()))
    spec = importlib.util.spec_from_file_location(
        "chemtools_multiplet_reference",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reference module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_record(reference, configuration: str) -> dict[str, object]:
    parsed = reference.parse_configuration(configuration)
    terms = reference.terms_for_configuration(parsed)
    jj = reference.jj_analysis(parsed, terms)
    distribution = reference.configuration_microstates(parsed)
    return {
        "configuration": parsed.label,
        "electron_count": parsed.electron_count,
        "parity": "-" if parsed.parity == -1 else "+",
        "microstates": sum(distribution.values()),
        "terms": [
            [term.label, term.L, term.two_s, term.occurrences]
            for term in terms
        ],
        "j_levels": [
            [two_j, count]
            for two_j, count in reference.j_block_counts(terms).items()
        ],
        "jj_rows": [
            [row.label, [[two_j, count] for two_j, count in row.levels]]
            for row in jj.rows
        ],
        "jj_census": [[two_j, count] for two_j, count in jj.jj_census],
    }


def _target_record(configuration: str) -> dict[str, object]:
    analysis = analyze_atomic_multiplets(configuration)
    return {
        "configuration": analysis["configuration"],
        "electron_count": analysis["electron_count"],
        "parity": analysis["parity"],
        "microstates": analysis["microstate_counts"]["determinant_weights"],
        "terms": [
            [term["term"], term["L"], term["two_s"], term["occurrences"]]
            for term in analysis["terms"]
        ],
        "j_levels": [
            [block["two_j"], block["levels"]]
            for block in analysis["j_parity_blocks"]
        ],
        "jj_rows": [
            [
                row["configuration"],
                [
                    [level["two_j"], level["csfs"]]
                    for level in row["j_levels"]
                ],
            ]
            for row in analysis["jj_coupling"]["configurations"]
        ],
        "jj_census": [
            [level["two_j"], level["levels"]]
            for level in analysis["jj_coupling"]["jj_census"]
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())

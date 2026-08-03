"""Check the fixed Orbitron NBO Python-API slice against an owned fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chemtools.integrations.science_runtime import (
    ScienceRuntimeClient,
    ScienceRuntimeCommandError,
    ScienceRuntimeUnavailableError,
)


EXPECTED = {
    "orbital_count": 142,
    "orbital_type_counts": {"BD": 6, "BD*": 6, "CR": 9, "LV": 15, "RY*": 106},
    "occupancy_range": {"minimum": 0.0, "maximum": 2.0},
    "per_atom_entry_counts": [
        {"atom_index": 0, "entry_count": 110},
        {"atom_index": 1, "entry_count": 22},
        {"atom_index": 2, "entry_count": 22},
    ],
}
EXPECTED_SOURCE_SHA256 = "f29c9a3275223c1fa28eed396a61d282a1333ef3dd259e4be3ca689dfa086311"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check Orbitron's bounded NBO Python-API MCP operation."
    )
    parser.add_argument(
        "--fixture",
        required=True,
        type=Path,
        help="Path to Orbitron's uo2-test.nbo fixture.",
    )
    parser.add_argument(
        "--python",
        help="Explicit chemtools-science Python path; defaults to CHEMTOOLS_SCIENCE_PYTHON.",
    )
    arguments = parser.parse_args(argv)
    try:
        response = ScienceRuntimeClient(arguments.python).orbitron_nbo({
            "schema_version": "chemtools.orbitron-nbo-request/1",
            "path": str(arguments.fixture.resolve()),
        })
    except (ScienceRuntimeUnavailableError, ScienceRuntimeCommandError) as error:
        print(json.dumps({"outcome": "tool_refused", "message": str(error)}))
        return 2

    actual = response.get("nbo")
    source = response.get("source", {})
    if (
        response.get("status") == "completed"
        and source.get("sha256") == EXPECTED_SOURCE_SHA256
        and isinstance(actual, dict)
        and {key: actual.get(key) for key in EXPECTED} == EXPECTED
        and len(actual.get("bonding_orbital_samples", [])) == 12
        and [sample.get("orbital_type") for sample in actual["bonding_orbital_samples"]]
        == ["BD"] * 6 + ["BD*"] * 6
    ):
        print(json.dumps({"outcome": "agree", "fixture": str(arguments.fixture)}))
        return 0
    print(json.dumps({
        "outcome": "disagree",
        "expected": EXPECTED,
        "actual": actual,
        "source": source,
        "status": response.get("status"),
    }, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

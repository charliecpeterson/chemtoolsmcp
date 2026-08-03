"""Check the fixed Orbitron structure-identity Python-API slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chemtools.integrations.science_runtime import (
    ScienceRuntimeClient,
    ScienceRuntimeCommandError,
    ScienceRuntimeUnavailableError,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "orbitron_identity"
    / "zncl2.xyz"
)
EXPECTED_SUMMARY = {
    "atom_count": 3,
    "bond_count": 2,
    "bond_order_counts": {"Dative": 2},
    "identifiers": {
        "formula": {"status": "available", "value": "Cl2Zn"},
        "inchi": {
            "status": "available",
            "value": "InChI=1S/2ClH.Zn/h2*1H;/q;;+2/p-2",
        },
        "inchikey": {
            "status": "available",
            "value": "JIAARYAFYJHUJI-UHFFFAOYSA-L",
        },
        "smiles": {"status": "available", "value": "[Cl][Zn][Cl]"},
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check Orbitron's structure-identity Python-API MCP operation."
    )
    parser.add_argument(
        "--python",
        help="Explicit chemtools-science Python path; defaults to CHEMTOOLS_SCIENCE_PYTHON.",
    )
    arguments = parser.parse_args(argv)
    try:
        response = ScienceRuntimeClient(arguments.python).orbitron_structure_identity({
            "schema_version": "chemtools.orbitron-structure-identity-request/1",
            "path": str(FIXTURE_PATH),
        })
    except (ScienceRuntimeUnavailableError, ScienceRuntimeCommandError) as error:
        print(json.dumps({"outcome": "tool_refused", "message": str(error)}))
        return 2

    actual = response.get("structure_identity")
    if response.get("status") == "completed" and actual == EXPECTED_SUMMARY:
        print(json.dumps({"outcome": "agree", "fixture": str(FIXTURE_PATH)}))
        return 0
    print(json.dumps({
        "outcome": "disagree",
        "expected": EXPECTED_SUMMARY,
        "actual": actual,
        "status": response.get("status"),
    }, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

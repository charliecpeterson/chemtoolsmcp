"""Check the fixed Orbitron periodic Python-API slice against its fixture."""

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
    / "orbitron_periodic"
    / "vasprun_band_dos.xml"
)
EXPECTED_SUMMARY = {
    "fermi_energy_ev": 1.2,
    "total_magnetization_bohr": None,
    "band_gap": {"status": "available", "value_ev": 7.0, "is_direct": False},
    "band_structure": {
        "status": "available",
        "sampling": "path",
        "spin_channels": ["total"],
        "kpoint_count": 2,
        "band_count_per_spin": [2],
        "label_count": 2,
        "segment_count": 0,
    },
    "density_of_states": {
        "status": "available",
        "spin_channels": ["total"],
        "energy_point_count": 3,
        "energy_min_ev": -5.0,
        "energy_max_ev": 5.0,
        "integrated_available": True,
    },
    "projected_data": "omitted",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check Orbitron's periodic Python-API MCP operation."
    )
    parser.add_argument(
        "--python",
        help="Explicit chemtools-science Python path; defaults to CHEMTOOLS_SCIENCE_PYTHON.",
    )
    arguments = parser.parse_args(argv)
    try:
        response = ScienceRuntimeClient(
            arguments.python
        ).orbitron_periodic_electronic_structure({
            "schema_version": "chemtools.orbitron-periodic-electronic-structure-request/1",
            "path": str(FIXTURE_PATH),
        })
    except (ScienceRuntimeUnavailableError, ScienceRuntimeCommandError) as error:
        print(json.dumps({"outcome": "tool_refused", "message": str(error)}))
        return 2

    actual = response.get("periodic_electronic_structure")
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

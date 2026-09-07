"""Compare Chemtools GRASP parsing with an independent summary reader.

Each case pairs any GRASP artifact with the authoritative ``.sum`` or ``.csum``
from the same run. The report distinguishes agreement, disagreement, and a
deliberate refusal to extract an energy from an execution-only log.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemtools.programs.grasp import GRASP


_FLOAT = r"[+-]?\d+\.\d+(?:[DdEe][+-]?\d+)?"
_HEADER_RE = re.compile(
    r"There are\s+(\d+)\s+electrons in the cloud\s+"
    r"in\s+(\d+)\s+relativistic CSFs\s+"
    r"based on\s+(\d+)\s+relativistic subshells",
    re.DOTALL,
)
_FERMI_RE = re.compile(
    rf"Fermi nucleus:\s+c\s*=\s*({_FLOAT})\s+Bohr radii,\s*"
    rf"a\s*=\s*({_FLOAT})\s+Bohr radii",
    re.DOTALL,
)
_EIGEN_ROW_RE = re.compile(
    rf"^\s*\d+\s+\S+\s+[+-]\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$",
    re.MULTILINE,
)


def compare_case(
    label: str,
    artifact_path: Path,
    summary_path: Path,
    *,
    energy_tolerance: float,
) -> dict[str, Any]:
    reference = read_summary_reference(summary_path)
    parsed = GRASP.parser.parse_output(str(artifact_path))
    derived = parsed.get("derived") or {}
    parser_energy = derived.get("final_energy_hartree")
    if parser_energy is None and derived.get("grasp:energy_source_required"):
        outcome = "tool-refused"
        reason = (
            "The artifact is an execution log; Chemtools requires the "
            "authoritative GRASP summary for an energy."
        )
    elif _close(parser_energy, reference["energy_hartree"], energy_tolerance):
        outcome = "agree"
        reason = "Chemtools and the independent summary reader agree."
    else:
        outcome = "disagree"
        reason = "The parsed energy differs from the authoritative summary."

    metadata_checks = {}
    for parser_key, reference_key in (
        ("grasp:n_electrons", "n_electrons"),
        ("grasp:n_csfs", "n_csfs"),
        ("grasp:n_subshells", "n_subshells"),
    ):
        parser_value = derived.get(parser_key)
        reference_value = reference.get(reference_key)
        metadata_checks[reference_key] = {
            "parser": parser_value,
            "reference": reference_value,
            "status": (
                "not-available"
                if parser_value is None
                else "agree"
                if parser_value == reference_value
                else "disagree"
            ),
        }

    return {
        "label": label,
        "outcome": outcome,
        "reason": reason,
        "artifact": str(artifact_path.resolve()),
        "summary": str(summary_path.resolve()),
        "artifact_kind": derived.get("grasp:file_kind"),
        "parser_energy_hartree": parser_energy,
        "reference_energy_hartree": reference["energy_hartree"],
        "energy_difference_hartree": (
            float(parser_energy) - reference["energy_hartree"]
            if parser_energy is not None
            else None
        ),
        "energy_tolerance_hartree": energy_tolerance,
        "metadata_checks": metadata_checks,
        "reference": reference,
    }


def read_summary_reference(path: Path) -> dict[str, Any]:
    """Read only fields needed by the differential check via local regexes."""
    text = path.read_text(encoding="utf-8", errors="replace")
    header = _HEADER_RE.search(text)
    if header is None:
        raise ValueError(f"GRASP count header not found in {path}")
    contributor = text.find("Weights of major contributors to ASF:")
    if contributor < 0:
        raise ValueError(f"authoritative eigenenergy section not found in {path}")
    eigen_header = text.rfind("Eigenenergies", 0, contributor)
    if eigen_header < 0:
        raise ValueError(f"authoritative eigenenergy header not found in {path}")
    energies = [
        _fortran_float(match.group(1))
        for match in _EIGEN_ROW_RE.finditer(text[eigen_header:contributor])
    ]
    if not energies:
        raise ValueError(f"authoritative eigenenergy rows not found in {path}")
    nucleus = {}
    fermi = _FERMI_RE.search(text)
    if fermi is not None:
        nucleus = {
            "fermi_c_bohr": _fortran_float(fermi.group(1)),
            "fermi_a_bohr": _fortran_float(fermi.group(2)),
            "stationary": "the nucleus is stationary" in text,
        }
    transverse = "H (Transverse)" in text
    frequency = re.search(
        rf"factor multiplying the photon frequency:\s*({_FLOAT})",
        text,
    )
    return {
        "energy_hartree": min(energies),
        "n_electrons": int(header.group(1)),
        "n_csfs": int(header.group(2)),
        "n_subshells": int(header.group(3)),
        "nucleus": nucleus,
        "transverse_breit": transverse,
        "photon_frequency_factor": (
            _fortran_float(frequency.group(1)) if frequency is not None else None
        ),
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    cases = [
        compare_case(
            label,
            Path(artifact).expanduser().resolve(),
            Path(summary).expanduser().resolve(),
            energy_tolerance=arguments.energy_tolerance,
        )
        for label, artifact, summary in arguments.case
    ]
    counts = {
        outcome: sum(case["outcome"] == outcome for case in cases)
        for outcome in ("agree", "disagree", "tool-refused")
    }
    return {
        "schema_version": "chemtools.grasp-output-differential/1",
        "cases": cases,
        "counts": counts,
        "success": counts["disagree"] == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        metavar=("LABEL", "ARTIFACT", "SUMMARY"),
        required=True,
    )
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--json", type=Path)
    arguments = parser.parse_args()
    report = run(arguments)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.json is not None:
        arguments.json.expanduser().resolve().write_text(
            encoded,
            encoding="utf-8",
        )
    print(encoded, end="")
    return 0 if report["success"] else 1


def _fortran_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _close(left: Any, right: float, tolerance: float) -> bool:
    return (
        not isinstance(left, bool)
        and isinstance(left, (int, float))
        and math.isfinite(float(left))
        and abs(float(left) - right) <= tolerance
    )


if __name__ == "__main__":
    sys.exit(main())

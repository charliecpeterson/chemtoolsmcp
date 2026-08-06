"""Run one input-ready f-block GRASP reference through RMCDHF and RCI.

The script validates CSF and ASF labels against the catalog and compares the
resulting (2J+1)-weighted configuration averages with the recorded energies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from chemtools.programs.grasp.parse.rmcdhf_log import parse_rmcdhf_log
from chemtools.programs.grasp.parse.sum_file import parse_sum
from chemtools.reference import (
    plan_fblock_atomic_state,
    validate_grasp_fblock_artifacts,
)


_EXECUTABLES = {
    "rnucleus": "/apps/grasp/bin/rnucleus",
    "rcsfgenerate": "/apps/grasp/bin/rcsfgenerate",
    "rangular": "/apps/grasp/bin/rangular",
    "rwfnestimate": "/apps/grasp/bin/rwfnestimate",
    "rmcdhf": "/apps/grasp/bin/rmcdhf",
    "rci": "/apps/grasp/bin/rci",
}


def _run(
    name: str,
    lines: list[str],
    work: Path,
    timeout: float,
) -> None:
    completed = subprocess.run(
        [_EXECUTABLES[name]],
        input="\n".join([*lines, ""]),
        text=True,
        cwd=work,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    (work / f"{name}.stdout").write_text(completed.stdout, encoding="utf-8")
    (work / f"{name}.stderr").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise ValueError(f"{name} exited {completed.returncode}")


def _configuration_average(summary: dict[str, object]) -> float:
    levels = summary.get("eigenenergies")
    if not isinstance(levels, list) or not levels:
        raise ValueError("GRASP summary contains no eigenenergies")
    weighted_energy = 0.0
    total_weight = 0
    for level in levels:
        j_label = level["j_str"]
        if "/" in j_label:
            numerator, denominator = j_label.split("/", 1)
            two_j = 2 * int(numerator) // int(denominator)
        else:
            two_j = 2 * int(j_label)
        weight = two_j + 1
        weighted_energy += weight * level["energy_hartree"]
        total_weight += weight
    return weighted_energy / total_weight


def check_reference(arguments: argparse.Namespace) -> dict[str, object]:
    work = arguments.scratch.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    plan = plan_fblock_atomic_state(arguments.element, arguments.state).to_dict()
    if plan["automation"] != {"status": "input_ready", "requirements": []}:
        raise ValueError("representative check requires an input-ready cold state")
    inputs = plan["grasp2018"]["inputs"]

    _run("rnucleus", inputs["rnucleus"], work, arguments.timeout)
    _run("rcsfgenerate", inputs["rcsfgenerate"], work, arguments.timeout)
    shutil.copy2(work / "rcsf.out", work / "rcsf.inp")
    _run("rangular", inputs["rangular"], work, arguments.timeout)
    _run("rwfnestimate", inputs["rwfnestimate"], work, arguments.timeout)
    _run("rmcdhf", inputs["rmcdhf"], work, arguments.timeout)

    rmcdhf_log = parse_rmcdhf_log(str(work / "rmcdhf.stdout"))
    if not rmcdhf_log["converged"]:
        raise ValueError("RMCDHF lacks positive convergence evidence")
    rmcdhf_validation = validate_grasp_fblock_artifacts(
        arguments.element,
        arguments.state,
        work / "rcsf.inp",
        mixing_path=work / "rmix.out",
    )

    shutil.copy2(work / "rcsf.inp", work / "ref.c")
    shutil.copy2(work / "rwfn.out", work / "ref.w")
    _run("rci", inputs["rci"], work, arguments.timeout)
    rci_validation = validate_grasp_fblock_artifacts(
        arguments.element,
        arguments.state,
        work / "ref.c",
        mixing_path=work / "ref.cm",
    )

    expected = plan["grasp2018"]["expected"]["energies_au"]
    dc_average = _configuration_average(parse_sum(str(work / "rmcdhf.sum")))
    dcb_summary = parse_sum(str(work / "ref.csum"))
    dcb_average = _configuration_average(dcb_summary)
    energy_differences = {
        "dirac_coulomb": dc_average - expected["dirac_coulomb"],
        "dirac_coulomb_breit": (
            dcb_average - expected["dirac_coulomb_breit"]
        ),
    }
    corrections = dcb_summary.get("rci_corrections")
    success = (
        rmcdhf_validation["valid"]
        and rci_validation["valid"]
        and all(abs(value) <= arguments.energy_tolerance for value in energy_differences.values())
        and isinstance(corrections, dict)
        and corrections["transverse_breit"] is True
        and corrections["vacuum_polarisation"] is False
        and corrections["self_energy"] is False
    )
    return {
        "schema_version": "chemtools.fblock-grasp-reference-check/1",
        "query": plan["query"],
        "catalog_sha256": plan["reference"]["dataset"]["catalog_sha256"],
        "rmcdhf": {
            "convergence": rmcdhf_log,
            "configuration_average_au": dc_average,
            "catalog_difference_au": energy_differences["dirac_coulomb"],
            "artifact_validation": rmcdhf_validation,
        },
        "rci": {
            "configuration_average_au": dcb_average,
            "catalog_difference_au": energy_differences[
                "dirac_coulomb_breit"
            ],
            "corrections": corrections,
            "artifact_validation": rci_validation,
        },
        "energy_tolerance_au": arguments.energy_tolerance,
        "success": success,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("element")
    parser.add_argument("state")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--energy-tolerance", type=float, default=1e-5)
    arguments = parser.parse_args()
    evidence = check_reference(arguments)
    evidence_path = arguments.scratch.expanduser().resolve() / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "evidence": str(evidence_path),
        "rmcdhf_difference_au": evidence["rmcdhf"]["catalog_difference_au"],
        "rci_difference_au": evidence["rci"]["catalog_difference_au"],
        "success": evidence["success"],
    }, sort_keys=True))
    return 0 if evidence["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

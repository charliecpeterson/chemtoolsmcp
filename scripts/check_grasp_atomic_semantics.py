"""Exercise GRASP multireference, excitation, ASF, and orbital-role semantics.

The cases reproduce small GRASP2018 manual examples and retain every generated
artifact under the caller-provided scratch directory for independent review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from chemtools.programs.grasp.binary.mixing import inspect_grasp_mixing
from chemtools.programs.grasp.input.heredoc import (
    rangular_input,
    rcsfgenerate_input,
    rmcdhf_input,
    rnucleus_input,
    rwfnestimate_input,
)
from chemtools.programs.grasp.parse.csf import load_grasp_csf_list
from chemtools.programs.grasp.parse.rmcdhf_log import parse_rmcdhf_log


_EXECUTABLES = {
    "rangular": "/apps/grasp/bin/rangular",
    "rcsfgenerate": "/apps/grasp/bin/rcsfgenerate",
    "rmcdhf": "/apps/grasp/bin/rmcdhf",
    "rnucleus": "/apps/grasp/bin/rnucleus",
    "rwfnestimate": "/apps/grasp/bin/rwfnestimate",
}


def _run(
    executable: str,
    stdin_lines: list[str],
    work: Path,
    *,
    label: str,
    timeout: float,
) -> None:
    completed = subprocess.run(
        [_EXECUTABLES[executable]],
        input="\n".join([*stdin_lines, ""]),
        text=True,
        cwd=work,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    (work / f"{label}.stdin").write_text(
        "\n".join([*stdin_lines, ""]),
        encoding="utf-8",
    )
    (work / f"{label}.stdout").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (work / f"{label}.stderr").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise ValueError(f"{label} exited {completed.returncode}")


def _csf_blocks(path: Path) -> list[dict[str, object]]:
    document = load_grasp_csf_list(path)
    return [
        {
            "j": block.j_label,
            "parity": block.parity,
            "ncsf": len(block.entries),
        }
        for block in document.blocks
    ]


def _manual_generation_checks(
    scratch: Path,
    timeout: float,
) -> list[dict[str, object]]:
    cases = [
        {
            "name": "be_multireference_sd_n4",
            "stdin": rcsfgenerate_input(
                configurations=["1s(2,*)2s(2,*)", "1s(2,*)2p(2,*)"],
                active_orbitals="4s,4p,4d,4f",
                twoj_min=0,
                twoj_max=0,
                excitations=2,
            ),
            "electron_count": 4,
            "expected_blocks": [{"j": "0", "parity": "+", "ncsf": 361}],
            "manual": "GRASP2018 manual section 6.1",
        },
        {
            "name": "li_reference_even_odd",
            "stdin": rcsfgenerate_input(
                configurations=["1s(2,i)2s(1,i)"],
                active_orbitals="2s",
                twoj_min=1,
                twoj_max=1,
                excitations=0,
                additional_lists=[
                    {
                        "configurations": ["1s(2,i)2p(1,i)"],
                        "active_orbitals": "1s,2p",
                        "twoj_min": 1,
                        "twoj_max": 3,
                        "excitations": 0,
                    }
                ],
            ),
            "electron_count": 3,
            "expected_blocks": [
                {"j": "1/2", "parity": "+", "ncsf": 1},
                {"j": "1/2", "parity": "-", "ncsf": 1},
                {"j": "3/2", "parity": "-", "ncsf": 1},
            ],
            "manual": "GRASP2018 manual section 7.1",
        },
        {
            "name": "li_merged_excitation_policies",
            "stdin": rcsfgenerate_input(
                configurations=["1s(2,*)2p(1,*)"],
                active_orbitals="5s,5p,5d,5f,5g",
                twoj_min=1,
                twoj_max=3,
                excitations=3,
                additional_lists=[
                    {
                        "configurations": ["1s(2,*)2p(1,*)"],
                        "active_orbitals": "7s,7p,7d,7f,7g,7h,7i",
                        "twoj_min": 1,
                        "twoj_max": 3,
                        "excitations": 2,
                    }
                ],
            ),
            "electron_count": 3,
            "expected_blocks": [
                {"j": "1/2", "parity": "-", "ncsf": 2408},
                {"j": "3/2", "parity": "-", "ncsf": 4174},
            ],
            "manual": "GRASP2018 manual section 6.7",
        },
    ]
    evidence = []
    for case in cases:
        work = scratch / str(case["name"])
        work.mkdir(parents=True, exist_ok=True)
        _run(
            "rcsfgenerate",
            case["stdin"],
            work,
            label="rcsfgenerate",
            timeout=timeout,
        )
        document = load_grasp_csf_list(work / "rcsf.out")
        actual_blocks = _csf_blocks(work / "rcsf.out")
        passed = (
            document.electron_count == case["electron_count"]
            and actual_blocks == case["expected_blocks"]
        )
        evidence.append({
            "name": case["name"],
            "manual": case["manual"],
            "electron_count": document.electron_count,
            "blocks": actual_blocks,
            "passed": passed,
        })
    return evidence


def _li_rmcdhf_checks(
    scratch: Path,
    timeout: float,
) -> list[dict[str, object]]:
    reference = scratch / "li_reference_rmcdhf"
    reference.mkdir(parents=True, exist_ok=True)
    _run(
        "rnucleus",
        rnucleus_input(z=3, a=7, nuclear_mass_amu=6.94, nuclear_spin=1.5),
        reference,
        label="rnucleus",
        timeout=timeout,
    )
    generation = rcsfgenerate_input(
        configurations=["1s(2,i)2s(1,i)"],
        active_orbitals="2s",
        twoj_min=1,
        twoj_max=1,
        excitations=0,
        additional_lists=[
            {
                "configurations": ["1s(2,i)2p(1,i)"],
                "active_orbitals": "1s,2p",
                "twoj_min": 1,
                "twoj_max": 3,
                "excitations": 0,
            }
        ],
    )
    _run(
        "rcsfgenerate",
        generation,
        reference,
        label="rcsfgenerate",
        timeout=timeout,
    )
    shutil.copy2(reference / "rcsf.out", reference / "rcsf.inp")
    _run(
        "rangular",
        rangular_input(),
        reference,
        label="rangular",
        timeout=timeout,
    )
    _run(
        "rwfnestimate",
        rwfnestimate_input(),
        reference,
        label="rwfnestimate",
        timeout=timeout,
    )
    _run(
        "rmcdhf",
        rmcdhf_input(
            block_level_selections=["1", "1", "1"],
            orbitals_to_optimize="*",
            spectroscopic_orbitals="*",
        ),
        reference,
        label="rmcdhf",
        timeout=timeout,
    )
    reference_mixing = inspect_grasp_mixing(
        reference / "rmix.out",
        csf_path=reference / "rcsf.inp",
    )
    reference_blocks = [
        {
            "j": block["j_label"],
            "parity": block["parity"],
            "ncsf": block["csf_count"],
            "nasf": block["eigenstate_count"],
        }
        for block in reference_mixing["blocks"]
    ]
    expected_reference_blocks = [
        {"j": "1/2", "parity": "+", "ncsf": 1, "nasf": 1},
        {"j": "1/2", "parity": "-", "ncsf": 1, "nasf": 1},
        {"j": "3/2", "parity": "-", "ncsf": 1, "nasf": 1},
    ]
    reference_log = parse_rmcdhf_log(str(reference / "rmcdhf.stdout"))

    correlation = scratch / "li_2s_n3_correlation"
    correlation.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reference / "isodata", correlation / "isodata")
    shutil.copy2(reference / "rwfn.out", correlation / "rwfn.inp")
    _run(
        "rcsfgenerate",
        rcsfgenerate_input(
            configurations=["1s(2,*)2s(1,*)"],
            active_orbitals="3s,3p,3d",
            twoj_min=1,
            twoj_max=1,
            excitations=3,
        ),
        correlation,
        label="rcsfgenerate",
        timeout=timeout,
    )
    shutil.copy2(correlation / "rcsf.out", correlation / "rcsf.inp")
    _run(
        "rangular",
        rangular_input(),
        correlation,
        label="rangular",
        timeout=timeout,
    )
    _run(
        "rwfnestimate",
        rwfnestimate_input(sources=["file:rwfn.inp", "2"]),
        correlation,
        label="rwfnestimate",
        timeout=timeout,
    )
    correlation_rmcdhf = rmcdhf_input(
        block_level_selections=["1"],
        orbitals_to_optimize="3*",
        spectroscopic_orbitals="",
    )
    _run(
        "rmcdhf",
        correlation_rmcdhf,
        correlation,
        label="rmcdhf",
        timeout=timeout,
    )
    correlation_mixing = inspect_grasp_mixing(
        correlation / "rmix.out",
        csf_path=correlation / "rcsf.inp",
    )
    correlation_block = correlation_mixing["blocks"][0]
    correlation_log = parse_rmcdhf_log(str(correlation / "rmcdhf.stdout"))

    return [
        {
            "name": "li_reference_all_spectroscopic",
            "blocks": reference_blocks,
            "rmcdhf_converged": reference_log["converged"],
            "passed": (
                reference_blocks == expected_reference_blocks
                and reference_log["converged"]
            ),
        },
        {
            "name": "li_2s_n3_correlation_layer",
            "stdin": correlation_rmcdhf,
            "block": {
                "j": correlation_block["j_label"],
                "parity": correlation_block["parity"],
                "ncsf": correlation_block["csf_count"],
                "nasf": correlation_block["eigenstate_count"],
            },
            "optimized_orbitals": "3*",
            "spectroscopic_orbitals": "",
            "weight_prompt_omitted": "5" not in correlation_rmcdhf,
            "rmcdhf_converged": correlation_log["converged"],
            "passed": (
                correlation_block["j_label"] == "1/2"
                and correlation_block["parity"] == "+"
                and correlation_block["csf_count"] == 79
                and correlation_block["eigenstate_count"] == 1
                and "5" not in correlation_rmcdhf
                and correlation_log["converged"]
            ),
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    arguments = parser.parse_args()
    scratch = arguments.scratch.expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    generation = _manual_generation_checks(scratch, arguments.timeout)
    rmcdhf = _li_rmcdhf_checks(scratch, arguments.timeout)
    success = all(case["passed"] for case in [*generation, *rmcdhf])
    evidence = {
        "schema_version": "chemtools.grasp-atomic-semantics-check/1",
        "generation_cases": generation,
        "rmcdhf_cases": rmcdhf,
        "success": success,
    }
    evidence_path = scratch / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "evidence": str(evidence_path),
        "generation_cases": len(generation),
        "rmcdhf_cases": len(rmcdhf),
        "success": success,
    }, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
